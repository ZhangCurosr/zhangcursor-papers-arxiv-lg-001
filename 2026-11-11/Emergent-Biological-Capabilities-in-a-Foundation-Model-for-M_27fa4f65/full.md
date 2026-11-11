# Emergent Biological Capabilities in a Foundation Model for Molecular Interactions

Adi Mashiach Output Labs adi@outputlabs.ai

## Abstract

Foundation models trained on large unlabeled corpora develop emergent capabilities, tasks that go beyond the training objective. We extend this paradigm to multimodal biology with Bonbon, a foundation model for molecular interactions trained on approximately 3.2 trillion tokens of paired protein-ligand sequences. From a single representation space, Bonbon exhibits three zero-shot emergent capabilities at hierarchical levels of resolution: functional (mechanism of action), residue (binding site localization, including orthosteric/allosteric discrimination), and bond (covalent versus non-covalent engagement). All three capabilities generalize to out-of-distribution complexes. Bonbon was trained on protein-ligand sequences alone, without any task-specific supervision and without structural coordinates. We conducted controlled scaling experiments across six model scales from 42M to 1.6B parameters; every capability evaluated shows a sharp jump at the largest scale on continuous metrics, even as pretraining loss decreases smoothly. This demonstrates that loss-based scaling laws alone are insuficient to predict when zero-shot emergent capabilities become available. The capabilities reported here emerge after training at a 1:2,000 parameter-to-token ratio to accommodate the sparsity of paired biological interaction data. The same frozen representations achieve state-of-the-art binding and afinity prediction at five orders of magnitude greater speed than co-folding methods. In prospective wet-lab validation across nine disease targets spanning seven protein families, Bonbon achieves a 61.5% dose-response hit rate on de novo compounds, yielding structurally novel molecules not previously reported to inhibit these proteins, including for out-of-distribution targets.

## 1 Introduction

Foundation models trained by self-supervised learning at scale exhibit emergent capabilities: tasks that go beyond the training objective [1]. This principle extends from language [2, 3] to multimodal vision-language [4, 5] and unimodal biological domains: ESM-2 predicts protein structure from masked language modeling [6], and Evo captures regulatory patterns across genomes [7]. Yet no model for molecular interactions has demonstrated capabilities beyond its explicit training objective without task-specific supervision. Co-folding models [8–10] have transformed protein-ligand structure prediction, yet their capabilities do not extend beyond predicting 3D atomic coordinates. Independent benchmarks report substantial memorization with limited generalization to unseen ligands [11, 12]; capabilities such as afinity prediction require supervised heads rather than emerging from the pretrained representation. The broader landscape of molecular interaction modeling remains fragmented, with task-specific models each trained with explicit supervision for a single capability [13–15].

We hypothesize that a foundation model for molecular interactions, with an architecture matched to the learning objective and trained at suficient scale, can extract high-order interaction properties encoded in the sequences of interacting molecules, without requiring structural coordinates or physics-based analysis. Molecular interaction modeling faces challenges in both data and architecture that distinguish it from vision-language foundation models. For example, paired protein-ligand data is sparse, carrying far less signal than a dense image-caption pair [16, 17], with the patterns distinguishing functional outcomes (agonism vs. antagonism, covalent vs. non-covalent engagement) distributed across millions of examples rather than concentrated within each. Similarly, molecular interactions form many-to-many relationships, where one protein binds thousands of ligands and one ligand hundreds of proteins, whereas standard multimodal architectures assume a one-to-one correspondence between modalities.

We introduce Bonbon, a transformer-based model trained by self-supervised learning on approximately 3.2 trillion tokens of paired protein-ligand sequences. Bonbon adopts a data-heavy scaling regime: a 1:2,000 parameter-to-token ratio, two orders of magnitude beyond the Chinchilla-optimal ratio for natural language [18].

From a single representation space, Bonbon exhibits three emergent capabilities arising at hierarchical levels of resolution. At the functional level, Bonbon zero-shot predicts mechanism of action across six classes, including agonist/antagonist discrimination on the same receptor. At the residue level, Bonbon localizes where interactions occur, enabling zero-shot binding site detection, including on targets such as GPCRs and transporters that lack well-defined pockets, and a parameter-free signature distinguishes orthosteric from allosteric pockets, including on the same protein. At the bond level, dense representations classify covalent versus non-covalent engagement and per-residue representations localize the reactive residue. All three capabilities emerge from paired sequences alone, without task-specific labels, structural coordinates, or binding geometries, and generalize to out-ofdistribution complexes. In a controlled scaling experiment across six model scales (42M to 1.6B parameters), all three capabilities show a sharp jump at 1.6B that is not predicted by the smoothly decreasing loss curve.

The same frozen representations transfer to standard predictive benchmarks: Bonbon achieves state-of-the-art zero-shot binding prediction (MF-PCBA AUPR 0.081, LIT-PCBA-AF3 EF1% 16.6) and afinity prediction matching physics-based free energy perturbation (CASP16 r = 0.74, FEP+4 r = 0.78), at approximately five orders of magnitude greater speed than co-folding methods. In prospective wet-lab validation across nine disease targets spanning seven protein families, Bonbon achieves a 61.5% dose-response hit rate on de novo compounds, yielding structurally novel molecules not previously reported to inhibit these proteins, including for out-of-distribution targets.

## Contributions.

• A foundation model for molecular interactions, extending the foundation-model paradigm to multimodal biology.

• Three emergent capabilities arising at hierarchical levels of resolution within a single representation space, without task-specific supervision.

• A controlled scaling experiment showing that capabilities remain flat or near baseline through approximately three orders of magnitude of compute and emerge sharply at the largest scale on continuous metrics, while pretraining loss decreases smoothly.

• State-of-the-art binding and afinity prediction at five orders of magnitude greater speed than co-folding methods, and prospective wet-lab validation across nine disease targets spanning seven protein families, with a 61.5% dose-response hit rate on de novo compounds, including for out-of-distribution targets, yielding molecules not previously reported to inhibit these proteins.

• A data-heavy scaling regime (1:2,000) adopted to accommodate the information sparsity of paired biological data.

![](images/b376fa45b39651816b3267e3bfc9ef95f42720855db7c7c33e0c4b9a8de2c739.jpg)

B  
![](images/1cf30316d785a632c380fd92ba4f3ef2085b3c189c8c6e8d2b656513a176198b.jpg)  
C

![](images/5609339eeb4159ba267454340bb323171a5839220f63205c1f775c918484af27.jpg)  
D

![](images/ea2a3736803c71e6ba7af3063ff17840c077349322b7696381bdbc2f4caef3a9.jpg)  
E

![](images/b3e25824530e222b938b60731b248289841a82101529598b6ca6d63346978dfd.jpg)  
Figure 1: Capability emergence at scale. Controlled scaling experiment across six model scales (42M, 315M, 800M, 1B, 1.3B, 1.6B parameters) under identical conditions except parameter count. Panels (b) to (e) use log FLOPs as the x-axis. (a) UMAP projections (visualization only) of frozen interaction representations at each scale, colored by mechanism-of-action class. (b) Pretraining loss vs. training FLOPs. (c) Six-class mechanism-of-action macro precision from a parameter-free signature, mean over ten evaluation splits. (d) Zero-shot orthosteric vs. allosteric AUROC on 6,032 OOD complexes. Morgan fingerprint baseline shown dashed. (e) Zero-shot covalent vs. non-covalent AUROC on the OOD complexes in which neither class carries a detectable electrophilic warhead. All capability evaluations use parameter-free signatures derived from frozen representations and applied unchanged across scales.

## 1.1 Related Work

Structure-based interaction modeling. Co-folding methods [8–10] predict 3D coordinates of protein-ligand complexes from structure or sequence inputs and have transformed binding pose prediction. However, capabilities beyond pose prediction require supervised heads: across recent structure-prediction models, afinity prediction is a fine-tuned regressor rather than a property recovered from the pretrained representation alone. Generalization is also limited: FoldBench reports substantial memorization on benchmarks with weak performance on unseen ligands [11]. In addition, these models systematically misplace allosteric ligands into orthosteric pockets [19].

Frozen-embedding interaction models. Paired protein-ligand encoders trained with contrastive objectives [20, 21] support binding-prediction retrieval at scale, but their representations have not been shown to learn capabilities beyond the binding signal they were trained on.

Task-specific interaction prediction. Pocket detection [13, 14] and allosteric site prediction [22] have historically been addressed by separately supervised models, each requiring residue-level annotations and 3D protein structures. Structure-based covalent docking tools [15] require the user to specify a priori that a covalent bond forms and predict only the resulting geometry, rather than identifying covalent engagement or the reactive residue.

Emergent capabilities in foundation models. Wei et al. [1] characterize emergent abilities as tasks not in the pretraining objective that appear with suficient scale. Schaefer et al. [23] argue that apparent emergence is often a metric artifact: capabilities scale smoothly, and discontinuities reflect threshold-imposing evaluation metrics.

## 2 Model

## 2.1 Architecture

Bonbon is a transformer-based model trained by self-supervised learning on paired proteinligand sequences. The architecture is designed for the many-to-many relationships inherent in molecular interaction data, with pre-layer normalization, rotary position embeddings, flash attention, and a 65,536-token context window suficient to encode the longest known protein sequences and accommodate multi-component inputs (Appendix Figure A1).

## 2.2 Training Data

Training data comprised of paired protein-ligand sequences from proprietary screening datasets, enriched with synthetic interaction data generated to match the distributional properties of experimentally measured binding data from that corpus. The generation methodology is outside the scope of this paper and is the subject of a separate technical report. Every evaluation in this paper uses real experimental data from public databases and the published literature; no synthetic data appears in any evaluation set. The corpus spans soluble enzymes, membrane receptors, ion channels, and transcription factors, with ligand chemistries ranging from small molecules to peptides and protein-protein interactions.

Bonbon was trained on approximately 3.2 trillion tokens drawn from this corpus. The sampling distribution was designed to ensure diverse representation across both protein and ligand space, attenuating bias toward well-studied therapeutic targets.

Multiple quality filters were applied. PAINS-pattern compounds [24] and other reactive or aggregator-prone substructures are excluded to avoid pan-assay artifacts. Proteins with disproportionately broad ligand sets were removed to prevent their patterns from dominating representations. Sequence-level redundancy filters were applied to both the protein and ligand sides of the corpus. All proteins sharing greater than 90% sequence identity with any evaluation target were excluded prior to training as the basis for all OOD evaluation reported in subsequent Sections 3 and 4.

## 2.3 Model Training and Scaling Experiment

The 1.6B production model uses 3.2T training tokens, a 1:2,000 parameter-to-token ratio.   
Hardware, optimizer, and training schedule are described in Appendix A.1.

We chose this data-heavy ratio to accommodate the information sparsity of paired interaction data. To characterize how capabilities relate to scale, we trained six model scales spanning approximately three orders of magnitude in compute (42M, 315M, 800M, 1B, 1.3B, 1.6B parameters). We then evaluated each scale on (i) pretraining loss, (ii) zero-shot mechanismof-action macro precision, (iii) zero-shot orthosteric/allosteric AUROC on OOD complexes, and (iv) zero-shot covalent vs. non-covalent AUROC on OOD complexes, with and without a detectable electrophilic warhead. Both AUROC evaluations use the parameter-free signatures of Sections 3.2 and 3.3, with signature features and signs derived once and applied unchanged across all six scales, isolating capability emergence from any signature-tuning efect. All models were trained on the same corpus, sub-sampled to roughly preserve the data-heavy regime, with the same sampling procedure drawing from the full corpus at every scale, so the data mixture, including the proportion of real and synthetic data, is preserved. The architecture family, optimizer configuration and warmup schedule match the 1.6B model, and only the parameter count and its corresponding token budget difer. Each run trained on a predetermined sample from the corpus, with no early stopping or convergence-based termination, and every capability was evaluated with the same protocol on the same held-out sets at every scale. Each scale is a single training run. Training-seed variability was not measured.

Pretraining loss decreases monotonically and continuously with compute, falling from 6.94 at 42M to 1.44 at 1.6B (Figure 1b). Each of the zero-shot capabilities shows a sharp jump at 1.6B, with the largest absolute increase concentrated at the final scale across all three. All capabilities are absent at the five sub-1.6B scales despite identical training conditions, confirming they are a property of the largest scale rather than of the corpus or training procedure alone. The capabilities show qualitatively diferent behavior: MoA macro precision stays between 0.567 and 0.608 across the five sub-1.6B scales (0.567, 0.598, 0.598, 0.586, 0.608), then jumps to 0.811 at 1.6B (Figure 1c). Orthosteric/allosteric AUROC stays at or near the Morgan fingerprint baseline (0.589) across the five smaller scales (0.609, 0.600, 0.646, 0.641, 0.602), then jumps to 0.834 (Figure 1d). Covalent AUROC measured on all molecules moves within a narrow band across the smaller scales (0.645, 0.678, 0.691, 0.675, 0.695) before reaching 0.838 at 1.6B (Appendix E.6). Ablating warhead presence, we evaluated only complexes in which neither class carries a detectable electrophile. AUROC on this subset is flat at 0.638, 0.645, 0.649, 0.659 and 0.645, then jumps to 0.775 at 1.6B (Figure 1e). All three sub-threshold series are non-monotone. Against log FLOPs their slopes are 0.012 (mechanism of action), 0.008 (orthosteric/allosteric) and 0.005 (covalent) per tenfold increase in compute, against 0.682, 0.780 and 0.437 respectively for the 1.3B to 1.6B step. Each capability is measured on a non-overlapping protein set, ensuring independence across evaluations, and on a continuous metric, addressing the criticism that apparent emergence can be an artifact of discontinuous evaluation metrics [23].

Loss-based scaling, the standard quantity used to predict compute-optimal training [18], is monotone across the six scales and gives no signal that capabilities are about to appear. Selecting a model size on the basis of loss alone would not reveal whether the next compute step crosses a capability threshold; the threshold is visible only when capabilities are measured directly. The 1.6B scale is the smallest configuration in this experiment at which the threshold is crossed. We report where the transition occurs, not why it occurs. A plausible hypothesis consistent with these observations is that recognizing an interaction property requires composing several pieces of information, none of which is a local property of either sequence, following the intuition of Wei et al. [1] about depth suficient for sequential composition (Appendix A.4).

## 3 Emergent Capabilities

Bonbon’s frozen representations encode biological information about molecular interactions at three hierarchical levels of granularity: functional outcome, binding site identity, and bond type. None of these properties is determined by either the protein or the molecule alone, making each a property of the interaction that unimodal models cannot capture. Bonbon’s training data contained no task-specific labels for any of these properties. All three emerge only at the largest model scale (Section 2.3).

## 3.1 Mechanism of action prediction

Mechanism of action describes the functional outcome when a molecule binds a target (e.g., pathway activation vs. suppression). UMAP projection of Bonbon’s interaction representations across six mechanism classes (inhibitor, agonist, antagonist, activator, substrate, product) visualizes the hierarchical organization, separating regulatory from metabolic mechanisms (Figure 1a). A 160-feature parameter-free signature (107,027 interactions, molecules below 0.60 Tanimoto similarity to training data; Appendix B.1) yields 81.1% macro precision.

To eliminate the confound that the model exploits protein-family shortcuts, we evaluated within-receptor agonist/antagonist discrimination on 28,126 interactions across 222 GPCRs where every receptor has both classes (Appendix B.3). Holding the receptor constant acts as an ablation for protein identity as a possible shortcut for agonist/antagonist discrimination. Bonbon achieves 85.5% balanced accuracy (Figure 2b). No prior computational method has demonstrated zero-shot mechanism-of-action prediction across protein families.

## 3.2 Binding site localization and orthosteric/allosteric discrimination

Having demonstrated functional-level information, we tested whether Bonbon also encodes where on the protein binding occurs, despite no residue-level annotations and no structura information in the training data. Bonbon’s attention weights correlate linearly with ligand proximity (Figure 2c). Across 1,176 co-crystal structures spanning nine protein families (Appendix C), approximately 25% of attention heads self-organize for spatial localization, achieving $\mathrm { F } 1 = 0 . 3 8 1 \pm 0 . 1 4 2$ (in-distribution, ID) and 0.388 ± 0.124 (OOD), consistent across families including nuclear receptors $( \mathrm { F 1 } =  { \mathrm { 0 . 3 9 8 } }  { }  { \mathrm { O O D } } )$ and GPCRs (0.393 OOD), where co-folding methods are known to place ligands unreliably [25] (Appendix Table A6). Four families perform comparably or better on OOD than on training proteins, ruling out overfitting. On COACH420 [14], Bonbon achieves $\mathrm { F 1 } = \ 0 . 4 9 7 .$ , outperforming all three baselines (Appendix Table A7), while using sequence alone, without the residue-level supervision, 3D coordinates, or pocket annotations the baselines require. The 1,176-structure set described above is a harder test than COACH420, spanning targets such as GPCRs and transporters, whose binding sites are less well defined than the enzyme pockets that dominate COACH420.

A  
![](images/e263ad9dd696b615a3482c6c8b5f12e580064ec322f88f5adf2a8080abe4a9ff.jpg)  
E

B  
![](images/5d92ab3f1e76523c736a94f9056288ac003ce9d2c3a7145385f29ea95d3b74b9.jpg)  
C

![](images/7fab8c4497de3e0e1596cb85ae32d3a5406ce128cdb5119d603d7f34eb61e88e.jpg)

![](images/1ae946efd551384dea4f54f1d7cec9dd1d84d18e39a9421b482c5639784e3f00.jpg)

![](images/56c0157a9e8c268dd2ccf2fe738f38d21e6789c04a1a85e1084eb34e91575d97.jpg)  
G

![](images/505de3f639cfc67221d13bd92ab51128e38d58686bfd1c4fc7c16a7907d7d8cf.jpg)

![](images/ad96f0c5422ee0dc4b55f595124ab4908ef732a4236ac9d2370183611831860a.jpg)

![](images/1717225b27d83a06082d369ffb771374909946c6c5f37ba156ed6891bda05d02.jpg)

H  
![](images/c9815d1000903e6b2dc6f5d993344fe25a77938c93530a7bc86d82f0dcb3051a.jpg)

J  
![](images/e631f0cf37d222945a84004d4e4f39de1623a00606091be37aa8ade862dfd870.jpg)

K  
![](images/e1453545377ae0f9ee930b4de68fa8eb19a37b3bd7f2ad2e963a4b0348dea507.jpg)

L  
![](images/853b71f0192dbc620d11c0c0253c12b87d9b19b25b99bffa419f852bab6e773a.jpg)  
M

![](images/d8684f823f15f2401e9187339a0f47c273fabaa63bfdd3f7374eda276a135c7c.jpg)  
Figure 2: Emergent capabilities across three hierarchical levels, all from frozen representations. Mechanism of action: (a) Six-class signature confusion matrix, column-normalized, mean over ten evaluation splits of a $^ { 1 0 7 , 0 2 7 - } \mathrm { p a i r }$ OOD interaction set. (b) Within-receptor agonist/antagonist discrimination across 222 GPCRs. Binding site localization and orthosteric/allosteric discrimination: (c) Per-residue attention vs. ligand distance, ID and OOD. (d) Per-family binding site F1, ID and OOD. (e) Orthosteric/allosteric AUROC on 6,032 OOD complexes, per family. (f) Within-protein classification on 1,860 OOD complexes. (g) Signature score distributions by binding-mode class. (h) Signed feature vector: orthosteric-associated (blue) and allosteric-associated (red) features, by protein family. Covalent binding mode: (i) Activation frequency of the 36 signature features in covalent versus non-covalent complexes, sorted by covalent enrichment. (j) AUROC under full, warhead-matched, and warhead-free conditions. (k) Warhead-controlled comparison. (l) Same-residue-type classification for serine, cysteine, and lysine. (m) Top-1 reactive residue localization accuracy.

We next tested whether Bonbon distinguishes orthosteric from allosteric binding. Orthosteric ligands occupy the primary endogenous binding site; allosteric modulators bind secondary pockets that ofer higher target specificity. From Bonbon’s frozen interaction representations, we identified a 145-feature parameter-free signature whose discriminative direction between orthosteric and allosteric complexes is consistent across protein families (Appendix D.2).

Across 6,032 OOD complexes (141 OOD proteins), the signature achieves AUROC 0.834 versus 0.589 for Morgan fingerprints (Figure 2e). On the harder task of classifying binding mode among ligands of the same target (where proteins with both orthosteric and allosteric ligands act as an ablation of using protein identity as a shortcut), Bonbon achieves AUROC 0.821 across 1,860 OOD complexes (Figure 2f); on OOD membrane receptors, 0.788. Transcription factors are near chance (0.548), consistent with their smaller training representation. To test whether the site-detection heads causally support this discrimination, we ablated them at inference time and re-ran the same OOD evaluation. Ablating the 25% site-detection heads drops orthosteric/allosteric AUROC from 0.834 to 0.751; ablating 25% of heads at random gives $0 . 8 1 6 \pm 0 . 0 0 2$ across 10 seeds, so removing the site-detection heads costs roughly 4.5 times what removing an arbitrary subset of the same size costs. The capability degrades but does not disappear: at 0.751 the ablated model remains well above the Morgan fingerprint baseline of 0.589 (Appendix D.5). No prior computational method has demonstrated orthosteric/allosteric discrimination per ligand.

## 3.3 Covalent binding mode recognition

Binding site localization identifies where a molecule binds; a finer question is how: whether engagement involves covalent bond formation. Covalent drugs can reach targets considered undruggable. Using a 36-feature parameter-free signature from frozen representations, Bonbon achieves AUROC 0.838 on 1,212 OOD pairs spanning 237 proteins (Figure 2j). Two ablations remove warhead presence as an explanation. Among molecules that all carry electrophilic warheads, Bonbon separates those that form covalent bonds from those that do not (AUROC 0.834); among molecules that carry none, it still identifies the covalent binders, whose bonds form through the protein’s catalytic machinery rather than ligand reactivity (AUROC 0.775; Appendix E.2). Per-residue representations distinguish the reactive residue from non-reactive same-type residues (serine AUROC 0.902, cysteine 0.817, lysine 0.701), persisting after distance matching (Figure 2l; Appendix E.5). A two-step pipeline that predicts amino-acid type from pair-level features then ranks pocket residues of the predicted type achieves 71.8% Top-1 localization accuracy without prior knowledge of the reactive residue (Figure 2m), a 12-fold improvement over the 5.8% chance baseline. No prior computational method has demonstrated covalent engagement prediction from sequence alone.

## 4 Capabilities Translate to Predictive Accuracy and Novel Inhibitors

The three emergent capabilities (Section 3) demonstrate that Bonbon’s representations encode interaction-level biological information without task-specific training. Two further evaluations test whether this information translates to quantitative predictive accuracy on standard benchmarks (Section 4.1) and to biochemical activity validated in prospective wet-lab experiments (Section 4.2).

## 4.1 Binding and afinity prediction at five orders of magnitude greater speed

For binary binding prediction we use frozen embeddings directly. For binding afinity prediction we train a two-layer MLP probe [26] for ∼12 minutes on frozen embeddings using BindingDB [27]. Decontamination protocol and dataset preparation are described in Appendix G.

On MF-PCBA, Bonbon achieves state-of-the-art AUPR 0.081 and EF 0.5% of 24.8 (Figure 3b,c; full per-method comparisons in Appendix Tables A14–A16). On LIT-PCBA, Bonbon achieves state-of-the-art EF1% 11.4, more than double the previous best (Appendix Table A14). On LIT-PCBA-AF3 [28], a five-target subset of LIT-PCBA restricted to libraries of fewer than 100,000 compounds and therefore tractable for co-folding methods, Bonbon achieves state-of-the-art EF1% 16.6, against 9.3 for the strongest co-folding baseline (Figure 3d; Appendix Table A15). On afinity prediction, Bonbon’s MLP probe achieves Pearson $r = 0 . 7 4$ on CASP16, $r = 0 . 7 8$ on $\mathrm { \dot { F } E P + 4 }$ , and $r = 0 . 6 9$ on OpenFE, matching physics-based free energy perturbation and surpassing co-folding methods (Figure 3a).

Accuracy is accompanied by orders-of-magnitude reduction in compute. Bonbon processes ∼2,500 pairs/s on-the-fly or ∼42,500 pairs/s with cached representations on a single H100, approximately five orders of magnitude faster than Boltz-2’s 0.05 pairs/s (Figure 3e). At proteome scale, screening DrugBank against the human proteome (1.66B pairs) completes in 11 hours versus an estimated ∼1,100 GPU-years for Boltz-2 (Appendix Table A17).

A  
![](images/1e31702327c54d37ff4a4e2d78c3cc88ceec1784fd2144f158254fb30aaf5e3f.jpg)

![](images/a013fa0d8b437653a357df3a6030669894959c86e840a1ff87292d0742843ce4.jpg)

![](images/751de1f7b1d8ffae178723287a18ec32a5b6c8a0cd437ba2f66cf252798833ff.jpg)

![](images/ae3bbd164b501ecee4067dd9d71dcaa2c3e6b296350e678b9257c9a02e1d9921.jpg)

![](images/fefaad9f197287bfa12f6d791b5331f16c25f458dba2cc78462646e4b25c037c.jpg)

![](images/46fb019b1272c75da3aeb7c2bb6dc7cb85597cfe43e05a67d7388cff7ef96515.jpg)

![](images/24207dd17d491947f21a0cf7ac14ade9f32eb5fe3e20748b6bbfc402324e09d4.jpg)  
Figure 3: Predictive accuracy and throughput. (a) Binding afinity prediction (Pearson r) on CASP16, FEP+4, and OpenFE. (b) MF-PCBA AUPR. (c) MF-PCBA enrichment factor across thresholds. (d) LIT-PCBA-AF3 enrichment factor at 1%. (e) Inference throughput: pairs evaluated per hour on a single H100, compared to co-folding and docking methods.

## 4.2 Prospective wet-lab experimental validation across nine disease targets

We used Bonbon to rank molecules generated de novo under synthetic-accessibility constraints against nine disease targets, and up to 24 compounds per target from the top-ranked set were synthesized and assayed, with dose-response $\mathrm { I C } _ { 5 0 }$ determination for actives (Figure 4; Appendix H). All tested compounds have Tanimoto similarity less than 0.60 to any compound in Bonbon’s training data. Structural novelty is assessed as described (Appendix H.3). The nine targets isolate three progressively harder tests.

Bonbon yielded confirmed inhibitors for all nine targets, with $\mathrm { I C } _ { 5 0 }$ values reaching singledigit nanomolar and, on several targets, chemical matter outside the established inhibitor classes.

Novel and eficient inhibitors on well-characterized targets. TYK2, PRMT5, JAK1 and CDK1 have rich SAR and known pharmacophore classes; they test whether Bonbon identifies eficient and novel inhibitors within established chemical space. On TYK2, 23 compounds were confirmed as inhibitors in dose-response assays, with $\mathrm { I C } _ { 5 0 }$ values from 146 nM to $4 0 \mu \mathrm { M }$ and four below $1 \mu \mathrm { M }$ . None is a single-substituent analog of a known inhibitor and $1 7$ are novel molecules. On PRMT5, 16 compounds were confirmed inhibitors, with $\mathrm { I C } _ { 5 0 }$ values from 186 nM to 143 µM. All 16 sit at 0.479–0.597 Tanimoto to the nearest known active and none is a single-substituent analog of any known inhibitor: Bonbon reached the productive region of PRMT5 chemistry without reproducing any individual compound in ${ \mathrm { i t } } ,$ in a single ranked set and without iterative SAR. On JAK1, 21 compounds were confirmed inhibitors, with $\mathrm { I C } _ { 5 0 }$ values from 3 nM to 28 µM, 13 below $1 \mu \mathrm { M }$ and three below 100 nM. Four are novel molecules, and single-digit nanomolar potency was reached without optimization. On CDK1, all 22 compounds tested were confirmed inhibitors, with $\mathrm { I C } _ { 5 0 }$ values from 114 nM to $2 3 \mu \mathrm { M }$ and eight below $1 \mu \mathrm { M } ;$ 10 are novel molecules, at 0.279–0.706 Tanimoto to the nearest known active.

New chemistry beyond established pharmacophores. HDAC6, MNK2 and MMP-9 have inhibitor literatures confined to a narrow region of chemical space; they test whether the model reaches chemical matter that is genuinely new rather than rewarding familiar pharmacophores. The HDAC inhibitor field has been bottlenecked by hydroxamic acid toxicity for two decades: hydroxamic acid appears in every FDA-approved HDAC inhibitor but causes dose-limiting toxicity that has stalled clinical programs. For HDAC6, 15 compounds were confirmed inhibitors in dose-response assays, with $\mathrm { I C } _ { 5 0 }$ values from 115 nM to $2 2 \mu \mathrm { M } ;$ 12 of the 15 lack hydroxamic acid entirely, and those same 12 are novel molecules. On MNK2, a kinase with no approved inhibitor and only two disclosed clinical programs, 19 compounds were confirmed inhibitors, with $\mathrm { I C } _ { 5 0 }$ values from 39 nM to $3 8 \mu \mathrm { M } .$ . Eighteen are novel molecules, at 0.269–0.492 Tanimoto to any known MNK2 active with no single-substituent analog among them, the strongest structural novelty in the panel. On MMP-9, where known inhibitors rely on zinc-chelating warheads and broad-spectrum inhibition has produced dose-limiting musculoskeletal toxicity attributed to cross-reactivity within the 23-member MMP family, three compounds were confirmed inhibitors, from 480 nM to $1 . 3 \mu \mathrm { M } ;$ none carries a strong zinc-chelating warhead (Appendix H.4).

A  
![](images/7b81f3cf8bab36ce0cfecfae8008234a5ec46f211a982b72ae9bbaccfd71001e.jpg)

B  
![](images/44aa7b2eafefcd98b0b24f60acf89da516b0e2385b2e57ad5436173d0339f8bb.jpg)

C  
![](images/9b729bcbd40e5894aac0f99e2292a226ac225005a3d37d3be56c8a3c0bc1961e.jpg)

D  
![](images/254531aa5ce30228bce4f2428f98f7eaca5528ca46577daab468d62c621b48f7.jpg)

E  
![](images/4603ce9886e0bc61f7ac2e3e88eb5c2e404c58c29dee20e7797b9544bf680f77.jpg)

F  
![](images/1f5ff5ce7463e7c4a6c824a0e7f08d821e98d30d197a591d1898bb3d8880a19e.jpg)

G  
![](images/c077a9d7b97f9d355ead89e0c1c61eac3f08047f759dfb3314235c05c1284518.jpg)

H  
![](images/3e416589f0cf074cd576bc2305ddea8ec1a8c703790aa429e2b621cb2e67dc93.jpg)

一  
![](images/5c6b01533ef83bd99111448bcfb40d65e74e04fc36e4e072a92bc41e0b56223c.jpg)  
Figure 4: Prospective wet-lab validation across nine disease targets spanning seven protein families. For each target, a ten-point dose-response curve with fitted $\mathrm { I C } _ { 5 0 }$ for one confirmed inhibitor (left) and that compound bound in the target (right; protein ribbon within a transparent surface, compound in orange; Appendix Table A18).

Out-of-distribution targets. ELANE and VEGFR2 are absent from Bonbon’s training data entirely, with no protein above 90% sequence identity in training. On ELANE, a serine protease, Bonbon yielded two confirmed inhibitors at 1.22 and 29.4 µM, both at or below 0.500 Tanimoto to the nearest known active and neither a single-substituent analog. On VEGFR2, 10 compounds were confirmed inhibitors in dose-response assays, with $\mathrm { I C } _ { 5 0 }$ values from 38 nM to $8 2 \mu \mathrm { M } ;$ they sit at 0.429–0.617 Tanimoto to the nearest known active, with one novel molecule by every criterion (Appendix H.4).

Across a target panel spanning seven protein families, including out-of-distribution targets, validated hits include molecules novel against the published inhibitor literature, evidence that the representations encode the functional requirements of target engagement rather than memorized pharmacophore patterns.

## 5 Discussion

Bonbon learned to predict the functional outcome when a protein and molecule interact, where on the protein, and how they engage, all within a single representation space. All three capabilities arise from frozen representations with no task-specific labels, and none has been demonstrated by any prior computational method. A controlled scaling experiment shows each capability emerges as a sharp jump at 1.6B on continuous metrics, addressing the metric-artifact concern [23].

Scaling. Previously reported foundation models exhibiting emergent capabilities converge on parameter-to-token ratios between 1:2 and 1:100. Bonbon’s 1:2,000 ratio reflects the sparsity of paired interaction data: the distinction between an agonist and an antagonist, for example, is distributed across millions of examples rather than concentrated in individual sequences, so we allocate far more data per parameter than language modeling conventionally requires. We arrived at this allocation through internal experimentation across token budgets rather than a fixed-compute sweep. A conventional IsoFLOP sweep optimizes pretraining loss [18], which is the quantity our scaling results indicate does not predict capability availability (Appendix A.4).

Implications for drug discovery. Each capability addresses an unmet need. Mechanismof-action filtering before synthesis replaces costly functional assays. Orthosteric/allosteric discrimination per ligand is a prerequisite for designing allosteric modulators, which ofer higher selectivity and access to targets considered undruggable at the orthosteric site. Covalent bond prediction enables both deliberate covalent design and early detection of of-target reactivity currently identifiable only through late-stage experimental assays.

These capabilities and the benchmarks arise from the same frozen representations in a single forward pass. Bonbon provides a computational foundation for drug discovery that replaces sequential, fragmented, structure-dependent tools and laboratory-only measurements with a single representation space, operating at up to 42,500 pairs per second. The practical consequence is a shorter design-make-test cycle: interaction properties that today require co-crystal structures, functional assays, or iterative SAR are available computationally before a molecule is synthesized, enabling the evaluation of more hypotheses and more ambitious ones. Appendix F reports an additional analysis at atomic resolution, identifying which atoms within a ligand drive its activity. The prospective wet-lab validation across nine targets confirms this translates to practice, including validated hits that are novel against the published inhibitor literature, and generalization to out-of-distribution proteins.

Limitations. The capabilities reported here resolve coarse distinctions; finer granularity and explicit 3D coordinate extraction remain for future work. All results here are obtained from frozen representations: Bonbon has not been fine-tuned on any downstream task, so what we report is a lower bound on what these representations encode. Additional interaction properties such as residence time, dissociation kinetics, and how conformation and function change after binding are not explored in this work.

Conclusion. Bonbon demonstrates that the functional grammar of molecular interactions is encoded in the sequences of interacting molecules and recoverable at suficient scale, without task-specific labels or structural coordinates. The capabilities generalize to out-ofdistribution complexes and translate to lab-validated novel inhibitors across nine disease targets. Bonbon’s representation space and the speed at which it operates open directions beyond pairwise prediction, such as generative molecular design conditioned on mechanistic intent, where mechanism, binding site, and binding mode constraints are enforced during generation rather than screened after synthesis; and systems-level interaction profiling, where the computational cost of characterizing every protein-ligand pair in a pathway becomes negligible, enabling the design of molecules with predetermined multi-target profiles and explicit selectivity across protein families. Characterizing not just whether molecules interact but how, at this scale, is a foundation for understanding biological systems.

Safety and ethics. A model that predicts interaction properties, covalent engagement, and mechanism of action could in principle guide the design of molecules with harmful biological activity. We mitigate this by not releasing model weights or training data publicly; access for academic, non-commercial research is granted case by case. Bonbon predicts interaction properties, not organismal toxicity; translating any prediction to a functional molecule requires synthesis, formulation, and safety testing outside the model’s scope.

## Acknowledgments

We would like to thank Itsik Pe’er and Pranam Chatterjee for insightful discussions and valuable feedback on the manuscript. We are especially grateful to Mirella Mashiach for her thoughtful guidance and feedback throughout this work.

## References

[1] Jason Wei, Yi Tay, Rishi Bommasani, Colin Rafel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent Abilities of Large Language Models, 2022.

[2] T. B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language Models are Few-Shot Learners, 2020.

[3] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H., Quoc V. Le, and Denny Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, 2022.

[4] Gemini Robotics Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024.

[5] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. PaLM-E: An Embodied Multimodal Language Model, 2023.

[6] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 2023. doi: 10.1126/sc ience.ade2574.

[7] Eric Nguyen, Michael Poli, Matthew G. Durrant, Brian Kang, Dhruva Katrekar, David Li, Liam J. Bartie, Armin W. Thomas, S. B. King, Garyk Brixi, et al. Sequence modeling and design from molecular to genome scale with Evo. Science, 2024. doi: 10.1126/science.ado9336.

[8] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J. Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 2024. doi: 10.1038/s41586-024-07487-w.

[9] Saro Passaro, Gabriele Corso, Jeremy Wohlwend, Mateo Reveiz, Stephan Thaler, Vignesh Ram Somnath, Noah Getz, Tally Portnoi, Julien Roy, H. Stärk, et al. Boltz-2: Towards Accurate and Eficient Binding Afinity Prediction. bioRxiv (Cold Spring Harbor Laboratory), 2025.

[10] Chai Discovery Team, Jacques Boitreaud, Jack Dent, Danny Geisz, Matthew McPartlon, Joshua Meier, Zhuoran Qiao, Alex Rogozhnikov, Nathan Rollins, Paul Wollenhaupt, and Kevin Wu. Zero-shot antibody design in a 24-well plate. bioRxiv, 2025.

[11] Sheng Xu, Qili Feng, Lifeng Qiao, Hao Wu, Tao Shen, Yu Cheng, Shuangjia Zheng, and Siqi Sun. Benchmarking all-atom biomolecular structure prediction with FoldBench. Nature Communications, 2025. doi: 10.1038/s41467-025-67127-3.

[12] Peter Škrinjar, Jérôme Eberhardt, Gerardo Tauriello, Torsten Schwede, and Janani Durairaj. Have protein-ligand cofolding methods moved beyond memorisation? bioRxiv (Cold Spring Harbor Laboratory), 2025.

[13] Vincent Le Guilloux, Peter Schmidtke, and Pierre Tuféry. Fpocket: An open source platform for ligand pocket detection. BMC Bioinformatics, 2009. doi: 10.1186/1471-2 105-10-168.

[14] Radoslav Krivák and David Hoksza. P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure. Journal of Cheminformatics, 2018. doi: 10.1186/s13321-018-0285-8.

[15] Kai Zhu, Kenneth Borrelli, Jeremy R. Greenwood, Tyler Day, Robert Abel, Ramy Farid, and Edward Harder. Docking Covalent Inhibitors: A Parameter Free Approach To Pose Prediction and Scoring. Journal of Chemical Information and Modeling, 2014. doi: 10.1021/ci500118s.

[16] Antonio Peón, Stefan Naulaerts, and Pedro J. Ballester. Predicting the Reliability of Drug-target Interaction Predictions with Maximum Coverage of Target Space. Scientific Reports, 2017. doi: 10.1038/s41598-017-04264-w.

[17] Kevin Yang, Kyle Swanson, Wengong Jin, Connor W. Coley, Philipp Eiden, Hua Gao, Angel Guzmán-Pérez, Timothy Hopper, Brian Kelley, Miriam Mathea, et al. Analyzing Learned Molecular Representations for Property Prediction. Journal of Chemical Information and Modeling, 2019. doi: 10.1021/acs.jcim.9b00237.

[18] Jordan Hofmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training Compute-Optimal Large Language Models, 2022.

[19] Eva Nittinger, Özge Yoluk, Alessandro Tibo, Gustav Olanders, and Christian Tyrchan. Co-folding, the future of docking – prediction of allosteric and orthosteric ligands. Artificial Intelligence in the Life Sciences, 2025. doi: 10.1016/j.ailsci.2025.100136.

[20] Rohit Singh, Samuel Sledzieski, Lenore Cowen, and Bonnie Berger. Learning the Drug-Target Interaction Lexicon. bioRxiv (Cold Spring Harbor Laboratory), 2022.

[21] Bowen Gao, Qiang Bo, Haichuan Tan, Minsi Ren, Yinjun Jia, Minsi Lu, Jingjing Liu, Weiying Ma, and Yanyan Lan. DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening, 2023.

[22] Kamila Riedlová, Vít Škrhák, Will Gatlin, Max Ludwick, Lucas Turano, Marián Novotný, David Hoksza, and Gennady M. Verkhivker. Protein Language Models and Structure-Based Machine Learning for Prediction of Allosteric Binding Sites in Protein Kinases: An Explainable AI Framework Grounded in Energy Landscape-Encoded Frustration. bioRxiv (Cold Spring Harbor Laboratory), 2026. doi: 10.64898/2026.01.05.697819.

[23] Rylan Schaefer, Brando Miranda, and Sanmi Koyejo. Are Emergent Abilities of Large Language Models a Mirage?, 2023.

[24] Jonathan B. Baell and Georgina A. Holloway. New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS) from Screening Libraries and for Their Exclusion in Bioassays. Journal of Medicinal Chemistry, 2010. doi: 10.1021/jm901137j.

[25] Lichirui Zhang, Richard A. Friesner, Edward B. Miller, and João P. G. L. M. Rodrigues. On the generalization and usability of cofolding models for GPCR drug discovery. npj Drug Discovery, 3:30, 2026. doi: 10.1038/s44386-026-00066-1.

[26] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2016.

[27] Michael K. Gilson, Tiqing Liu, Michael Baitaluk, George Nicola, Linda Hwang, and Jenny Chong. BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology. Nucleic Acids Research, 2015. doi: 10.1093/nar/ gkv1072.

[28] C. Shen, X. Zhang, S. Gu, O. Zhang, Q. Wang, G. Du, Y. Zhao, L. Jiang, P. Pan, Y. Kang, Q. Zhao, C.-Y. Hsieh, and T. Hou. Unlocking the application potential of AlphaFold3-like approaches in virtual screening. Chemical Science, 17(5):2858–2879, 2025. doi: 10.1039/d5sc06481c.

[29] Amelie Schreiber. ESMBind and QBind: LoRA, QLoRA, and ESM-2 for Predicting Binding Sites and Post Translational Modification. bioRxiv, 2023. doi: 10.1101/2023.11. 13.566930.

[30] Luis A. Caro-Gómez, Jorge L. Rosas-Trigueros, Edgar Mixcoha, José Luis Vique-Sánchez, Humberto Gasperin-Sánchez, Claudia G. Benítez-Cardoza, and Absalom Zamorano-Carillo. Exploring the Conformational Space of Bcl-2 Protein Variants: Dynamic Contributions of the Flexible Loop Domain and Transmembrane Region. Molecules, 2019. doi: 10.3390/molecules24213896.

[31] Andrew M. Petros, A Medek, David G. Nettesheim, Daniel H. Kim, Ho Sup Yoon, Kerry M. Swift, Edmund D. Matayoshi, Tilman Oltersdorf, and Stephen W. Fesik. Solution structure of the antiapoptotic protein bcl-2. Proceedings of the National Academy of Sciences, 2001. doi: 10.1073/pnas.041619798.

[32] Gregory L. Verdine and Loren D. Walensky. The Challenge of Drugging Undruggable Targets in Cancer: Lessons Learned from Targeting BCL-2 Family Members. Clinical Cancer Research, 2007.

[33] Gilles J. P. Rautureau, Catherine L. Day, and Mark G. Hinds. Intrinsically Disordered Proteins in Bcl-2 Regulated Apoptosis. International Journal of Molecular Sciences, 2010.

[34] Yu-Jing Lan, Pei-Shan Yeh, Te-Yu Kao, Yuan-Chao Lo, Shih-Che Sue, Yuwen Chen, Dennis W. Hwang, and Yun-Wei Chiang. Anti-apoptotic BCL-2 regulation by changes in dynamics of its long unstructured loop. Communications Biology, 2020. doi: 10.103 8/s42003-020-01390-6.

[35] Vincent B. Luscombe, Daniel Lucy, Carole J. R. Bataille, Angela J. Russell, and David R. Greaves. 20 Years an Orphan: Is GPR84 a Plausible Medium-Chain Fatty Acid-Sensing Receptor? DNA and Cell Biology, 2020.

[36] Stephen P.H. Alexander, Chido Mpamhanga, Richard Neubig, Adam J. Pawson, Jean-Philippe Pin, et al. Class A Orphans in GtoPdb v.2025.4, 2025.

[37] Heng Liu, Qing Zhang, Xinheng He, Mengting Jiang, Siwei Wang, Xiaoci Yan, Xi Cheng, Yang Liu, Fajun Nan, H. Eric Xu, et al. Structural insights into ligand recognition and activation of the medium-chain fatty acid-sensing receptor GPR84. Nature Communications, 2023. doi: 10.1038/s41467-023-38985-6.

[38] Xuan Zhang, Yujing Wang, Shreyas Supekar, Xu Cao, Jingkai Zhou, Jessica Dang, Siqi Chen, Laura Jenkins, Sara Marsango, Xiu Li, et al. Pro-phagocytic function and structural basis of GPR84 signaling. Nature Communications, 2023. doi: 10.1038/s414 67-023-41201-0.

[39] Martin Steinegger and Johannes Söding. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature Biotechnology, 35(11):1026–1028, 2017. doi: 10.1038/nbt.3988.

[40] Andrew T. McNutt, Abhinav K. Adduri, Caleb N. Ellington, Monica T. Dayao, Eric P. Xing, Hosein Mohimani, and David R. Koes. Scaling Structure Aware Virtual Screening to Billions of Molecules with SPRINT. arXiv:2411.15418, 2024.

[41] Michael K. Gilson, Jérôme Eberhardt, Peter Škrinjar, Janani Durairaj, Xavier Robin, and Andriy Kryshtafovych. Assessment of Pharmaceutical Protein–Ligand Pose and Afinity Predictions in CASP16. Proteins Structure Function and Bioinformatics, 2025. doi: 10.1002/prot.70061.

[42] David F. Hahn, Christopher I. Bayly, Melissa L. Boby, Hannah E. Bruce Macdonald, John D. Chodera, Vytautas Gapsys, Antonia S. J. S. Mey, David L. Mobley, Laura Pérez-Benito, Christina Schindler, et al. Best Practices for Constructing, Preparing, and Evaluating Protein-Ligand Binding Afinity Benchmarks [Article v1.0]. Living Journal of Computational Molecular Science, 2022. doi: 10.33011/livecoms.4.1.1497.

[43] Leon Qiao, Wayne Bai, He Yan, Gary Liu, Nova Xi, Xiang Zhang, and Siqi Sun. IntFold: A Controllable Foundation Model for General and Specialized Biomolecular Structure Prediction. arXiv:2507.02025, 2025.

[44] SandboxAQ. AQAfinity: Open-Source Structure-Free Afinity on OpenFold3. SandboxAQ blog post, January 2026, 2026. URL https://www.sandboxaq.com/post/aqa ffinity-open-source-structure-to-affinity-built-on-openfold3-by-sandb oxaq.

[45] Matteo Rossi, Ryan Pederson, Miles Wang-Henderson, Ben Kaufman, Edward C. Williams, Carl Underkofler, Owen Lewis Howell, Adrian Layer, Stephan Thaler, Narbe Mardirossian, and John Anthony Parkhill. TerraBind: Fast and Accurate Binding Afinity Prediction through Coarse Structural Representations. arXiv:2602.07735, 2026.

[46] Andrew T. McNutt, Abhinav K. Adduri, Caleb N. Ellington, Monica T. Dayao, Eric P. Xing, Hosein Mohimani, and David R. Koes. Scaling Structure Aware Virtual Screening to Billions of Molecules with SPRINT. arXiv:2411.15418, 2024.

[47] Jameed Hussain and Ceara Rea. Computationally Eficient Algorithm to Identify Matched Molecular Pairs (MMPs) in Large Data Sets. Journal of Chemical Information and Modeling, 50(3):339–348, 2010. doi: 10.1021/ci900450m.

[48] Andrew Dalke, Jerome Hert, and Christian Kramer. mmpdb: An Open-Source Matched Molecular Pair Platform for Large Multiproperty Data Sets. Journal of Chemical Information and Modeling, 58(5):902–910, 2018. doi: 10.1021/acs.jcim.8b00173.

[49] Theonie Anastassiadis, Sean Deacon, Karthik Devarajan, Haiching Ma, and Jefrey R. Peterson. Comprehensive assay of kinase catalytic activity reveals features of kinase inhibitor selectivity. Nature Biotechnology, 2011. doi: 10.1038/nbt.2017.

[50] Xianglei Jin, Ruijuan Yu, Xiaomei Wang, Christopher G. Proud, and Tao Jiang. Progress in developing MNK inhibitors. European Journal of Medicinal Chemistry, 219:113420, 2021. doi: 10.1016/j.ejmech.2021.113420.

[51] Judith A. Voynow and Meagan Shinbashi. Neutrophil Elastase and Chronic Lung Disease. Biomolecules, 2021.

## Appendix

1 Introduction 1   
1.1 Related Work . 3   
2 Model 4   
2.1 Architecture 4   
2.2 Training Data 4   
2.3 Model Training and Scaling Experiment 4   
3 Emergent Capabilities 5   
3.1 Mechanism of action prediction 5   
3.2 Binding site localization and orthosteric/allosteric discrimination 5   
3.3 Covalent binding mode recognition 7   
4 Capabilities Translate to Predictive Accuracy and Novel Inhibitors 7   
4.1 Binding and afinity prediction at five orders of magnitude greater speed . . 7   
4.2 Prospective wet-lab experimental validation across nine disease targets . 8   
5 Discussion 9   
A Model architecture, training, and scaling experiment 17   
A.1 Hardware, optimizer, and training schedule 17   
A.2 Training-scale reference table 17   
A.3 Scaling experiment configurations 18   
A.4 On the mechanism of emergence 18   
B Mechanism of action: evaluation datasets 19   
B.1 Six-class mechanism dataset . 19   
B.2 Mechanism overlap at the molecule level 20   
B.3 GPCR agonist/antagonist evaluation dataset 20   
C Binding site localization: dataset, head selection, and per-family results 21   
C.1 Dataset construction and methodology 21   
C.2 Head selection 21   
C.3 Per-family binding site prediction 21   
C.4 COACH420 comparison 22   
C.5 Structurally challenging targets 22   
D Orthosteric/allosteric discrimination: dataset, signature construction, and   
analysis 24   
D.1 Dataset 24   
D.2 Signature construction 24   
D.3 Classification results 24   
D.4 Score distribution by family 24   
D.5 Head ablation 25   
E Covalent binding mode recognition: dataset, controls, and residue localiza  
tion 25   
E.1 Dataset construction 25   
E.2 Warhead controls 26   
E.3 Classification methodology 26   
E.4 Reactive residue localization 27   
E.5 Distance-controlled analysis 27   
E.6 Covalent recognition across model scales 27   
F Active moiety identification: dataset construction, metric, and generaliza  
tion 28   
G Computational benchmarks: protocols, datasets, and per-method compar  
isons 31   
G.1 Decontamination 31   
G.2 Dataset preparation 32   
G.3 Afinity benchmark configurations . 32   
G.4 Afinity probe 32   
G.5 Binary binding prediction: full per-method comparison . 33   
G.6 Binding afinity prediction: per-target results and full per-method comparison 33   
G.7 Inference throughput . 33   
H Prospective experimental validation: methods and per-target detail 34   
H.1 Compound ranking and selection 34   
H.2 Compound sourcing and handling . 35   
H.3 Novelty analysis 35   
H.4 Per-target results 35   
H.5 TYK2 kinase assay 36   
H.6 PRMT5 methyltransferase assay 36   
H.7 JAK1 kinase assay 36   
H.8 CDK1 kinase assay 36   
H.9 HDAC6 deacetylase assay 37   
H.10 MNK2 kinase assay 37   
H.11 MMP-9 protease assay 37   
H.12 ELANE (neutrophil elastase) protease assay 37   
H.13 KDR/VEGFR2 kinase assay 37

This appendix provides supplementary methodological and experimental detail referenced throughout the main text. Sections are lettered A through H in the order they are first referenced from the main text. All figures and tables in this appendix are prefixed $^ { 6 } \mathrm { A } ^ { \dag }$ (e.g., Table A1, Figure A1) and are numbered sequentially within the appendix.

## A Model architecture, training, and scaling experiment

## A.1 Hardware, optimizer, and training schedule

Bonbon was trained on a cluster of NVIDIA H100 GPUs using PyTorch 2.0 with Fully Sharded Data Parallel (FSDP). Optimization used AdamW with a learning rate of $1 \times 1 0 ^ { - 3 }$ and 10,000 warmup steps. The model uses pre-layer normalization, rotary position embeddings, flash attention, and a 65,536-token context window.

![](images/7281dc29d602d04c5c8e0f7cea044b854bb4af37d0470b014084746e1c1750b8.jpg)

![](images/1e89df9b48b2894338aa4bab98c007332f5b33f780e3d9fbce539904dd2a4660.jpg)  
Figure A1: Bonbon design and scale. (a) Schematic of the input-to-prediction pipeline: paired protein and ligand sequences are jointly encoded by a transformer trained with a self-supervised objective, yielding the interaction representations from which the zero-shot capabilities of Section 3 are read out. (b) Parameter-to-token scatter for representative foundation models. Bonbon sits in a data-heavy regime relative to Chinchilla-optimal scaling. Reference table: Table A1.

## A.2 Training-scale reference table

Table A1 lists training scale and compute for representative foundation models in language and biology, together with Bonbon. Figure A1b plots the parameter-to-token ratios from this table.

Table A1: Training scale of Bonbon compared to representative foundation models. Bonbon’s 1:2,000 parameter-to-token ratio places it in a data-heavy regime distinct from language and unimodal biology models.
<table><tr><td>Model</td><td>Domain</td><td>Params</td><td>Tokens</td><td>Param:Token</td><td>FLOPs</td><td>Emergent capabilities</td></tr><tr><td>GPT-2</td><td>Language</td><td>1.5B</td><td>10B</td><td>1:7</td><td> $9 \times 1 0 ^ { 1 9 }$ </td><td>Coherent generation</td></tr><tr><td>ESM-2</td><td>Proteins</td><td>650M</td><td>65B</td><td>1:100</td><td> $2 . 5 \times 1 0 ^ { 2 0 }$ </td><td>Weak structure</td></tr><tr><td>ESM-2</td><td>Proteins</td><td>15B</td><td>65B</td><td>1:4</td><td> $5 . 8 \times 1 0 ^ { 2 1 }$ </td><td>Structure prediction</td></tr><tr><td>Chinchilla</td><td>Language</td><td>70B</td><td>1.4T</td><td>1:20</td><td> $5 . 9 \times 1 0 ^ { 2 3 }$ </td><td>Few-shot reasoning</td></tr><tr><td>GPT-3</td><td>Language</td><td>175B</td><td>300B</td><td>1:2</td><td> $3 . 1 \times 1 0 ^ { 2 3 }$ </td><td>In-context learning</td></tr><tr><td>Evo</td><td>DNA/RNA</td><td>7B</td><td>300B</td><td>1:43</td><td> $1 . 3 \times 1 0 ^ { 2 2 }$ </td><td>Gene function</td></tr><tr><td>Bonbon</td><td>Protein-ligand</td><td>1.6B</td><td>3.2T</td><td>1:2,000</td><td> $\mathbf { 3 . 6 \times 1 0 ^ { 2 2 } }$ </td><td>Interaction biology</td></tr></table>

## A.3 Scaling experiment configurations

Figure 1 in the main text reports a controlled scaling experiment across six model scales: 42M, 315M, 800M, 1B, 1.3B, and 1.6B parameters. All six models were trained on the same corpus, sub-sampled to roughly preserve the data-heavy regime. Each model uses the same architecture family, optimizer configuration, and warmup schedule as the production 1.6B model; only the parameter count and corresponding token budget difer.

For each scale we evaluated:

• Pretraining loss vs. compute (Figure 1b): logged during training and plotted against approximate FLOPs.

• Six-class mechanism-of-action macro precision (Figure 1c): computed on the same evaluation set as Section 3.1 / Appendix B.1 (107,027 interactions, molecules split at 0.60 Tanimoto, 160-feature parameter-free signature on frozen representations). Every scale is evaluated on the same ten train/test splits of this set and we report the mean across them. These splits resample the evaluation partition only; they do not resample model training.

• Orthosteric vs. allosteric AUROC (Figure 1d): computed using the 145-feature parameter-free signature derived from Section 3.2 / Appendix D, evaluated on the same 6,032 OOD complexes from 141 OOD proteins.

• Covalent vs. non-covalent AUROC (Figure 1e): computed using the 36-feature parameter-free signature derived from Appendix E, evaluated on the subset of the 1,212 OOD pairs from 237 OOD proteins in which neither class carries a detectable electrophilic warhead (Appendix E.2).

The signature features and signs for all three scoring rules, mechanism of action, orthosteric/allosteric and covalent, are derived once from in-distribution data at the 1.6B scale and applied unchanged to all six model scales; this isolates capability emergence from any signature-tuning efect. The orthosteric/allosteric and covalent OOD protein sets share no proteins.

## A.4 On the mechanism of emergence

The mechanism of emergence is an open question in the field. Wei et al. [1] note that few compelling explanations exist for why capabilities emerge in the way they do, and ofer two intuitions rather than accounts: depth suficient for sequential composition, and capacity suficient for tasks requiring knowledge memorization. Our primary hypothesis is that emergence depends on data and computation.

First, the property has to be representable. Whether a bond forms depends on the reactivity of the ligand’s electrophile and the environment of the candidate residue, not on proximity. Two molecules can occupy the same pocket in similar poses while producing opposite functional outcomes. Neither distinction is recoverable from coordinates alone, so a representation organized around geometry cannot hold it. Training on multimodal sequences imposes no such constraint, leaving the model free to learn the interaction pattern directly, putting the capability within reach.

Second, the corpus has to be large enough, also considering patterns are not present in every single pair. For example, whether a given electrophile reacts with a given nucleophile in a given environment is a regularity that exists only across many pairs sharing a protein or a chemotype. This is the reason for training at 1:2,000 rather than near the Chinchilla-optima ratio.

Third, the pattern has to be computable. Recognizing bond formation, for example, requires locating the reactive group on the ligand, identifying candidate nucleophilic residues, assessing the local environment of each, and judging whether that specific combination reacts. These are several composed steps, and none of them is a local property of either sequence. Following Wei’s intuition about depth suficient for sequential composition, capacity below threshold may support the individual steps while the composition remains out of reach.

Our explanation is a hypothesis grounded in the nature of molecular interactions. It is consistent with the observations reported in Section 2.3 and is not an experimentally established causal mechanism. We view our explanation as another data point in the open debate of explaining the mechanism of emergence.

## B Mechanism of action: evaluation datasets

## B.1 Six-class mechanism dataset

We curated 107,027 protein-ligand interactions with literature-derived mechanism labels across six classes (inhibitor, agonist, antagonist, activator, substrate, product). Bonbon was trained without any of these labels; the dataset is used exclusively for evaluating whether the embedding space organizes by mechanism. To prevent scafold memorization in the evaluation, molecules were clustered at 0.60 Tanimoto similarity using Morgan fingerprints (radius 2, 2048 bits), and entire clusters were assigned to either the evaluation-train split or the test split.

Class composition. The six classes are not represented uniformly across pharmacology databases: regulatory categories (inhibitor, agonist, antagonist, activator) are overrepresented relative to metabolic categories (substrate, product), reflecting the focus of historical pharmaceutical research on small-molecule modulators of regulatory targets. Aggregate metrics are reported as macro-averaged precision (unweighted mean across the six classes) to prevent the larger inhibitor class from dominating the score. Because entire Tanimoto clusters are assigned to one side of the split, the held-out fraction varies by class; Table A2 reports train and test sizes as means over the ten splits.

Table A2: Six-class mechanism dataset composition. Train and test sizes are means over the ten evaluation splits.
<table><tr><td>Class</td><td>Train</td><td>Test</td><td>Total</td></tr><tr><td>Inhibitor</td><td> $2 7 , 0 5 0 \pm 9 9$ </td><td> ${ 2 , 9 5 0 \pm 9 9 }$ </td><td>30,000</td></tr><tr><td>Substrate</td><td> $2 2 , 9 0 5 \pm 3 1 8$ </td><td> $1 , 1 4 4 \pm 3 1 8$ </td><td>24,049</td></tr><tr><td>Product</td><td> $2 1 , 2 4 4 \pm 2 6 5$ </td><td> $1 , 0 7 8 \pm 2 6 5$ </td><td>22,322</td></tr><tr><td>Agonist</td><td> $1 2 , 1 4 4 \pm 1 1 6$ </td><td> $9 9 1 \pm 1 1 6$ </td><td>13,135</td></tr><tr><td>Antagonist</td><td> $^ { 9 , 2 1 8 \pm 1 0 8 }$ </td><td> $1 , 2 7 8 \pm 1 0 8$ </td><td>10,495</td></tr><tr><td>Activator</td><td> ${ 5 , 5 0 6 \pm 6 8 }$ </td><td> $1 , 5 2 0 \pm 6 8$ </td><td>7,026</td></tr><tr><td>Total</td><td> $9 8 , 0 6 6 \pm 4 9 6$ </td><td> $8 { , } 9 6 1 \pm 4 9 6$ </td><td>107,027</td></tr></table>

Signature construction. The 160-feature signature is a parameter-free scoring rule derived entirely from in-distribution interactions. For each feature in Bonbon’s frozen representations we computed a one-versus-rest efect size for each of the six mechanism classes, and retained the 160 features whose discriminative direction is consistent across protein families rather than driven by any single family. Each retained feature carries a sign per class, taken from the direction of its efect size for that class. Scoring an interaction reduces to one dot product per class between the interaction’s representation and that class’s signed feature vector, with the predicted class the argmax over the six scores. There are no trainable parameters: feature selection and signs are fixed from in-distribution statistics and applied unchanged to held-out interactions.

Per-class precision. Table A3 reports signature precision per class as the mean over the ten splits with a 95% confidence interval on the mean.

Table A3: Six-class signature precision on held-out interactions, mean over ten evaluation splits.
<table><tr><td>Class</td><td>Precision</td><td>SD</td><td>95% CI</td></tr><tr><td>Activator</td><td>0.987</td><td>0.010</td><td>[0.980, 0.994]</td></tr><tr><td>Inhibitor</td><td>0.879</td><td>0.020</td><td>[0.865, 0.893]</td></tr><tr><td>Antagonist</td><td>0.829</td><td>0.020</td><td>[0.815, 0.843]</td></tr><tr><td>Agonist</td><td>0.792</td><td>0.023</td><td>[0.776, 0.808]</td></tr><tr><td>Substrate</td><td>0.691</td><td>0.093</td><td>[0.625, 0.758]</td></tr><tr><td>Product</td><td>0.686</td><td>0.069</td><td>[0.637, 0.735]</td></tr><tr><td>Macro</td><td>0.811</td><td>0.010</td><td>[0.803, 0.818]</td></tr></table>

Substrate/product directionality. Confusion between substrate and product accounts for 97% and 90% of each class’s total error (Figure 2a). These two categories difer by reaction direction rather than by chemistry, and they carry the widest intervals in Table A3.

## B.2 Mechanism overlap at the molecule level

Some molecules appear under more than one mechanism, always against diferent proteins, since each interaction pair is unique. At the molecule level, 1,925 of the 58,886 molecules in the six-class evaluation set (3.27%) carry more than one mechanism label. Removing them and repeating the signature classification on the same ten splits gives macro precision of $0 . 8 1 7 \pm 0 . 0 1 7$ against $0 . { \overset { - } { 8 } } 1 1 \pm 0 . 0 1 0$ on the full set. The paired diference across splits is +0.007 with a 95% confidence interval of $[ - 0 . 0 0 3 , + 0 . 0 1 6 ]$ , which contains zero. The six-class result is therefore not an artifact of molecules carrying several labels.

## B.3 GPCR agonist/antagonist evaluation dataset

Table A4 gives the composition of the within-receptor evaluation set, and Table A5 the resulting agonist versus antagonist discrimination.

Table A4: GPCR agonist/antagonist evaluation dataset. Every GPCR has both agonist and antagonist ligands; molecule splits ensure structural novelty (Tanimoto < 0.60 to all training molecules).
<table><tr><td></td><td>Train</td><td>Test</td><td>Val</td><td>Total</td></tr><tr><td>Agonist</td><td>12,352</td><td>1,543</td><td>1,543</td><td>15,438</td></tr><tr><td>Antagonist</td><td>10,152</td><td>1,268</td><td>1,268</td><td>12,688</td></tr><tr><td>Total</td><td>22,504</td><td>2,811</td><td>2,811</td><td>28,126</td></tr></table>

Table A5: Signature-based agonist vs. antagonist discrimination on GPCRs. Frozen representations; molecule split ensures structural novelty.
<table><tr><td></td><td>BONBON</td></tr><tr><td>Balanced accuracy</td><td> $8 5 . 5 \pm 2 . 1 \%$ </td></tr><tr><td>Agonist precision</td><td> $8 8 \pm 1 \%$ </td></tr><tr><td>Antagonist precision</td><td> $8 3 \pm 4 \%$ </td></tr></table>

## C Binding site localization: dataset, head selection, and per-family results

## C.1 Dataset construction and methodology

We collected 1,176 protein-ligand complexes with experimentally determined co-crystal structures from the PDB, spanning kinases, GPCRs, oxidoreductases, proteases, epigenetic regulators, transporters, ion channels, nuclear receptors, and receptor tyrosine kinases. Targets were selected based on structural availability in the PDB and presence of a cocrystallized drug ligand, without filtering on pocket or cavity characteristics or the existence of prior binding site annotations. Complexes were labeled based on whether the protein appeared in Bonbon’s training data: 782 proteins were found in the training set (indistribution, ID), while 394 proteins sharing less than 90% sequence identity with any training protein were designated as the out-of-distribution (OOD) group, representing a stricter test of generalization.

We extracted attention weights from all layers and heads across all 1,176 complexes. For each residue, we computed the minimum heavy-atom distance to the ligand and defined binding site residues as those within 8 Å of any ligand atom. We then measured the correlation between attention weights and inverse ligand distance across all residues.

Attention weights show a monotonic inverse relationship with distance to the nearest ligand atom. Residues within 5 Å of the ligand receive 1.74× the per-protein mean attention, while residues beyond 25 Å receive 0.58×, a 3× contact-to-distant ratio across the 782 in-distribution proteins. The relationship is strictly monotonic across all six distance bins and consistent across all nine protein families, and it generalizes to the 394 held-out proteins with an equal or larger ratio: contact residues receive 1.71× the mean and distant residues 0.52×, a 3.3× ratio, so attention weights reflect learned interaction principles rather than memorization of training examples.

## C.2 Head selection

We identified a subset of attention heads, approximately 25% of all heads, that show consistently high performance for binding site detection. We refer to these as site-detection heads and use them for all subsequent binding site evaluation. Early-layer heads act as universal binding site detectors and generalist supporting heads, while later-layer heads specialize for particular protein families; for example, certain heads show preferential performance on membrane protein families such as GPCRs and transporters. The emergence of this hierarchical specialization is itself noteworthy: without any residue-level supervision, the model has self-organized a subset of its attention mechanism for spatial localization, with progressive refinement from general to family-specific detection.

## C.3 Per-family binding site prediction

Table A6 reports binding site prediction F1 per protein family, in distribution and out of distribution.

Table A6: Zero-shot binding site prediction across protein families. F1 for residue-level classification using frozen attention weights, mean ± standard deviation. No residue-level supervision during training. ID = proteins in training set; OOD = proteins held out entirely (< 90% sequence identity to any training protein). n = number of complexes per family.
<table><tr><td>Family</td><td>n</td><td>ID</td><td>OOD</td></tr><tr><td>Kinase</td><td>213</td><td> $0 . 3 8 1 \pm 0 . 0 9 0$ </td><td> $0 . 3 7 0 \pm 0 . 0 9 1$ </td></tr><tr><td>Protease</td><td>180</td><td> $0 . 4 1 0 \pm 0 . 1 6 3$ </td><td> $0 . 4 0 3 \pm 0 . 1 5 4$ </td></tr><tr><td>GPCR</td><td>156</td><td> $0 . 4 0 4 \pm 0 . 0 7 8$ </td><td> $0 . 3 9 3 \pm 0 . 1 0 2$ </td></tr><tr><td>Oxidoreductase</td><td>100</td><td> $0 . 3 5 0 \pm 0 . 1 5 5$ </td><td> $0 . 3 7 8 \pm 0 . 1 1 3$ </td></tr><tr><td>Epigenetic</td><td>61</td><td> $0 . 3 9 2 \pm 0 . 1 2 2$ </td><td> $0 . 4 0 9 \pm 0 . 0 6 9$ </td></tr><tr><td>Ion channel</td><td>56</td><td> $0 . 2 7 4 \pm 0 . 1 3 1$ </td><td> $0 . 2 5 2 \pm 0 . 1 2 3$ </td></tr><tr><td>RTK</td><td>38</td><td> $0 . 3 8 6 \pm 0 . 0 6 7$ </td><td> $0 . 3 8 8 \pm 0 . 0 5 8$ </td></tr><tr><td>Nuclear receptor</td><td>31</td><td> $0 . 4 6 1 \pm 0 . 1 2 8$ </td><td> $0 . 3 9 8 \pm 0 . 0 7 9$ </td></tr><tr><td>Transporter</td><td>30</td><td> $0 . 3 5 9 \pm 0 . 1 7 6$ </td><td> $0 . 3 3 3 \pm 0 . 1 2 9$ </td></tr><tr><td>Other</td><td>311</td><td> $0 . 3 8 9 \pm 0 . 1 6 7$ </td><td> $0 . 3 9 3 \pm 0 . 1 4 2$ </td></tr><tr><td>Overall</td><td>1,176</td><td> $\mathbf { 0 . 3 8 1 \pm 0 . 1 4 2 }$ </td><td> $\mathbf { 0 . 3 8 8 \ : \pm { \ : 0 . 1 2 4 } }$ </td></tr></table>

## C.4 COACH420 comparison

Table A7 compares Bonbon against established binding site prediction methods on COACH420.

Table A7: Binding site prediction on COACH420. F1 for residue-level binding site identification. Bonbon uses sequence alone, with no residue-level supervision, no 3D coordinates and no pocket annotations.

<table><tr><td>Method</td><td>F1</td></tr><tr><td>Bonbon</td><td>0.497</td></tr><tr><td>Fpocket [13]</td><td>0.45</td></tr><tr><td>P2Rank [14]</td><td>0.44</td></tr><tr><td>ESMBind [29]</td><td>0.32</td></tr></table>

## C.5 Structurally challenging targets

The results above aggregate across many proteins. To examine whether Bonbon localizes binding sites on individual targets where structure-dependent methods face fundamental limitations, we visualized attention patterns on four structurally diverse proteins spanning orphan GPCRs, ion channels, nuclear receptors, and a protein-protein interaction target with a conformationally plastic binding site (Figure A2). Two representative cases are described below.

BCL-2. BCL-2 is an anti-apoptotic protein whose drug-binding site is a protein-protein interaction interface, the BH3-binding groove, rather than a conventional pocket. The groove is a shallow, extended hydrophobic surface formed by the BH1, BH2, and BH3 domains [30, 31] that lacks the deep pockets typical of small-molecule targets [32]. The groove exists in the apo state but is conformationally plastic [33]: an adjacent 58-residue intrinsically disordered flexible loop domain can fold toward the protein core and occlude the BH1–BH3 binding surface [30], and phosphorylation of this loop allosterically modulates groove accessibility from the opposite face of the protein [34]. A shallow PPI surface, conformational plasticity, and dynamic occlusion by an adjacent disordered region together make BCL-2 a challenging target for geometry-based methods. Bonbon’s attention concentrates on the BH1 and BH2 domains, which together with BH3 form the hydrophobic groove targeted by BH3-only proteins and small-molecule mimetics such as navitoclax [30]. Attention on BH3 is moderate and on BH4 minimal, consistent with its distance from the binding surface. The model identifies the functional binding region from sequence alone, without structural coordinates or pocket geometry.

GPR84. GPR84 is a GPCR that has been classified as an orphan receptor for over two decades [35, 36], had no experimentally determined structure until 2023 [37, 38], and features a fully occluded orthosteric pocket that is inaccessible from the extracellular surface [38]. Attention concentrates on residues within 5 Å of the ligand (Spearman $\rho = + 0 . 5 5 8 )$ , correctly localizing the orthosteric binding site across transmembrane helices despite membrane embedding and conformational flexibility.

Two further targets are noted briefly: the potassium voltage-gated channel KCNQ2, where attention localizes the pore-domain binding site $( \rho = + 0 . 5 2 6 , \mathrm { O O D } )$ , concentrating on residues near the selectivity filter region; and Tankyrase-1, a poly(ADP-ribose) polymerase with 153 intrinsically disordered residues flanking the PARP catalytic domain, where attention concentrates on the catalytic domain binding site $( \rho = + 0 . 5 7 0 , \mathrm { I D } )$ while ignoring the extensive disordered regions. Binding site localization therefore extends to protein classes where structural methods face fundamental limitations, including conformationally plastic PPI targets, membrane-embedded receptors, and ion channels, capturing sequence-encoded determinants of ligand engagement rather than geometric pocket features.

A  
![](images/6a792417ccb417f2f8b249898c22c0a326b99705330de3d6e6139af4dba5bf87.jpg)

![](images/f0cab040e993c481d57672e4f3db493934030cb3f3541eb3b5f194c8bcce8143.jpg)

![](images/d6494a2692129fe51d053217988eb0d0689bc37a1aba36d2f197365409ded272.jpg)

![](images/4693c7cce4ee1b001266b8c84ba2c77f087a10d96bfc79cbe6bc6275069ac14b.jpg)

B  
![](images/8635a6b1c4f8c56821e146c1d75e0a7697cca5a9f0a03f2df1b30f1439ff91a5.jpg)  
Figure A2: (a) Attention overlay traces on four structurally challenging targets. Blue solid: attention signal; red dashed: inverse distance to ligand; grey shading: binding site residues. (b) 3D attention visualization on the same four targets. KCNQ2: attention concentrates on helices surrounding the ligand binding site. BCL-2: attention localizes to the BH3-binding groove on a shallow PPI surface. GPR84: attention maps the orthosteric pocket of an orphan GPCR. Tankyrase-1: attention concentrates on the ligand-binding chain of the homodimer while the non-binding chain remains cold, demonstrating ligand-conditioned localization. Bonbon also distinguishes orthosteric from allosteric binding sites from frozen representations.

## D Orthosteric/allosteric discrimination: dataset, signature construction, and analysis

## D.1 Dataset

Allosteric and orthosteric binding annotations were curated from published databases, yielding 19,027 protein-ligand complexes across 463 proteins. Complexes were split by protein identity: 322 in-distribution proteins (12,995 complexes) for signature derivation, and 141 out-ofdistribution proteins (6,032 complexes) for evaluation. Both-class proteins (those with at least 2 ligands of each binding type) were used for the within-protein evaluation: 18 OOD proteins (1,860 complexes).

## D.2 Signature construction

The 145-feature signature is a parameter-free scoring rule derived entirely from in-distribution complexes. For each feature in Bonbon’s learned representations, we computed the efect size between allosteric and orthosteric complexes and selected 145 features that show consistent discriminative signal across protein families. Each selected feature is assigned a sign from the direction of its efect size. Scoring a new complex reduces to a single dot product between the complex’s representation and the signed feature vector. There are no trainable parameters: feature selection and signs are fixed from in-distribution statistics and applied to out-of-distribution complexes without modification.

## D.3 Classification results

Table A8 reports zero-shot orthosteric versus allosteric classification against the Morgan fingerprint baseline, overall and per family.

Table A8: Zero-shot orthosteric vs. allosteric classification. 145-feature signature (no trained parameters) on OOD proteins. MFP = Morgan fingerprints. Bold indicates AUROC > 0.6. The TF BC OOD result (0.548) is consistent with the smaller training representation of transcription factors and is reported as a domain coverage limitation.

<table><tr><td>Evaluation subset</td><td>Proteins</td><td>Complexes</td><td>Signature</td><td>MFP</td></tr><tr><td>All OOD</td><td>141</td><td>6,032</td><td>0.834</td><td>0.589</td></tr><tr><td>GPCR OOD</td><td>60</td><td>3,684</td><td>0.759</td><td>0.487</td></tr><tr><td>Enzyme OOD</td><td>61</td><td>1,480</td><td>0.746</td><td>0.646</td></tr><tr><td>TF OOD</td><td>5</td><td>445</td><td>0.669</td><td>0.320</td></tr><tr><td>Both-class OOD</td><td>18</td><td>1,860</td><td>0.821</td><td>0.609</td></tr><tr><td>GPCR BC OOD</td><td>9</td><td>1,229</td><td>0.788</td><td>0.660</td></tr><tr><td>Enzyme BC OOD</td><td>5</td><td>226</td><td>0.793</td><td>0.568</td></tr><tr><td>TF BC OOD</td><td>2</td><td>361</td><td>0.548</td><td>0.270</td></tr></table>

## D.4 Score distribution by family

Figure A3 shows the distribution of signature scores by protein family underlying the AUROCs in Table A8.

![](images/a65091a0f1386e8dcf045b717cfa05caa5f78ea251f96e8b01c565acac8ded94.jpg)  
Figure A3: Signature score distributions by binding-mode class (violin plots). Bimodal separation between orthosteric and allosteric classes is visible across membrane receptors and enzymes; on transcription factors, the separation collapses, consistent with the lower AUROC on that family.

## D.5 Head ablation

Ablation was performed at inference time with no retraining, on the OOD dataset described above. Site-detection heads (25% of all heads, selected as in Appendix C.2) were zeroed; the control zeroes a random 25% of heads, repeated over 10 seeds (Figure A4).

The orthosteric/allosteric discrimination capability can be explained as a combination of spatial localization of the binding site from the site-specific heads and binding-mode features related to the interaction representations, with both the heads and the representations contributing to the signature.

![](images/10f3e3f9a256bd6b137c061566d61f5689195a45b2b5177d652d78d9c1592c6a.jpg)  
Figure A4: Orthosteric/allosteric AUROC after ablating the site-detection heads, compared with random-head controls and the Morgan fingerprint baseline.

## E Covalent binding mode recognition: dataset, controls, and residue localization

## E.1 Dataset construction

We curated 1,490 protein-ligand pairs (970 covalent, 520 non-covalent) from covalent and non-covalent co-crystal structures in the RCSB PDB, spanning 322 unique proteins. Pocket residues were defined as heavy atoms within 5 Å of any ligand heavy atom (24,811 total). Covalent bond assignments were derived from PDB mmCIF records cross-referenced with the Chemical Component Dictionary. Proteins were split into in-distribution (85 proteins, 278 pairs) and out-of-distribution (237 proteins, 1,212 pairs) by sequence identity (< 90% to training data). All main-text results report OOD performance.

## E.2 Warhead controls

Each molecule was screened for 20 electrophilic substructure patterns (acrylamide, Michael acceptor, nitrile, haloacetamide, epoxide, aldehyde, sulfonyl fluoride, and others) using RDKit SMARTS matching. This yields the four groups in Table A9 and Figure A5, which enable two controlled comparisons.

Table A9: Warhead control groups for covalent binding analysis.
<table><tr><td>Group</td><td>Bond type</td><td>Electrophile</td><td>n</td><td>Description</td></tr><tr><td>A</td><td>Covalent</td><td>Present</td><td>731</td><td>Warhead engages the target</td></tr><tr><td>B</td><td>Covalent</td><td>Absent</td><td>239</td><td>Enzymatic covalent mechanism</td></tr><tr><td>C</td><td>Non-covalent</td><td>Present</td><td>131</td><td>Warhead present but does not react</td></tr><tr><td>D</td><td>Non-covalent</td><td>Absent</td><td>389</td><td>No electrophilic groups</td></tr></table>

![](images/0e22d155043d6aad2d175e39ee2a6a86b7e2a4aabb9347bdae6db99d0b44d3d5.jpg)  
Figure A5: The four warhead control groups. Rows give the observed bond type, columns whether the molecule carries a detectable electrophilic warhead. The two controlled comparisons are A vs. C, where both groups carry warheads and only one reacts, and B vs. D, where neither carries a detectable warhead. Counts match Table A9.

Groups A vs. C: both carry reactive groups, only one reacts. Both groups carry electrophilic warheads; Group A molecules form covalent bonds and Group C molecules do not, despite carrying the same types of reactive substructure. A model that detects warhead presence alone would score the two groups identically. Bonbon achieves AUROC 0.834 against 0.728 for Morgan fingerprints, and MACCS structural keys, which reach 0.779 on the full comparison, fall to 0.682 here, the weakest fingerprint baseline.

Groups B vs. D: neither carries a detectable reactive group. This comparison removes warhead detection as an explanatory variable entirely, since covalent binding in Group B proceeds through enzymatic mechanisms in which the protein’s catalytic machinery drives bond formation rather than the ligand’s intrinsic reactivity. Bonbon achieves AUROC 0.775 against 0.697 for Morgan fingerprints and 0.676 for MACCS, so discriminative signal persists on molecules carrying no electrophilic substructure pattern.

## E.3 Classification methodology

Binary covalent vs. non-covalent signature (Section 3.3). From frozen interaction representations, we identified 36 features whose discriminative direction between covalent and non-covalent complexes is consistent across protein families on in-distribution data, selected by the same efect-size procedure as the orthosteric/allosteric signature (Appendix D.2), and constructed a parameter-free scoring rule: a single dot product between each complex’s representation and the signed feature vector. There are no trained parameters. On the out-of-distribution split the signature reaches F1 0.840 and average precision 0.927, with precision 85.9% and recall 82.1%. Per-protein validation was performed on within-protein covalent vs. non-covalent pairs from 215 both-class proteins, computing the AUROC per protein using the same signature.

Reactive residue localization. For per-residue classification of “is this the covalent residue,” we trained logistic regression classifiers (L1/L2 regularization, C = 0.1, saga solver) on per-residue frozen representations with standardized features.

## E.4 Reactive residue localization

Reactive residue localization was evaluated on 883 OOD covalent complexes.

Two-step pipeline. Step 1 predicts amino acid type from pair-level features; Step 1 performance is reported in Table A10. Step 2 ranks pocket residues of the predicted type, reaching 71.8% Top-1 accuracy against a 5.8% chance baseline, with 84.1% Top-3 and a mean rank of 2.3.

Table A10: Step 1 type prediction AUROC (out-of-distribution).
<table><tr><td>Target residue</td><td>AUROC</td><td>n positive (OOD)</td></tr><tr><td>Serine</td><td>0.879</td><td>327</td></tr><tr><td>Threonine</td><td>0.861</td><td>151</td></tr><tr><td>Lysine</td><td>0.618</td><td>60</td></tr><tr><td>Cysteine</td><td>0.561</td><td>149</td></tr></table>

## E.5 Distance-controlled analysis

To rule out the confound that reactive residues are systematically closer to the ligand, we retained only same-type structural residues within ±0.5 Å or ±0.25 Å of any reactive residue’s distance. This is an ablation of proximity: distance matching removes distance as an available cue. After distance matching, the signal persists for all three residue types at ±0.5 Å, and serine and cysteine remain above chance even under strict ±0.25 Å matching, while lysine’s separation is largely proximity-driven. Under five-fold cross-validation on OOD proteins the same discrimination gives 0.894 for serine, 0.838 for cysteine and 0.801 for lysine.

Table A11: Distance-controlled AUROC for reactive residue identification. Serine and cysteine retain signal well above chance after strict distance matching, confirming the representations encode catalytic-site context beyond proximity. Lysine collapses to chance at the strictest matching, suggesting its signal is primarily distance-driven.
<table><tr><td>Residue</td><td>Original AUROC</td><td>±0.5Å matched</td><td>±0.25 Å matched</td><td>n (original)</td></tr><tr><td>Serine</td><td>0.902</td><td>0.877</td><td>0.725</td><td>412 positive</td></tr><tr><td>Cysteine</td><td>0.817</td><td>0.796</td><td>0.758</td><td>179 positive</td></tr><tr><td>Lysine</td><td>0.701</td><td>0.698</td><td>0.500</td><td>72 positive</td></tr></table>

## E.6 Covalent recognition across model scales

Covalent versus non-covalent AUROC measured on all molecules is 0.645, 0.678, 0.691, 0.675, 0.695, and 0.838 from 42M to 1.6B parameters (Figure A6). This contrast includes molecules carrying detectable electrophiles, whose presence is a local property of the ligand and is learnable at smaller scales, which lifts the sub-1.6B values above those of the warhead-free evaluation in Figure 1e without producing a trend that anticipates the jump at 1.6B.

![](images/16bbd011f7fed511d6fca8c68e160171b75cd52aae3403bae99d4f73293d8c0d.jpg)  
Figure A6: Covalent versus non-covalent AUROC across model scales, measured on all molecules.

## F Active moiety identification: dataset construction, metric, and generalization

Active moieties are the chemical substructures responsible for engaging the binding site: a kinase inhibitor’s hinge-binding heterocycle is its active moiety, while its solvent-exposed substituents tune selectivity but can be replaced without disrupting binding. Identifying them is a prerequisite for rational optimization, since knowing which atoms to preserve and which to modify is the basis of SAR analysis. Existing approaches require a ligand-bound 3D structure or a docking pose, and docking and co-folding predict contacts rather than which atoms drive activity. No moiety labels appear in Bonbon’s training data.

Ground-truth active moieties were derived by automated matched molecular pair analysis of ChEMBL structure-activity data. For each target with at least 100 active compounds $( \mathrm { I C } _ { 5 0 } < 1 \mu \mathrm { M } )$ , actives were grouped by Butina clustering at Tanimoto 0.40 on Morgan fingerprints, and matched pairs difering by a single structural transformation were generated within each cluster. Atoms whose single-atom modification shifted activity more than tenfold across at least three pairs were scored as essential and converted to SMARTS patterns; the pipeline was validated against five proteins with expert-curated SMARTS from the medicinal chemistry literature. The resulting dataset spans 145 proteins across five target classes with 241,035 protein-ligand pairs.

Attention coverage for a molecule is the fraction of total ligand attention allocated to active moiety atoms, bounded in [0, 1]. The chance baseline of 0.36 is the average fraction of moiety atoms per molecule, that is, the coverage uniform attention would produce; coverage is bounded and threshold-free, so it imposes no discontinuity on the scaling results [23]. Mean coverage is 0.85 across 241,035 molecules and 145 proteins, and is uniform across the five target classes, from 82.4% for ion channels to 85.9% for GPCRs, which span diferent binding site architectures (Table A12; Figure A7).

To test generalization to unseen chemical matter, all molecules were clustered into 24,525 chemotype groups at Tanimoto 0.60 (Morgan fingerprints, radius 2, 2048 bits) by leader clustering, and 6,197 complete clusters were held out, so that no held-out molecule shares greater than 0.60 Tanimoto similarity with any in-distribution molecule. Coverage is 84.6% in-distribution and 84.7% across the 60,514 held-out molecules, with per-class divergence below 0.7 percentage points. With the protein held constant, attention tracks a diferent moiety in each of four structurally distinct ligands for JAK2, BACE1 and adenosine A2A, all out of distribution: the pattern depends on the pair, not the protein alone (Figure A8).

Across model scales, mean attention coverage is 0.362 at 42M, 315M, 800M and 1B and 0.365 at 1.3B, against a chance baseline of 0.36, then rises to 0.85 at 1.6B.

Table A12: Active moiety attention coverage across protein classes. Coverage is the fraction of total ligand attention allocated to active moiety atoms; chance baseline 0.36. No substructure annotations appear in training.
<table><tr><td>Target class</td><td>Proteins</td><td>Molecules</td><td>Mean coverage</td></tr><tr><td>Enzyme</td><td>7</td><td>31,159</td><td>82.9%</td></tr><tr><td>GPCR</td><td>36</td><td>80,270</td><td>85.9%</td></tr><tr><td>Ion channel</td><td>37</td><td>40,584</td><td>82.4%</td></tr><tr><td>Nuclear receptor</td><td>36</td><td>26,312</td><td>85.3%</td></tr><tr><td>RTK</td><td>29</td><td>62,710</td><td>85.0%</td></tr><tr><td>All</td><td>145</td><td>241,035</td><td>84.6%</td></tr></table>

Attention at the moiety boundary. Coverage measures how much attention falls inside the moiety; it does not say how sharply attention stops at the moiety’s edge. Boundary attention coverage restricts the measurement to the boundary region, the moiety’s edge atoms together with the one-bond-outside ring, and reports the share of attention in that region falling on the edge atoms themselves. Chance is 0.5 and values above 0.59 indicate a sharp boundary. Median boundary coverage sits at chance at every scale through 1.3B and reaches 0.904 at 1.6B (Table A13): 95% of proteins show a sharp moiety boundary at the largest scale, against roughly 1% at every smaller scale.

Table A13: Boundary sharpness across model scales. Median boundary attention coverage and per-protein counts in the three bucket categories.
<table><tr><td>Scale</td><td>Median boundary coverage</td><td>Sharp (&gt;0.59)</td><td>Soft (0.50–0.59)</td><td>Bleeding (&lt;0.50)</td></tr><tr><td>42M</td><td>0.501</td><td>1/90</td><td>57/90</td><td>32/90</td></tr><tr><td>315M</td><td>0.501</td><td>1/90</td><td>66/90</td><td>23/90</td></tr><tr><td>800M</td><td>0.500</td><td>1/90</td><td>59/90</td><td>30/90</td></tr><tr><td>1B</td><td>0.501</td><td>1/90</td><td>60/90</td><td>29/90</td></tr><tr><td>1.3B</td><td>0.502</td><td>1/90</td><td>75/90</td><td>14/90</td></tr><tr><td>1.6B</td><td>0.904</td><td>113/119</td><td>1/119</td><td>5/119</td></tr></table>

![](images/ebb4a44ec9dfa1946ad362f426fa7e0b31b3da271fe4cdf7ab99bc8b4bc8a205.jpg)

![](images/968a38289ed7bf5d477e77b31ce9719dcb76b20eb998bd869dc3520dc4da5680.jpg)  
D

B  
![](images/6fb4f5c78fe295fe2807269553f0b5e1e510c89ca178a9432a792b42923addad.jpg)  
C  
E

![](images/f53bdb4277ded4baff6976d57c24546ae499c0bcf8d80a9e03f9d60d626c75c0.jpg)  
Figure A7: (a) Attention boundary profile: mean attention per atom as a function of topological distance from the moiety edge. (b) Mean attention coverage by protein class. (c) Chemotype generalization: mean attention coverage for in-distribution and held-out molecules across protein classes. (d) Per-protein boundary heatmap across 119 proteins sorted by boundary sharpness; the green line marks the moiety edge. (e) Per-atom attention across five protein classes: attention highlights the annotated active moiety (boxed) regardless of protein family or molecular scafold.

![](images/05dc9dc50ce8cec312df54db172fa3146494e39e58cb82ba174992c58dc31c45.jpg)  
Figure A8: (a–c) Interaction-conditioned moiety identification on three out-of-distribution proteins, JAK2, BACE1, and adenosine A2A: four structurally distinct ligands per protein, each with a diferent active moiety, showing that moiety detection depends on the proteinligand interaction rather than on the protein alone.

## G Computational benchmarks: protocols, datasets, and per-method comparisons

## G.1 Decontamination

To hold out the evaluation benchmarks (CASP16, FEP+4, OpenFE, MF-PCBA, LIT-PCBA and its subset LIT-PCBA-AF3, COACH420, Davis, and BindingDB), we used MMseqs2 [39] (--min-seq-id 0.9 -c 0.8 --cov-mode 2 --alignment-mode 3 -s 7.5) to remove from training any protein sharing ≥90% sequence identity over ≥80% coverage with a benchmark target, along with all of its associated protein-ligand pairs. Ligand decontamination used Tanimoto below 0.60 on Morgan fingerprints (radius 2, 2048 bits).

The afinity probe of Appendix G.4 is trained on measurements drawn from the BindingDB database [27], not from the published BindingDB benchmark [40], which is held out in full. Proteins sharing ≥90% sequence identity with any CASP16, FEP+4, or OpenFE target, together with their associated pairs, were removed from the probe’s training data before fitting, so no evaluation target appears in it.

## G.2 Dataset preparation

For MF-PCBA, we selected bioassays and downsampled to 50,000 pairs as described in Passaro et al. [9]. For LIT-PCBA, we used the AVE set, which removes duplicates and structural bias, a stricter evaluation than the “full” split used by some prior work.

## G.3 Afinity benchmark configurations

We evaluated on the CASP16 binding afinity challenge comprising 140 protein-ligand pairs across two targets [41], the FEP+ four-target subset (CDK2, TYK2, JNK1, p38) as defined in Boltz-2 [9], and the Open Force Field Protein-Ligand Benchmark [42]. FEP+ computes relative binding free energies within a congeneric series; ABFE is the absolute variant and is reported as a separate row in Table A16. All methods in Table A16 are compared by Pearson correlation on the same targets, and Pearson r is invariant to scale and ofset, so the comparison does not depend on any method being calibrated in absolute terms. We ran Boltz-2 using default settings with predicted structures and its integrated afinity scoring module; all other baselines are reported from published results. We note that our reproduced Boltz-2 CASP16 correlation $( r = 0 . 5 2 )$ difers from the value reported in the original publication $( r = 0 . 6 5 )$ . This discrepancy has been independently observed by other groups: IntFold [43], AqAfinity [44], and TerraBind [45] all report reproduced Boltz-2 correlations in the 0.46–0.59 range on CASP16 subsets. Figure A9 plots the CASP16 predictions against experiment for Bonbon and these baselines.

![](images/260e6d71ef63605fd306fdc716b2bb615bbaa5f74cb9f3c224cc09936e8254d5.jpg)

![](images/61a025f9efcb7e67f0331c9a9cd4ff0493d2ac7cd1e87526cd6b1730832d5bce.jpg)  
Figure A9: CASP16 binding afinity scatter plots. Predicted vs. experimental $\mathrm { p I C } _ { 5 0 }$ on L3000 (left) and L1000 (right). Bonbon $( r = 0 . 7 2 , 0 . 7 5 )$ ) shows tighter clustering along the diagonal than Boltz-2 $( r \doteq 0 . 5 9 , 0 . 4 6 )$ , IntFold $( r = 0 . 6 0 , 0 . 5 3 )$ , and AqAfinity $\bar { ( r = 0 . 5 7 , 0 . 4 0 ) }$ . The advantage is consistent across both targets despite diferences in binding site topology and ligand chemotype.

## G.4 Afinity probe

The two-layer MLP probe maps Bonbon’s frozen interaction representations to a scalar afinity value (pKi), the minimal supervised step required to project the representation space onto the afinity axis. No representation fine-tuning is performed. Probe architecture: input dimension = embedding dimension, hidden dimension = 512, output dimension = 1. Training used AdamW with default hyperparameters and a Huber objective for 1,500 steps (∼12 minutes) on BindingDB [27], with no early stopping.

## G.5 Binary binding prediction: full per-method comparison

Table A14: Binary binding prediction. Bold indicates best performance. “—” indicates not reported. † = structure-informed (uses protein structure embeddings). ‡ = structure-based (uses full 3D coordinates or docking). \* AVE split (deduplicated). \*\* SPRINT evaluated on “full” split without deduplication [46].
<table><tr><td>Benchmark</td><td>Metric</td><td>BONBON</td><td> $\mathrm { B o l t z { - } 2 ^ { \frac { \dag } { \ d } } }$ </td><td> $\mathrm { I n t F o l d ^ { \ddagger } }$ </td><td>SPRINT†</td><td> $\mathrm { D r u g C L I P ^ { \dag } }$ </td><td>Docking</td></tr><tr><td>Davis</td><td>AUPR</td><td>0.825</td><td>0.46</td><td>0.58</td><td>0.51</td><td></td><td></td></tr><tr><td>BindingDB</td><td>AUPR</td><td>0.752</td><td>0.53</td><td>0.67</td><td>0.72</td><td></td><td></td></tr><tr><td>MF-PČBA</td><td>AUPR</td><td>0.081</td><td>0.025</td><td></td><td></td><td></td><td>0.016</td></tr><tr><td>LIT-PCBA*</td><td>EF1%</td><td>11.4</td><td></td><td></td><td>10.8**</td><td>5.5</td><td>3.4-4.6</td></tr></table>

Table A15: Enrichment factor at 1% on LIT-PCBA-AF3 [28], a five-target subset of LIT-PCBA restricted to libraries of fewer than 100,000 compounds.
<table><tr><td>Method</td><td>EF1%</td></tr><tr><td>Bonbon</td><td>16.6</td></tr><tr><td>Boltz-2</td><td>9.3</td></tr><tr><td>Protenix</td><td>7.6</td></tr><tr><td>Glide</td><td>6.8</td></tr><tr><td>ConPlex</td><td>4.3</td></tr><tr><td>Gnina</td><td>2.8</td></tr><tr><td>SPRINT</td><td>1.8</td></tr></table>

## G.6 Binding afinity prediction: per-target results and full per-method comparison

Table A16: Binding afinity prediction, full comparison (Pearson correlation). Methods grouped by type. Bold indicates best within ML methods. † = independently reproduced. “—” indicates not reported.

<table><tr><td>Method</td><td>Type</td><td>CASP16</td><td>FEP+4</td><td>OpenFE</td></tr><tr><td>FEP+</td><td>Physics (&gt;1h)</td><td></td><td>0.78</td><td>0.72</td></tr><tr><td>ABFE</td><td>Physics (&gt;1h)</td><td></td><td>0.75</td><td>0.73</td></tr><tr><td>OpenFE-calc</td><td>Physics (&gt;1h)</td><td></td><td>0.66</td><td></td></tr><tr><td>FMO</td><td>Physics (&lt;1h)</td><td></td><td>0.55</td><td></td></tr><tr><td>MM/PBSA</td><td>Physics (&lt;1h)</td><td></td><td>0.18</td><td></td></tr><tr><td>Chemgauss4</td><td>Docking</td><td></td><td>0.26</td><td></td></tr><tr><td>Boltz-2 [9]</td><td>Structure-based ML</td><td>0.52†</td><td>0.66</td><td>0.63</td></tr><tr><td>IntFold [43]</td><td>Structure-based ML</td><td>0.54</td><td></td><td></td></tr><tr><td>AqAffinity [44]</td><td>Sequence-based ML</td><td>0.49</td><td></td><td></td></tr><tr><td>GAT</td><td>Structure-based ML</td><td></td><td>0.40</td><td></td></tr><tr><td>BACPI</td><td>Structure-based ML</td><td>0.38</td><td>0.14</td><td>0.28</td></tr><tr><td>Haiping</td><td>Sequence-based ML</td><td>0.54</td><td></td><td></td></tr><tr><td>VnDeep</td><td></td><td>0.49</td><td></td><td></td></tr><tr><td>MULTICOM</td><td></td><td>0.38</td><td></td><td></td></tr><tr><td>Haddock</td><td>Structure-based</td><td>0.38</td><td></td><td></td></tr><tr><td>Bonbon</td><td>Sequence-based ML</td><td>0.74</td><td>0.78</td><td>0.69</td></tr></table>

## G.7 Inference throughput

Table A17 reports measured and extrapolated throughput; Figure A10 places Bonbon against structure-dependent and physics-based methods on the speed-accuracy plane.

Table A17: Inference throughput on a single H100 GPU. Boltz-2 was empirically measured at 20.6 s/pair on the same hardware, consistent with the 20 s/pair reported in the Boltz-2 paper on FEP+ using an H100 GPU. \* = extrapolated from measured per-pair rate. $^ { * * } =$ extrapolated for both models.
<table><tr><td>Task</td><td>Pairs</td><td>Mode</td><td>BONBON</td><td>Boltz-2</td><td>Speedup</td></tr><tr><td>FEP+4</td><td>87</td><td> $\mathrm { O n \mathrm { - t h e - } f l y }$ </td><td>0.08s</td><td> $^ { 3 , 8 6 7 \mathrm { s } }$ </td><td>~48,600×</td></tr><tr><td>MF-PCBA (10 targets)</td><td>500K</td><td> $\mathrm { O n \mathrm { - t h e - } f l y }$ </td><td>5min</td><td> ${ \sim } 4 \mathrm { m o n t h s } ^ { \ast }$ </td><td> ${ \sim } 3 4 { , } 7 0 0 \times$ </td></tr><tr><td>LIT-PCBA</td><td>2.65M</td><td> $\mathrm { O n \mathrm { - t h e - } f l y }$ </td><td>18 min</td><td> ${ \sim } 1 . 7 5 \mathrm { y e a r s ^ { * } }$ </td><td> $\sim 5 1 , 1 0 0 \times$ </td></tr><tr><td>Proteome × DrugBank</td><td>1.66B</td><td>Cached</td><td> $1 1 \mathrm { { h o u r s } ^ { * * } }$ </td><td> ${ \sim } 1 , 1 0 0 \mathrm { G P U } \mathrm { - y e a r s } ^ { \ast \ast }$ </td><td> ${ \sim } 8 7 3 , 0 0 0 \times$ </td></tr></table>

![](images/aec46e2635d2a6e144d6715ae544371b67432c7cdb1d88e93e1b7fd835da8ad0.jpg)  
Figure A10: Speed-accuracy Pareto front on FEP+4: Bonbon achieves $r = 0 . 7 8$ in 0.08 seconds, against minutes for structure-dependent machine learning methods and hours for physics-based methods.

## H Prospective experimental validation: methods and per-target detail

Attribution and reproducibility statement. All biochemical assays in this section were performed at an external contract research organization (CRO) under protocols specified in advance by the authors. Compounds were selected and ranked by the authors using Bonbon scores (H.1) and delivered to the CRO in a randomized format; the CRO returned raw dose-response data, which the authors then mapped back to compound identities for hit analysis. The CRO did not filter compounds, repeat assays, or alter protocols. The CRO’s identity is withheld under a confidentiality agreement; per-target protocols (H.5–H.13) reference standard assay formats and published methods and are suficient to reproduce all results at any comparably equipped facility.

## H.1 Compound ranking and selection

For each target, we used Bonbon to rank molecules produced by a separate generative method under synthetic-accessibility constraints; that method is outside the scope of this paper and is the subject of a separate technical report. Bonbon’s role in this experiment is ranking alone. Compounds with Tanimoto similarity $\geq 0 . 6 0$ to any compound in Bonbon’s training data were removed (Morgan fingerprints, radius 2, 2048 bits). PAINS-pattern compounds [24] were filtered. The remaining top-128 compounds by Bonbon score were clustered using hierarchical agglomerative clustering (distance = 1− Tanimoto similarity, fusion cutof = 0.5). Twenty-four compounds were selected from the clustered pool to maximize chemical diversity within the top-ranked set. No additional human selection or medicinal chemistry triage was applied. For MMP-9 one compound and for CDK1 two compounds could not be obtained, yielding 23 and 22 tested compounds respectively and 213 compounds across the nine targets. All delivered compounds were profiled in a single-dose primary screen, followed by ten-point dose-response confirmation for $\mathrm { I C } _ { 5 0 }$ determination (H.2). Confirmed inhibitors are compounds active in both the primary and dose-response assays. Dose-response hit rates are reported over all delivered compounds.

## H.2 Compound sourcing and handling

Compounds were synthesized by a contract research organization and delivered as 10 mM stocks in DMSO, verified to at least 90% purity by LC-MS. Stocks were stored at $- 8 0 ^ { \circ } \mathrm { C }$ (long-term) $\mathrm { o r - 2 0 ^ { \circ } C }$ (working) in desiccated storage and thawed at room temperature before use. All compounds were first assayed in a single-dose primary screen in duplicate at $1 0 \mu \mathrm { M }$ and 1% DMSO. Compounds meeting the progression criterion were assayed in dose-response in duplicate across a ten-point, three-fold serial dilution from a top final concentration of $1 0 0 \bar { \mu } \mathrm { M }$ to $5 \mathrm { n M }$ , with reference inhibitors titrated in parallel on every plate. Per-concentration responses were normalized to percent activity relative to DMSO vehicle controls after subtraction of no-enzyme background; for the protease assays the response was the initial reaction velocity normalized to vehicle. Normalized responses were fit against compound concentration with a four-parameter logistic (sigmoidal, variable-slope) model,

$$
Y = { \mathrm { B o t t o m } } + { \frac { \mathrm { T o p } - { \mathrm { B o t t o m } } } { 1 + 1 0 ^ { ( \log \mathrm { I C _ { 5 0 } } - \log [ \mathrm { c o m p o u n d } ] ) } \cdot { \mathrm { H i l l S l o p e } } } } ,
$$

reporting the midpoint potency $\mathrm { I C } _ { 5 0 }$ (and $\mathrm { p I C _ { 5 0 } = - \Delta \log _ { 1 0 } \Delta \mathrm { I C _ { 5 0 } \ [ M ] ) } }$ together with the Hill slope. A compound counts as a confirmed inhibitor if its dose-response series returns a determinable $\mathrm { I C } _ { 5 0 }$ from this fit; no compound that progressed to dose-response was excluded from the analysis.

## H.3 Novelty analysis

Known actives for each target were aggregated from ChEMBL and BindingDB. A hit counts as a novel molecule only if it clears three criteria: it is not a duplicate of any known active; its maximum ECFP4 Tanimoto (Morgan, radius 2, 2048 bits) to any known active is below 0.45; and it is not a single-substituent analog of any known active by matched-molecular-pair analysis [47], with the changing fragment limited to at most 10 heavy atoms and the constant core required to be at least twice its size, following standard mmpdb limits [48].

## H.4 Per-target results

Table A18 reports, for each of the nine targets, the number of compounds tested and confirmed, the $\mathrm { I C } _ { 5 0 }$ range, and the Tanimoto range to the nearest known active. Figure A11 shows the corresponding per-compound distributions against training data and against known actives.

Table A18: Per-target dose-response results across the nine-target panel. Confirmed inhibitors are compounds active in both the primary and dose-response assays. Tanimoto is the range of maximum ECFP4 (Morgan, radius 2, 2048 bits) similarity of each confirmed inhibitor to any known active for that target.
<table><tr><td>Target</td><td>Tested</td><td>Confirmed</td><td>IC50 range</td><td>Tanimoto</td></tr><tr><td>TYK2</td><td>24</td><td>23</td><td> $1 4 6 \mathrm { n M - } 4 0 \mu \mathrm { M }$ </td><td>0.342–0.567</td></tr><tr><td>PRMT5</td><td>24</td><td>16</td><td> $1 8 6 \mathrm { n M - } 1 4 \dot { 3 } \mu \mathrm { M }$ </td><td>0.479–0.597</td></tr><tr><td>JAK1</td><td>24</td><td>21</td><td> $3 \mathrm { n M - 2 8 } \mu \mathrm { M }$ </td><td>0.385-0.684</td></tr><tr><td>CDK1</td><td>22</td><td>22</td><td> $1 1 4 \mathrm { n M - 2 3 } \mu \mathrm { M }$ </td><td>0.279-0.706</td></tr><tr><td>HDAC6</td><td>24</td><td>15</td><td> $1 1 5 \mathrm { n M - 2 2 } \dot { \mu } \mathrm { M }$ </td><td>0.278-0.625</td></tr><tr><td>MNK2</td><td>24</td><td>19</td><td> $3 9 \mathrm { n M - 3 8 } \mu \mathrm { M }$ </td><td>0.269-0.492</td></tr><tr><td>MMP-9</td><td>23</td><td>3</td><td> $4 8 0 \mathrm { n M - 1 . 3 } \mu \mathrm { M }$ </td><td>0.443-0.563</td></tr><tr><td>KDR/VEGFR2</td><td>24</td><td>10</td><td> $3 8 \mathrm { n M } { - } 8 2 \mu \mathrm { M }$ </td><td>0.429–0.617</td></tr><tr><td>ELAÑE</td><td>24</td><td>2</td><td> $1 . 2 2 \mathrm { - } 2 9 . 4 \mu \mathrm { M }$ </td><td>0.407–0.500</td></tr><tr><td>Overall</td><td>213</td><td>131</td><td></td><td></td></tr></table>

A  
![](images/068751d8d523232cbf4bb38b44b9f7ce99f470500b10e19ca703378231b94001.jpg)  
B

![](images/1c4ac611a0d9a5b97ae87cec4540996015aa6b3daa330e081d27c98c21c97d8f.jpg)  
Figure A11: (a) Tanimoto similarity of the confirmed inhibitors to the nearest molecule in the training data; all fall below 0.60. KDR/VEGFR2 and ELANE are excluded: neither protein appears in the training corpus, nor does any protein above 90% sequence identity to either, so neither target has training molecules to compare against. (b) Tanimoto similarity of the confirmed inhibitors to the nearest known active for their target, across all nine targets.

## H.5 TYK2 kinase assay

TYK2 activity was measured in a radiometric filter-binding assay [49]. TYK2 transfers the γ-phosphate from a mixture of ATP and $[ \gamma - ^ { 3 3 } \mathrm { P ] - A T P }$ to the peptide substrate KKSRGDYMT-MQIG $( \mathrm { { 2 0 } } \mu { \mathrm { M } } )$ . Reactions were run in standard kinase bufer (20 mM HEPES pH 7.5, 10 mM MgCl<sub>2</sub>, 1 mM EGTA, 0.01% Brij-35, 0.02 mg/mL BSA, 0.1 mM $\mathrm { N a _ { 3 } V O _ { 4 } }$ , 2mM DTT, 1% DMSO) with ATP at its apparent $K _ { m } \ ( 1 0 \mu \mathrm { M } ) ;$ ; compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes, the reaction then proceeded 2 hours at $2 5 ^ { \circ } \mathrm { C } ,$ , and <sup>33</sup>P-substrate was captured on P81 ion-exchange filter paper and washed in 0.75% phosphoric acid. Staurosporine was the reference inhibitor $( \mathrm { I C } _ { 5 0 } = 0 . 3 9 \mathrm { n M } )$

## H.6 PRMT5 methyltransferase assay

PRMT5/MEP50 activity was measured in a radiometric filter-binding assay. Transfer of a tritiated methyl group from [<sup>3</sup>H]-S-adenosylmethionine (SAM, 1 µM) to histone H2A $( 5 \mu \mathrm { M } )$ proceeded for 60 minutes at $\mathrm { \dot { 3 } 0 ^ { \circ } C }$ in 50 mM Tris-HCl (pH 8.5), 5 mM $\mathrm { M g C l _ { 2 } }$ , 50 mM NaCl, 1 mM DTT, 0.01% Brij-35, and 1% DMSO. Compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes. Reactions were quenched and transferred to nylon filter paper, which was washed in PBS to remove unreacted radioactive SAM. S-adenosyl-L-homocysteine (SAH) was the reference inhibitor $( \mathrm { I C } _ { 5 0 } = 3 . 4 \mu \mathrm { M } )$

## H.7 JAK1 kinase assay

JAK1 activity was measured in a radiometric filter-binding assay. JAK1 transfers the γ-phosphate from a mixture of ATP and $[ \gamma - ^ { 3 3 } \mathrm { P ] - A T P }$ to a poly[Glu:Tyr] (4:1) substrate $( 0 . { \overset { - } { 2 } } \operatorname { m g } / \operatorname { m L } )$ . Reactions were run in standard kinase bufer with ATP at its apparent $K _ { m }$ $( 1 5 \mu \mathrm { M } )$ ; compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes, the reaction then proceeded 2 hours at $2 5 ^ { \circ } \mathrm { C } .$ , and <sup>33</sup>P-substrate was captured on P81 ion-exchange filter paper and washed in 0.75% phosphoric acid. Staurosporine was the reference inhibitor $( \mathrm { I C } _ { 5 0 } = 0 . 7 6 \mathrm { n M } )$

## H.8 CDK1 kinase assay

CDK1 activity was measured in a radiometric filter-binding assay of the CDK1/cyclin B complex. CDK1 transfers the γ-phosphate from a mixture of ATP and $[ \gamma - ^ { 3 3 } \mathrm { P } ] \mathrm { - A T P }$ to retinoblastoma protein substrate $( { \bar { 5 } } \mu \mathrm { M } )$ . Reactions were run in standard kinase bufer with ATP at its apparent $K _ { m } \ \mathrm { ( 5 0 } \mu \mathrm { M ) }$ ; compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes, the reaction then proceeded 2 hours at $2 5 ^ { \circ } \mathrm { C }$ , and <sup>33</sup>P-substrate was captured on P81 ion-exchange filter paper and washed in 0.75% phosphoric acid. Staurosporine was the reference inhibitor $( \mathrm { I C } _ { 5 0 } \dot { = } \dot { 1 } . 8 \mathrm { n M } )$

## H.9 HDAC6 deacetylase assay

HDAC6 deacetylase activity was measured in a two-step fluorogenic assay using the acetylated peptide substrate from p53 residues 379–382 $( \mathrm { R H \bar { K } K ( A c ) \mathrm { - } A M C , \ 1 0 \mu M } ) ^ { \sim }$ ; deacetylation followed by trypsin digestion frees 7-amino-4-methylcoumarin (AMC) for quantification. Reactions were run in 50 mM Tris (pH 8.0), 137 mM NaCl, 2.7 mM KCl, 1 mM $\mathrm { M g C l _ { 2 } } .$ 1 mg/mL BSA, and 1% DMSO. Compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes; the reaction then proceeded 1 hour at $3 0 ^ { \circ } \mathrm { C }$ . Trichostatin A was the reference inhibitor $( \mathrm { I C } _ { 5 0 } = 1 . 3 5 \mathrm { n M } )$ ).

## H.10 MNK2 kinase assay

MNK2 activity was measured in a radiometric filter-binding assay. MNK2 transfers the γ-phosphate from a mixture of ATP and $[ \gamma - ^ { 3 3 } \mathrm { P ] - A T P }$ to myelin basic protein from porcine brain $( \mathrm { { 2 0 } } \mu \mathrm { { M } ) }$ . Reactions were run in standard kinase bufer with ATP at its apparent $K _ { m }$ $( 1 0 0 \mu \mathrm { { M } ) }$ ; compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes, the reaction then proceeded 2 hours at $2 5 ^ { \circ } \mathrm { C } ,$ and <sup>33</sup>P-substrate was captured on P81 ion-exchange filter paper and washed in 0.75% phosphoric acid. Staurosporine was the reference inhibitor $( \mathrm { I C } _ { 5 0 } = 1 8 . 6 \mathrm { n M } )$ [50].

## H.11 MMP-9 protease assay

MMP-9 activity was measured in a continuous fluorogenic protease assay using the MMP-9 catalytic domain (residues 88–438) and the FRET peptide substrate 5-FAM/QXL $( 5 \mu \mathrm { M } )$ Compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes; the reaction was started by substrate addition and monitored as a time course, with the initial linear portion of the slope (signal/min) analyzed. GM6001 (ilomastat) was the reference inhibitor $( \mathrm { I C } _ { 5 0 } = 0 . 1 3 \mathrm { n } \bar { \mathrm { M } } )$

## H.12 ELANE (neutrophil elastase) protease assay

ELANE activity was measured in a continuous fluorogenic protease assay using the peptide substrate MeOSuc-AAPV-AMC $( 1 0 \mu \mathrm { M } )$ . Compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes; the reaction was started by substrate addition and monitored as a time course, with the initial linear portion of the slope (signal/min) analyzed. Sivelestat was the reference inhibitor $( \mathrm { I C } _ { 5 0 } = 3 . 4 \mathrm { \dot { n } M ) \ [ 5 1 ] }$

## H.13 KDR/VEGFR2 kinase assay

KDR/VEGFR2 activity was measured in a radiometric filter-binding assay. KDR/VEGFR2 transfers the γ-phosphate from a mixture of ATP and $[ \gamma - ^ { 3 3 } \mathrm { P ] - A T P }$ to a poly[Glu:Tyr] (4:1) substrate $( 0 . 2 \mathrm { { \dot { m g } } / \mathrm { { m L } } ) }$ . Reactions were run in standard kinase bufer with ATP at its apparent $K _ { m } \ ( 2 . 5 \mu \mathrm { M } ) ;$ compounds were delivered by acoustic transfer (Echo 550) and pre-incubated 20 minutes, the reaction then proceeded 2 hours at $2 5 ^ { \circ } \mathrm { C }$ , and <sup>33</sup>P-substrate was captured on P81 ion-exchange filter paper and washed in 0.75% phosphoric acid. Staurosporine was the reference inhibitor $( \mathrm { I C } _ { 5 0 } \overset { \cdot } { = } 5 . 4 \mathrm { n M } )$ .