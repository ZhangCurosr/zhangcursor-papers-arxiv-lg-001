## Title

OPRD-Imp: a dataset of experimentally observed impurities and side products in multi-step pharmaceutical syntheses

## Authors

Heejeong Kim<sup>1</sup>, Seonbin Kim<sup>2</sup>, Jihwan Kim<sup>2</sup>, Tim Pinkhassik<sup>3</sup>, Magdalena Lederbauer<sup>3</sup>, Suong Tran<sup>4</sup>, Connor W. Coley<sup>3,5</sup>, Juyoung Yoon<sup>1</sup> \*, Joonyoung F. Joung<sup>2</sup>\*

Affiliations

1. Department of Chemistry and Nanoscience, Ewha Womans University, Seoul, 03760, South Korea

2. Department of Chemistry, Kookmin University, Seoul, 02707, South Korea

3. Department of Chemical Engineering, Massachusetts Institute of Technology, Cambridge, MA, USA

4. Department of Chemistry, Massachusetts Institute of Technology, Cambridge, MA, USA

5. Department of Electrical Engineering and Computer Science, Massachusetts Institute of Technology, Cambridge, MA, USA

corresponding author(s): Juyoung Yoon (jyoon@ewha.ac.kr), Joonyoung F. Joung (jjoung@kookmin.ac.kr)

## Abstract

Predicting reaction impurities is a critical yet persistently data-limited challenge in pharmaceutical process chemistry. While machine learning models for reaction outcome prediction have advanced rapidly, benchmarking their ability to predict impurities remains impractical due to the absence of curated, publicly accessible datasets. We present OPRD-Imp, a dataset of 229 organic reactions drawn from 62 publications in Organic Process Research & Development, each annotated with experimentally observed impurities and their mechanistic origins. A total of 784 impurity entries is provided, classified into four origin categories. Reactions are represented as reaction SMILES and impurities as individual SMILES strings, with multi-step synthetic route connectivity preserved. All data were manually extracted from the source publications by trained chemists following a standardized extraction protocol, with expert chemical validation and manual structure drawing. OPRD-Imp is intended to serve as a benchmark resource for evaluating models that predict reaction impurities, providing experimentally reported impurity annotations and mechanistic origin labels that are not systematically represented in conventional reaction datasets.

## Background & Summary

Identifying and controlling impurities is one of the central challenges in synthetic chemistry, from early-stage laboratory research to large-scale pharmaceutical manufacturing. Unanticipated impurities complicate purification, compromise yields, and obscure structure– activity relationships by introducing uncharacterized compounds into biological assays.<sup>1,2</sup> Critically, their formation is difficult to anticipate even for experienced chemists. Side products can arise from competing or cascading pathways, trace metal or catalyst residues, and impurities already present in the starting materials that are carried through and chemically transformed.<sup>3,4</sup> As a result, chemists are frequently confronted with unexpected products and must resort to laborious analytical work to detect and assign them. Structures are typically confirmed either by isolating the impurity on scale-up for NMR characterization, or by forming a structural hypothesis from mass spectrometry and mechanistic reasoning and comparing it against an independently synthesized authentic standard.<sup>5</sup> This makes exhaustive impurity mapping resource-intensive even at the bench scale.

These challenges are amplified as synthetic routes progress toward process development and manufacturing, where incomplete impurity knowledge creates critical blind spots during scaleup and intersects with strict regulatory obligations. Comprehensive impurity profiling is mandatory at every stage of drug development, as governed by the International Council for Harmonisation (ICH) guidelines Q3A, Q3B, and M7, which control impurities, including genotoxic impurities, down to the parts-per-million level.<sup>6–8</sup> Failure to identify and control impurities can result in delayed regulatory submissions, costly batch reprocessing, or postmarket drug recalls. Consequently, the experimental burden of impurity identification and these regulatory demands motivate a complementary, data-driven strategy that learns the patterns of impurity formation directly from accumulated experimental observations.

The field of machine learning for organic chemistry has become increasingly effective for a range of reaction-related tasks, including forward reaction prediction, retrosynthetic planning, and reaction-condition recommendation.<sup>9–18</sup> More recently, mechanistic reaction prediction approaches such as FlowER<sup>19</sup> and Reactron<sup>20</sup> have extended reaction modeling beyond majorproduct prediction by representing elementary reaction steps. These developments suggest that mechanistic and data-driven models could also be applied to impurity and side-product prediction. However, most existing reaction-prediction benchmarks remain focused on the major intended product, while impurity prediction has received comparatively limited attention.<sup>21–24</sup>

A critical bottleneck is the absence of publicly available datasets containing experimentally reported impurity annotations for reaction-level benchmarking. Large reaction datasets such as the United States Patent and Trademark Office (USPTO),<sup>25</sup> the Open Reaction Database (ORD),<sup>26</sup> and ORDerly<sup>17</sup> provide broad coverage of reaction conditions and major products but do not systematically annotate experimentally observed impurities. Even commercially licensable datasets based on the patent literature or journal articles contain a small fraction of entries reporting multiple products. Consequently, even models designed to predict side products or reaction mechanisms cannot be rigorously evaluated against a curated set of experimentally observed impurities.

We selected Organic Process Research & Development (OPRD) as the primary literature source for curating an impurity-focused dataset for several reasons. First, OPRD publications focus specifically on the development and scale-up of synthetic routes for active pharmaceutical ingredients and pharmaceutical intermediates, where impurity identification is particularly consequential. Second, OPRD papers routinely characterize and report the structures of observed impurities in detail, including analytical evidence, as part of the process development procedure.<sup>27</sup> This contrasts with journals focusing more on academic synthesis or discovery chemistry, where impurities are less systematically reported, and with large patent-derived datasets, where impurity data are typically absent. Third, the pharmaceutical process chemistry context of OPRD publications makes the resulting dataset directly relevant to impurity prediction in real-world synthetic workflows.

Herein, we present OPRD-Imp, a publicly available benchmark dataset of experimentally observed organic reaction impurities from selected Organic Process Research & Development literature. The dataset comprises 229 reactions sourced from 62 publications published between 2002 and 2026, containing 784 annotated impurity entries with four mechanistic origin classifications. Key distinguishing features include (1) experimentally reported and structurally curated impurity annotations; (2) origin labels that enable category-specific evaluation of impurity prediction models; (3) connectivity between sequential reaction steps, together with per-impurity provenance paths that trace each carried-over impurity to the species it originated from, enabling analysis of impurity propagation across synthetic routes; and (4) reaction-class annotation of every product and impurity. The dataset was constructed through structured manual extraction followed by multi-round expert curation and validation against the primary literature. Together, these features make OPRD-Imp a curated benchmark for evaluating models for impurity prediction, designed to prioritise annotation reliability over scale.

## Methods

![](images/921ab4372ae7e32c4e1eada1365a1359983dc84a72191d3eabce6d5c551f21e5.jpg)  
Figure 1. Overview of the dataset construction workflow.

## Data Source and Inclusion Criteria

Publications were identified by searching the Organic Process Research & Development (OPRD) journal website (pubs.acs.org/journal/oprdfk) using the keywords “impurity” or “impurity validation.” All publications published between 2000 and 2026 were considered and the literature search was conducted in May 2026. Articles published before 2000 were not included because, in preliminary screening, they seldom reported impurity structures with supporting analytical evidence and therefore did not meet the inclusion criteria.

Following keyword-based retrieval, each candidate paper was manually reviewed to determine eligibility. We included papers reporting primary experimental data—at least one reaction step performed by the authors, with the reagents, solvents, and reaction conditions specified—in which impurity information is explicitly provided for at least one reaction step, either as a structural assignment or as a description sufficient for unambiguous structure determination. Review articles, perspective pieces, and papers reporting only computational or in silico results were excluded, as were papers in which impurities are mentioned only by name without a clear structural assignment. Following this screening procedure, 62 publications were included in the final dataset.<sup>27–88</sup>

## Structured Information Extraction

The dataset construction workflow is summarized in Figure 1. To ensure systematic and consistent extraction across 62 papers with varying reporting styles, all information was extracted manually from the article PDFs, including the supporting information, by trained chemists following a standardized extraction protocol. Manual extraction without a standardized protocol risks inconsistency between different annotators and inadvertent omission of impurities mentioned only briefly in one location. Therefore, an extraction checklist was designed to provide a systematic framework that can be applied across all papers, reducing annotator-dependent variability.

## For each paper, the annotator was required to:

1. Extract the DOI of the paper.

2. Identify and enumerate each reaction step, formatted as "[Starting material].[Reagents] → [Product]" using compound names or numbers given in the paper; these identifiers were converted to SMILES during curation (see Expert Curation and Structure Drawing).

3. Apply an exhaustive impurity extraction checklist, explicitly requiring examination of: abstract, introduction, results and discussion section, experimental section, analytical data, all tables, figures and schemes, and supporting information.

4. For each reaction step, populate a structured table with: compound identifier (name or scheme number as used in the paper), mechanistic origin classification, quantification status, and a mandatory direct quotation from the paper as evidence.

The following categories of impurity report were captured: (i) quantitatively reported impurities (e.g., "X was present at 1.2% by HPLC area"); (ii) qualitatively mentioned impurities (e.g., "trace amounts of compound Y were detected"); (iii) negatively mentioned compounds (e.g., "no formation of Z was observed"); and (iv) implicitly referenced compounds (e.g., "conditions were optimized to suppress formation of W"). This approach was adopted to minimize selective reporting bias in the extraction step itself. Categories (iii) and (iv) were captured so that no compound discussed in the paper was overlooked during screening; however, only compounds that the source reports as actually observed were retained as impurity entries, and compounds explicitly reported as not formed were excluded from the released dataset. Furthermore, a direct supporting quotation from the source text was recorded for every extracted impurity. This provided a concrete anchor for verification in the subsequent curation step. Any impurity lacking a plausible direct quote was flagged for cautious cross-check during expert review.

Each impurity was assigned to one of five origin labels during extraction: the four mechanistic categories defined in Table 1 plus an additional “unknown” label used when the paper does not indicate the origin. Entries labelled “unknown” were resolved into one of the four final categories, or removed, during expert curation, so that the released dataset uses only the four categories in Table 1. The final origin label for every entry was assigned by consensus of the three expert curators; the assignment and verification procedure is described in the Technical Validation section.

![](images/dad86215f94aea6181f76ffc966ec51317a1ec7ca3d8a92ea906dade7ae3097d.jpg)  
Figure 2. Representative multi-step synthetic route illustrating impurity origin classification and cross-step propagation.<sup>27</sup> (a) Two consecutive steps from a six-step synthetic route: 16\_2 and 16\_3. (b) Impurity origin assignments for each step. Arrows linking impurities across the two steps are drawn for illustrative purposes.

<table><tr><td>Origin label</td><td>Definition</td></tr><tr><td>minor_product</td><td>An impurity formed through a side reaction during the current reaction step, including regioselectivity or stereoselectivity failures, over- reaction, or other competing transformations.</td></tr><tr><td>SM_impure</td><td>An impurity already present in a starting material or reagent before the current reaction step and not generated by the current reaction.</td></tr><tr><td>SM_impure_carryover</td><td>An impurity originating from an SM_impure species that undergoes chemical transformation during the current reaction step.</td></tr><tr><td>carryover</td><td>An impurity inherited from a preceding synthetic step and transferred into the current reaction without chemical transformation. Unreacted starting material or reagent remaining in the product stream is</td></tr></table>

Table 1. Impurity origin categories and their definitions

Origin labels were assigned according to the earliest identifiable source of the impurity relative to the current reaction step: impurities originating from the starting-material feed were classified as SM\_impure, their chemically transformed products as SM\_impure\_carryover, impurities inherited from preceding steps as carryover, and impurities newly formed during the current step as minor\_product. Unreacted starting material is recorded only where the source publication explicitly reports it as an impurity in the product stream; such entries are directly identifiable through their parent path referencing a reactant or reagent, so they can be retained or excluded depending on the intended application. Likewise, carryover entries are recorded only for impurities explicitly reported in the source publication.

A representative multi-step synthetic route in which all four origin categories are observed is shown in Figure 2, together with a detailed view of a single reaction step illustrating each category.

The SM\_impure category is a special case. These impurities originate from the startingmaterial feed and are therefore not predictable from the reaction mechanism. Their inclusion nonetheless provides a complete picture of the observable impurity profile as encountered in real process chemistry settings.

## Expert Curation and Structure Drawing

The initial extraction records were reviewed and processed by a team of three trained chemists. Expert curation served four functions.

1. Verification against source documents: Each extracted impurity, together with its recorded evidence quotation, was checked against the original publication, and unsupported or chemically implausible entries were flagged for rejection or correction. The multi-round verification protocol is detailed in the Technical Validation section.

2. Chemical structure drawing and SMILES generation: Each retained impurity, together with the reactants, reagents, solvents, and product of its reaction, was drawn as a chemical structure and converted into a canonical SMILES string, yielding a machine-readable representation of every reaction and its associated impurities. Structures, including stereochemistry, were transcribed as drawn in the source publication. For compounds not explicitly drawn in the source publication (for example, common reagents and solvents referred to only by name), SMILES strings were taken from the corresponding PubChem entries.

3. SMILES representability: Species lacking a standard covalent SMILES representation, such as heterogeneous catalysts and organometallic reagents with coordinate or non-covalent bonds, were encoded using simplified or disconnected ionic conventions, following PubChem representations where available (see Usage Notes).

4. Data structuring and format: Curated data were compiled and structured as JSON to enable structured querying and ease of integration into predictive model evaluation scripts. For papers reporting multi-step synthetic sequences, sequential steps within the same paper were linked using the next\_reaction\_id field. This encodes the directed structure of the synthesis route, allowing downstream users to trace the propagation of carryover impurities through overall steps. Branching routes (where a single intermediate leads to multiple synthetic paths) are represented by providing multiple entries in the next\_reaction\_id list.

Procedures that the source publication reported as one-pot were retained with the step delineation shown in the paper’s reaction scheme, recorded as separate entries linked by next\_reaction\_id and grouped through the one\_pot\_group field, which lists the reaction\_id keys of all steps in the procedure. Because the intermediates in such procedures were not isolated or independently analysed, observed impurities were attributed to the step at which the source reported them (typically the final step); the empty impurity lists of the preceding one-pot steps therefore do not represent negative examples.

## Data Records

## Repository and File Availability

OPRD-Imp is deposited at figshare<sup>89</sup> and is freely accessible under a Creative Commons Attribution 4.0 International (CC BY 4.0) licence. The repository contains the following files:

OPRD\_imp.json — primary dataset in JSON format (schema described in the JSON Schema section below)

OPRD\_imp\_dashboard.html — interactive dashboard for browsing the dataset reaction by reaction, distributed as a single self-contained HTML file (see Usage Notes)

## JSON Schema

The file contains a single top-level JSON object keyed by reaction\_id (e.g., "16\_2"), so that any reaction can be retrieved directly by its identifier. Each value is a reaction object with the fields described in Table 2.

<table><tr><td>Field</td><td>Type</td><td>Description</td></tr><tr><td>reaction_id (key) string</td><td></td><td>Unique identifier in the format N_M (e.g., 16_2), where N (1– 62) indexes the source publication and M denotes the reaction step within that publication; used as the key of the top-level object rather than as a field. All eligible reaction steps meeting the inclusion criteria were retained from each selected publication.</td></tr><tr><td>paper_id</td><td>integer</td><td>Index of the source publication (1-62); identical to the N component of reaction_id, provided as a separate field for convenient grouping by publication.</td></tr><tr><td>step</td><td>integer</td><td>Step number within the synthetic route; identical to the M component of reaction id.</td></tr><tr><td>doi</td><td>string</td><td>DOl of the source publication, recorded on every step of the route.</td></tr></table>

<table><tr><td>Field</td><td>Type</td><td>Description</td></tr><tr><td>reaction_smiles</td><td>string</td><td>Reaction SMILES, reactants.reagents.solvents&gt;&gt;product, assembled from the canonical component SMILES of the fields below.</td></tr><tr><td>reactants</td><td>array of strings</td><td>Canonical SMILES of the substrate(s); one entry per reaction, except reaction 26_3, which has two substrates.</td></tr><tr><td>reagents</td><td>array of strings</td><td>Canonical SMILES of the reagents; catalysts are recorded as reagents.</td></tr><tr><td>solvents</td><td>array of strings</td><td>Canonical SMILES of the solvents; empty array for neat reactions.</td></tr><tr><td>product name_rxn</td><td>string</td><td>Canonical SMILES of the major intended product. NameRxn class of the product-forming reaction (see</td></tr><tr><td></td><td>string</td><td>Dataset Statistics).</td></tr><tr><td>reactants_for_i mpurities</td><td>string</td><td>Reactant side used for impurity prediction: the components of reaction_smiles together with every additional species needed to account for the impurities of this reaction, namely the starting-material impurities and the parent species of carried-over impurities. Given as a single dot-separated</td></tr><tr><td>impurities</td><td>array of objects</td><td>SMILES string. One object per reported impurity with keys smiles (canonical SMILES), origin (one of the four labels in Table 1), parent (see below), name_rxn (NameRxn class of the impurity- forming reaction), name_rxn_input (the reaction SMILES submitted to NameRxn), and note; empty array for reactions</td></tr><tr><td>notes</td><td>string or null</td><td>with no reported impurities. Free-text annotation, mainly documenting non-standard SMILES conventions for heterogeneous catalysts (see Usage Notes); null otherwise.</td></tr><tr><td>one_pot_group</td><td>array of strings or null</td><td>reaction_id keys of all steps belonging to the same one-pot procedure, including the entry itself (e.g., [&quot;16_1&quot;, &quot;16_2&quot;]); null for reactions that are not part of a one-pot procedure.</td></tr><tr><td>next_reaction_id array of</td><td>strings</td><td>Subsequent step(s), each a key of the top-level object; empty array for terminal steps.</td></tr></table>

Table 2. JSON schema for each reaction entry

For every carryover, SM\_impure\_carryover, and SM\_impure entry, the parent field records the specific species from which the impurity originated: an unconsumed reactant or reagent, or an impurity entry of an earlier step. The value is a path expressed as a JSON array: [reaction\_id, field, index] for a reactant, reagent, or solvent (e.g., ["16\_1", "reactants", 0]), and [reaction\_id, "impurities", index, "smiles"] for an impurity of a preceding step. Following the path by successive indexing of the top-level object (e.g., data["16\_1"]["impurities"][0]["smiles"]) returns the SMILES string of the parent species. Omitting the final "smiles" element instead returns the parent impurity record, which carries its own parent field; repeating this step traces the formation chain of an impurity back to its original source. The field is null for minor\_product entries, whose source is the reaction step itself. For five entries of other origins whose source could not be determined from the publication, parent is also null and the note field reads "parent unidentified in source". For model input, the reaction-level reactants\_for\_impurities field provides the union of all species needed to account for the impurities of a reaction, so that carried-over impurities can be explained from the reactant side alone; 299 additional species are recorded across 128 reactions.

## Dataset Statistics

The dataset contains 229 reactions from 62 publications spanning 2002–2026, with a total of 784 experimentally observed impurity entries. Summary statistics are provided in Figure 3.

a  
![](images/36327728a1d9ba5ea193ff8e14fab9c124634bd4d63862a0dc9866d77a6ec84a.jpg)

b  
![](images/1053723690d5743141472d6428e8266e01c6c555f5a2d74a9c6700f6e01078c3.jpg)

C  
![](images/d62e3f4abe57eff518030479410566bf051812f60787720c77eca1285311ff76.jpg)

d  
![](images/b2d9a3e14ce9ff50a34b9edc1911ff23389cee28d224d1f93f534c98a45b3d67.jpg)  
Figure 3. Statistical overview of the dataset. (a) Distribution of the 62 source publications by publication year. (b) Distribution of the number of impurity entries per reaction. (c) Number of major products (229) and of impurity entries in each origin category (784 in total). (d) Distribution of synthetic route lengths, i.e., the number of reaction steps recorded per publication.

Reactions with an empty impurity list fall into two groups. Nine reaction steps are intermediate steps of one-pot procedures that were not independently analysed.<sup>27,61,64,65,77,79</sup> The remaining 19 steps (8.3% of 229) are those for which the source publication reports no impurity for that step; they belong to routes whose other steps report impurities and are retained to keep each route complete.<sup>27,28,37,50,57,58,62,64,67</sup> None of them reports complete conversion or a quantitative (100%) yield (isolated yields, where given, range from 44% to 99%), eight of the 19 were telescoped into the following step without isolation or analysis of the intermediate, and only three are reported with 100% HPLC purity of the isolated product, at 77–92% isolated yield. Impurities may therefore have been removed during workup prior to analysis or simply not detected.

A small number of reactions contain two entries with the same structure. These are retained because the source publication reports the compound as arising through two distinct formation pathways, and each entry carries its own origin label and parent assignment.

The dataset spans a broad range of reaction types encountered in pharmaceutical process chemistry, including amide bond formation, reductive amination, N- and O-alkylation, protecting group installation and removal, palladium-catalyzed cross-coupling, and asymmetric transformations. The distribution of impurity counts across reactions is rightskewed, with a median of three impurity entries per reaction (mean 3.4) and a long tail of twelve reactions with ten or more impurities reflecting complex process development scenarios (Figure 3b).

Every reaction was classified with NameRxn<sup>90</sup> (NextMove Software), which assigns a hierarchical reaction name. The recorded major product and each impurity were represented as a reaction whose reactant side comprises the recorded reactants, reagents and solvents together with the starting-material impurities and the parent species of carried-over impurities, and whose product side is the major product or the impurity. Entries that involve no net chemical transformation—unreacted starting material or reagent remaining in the product stream, and impurities carried over unchanged from a preceding step—were nevertheless submitted to NameRxn, and were almost always assigned to the “Resolution” class (typically “Other separation”). In total, 1,013 reactions (229 products and 784 impurities) were classified; the assigned class is stored in the name\_rxn field of the reaction or impurity, and the exact reaction SMILES submitted to NameRxn in name\_rxn\_input. We then compared the class distribution of our dataset with those of the pre-processed one-step reaction datasets USPTO (\~0.97 M reactions) and Pistachio (\~4.3 M reactions).<sup>91</sup>

a  
![](images/b68d4971069047cfbdbfb8fd758eecf518ca3942cf576ecaae089b69f7a917a9.jpg)

![](images/e7ac3cb5aa6bed9c8d187a11085add627fdab46c2fc11204b906c54bde02e91a.jpg)  
C  
Figure 4. Reaction-class analysis of the dataset. (a) Fraction of reactions in each first-level NameRxn class for the major products and the impurities of OPRD-Imp, compared with the USPTO and Pistachio reaction sets. (b) Percentage of “Unrecognized” reactions among impurities, split by annotated origin (n = 341, 294, 101 and 48 for minor\_product, carryover, SM\_impure\_carryover and SM\_impure, respectively); the dashed line marks the rate for the major products (19%). (c) A representative unrecognized impurity that is nonetheless a genuine transformation (reaction 4\_3): from the same reactants, a single ${ \mathsf { S } } _ { \mathsf { N } } { \mathsf { A } } { \mathsf { r } }$ gives the intended N-arylation product (left, a named reaction), whereas displacement of the second chloride by a second equivalent of the aminoindazole gives the bis-adduct impurity (right), a net double substitution with no single NameRxn label.

The major products of our impurity dataset populate the classes expected of medicinal- and process-chemistry routes and are almost indistinguishable from USPTO and Pistachio (Figure 4a). The impurities show a different trend: relative to the products, they are enriched mainly in "Resolution" and "Unrecognized". The "Resolution" enrichment, however, is largely explained by unreacted precursors. Of the 224 impurities assigned to this class, 217 (97%) are species already present on the reactant side, i.e., leftover starting material or reagent. The unrecognized fraction is larger for impurities than for the major products (Figure 4a). Splitting the impurities by their annotated origin, we also observed that genuine minor products, the actual products of the step's chemistry, are "Unrecognized" 58% of the time (Figure 4b). The "Unrecognized" label reflects the coverage of the NameRxn ontology as much as the novelty of the chemistry: many impurities are genuine net transformations, including over-reactions, eliminations, and rearrangements, that simply lack a single named-reaction label. Figure 4c shows a representative case. From the same two reactants, a single ${ \mathsf { S } } _ { \mathsf { N } } { \mathsf { A } } { \mathsf { r } }$ coupling of the aminoindazole with 2,4-dichloropyrimidine gives the intended N-arylation product, while displacement of the second chloride by a second equivalent of the aminoindazole gives the bis-adduct. This is a net double substitution that NameRxn cannot assign to a single class and therefore labels as "Unrecognized".

## Technical Validation

All SMILES strings in the dataset were parsed using RDKit (version 2025.03).<sup>92</sup> Canonical SMILES were generated for every entry, and any entry for which RDKit failed to produce a valid molecular graph was investigated and either corrected or excluded; all SMILES in the released file are canonical. The internal consistency of the released file was then checked programmatically. Every reaction\_smiles was verified to reconstruct exactly from the canonical component SMILES of its reactants, reagents, solvents and product, and every name\_rxn\_input to terminate in the SMILES of the entry it annotates. All 164 next\_reaction\_id references resolve to keys of the top-level object, and the nine one\_pot\_group sets are mutually consistent, each member listing the same group. Of the 784 impurity entries, the 438 carrying a parent path all resolve to a SMILES string of a species recorded at the same or an earlier step of the same publication; the remaining entries are the 341 minor\_product entries, whose source is the reaction step itself, and the five entries whose source could not be determined from the publication.

In addition to automated SMILES validation, every entry in the dataset was subjected to a multi-round expert review. Each source publication was examined at least three times in total, once during the initial curation and twice more by curators who had not performed the original extraction. Reviewers cross-checked every impurity against the original publication, using the direct evidence quotation recorded during the initial extraction as the primary reference. Entries whose evidence quotation could not be found in the source, or whose structural assignment was chemically implausible, were corrected or removed, yielding the final set of 784 curated impurity entries.

Origin labels were assigned by consensus of the three curators. Where the source publication explicitly stated the origin of an impurity, that assignment was followed. In all other cases, the most plausible origin was determined from the reaction mechanism and recorded only after agreement among all three curators. Stereochemical assignments in the SMILES strings were transcribed as drawn in the source publications and were not independently re-assigned.

## Usage Notes

OPRD-Imp is distributed as a single JSON file (OPRD\_imp.json). To reconstruct multi-step synthetic routes, follow the next\_reaction\_id field, which links each reaction to its subsequent step(s) within the same publication. Tracking these links recovers the directed route graph and allows carryover impurities to be tracked across sequential steps. An empty impurities list means that no impurity was reported in the source publication under its described analysis. Because absence of a report is not evidence of absence of impurities, these steps should be treated as unannotated rather than as negative examples (see Dataset

Statistics). Empty lists belonging to intermediate steps of one-pot procedures (identified by the one\_pot\_group field) are not negative examples, as those steps were not independently analysed. Quantitative information (for example HPLC area percentages) was captured during extraction where reported, but is not included in the released schema because reporting conventions were heterogeneous and frequently incomplete; incorporating normalised quantification is left for future work.

Reactions carried out in a single vessel without isolation of intermediates are identified by the one\_pot\_group field, an array listing the reaction\_id keys of all steps belonging to the same one-pot procedure (e.g., ["16\_1", "16\_2"]); the field is null for all other reactions, so no string parsing is required. The notes field documents simplified SMILES conventions adopted for species that lack a well-defined molecular structure: palladium on carbon is denoted [C].[Pd], with the carbon support represented by a bare carbon atom, Raney nickel [Al].[Ni], reflecting its Ni–Al alloy composition, activated carbon [C], and palladium(II) chloride [Cl][Pd][Cl]. Molecular hydrogen and nitrogen follow the PubChem notations [HH] and N#N, respectively. Ligands containing ferrocene units, such as 1,1′- bis(diphenylphosphino)ferrocene (dppf), are represented by a disconnected ionic convention, two cyclopentadienylphosphine anions with [Fe+2], as their coordination bonds cannot be expressed in standard SMILES.

To inspect the dataset, we provide an interactive web dashboard, distributed as a single selfcontained HTML file (OPRD\_imp\_dashboard.html) in the same figshare repository<sup>89</sup> that renders each entry as chemical structures, showing its reactants, reagents, solvents, product, and observed impurities with their origin labels, and follows the next\_reaction\_id links to display multi-step synthetic routes. The dashboard allows users to search and view individual reactions without parsing the raw JSON.

## Data Availability

The OPRD-Imp dataset (OPRD\_imp.json) and the interactive dashboard (OPRD\_imp\_dashboard.html) are available from figshare under a CC BY 4.0 licence at https://doi.org/10.6084/m9.figshare.34075524.<sup>89</sup>

## Code Availability

Reaction-class annotation used NameRxn 3.7.7 (NextMove Software)<sup>90</sup>, a commercial tool; the exact reaction SMILES submitted to NameRxn and the returned class labels are stored in the dataset (name\_rxn\_input and name\_rxn fields) so that the annotation can be reproduced by licence holders.

## Funding

This work was supported by the AI Seoul Tech Research Support Program of the Seoul Future Foundation (to H.K.), the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (RS-2026-25484674, to J.F.J.), and by the Yangyoung Foundation through the Academic Research Support Program (to J.F.J.).

## Author contributions

H.K. conceived and designed the study, designed the data extraction protocol and developed the associated software, and performed the formal analysis and visualization. H.K., S.K., and J.K. carried out the data curation, chemical structure drawing, and validation. T.P. and M.L. reviewed and manually corrected entries in the dataset. S.T. performed the reaction-class analysis. H.K. wrote the original draft, and H.K., S.K., J.K., J.F.J., T.P., M.L., S.T., and C.W.C.

5. Maggio, R. M., Calvo, N. L., Vignaduzzo, S. E. & Kaufman, T. S. Pharmaceutical impurities and degradation products: uses and applications of NMR techniques. J. Pharm. Biomed. Anal. 101, 102–122 (2014).

reviewed and edited the manuscript. J.F.J. supervised the project. J.F.J. and J.Y. acquired funding and administered the project. All authors reviewed and approved the final manuscript.

## Competing interests

The authors declare no competing interests.

## References

1. Görög, S. The importance and the challenges of impurity profiling in modern pharmaceutical analysis. TrAC Trends Anal. Chem. 25, 755–757 (2006).

2. Görög, S. Critical review of reports on impurity and degradation product profiling in the last decade. TrAC Trends Anal. Chem. 101, 2–16 (2018).

3. Garrett, C. E. & Prasad, K. The art of meeting palladium specifications in active pharmaceutical ingredients produced by Pd-catalyzed reactions. Adv. Synth. Catal. 346, 889– 900 (2004).

4. Teasdale, A. et al. Risk assessment of genotoxic impurities in new chemical entities: strategies to demonstrate control. Org. Process Res. Dev. 17, 221–230 (2013).

6. ICH Harmonised Guideline. Q3A(R2): Impurities in New Drug Substances (International Council for Harmonisation, 2006).

7. ICH Harmonised Guideline. Q3B(R2): Impurities in New Drug Products (International Council for Harmonisation, 2006).

8. ICH Harmonised Guideline. M7(R2): Assessment and Control of DNA Reactive (Mutagenic) Impurities in Pharmaceuticals to Limit Potential Carcinogenic Risk (International Council for Harmonisation, 2023).

9. Tu, Z., Stuyver, T. & Coley, C. W. Predictive chemistry: machine learning for reaction deployment, reaction development, and reaction discovery. Chem. Sci. 14, 226–244 (2023).

10. Jin, W., Coley, C. W., Barzilay, R. & Jaakkola, T. Predicting organic reaction outcomes with Weisfeiler-Lehman network. In Advances in Neural Information Processing Systems 30 (eds Guyon, I. et al.) 2607–2616 (Curran Associates, 2017).

11. Fooshee, D. et al. Deep learning for chemical reaction prediction. Mol. Syst. Des. Eng. 3, 442–452 (2018).

12. Schwaller, P. et al. Molecular Transformer: a model for uncertainty-calibrated chemical reaction prediction. ACS Cent. Sci. 5, 1572–1583 (2019).

13. Tetko, I. V., Karpov, P., Van Deursen, R. & Godin, G. State-of-the-art augmented NLP transformer models for direct and single-step retrosynthesis. Nat. Commun. 11, 5575 (2020).

14. Chen, S. & Jung, Y. Deep retrosynthetic reaction prediction using local reactivity and global attention. JACS Au 1, 1612–1620 (2021).

15. Roh, J. et al. Higher-level strategies for computer-aided retrosynthesis. ACS Cent. Sci. 12, 345–357 (2026).

16. Gao, H. et al. Using machine learning to predict suitable conditions for organic reactions. ACS Cent. Sci. 4, 1465–1476 (2018).

17. Wigh, D. S., Arrowsmith, J., Pomberger, A., Felton, K. C. & Lapkin, A. A. ORDerly: data sets and benchmarks for chemical reaction data. J. Chem. Inf. Model. 64, 3790–3798 (2024).

18. Sun, X., Liu, J., Mahjour, B., Jensen, K. F. & Coley, C. W. Data-driven recommendation of agents, temperature, and equivalence ratios for organic synthesis. Chem. Sci. 16, 18176– 18189 (2025).

19. Joung, J. F. et al. Electron flow matching for generative reaction mechanism prediction. Nature 645, 115–123 (2025).

20. Chen, S., Park, K. S., Kim, T., Han, S. & Jung, Y. Predicting chemical reaction outcomes based on electron movements using machine learning. Preprint at https://arxiv.org/abs/2503.10197 (2025).

21. Joung, J. F., Casetti, N., Raghavan, P. & Coley, C. W. An overview of reaction outcome prediction with physics-based and data-driven methods. Chem. Soc. Rev. 55, 6768–6813 (2026).

22. Joung, J. F. et al. Reproducing reaction mechanisms with machine-learning models trained on a large-scale mechanistic dataset. Angew. Chem. Int. Ed. 63, e202411296 (2024).

23. Hoque, A., Das, M., Baranwal, M. & Sunoj, R. B. ReactAIvate: a deep learning approach to predicting reaction mechanisms and unmasking reactivity hotspots. In Proc. 27th European Conference on Artificial Intelligence (ECAI 2024) 2645–2652 (IOS Press, 2024).

24. Coley, C. W. et al. Grand challenges for predictive modeling in small molecule drug discovery. Preprint at https://doi.org/10.26434/chemrxiv.15000615 (2026).

25. Lowe, D. M. Extraction of Chemical Structures and Reactions from the Literature. PhD thesis, Univ. Cambridge (2012).

26. Kearnes, S. M. et al. The Open Reaction Database. J. Am. Chem. Soc. 143, 18820–18826 (2021).

27. Haneishi, T. et al. Development of a kilogram-scale synthesis of a novel anti-HCV agent, CH4930808. Org. Process Res. Dev. 22, 236–240 (2018).

28. Yang, A. et al. Improved synthesis and impurity identification of (R)-lacosamide. Org. Process Res. Dev. 23, 818–824 (2019).

29. Gao, S., Gao, X., Yang, Z. & Zhang, F. Process research and impurity control strategy of esketamine. Org. Process Res. Dev. 24, 555–566 (2020).

30. Maeda, S. et al. Identification of tamsulosin-dimeric impurity, leading to the efficient manufacturing process of the active pharmaceutical ingredient. Org. Process Res. Dev. 29, 2715–2724 (2025).

31. Terentjeva, S., Muceniece, D. & Lu̅sis, V. Process-related impurities of pazopanib. Org. Process Res. Dev. 23, 2057–2068 (2019).

32. Guvvala, V., Subramanian, V. C., Anireddy, J. & Konda, M. Study on the isolation and chemical investigation of potential impurities in dexrazoxane using 2D-NMR and LC-PDA-MS. Org. Process Res. Dev. 21, 11–17 (2017).

33. Lu, H. et al. Identification, synthesis, characterization, and control strategy establishment for process impurities of baloxavir marboxil. Org. Process Res. Dev. 28, 1170–1185 (2024).

34. Anderson, B. G., Bauta, W. E., Cantrell, W. R. Jr, Engles, T. & Lovett, D. P. Isolation, synthesis, and characterization of impurities and degradants from the clofarabine process. Org. Process Res. Dev. 12, 1229–1237 (2008).

35. Zhong, Z. et al. Identification, characterization, synthesis, and strategy for minimization of potential impurities observed in the synthesis of solithromycin. Org. Process Res. Dev. 21, 1371–1382 (2017).

36. Zhang, T. et al. One-pot process for synthesis of nalbuphine hydrochloride and impurity control strategy. Org. Process Res. Dev. 24, 1707–1717 (2020).

37. Srinivas, K. et al. An improved, scalable, and impurity-free process for tolterodine tartrate. Org. Process Res. Dev. 9, 314–318 (2005).

38. Tyagi, R. et al. Identification, synthesis, and control of process-related impurities in the antipsychotic drug substance brexpiprazole. Org. Process Res. Dev. 22, 1471–1480 (2018).

39. Hu, Z. et al. Identification, synthesis, and strategy for minimization of potential impurities in the preclinical anti-HBV drug Y101. Org. Process Res. Dev. 17, 1156–1167 (2013).

40. Ding, Z. et al. Improved synthesis and impurity control strategy of penehyclidine hydrochloride. Org. Process Res. Dev. 28, 210–224 (2024).

41. Mezei, T. et al. Manufacture of high-purity meloxicam via its novel potassium salt monohydrate. Org. Process Res. Dev. 13, 567–572 (2009).

42. Feng, W. D., Zhuo, S. M. & Zhang, F. L. Process research and impurity control strategy for obeticholic acid, a farnesoid X receptor agonist. Org. Process Res. Dev. 23, 1979–1989 (2019).

43. Goura, R. et al. An alternate, efficient synthetic process for a hemolytic anemia drug: mitapivat sulfate. Org. Process Res. Dev. 28, 4024–4033 (2024).

44. Li, C. et al. Process development and large-scale synthesis of BTK inhibitor BIIB068. Org. Process Res. Dev. 24, 1199–1206 (2020).

45. Shi, X., Chang, H., Grohmann, M., Kiesman, W. F. & Kwok, D. I. A. Process development of an N-benzylated chloropurine at the kilogram scale. Org. Process Res. Dev. 19, 437–443 (2015).

46. Zheng, L. et al. Reaction of nicardipine with nitrous acid or tert-butylnitrite produces nicardipine pyridine analog and nicardipine oxime derivatives, rather than N-nitroso nicardipine. Org. Process Res. Dev. 30, 683–691 (2026).

47. Walker, D. G. et al. Development of an early phase process for a potent oligodendrocyte precursor cell differentiator. Org. Process Res. Dev. 28, 2203–2212 (2024).

48. Lu, J. et al. Identification, synthesis, and comprehension of an imidazole N-3 regioisomeric impurity of olmesartan medoxomil key intermediate. Org. Process Res. Dev. 26, 1247–1257 (2022).

49. Mortensen, M. A. et al. Process development, impurity control, and production of a novel tubulin inhibitor. Org. Process Res. Dev. 16, 1811–1817 (2012).

50. Delacroix, K. et al. Process development of a novel route to rilpivirine hydrochloride. Org. Process Res. Dev. 28, 524–531 (2024).

51. Standley, E. A. et al. Synthesis of rovafovir etalafenamide (Part I): active pharmaceutical ingredient process development, scale-up, and impurity control strategy. Org. Process Res. Dev. 25, 1215–1236 (2021).

52. Goto, Y. et al. Practical synthesis of triethylammonium (Z)-2-[2-(Boc-amino)thiazol-4-yl]-2- (trityloxyimino)acetate, the side chain of cefmatilen. Org. Process Res. Dev. 9, 57–61 (2005).

53. Santandrea, E., Waldraff, C., Gerber, G., Moreau, M. & Beney, P. Development of the latephase manufacturing process of ZPL389: control of process impurities by enhanced process knowledge. Org. Process Res. Dev. 25, 1190–1205 (2021).

54. Tamboli, Y. et al. Process development and scale-up of the novel β-lactamase inhibitor WCK 6395. Org. Process Res. Dev. 27, 496–504 (2023).

55. Lin, B. et al. Development of a robust and scalable process for the large-scale preparation of vadadustat. Org. Process Res. Dev. 25, 960–968 (2021).

56. Jiao, S. et al. Impurity identification and scale-up of a novel glycopeptide antibiotic. Org. Process Res. Dev. 25, 2390–2402 (2021).

57. Goverdhan, G. et al. An improved and scalable process for zafirlukast: an asthma drug. Org. Process Res. Dev. 13, 67–72 (2009).

58. Kumar, U. S., Sankar, V. R., Rao, M. M., Jaganathan, T. S. & Buchi Reddy, R. Investigational study into the formation of methoxy derivative and other impurities during the optimization of eletriptan hydrobromide. Org. Process Res. Dev. 16, 1917–1920 (2012).

59. Ide, N. D. et al. Synthesis of filibuvir. Part III. Development of a process for the reductive coupling of an aldehyde and a β-keto-lactone. Org. Process Res. Dev. 18, 45–56 (2014).

60. Li, B., Kasthurikrishnan, N., Ragan, J. A. & Young, G. R. Structural determination of a process impurity in a furan–pyrrole heteroatom exchange reaction. Org. Process Res. Dev. 6, 64–66 (2002).

61. Holte, D. et al. Development of a scalable process for an LPAR1 antagonist with a complex form landscape, GS-2278. Org. Process Res. Dev. 29, 2378–2387 (2025).

62. He, J. et al. Development of a scalable synthetic route and a crystal morphology characterization method for the PDE4B inhibitor HY-072808. Org. Process Res. Dev. 30, 407– 418 (2026).

63. Hyde, A. M., Calabria, R., Arvary, R., Wang, X. & Klapars, A. Investigating the underappreciated hydrolytic instability of 1,8-diazabicyclo[5.4.0]undec-7-ene and related unsaturated nitrogenous bases. Org. Process Res. Dev. 23, 1860–1871 (2019).

64. Wagner, A. M. et al. Synthesis of lenacapavir sodium: active pharmaceutical ingredient process development and scale-up. Org. Process Res. Dev. 28, 3382–3395 (2024).

65. Halama, A., Stach, J., Rádl, S. & Benediktová, K. Identification of an unexpected impurity in a new improved synthesis of lesinurad. Org. Process Res. Dev. 22, 1861–1867 (2018).

66. Miller, W. D., Fray, A. H., Quatroche, J. T. & Sturgill, C. D. Suppression of a palladiummediated homocoupling in a Suzuki cross-coupling reaction. Development of an impurity control strategy supporting synthesis of LY451395. Org. Process Res. Dev. 11, 359–364 (2007). 67. Haycock-Lewandowski, S. J., Wilder, A. & Åhman, J. Development of a bulk enabling route to maraviroc (UK-427,857), a CCR-5 receptor antagonist. Org. Process Res. Dev. 12, 1094–1103 (2008).

68. Sampath Kumar, U., Ravi Sankar, V., Bharani Kumar, S., Pandi Prabhu, M. & Mahender Rao, S. An investigation into the formation of impurity B during the optimization of naratriptan hydrochloride. Org. Process Res. Dev. 13, 468–470 (2009).

69. Young, I. S., Qiu, Y., Smith, M. J., Hay, M. B. & Doubleday, W. W. Preparation of a tricyclopropylamino acid derivative via Simmons–Smith cyclopropanation with downstream intramolecular aminoacetoxylation for impurity control. Org. Process Res. Dev. 20, 2108–2115 (2016).

70. Liu, Z. et al. Development and implementation of an aluminum-promoted phosphorylation in the uprifosbuvir manufacturing route. Org. Process Res. Dev. 25, 661–667 (2021).

71. Zhou, Z. et al. Development of a robust and scalable process for the large-scale preparation of vilazodone. Org. Process Res. Dev. 26, 2855–2863 (2022).

72. Carroll, M. P. et al. Development of an improved route for the synthesis of an abemaciclib intermediate. Org. Process Res. Dev. 23, 2549–2555 (2019).

73. Yang, Y. & Hansen, L. Regioisomer formation with agmatine guanidino group, its implications for agmatine peptide cyclization, and application of bis-Boc-agmatine. Org. Process Res. Dev. 26, 1837–1844 (2022).

74. Aluri, K., Pullagurla, H. R., Annareddy, S. P. R. & Pitta, B. R. Investigation of Pd catalyst for key coupling reaction in asciminib synthesis and impurity characterization. Org. Process Res. Dev. 29, 2520–2529 (2025).

75. Huang, K. et al. Improved synthetic process of baloxavir marboxil intermediate 3- benzyloxy-4-oxo-4H-pyran-2-carboxylic acid. Org. Process Res. Dev. 27, 1015–1026 (2023).

76. Aydin, G. et al. Metal-catalyzed C–N bond forming reaction selection and process development for the manufacture of AZD7594. Org. Process Res. Dev. 28, 559–576 (2024).

77. Lobben, P. C. et al. Control strategy for the manufacture of brivanib alaninate, a novel pyrrolotriazine VEGFR/FGFR inhibitor. Org. Process Res. Dev. 19, 900–907 (2015).

78. Hanselmann, R., Johnson, G., Reeve, M. M. & Huang, S. T. Identification and suppression of a dimer impurity in the development of delafloxacin. Org. Process Res. Dev. 13, 54–59 (2009).

79. Sankareswaran, S., Mannam, M., Chakka, V., Mandapati, S. R. & Kumar, P. Identification and control of critical process impurities: an improved process for the preparation of dolutegravir sodium. Org. Process Res. Dev. 20, 1461–1468 (2016).

80. Kumar, Y., Tewari, N., Nizar, H., Rai, B. P. & Singh, S. K. Identification and control of a process-related impurity in the chlorination of 3-hydroxy-3-carbacephem. Org. Process Res. Dev. 7, 758–761 (2003).

81. DeBaillie, A. C. et al. Optimization of an aminothiazine ring formation: integrating modeling with experiments to maximize yield by minimizing impurity formation. Org. Process Res. Dev. 19, 1244–1256 (2015).

82. Pramanik, C. et al. An efficient, scalable process for benzphetamine hydrochloride. Org. Process Res. Dev. 18, 495–500 (2014).

83. Bova, A., Saba, M., Novelli, S. & Pecora, F. Development of a novel process for the preparation of the anti-inflammatory drug betamethasone sodium phosphate. Org. Process Res. Dev. 27, 189–197 (2023).

84. Zhou, A. N., Chen, X. W., Qi, Y. L., Duan, G. L. & Li, J. Q. Process development of an efficient kilogram-scale preparation of a preferential dopamine D3 versus D2 receptor antagonist SIPI 6398 as a new antipsychotic candidate. Org. Process Res. Dev. 23, 1442–1449 (2019).

85. Jiang, B. et al. Optimized process and quality evaluation for ketamine hydrochloride. Org. Process Res. Dev. 27, 1330–1338 (2023).

86. Yang, Q. et al. Development of a scalable process for an IL-17A inhibitor LY3509754: Part III. Assembly of drug substance, salt formation, and impurity control. Org. Process Res. Dev. 29, 1019–1035 (2025).

87. Saravanan, M., Satyanarayana, B. & Reddy, P. P. An improved and impurity-free large-scale synthesis of venlafaxine hydrochloride. Org. Process Res. Dev. 15, 1392–1395 (2011).

88. Schülé, A. et al. Monitoring and control of genotoxic impurity acetamide in the synthesis of zaurategrast sulfate. Org. Process Res. Dev. 14, 1008–1014 (2010).

89. Kim, H. et al. OPRD-Imp: a dataset of experimentally observed impurities and side products in multi-step pharmaceutical syntheses. figshare 10.6084/m9.figshare.34075524 (2026).

90. Schneider, N., Lowe, D. M., Sayle, R. A. & Landrum, G. A. Development of a novel fingerprint for chemical reactions and its application to large-scale reaction classification and similarity. J. Chem. Inf. Model. 55, 39–53 (2015).

91. Tran, S. B. A., Roh, J. & Coley, C. W. Quantifying the failure modes of current one-step retrosynthesis models. Chem. Sci. 17, 14054–14063 (2026).

92. RDKit: Open-source Cheminformatics. https://www.rdkit.org