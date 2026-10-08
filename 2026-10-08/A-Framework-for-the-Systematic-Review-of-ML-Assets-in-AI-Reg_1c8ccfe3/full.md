# A Framework for the Systematic Review of ML Assets in AI Registries

Alexandra González #

Universitat Politècnica de Catalunya - BarcelonaTech (UPC), Barcelona, Spain

Quim Motger #

Universitat Politècnica de Catalunya - BarcelonaTech (UPC), Barcelona, Spain

Xavier Franch #

Universitat Politècnica de Catalunya - BarcelonaTech (UPC), Barcelona, Spain

Silverio Martínez-Fernández #

Universitat Politècnica de Catalunya - BarcelonaTech (UPC), Barcelona, Spain

## Abstract

Background: Modern software systems increasingly rely on Machine Learning (ML) assets (i.e., pre-trained models, datasets, benchmarks) for building, evaluating, and integrating ML-based systems. However, current exploration, selection and reuse practices of ML assets are not supported by systematic retrieval methodologies comparable to those used in traditional evidence synthesis. Consequently, in practice, ML asset selection is often presented as a settled design decision, supported by informal justification rather than a traceable, evidence-based, and updatable selection process. Aims: This paper explores how systematic review methods can support ML asset retrieval. In doing so, we aim to make their selection transparent and reproducible, grounded in explicit evidence, and ultimately better suited to its intended use. Method: We analyze established systematic review practices from scientific literature and adapt their phases (i.e., planning, conducting, and documenting) to Artificial Intelligence (AI) registries, treating ML assets as first-class units of analysis. The resulting framework integrates registry-aware search strategies, cross-registry schema alignment, and dependency-driven ML asset exploration. Results: We conceptualize ML asset retrieval as a systematic and reproducible process rather than an ad hoc activity, and propose a framework for structured ML asset discovery. Conclusions: This work illustrates how systematic review principles can be extended beyond scientific literature to support evidence synthesis over evolving AI registries.

2012 ACM Subject Classification General and reference → Surveys and overviews; Software and its engineering → Software creation and management

Keywords and phrases Models, Datasets, AI registries, Systematic Reviews

Digital Object Identifier 10.4230/LIPIcs.CVIT.2016.23

## 1 Introduction

Modern software development workflows increasingly rely on Machine Learning (ML) assets, including pre-trained models, datasets, and benchmarks [34, 30, 1]. In this context, researchers and practitioners use open, large-scale registries such as Hugging Face and Kaggle, often referred to as Artificial Intelligence (AI) ecosystems [33], to select models for system integration, reuse datasets in experiments, or identify benchmarks for evaluation.

Despite this growing reliance on ML assets, current practices for selecting them remain ad hoc and are rarely documented in a systematic way. This lack of standardization reflects a fundamental mismatch between ML assets and traditional scholarly publications. Unlike research papers, ML assets are hosted in rapidly evolving registries with sparse documentation and volatile (if any) quality signals [6, 24, 30]. Although initiatives such as model cards [21], datasheets for datasets [13], and more recent eforts such as AI Bill of Materials [23] aim to improve transparency and standardize artifact documentation, they do not provide methodological guidance for systematically retrieving ML assets in AI registries. Furthermore, diferent registries expose heterogeneous schemes and traceability signals. For instance, registries such as Hugging Face provide popularity indicators (e.g., likes) and explicit traceability links to downstream models trained or fine-tuned on a given dataset. In contrast, others, such as Kaggle, rely on alternative metrics (e.g., upvotes) while ofering limited support for tracing reuse. This leads to uneven levels of traceability across registries, where relationships between ML assets may be explicitly documented or only implicitly inferred. When considered jointly, these heterogeneous signals form a complex network, spanning links between datasets and models, model-to-model adaptations (e.g, quantization, fine-tuning) [35], and dataset-to-dataset lineages (e.g., subsets, augmentations) [20]. This network is critical, as a model is unlikely to perform reliably in production if its deployment context diverges from its training data, or if these datasets reflect unwanted biases [13, 26].

These challenges reveal the need for more rigorous and reproducible methodologies for ML asset discovery. To address this need, systematic reviews and related evidence synthesis methodologies ofer a promising foundation. They provide rigorous procedures to guide the identification, selection, and analysis of relevant studies, consolidating the state of the art within a research field. In Software Engineering (SE), secondary studies methodologies such as Systematic Literature Reviews (SLRs) [19, 3], Systematic Mapping Studies (SMSs) [27], and Rapid Reviews [4] support diferent evidence synthesis objectives, ranging from in-depth analysis to broad topic exploration and accelerated review processes. These methodologies define structured processes for planning, conducting, and reporting reviews, while emphasizing transparency and reproducibility. They rely on explicit metadata (e.g., titles, abstracts, keywords) and stable indexing provided by traditional academic databases (e.g., Scopus, IEEE Xplore, ACM Digital Library), which together enable controlled and repeatable study retrieval. However, traditional evidence synthesis methods are not directly applicable to AI registries. While they assume structured and standardized corpora, AI ecosystems are dynamic, heterogeneous, and governed by evolving and partially implicit dependency relationships. These diferences are summarized in Table 1.

Table 1 Key diferences between systematic reviews of scientific papers and ML assets.
<table><tr><td rowspan="2">Aspect</td><td colspan="2">Systematic Reviews of...</td></tr><tr><td>Scientific Papers</td><td>ML Assets</td></tr><tr><td>Primary unit</td><td>Academic publications</td><td>Models, datasets, benchmarks</td></tr><tr><td>Data sources</td><td>Academic databases</td><td>AI registries</td></tr><tr><td>Metadata quality</td><td>Structured and standardized</td><td>Sparse, inconsistent, heterogeneous</td></tr><tr><td>Relationships</td><td>Citation networks</td><td>Model-dataset dependencies</td></tr><tr><td>Reproducibility</td><td>Stable indexed records</td><td>Rapidly evolving registries and metadata</td></tr></table>

This paper addresses this mismatch by proposing a framework for systematically retrieving ML assets from AI registries, inspired by systematic reviews of scientific literature and adapted to the characteristics of open, heterogeneous, and continuously evolving data sources.

## 2 Related Work

Empirical SE studies that rely on ML assets typically treat their retrieval as an implementation detail rather than a formalized research step. In practice, datasets, pre-trained models, and benchmarks are selected through ad hoc filtering strategies, manually curated subsets, or registry-specific heuristics that are rarely documented in a systematic form. For instance, Prenner et al. [28] evaluated pre-trained transformer models on 13 small SE datasets selected from the literature guided by task diversity, dataset size, and availability of baselines. Similarly, other studies adopt pragmatic selection criteria when constructing experimental corpora, although these vary across objectives. Durán et al. [11] selected 12 code generation models from Hugging Face by constraining model size, focusing on base models, and ensuring representativeness across sizes, release dates, and afiliations. González et al. [15] further filtered models based on the availability of training dataset information, popularity signals, and size constraints. Across these works, ML asset inclusion is driven by experimental requirements rather than reproducible retrieval strategies grounded in registry structure.

A complementary line of work investigates AI ecosystems as interconnected networks of models, datasets, and related artifacts. These studies shift the focus from individual ML assets to their relationships, highlighting that ML artifacts form complex dependency structures. Dasoulas et al. [8] proposed a knowledge graph integrating datasets, pipelines, implementations and scientific works from sources such as OpenML, Kaggle, and Papers with Code. Similarly, Yang et al. [33] analyzed the ecosystem of Large Language Models (LLMs) for code in Hugging Face, combining manual and LLM-assisted analysis, and incorporating relationships between models and their training datasets through snowballing-based explora tion. Jiang et al. [18] explored cross-repository organization of AI artifacts across Hugging Face and GitHub. From an SE perspective, cataloguing eforts have examined registries such as Hugging Face and highlighted inconsistencies in metadata quality [16]. While these works provide important insights into the structure of AI ecosystems, they primarily focus on analyzing them rather than using their structure to support systematic retrieval.

Taken together, these streams of work point to a dual reality: ML assets are central to empirical SE and its intersection with AI, and are embedded in complex ecosystems with rich dependency structures. However, evidence synthesis practices remain grounded in document-centric corpora and have not evolved accordingly. From the perspective of prior research on ML asset usage and AI ecosystem analysis, there is still no established approach. The following points summarize the most relevant implications of this lack of formalization:

Obscure selection rationale. ML asset choices are rarely accompanied by explicit criteria, leaving the reasoning behind a given selection opaque and dificult to assess.

Limited reproducibility. Without documented selection criteria, verifying if an ML asset suits a specific context is dificult, hindering asset reuse and cross-study comparison.

Suboptimal selection outcomes. Without the guidance on where, what, and how to search, ML asset discovery becomes ad hoc, rarely yielding the best results for the task.

Costly and unreliable re-execution. Without documented retrieval processes, fastevolving ML catalogs make repeating a data selection efortful, error-prone, and unreliable.

This gap exposes the need for methodological support that enables structured, transparent, and reproducible ML asset discovery in evolving AI ecosystems.

## 3 The Review Process

To formalize the derivation of the proposed framework, we adopted a structured design process rooted in established systematic practices and state-of-the-art literature. First, we analyzed protocols for systematic reviews, focusing on SLRs and their operational phases of planning, conducting, and documenting reviews [19, 3]. Second, we conducted a thematic analysis of primary studies addressing ML asset selection, encompassing recent model-centric [10, 16, 33] and data-centric [22] perspectives. This allowed us to isolate both their operational processes and recurring methodological challenges, such as metadata incompleteness, dependencyaware exploration, and reproducibility threats. Third, we explicitly aligned the processes and challenges extracted from these studies and the core SLR phases and steps. This synthesis directly shaped the dimensions of the resulting methodology, ensuring that these emerging results remain grounded while addressing the concrete, open challenges of evolving AI registries.

![](images/59278e427ae4ff98de70c1a61e56eeafc72404eedc028fe68d2bcd0f0df85085.jpg)  
Figure 1 Systematic Review of ML Assets process.

## 3.1 Plan the Review

Before conducting the review, it is necessary to assess its necessity, as this determines whether a systematic retrieval efort is justified. Once this need is established, the next step is to formulate the search questions that guide the review, ensuring that they are precise enough to delimit the ML asset space. This phase concludes with the definition and validation of the review protocol, which specifies the operational procedures, inclusion boundaries, and methodological choices that will structure the subsequent phases.

1 Identify Review Need. Given the rapid evolution and scale of AI registries, an ML asset review begins with a preliminary assessment to determine whether a systematic retrieval efort is warranted and not already addressed by prior studies. Researchers must examine limitations in existing asset collections, such as missing quality assessments, incomplete dependency mappings, or outdated evaluation results. This evidence is then used to justify the resources required for a rigorous systematic review.

2 Specify Search Question. The review process is grounded in a clearly articulated search question that can be derived from a broader Research Question (RQ). Considering ML assets, this step requires translating a study’s high-level objective into a specification of the ML asset type and its intended role within the study. Here, we need to clarify whether the intended retrieval target corresponds to a model, dataset, or benchmark, and how this ML asset is expected to support the research goal.

For example, a broader RQ such as: “How efective are ML approaches for automating code generation tasks?” may be decomposed into multiple ML asset search questions:

“Which models are designed or adapted for source code generation?”

“Which datasets are used to train ML models for code generation?”

“Which benchmarks are used to evaluate the performance of code generation models?” In this case, each search question operationalizes a specific aspect of the broader research intent by targeting distinct types of ML assets relevant to code generation tasks.

3 Develop Review Protocol. Establishing the protocol in advance is essential to reduce researcher bias and improve reproducibility. In this context, a pre-defined protocol documents all elements of the review together with the associated planning information. Beyond the background, which motivates the review, and the search questions that guide the investigation, the protocol must define the following elements:

Search strategy: Specifies target registries while accounting for their structural heterogeneity and considering their primary focus. Model-centric registries are primarily oriented toward hosting pre-trained models, including registries such as Model Zoo, PyTorch Hub, ONNX Model Zoo, NVIDIA NGC, and MATLAB Model Hub. In contrast, dataset-centric registries focus on the reuse of datasets and include OpenML, Harvard Dataverse, Zenodo, and Figshare. Finally, mixed-purpose registries such as Hugging Face and Kaggle support multiple ML asset types, including both models and datasets.

Identification strategy: Determines how registry data will be accessed, including public APIs, downloadable registry dumps, web scraping procedures, or hybrid approaches. Such decisions are important because registries difer in the amount and granularity of accessible data. For instance, popularity signals are not consistently exposed across registries. At the time of writing, Kaggle provides dataset-level download counts via its web interface and API, whereas equivalent signals for models are only available through the web interface. Registries also difer in identifier schemes, ranging from human-readable composite identifiers (e.g., user\_id/asset\_id in Hugging Face and Kaggle) to opaque numeric identifiers in their APIs (e.g., OpenML datasets identified by integer IDs).

Selection criteria: Defines the inclusion and exclusion criteria for ML assets. These should be adapted to the characteristics of the target registries, as each registry may difer in structure and available data. They may include several dimensions, such as scope, traceability, and maintenance activity. Since registry information may be incomplete, the protocol should also specify how missing information will be handled.

Quality assessment strategy: Defines how ML assets will be evaluated in terms of verifiability of reported claims (e.g., performance metrics, training configurations), internal validity of the experimental evidence (suficiency of methodological detail), and external validity of the ML assets.

Data extraction and synthesis strategy: Specifies which attributes and dependencies will be extracted from selected ML assets, as well as how these data will be synthesized.

Dissemination strategy: Covers reporting guidelines, versioning practices, and the planned schedule for conducting and re-executing the review.

4 Validate Review Protocol. Evaluating the protocol before full-scale execution is critical for discovering mistakes. Protocol validation should confirm that the proposed search strategy retrieves relevant ML assets across the selected registries. In practice, validation may involve executing pilot searches over a representative subset, inspecting the retrieved data, and verifying whether the resulting set adequately reflects the intended review scope. Based on these results, the protocol may be iteratively refined.

## 3.2 Conduct the Review

The second phase of the review operationalizes the protocol through systematic identification, filtering, and analysis of ML assets. During this phase, researchers execute the search strategy by identifying ML assets, and applying the inclusion, exclusion, and quality assessment criteria established in the protocol. In addition, snowballing should be employed to identify further relevant ML assets. Once the final set of ML assets has been selected, researchers can perform data extraction and synthesize the collected evidence.

Given the potentially large number of ML assets retrieved from AI registries such as Hugging Face, which hosts 2.9M models and 1M datasets as of May 2026, the execution of this phase may require support from automated techniques. Recent studies explore the use of LLMs to support empirical workflows in mining software repositories [14, 29], as well as structured prompting and validation frameworks such as PRIMES [9] and empirical guidelines for SE studies involving LLMs [2]. These approaches emphasize combining LLM-assistance with rule-based filtering and human validation to improve scalability while preserving reliability and mitigating risks such as hallucinations.

5 Primary Search: Identify ML Assets. The primary search aims to construct an initial set of candidate ML assets. Each search question specified in the protocol is translated into one or more search queries by identifying core concepts (e.g., ML asset type, task) and expanding them with synonyms and registry-specific terminology. These queries are not executed against a uniform bibliographic database but across heterogeneous AI registries that expose diferent retrieval mechanisms, including free-text search and metadata filters. As a result, search strings are operationalized as registry-specific queries, combining keywords with filtering tags supported by each registry and adapted to their indexing mechanisms.

Search execution is performed using the available interfaces of each registry. Access to registry data can be achieved either through public APIs or via downloadable snapshots. APIs enable fine-grained querying over metadata fields and textual descriptions, while static dumps provide a complete view of the registry at a given point in time. In API-supported registries, queries are issued through structured endpoints, whereas in registries without queryable APIs, search is performed through web interfaces or downloaded snapshots. A key challenge in API-based retrieval is that APIs often expose incomplete data, omitting information that is only available on the registry’s web interface. To address this limitation, primary search may rely on hybrid retrieval strategies that combine structured API queries with web scraping of ML asset pages to recover missing information directly from the source. However, researchers must account for registry constraints such as rate limits, which bound the volume and frequency of requests, thereby limiting large-scale or highly iterative extraction workflows. As a result, primary search is often conducted in an incremental and iterative manner.

After identifying candidates, duplicates may arise due to mirrored repositories or crossregistry republishing. To address this, deduplication must be applied based on information available in the documentation. ML assets with highly similar documentation are considered copies, and timestamps are used to distinguish between original and replicas [16].

6 Select ML Assets. Following the identification of candidate primary ML assets, the next step consists of selecting those that satisfy the inclusion and exclusion criteria defined in the review protocol. To perform this task, reviewers must examine the information associated with each ML asset in the registry, since the relevance of an ML asset cannot be determined from its identifier alone. In AI registries, a natural starting point for this selection process is the inspection of model and dataset cards [21, 13]. These combine structured metadata, commonly represented through YAML fields (e.g., license type, library), and unstructured natural language descriptions written in Markdown that provide contextual information, including intended use and limitations. As summarized in Table 2, the selection process comprises multiple complementary dimensions to capture ML asset suitability. It is worth noting that these dimensions are conditioned by the signals available in each registry, since not all registries expose the same information.

The extent to which an ML asset matches the review is captured through scope alignment. This is operationalized by examining the tasks explicitly associated with the ML asset in the registry, such as ‘text classification’, as well as related task annotations provided in its card.

In addition, adoption within the ecosystem is approximated through popularity. This is derived from signals such as download counts, likes, or upvotes, when available. While highly popular ML assets may indicate broad community validation and reuse, this measure must be interpreted cautiously due to temporal bias, since recently published ML assets may not have had suficient time to accumulate engagement despite being relevant.

Similarly, the level of ongoing support is reflected through maintenance activity, which captures how actively an ML asset is updated over time. This is inferred from indicators such as the date of the last update, commit history, and contributor engagement. ML assets with recent and sustained maintenance are generally preferred, as they are more likely to remain compatible with evolving frameworks and dependencies.

Another relevant aspect concerns traceability, which describes the ability to connect an ML asset to related resources, such as parent/derived variants, datasets used in downstream or upstream pipelines, or publications associated with a given ML asset. This information is typically available through dedicated sections in the cards, or as unstructured references embedded in the description. For example, a model may detail the base model from which it was derived, fine-tuned, or adapted versions of it, as well as the training datasets used.

Beyond these aspects, the quality of documentation is reflected through metadata completeness, which captures how thoroughly an ML asset is described to support informed assessment and reuse. This is evaluated based on the presence of fields, such as license information, library dependencies, and ML asset size. When such information is missing or incomplete, the ability to verify eligibility is significantly reduced.

Finally, technical characteristics are considered through two complementary perspectives. On the one hand, model-specific characteristics capture properties of trained models, including training data, evaluation benchmarks, and reported performance metrics. On the other hand, dataset-specific ones describe properties of the underlying data, such as modality and format.

<table><tr><td>Dimension</td><td>Purpose</td><td>Examples</td></tr><tr><td>Scope</td><td>Determine relevance to the review</td><td>ML task, application domain</td></tr><tr><td>Popularity</td><td>Estimate usage and visibility</td><td>Likes, downloads, upvotes</td></tr><tr><td>Maintenance</td><td>Evaluate evolution and support</td><td>Recent updates, contributors</td></tr><tr><td>Traceability</td><td>Identify dependency relationships</td><td>Parent/derived ML assets</td></tr><tr><td>Metadata</td><td>Assess documentation sufficiency</td><td>License, library, size</td></tr><tr><td>Model-Specific</td><td>Capture model characteristics</td><td>Training dataset, evaluation data</td></tr><tr><td>Dataset-Specific</td><td>Characterize data properties</td><td>Modalities, format</td></tr></table>

Table 2 Selection dimensions used in the systematic review of ML assets.

7 Secondary Search: Snowballing. After identifying and selecting primary ML assets, researchers can extend the retrieval process by applying snowballing strategies [32] adapted to AI registries. Here, ML assets form interconnected ecosystems in which models and datasets are often linked through explicit metadata or implicit references.

In this context, backward snowballing can be performed by inspecting the provenance information of a selected ML asset, i.e., identifying the resources from which it originates. For models, this involves tracing the base model and the training datasets. For instance, a code-generation model may indicate that it was fine-tuned from a general-purpose model and trained on curated code corpora. In Hugging Face, this is explicitly supported through structured metadata fields such as base\_model, which links a model to its parent models, and datasets, which indicates training datasets. For datasets, backward snowballing involves identifying their origin sources, such as upstream datasets from which they were derived.

## 23:8 A Framework for the Systematic Review of ML Assets in AI Registries

Conversely, forward snowballing can be conducted by identifying downstream ML assets that reuse a given model or dataset. For models, this includes fine-tuned versions, adapters (e.g., LoRA or PEFT variants), quantized models, or merged architectures derived from a base model. For datasets, forward snowballing involves identifying downstream ML assets that are built on top of a given dataset, such as models trained or fine-tuned using that dataset, or benchmarks that incorporate it as part of their evaluation pipeline.

More broadly, this process reflects that ML assets do not exist in isolation but form interconnected ecosystems of reuse. In their work, Yang et al. [33] operationalize this idea by manually performing snowballing over a selected set of LLMs for code, where they identified training datasets and related models. However, such relationships are not always explicitly encoded in structured metadata and often appear only in unstructured descriptions. As a result, snowballing may require inspection of textual fields to recover implicit dependencies.

Lastly, all newly identified ML assets through snowballing are subsequently subjected to the same inclusion and exclusion criteria defined in the review protocol, and are only incorporated into the final set after passing the selection step.

8 Assess ML Asset Quality. Once the final set of ML assets has been identified and expanded through snowballing, a quality assessment is performed to evaluate its reliability. In contrast to selection, which determines whether an ML asset is relevant to the review scope, this step focuses on the strength of the information that supports its use in subsequent steps. Following established systematic review principles, quality is understood in terms of the extent to which bias is minimized and internal and external validity are maximized. In this context, bias relates to uncertainty introduced by inconsistently reported information, while validity relates to the applicability of the evidence provided.

The assessment considers the extent to which claims associated with an ML asset can be independently verified, including reported performance, training data composition, or intended use, and whether these elements are supported by external sources such as publications. This relates to the presence of potential bias in registry-reported information. It also evaluates the credibility of the experimental evidence provided, focusing on whether reported results are accompanied by suficient methodological detail to support internal validity, including the clarity of evaluation protocols, the definition of benchmarks, and the availability of configuration information that contextualizes performance claims. A further aspect concerns the applicability of the evidence outside the original reporting context, particularly in situations where evaluation settings are ambiguously defined, where training and evaluation boundaries are not clearly specified, or where preprocessing and data construction steps are insuficiently documented, thereby limiting external validity during synthesis.

9 Extract Data. Once the final set of ML assets has been selected, relevant attributes are collected to answer the initial research questions and enable subsequent analysis.

Given the variability of metadata across registries, extraction involves harmonizing representations of equivalent attributes and extracting relevant unstructured information. This includes aligning diferent naming conventions, resolving inconsistencies in identifiers, and mapping registry-specific fields to a unified schema defined in the review protocol. In addition, relevant information is also retrieved from descriptive text associated with each ML asset. Model cards and dataset cards often contain details such as intended use, training procedures, limitations, and known biases. For instance, a dataset card may describe filtering steps applied to remove low-quality or duplicated samples, while a model card may provide information about architecture choices or fine-tuning strategies. These details are extracted through manual inspection or automated parsing techniques, depending on the review setup.

When certain attributes are not available, their absence is explicitly recorded. This is particularly relevant in cases where key information, such as training data provenance or evaluation benchmarks, is omitted. In such situations, the missingness itself becomes an analytical signal that may inform the interpretation of results in subsequent phases.

10 Synthesise Data. The final step of the conducting phase aggregates, normalizes, and categorizes the extracted data to answer the initial search questions. As AI registries often yield an unmanageably large corpus of valid ML assets, a stratified sampling approach can be applied to extract a representative subset [5]. This strategy ensures that the synthesis is not biased toward dominant ML assets, but remains inclusive of the ecosystem’s variety.

## 3.3 Document the Review

The last phase of the review involves documenting it. Specifically, it focuses on ensuring transparency, reproducibility, and longitudinal traceability of the retrieval process. It records search strategies, selection decisions, and extracted data. In addition, it supports iterative updates to account for the dynamic nature of AI registries, enabling continuous refinement and replication of the review process over time.

11 Write Review Report: A structured report captures the full review process, including the rationale behind methodological decisions, the applied search strategy, and the criteria used for ML asset selection. It also records deviations from the initial protocol and provides a clear mapping between search questions and retrieved ML assets. Following established reporting guidelines such as PRISMA [25], the goal is to ensure that the review can be critically assessed, reproduced, and extended by other researchers.

12 Version Snapshots: Version-controlled snapshots of the review must be maintained to preserve the state of AI registries at specific points in time. Unlike traditional literature corpora, ML assets are continuously updated, modified, or deprecated, which introduces a significant threat to reproducibility if only static records are retained. For instance, model weights may be updated, datasets may be expanded with new samples or corrected labels, and metadata such as performance metrics, licenses, or documentation may be revised as communities refine or correct earlier releases. For this reason, review reporting cannot be treated as a single static extraction, but as a sequence of reproducible snapshots. Each snapshot represents a frozen view of the unified database, enabling consistent analysis and comparison across diferent time points.

## 3.4 Need of Iteration

13 Periodic Re-execution: Given the dynamic and continuously evolving nature of AI registries, a one-of execution of the review process is insuficient to maintain up-to-date results. Therefore, we define an iterative loop over the Conduct Review and Document Review phases, while the Plan Review phase remains fixed after its validation to ensure consistency across iterations. The inclusion of new AI registries or any change to the protocol is excluded from the iterative mechanism, as it alters the study definition and afects comparability. Such cases require a new review instance rather than a new iteration.

Within this loop, the Conduct Review phase is periodically re-executed to refresh retrieval, filtering, and analysis of ML assets as registries evolve. This re-execution captures newly added or modified ML assets within those registries. Then, the Document Review phase is updated to record new versioned snapshots and preserve traceability of changes over time. This iterative mechanism reduces manual efort, mitigates drift in ML asset collections, and ensures that the review remains aligned with the current state of the ecosystem within a fixed methodological scope.

## 4 Discussion

Systematic retrieval of ML assets sits at the intersection of established empirical methodologies and emerging challenges introduced by AI registries. Established empirical standards (e.g., ACM/SIGSOFT guidelines) are grounded in document-centric assumptions, and search strategies such as snowballing [32] extend retrieval by exploiting citation relationships. In parallel, mining software repositories research has extended empirical analysis toward software artifacts in platforms such as GitHub [7]. However, AI registries introduce a setting in which metadata is sparse, inconsistent, and heterogeneous [17]. These characteristics make ML assets dificult to fully capture using either traditional systematic reviews of scientific literature or repository mining approaches. Therefore, we position the proposed framework as an adaptation of existing evidence synthesis principles that highlights ML assets as first-class analytical units. The key distinction lies in shifting the unit of analysis from documents or repositories to ML assets as interconnected artifacts.

This perspective aligns with Multivocal Literature Reviews (MLRs) [12], which incorporate grey literature (e.g., preprints, e-prints) alongside published literature (i.e., journal papers, conference proceedings, and books). AI registries introduce a comparable challenge, since relevant information about ML assets (e.g., evaluation signals) is not always peerreviewed. However, while MLRs primarily integrate textual evidence, this framework extends multivocality toward ML assets.

From a broader empirical SE perspective, the proposed framework is also related to data strategies in empirical data-driven research [31]. In particular, it resonates with the idea that the validity of empirical findings depends not only on analysis techniques but also on how data is selected, sampled, and curated from large and evolving AI ecosystems. In this sense, ML asset retrieval can be seen as a form of structured sampling over AI registries, where representativeness, traceability, and bias control become central concerns.

Finally, the lack of traceable ML asset retrieval directly impacts reproducibility in empirical SE. As open science policies become more important in venues such as ESEM or ICSE, transparency must extend beyond code to include the full process of artifact selection. This includes how models and datasets are identified, filtered, and justified. By treating ML asset retrieval as a versioned process, the framework operationalizes reproducibility at the level of ecosystem exploration, turning what is typically an ad hoc methodological step into a verifiable and repeatable component of empirical research.

## 5 Conclusion & Future Work

This emerging results paper proposes a framework for the systematic review of ML assets in AI registries, addressing the lack of structured practices in state-of-the-art studies. By adapting established evidence synthesis principles and lessons learned from pioneering studies, the framework accounts for the heterogeneous, dynamic, and weakly standardized nature of AI ecosystems. It provides a structured process for planning, conducting, and documenting ML asset retrieval, enabling more transparent selection compared to ad hoc practices. Therefore, the framework ofers a foundation for more systematic ML asset discovery.

Future work will focus on further formalizing the proposed methodology into a more operational framework, including mapping its steps against existing literature to identify established, partially addressed, and implicit practices. We also aim to investigate how the framework can be empirically validated, including what constitutes evidence of efectiveness for ML asset retrieval, how consistently the approach can be applied across heterogeneous AI registries, and how it compares against ad hoc selection practices commonly used in empirical studies. At this stage, the approach is grounded in prior systematic review methodologies and empirical studies of AI registries, and should be interpreted as an empirically informed design rather than a fully validated method. Finally, we envision the construction of a knowledge graph of the AI ecosystem where models, datasets, and their relationships are explicitly represented. This would enable ecosystem-level modeling and make retrieval strategies such as snowballing more systematic and straightforward.

## 6 Data Availability

This paper presents a conceptual and methodological framework. No datasets, code, or other research artifacts were collected, generated or analyzed as part of this article.

## References

1 Adem Ait, Javier Luis Cánovas Izquierdo, and Jordi Cabot. Hfcommunity: A tool to analyze the hugging face hub community. In SANER, pages 728–732. IEEE, 2023.

2 Sebastian Baltes, Florian Angermeir, Chetan Arora, Marvin Muñoz Barón, Chunyang Chen, Lukas Böhme, Fabio Calefato, Neil Ernst, Davide Falessi, Brian Fitzgerald, et al. Guidelines for empirical studies in software engineering involving large language models. arXiv preprint arXiv:2508.15503, 2025.

3 Pearl Brereton, Barbara A Kitchenham, David Budgen, Mark Turner, and Mohamed Khalil. Lessons from applying the systematic literature review process within the software engineering domain. Journal of Systems and Software, 80(4):571–583, 2007.

4 Bruno Cartaxo, Gustavo Pinto, and Sergio Soares. Rapid reviews in software engineering. In Contemporary empirical methods in software engineering, pages 357–384. Springer, 2020.

5 Joel Castaño, Silverio Martínez-Fernández, and Xavier Franch. Lessons learned from mining the hugging face repository. In WSESE@ICSE, pages 1–6, 2024.

6 Joel Castaño, Silverio Martínez-Fernández, Xavier Franch, and Justus Bogner. Analyzing the evolution and maintenance of ml models on hugging face. In MSR, pages 607–618, 2024.

7 Ozren Dabic, Emad Aghajani, and Gabriele Bavota. Sampling projects in github for msr studies. In MSR, pages 560–564. IEEE, 2021.

8 Ioannis Dasoulas, Duo Yang, and Anastasia Dimou. MLSea: A semantic layer for discoverable machine learning. In European Semantic Web Conference, pages 178–198. Springer, 2024.

9 Vincenzo De Martino, Joel Castaño, Fabio Palomba, Xavier Franch, and Silverio Martínez-Fernández. A framework for using llms for repository mining studies in empirical software engineering. In WSESE@ICSE, pages 6–11. IEEE, 2025.

10 Claudio Di Sipio, Riccardo Rubei, Juri Di Rocco, Davide Di Ruscio, and Phuong T Nguyen. Automated categorization of pre-trained models in software engineering: A case study with a Hugging Face dataset. In EASE, pages 351–356, 2024.

11 Francisco Durán, Matias Martinez, Patricia Lago, and Silverio Martínez-Fernández. Insights into resource utilization of code small language models serving with runtime engines and execution providers. Journal of Systems and Software, page 112574, 2025.

12 Vahid Garousi, Michael Felderer, and Mika V Mäntylä. Guidelines for including grey literature and conducting multivocal literature reviews in software engineering. IST, 106:101–121, 2019.

13 Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. Datasheets for datasets. Communications of the ACM, 64(12):86–92, 2021.

14 Victoria Gomes, Delaney Selb, Fabio Palomba, Rodrigo Spinola, and David Lo. LLM-Assisted Empirical Software Engineering: Systematic Literature Review and Research Agenda. arXiv preprint arXiv:2604.26192, 2026.

15 Alexandra González, Joel Castaño, Xavier Franch, and Silverio Martínez-Fernández. Impact of ML optimization tactics on greener pre-trained ML models. Computing, 107(4), 2025.

16 Alexandra González, Xavier Franch, David Lo, and Silverio Martínez-Fernández. Cataloguing Hugging Face Models to Software Engineering Activities: Automation and Findings. arXiv preprint arXiv:2506.03013, 2025.

17 Wenxin Jiang, Mingyu Kim, Chingwo Cheung, Heesoo Kim, George K Thiruvathukal, and James C Davis. “i see models being a whole other thing”: an empirical study of pre-trained model naming conventions and a tool for enhancing naming consistency. Empirical Software Engineering, 30(6):155, 2025.

18 Wenxin Jiang, Jerin Yasmin, Jason Jones, Nicholas Synovic, Jiashen Kuo, Nathaniel Bielanski, Yuan Tian, George K. Thiruvathukal, and James C. Davis. PeaTMOSS: A Dataset and Initial Analysis of Pre-Trained Models in Open-Source Software. In MSR, page 431–443. ACM, 2024. doi:10.1145/3643991.3644907.

19 Barbara Kitchenham, Stuart Charters, et al. Guidelines for performing systematic literature reviews in software engineering. 2007.

20 Bernard Koch, Emily Denton, Alex Hanna, and Jacob G Foster. Reduced, reused and recycled: The life of a dataset in machine learning research. In NeurIPS Datasets and Benchmarks Track, 2021.

21 Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. Model cards for model reporting. In FAccT’19, pages 220–229, 2019.

22 Quim Motger, Carlota Catot, and Xavier Franch. Characterizing Datasets for LLM-based Requirements Engineering: A Systematic Mapping Study. arXiv preprint arXiv:2510.18787, 2025.

23 Sabato Nocera, Massimiliano Di Penta, Fatima Ahmed, Simone Romano, and Giuseppe Scanniello. What We Know about AIBOMs: Results from a Multivocal Literature Review on Artificial Intelligence Bill of Materials. ACM TOSEM, December 2026. doi:10.1145/3786773.

24 Ernesto Lang Oreamuno, Rina Mrinmoyee Khan, Abdul Ali Bangash, Catherine Stinson, and Bram Adams. The state of documentation practices of third-party machine learning models and datasets. IEEE Software, 41(5):52–59, 2024.

25 Matthew J Page, Joanne E McKenzie, Patrick M Bossuyt, Isabelle Boutron, Tammy C Hofmann, Cynthia D Mulrow, Larissa Shamseer, Jennifer M Tetzlaf, and David Moher. Updating guidance for reporting systematic reviews: development of the PRISMA 2020 statement. Journal of Clinical Epidemiology, 134:103–112, 2021.

26 Federica Pepe, Vittoria Nardone, Antonio Mastropaolo, Gabriele Bavota, Gerardo Canfora, and Massimiliano Di Penta. How do hugging face models document datasets, bias, and licenses? an empirical study. In IEEE/ACM ICPC, pages 370–381, 2024.

27 Kai Petersen, Robert Feldt, Shahid Mujtaba, and Michael Mattsson. Systematic mapping studies in software engineering. In EASE. BCS Learning & Development, 2008.

28 Julian Aron Prenner and Romain Robbes. Making the most of small software engineering datasets with modern machine learning. IEEE TSE, 48(12):5050–5067, 2021.

29 Miguel Romero-Arjona, Saman Barakat, Ana B Sánchez, and Sergio Segura. The Rise of Language Models in Mining Software Repositories: A Survey. arXiv preprint arXiv:2604.00787, 2026.

30 Trevor Stalnaker, Nathan Wintersgill, Oscar Chaparro, Laura A Heymann, Massimiliano Di Penta, Daniel M German, and Denys Poshyvanyk. An Empirical Analysis of Machine Learning Model and Dataset Documentation, Supply Chain, and Licensing Challenges on Hugging Face. ACM TOSEM, 2025.

31 Margaret-Anne Storey, Neil A Ernst, Courtney Williams, and Eirini Kalliamvakou. The who, what, how of software engineering research: a socio-technical framework. Empirical Software Engineering, 25(5):4097–4129, 2020.

32 Claes Wohlin. Guidelines for snowballing in systematic literature studies and a replication in software engineering. In EASE, pages 1–10, 2014.

33 Zhou Yang, Jieke Shi, Premkumar Devanbu, and David Lo. Ecosystem of large language models for code. ACM TOSEM, 35(1):1–30, 2025.

34 Zhimin Zhao, Yihao Chen, Abdul Ali Bangash, Bram Adams, and Ahmed E Hassan. An empirical study of challenges in machine learning asset management. Empirical Software Engineering, 29(4):98, 2024.

35 Da-Wei Zhou and Han-Jia Ye. A Unifying Perspective on Model Reuse: From Small to Large Pre-Trained Models. In IJCAI, pages 10826–10835, 2025.