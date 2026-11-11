# Chain of Physics: Supporting Fault Diagnosis with Engineering-Inspired Large Language Model Reasoning

Wenqi Zhou<sup>a</sup>, Leonardo Baldo<sup>a,∗</sup>, Kai Goebel<sup>b,c</sup> and Manuel Arias Chao<sup>a,d</sup>

<sup>a</sup>Institutefor Data Science, Zurich University ofApplied Science, Winterthur, 8401, Switzerland

<sup>b</sup>Fragum Global, Mountain View, CA 94040, USA

<sup>c</sup>Luleå University ofTechnology, Luleå, 971 87, Sweden

<sup>d</sup>Faculty of Aerospace Engineering, Delft University of Technology, HS 2926 Delft, The Netherlands

## A R T I C L E I N F O

Keywords: Fault diagnosis Diagnostic reasoning Large Language Models Knowledge Graphs Engineering knowledge In-context learning

## A BS T R AC T

Fault diagnosis in complex systems requires linking observed changes in physical quantities to the failure mechanisms that produce them. Experienced engineers perform this task by combining sensor information with knowledge of system behavior and fault propagation. Large Language Models (LLMs) can support this process, but their diagnostic reasoning depends on the availability of asset-specific engineering knowledge and on how this knowledge is represented and used during inference. This work introduces Chain of Physics (CoP), an LLM-based fault-diagnosis framework built around a Constrained Knowledge Graph (CKG) generated automatically from asset-specific reference documentation. CoP addresses failure– sensor association from textual engineering knowledge. The CKG stores diagnostic chains connecting failure modes, changes in physical quantities, and monitoring sensors. During inference, semantically similar graph paths are retrieved for each question and used as both information and in-context examples for the reasoning process. Candidate failure modes are extracted from the question and analysed independently through dedicated LLM calls, which generate plausible failure-to-observation chains. A final decision module compares these chains with the available answer choices. CoP is evaluated on the FailureSensorIQ benchmark across four industrial asset classes and three LLMs. Compared with direct prompting and conventional Retrieval-Augmented Generation (RAG), CoP improves diagnostic accuracy across all evaluated models, with gains ranging from 7.8 to 19.6 percentage points and from 11.7 to 18.2 percentage points, respectively. The results show that structured engineering knowledge can improve LLMbased diagnostic reasoning when graph retrieval is combined with in-context examples of fault propagation.

## 1. Introduction

Fault diagnosis in complex systems requires identifying the failure mechanisms responsible for an observed deviation from nominal behavior. Experienced engineers and technicians perform this task by combining available measurements with their knowledge of system behavior and reasoning about the physical quantities through which failure efects become observable [1]. These relationships link observed sensor behavior to specific failure mechanisms, enabling Fault Detection and Isolation (FDI) and supplying diagnostic information for downstream health assessment and maintenance decisions in Prognostics and Health Management (PHM) frameworks [2]. In industrial applications, the knowledge required for this process is generally distributed across technical manuals, reference documents, troubleshooting procedures, maintenance records, and accumulated expert experience [3]. Its interpretation and integration therefore remain largely dependent on experienced personnel, who must retrieve and reconcile information scattered across heterogeneous technical documents and information systems [4, 5]. The problem considered in this work focuses specifically on diagnosis from engineering knowledge, not on the direct classification of measured sensor signals. Given a textual description of a failure mode or sensor observation, the objective is to recover the relevant failure–sensor association and the physical relationships that connect them. The task can therefore be viewed as knowledge association from engineering text, requiring the relationships among failure mechanisms, their physical efects, and the corresponding sensing modalities to be established from textual information.

The increasing interest in Large Language Models (LLMs) introduces the possibility of assisting engineers in this knowledge-intensive task [6]. Through natural-language interaction, LLMs can process technical information, interpret diagnostic questions, and formulate possible explanations without requiring a dedicated model for each asset [5]. Despite the strong push towards Artificial Intelligence (AI) adoption and the proliferation of AI initiatives across industrial sectors, the integration of LLMs into fault diagnosis remains challenging. Conventional taskspecific models generally require representative labelled data and retraining when new assets or failure conditions are introduced, limiting their applicability in industrial settings. LLMs ofer greater flexibility, but often lack the assetspecific and practical engineering knowledge required to interpret abnormal conditions correctly. This limitation can lead to hallucinated or physically inconsistent diagnostic explanations, particularly when the model must establish relationships between failure mechanisms, their efects on system behavior, and the available sensor measurements [7]. Their efective deployment therefore requires incorporating established engineering knowledge and guiding the reasoning process according to the physical relationships governing the system.

![](images/a034c15382f67efba6010cfc6e4b36a7a2126456183d1f60b1f5b36cc98b5efb.jpg)  
Figure 1: Overview of CoP. Once per asset, constrained extraction turns reference documentation into a CKG whose paths are complete failure-to-observation chains. For each diagnostic question, retrieved CKG paths provide both asset-specific knowledge and reasoning examples. These guide failure-specific reasoning, which produces new chains for the candidate failure modes. A separate answer-selection stage makes the decision based on those chains.

One of the most widely adopted approaches to incorporate external engineering knowledge into LLM inference is Retrieval-Augmented Generation (RAG) [8]. By retrieving relevant passages from technical documentation, RAG can provide information beyond that acquired during pre-training without requiring task-specific model training. Nevertheless, the availability of relevant information does not guarantee that the model can reconstruct the physical sequence linking a fault to its measurable efects. However, in fault diagnosis, these relationships are central to the reasoning process: a failure afects one or more physical quantities, its efects may propagate through the system, and the resulting changes are detected by specific sensors. Representing this sequence explicitly may therefore provide a more suitable basis for diagnostic reasoning than supplying unstructured technical information alone. This type of explicit failure-to-observation reasoning has a long tradition in engineering diagnosis, including qualitative physics and model-based diagnosis [9], Signed Directed Graph (SDG) representations [10], and Failure Mode and Symptoms Analysis (FMSA) [11]. These approaches similarly rely on structured relationships between failure mechanisms, their physical manifestations, and the observations through which they can be detected. In recent years, Knowledge Graphs (KGs) have emerged as a possible means of representing this type of diagnostic structure through explicit entities and relationships [12]. For fault diagnosis, a KG can represent the progression from a failure mode to the physical changes and the measurements through which those changes become observable. Even more, the close integration of LLMs and KGs is increasingly being investigated for industrial fault diagnosis [13]. In fact, LLMs provide natural language processing and reasoning capabilities, while KGs organize domain knowledge into explicit entities and relationships. Their integration allows diagnostic systems to retrieve and use structured engineering knowledge during inference, supporting more informed and interpretable diagnostic reasoning. Still, the practical integration of LLMs and KGs into an industrial diagnostic framework presents two main dificulties [14]. First, manually constructing asset-specific knowledge representations requires considerable engineering efort, particularly when multiple systems and failure mechanisms are considered. Second, a graph containing relevant entities and generic relationships does not necessarily preserve the physical sequence required for diagnostic reasoning.

Against this backdrop, this work introduces Chain of Physics (CoP) as an engineering-inspired framework for LLMassisted fault diagnosis based on the CKG, aimed to tackle these exact two challenges. The methodology follows the way an experienced engineer approaches a diagnostic problem: plausible failure mechanisms are considered, their physical consequences are traced through the system, and the resulting efects are connected to the available measurements. This places CoP within the broader line of work that translates human reasoning procedures into explicit intermediate steps that can be reproduced algorithmically during LLM inference [15, 16]. CoP organizes both knowledge extraction and diagnostic reasoning around the same failure-to-observation structure. Figure 1 summarizes this logic, with the ofline construction of reusable asset knowledge on the left and the question-specific reasoning process on the right. During graph construction, asset-specific reference documentation is processed automatically and candidate chains are retained only when the source material supports a failure mode, at least one change in a physical quantity, and a monitoring sensor. These elements are represented through a fixed set of relations describing the initial failure efect, its propagation through the system, and its observation by a sensor. The resulting CKG is therefore a diagnostic representation of the source knowledge in which each complete path corresponds to a physically structured failure-to-observation chain. Since the graph is generated once for each asset, the extracted knowledge can be reused across subsequent diagnostic questions. During inference, relevant CKG paths are retrieved for the question under analysis and used in two ways. They provide asset-specific information about failure mechanisms and measurable efects, and they are supplied to the LLM as in-context examples of the reasoning structure to follow. Candidate failure modes extracted from the question are then analysed independently, allowing the model to generate new failure-to-observation chains that are not restricted to paths already contained in the graph. A separate decision stage gives the answer based on the generated chains. CoP therefore combines graph-grounded retrieval with in-context reasoning over a constrained engineering representation, while maintaining a clear separation between failure-specific reasoning and final answer selection.

The proposed approach is evaluated using FailureSensorIQ, an open-source benchmark addressing the relationships between industrial failure modes and monitoring sensors. The assessment considers two complementary tasks: identifying relevant sensors for a given failure mode and identifying possible failure modes from sensor evidence. Experiments are conducted using three LLMs on four asset classes: an aero gas turbine, an electric motor, a fan, and a power transformer. CoP is compared with direct LLM prompting and conventional RAG based on the same reference documentation. An ablation study further examines the contribution of question clarification and the separate diagnostic-reasoning and decision stages. The results show that CoP improves diagnostic accuracy across all evaluated models, supporting the use of explicitly structured engineering knowledge to assist fault diagnosis without task-specific model training. In summary, the main contributions of this work are:

• An automatic procedure for constructing an asset-specific CKG from technical reference documentation. The graph is restricted to complete diagnostic chains linking failure modes, changes in physical quantities, and monitoring sensors through a fixed set of relations.

• A diagnostic inference framework that uses retrieved CKG paths as both engineering knowledge and in-contex reasoning exemplars. These paths guide the generation of failure-specific chains without requiring the target failure mode to be explicitly represented in the retrieved graph paths.

• A staged reasoning procedure that separates question clarification, failure-mode extraction, graph retrieval, failure-specific reasoning, and final answer selection, allowing the physical reasoning process to be performed independently from the multiple-choice decision.

• An evaluation on the FailureSensorIQ benchmark across four industrial asset classes and three LLMs, including comparisons with direct prompting and conventional RAG and an ablation analysis of the main inference components.

The remainder of the paper is organised as follows. Section 2 reviews related work on LLM-based fault diagnosis, KGs, RAG, and prompt-based reasoning. Section 3 presents the proposed Chain of Physics framework, including the ofline construction of the CKG and the online diagnostic inference procedure. Section 4 introduces the FailureSensorIQ benchmark and the experimental setting. Section 5 presents and discusses the results. Finally, Section 6 summarises the main findings and outlines directions for future work.

## 2. Related Works

The proposed framework lies at the intersection of three research directions: (i) the use of external knowledge during LLM inference, (ii) the representation of domain knowledge through graph-based structures, and (iii) the use of in-context examples to guide reasoning. These areas have developed largely in parallel, but their combination is particularly relevant for fault diagnosis, where both asset-specific knowledge and the structure of the reasoning process matter. The following paragraph briefly introduces representative developments from the broader Machine Learning (ML) literature that are relevant to the methodological building blocks considered in this work, including RAG, graphbased knowledge access, and In-Context Learning (ICL). After this broader methodological overview, the second part of the section reviews works more directly connected to PHM and fault diagnosis. The purpose of the next paragraph is then not to provide a comprehensive review of current ML research, but to establish the technical background and state of the art needed to position the proposed framework. For a broader overview of the interaction between LLMs and structured knowledge, the interested reader is referred to Pan et al. [14], who provide a comprehensive review of KG-enhanced LLMs, LLM-augmented KGs, and complete LLM–KG frameworks.

A central limitation of LLMs in knowledge-intensive applications is their dependence on information acquired during pre-training. This has motivated the development of methods that provide models with external knowledge at inference time. RAG, introduced by Lewis et al. [17], is one of the most established approaches in this direction. Conventional RAG retrieves relevant passages from an external document collection and supplies them to the model as additional context for question answering or text generation. Beyond unstructured text retrieval, subsequent work has investigated how external knowledge can be represented and accessed in a more structured form. Jiang et al. [18] proposed StructGPT, which enables LLMs to interact with structured data through dedicated interfaces. Graph-based approaches then extended this idea by exploiting explicit entities, relations, and paths during retrieval and reasoning. Wang et al. [19], for example, introduced Knowledge Graph Prompting, where graph construction and traversal are used to retrieve supporting evidence from multiple documents. Luo et al. [20] developed Reasoning on Graphs (RoG), in which relation paths are used to guide the retrieval of graph-grounded evidence. More recently, Ma et al. [21] proposed Think-on-Graph 2.0, which combines structured graph retrieval with unstructured textual information. These approaches show how graph structure can be used to organize external knowledge and make relational information directly available during LLM inference. A related line of research concerns ICL, where examples included in the input prompt guide the model in performing a task without modifying its parameters. The choice of these examples can substantially afect model behavior, motivating retrieval-based approaches in which demonstrations are selected dynamically for each input [22, 23]. This principle has also been applied to graph-structured knowledge. Wei et al. [24], for instance, proposed KICGPT, where structural knowledge retrieved from a KG is encoded into demonstrations that guide the LLM during knowledge-graph completion. In this setting, retrieval determines both the information supplied to the model and the examples used to shape its reasoning process.

In this sense, the use of LLMs in PHM has received increasing attention, with recent studies investigating their use across diferent stages of the PHM process. In their paper, Li et al. [35] reviewed the development of largescale foundation models for PHM, outlining their potential applications, methodological requirements, and remaining challenges. Chen et al. [36] reviewed large models for machine monitoring and fault diagnostics, while Tsallis et al. [37] provided a systematic review of large language and foundation models for machinery health monitoring. Razaq et al. [5] conducted a systematic literature review of 37 studies combining LLMs and KGs for fault diagnosis, identifying two dominant integration paradigms (i.e., automated KG construction and KG-grounded (GraphRAG-style) reasoning) and reporting that current work concentrates almost exclusively on rotating machinery, steel manufacturing, and Computer Numerical Control (CNC) equipment, highlighting limited multilingual coverage and persistent challenges concerning the explainability of diagnostic reasoning. Other contributions have investigated broader frameworks intended to support multiple PHM tasks. Cha et al. [38], for instance, proposed an LLM-based autonomous agent that orchestrates data preprocessing, model selection, training, and optimization for fault classification and Remaining Useful Life (RUL) prediction. Tao et al. [39] introduced the concept of a PHM Large Model (PHM-LM), discussing its underlying paradigms and the technical challenges associated with its development and application.

Alongside these broader frameworks, a growing body of research addresses individual PHM functions, including fault detection and isolation, prognostics, and maintenance decision support. Fault diagnosis is particularly relevant for knowledge-based LLM applications, as the interpretation of abnormal conditions often requires combining sensor evidence with information contained in technical manuals, troubleshooting procedures, and maintenance records. Such information is frequently expressed in natural language, making it accessible to LLM-based reasoning and questionanswering approaches.

Table 1  
Representative studies grouped by methodological similarity and their relationship to CoP.
<table><tr><td>Group</td><td>Representative studies</td><td>Key characteristic</td><td>Distinction from CoP</td></tr><tr><td>Reasoning procedure without persistent KG</td><td>Gong et al. [16]</td><td>Structured prompting and agent-based architectures, including multi-agent debate and planning.</td><td>No reusable, offline-built engineering-knowledge representation.</td></tr><tr><td>KG-based symptom /solution retrieval</td><td>Lin and Miao [25]; Nie et al. [26]; Liu et al. [27]</td><td>Graph retrieval links symptoms and faults to causes and maintenance actions.</td><td>Do not explicitly impose CoP&#x27;s constraining schema.</td></tr><tr><td>Graph-grounded retrieval and reasoning</td><td>Gu et al. [28]; Xie et al. [29]; Miao et al. [30]</td><td>Retrieved causal paths or subgraphs provide structured, query-specific context to support subsequent LLM reasoning.</td><td>Evaluated in individual application domains, without a common cross-asset benchmark or not reproducible.</td></tr><tr><td>Causal / mechanism-explicit KG</td><td>Guo et al. [31]</td><td>Weighted causal graph of fault-propagation paths derived from monitoring data.</td><td>Constructed from operational measurements instead of reference technical documentation.</td></tr><tr><td>Automatic knowledge construction from documentation</td><td>Zhou et al. [32]; Wang et al. [33]; Marandi et al. [34]</td><td>Automatic extraction of structured knowledge from technical documents. Marandi et al. also reuse the resulting representation for graph traversal and LLM-based reasoning.</td><td>Use different ontologies or functional representations, without CoP&#x27;s predefined constraining schema</td></tr><tr><td>Comparable evaluation design</td><td>Ma et al. [7] (FDRKG-LLM)</td><td>Comparison of a KG-grounded LLM with direct prompting and multiple conventional RAG baselines.</td><td>Single equipment class, with open-ended generation metrics instead of multi-asset multiple-choice accuracy.</td></tr></table>

Before considering recent LLM-based approaches to this problem, it is worth recalling that explicit representations of physical behaviour have a long history in engineering fault diagnosis. Classical work on qualitative physics represented system behaviour through qualitative changes in physical quantities and causal relationships, providing a basis for reasoning about how physical efects propagate through engineered systems [9]. Closely related model-based diagnostic approaches infer possible faults by comparing manifested system behaviour with the behaviour predicted from explicit models of component structure and function. In process diagnosis, causal and SDG representations have similarly been used to describe qualitative relationships among process variables and support the propagation of fault efects [10]. FMSA provides a further connection from the condition-monitoring domain, relating identified failure modes to their symptoms and to the monitoring techniques suitable for their detection [11].

The use of LLMs for fault diagnosis and diagnostic reasoning is therefore the research direction most closely related to the present work and requires a deeper look. The main axis along which diferent approaches can be distinguished is the way engineering knowledge is made available to the model and subsequently used: (i) some methods rely primarily on knowledge acquired during pre-training, (ii) others retrieve relevant passages from technical documentation, (iii) while a third group organizes domain information into structured representations such as KGs.

Approaches based mainly on pretrained model knowledge use prompting, reasoning strategies, or agent-based decomposition without introducing an external persistent representation of asset-specific engineering knowledge [16]. As already mentioned, their main limitation is that diagnostic performance depends on information already encoded in the model parameters, which may be incomplete, too generic, or inconsistent with the the physics of the specific asset under analysis. This becomes particularly critical when diagnosis requires detailed relationships between failure mechanisms, local changes in physical quantities, and the sensors through which these efects become observable. A second group augments the LLM with unstructured technical information retrieved from manuals, maintenance documents, or troubleshooting material. Retrieval can provide relevant asset-specific evidence and reduce the dependence on pre-training knowledge. However, the retrieved passages remain expressed in natural language and the model must reconstruct the diagnostic relationships contained in them during inference. Relevant evidence may therefore be available in the context without explicitly preserving the sequence through which a failure produces physical efects and eventually becomes observable at a sensor. Lukens et al. [8], for example, investigated the use of technical manuals within a RAG framework for fault diagnosis, providing a direct example of document-based knowledge augmentation in PHM. Finally, structured approaches address this limitation by representing engineering knowledge explicitly through entities and relations. In these methods, knowledge extraction and representation become part of the diagnostic framework, allowing information about components, failure mechanisms, symptoms, and monitoring variables to be reused during inference. However, the usefulness of the resulting graph depends strongly on the adopted representation. Generic entity–relation graphs can store relevant domain information without necessarily preserving the physical progression required to reason from a failure mechanism to its measurable efects. A further refinement of this third category is provided by constrained knowledge representations, in which the admissible relations between entities are defined a priori according to the semantics of the target problem. In this case, the graph structure itself restricts how knowledge can be represented and connected, reducing arbitrary relation types and enforcing paths that follow a predefined engineering logic. For diagnostic applications, such constraints can be used to preserve the progression from a failure mechanism, through changes in physical quantities, to the sensor that observes their efects.

The next paragraph focuses on applications of this last paradigm, which is particularly relevant and promising for fault-diagnosis reasoning. Among the applications combining LLMs and structured diagnostic knowledge, Liu et al. [40] proposed a joint KG–LLM model for fault localization in aviation assembly. The method incorporates KG embeddings through prefix-tuning and uses knowledge retrieval to support the identification of fault causes and troubleshooting solutions. Ma et al. [7] introduced FDRKG-LLM, combining a fault diagnostic reasoning KG with an LLM to support equipment diagnosis and maintenance decisions. Their evaluation compares the proposed framework with direct prompting and conventional RAG configurations on a self-constructed diagnostic database, reporting improvements in text-generation metrics. Liu et al. [27] and Nie et al. [26] investigated related approaches for CNC equipment, combining a domain KG, retrieval, and engineering information to support interactive fault diagnosis. Guo et al. [31] combined constrained causal discovery with LLM-enhanced knowledge-graph completion for rotating machinery, deriving a weighted causal directed acyclic graph that traces fault-propagation paths from monitoring data. A comparably large empirical gain was reported by Lin and Miao [25], who improved automotive fault-diagnosis accuracy using a dual-mode knowledge-graph and LLM retrieval scheme. These studies establish the use of structured engineering knowledge in LLM-assisted industrial diagnosis, with diferent approaches to knowledge integration and model adaptation.

Since the integration of LLMs with structured engineering knowledge appears especially promising for scalable and reproducible fault diagnosis, the next question is: how are diagnostic knowledge representations created? For instance, Liao et al. [41] investigated LLM-assisted extraction of fine-grained information from robotic maintenance texts for KG construction, and comparable automatic pipelines have been proposed for automotive [32] and aviation [42] components. Wang et al. [33] proposed GraphRAG-KM, an automated framework that converts industrial technical documentation into ontology and conceptual models through document parsing, graph-based retrieval, LLM extraction, and clustering. Marandi et al. [34] proposed a framework for nuclear power plant diagnostics in which a hierarchical functional model is automatically constructed from technical documentation and encoded into a KG. The resulting representation supports both structured graph traversal and LLM-based interpretive reasoning through Graph-RAG. These contributions address the preparation and reuse of engineering knowledge, including its integration into diagnostic inference.

Other approaches investigate how retrieved knowledge can be used to structure the reasoning process itself. Gong et al. [16] proposed a multi-agent framework that generates diagnostic reasoning blueprints and adapts the subsequent decision process according to question complexity. Evaluated on FailureSensorIQ [43], the framework improves Llama-3.1-8B accuracy from 36.5% to 54.6%. Gu et al. [28] combined domain knowledge and data-driven information to construct a unified causal graph for fault diagnosis in distributed computing systems, supplying retrieved paths as in-context reasoning exemplars. Xie et al. [29] and Miao et al. [30] investigated related graph-grounded reasoning approaches for sensor networks and train bogies, respectively. Gao et al. [44] proposed LLM-FDQA, a KG-enhanced framework for industrial fault diagnosis that combines semantic extraction, two-stage entity linking, and multi-hop causal reasoning. The method retrieves and validates symptom–cause–solution chains to support diagnostic question answering. Evaluated on a proprietary CNC fault-diagnosis dataset, it reports an end-to-end diagnostic accuracy of 94.04%. These studies show how graph-derived evidence can be used both as retrieved knowledge and as structure for subsequent diagnostic reasoning.

Reproducibility remains a practical limitation when these approaches are compared. Several studies rely on proprietary datasets or provide insuficient implementation detail for a controlled reimplementation of the complete reasoning pipeline. The paper from Gong et al. [16] is among the few works evaluated on the public FailureSensorIQ benchmark whose reasoning architecture is described in suficient detail to reproduce. This makes their framework particularly suitable for the additional controlled comparison reported in Section 5.7.

The present work combines these research directions within a single diagnostic framework. From the perspective of classical diagnostic methods discussed above, the CKG follows the same general principle of representing explicitly the relationships between a failure and its observable manifestations, while adopting a structure tailored to LLM-based reasoning. Its paths contain failure modes, directional changes in physical quantities, and monitoring sensors connected through a constrained sequence of relations. Moreover, the methodology is shaped by the way an engineer approaches a diagnostic problem, by following the physical relationships that connect a failure mechanism to the changes it produces in the system and to the measurements through which those changes become observable. CoP automatically constructs an asset-specific CKG from reference documentation, retaining chains that connect failure modes, changes in physical quantities, and monitoring sensors. During inference, query-conditioned graph retrieval recovers complete CKG paths that provide both asset-specific engineering knowledge and in-context exemplars of failure-to-observation reasoning. Candidate failure modes are extracted independently from the diagnostic question and analysed through separate LLM calls using the retrieved chains as reasoning examples. The resulting failure-specific chains are then passed to a separate Judge module that selects the final answer. CoP therefore combines graph-grounded retrieval with retrieval-conditioned in-context reasoning and a distinct final decision stage.

## 3. Methodology

Evaluating diagnostic reasoning is not straightforward. In realistic applications, the available evidence, operating conditions, and level of diagnostic detail can vary substantially, making direct comparison between reasoning approaches dificult. A common way of reducing this variability is to formulate the task as question answering, where the information required to reach the answer is controlled and the final response can be evaluated against a known target. In Knowledge Graph Question Answering (KGQA), this setting is commonly distinguished between simple questions, whose answer can be obtained from a single relation, and more complex multi-hop questions, which require combining several entities and relations before reaching the target answer [45, 46]. This formulation provides a practical way of evaluating whether a model can recover isolated associations or compose a sequence of intermediate relations and it is one of the standard methodologies to test and compare reasoning algorithm in ML also thanks to the various published ML datasets.

A similar setting is commonly used for diagnostic reasoning. A failure-mode-to-sensor question may be answered through a direct association when the relevant relation is explicit, or it may require reconstructing a sequence in which a failure afects one or more physical quantities before its efect becomes observable at a sensor. FailureSensorIQ [43] is, to the best of the authors’ knowledge, one of the few and the most comprehensive publicly available benchmarks designed specifically to evaluate this type of reasoning across industrial assets. The dataset formulates diagnosis as multiple-choice question answering and tests both directions of the failure-mode–sensor relationship: identifying the sensors relevant to a given failure mode and inferring plausible failure modes from abnormal sensor observations. Accordingly, the task considered here difers from conventional fault classification based on measured sensor data. FailureSensorIQ evaluates diagnostic association from textual information. Given a failure mode, the model must identify the sensing modality most relevant to its detection, or, in the inverse direction, determine which failure modes are consistent with a stated sensor observation. An example of a Failure Mode to Sensor (FM2Sensor) question is reported below, where the model receives a diagnostic stem and a fixed set of candidate sensor classes and must identify the right (gold) response

## Example Diagnostic Question

Stem. In an aero gas turbine, when air inlet blockage occurs, which sensor from the choices is most critical in detecting the occurrence of the failure event?

<table><tr><td>A</td><td>Gas generator temperature</td></tr><tr><td>B</td><td>Power turbine temperature</td></tr><tr><td>C</td><td>Oil debris</td></tr><tr><td>D</td><td>Compressor temperature Gold</td></tr><tr><td>E</td><td>Pressure / pressure ratio</td></tr></table>

The example also illustrates a practical characteristic of the benchmark: diagnostic questions are not always expressed with fully standardised or technically precise wording. Since FailureSensorIQ is generated through a template-based procedure, this should not be interpreted as reproducing real industrial conversations. However, redundant formulations, broad sensor names, and ambiguous expressions are present in the benchmark and resemble the linguistic imprecision that can occur in technical diagnostic requests. This makes the dataset suitable for evaluating a pipeline that must first establish a clear interpretation of the question before reasoning about the underlying failure.

CoP organizes diagnostic reasoning around the physical consequences expected from each candidate failure mechanism. The procedure follows the way an engineer approaches diagnosis by tracing the changes that a possible cause would produce in the system and identifying the measurements through which those changes become observable. For failure-mode-to-sensor questions, the reasoning proceeds from a specified failure to its observable efects. For sensor-to-failure-mode questions, candidate causes are assessed through their expected efects and their relevance to the stated observations. The method comprises an ofline stage, performed once for each asset, in which reference documentation is converted into the CKG, and an online stage, executed for each diagnostic question, in which retrieved chains support failure-specific reasoning and subsequent answer selection. Figure 2 visualizes the modularity: the upper part constructs the reusable asset-specific CKG, while the lower part shows how that representation is used for each diagnostic question.

All LLM calls in CoP use dedicated instruction templates designed following established prompt-design principles: the instructions define the task of each module, the context made available to it, the permitted operations, and the required output structure as indicated in [47, 48]. Complex diagnostic reasoning is decomposed into separate calls, retrieved failure-to-observation chains are supplied as in-context demonstrations where appropriate [15], and structured outputs are used to constrain the information exchanged between stages. The complete prompts for all calls are reported in Appendix A.

## 3.1. Ofline step: CKG generation from Reference Knowledge

The ofline stage is performed once for each asset and converts the corresponding reference documentation into the CKG used during online inference. As illustrated in Figure 2, the procedure comprises diferent operations: document parsing and segmentation, extraction of grounded failure-to-observation chains, and aggregation of the accepted chains into the asset-specific CKG.

## 3.1.1. Document parsing and segmentation

The reference documentation is first converted from PDF to Markdown using Docling [49]. The parsing pipeline preserves the document structure and enables the extraction of tables and mathematical expressions, while Optical Character Recognition (OCR) can be enabled when required by the source document. The resulting Markdown representation is then divided into text units of approximately 2,000 characters, which are processed independently in the following stage. This size has been selected following the commonly recommended starting point for fixed-size chunking [50] and lies within the range shown to benefit retrieval on technical, context-dependent documents [51].

This preprocessing step provides a common textual representation of the source material while keeping the subsequent LLM calls local to relatively short portions of the document. Each unit is treated independently, so the generation of a diagnostic chain is based only on the information contained in that portion of the reference text.

## 3.1.2. Grounded diagnostic-chain extraction

Each text unit is passed to a dedicated LLM call using a structured output schema designed specifically for the diagnostic representation adopted in CoP. The extraction prompt is reported in Appendix A.1. For each unit, the model is asked to extract one coherent monitoring scenario composed of a failure mode, an ordered sequence of changes in physical quantities, and the sensor that observes the final quantity. Each intermediate step contains the physical

![](images/b87b43c602dd232c1873a7a1b619cccf2b09b3dc9516b489aa992437bcc6c0ea.jpg)  
Figure 2: Complete CoP methodology. The framework consists of an ofline stage, performed once for each asset, and an online stage executed for each diagnostic question. Ofline, the asset-specific reference documentation is divided into chunks and processed through a grounding gate. The accepted chains are converted into triples using the fixed relations results in, leads to, and is monitored by, and subsequently merged into the CKG. Online, the clarified question is used independently to retrieve and expand relevant CKG paths and to extract the failure modes under consideration. The retrieved paths guide the generation of new diagnostic chains for each failure mode, after which the Judge maps the resulting reasoning to the available answer choices.

variable, its location on the asset when available, and the direction of change, restricted to increases or decreases.   
Between one and five physical-variable changes are allowed in each chain to avoid excessively long chains.

The extraction prompt acts as a grounding gate. A chain is retained only when the text unit supports the complete relationship between a failure mode, at least one physical-variable change, and a monitoring sensor within the same scenario. General theoretical descriptions, historical information, design equations, or passages without an explicit failure-to-monitoring relationship are therefore discarded. The model is also instructed to use only information contained in the current text unit, without introducing additional physical propagation steps. When several diagnostic scenarios are present, the dominant one is selected.

The structured LLM output is subsequently checked before graph generation. A candidate chain is rejected if the failure mode or sensor is missing, if no physical-variable step is available, or if any step does not contain a physical variable. Repeated node descriptions within the same chain are also rejected, since their direct conversion into the graph representation could introduce an artificial cycle. This second validation therefore complements the constraints imposed through the extraction prompt. For an accepted result, the resulting sequence has the form

$$
{ \mathrm { f a i l u r e ~ m o d e } } \to { \mathrm { p h y s i c a l ~ c h a n g e } } _ { 1 } \to \cdots \to { \mathrm { p h y s i c a l ~ c h a n g e } } _ { n } \to { \mathrm { s e n s o r , w h e r e ~ } } n \in [ 1 , 5 ] .
$$

Throughout the chain, each physical-change node combines the variable, its location when reported in the source, and its direction of change

## 3.1.3. Graph construction and aggregation

Each accepted chain is converted directly into a small graph using the three relations defined by the CKG schema. The transition from the failure mode to the first physical efect is assigned the relation results in; intermediate propagation steps are connected through leads to; and the final physical efect is connected to the corresponding sensor through is monitored by. The graph structure is therefore generated directly from the ordered chain returned by the extraction stage. A separate graph is produced for every accepted text unit. Once all units associated with an asset have been processed, the resulting graphs are aggregated by taking the union of their entities, edge types, and relations using KGGen [52]. The outcome is a single asset-specific CKG containing the failure-to-observation chains supported by the reference documentation.

This construction procedure imposes the same reasoning structure across all retained knowledge. Each complete path starts from a failure mechanism, follows the physical quantities through which its efects propagate, and terminates at the sensor through which the final efect becomes observable. The intuitive structure of the CKG allows experts to cross check and edit the information easily. The CKG can consequently be reused during online inference both as a condensed representation of asset-specific engineering knowledge and as a repository of complete diagnostic reasoning examples.

The extracted chains used in the evaluation were reviewed by a Subject-Matter Expert (SME) and corrected where required before online inference. The constrained representation makes this review direct: the initiating condition, every physical-variable change, its direction, and the monitoring sensor are exposed as an ordered path whose individual steps can be checked against engineering knowledge. The resulting knowledge bases remain compact enough for complete expert inspection. Section 5.1 and Figure 4 show an excerpt of the resulting CKG and illustrate how the stored chains can be followed and verified relation by relation.

## 3.2. Online step: Inference on a Specific Question

The online stage, reported in the lower half of Figure 2, is executed independently for each diagnostic question using the CKG previously constructed for the corresponding asset. Its purpose is to combine the information contained in the question with the reasoning patterns stored in the graph, while keeping knowledge retrieval, failure-mode identification, physical reasoning, and answer selection as separate operations.

## 3.2.1. Question clarification

The first operation is a clarification step performed by a dedicated LLM call using the prompt reported in Appendix A.2. The rationale follows a common pattern in human diagnostic reasoning: when a problem statement is unclear or ambiguous, an engineer first reformulates it to obtain a more precise interpretation before analysing the underlying failure mechanisms. The model receives the original question together with all candidate answers and determines whether the wording is suficiently clear for the subsequent reasoning stages. If no ambiguity is identified, the original question and options are retained unchanged. Otherwise, the model reformulates the question and, where required, the answer options while preserving their number, order, meaning, asset, and diagnostic intent. The clarification module is explicitly instructed not to select an answer or introduce failure modes or sensors that are not implied by the original item. If the structured output cannot be parsed, or if the number of returned options difers from the original one, the pipeline falls back to the unmodified question.

The resulting question and options are used by all subsequent modules. The clarification stage therefore acts only on the interpretation of the diagnostic problem and does not access the CKG. Figure 5 later shows a question that is left unchanged, while Appendix B reports a complementary worked example in which ambiguity is detected and the question is reformulated before inference.

## 3.2.2. CKG retrieval andfailure-mode extraction

After clarification, the pipeline separates into two parallel operations.

The first retrieves relevant reasoning examples from the asset-specific CKG. A retrieval query is formed by concatenating the clarified question and all answer options. This query is embedded and compared through cosine similarity with cached embeddings of the failure-mode and sensor entities contained in the CKG. The top-� highestscoring entities are retained as retrieval seeds, where $k = 2$ in the experiments reported in this work. The selected seeds are then expanded along the graph to recover complete failure-to-sensor chains. This operation corresponds to graph traversal, i.e., following the stored relations from a selected node to reconstruct a complete path. When the retrieved seed is a failure mode, the algorithm follows the relations forward until a monitoring sensor is reached. When the seed is a sensor, the relations are followed backward until the corresponding failure mode is reached. The path expansion follows the constrained graph structure and recovers complete chains of the general form

$$
\mathrm { f a i l u r e ~ m o d e }  \mathrm { p h y s i c a l ~ e f f e c t }  \cdots  \mathrm { s e n s o r } ,
$$

using the relations results in, leads $t o ,$ and is monitored by. Duplicate paths are removed before the retrieved chains are supplied to the reasoning stage.

In parallel, a second LLM call extracts the failure modes mentioned or implied by the clarified question and its answer options using the prompt reported in Appendix A.3. This operation is independent of the CKG. The extracted failure modes are not required to correspond to nodes already present in the graph. The result is a list of candidate failure modes defining what must subsequently be analysed.

This separation assigns diferent roles to the two branches to define what to reason for and how to reason. The question determines the failure modes under consideration, while the CKG provides asset-specific examples of how failure efects propagate toward measurable quantities. The target failure mode therefore does not need to be explicitly represented in the retrieved graph paths.

## 3.2.3. Retrieval-conditioned in-context reasoning

The reasoning stage follows an ICL formulation. As already mentioned, in ICL, examples placed in the input context guide the model in performing the target task without updating its parameters [22]. Retrieval-based variants select these demonstrations dynamically according to the current input, allowing the context presented to the model to change from one query to another [23]. Related graph-based approaches have used structured KG knowledge as in-context demonstrations to guide LLM inference [24].

CoP adopts retrieval-conditioned few-shot prompting at the level of diagnostic reasoning. Complete CKG chains are dynamically retrieved for each question and supplied as demonstrations of the failure-to-observation structure to follow. The complete CKG chains retrieved in the previous step are inserted into the prompt as examples of the expected failure-to-observation reasoning structure. They therefore provide both asset-specific engineering context and demonstrations of the sequence "failure mode → physical quantity changes → sensor".

A separate LLM call is performed for every extracted failure mode using the prompt reported in Appendix A.4. This separation mirrors the engineering reasoning process adopted throughout CoP. Each candidate failure is examined through its expected physical consequences before the final diagnostic comparison. The model receives the asset name, the target failure mode, the retrieved CKG paths, and the chain format associated with the graph. Its task is to trace plausible changes in physical quantities and identify the sensors capable of observing them. The reasoning module does not receive the multiple-choice question or its answer options directly; the assessment of the generated chains against the diagnostic question is assigned to the subsequent Judge module. Using the retrieved chains as in-context reasoning exemplars, together with its domain knowledge, the LLM generates one or more plausible chains linking the target failure mode to intermediate physical efects and ultimately to sensors capable of observing those efects. The generated chains are not required to correspond to paths already stored in the CKG. The structured output of this stage is therefore a set of newly inferred diagnostic chains for each extracted failure mode.

## 3.2.4. Final judgement

Once all failure modes have been analysed, their inferred chains are passed to a separate LLM-based decision module, hereafter referred to as the Judge, together with the clarified question and the complete set of answer options. The Judge prompt is reported in Appendix A.5. The Judge then selects a single option identifier based on the generated diagnostic reasoning. The resulting inference procedure separates retrieval of stored graph knowledge, generation of failure-specific diagnostic chains, and matching of those chains to the available answer choices. This separation assigns failure-specific analysis and final answer selection to distinct stages. The reasoning module receives candidate failure modes and retrieved exemplars, while the Judge receives the diagnostic question and answer choices to assess the relevance of the generated consequences. The answer choices can therefore influence reasoning indirectly through retrieval and failure-mode extraction.

## 4. Dataset

The proposed approach is evaluated on FailureSensorIQ [43], one of the most complete and one of the few publicly available benchmarks specifically designed to assess the ability of LLMs to establish relationships between failure modes and monitoring sensors across diferent industrial assets. Developed by IBM Research, the dataset comprises 8,296 multiple-choice questions covering ten asset classes. The questions are generated through a templatebased procedure using condition-monitoring knowledge from International Organization for Standardization (ISO) documentation, including ISO 17359. The benchmark addresses two complementary diagnostic tasks: FM2Sensor, which identifies the sensors relevant to a given failure mode, and Sensor-to-Failure-Mode (Sensor2FailureMode), which identifies possible failure modes associated with a sensor observation. Both tasks include positive and negative formulations, requiring the model to identify either the most relevant or the irrelevant option among the available answers.

Industrial assets considered in the evaluation, reference documentation used for CKG construction, and number of singlecorrect FailureSensorIQ questions [43].
<table><tr><td>Asset</td><td>Reference documentation</td><td>Questions</td></tr><tr><td>Aero gas turbine</td><td>Gas Turbine Theory, 7th ed. [53]</td><td>336</td></tr><tr><td>Electric motor</td><td>Noninvasive Methods for Condition Monitoring and Electrical Fault Diagnosis of Induction Motors [54]</td><td>234</td></tr><tr><td>Fan</td><td>Improving Fan System Performance: A Sourcebook for Industry [55]</td><td>200</td></tr><tr><td>Power transformer</td><td>Transformers: Basics, Maintenance, and Diagnostics [56]</td><td>544</td></tr><tr><td>Total</td><td></td><td>1,314</td></tr></table>

FailureSensorIQ provides 2,667 Single-Correct Multiple-Choice Question Answering (SC-MCQA) items and 5,629 Multiple-Correct Multiple-Choice Question Answering (MC-MCQA) items. Consistently with the primary evaluation setting adopted in the original benchmark, this study considers only the single-correct subset. The assessment is further restricted to four asset classes for which suitable reference documentation was available to construct the corresponding CKGs: aero gas turbine, electric motor, fan, and power transformer. The resulting evaluation set comprises 1,314 questions. Table 2 reports the selected assets, the reference documents used for graph construction, and the number of questions considered for each asset.

An additional consideration worth mentioning concerning the interpretation of accuracy on FailureSensorIQ is raised in the original dataset study by Constantinides et al. [43]. The benchmark is generated from failure-mode– sensor associations contained in ISO documentation and expressed through predefined question templates. These engineering relationships can be many-to-many: a failure mechanism may afect several measurable quantities and may therefore be observable through diferent sensing modalities. The single-correct formulation nevertheless requires one designated answer for each item, and expressions such as "most relevant sensor" consequently impose a ranking among relationships that may all be technically plausible to diferent degrees. Some sensor descriptions are also broad or technically underspecified, as illustrated by the "length measurement" example discussed in Appendix B.

This characteristic should be considered when interpreting the absolute accuracy values reported in the following section. The original FailureSensorIQ study evaluated five human participants on a balanced subset of 80 questions and reported a mean expert accuracy of approximately 60%, with the best-performing expert reaching 66.19%. Inter-rater agreement among the experts was moderate $( \kappa = 0 . 4 6 2 )$ [43]. These results do not establish a formal upper bound for the benchmark, since the human evaluation covers only a subset of the questions and expertise difers across asset classes. They do, however, indicate that disagreement and ambiguity are present even among human domain experts. Accordingly, no matter the methodology, the gap to 100 per cent accuracy should not be interpreted solely as model error, since a portion of the residual error may reflect ambiguity in the benchmark formulation and the non-unique nature of some failure–sensor relationships.

## 5. Results

The evaluation was conducted on the 1,314 single-correct FailureSensorIQ questions retained for the four assets considered in this work. The section evaluates the proposed solution from the construction of the knowledge representation to the final diagnostic decision. Section 5.1 first examines the CKGs produced ofline and their inspectability. Section 5.2 then follows a worked reasoning example. Sections 5.3–5.5 assess diagnostic accuracy, the contribution of the inference stages, and consistency across assets. Section 5.7 removes the CKG entirely to isolate the contribution of the failure-to-observation reasoning structure. Together, these analyses assess the quality of the constructed knowledge, the benchmark performance of CoP, the role of its individual stages, and the contribution of its reasoning representation.

## 5.1. CKG construction

Table 3 summarises the outcome of the ofline graph-construction procedure. The number of accepted chains varies substantially across the four reference documents. From 981 text chunks extracted from the aero gas turbine textbook,

Table 3  
Outcome of the ofline CKG construction for the four evaluated assets.
<table><tr><td>Asset</td><td>Chunks</td><td>Chains</td><td>Pass rate</td><td>Entities</td><td>Relations</td></tr><tr><td>Aero gas turbine</td><td>981</td><td>55</td><td>5.6%</td><td>209</td><td>164</td></tr><tr><td>Electric motor</td><td>38</td><td>14</td><td>36.8%</td><td>44</td><td>35</td></tr><tr><td>Fan</td><td>152</td><td>25</td><td>16.4%</td><td>99</td><td>79</td></tr><tr><td>Power transformer</td><td>333</td><td>135</td><td>40.5%</td><td>429</td><td>348</td></tr></table>

![](images/ecd89f0061d869a30c660ce72ccca935cf60d763b491a4396ccafa1ca87f5639.jpg)  
Figure 3  
Structure of the four CKGs. Left: distribution of chain lengths, measured as the number of relations between the failure mode and the sensor. Right: number of failure-mode, intermediate physical-change, and sensor entities in each merged graph.

55 complete diagnostic chains were retained, corresponding to a pass rate of 5.6%. The fan source produced 25 chains from 152 chunks (16.4%), while the electric-motor and power-transformer documents produced 14 from 38 chunks (36.8%) and 135 from 333 chunks (40.5%), respectively.

Figure 3 describes the structure of the resulting graphs. Most chains contain two or three relations, corresponding to one or two intermediate changes in physical quantities between the failure mode and the sensor. Longer chains, with up to five relations, are mainly extracted from the fan and aero gas turbine documents, whereas the electric-motor graph contains no chain longer than four relations. For all assets, intermediate physical-change entities form the largest group of nodes.

The extraction rates in Table 3 reflect the selective purpose of the CKG. Reference documents contain extensive theoretical, historical, and design material, while the extraction procedure retains passages supporting complete failureto-observation chains. The resulting graphs contain between 14 and 135 accepted chains per asset, providing a manageable body of diagnostic knowledge for independent SME review. This compactness is a practical advantage of the representation: each retained example exposes the physical relationships used to connect an initiating condition to a measurable efect.

Figure 4 illustrates a section of the aero gas turbine CKG. The stored chains can contain diferent numbers of intermediate physical changes, reflecting the amount of propagation supported by the source material. The fixed relations distinguish the initial efect, subsequent propagation, and sensor observation. This structure makes every chain readable from the initiating failure to the monitored quantity and allows an engineer to check its correctness step by step. The figure therefore also illustrates the practical basis for the SME review described in Section 3.1.3.

During inference, retrieved chains serve as engineering knowledge and few-shot demonstrations of diagnostic reasoning. Their role is to provide complete examples of how physical efects connect an initiating condition to observable quantities. CoP therefore uses a selective set of reviewed diagnostic exemplars to guide the analysis of candidate failure modes, including those absent from the retrieved paths.

## 5.2. Analysis of the reasoning process

The illustrative example in Figure 5 illustrates one case of how the CoP inference procedure operates in practice. A second example reporting the complete end-to-end output of every online stage, including retrieval seeds, expanded CKG chains, inferred chains, and Judge reasoning, is provided in Appendix C. The question concerns a mounting fault in a fan and asks which of five candidate measurements has the highest diagnostic relevance. The clarification module identifies the item as suficiently clear and leaves the question unchanged. The failure-mode extraction stage identifies mountingfault infan as the condition to be analysed.

![](images/a9f7bf56b0a4b557bc074b040ff7be95535517c7a13a17ace2446fb609b96c3d.jpg)  
Figure 4: Excerpt from the aero gas turbine CKG. Orange nodes denote initiating conditions, blue nodes denote physicalvariable changes, and green nodes denote monitoring sensors. The relations results in, leads to, and is monitored by distinguish initial efects, physical propagation, and observation. The explicit chain structure supports step-by-step inspection by domain experts.

In parallel, the retrieval stage searches the fan CKG using the clarified question and answer options. The two recovered chains concern contaminant buildup on fan blades and fouling of the inlet-guide-vane linkage. Their physica paths terminate at a power meter and a pressure transducer, respectively. Neither retrieved chain contains the mounting fault appearing in the question or directly points to vibration, which is the correct answer.

The reasoning stage then uses these stored chains as examples of the expected failure-to-observation structure and applies the same reasoning pattern to the extracted mounting fault. Among the generated chains, the model links mounting faults to loose mounting bolts, shaft misalignment, and transmitted imbalance forces. Two of these chains terminate in structural or lateral vibration measurements, while another terminates in motor-current or power measurement. The Judge subsequently compares the complete set of inferred chains with the available answer choices and selects vibration, corresponding to the gold label.

This example is useful because the retrieved CKG paths provide limited topical overlap with the target failure mode. Their contribution lies in supplying an asset-specific representation of how a failure is connected to physical efects and observable quantities. The new mounting-fault chains are generated during the reasoning stage and are therefore distinct from the paths stored in the CKG. Figure 5 consequently shows the intended interaction between graph retrieval and LLM inference: the graph provides structured diagnostic examples, while the model extends the same structure to the failure mode under analysis.

The trace should be interpreted as an explanatory example of the inference process, while the quantitative results reported in the previous sections provide the evidence for the observed performance improvements. The clarification and decision stages further improve overall accuracy. The results are based on one run for each model with zero temperature.

## 5.3. Diagnostic accuracy

The overall accuracy obtained by the three approaches is reported in Table 4. CoP achieves the highest accuracy for all three models. With Claude Opus 4.7, accuracy increases from 51.98% with direct prompting and 50.61% with RAG to 68.80% with CoP. The same trend is observed for the smaller models: Llama 3.1 8B increases from 30.75%

![](images/4aab17941aaccb4d6e3b6723d7a2e4b40a751c26e7280d384912dc6755202019.jpg)  
Figure 5: Example of the CoP inference process for a fan mounting fault. The input question is first assessed by the clarification module and is left unchanged because no ambiguity is detected. The target failure mode is extracted directly from the clarified question, while the CKG retrieval stage independently recovers complete stored chains associated with the most relevant graph entities. These paths provide examples of the expected failure–physical efect–sensor reasoning structure. The reasoning stage then generates new plausible chains for the mounting fault, including paths leading to structural and lateral vibration measurements. The Judge receives the question, answer choices, and all inferred chains, and selects vibration as the final answer.

## Table 4

Overall diagnostic accuracy on the 1,314 single-correct FailureSensorIQ questions.
<table><tr><td>Model</td><td>Direct</td><td>RAG</td><td>CoP</td></tr><tr><td>Claude Opus 4.7</td><td>51.98</td><td>50.61</td><td>68.80</td></tr><tr><td>Llama 3.1 8B</td><td>30.75</td><td>26.86</td><td>38.51</td></tr><tr><td>Qwen 2.5 7B</td><td>30.52</td><td>35.24</td><td>50.15</td></tr></table>

to 38.51%, while Qwen 2.5 7B increases from 30.52% to 50.15%. Overall, the gain over direct prompting ranges from 7.8 to 19.6 percentage points, while the gain over conventional RAG ranges from 11.7 to 18.2 percentage points. The absolute values should be interpreted in light of the benchmark characteristics discussed in Section 4.

The comparison with RAG is particularly relevant because both approaches rely on the same reference documentation and the same initial chunking procedure. The diference therefore lies in the representation and use of the retrieved information. RAG provides the model with semantically similar text passages, while CoP retrieves complete failure-to-observation chains from the CKG and uses them as reasoning examples. The results indicate that the structure imposed on the engineering knowledge has a substantial influence on the final diagnostic performance. This efect is visible in all three models and is especially pronounced for Qwen 2.5 7B, for which CoP improves accuracy by almost 15 percentage points over RAG.

The RAG results also show that the availability of additional textual context does not translate consistently into higher accuracy. This result is aligned with the conclusion in [43], where a RAG pipeline is implemented. Claude Opus 4.7 and Llama 3.1 8B both obtain slightly lower performance with RAG than with direct prompting, while Qwen

Table 5 Ablation study of the CoP inference procedure.
<table><tr><td>Model</td><td>w/o clarify + Judge w/o Judge</td><td></td><td>Full CoP</td></tr><tr><td>Claude Opus 4.7</td><td>66.36</td><td>67.81</td><td>68.80</td></tr><tr><td>Llama 3.1 8B</td><td>35.16</td><td>36.61</td><td>38.51</td></tr><tr><td>Qwen 2.5 7B</td><td>43.53</td><td>48.55</td><td>50.15</td></tr></table>

![](images/ea490b2d849f9901ae5cc4ac350d7bdd0cb0b484b1b24f0075a1e6106dd85de1.jpg)  
Figure 6  
Accuracy gain contributed by each inference stage, starting from direct prompting. Each point adds one component to the previous configuration. Labels report the gain in percentage points.

2.5 7B benefits from retrieved context. This variability suggests that the contribution of retrieval depends on how efectively the model can reconstruct the relevant diagnostic relationship from the retrieved passages. CoP reduces this burden by retrieving information already organised according to the failure–physical efect–sensor structure used during reasoning.

## 5.4. Contribution of the inference stages

The ablation study in Table 5 examines how the additional inference stages afect the performance obtained from the CKG. All configurations use the same CKG, the same embedding-based seed selection (� = 2), and the same path expansion. The configurations difer in the LLM calls surrounding retrieval:

• CoP w/o clarify + Judge. The original question and options are used directly for retrieval. No clarification call, failure-mode extraction call, failure-specific reasoning call, or separate Judge is used. One LLM call receives the original question, the answer options, the retrieved CKG chains, and the chain pattern, and jointly produces the diagnostic reasoning and the selected option.

• CoP w/o Judge. The clarification call is added before retrieval. The clarified question and options are then passed to the same single answer-generation call used in the previous configuration. Failure-mode extraction and separate failure-specific reasoning are still absent, and reasoning and option selection remain within one LLM call. The comparison with the previous configuration therefore isolates the efect of clarification.

• Full CoP. The clarification stage is retained, and the single answer-generation call is replaced by separate failure-mode extraction, one reasoning call for each extracted failure mode, and the final Judge call described in Section 3.2. The reasoning calls do not receive the answer choices directly. The comparison with CoP w/o Judge therefore evaluates the combined efect of failure-mode extraction, separate failure-specific reasoning, and final answer selection through the Judge.

The configuration combining CKG retrieval with a single answer-generation call already reaches 66.36%, 35.16%, and 43.53% for the three models, respectively. These values account for a large part of the improvement observed with the complete method and support the role of the constrained graph representation itself. Clarification adds 1.45 percentage points for Claude Opus 4.7 and Llama 3.1 8B, and 5.02 points for Qwen 2.5 7B. The full staged inference procedure adds a further 0.99, 1.90, and 1.60 percentage points, respectively. Figure 6 shows the same results as a cumulative sequence starting from direct prompting. For all three models, the largest single increase is obtained when CKG retrieval is introduced, while clarification and the Judge provide smaller additional gains.

Diagnostic accuracy (%) by asset class, model, and inference configuration. CKG denotes graph retrieval with direct answer generation; CoP denotes the complete inference procedure, including clarification, separate failure-specific reasoning, and final judgement. The number of evaluated questions is reported once for each asset. Bold values indicate the highest accuracy within each row.
<table><tr><td>Asset (N)</td><td>Model</td><td>Direct</td><td>RAG</td><td>CKG</td><td>CoP</td></tr><tr><td>Aero gas turbine (336)</td><td>Claude Opus 4.7</td><td>43.45</td><td>44.05</td><td>59.82</td><td>61.01</td></tr><tr><td rowspan="4">Electric motor (234)</td><td>Llama 3.1 8B</td><td>29.17</td><td>30.06</td><td>34.52</td><td>38.99</td></tr><tr><td>Qwen 2.5 7B</td><td>27.68</td><td>31.85</td><td>37.80</td><td>49.40</td></tr><tr><td>Claude Opus 4.7</td><td>66.67</td><td>54.70</td><td>66.67</td><td>75.21</td></tr><tr><td>Llama 3.1 8B</td><td>36.32</td><td>24.36</td><td>40.60</td><td>40.60</td></tr><tr><td>Fan (200)</td><td>Qwen 2.5 7B</td><td>29.91</td><td>35.47</td><td>44.44</td><td>50.43</td></tr><tr><td rowspan="4"></td><td>Claude Opus 4.7</td><td>51.50</td><td>51.00</td><td>63.00</td><td>65.00</td></tr><tr><td>Llama 3.1 8B</td><td>37.50</td><td>27.00</td><td>43.50</td><td>45.50</td></tr><tr><td>Qwen 2.5 7B</td><td>38.00</td><td>46.50</td><td>53.00</td><td>60.00</td></tr><tr><td>Claude Opus 4.7</td><td>51.10</td><td>52.76</td><td>71.51</td><td>72.24</td></tr><tr><td>Power transformer (544)</td><td>Llama 3.1 8B</td><td>26.84</td><td>25.92</td><td>30.15</td><td>34.74</td></tr><tr><td></td><td>Qwen 2.5 7B</td><td>29.78</td><td>33.09</td><td>43.01</td><td>46.88</td></tr></table>

![](images/ff68b06d077c9abbbca0dd3859067dcc600fd2d08af0523114b4999b12addfc5.jpg)  
Diagnostic accuracy (%) per asset for direct prompting, RAG, and CoP with the three evaluated LLMs.

## 5.5. Accuracy across asset classes

Table 6 reports diagnostic accuracy separately for each asset class, including the intermediate inference configurations. Full CoP outperforms direct prompting and conventional RAG in all twelve asset–model combinations. The gains range from 4.28 to 21.14 percentage points over direct prompting and from 8.93 to 20.51 percentage points over RAG. The aggregate improvement is therefore observed across all four evaluated asset classes. Figure 7 reports the same comparison for each asset. With Claude Opus 4.7 and Qwen 2.5 7B, CoP obtains the highest accuracy on all four assets. With Llama 3.1 8B, the margins are smaller. CoP is most clearly ahead on the electric motor and the fan, while the three approaches remain close on the aero gas turbine and the power transformer.

![](images/7a618ee572b5a28371742c4bffd265099b35eb795873f0af0137809f880d3758.jpg)  
Figure 8: Accuracy (%) by question type for direct prompting, RAG, and the three CoP configurations. (FM→S: failure mode to sensor; S→FM: sensor to failure mode; �: number of questions of each type.)

The intermediate stages show greater variation. For Llama 3.1 8B, clarification reduces accuracy from 34.52% to 32.44% on the aero gas turbine and from 40.60% to 40.17% on the electric motor. Adding separate failure-specific reasoning and finaljudgement reduces fan accuracy from 46.50% to 45.50% for Llama, and power-transformer accuracy from 72.98% to 72.24% for Claude and from 47.98% to 46.88% for Qwen. It leaves Qwen’s fan accuracy unchanged at 60.00%. These results qualify the aggregate ablation: the additional stages improve overall accuracy, but their benefit depends on the asset and underlying model.

These results also indicate that the contribution of the individual stages depends on the underlying model. Question clarification has a limited efect for Claude Opus 4.7 and Llama 3.1 8B, while its efect is more marked for Qwen 2.5 7B. The Judge provides a smaller increase in aggregate accuracy across all three models, although the asset-level results show exceptions.

## 5.6. Accuracy by question type

FailureSensorIQ contains four question types, obtained by combining the reasoning direction (from failure mode to sensor, FM→S, or from sensor to failure mode, S→FM) with the polarity of the question (selecting the relevant or the irrelevant option). Figure 8 reports the accuracy of all configurations for each type

With Claude Opus 4.7, the largest improvements occur for questions that require selecting the relevant option, where CoP increases accuracy by approximately 20 to 30 percentage points in both directions, and for S→FM questions with negative polarity. FM→S questions with negative polarity are already answered comparatively well by direct prompting and show a smaller gain. With Qwen 2.5 7B, CoP improves accuracy for all four question types. With Llama 3.1 8B, CoP improves accuracy for the two positive-polarity types and for S→FM questions with negative polarity, but not for FM→S questions with negative polarity, where direct prompting remains slightly better. S→FM questions with negative polarity are the most dificult type for this model across all approaches. Overall, the improvement obtained with CoP is not limited to a single question formulation, although its magnitude depends on both the question type and the model.

## 5.7. Reasoning structure without the CKG

The previous experiments evaluate CoP with both the constrained failure-to-observation reasoning structure and the asset-specific knowledge stored in the CKG. To examine the contribution of the reasoning structure independently of graph retrieval, we also evaluate a variant without CKG support using the multi-agent architecture proposed by Gong et al. [16]. This study was selected because it is one of the few approaches evaluated on FailureSensorIQ for which the agent roles, routing logic, and overall decision procedure are described in suficient detail to support an independent reimplementation.

The architecture described by Gong et al. comprises a Guide Agent, a Discriminator Agent, a Simple Answerer Agent, one Option Advocate Agent for each answer choice, and a Judge Agent. The Guide produces a reasoning blueprint, while the Discriminator classifies each question as SIMPLE or COMPLEX. SIMPLE questions are routed directly to the Simple Answerer. COMPLEX questions are processed by the Option Advocates, after which the Judge selects the final answer from the generated arguments.

Accuracy (%) on the 2,667 single-correct FailureSensorIQ questions without graph retrieval. Gong et al. denotes the reimplemented multi-agent framework of [16] with all agents instantiated using the same model.
<table><tr><td>Model</td><td>Method</td><td>AGT</td><td>CMP</td><td>EG</td><td>EM</td><td>FAN</td><td>IGT</td><td>PT</td><td>PMP</td><td>RICE</td><td>ST</td><td>Overall</td></tr><tr><td rowspan="2">Llama 3.1 8B</td><td>Gong et al.</td><td>41.96</td><td>40.45</td><td>41.88</td><td>44.44</td><td>46.00</td><td>52.50</td><td>36.95</td><td>51.97</td><td>47.92</td><td>38.01</td><td>43.34</td></tr><tr><td>CoP w/o CKG</td><td>41.67</td><td>45.91</td><td>45.30</td><td>46.58</td><td>50.50</td><td>52.08</td><td>38.79</td><td>51.97</td><td>48.81</td><td>39.77</td><td>45.14</td></tr><tr><td rowspan="2">Qwen 2.5 7B</td><td>Gong et al.</td><td>41.67</td><td>47.73</td><td>52.99</td><td>53.85</td><td>54.00</td><td>55.00</td><td>43.38</td><td>52.63</td><td>56.85</td><td>43.27</td><td>49.34</td></tr><tr><td>CoP w/o CKG</td><td>45.54</td><td>51.82</td><td>58.55</td><td>50.00</td><td>57.50</td><td>57.50</td><td>50.74</td><td>53.95</td><td>62.20</td><td>46.78</td><td>53.28</td></tr><tr><td>Questions</td><td></td><td>336</td><td>220</td><td>234</td><td>234</td><td>200</td><td>240</td><td>544</td><td>152</td><td>336</td><td>171</td><td>2,667</td></tr></table>

Since the original study does not provide a public code repository or the complete prompts used for the individual agents, the framework was reconstructed from the methodological description, Algorithm 1, and the agent responsibilities reported in the paper. The resulting implementation follows the published agent structure and SIMPLE/COMPLEX routing procedure.

Within this reimplementation, the comparison is designed to isolate the form of the reasoning exchanged between the Option Advocates and the Judge. In the Gong configuration, each advocate produces a free-text argument supporting its assigned answer option. In the CoP w/o CKG configuration, the same advocate is required to express its argument as a diagnostic chain of the form

failure mode → physical change → ⋯ → sensor.

The chain is enforced through structured output and then passed to the Judge in place of the free-text argument. The remaining architecture and routing procedure are kept identical within our reimplementation. No CKG, graph retrieval, or reference documentation is used in either configuration. The diagnostic chains are therefore generated entirely from the model’s internal knowledge.

Since this experiment does not require asset-specific graph construction, it is conducted on all ten asset classes and on the complete set of 2,667 single-correct FailureSensorIQ questions. Both configurations are evaluated with Llama 3.1 8B and Qwen 2.5 7B. In the original Gong et al. study, the Guide, Discriminator, and Judge use GPT-4o-mini while the evaluated open-source model is used for the remaining roles. In our comparison, all agents are instantiated with the same model. This removes the contribution of a stronger orchestration model and allows the efect of the reasoning representation to be assessed under a homogeneous-model setting.

The results are reported in Table 7. For Llama 3.1 8B, replacing free-text advocacy with the constrained failureto-observation chain increases overall accuracy from 43.34% to 45.14%. For Qwen 2.5 7B, accuracy increases from 49.34% to 53.28%. Across the individual asset classes, CoP w/o CKG improves performance in 16 of the 20 asset– model combinations, matches the Gong configuration in one case, and performs lower in three.

The improvement is observed without any contribution from graph-derived knowledge or retrieved engineering documentation. The diference between the two configurations lies in the representation of the intermediate diagnostic argument presented to the Judge. Constraining this argument to an explicit failure-to-observation chain therefore improves the final decision even when the chain is generated entirely from the model’s internal knowledge.

This result is relevant for the interpretation of the complete CoP framework. The CKG contributes asset-specific engineering knowledge and retrieved examples of diagnostic propagation, while the chain representation determines how the intermediate reasoning is expressed. The experiment in this subsection shows that the latter remains beneficial when the CKG is removed. The gains obtained with the full framework can therefore be associated with two distinct elements: the engineering knowledge supplied by the graph and the explicit physical structure imposed on the reasoning process.

## 6. Conclusions and Future Work

LLMs ofer a practical means of supporting knowledge-intensive engineering tasks, provided that the information and reasoning structure required by the task are made explicit. This study introduced CoP for industrial fault diagnosis, translating the way an engineer traces a failure through its physical consequences into a constrained failure-toobservation representation. Asset-specific reference documentation is converted ofline into a CKG, and complete graph paths are retrieved during inference as engineering knowledge and reasoning examples for the generation of failure-specific diagnostic chains. CoP addresses a complementary diagnostic problem to conventional data-driven fault classification: reasoning over engineering knowledge to establish the relationships between failure mechanisms, their physical efects, and the sensing modalities through which those efects can be observed.

On 1,314 single-correct FailureSensorIQ questions, CoP improved accuracy across all three evaluated LLMs, with gains of 7.8–19.6 percentage points over direct prompting and 11.7–18.2 points over conventional RAG. The ablation analysis showed that the CKG provides a substantial part of this improvement, while the additional inference stages provide further gains. The comparison without graph retrieval provides a second result: imposing the same failureto-observation structure within the reproducible framework of Gong et al. improves overall accuracy for both tested models. This supports the use of physically ordered diagnostic chains as a reasoning representation independently of the asset-specific knowledge supplied by the CKG.

Future work will extend the evaluation to real industrial datasets combining sensor measurements, maintenance records, and naturally occurring diagnostic requests. Further work should also test alternative reasoning architectures and richer graph constraints, including component topology, operating-state dependencies, and admissible propagation paths, and develop confidence or abstention mechanisms for cases in which the available engineering knowledge does not support a suficiently reliable diagnostic conclusion.

## CRediT Authorship Contribution Statement

Wenqi Zhou: Conceptualization, Methodology, Software, Validation, Investigation, Visualization, Writing - Original Draft, Writing - Review & Editing Leonardo Baldo: Methodology, Visualization, Writing - Original Draft, Writing - Review & Editing Kai Goebel: Conceptualization, Methodology, Writing - Review & Editing Manuel Arias Chao: Conceptualization, Methodology, Validation, Writing - Review & Editing, Supervision, Funding acquisition

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Code and Data Availability Statement

Data sharing is not applicable to this article as no new data were created or generated. The FailureSensorIQ dataset employed in this study is publicly available [43]. The code will be made available upon request.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used ChatGPT and Claude to assist with language editing, image editing, improving readability, and refining the presentation of the manuscript. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] International Organization for Standardization. ISO 17359:2018 – condition monitoring and diagnostics of machines – general guidelines. Technical Report ISO 17359:2018, International Organization for Standardization, 2018.

[2] Andrew K. S. Jardine, Daming Lin, and Dragan Banjevic. A review on machinery diagnostics and prognostics implementing condition-based maintenance. Mechanical Systems and Signal Processing, 20(7):1483–1510, 2006.

[3] Rui Kang, Junxiang Zhu, Stephen Green, and Ioannis Brilakis. Knowledge graphs for operational decision-making in industrial maintenance: A systematic review. Expert Systems with Applications, 299:130172, 2026.

[4] Yanying Wang, Ying Cheng, Qinglin Qi, and Fei Tao. IDS-KG: An industrial dataspace-based knowledge graph construction approach for smart maintenance. Journal ofIndustrial Information Integration, 38:100566, 2024.

[5] Wan Barmawi Muhammad Razaq, Hao Chen, Marcos R. Machado, and João Luiz Rebelo Moreira. How can the integration of ai large language models and knowledge graph enhance fault diagnosis? a systematic literature review. Applied Soft Computing, 194:114908, 2026.

[6] Huihui Han, Jian Wang, and Xiaowen Wang. Enhancing reliability of steel production lines through advanced knowledge graph-based fault diagnosis model. IEEE Transactions on Reliability, 74(2):2880–2892, 2025.

[7] Yunfei Ma, Shuai Zheng, Zheng Yang, Hongcheng Pan, and Jun Hong. A knowledge-graph enhanced large language model-based fault diagnostic reasoning and maintenance decision support pipeline towards industry 5.0. International Journal of Production Research, 64(12):5239–5260, 2026.

[8] Sarah Lukens, Matthew Bishof, Nadir Siddiqui, and Destiny West. An evaluation framework for fault diagnosis using technical manuals in retrieval-augmented large language models. In Annual Conference ofthe PHM Society, volume 17, 2025.

[9] Johan de Kleer and Brian C. Williams. Diagnosing multiple faults. Artificial Intelligence, 32(1):97–130, 1987.

[10] Mark A. Kramer and Jr. Palowitch, B. L. A rule-based approach to fault diagnosis using the signed directed graph. AIChE Journal, 33(7):1067– 1078, 1987.

[11] International Organization for Standardization. Condition monitoring and diagnostics of machines—data interpretation and diagnostics techniques—part 1: General guidelines, 2012.

[12] Nourhan Ibrahim, Samar Aboulela, Ahmed Ibrahim, and Rasha Kashef. A survey on augmenting knowledge graphs (KGs) with large language models (LLMs): models, evaluation metrics, benchmarks, and challenges. Discover Artificial Intelligence, 4(1):76, 2024.

[13] Jorge Martinez-Gil, Georg Buchgeher, David Gabauer, Bernhard Freudenthaler, Dominik Filipiak, and Anna Fensel. Root cause analysis in the industrial domain using knowledge graphs: A case study on power transformers. Procedia Computer Science, 200:944–953, 2022. 3rd International Conference on Industry 4.0 and Smart Manufacturing

[14] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering, 36(7):3580–3599, 2024.

[15] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc., 2022.

[16] Wei Gong, Shuang Qiao, Chenhong Cao, Shilei Tan, Junliang Ye, Haoxiang Liu, Si Chen, and Xuesong Wang. Harnessing collective intelligence of multi-agent LLM systems for sensor failure reasoning in smart manufacturing. Journal ofIndustrial Information Integration, 49:101012.2026

[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.

[18] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. StructGPT: A general framework for large language model to reason over structured data. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 9237–9251, Singapore, December 2023. Association for Computational Linguistics.

[19] Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. Knowledge graph prompting for multi-document question answering. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(17):19206–19214, 2024.

[20] Linhao Luo, Yuan-Fang Li, Reza Hafari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. In International Conference on Learning Representations, 2024.

[21] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. Think-on-graph 2.0: Deep and faithful large language model reasoning with knowledge-guided retrieval augmented generation. In International Conference on Learning Representations, 2025.

[22] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655– 2671, Seattle, United States, 2022. Association for Computational Linguistics.

[23] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for GPT-3? In Proceedings ofDeep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integrationfor Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online, 2022. Association for Computational Linguistics.

[24] Yanbin Wei, Qiushi Huang, Yu Zhang, and James T. Kwok. KICGPT: Large language model with knowledge in context for knowledge graph completion. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 8667–8683, Singapore, 2023. Association for Computational Linguistics.

[25] Weikun Lin and Kehua Miao. An automotive fault diagnosis framework based on knowledge graphs and large language models. Electronics, 14(21), 2025.

[26] Qingwei Nie, Junsai Geng, Dunbing Tang, and Changchun Liu. Industrial knowledge-enhanced fault diagnosis method: Integrating llm and knowledge graph for fault reasoning and maintenance recommendation in cnc machine tools. Computers & Industrial Engineering, 214:111879, 2026.

[27] Yuhan Liu, Yuan Zhou, Yufei Liu, Zhen Xu, and Yixin He. Intelligent fault diagnosis for CNC through the integration of large language models and domain knowledge graphs. Engineering, 53:311–322, 2025.

[28] Yu Gu, Jian Zhang, and Yugen Du. Causal graph-enhanced large language models for automated fault diagnosis and intelligent operation and maintenance in distributed computing systems. Electronics, 15(11):2359, 2026.

[29] Xin Xie, Junbo Wang, Yu Han, and Wenjuan Li. Knowledge graph-based in-context learning for advanced fault diagnosis in sensor networks. Sensors, 24(24), 2024.

[30] Runsheng Miao, Tao Wu, and Zhenyu Zhang. Graph rag-based fault diagnosis for train bogies using knowledge graphs and large language model. Knowledge-Based Systems, 331:114855, 2026.

[31] Zhen Guo, Zhiping Liu, Edgar Estupiñan, and Diego Cabrera. Fault inference diagnosis based on causal network and large language model enhanced knowledge graph network. Chinese Journal ofMechanical Engineering, 39:100162, 2026.

[32] Bin Zhou, Xingwang Shen, Runze Miao, and Jinsong Bao. Llm-driven knowledge graph construction for automotive parts fault diagnosis. Digital Engineering, page 100137, 2026.

[33] Duyun Wang, Peilin Han, Shmuel Tyszberowicz, Mingyue Zhang, and Bo Liu. GraphRAG-KM: An automated framework for transforming industrial documents into ontology and conceptual models. In Knowledge Science, Engineering and Management, volume 15920 of Lecture Notes in Computer Science, pages 108–120. Springer Nature Singapore, 2026. KSEM 2025, Proceedings, Part II.

[34] Saman Marandi, Yu-Shu Hu, and Mohammad Modarres. Complex system diagnostics using a knowledge graph-informed and large language model-enhanced framework. Applied Sciences, 15(17):9428, 2025.

[35] Yan-Fu Li, Huan Wang, and Muxia Sun. ChatGPT-like large-scale foundation models for prognostics and health management: A survey and roadmaps. Reliability Engineering & System Safety, 243:109850, 2024.

[36] Xuefeng Chen, Yaguo Lei, Yanfu Li, Simon Parkinson, Xiang Li, Jinxin Liu, Fan Lu, Huan Wang, Zisheng Wang, Bin Yang, Shilong Ye, and Zhibin Zhao. Large models for machine monitoring and fault diagnostics: Opportunities, challenges, and future direction. Journal of Dynamics, Monitoring and Diagnostics, 4(2):76–90, 2025.

[37] Christos Tsallis, Panagiotis Papageorgas, Radu Adrian Munteanu, and Sofiene Dellagi. Large language and foundation models for machinery health monitoring: A systematic review. Applied Sciences, 16(5):2493, 2026.

[38] Minhyeok Cha, Sang-il Yoon, Seongrae Kim, Daeyoung Kang, Keonwoo Nam, Teakyong Lee, and Joon-Young Kim. Large language model based autonomous agent for prognostics and health management. Machines, 13(9):831, 2025.

[39] Laifa Tao, Shangyu Li, Haifei Liu, Qixuan Huang, Liang Ma, Guoao Ning, Yi-Ling Chen, Yunlong Wu, Bin Li, Weiwei Zhang, Zhengduo Zhao, Wenchao Zhan, Wenyan Cao, Chao Wang, Hongmei Liu, Jian Ma, Mingliang Suo, Yujie Cheng, Yu Ding, Dengwei Song, and Lü Chen. An outline of prognostics and health management large model: Concepts, paradigms, and challenges. Mechanical Systems and Signal Processing, 232:112683, 2025.

[40] Peifeng Liu, Lu Qian, Xingwei Zhao, and Bo Tao. Joint knowledge graph and large language model for fault diagnosis and its application in aviation assembly. IEEE Transactions on Industrial Informatics, 20(6):8160–8169, 2024.

[41] Xingming Liao, Chong Chen, Zhuowei Wang, Ying Liu, Tao Wang, and Lianglun Cheng. Large language model assisted fine-grained knowledge graph construction for robotic fault diagnosis. Advanced Engineering Informatics, 65:103134, 2025.

[42] Lunyong Li, Auwal Haruna, Wanming Ying, Khandaker Noman, and Yongbo Li. Knowledge graph-driven fault diagnosis for aviation equipment: Integrating improved joint extraction with large language model. Journal ofIndustrial Information Integration, 50:101039, 2026.

[43] Christodoulos Constantinides, Dhaval C. Patel, Shuxin Lin, Claudio Guerrero, Sunil Dagajirao Patil, and Jayant Kalagnanam. FailureSensorIQ: A multi-choice QA dataset for understanding sensor relationships and failure modes. In Advances in Neural Information Processing Systems, volume 38, 2025. Datasets and Benchmarks Track.

[44] Yun Gao, Yuhua Qin, Tao Wang, Yue Cui, and Pengxiang Wang. Knowledge graph enhanced large language model framework for causal chain reasoning in industrial fault diagnosis. Expert Systems with Applications, 318:132027, 2026.

[45] Yiqing Song, Wenfa Li, Guiren Dai, and Xinna Shang. Advancements in complex knowledge graph question answering: A survey. Electronics, 12(21):4395, 2023.

[46] C. Peng, F. Xia, M. Naseriparsa, and F. Osborne. Knowledge graphs: Opportunities and challenges. Artificial Intelligence Review, 56:13071–13102, 2023.

[47] Sander Schulhof, Michael Ilie, Nishant Balepur, Konstantine Kahadze, Amanda Liu, Chenglei Si, Yinheng Li, Aayush Gupta, HyoJung Han, Sevien Schulhof, Pranav Sandeep Dulepet, Saurav Vidyadhara, Dayeon Ki, Sweta Agrawal, Chau Pham, Gerson Kroiz, Feileen Li, Hudson Tao, Ashay Srivastava, Hevander Da Costa, Saloni Gupta, Megan L. Rogers, Inna Goncearenco, Giuseppe Sarli, Igor Galynker, Denis Peskof, Marine Carpuat, Jules White, Shyamal Anadkat, Alexander Hoyle, and Philip Resnik. The prompt report: A systematic survey of promp engineering techniques, 2025.

[48] Sondos Mahmoud Bsharat, Aidar Myrzakhan, and Zhiqiang Shen. Principled instructions are all you need for questioning LLaMA-1/2, GPT-3.5/4, 2024.

[49] Deep Search Team. Docling technical report. Technical report, 8 2024.

[50] Microsoft. Chunk large documents for RAG and vector search in Azure AI Search. https://learn.microsoft.com/en-us/azure/ search/vector-search-how-to-chunk-documents. Accessed: 2026-10-09.

[51] Sinchana Ramakanth Bhat, Max Rudat, Jannis Spiekermann, and Nicolas Flores-Herr. Rethinking chunk size for long-document retrieval: A multi-dataset analysis. arXiv preprint arXiv:2505.21700, 2025.

[52] Belinda Mo, Kyssen Yu, Joshua Kazdan, Joan Cabezas, Proud Mpala, Lisa Yu, Chris Cundy, Charilaos Kanatsoulis, and Sanmi Koyejo. KGGen: Extracting knowledge graphs from plain text with language models. In Advances in Neural Information Processing Systems, volume 38, 2025.

[53] H. I. H. Saravanamuttoo, G. F. C. Rogers, Henry Cohen, Paul Straznicky, and Andrew Nix. Gas Turbine Theory. Pearson, 7 edition, 2017.

[54] Muhammad Aman Sheikh, Nursyarizal Mohd Nor, Taib Ibrahim, Sheikh Tahir Bakhsh, Muhammad Irfan, and Hanita Binti Daud. Noninvasive methods for condition monitoring and electrical fault diagnosis of induction motors. In Mustafa Demetgul and Muhammet Ünal, editors, Fault Diagnosis and Detection. IntechOpen, 2017.

[55] U.S. Department of Energy and Air Movement and Control Association International. Improving fan system performance: A sourcebook for industry. Technical Report DOE/GO-102003-1294, U.S. Department of Energy, 2003.

[56] U.S. Department of the Interior, Bureau of Reclamation. Transformers: Basics, maintenance, and diagnostics. Technical report, Bureau of Reclamation, Denver, Colorado, April 2005.

## A. Prompts

The online CoP pipeline uses four LLM calls: clarification, failure-mode extraction, failure-mode reasoning, and finaljudging. Each module uses a dedicated system instruction and a structured user prompt. The templates are reported below. Terms enclosed in angle brackets indicate values populated at inference time.

## A.1. Ofline CKG Construction

The ofline CKG construction uses one LLM call for each text unit extracted from the reference documentation. The model is asked to identify one complete failure-to-observation chain grounded in the current text unit. The output is constrained to a structured schema containing one failure mode, one to five changes in physical quantities, and one monitoring sensor.

## Diagnostic-Chain Extraction System Prompt

You extract ONE failure-mode → sensor causal chain for asset: <Asset>.

Describe ONE coherent monitoring scenario on a <Asset> as a chain:

failure\_mode → variable\_1 changes → variable\_2 changes → ... → sensor

Fields:

∙ failure\_mode: named fault / degradation / abnormal condition of the <Asset> (or a named subcomponent).

∙ steps: ordered list of 1–5 physical-variable changes. Each step contains:

– variable: the physical quantity that moves;

– location: where on the <Asset> it is taken; empty if the CHUNK does not say;

– direction: “increases” or “decreases”.

∙ steps[0] is what the failure mode drives directly; each later step is caused by the one before it; the LAST step is what the sensor reads.

∙ sensor: instrument that reads the LAST step’s variable.

If the CHUNK does not ground the failure\_mode, at least one step, AND the sensor for the SAME scenario or is only general theory / history / equations / design with no failure-to-monitoring link — leave failure\_mode and sensor empty and steps=[].

## Rules:

∙ Use ONLY facts grounded in the CHUNK. Do NOT add physics hops the CHUNK does not state, and do NOT pad the chain to make it longer. A 1-step chain is correct when that is all the CHUNK supports.

∙ Every step must be a physical variable that goes up or down — not a fault, not an instrument, not a maintenance action.

∙ Do not repeat the same variable+location+direction twice in the chain.

∙ Prefer the dominant failure scenario when several appear.

∙ Stay specific to <Asset>; do not import failure modes from other asset classes.

∙ Keep each field a short phrase; no markdown.

## Diagnostic-Chain Extraction User Prompt Template

CHUNK:   
<Reference Text Unit>

The structured output contains the fields failure\_mode, steps, and sensor. Each element of steps contains a variable, an optional location, and a direction restricted to increases or decreases. Empty outputs are interpreted as rejected text units and do not contribute to the CKG.

## A.2. Clarification

The clarification module checks whether the original multiple-choice question or its options require rewording before the diagnostic process starts. The answer choices are preserved in number, order, and meaning.

## Clarification System Prompt

You clarify failure-mode / sensor multiple-choice questions before they are answered.

## Rules:

∙ First decide needs\_clarification.

∙ If already clear: set needs\_clarification=false and leave clarified\_question empty and clarified\_options as []. Do not rewrite anything.

∙ If clarification is needed: set needs\_clarification=true, rewrite clarified\_question, and return clarified\_options with the same number/order of options.

∙ Preserve the original meaning, asset, failure event, and which sensor each option refers to.

∙ Do not add, drop, merge, or reorder options when rewriting.

∙ Do not pick an answer. Do not invent new failure modes or sensors not implied by the text.

∙ Make vague wording concrete without changing substance.

## Clarification User Prompt Template

Question: <Question>   
Options:   
<Option ID 1> <Option 1>   
<Option ID 2> <Option 2>   
There are exactly <N> options with ids <Option IDs>.   
If needs\_clarification=true, return clarified\_options with the same length and order; if false, leave   
clarified\_question empty and clarified\_options=[].

The structured output contains needs\_clarification, clarified\_question, clarified\_options, and a short note describing the ambiguity when present.

## A.3. Failure-Mode Extraction

Failure modes are extracted from the clarified question and answer options. This step is independent of the CKG.

## Failure-Mode Extraction System Prompt

You are a failure analysis expert. Extract all failure modes mentioned or implied in the given question and answer options.

A failure mode is a way in which a component, system, or process can fail to perform its intended function.

Look for:

∙ Direct mentions of failure modes, e.g. “bearing wear” or “shaft misalignment”.

∙ Implied failures described by symptoms or efects.

∙ Failure conditions mentioned in context.

Extract the failure mode names as they naturally appear in the text. Be precise and avoid over-interpretation.

Failure-Mode Extraction User Prompt Template   
Question: <Clarified Question>   
Options:   
<Option ID 1> <Option 1>   
<Option ID 2> <Option 2>   
Extract all failure modes mentioned or implied in the above text.   
A failure mode is a way a component or system can fail.   
Look for direct mentions, descriptions of failures, or failure conditions.

The output is a list of extracted failure modes. Each failure mode is subsequently analysed independently.

## A.4. Failure-Mode Reasoning

For each extracted failure mode, the model receives the retrieved CKG chains and the corresponding chain structure. The retrieved paths provide examples of failure-to-sensor reasoning and are used to guide the generation of plausible diagnostic chains for the target failure mode.

## Failure-Mode Reasoning System Prompt

You are a failure analysis expert. Given a specific failure mode and knowledge graph examples, identify what diagnostic chains this failure mode could produce.

Use the knowledge graph examples as patterns to understand how failure modes typically lead to signatures and then to sensors.

Do NOT try to answer any specific question. Analyze the failure mode itself and list the possible diagnostic chains it could lead to.

## Failure-Mode Reasoning User Prompt Template

Asset: <Asset>   
Failure mode to analyze: <Failure Mode>   
Knowledge graph examples showing diagnostic chain patterns for <Asset>:   
<Retrieved CKG Chains>   
Chain pattern:   
<Failure Mode> – results in –> <Signature> – leads to –> <Signature> – is monitored   
by –> <Sensor>   
Based on these examples and your domain knowledge about <Asset> systems, analyze what diagnostic chains   
the failure mode <Failure Mode> could produce.   
What signatures might this failure mode cause?   
What sensors could detect those signatures?   
Do not reference any specific question or try to pick an answer. Analyze the failure mode itself in the context   
of <Asset> systems.

The structured output contains the list of possible diagnostic chains inferred for the target failure mode.

## A.5. Judge

The Judge receives the clarified question, the answer choices, the task polarity, and the reasoning generated for all extracted failure modes. It compares the inferred chains with the available choices and returns one option

## Judge System Prompt

You are the Judge Agent for failure-mode / sensor multiple-choice questions.

You will receive independent reasoning about failure modes extracted from the question/options, along with the original question and options.

Use the failure mode reasonings to understand what diagnostic chains are possible, then match those to the options to select the best answer.

Follow the task polarity exactly: for MOST relevant select the strongest match; for LEAST relevant select the weakest or most contradicted match.

Judge User Prompt Template   
Question: <Clarified Question>   
Options:   
<Option ID 1> <Option 1>   
<Option ID 2> <Option 2>   
<Task Polarity Instruction>   
Independent failure mode analysis:   
Failure Mode: <Failure Mode 1>   
Possible chains: <Possible Chains 1>   
Based on the failure mode reasonings above, determine which option best matches the diagnostic chains.   
Select exactly one option from <Candidate Option IDs>.

For positive questions, the task instruction asks for the strongest and most direct diagnostic relationship. For negative questions, it asks for the weakest, most indirect, or most contradicted relationship. The Judge returns a comparative reasoning field and one answer identifier.

## B. Worked Example of Question Clarification

The ablation study in Table 5 shows that the clarification stage increases accuracy for all three models. The trace in Figure 5 illustrates the case in which the question is already clear and is passed through unchanged. This appendix reports the complementary case, in which the clarification module identifies an ambiguity, rewrites the question, and changes the final answer.

The example is taken from the fan subset of FailureSensorIQ and was obtained with Claude Opus 4.7.

## B.1. Original question

Input Question (Fan, Sensor2FailureMode, negative)

Stem. Which failure event is irrelevant for fan if there is an abnormal reading from the sensor length measurement?

A damaged oil seals   
B bearing wear   
C bearing damage   
D rotor fouled Gold   
E misalignment

The question contains two sources of ambiguity. First, the sensor is described as a length measurement, which does not specify the physical quantity being measured. Second, the negative formulation (irrelevant . . . if there is an abnormal reading) can be read either as “which failure is unimportant in general” or as “which failure cannot be indicated by this sensor”. Only the second reading corresponds to the diagnostic intent of the item.

## B.2. Clarification output

Clarification Output   
needs\_clarification: true   
Reasoning. “Sensor length measurement” is vague; clarified as a displacement/position measurement sensor.   
Also clarified that the question asks whichfailure mode is NOT relevant.   
Clarified question. For a fan, if a displacement/position sensor (length measurement, e.g., axial or radial   
displacement) shows an abnormal reading, which of the following failure events is NOT relevant to (i.e., cannot   
be indicated by) this sensor?   
Clarified options.   
A Damaged oil seals   
B Bearing wear   
C Bearing damage   
D Fouled rotor   
E Shaft misalignment

The rewrite acts on both ambiguities. The sensor is grounded as a displacement/position probe measuring axial or radial displacement, and the negative polarity is restated explicitly as “cannot be indicated by this sensor”. The answer options are only lightly normalised: their number, order, and meaning are preserved, and no answer is suggested by the rewrite.

## C. Full Inference Trace

This appendix reports the complete output of every stage of the online CoP pipeline for a single FailureSensorIQ item. The item belongs to the aero gas turbine subset and was processed by the full CoP configuration with Claude Opus 4.7. The prompt templates used at each step are given in Appendix A.

## 1 Question

Stem. For aero gas turbine, if a failure event air inlet blockage occurs, which sensor out of the choices is the most relevant sensor regarding the occurrence of the failure event?

<table><tr><td>A</td><td>vibration</td></tr><tr><td>B</td><td>oil leakage/ consumption</td></tr><tr><td>C</td><td>compresor pressure/ pressure ratio Gold</td></tr><tr><td>D</td><td>fuel pressure/ fuel flow</td></tr><tr><td>E</td><td>power turbine temperature</td></tr></table>

## 2 Clarify

Is the question ambiguous or unclear?

needs\_clarification: false

clarified\_options: []

The question is deemed unambiguous, so no changes are made. The original stem and options are passed unchanged to steps 3a and 3b.

## 3a Retrieve & expand

What reasoning examples exist in the CKG?

Retrieval seeds. The question and options are embedded and compared with the failure-mode and sensor entities of the aero gas turbine CKG. The top-2 seeds are:

<table><tr><td>Rank</td><td>Seed entity</td><td>Cosine similarity</td></tr><tr><td>1</td><td>Turbine inlet temperature sensor</td><td>0.6941</td></tr><tr><td>2</td><td>turbine inlet temperature sensor</td><td>0.6923</td></tr></table>

Expanded CKG chains. Backward expansion from the two sensor seeds yields nine complete chains. Each chain is written as failure mode → physical change(s) → sensor, where the arrows correspond, in order, to the CKG relations results in, leads to, and is monitored by:

1. Variable stator overheating → Stator temperature at variable stators increases → Turbine eficiency at power turbine decreases → Turbine inlet temperature at turbine inlet increases → Turbine inlet temperature sensor

2. High turbine inlet temperature causes excess NOx → Turbine inlet temperature at turbine inlet increases → Turbine inlet temperature sensor

3. build-up of deposits on compressor blading → compressor isentropic eficiency at compressor decreases → turbine operating temperature at turbine increases → turbine inlet temperature sensor

4. excessive accessory power demand → fuel flow at combustor increases → turbine inlet temperature at turbine inlet increases → turbine inlet temperature sensor

5. turbine inlet temperature exceeds safe limit → turbine inlet temperature at turbine inlet increases → turbine inlet temperature sensor

6. nozzle area reduction → nozzle area at propelling nozzle decreases → turbine inlet temperature at turbine inlet increases → turbine inlet temperature sensor

7. Turbine inlet temperature exceeds limit → turbine inlet temperature at turbine inlet increases → turbine inlet temperature sensor

8. turbine overheating → turbine inlet temperature increases → turbine inlet temperature sensor

9. flame-out during rapid climb → fuel flow rate at fuel metering unit decreases → combustor temperature at combustor chamber decreases → turbine inlet temperature at turbine inlet decreases → turbine inlet temperature sensor

Output. {"failure\_modes": ["air inlet blockage"]}

One failure mode is extracted from the question and options. This step does not access the CKG.

## 4 Reason about the failure mode

What physical chains could thisfailure mode produce?

Input. Asset (aero gas turbine), target failure mode (air inlet blockage), the nine CKG chains from step 3a as in-context examples, and the chain pattern. The question and options are not provided.

Output. One LLM call produces ten candidate chains, none of which is copied from step 3a:

1. air inlet blockage → inlet air mass flow at compressor inlet decreases → compressor pressure ratio at compressor decreases → compressor discharge pressure sensor

2. air inlet blockage → inlet air pressure at compressor inlet decreases → inlet pressure sensor

3. air inlet blockage → inlet air mass flow at compressor inlet decreases → air-fuel ratio at combustor decreases → turbine inlet temperature at turbine inlet increases → turbine inlet temperature sensor

4. air inlet blockage → compressor inlet pressure at compressor inlet decreases → compressor surge margin at compressor decreases → compressor pressure sensor

5. air inlet blockage → inlet air mass flow at compressor inlet decreases → engine thrust at nozzle decreases → thrust sensor

6. air inlet blockage → pressure drop across inlet at air intake increases → diferential pressure sensor

7. air inlet blockage → compressor rotational speed at compressor decreases → shaft speed sensor (N1/N2)

8. air inlet blockage → inlet air mass flow at compressor inlet decreases → exhaust gas temperature at turbine exhaust increases → exhaust gas temperature sensor

9. air inlet blockage → compressor inlet temperature at compressor inlet changes → inlet temperature sensor

10. air inlet blockage → inlet air mass flow at compressor inlet decreases → compressor stall margin at compressor decreases → vibration at compressor increases → vibration sensor

## 5 Judge

Sees question, options and all inferred chains

Input. Original question and options, the task polarity instruction (MOST relevant: select the option with the strongest, most direct diagnostic relationship), and the ten chains from step 4.

Judge reasoning. “Air inlet blockage most directly reduces inlet air massflow andpressure, which immediately manifests as a drop in compressor pressure/pressure ratio. Multiple diagnostic chains point to compressor pressure as the primary monitored parameter. EGT/turbine temp changes and vibration are secondary efects, while oil andfuel are unrelated.”

## Answer

C — compresor pressure/ pressure ratio

Correct

The trace shows the same behaviour as the fan example in Figure 5. The retrieved CKG chains concern diferent failure mechanisms and point to a sensor that is not the correct answer. They nevertheless provide the failure–physical change–sensor structure that the reasoning step applies to air inlet blockage. The correct answer follows from the newly inferred chains.