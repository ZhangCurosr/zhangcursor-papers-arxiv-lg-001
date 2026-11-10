# FALC: An Artifact-Centric Life Cycle Model and Research Agenda for AI Forensics

Adrian Johannes Staeves<sup>a,∗</sup>, Mayur Rattan Jaisinghani<sup>b</sup>, Rebekah Overdorf<sup>b</sup>, Frank Breitinger<sup>a,∗</sup>

<sup>a</sup>Chair for Cybersecurity, University of Augsburg, 86159 Augsburg, Germany <sup>b</sup>Chair for Security and Privacy in Socio-technical Systems, Ruhr-University Bochum, 44801 Bochum, Germany

## Abstract

AI systems increasingly appear in criminal investigations as tools, targets, and sources of evidence, yet AI forensics lacks a systematic account of the artifacts they produce: investigators often do not know which traces to look for, and for many traces they do find, no established analysis technique exists. In this paper we first developed the Forensic AI Life Cycle (FALC) that unlike6 other life cycle models shifts the focus onto the forensic artifacts created during the life cycle of an AI system. Through FALC we<sup>2</sup> derived a reference list of artifacts that might serve an investigation. Second, we collected examples for crimes with AI involvement<sup>0</sup> throughout FALC and questions likely to be asked when investigating these crimes. Third, we propose a taxonomy of forensic fields of expertise with a focus on the current possibility to analyze AI forensic artifacts using established techniques. For investigators,<sup>v</sup> the model provides a systematic basis for identifying which artifacts to look for when an AI system is relevant to a case. For researchers, the classification shows where existing forensic methods already sufice and where new techniques are most urgently needed.9

Keywords: AI Forensics, AI life-cycle, artifact-centric life-cycle, AI forensic taxonomy

## 1. Introduction.

Due to its recent democratization and versatility, AI can ap-<sup>[</sup> pear in a forensic investigation in various ways. It can be an asset to aid investigators, the target of an attack, or a medium for committing a crime. The digital forensics community has acknowledged the need to investigate AI systems (e.g., Dragonas et al. (2024); Jeong et al. (2025); Comeaux et al. (2026)). What is still missing is a structure that tells an investigator where to look for traces in an arbitrary AI system and that tells re-. searchers which parts of the problem are already solved and which are not.

To fill this gap, we developed a life-cycle model for AI systems which focuses on the artifacts that each stage in the life of an AI system leaves behind. We call it the Forensic AI Life Cycle (FALC). Because of its focus, FALC is both an artifactcentric life-cycle model and an artifact-centric reference model<sup>r</sup> for forensic investigations. FALC provides a systematic way to enumerate questions that can arise during an investigation around an AI system and the artifacts available to answer them. Comparing those questions against the current state of forensic practice yields our second contribution: a classification of the fields of expertise that AI forensics requires into fields where existing methods transfer unchanged, fields where existing methods must be adapted, and fields where no adequate method exists yet. This last group marks where research is most urgently needed. In particular, our contributions are:

• FALC, an artifact-centric reference model of the AI life cycle designed for forensic investigation rather than for process optimization.

• A catalogue of forensic artifacts produced at each stage of the AI life cycle, together with where they can be expected to be found.

• A collection of crimes and forensic questions associated with each stage, illustrating how AI can be involved in an ofense.

• A taxonomy of the forensic fields of expertise required by AI forensics, distinguishing directly applicable fields, fields requiring adaptation, and fields that do not yet exist.

The remainder of this paper is structured as follows: Sec. 2 situates our work within related research on AI forensics and AI crime. Sec. 3 describes our methodology. Sec. 4 highlights the FALC and catalogues forensic artifacts produced in each stage. Sec. 5 gives, for each stage, examples of crimes and the forensic questions they raise. Sec. 6 derives the taxonomy of forensic fields of expertise from those questions. In Sec. 7 we discuss our findings and conclude this paper.

## 2. Background and Related Work

Research into how AI changes digital forensics largely falls into two groups: AIforforensics, which applies AI to improve forensic investigations (Dunsin et al. (2024); Yin et al. (2025);

Solanke (2022)); and AI forensics, which asks how investigations change when AI systems are involved in a crime (Schneider & Breitinger (2023)). This paper addresses the latter by introducing a forensic-focused, artifact-based AI life cycle model that provides a systematic structure for locating evidence when AI systems are investigated.

Baggili & Behzadan (2019) laid the conceptual groundwork for AI forensics, defining it as the investigation of “failures in AI enabled systems.” This definition notably omits cases where AI can serve as evidence without having failed, such as when it was used as a tool to commit a crime. Their taxonomy is correspondingly organized around what is examined, whereas FALC is organized around where evidence originates. The diferences also reflect how the field of AI has changed over the past years and the rapid change in how people interact with and use AI systems. In Appendix A we relate FALC to their resulting taxonomy.

Several papers propose classifications of AI crime (Jeong (2020); Sibai (2020); King et al. (2020); Blauth et al. (2022)). While these papers discuss which AI crimes can happen, our approach focuses on which artifacts are created and available for investigators.

Several works investigated specific AI systems or a group of similar AI systems, determining what can be found in these systems during a forensic investigation. Noteworthy examples are AI companion apps (Comeaux et al. (2026)), the ChatGPT mobile app (Dragonas et al. (2024)) and OpenClaw (Gruber & Hilgert (2026)). While these are valuable case studies, they can not show the bigger picture of AI forensics.

Outside of forensics, many AI lifecycle models exist. Wang et al. (2025) identified and compared 20 of the most important ones. These models serve to structure industry processes or ensure ethical AI, and are therefore not well-suited to be directly applied to AI forensic questions. We use these as a starting point for our life cycle model.

## 3. Methodology

As mentioned in Sec. 2, multiple AI life cycle models exist, but none of them is suitable for AI forensics. We therefore developed a new life cycle model that separates stages by the artifacts they produce rather than by when they are performed. The model was derived through conceptual synthesis of existing life cycle models, not a systematic review. We first identified common and unique stages and development steps for the 20 most important life cycle models<sup>1</sup> identified by Wang et al. (2025). Next, we determined the forensically relevant artifacts created during each step by reading diverse academic literature describing common practices and challenges faced during each step, looking for direct and implicit mentions of created artifacts<sup>2</sup>. We deemed the artifact list complete once at least three relevant publications resulted in no change to the list. Finally, we combined the individual steps into stages in a way that each stage creates diferent types of artifacts. Consequently, we arrived at the Forensic AI Life Cycle (FALC) shown in Fig. 1. In Appendix B we test FALC by applying it to a real investigation.

![](images/72726817a0f0df3a38ef4f74908f793cf234a74078620c008d57f8272f586080.jpg)  
Figure 1: The FALC model. Governance and quality assurance spans the entire life cycle. The remaining six stages are deliberately drawn without ordering: in practice they are interwoven, revisited, and sometimes skipped entirely, so an investigator should search for artifacts from every stage regardless of the stage in which the suspected crime occurred.

When collecting crimes and forensic questions, our goal was not to create a comprehensive list of all AI-related crimes, but instead to show that crimes can happen during each stage of FALC. Therefore, we decided against a structured literature review and instead searched Google Scholar for AI crime-related literature until we had found at least one example of a crime for each stage of FALC. From these crimes, we derived the questions an investigator may need to answer in such cases.

Lastly, we created a taxonomy to help AI forensic researchers determine which AI forensic subfields need more research and to help investigators determine which artifacts provide the most value to an investigation. For creating the taxonomy, we followed the empirical-to-conceptual approach described by Nickerson et al. (2013).

## 4. FALC: The Forensic AI Life Cycle Model

Most AI life cycle models difer primarily in how they order stages, and to a lesser extent in how they group the steps needed to build an AI model. Since our goal is to give investigators an overview of which artifacts might exist when examining an AI system, we deliberately leave the stage order undefined. In practice, stages are often interwoven, reordered, or revisited, so investigators should look for artifacts from all stages regardless of a case’s specifics.

We considered adding typical connections between stages, but decided against it: doing so could mislead investigators into assuming that a later stage does not need examination if the crime occurred at an earlier one. As artifacts from any stage may be relevant regardless of where the crime took place, we impose no ordering.

Removing the ordering allowed us to regroup the steps of AI system development by the artifacts they produce. This resulted in the FALC consisting of the stages shown in Fig. 1. Not every stage is visited in every case; the model categorizes the steps that can occur during the development of an AI system, not those that always do.

In the remainder of this section we briefly describe each of the 7 stages in FALC, and list the artifacts that belong to each stage. We include the list of artifacts, as we reference them in Section 6, and they are an integral part of FALC as a reference model, that can help investigators find all relevant artifacts during an investigation. However, reading the complete list of artifacts is not necessary to understand subsequent sections.

## 4.1. Governance and Quality Assurance

This stage includes all steps taken to prepare for developing an AI system, such as business understanding, problem design, defining requirements, and creating feasibility studies. It also includes all steps taken to ensure the quality of the resulting system, such as documenting, auditing, testing, and monitoring. As defined, this stage is active before and throughout the other stages. All steps of this stage produce documentation and reports that are usually restricted to company internal use. Namely, the following artifacts may exist:

Policies and procedures: All documents that describe how processes are supposed to be done or structured.<sup>3</sup>

Licenses: Usually, software use is subject to a license that describes what the user is allowed to do with the software. For bigger projects, diferent parts can have diferent licenses.

Design documents: Documents that describe what the system is supposed to do and the problem it is supposed to solve. Examples are the problem statement, requirements, risk assessments, feasibility studies, and ethical evaluations (Martínez-Plumed et al. (2017); Haakman et al. (2021); Diaz et al. (2021)).

Documentation: Documents that describe how the product works and how the parts work together.

Audit results: Various audits might be performed throughout an AI system’s life, such as data quality-, model performance-, risk-, ethical- or law compliance audits (Li & Goel (2025)). The findings of these audits can be of great value to an investigation, as they might confirm the system’s state at a specific point in time, hint at potential weaknesses, reveal previously unknown processes, or provide various other insights.

Test results: The results of manual and automated testing can provide insights into the system’s functionality at a specific point in time. These include unit, platform, application, and post-deployment tests, as well as model evaluation scores.

Test procedures: These can be automated test pipelines but also test documentation or instructions for manual testing (Choudhary et al. (2026)). The test procedures can give the test results meaning.

Monitoring: Usually companies monitor their system’s performance and user behavior after deployment to get early insights in case of failure or if performance degrades (Grover (2025)). Artifacts created through monitoring can give an investigator these same insights.

User tracking: For reasons such as improving the product or selling the data for advertising, AI systems might track the user’s behavior and collect vast amounts of user data (Comeaux et al. (2026)). Such data can give investigators additional insights into the users’ behavior and motives. Because such data is often collected by or sold to third parties, it may remain obtainable from them even if one party deleted it before the investigation.

User feedback: Companies may solicit feedback directly, or users may post opinions on review sites and social media. Indirect feedback is subjective, hard to locate, and dificult to analyze automatically, but it is usually publicly available and can reveal how user perception and user-facing behavior changed over time.

Backups and version history: As in any software project, backups of the system or its components may exist, and version control software such as git is commonly used (Choudhary et al. (2026)). Both reveal the system’s state at a given point in time and how it changed during development and use.

## 4.2. Data Acquisition

This stage covers all steps taken to obtain the initial data, which can be acquired in several ways.

First party data: A company or other entity that collects user data might already have all the information they need. In these cases the data acquisition can be as simple as accessing the own data lake (Roh et al. (2021)).

Second or third party data: Many datasets are ofered for sale by data brokers (Chen et al. (2023)). Some datasets are also available for free. If one of these datasets is suitable for the intended use case, one might decide to acquire one or multiple of them.

Data generation: Data can be generated by another AI system or through manual efort such as labeling or surveys (Roh et al. (2021)). It can also be derived from existing sources, for example by scraping image search results or by combining datasets in new ways.

These approaches are often combined. Datasets frequently contain metadata that is unused during training but essential to an investigation, for example, to establish the data’s source. Depending on how a dataset was acquired, the following artifacts may be available:

Raw data: The data that was obtained itself can be essential to an investigation. Note that the raw data can possibly be found at any party involved in the data acquisition.

Inspection: Before and/or after the acquisition, the data was probably inspected for its usability. This includes inspecting individual samples as well as metrics that relate to part of or the entire dataset (Chen et al. (2023)). These metrics may be stored, and the inspections themselves can leave traces on the devices used.

Networking: During acquisition, the data was probably transmitted over the internet or a local area network at least once. Therefore, networking-related artifacts might exist.

Storage: The data has to be stored. As AI requires large amounts of data, possibly dedicated storage servers might be utilized.

Access logs: If the data was extracted from a data lake or downloaded from a broker’s server, traces of the access can be found at the storage server.

Generation system: If the data was generated, that system is an artifact of this stage. Among others such an artifact can be a survey system, manual labeling tool or a diferent AI system that itself might be worth to (partially) inspect.

Negotiations and applications: Buying a dataset from a broker or similar source might involve sending an application and/or negotiating the price (Chen et al. (2023)).

Monetary Transactions If the dataset was bought, traces of the monetary transaction might exist.

## 4.3. Data Preparation

This stage encompasses all data processing steps taken before training the model. It can take place before data acquisition if preprocessed data is acquired, after the acquisition, or both. In the case of online data augmentation, some data preparation steps can even happen as part of the training pipeline (Zhou et al. (2025)). It includes data cleaning, data reduction, data preprocessing, and data labeling. What these steps have in common is that they typically process large volumes of data. Depending on when, where, how, and which processing took place, the following artifacts may be available:

Prepared data: Creating the prepared data is the main goal of this stage. Because of the high resource intensity of AI systems, new data formats that are optimized for AI might be used (Zhou et al. (2025))

Partially prepared data: Data might be saved after part of the preparation took place. Access to such data can answer questions about the state of the data during diferent stages of the data preparation.

Processing pipelines: Data preparation usually consists of multiple smaller steps. Pipelines are used to combine these steps (Zhou et al. (2025)). Access to these pipelines can tell investigators what happened during the data preparation.

Collaboration artifacts: Some data preparation steps such as data labeling are hard to automate and too much efort for one person. Therefore, they are often done crowd-based or performed by a possibly spatially spread-out team (Roh et al. (2021)), which can result in artifacts such as calls for participation, websites as well as network artifacts and collaboration software that was used. If a third-party service was used for the collaboration, some of these artifacts might be found at that third party as well.

AI systems: AI can be used to generate new training data, augment, or label it (Shorten et al. (2021)). If such an AI system was used, analyzing it can help determine whether it introduced errors.

Operating system artifacts: Program execution and file modifications can leave operating system-level traces. These can be found even if other traces of this stage are not available.

Augmentation input Some data augmentation methods require inputs, such as a knowledge graph for synonym replacements (Shorten et al. (2021)) or a prompt for AIbased data augmentation.

## 4.4. Model Development

This stage covers implementing and training the AI model itself, and is usually the most compute-intensive part of an AI system’s life. Fine-tuning and training Low-Rank Adaptations (LoRAs) also belong here. Depending on how training was performed, the following artifacts may be generated:

Final model or adaptation: The trained or fine-tuned model is the goal. As it will be at the core of the final system, being able to analyze it can greatly benefit an investigation.

Training checkpoints: Because the training is often resourceintensive and might require multiple attempts, intermediate results of the training process might be saved (Zhou et al. (2025)). These checkpoints can help investigators understand how the model changed during the training process and how it became the way it ended up.

Base model: If a model is fine-tuned or adapted in any other way, the base model might be a useful artifact, as it can help understand what the training changed about it.

Provenance markings: Some AI companies leave specific markings in the models they train. One way to mark an AI model is to let it memorize an artificial and abnormal data point (Yang et al. (2026)).

Training parameters: The success of model training as well as its susceptibility to certain attacks can be heavily influenced by the set of parameters chosen during training (DeCastro-García et al. (2019)).

Training metrics: As the optimal set of training parameters usually is not known in advance (DeCastro-García et al. (2019)), some success metrics such as the validation loss can be tracked during training. These can give an estimate of the model’s performance and error rates.

Training framework and tools: Training is typically performed using frameworks such as TensorFlow or PyTorch, whose configuration and version can indicate how the model was built.

Model code: Even when using a framework, some code is required to implement the model. This code can help understand how the model is structured and its intended use.

Dedicated training hardware: Because training can be highly compute-intensive, it may have been performed on dedicated servers. This can leave traces such as data transfers between compute nodes (Zhou et al. (2025)), or, if the hardware was rented, contracts, monetary transactions, and usage logs.

Training optimizations: To speed up the training, some data might be stored in an optimized file format, or optimizations such as caches can be used (Zhou et al. (2025)).

## 4.5. Model Integration

An AI Model is not deployed directly and instead integrated into a larger system. Even in cases as ChatGPT, Claude or DALL-E where users think they are directly interacting with the AI model, it is part of a larger system that contains a web interface, input validation, and output post-processing (Weber (2024); Shao et al. (2025)). This stage contains all steps taken to integrate the AI model into a larger system. On a high level, this stage produces the same artifacts as traditional software development, such as source code, compiled programs, default configurations, and databases. Not in this stage belong artifacts that better fit into the governance and quality assurance stage. On a more detailed level, the created artifacts are highly diverse and dependent on the purpose of the AI system and its capabilities, which are as diverse as any software can be. This makes it infeasible within the scope of this paper to list them all here. Instead, we recommend that investigators approach the artifacts created by this stage by first determining which software components make up the entire software system and then determining the artifacts for each subsystem. These software components can be grouped by when they are executed in relation to the AI model.

Before the model: Examples include input generation (e.g., for agentic invocations), input validation, input preprocessing, context enhancement (e.g., for RAG), tokenization, and prompt templates.

During model execution: Examples include tools, memory access, and environmental interactions.

After the model: Examples include output filtering, output conversion, output projectors, and modality generators<sup>4</sup>.

Independent of the model: AI systems can contain software components that operate completely independent of the AI model, e.g., because they provide a diferent functionality.

## 4.6. Deployment

Depending on the scale of the AI system, various steps might be necessary to deploy it. Deployment can happen to a server, to the devices of end users, or both. Note that tests, which are often seen as part of deploying a software system in FALC, are part of the governance and quality assurance stage.

Packaged software: For deployment, the software gets packaged. Among other options, this can be as containers (John et al. (2021)) for deployment into the cloud or to serverless architectures or as apps for deployment to end-user devices.

Compressed model/software: To deploy software and models they can be compressed (Brayford et al. (2019)). Access to these compressed versions, before and after they were transmitted to the production server, can help determine whether they have been tampered with.

Rewrites and re-implementations: Due to diferent requirements in production systems, some parts of an AI system might get re-implemented (e.g., in a diferent programming language) as part of the deployment (John et al. (2021)). As this can introduce vulnerabilities, errors, or even backdoors, being able to investigate how the software changed during this stage can be valuable for an investigation.

Deployment pipelines: One goal of MLOps and DevOps is to automate the deployment of the system using automated pipelines (Gopinathan (2025)). Access to these pipelines can tell investigators what happened during deployment and at what point an adversary might have intercepted it.

Hosting agreements: If the software is deployed to rented servers, a hosting agreement as well as monetary transactions are likely to exist. Sometimes, the AI model itself is hosted separately due to its high computational demands.

Deployment Configurations: During deployment, a system connects to the services it relies on by configuring endpoints and provisioning API keys, access tokens, or service accounts, often stored in configuration files, environment variables, or secret managers. It can help establish a link between the external services on which a system depends and who is responsible for them.

Note that, except for the deployment pipeline, these artifacts might be found at the deploying/distributing entity as well as at the entity the system is distributed to, and diferences between the two versions can be essential evidence.

## 4.7. Use

After an AI system is deployed, it can be used. This stage only encompasses the providing of the system and the users using the system. All monitoring steps taken, including the ones to monitor the use, are part of governance and quality assurance. Nevertheless, the use of an AI system can produce a vast amount of artifacts that can be of interest. Depending on how the system is set up, these artifacts can be found on the end user’s device, on a server used to provide the service, or both.

Inputs: AI systems might store the inputs the user provided even after they were processed. This category also includes other forms of input such as sensor data.

Model outputs: An AI system may store the result of AI model invocations. Examples include chat history with an LLM (Dragonas et al. (2024)) or generated images.

Intermediate generations: Some AI systems generate output that is not shown to the user. Examples include the chain of thought from a reasoning LLM or a planned trajectory from an agentic system.

Provenance markings: Some AI systems leave specific markings in the content they generate, such as a watermark, C2PA manifests, or generation settings stored in the metadata.

Tool implementations: Tool use means that an AI model, usually an LLM can call certain tools to interact with the environment. These tools can be any program with a predefined API and range from simple data retrieval to file manipulation, entire workflows, and delegating subtasks to a diferent AI system. The existence of such tools can tell investigators what an AI might have been able to do (Gruber & Hilgert (2026)).

Tool interactions: When an AI uses a tool, it can leave traces. Besides the expected outcome of the tool use, networking artifacts and log entries might be created (Gruber & Hilgert (2026)). These artifacts show that the tool was available to the AI and used.

Databases and storage files: Model memory and other persistent data can be found in databases and files that serve a similar purpose (Alnaqbi et al. (2025)).

Vector databases: Retrieval Augmented Generation (RAG) and similar methods might use a vector database to store additional data injected during generation (Zhou et al. (2025)).

Log files: Like their non-AI counterparts, AI systems can log certain events with varying amounts of detail (Alnaqbi et al. (2025)).

Caches: To prevent the same data from having to be transmitted multiple times, it might be cached and therefore available to an investigation (Dragonas et al. (2024)).

Configuration files: They store information about the software’s settings (Gruber & Hilgert (2026); Alnaqbi et al. (2025)). While these might not directly be of value to an investigation, they can reveal which other traces might exist.

Client-server communication: If the software consists of a client and a server, their communication can serve as artifacts. This includes the network communication, authentication details, and APIs (Dragonas et al. (2024)). These can also be of interest indirectly if the investigators only have access to one of the devices (e.g., the end-user’s device), as they can be used to access some data on the other system (Comeaux et al. (2026)).

Computer use traces: As with any other application, operating system artifacts can reveal additional information about how the user interacted with an AI system.

Usage and billing logs: Models hosted on external infrastructure may have their requests and responses logged, and providers typically retain usage and billing records. Anomalies in these records can serve as early indicators of misuse, and they may be the only evidence available when the model itself is beyond the investigators’ reach.

## 5. Crimes and AI Forensic Investigative Questions

This section examines each stage of the AI life cycle in turn, giving examples of crimes in which AI is involved at that stage. In a second step, we use the crimes to determine AI forensic investigative questions that can be relevant to real cases. We use the term “AI forensic investigative questions" to describe technical questions directly applicable to a specific AI crime, analogous to digital forensic investigative questions<sup>5</sup>. This is to diferentiate them from forensic questions, which are often understood as the 5W+1H questions.

The purpose of this section is twofold. First to illustrate how AI can be involved in a crime and to show that crimes can happen during all stages of AI development. Second to establish a set of AI forensic investigative questions, for use in Section 6. Neither the list of crimes nor the set of AI forensic investigative questions are conclusive, but they span all stages of FALC. Conclusivity is arguably not possible, as AI is still quickly evolving, but also not required for the purposes of this paper.

## 5.1. Governance and Quality Assurance

Crimes that happen during the governance and quality assurance stage tend to impact the AI system only indirectly. An example is purposeful neglect of due diligence, such as not addressing model drift, inadequate testing, or inadequate design leading to harm (Cummings (2024); Sibai (2020)). Another example is an adversary sabotaging the design of the project, e.g., data poisoning during early planning, leading to harm down the line. The main AI forensic investigative questions regarding this stage revolve around accountability; either the company’s accountability for some failure or what led to a crime and the role certain individuals played in that. Artifacts from this stage can also be useful for answering investigative questions raised by other stages. For example, to narrow down when a certain behavior first occurred.

## 5.2. Data Acquisition

Illegal data acquisition is one example of a crime in the data acquisition stage. It ranges from obtaining data without the required permissions to acquiring material that is illegal to possess, such as CSAM. Here, the investigative questions typically concern how and from where the data was obtained.

Data poisoning and backdoor attacks (Hossain et al. (2024)) are also crimes that belong to the data acquisition stage, if the malicious modifications were performed before the data was acquired. Here, the AI forensic investigative questions are likely to revolve around the state of the data at acquisition time and how that afects the resulting model.

## 5.3. Data Preparation

Data poisoning is a crime that belongs to the data preparation stage if the malicious alterations were performed after the dataset was acquired. Here, AI forensic investigative questions are likely to revolve around how the data changed. That includes what the data looked like before and after the transformations, which transformations were performed, and the efects of the transformations.

## 5.4. Model Development

Training parameters can have a huge impact on the resulting model’s performance (DeCastro-García et al. (2019)). Making malicious modifications to them is a potential attack to sabotage a model’s performance. If such a crime is suspected, the main investigative questions are likely to revolve around whether the training parameters were modified, what impact that had, and who is accountable.

Another crime we want to discuss here is training a model with a malicious goal, such as aiding in criminal activities. Here, AI forensic investigative questions are likely to revolve around what the model does, why it behaves the way it does, and whether that behavior is intended or not.

Generative AI systems might exactly reproduce some of their training data. This phenomenon is called memorization and can become a problem if the reproduced training data is copyrighted material. While the exact mechanisms behind memorization are the subject of ongoing research, some court rulings already exist<sup>6</sup>, with the main question of whether the AI model stored the copyrighted material.

The goal of model theft is to create a functionally identical copy of a proprietary AI model. The main investigative question is likely to be whether the created model is stolen or was legitimately trained to fulfill a similar goal. Model theft is a particularly interesting example, as two AI models that are in diferent stages are involved. As a result, FALC has to be applied to both models separately.

## 5.5. Model Integration

A crime that belongs to the model integration stage is the development of malicious-by-design software that has an integrated AI model (Fritsch et al. (2022)). An example of this is a computer virus that uses AI to avoid detection. Forensic investigations surrounding such a system are likely similar to ones where the malicious software does not use AI, with some additional questions regarding the roles, capabilities, and functionality of the integrated AI.

The analysis of the model’s integration is also relevant for some crimes that belong to the use stage. For example, if AI forensic investigative questions ask for input validation or the capabilities and tools available to the model.

## 5.6. Deployment

During deployment crimes can happen that target the AI and ones where AI is part of the crime. An example of a crime targeting the AI system is replacing the model during deployment with a malicious one. Here, AI forensic investigative questions likely revolve around whether that replacement took place and which safety mechanisms were used to detect or prevent such a crime.

An AI system is part of a crime if the system is deployed without the consent of the owner of the hardware the system is deployed to. That can be because the hosting agreement does not allow for running an AI system or because the AI is part of malware. Here investigative questions are likely to revolve around what exactly the system does, what it consists of, how it works, and the targets the software was deployed to.

## 5.7. Use

Most crimes belong to the use stage, as this is where the system provides the functionality that can be abused and where it is most exposed to the public. All crimes in which AI serves as the instrument of the ofense, i.e., Jeong’s AI as a tool category (Jeong (2020)), fall into this stage. Examples include asking an LLM for advice on how to commit the crime, (partially)

automating attacks (Blauth et al. (2022)) or using AI to generate fake news. For these kinds of crimes, AI forensic investigative questions are likely to revolve around the user’s actions, interactions with the AI system, and the user’s intent. Some of these crimes can also require investigators to detect whether media was generated using an AI system (deepfake detection).

Just as in all other stages, an AI system can also be attacked. Model inversion attacks (Hossain et al. (2024)), model theft, adversarial attacks (Kurakin et al. (2016); Hossain et al. (2024)) and prompt injection are examples. Three forensic questions are likely to arise here. The first concerns which sources the AI system drew on and who is accountable for their content. Secondly, investigative questions might revolve around the actions the AI model took. Finally, questions may concern what caused the system to take a particular action, since only this link allows a malicious action by an AI system to be attributed to a person. This type of question will become more relevant with growing complexity and autonomy of AI systems, as with these, the output is no longer directly related to a specific user input.

## 6. Taxonomy

This section presents a taxonomy of the required forensic fields of expertise. The goal of this taxonomy is twofold. First, the taxonomy can help AI forensic researchers determine which subfields need more research. Second, it can help investigators determine which artifacts to look for, as those that are not possible to obtain or analyze with existing techniques are likely of lower value to an ongoing investigation. We define the analysis of each artifact to contribute to answering a given AI forensic investigative question as its own forensic field. Put diferently, we treat each pairing of investigative question and artifact as a diferent forensic field of expertise.

For developing the taxonomy, we followed the approach described by Nickerson et al. (2013). Considering the mentioned purposes, we decided on the meta-characteristic of current ability to analyze an artifact in a way that it contributes to answering an AI forensic investigative question as part of a forensic investigation. For ending conditions, we decided on the 5 subjective ending conditions: concise, robust, comprehensive, extensible, and explanatory as described in Nickerson et al. (2013). As objective ending conditions, we choose:

• All possible pairs of artifacts from Sec. 4 with AI forensic investigative questions from Sec. 5 have been examined.

• At least one object is classified under every characteristic of every dimension (Nickerson et al. (2013)).

• Every characteristic is unique within its dimension (i.e., there is no characteristic duplication within a dimen sion) (Nickerson et al. (2013)).

Due to the large number of artifacts and AI forensic investigative questions determined in Sec. 4 and Sec. 5, respectively, we mainly used an empirical-to-conceptual approach.

Once all ending conditions were met, our taxonomy consisted of two dimensions. Ability to acquire with the characteristics of does not apply (e.g., for follow-up questions, where the artifact is expected to be obtained already), obtainable with established techniques, and obtainable when adapting established techniques. The second dimension, Ability to analyze, has the characteristics impossible, analyzable with established techniques, analyzable when adapting established techniques, and new techniques required for analysis. This taxonomy fulfills all three objective ending conditions, and it is concise, robust, comprehensive, extendible, and explanatory. We argue for concise and robust despite the low number of dimensions, as the dimensions are directly related to the intended purpose. We also consider the taxonomy comprehensive despite the ability to acquire dimension not having a characteristic for needing new methods. We did not add this characteristic because it was not required for any of the sample cases we collected. This case also arguably does not exist, as AI systems fundamentally are computer programs that leave artifacts in the same places as non-AI computer programs. However, if an artifact that requires a new technique to acquire is created, discovered, or theorized in the future, adding this characteristic to the taxonomy is a simple extension.

Table 1 shows our final taxonomy. In many cases, an artifact can not be used to answer an AI forensic investigative question because they are completely unrelated. These cases are impossible to analyze for answering the question characteristic and are not included in the table.

In the following, we describe some challenges and open research questions when obtaining an artifact or analyzing it to answer a specific question, which served as the rationale behind placing them in a diferent characteristic than obtainable/analyzable using established techniques. These are the forensic fields of expertise where most research is still needed.

Large amounts of data. The training of AI models requires large amounts of data. If an investigation requires finding malicious modifications of such data, manually investigating the dataset is not feasible due to its volume, and automated anomaly detection is currently too unreliable to be used. Therefore, we placed the forensic fields that require in-depth analysis of large amounts of training data, such as finding data modifications based on the prepared data of the data preparation stage, as needing adaptations to the existing automated anomaly detection mechanisms.

Impact of training input on model. A further challenge arises from open questions surrounding model training. Although research has examined how training data, hyperparameters, and model architecture shape the resulting model (Mohammed et al. (2025); Liao et al. (2022); Raghu et al. (2021)), training in practice still relies largely on trying diferent configurations and selecting the most successful one. As a result, a given training choice cannot reliably be traced back to an intention. Investigators are therefore rarely able to determine from a model’s behavior alone whether a particular property was introduced deliberately or emerged incidentally. As far as we are aware, there are currently no forensic techniques that allow for such predictions, and most other techniques for predicting the outcome of model learning are unreliable. Therefore, we categorized all forensic fields of expertise where the question requires predicting the impact of something on the outcome of the training process as requiring new forensic analysis techniques.

<table><tr><td>Stage</td><td>Forensic Question</td><td>Relevant Artifacts</td><td>Obtain</td><td>Analysis</td></tr><tr><td>Governance and Quality Assurance</td><td>Determining account- ability</td><td rowspan="5">policies and procedures, licenses, design documents, audit results, test procedures, test results, monitoring</td><td>est</td><td>est</td></tr><tr><td>Governance and Quality Assurance</td><td>First occurrence of test results, audit results, user feedback, backups, and ver- abnormal behavior sion history</td><td>est</td><td>est</td></tr><tr><td>Data Acquisition</td><td>Data source and ac- networking, access logs, inspection, generation system, quisition method</td><td>est</td><td>est</td></tr><tr><td>Data Acquisition</td><td>negotiations and applications, monetary transactions State of the data when raw data, inspection, storage, access logs, generation sys- acquired tem</td><td>est</td><td>adapt</td></tr><tr><td>Data Preparation</td><td>prepared data, partially prepared data, processing pipeline, Data modifications</td><td>est</td><td>adapt</td></tr><tr><td>Data Preparation Data Acquisition</td><td>Data modifications</td><td>collaboration artifacts, operating system artifacts AI system, augmentation input raw data, inspection, storage, generation system, prepared</td><td>adapt</td><td>new</td></tr><tr><td>and Data Prepara- tion</td><td>Expected impact of data on the resulting model</td><td>data, partially prepared data, processing pipeline, collabo- ration artifacts, AI system, augmentation input</td><td>dna</td><td>new</td></tr><tr><td>Model ment</td><td>Modifications Develop- training</td><td>of training parameters, training metrics, training frame- parameters work/tools, training optimizations, final model or adapta- tion, training checkpoints</td><td>est</td><td>new</td></tr><tr><td>Model ment</td><td>and their impact Develop- Model&#x27;s intended pur- pose and capabilities</td><td>final model or adaptation, training checkpoints, training framework/tools, model code</td><td>est</td><td>new</td></tr><tr><td>Model Develop- ment</td><td>Memorization of con- tent</td><td>final model or adaptation, training checkpoints, training parameters, training metrics, training optimizations</td><td>est</td><td>new</td></tr><tr><td>Model Develop- ment</td><td>Is the model the result of model theft</td><td>final model or adaptation, training checkpoints, base model, provenance markings, training parameters, training metrics, training framework/tools, model code, dedicated</td><td>adapt</td><td>new</td></tr><tr><td>Model Integration</td><td>Role, capabilities, and functionality of AI that is part of malware</td><td>training hardware, training optimizations software, final model or adaptation (Model Development stage)</td><td>adapt</td><td>new</td></tr><tr><td>Deployment</td><td>Was the model re- placed during deploy- ment</td><td>deployment pipeline, hosting agreements, packaged soft- ware, compressed models/software</td><td>est</td><td>est</td></tr><tr><td>Deployment</td><td>Purpose of the soft- ware</td><td>packaged software, compressed model/software</td><td>est</td><td>new</td></tr><tr><td>Deployment</td><td>Target software was deployed to</td><td>packaged software, rewrites and re-implementations, de- ployment pipelines, hosting agreements, deployment con- figurations</td><td>est</td><td>adapt</td></tr><tr><td>Use</td><td>User interactions and intent</td><td>inputs, model outputs, intermediate generations, prove- nance markings, databases and storage files, log files, caches, configuration files, client server communication,</td><td>est</td><td>est</td></tr><tr><td>Use</td><td>Deep fake detection</td><td>computer use traces model outputs, provenance markings model outputs, intermediate generations, tool implementa-</td><td>est</td><td>new</td></tr><tr><td>Use</td><td>Actions performed by AI</td><td>tions, tool interactions, databases and storage files, vector databases, log files, caches, configuration files, and client- server communication, computer use traces</td><td>est</td><td>est</td></tr><tr><td>Use</td><td>Linking AI system ac- tions to user input</td><td>inputs, model outputs, intermediate generations, tool im- plementations, tool interactions, databases and storage files, vector databases, log files, caches, configuration</td><td>est</td><td>new</td></tr><tr><td>Use</td><td>Was this model stolen</td><td>files, client server communication, computer use traces inputs, model outputs, intermediate generations, prove- nance markings, log files, caches, client-server commu- nication</td><td>est</td><td>new</td></tr></table>

Table 1: Taxonomy of AI forensic investigative questions and artifact pairs. Rows that difer only in the artifact are grouped. Not included are the forensic disciplines where it is fundamentally impossible to contribute to answering the question by analyzing the artifact. dna: does not apply, est: obtainable/analyzable with established techniques, adapt: obtaining/analysis need adaptations to established techniques, new: new analysis techniques are required

XAI. AI forensics faces a similar problem for questions that require understanding how a model reaches its conclusions, such as its intended purpose and capabilities, its reliability, or whether it memorized particular content. The way models are trained makes their internal reasoning dificult to reconstruct, and such understanding is usually not required to operate the system. Explainable AI (XAI) develops methods to determine how and why a model produced a given output, but these methods have not yet been evaluated for their forensic value. Most require valid input data, which may not be available to an investigation, and they produce results with unquantified uncertainty, making them dificult to defend in court. We therefore classify these questions as requiring new analysis techniques.

Deepfake detection. Finally, deepfake detection remains an active research problem (Soundarya & Gururaj (2026); Chandra et al. (2026); Croitoru et al. (2026)), indicating that established forensic techniques do not yet solve it adequately. We therefore classify the fields concerned with deepfake detection as requiring new analysis techniques.

## 7. Discussion and Conclusion

Applying the taxonomy across all stages reveals that forensic dificulty is not evenly distributed over the life cycle of an AI system. The stages surrounding the model are largely tractable: governance and quality assurance, data acquisition and deployment mostly produce known artifacts that established methods can acquire and analyze. Dificulty concentrates in the stages closest to the model itself. Data preparation strains existing methods through sheer volume, and model development and use raise questions current techniques cannot answer at all, as they require an understanding of the model’s internal behavior that does not yet exist. AI forensics is therefore not uniformly immature: its periphery is well served, and its center is not, which suggests that research efort is best directed at the model rather than at the surrounding infrastructure.

This also shapes how investigations should proceed. Rather than starting from the model, investigators can ask which stages a system passed through and work outwards from those whose artifacts are conventional and relevant to the specific case. Governance and quality assurance is usually the most productive entry point: its artifacts are ordinary documents, audit results and version histories; they need no AI-specific techniques, and they frequently date and corroborate findings from other stages. Currently, artifacts such as the model itself are often best treated as a last resort, as they are the ones for which no reliable analysis method exists.

AI systems are usually built by more than one party, with dataset preparation outsourced, and the service hosted on rented infrastructure; the same artifacts are produced and only their location changes. That location determines which jurisdiction and legal process applies, so an artifact may or may not be obtainable depending on who holds it.

Several AI systems may also be involved in one crime, for instance, when an adversary uses one model to attack another. Where the models remain separate, they can be investigated independently. Where they collaborate or form one system, stages completed before the models were combined are investi gated separately, while later stages treat the combined system as a unit: data acquisition, data preparation and model development are examined per model, model integration and use jointly. Governance and quality assurance follows whichever party was accountable for the stage, and deployment is treated jointly if the models were deployed together.

FALC does not feature an end-of-life or decommissioning stage. Decommissioning does produce artifacts, such as deletion logs, backup expiry records and certificates of destruction, but the questions they raise are predominantly anti-forensic ones: whether data claimed to be deleted was in fact deleted, and whether residual copies remain. As anti-forensics is out of scope for this paper and warrants treatment of its own, we exclude the stage here. Investigators should nonetheless bear in mind that artifacts from any stage may have been deliberately destroyed, and we consider the forensic analysis of decommissioning a worthwhile direction for future work.

Several limitations constrain our results. FALC was derived conceptually rather than empirically and evaluated against a single case study, selected because it exercises diferent stages across the life cycle. This demonstrates applicability rather than coverage: the model can be used to structure an investigation, but its artifact catalogue is not shown to be complete for other system types. The artifact catalogue reflects current practice; new artifacts and possibly new stages will have to be added as AI develops, although the model accommodates new artifact types without restructuring. The crimes we list are illustrative rather than systematically collected. The assignment of fields to categories also involves judgment: a field requiring new domain knowledge could be argued to require adaptation instead, and we draw the line at whether an existing method yields a valid result when applied to the artifact. We expect fields to migrate between categories as adapted techniques become established, making the taxonomy a snapshot whose re-application would show where the field has matured. Finally, our sources are drawn predominantly from English-language literature and industry practice.

## 8. Conclusion

We presented FALC, an artifact-centric life cycle model for AI systems, a catalogue of the forensic artifacts each stage produces, and a taxonomy classifying the forensic fields AI investigations require by whether existing methods apply, need adaptation, or do not yet exist. Applying the taxonomy shows that forensic dificulty is concentrated in the stages closest to the model itself, while the surrounding stages are largely served by established methods. For practitioners, the catalogue ofers a structured basis for scoping what to preserve and request in a case involving AI, including artifacts that lie outside the system under examination. Together, these provide a common foundation for AI forensics, but they are only a first step. As AI systems and the techniques to analyze them continue to evolve, we expect both to be refined and extended. We hope this work encourages the forensic and AI communities to jointly build the methods, tools, and practices needed to make all parts of AI systems investigable.

## Disclosure of AI-Assisted Writing Tools

Several authors used ChatGPT, Claude, and Grammarly to help with tasks such as revising, condensing text, and addressing grammatical errors, typos, and awkward phrasing. All AIgenerated suggestions were thoroughly reviewed and adjusted where needed to ensure they accurately reflected the authors intended meaning before being incorporated into the paper.

## CRediT Authorship Contribution Statement

Adrian Johannes Staeves: Conceptualization, Methodology, Validation, Investigation, Writing - Original Draft, Visualization Mayur Rattan Jaisinghani: Writing - Review & Editing Rebekah Overdorf: Writing - Review & Editing Frank Breitinger: Conceptualization, Writing - Review & Editing, Visualization, Supervision

## Declaration of Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

Alnaqbi, A., Alblooshi, M., Nasser, H. A., Habtom, N., & Iqbal, F. (2025). Forensic Investigations in the Age of AI: Identifying and Analyzing Artifacts from AI-Assisted Crimes. In 2025 13th International Symposium on Digital Forensics and Security (ISDFS) (pp. 1–6). doi:10 1109/ ISDFS65363 2025 11012109.

Baggili, I., & Behzadan, V. (2019). Founding The Domain of AI Forensics. doi:10 48550/arXiv 1912 06497. arXiv:1912.06497.

Blauth, T. F., Gstrein, O. J., & Zwitter, A. (2022). Artificial Intelligence Crime: An Overview of Malicious Use and Abuse of AI. IEEE Access, 10, 77110–77122. doi:10 1109/ ACCESS 2022 3191790.

Brayford, D., Vallecorsa, S., Atanasov, A., Barufa, F., & Riviera, W. (2019). Deploying AI Frameworks on Secure HPC Systems with Containers. In 2019 IEEE High Performance Extreme Computing Conference (HPEC) (pp. 1–6). doi:10 1109/HPEC 2019 8916576.

Chandra, N. A., Lee, H., Murtfeldt, R., Qiu, L., Karmakar, A., Tanumihardja, E., Farhat, K., Cafee, B., Lee, C., Choi, J., Paik, S., Kim, A., & Etzioni, O. (2026). Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024. doi:10 48550/arXiv 2503 02857. arXiv:2503.02857.

Chen, L., Acun, B., Ardalani, N., Sun, Y., Kang, F., Lyu, H., Kwon, Y., Jia, R., Wu, C.-J., Zaharia, M., & Zou, J. (2023). Data Acquisition: A New Frontier in Data-centric AI. doi:10 48550/arXiv 2311 13712. arXiv:2311.13712.

Choudhary, A., Vatsa, A., Puppala, R., & Rao, A. (2026). An Empirical Study of Version-Control and Validation Workflows for Enterprise AI Systems. In 2026 35th Wireless and Optical Communications Conference (WOCC) (pp. 1– 6). doi:10 1109/WOCC69802 2026 11556187.

Comeaux, K. J., Spinosa, T. T., Ghosn, A., & Baggili, I. (2026). Ex Machina: A forensic evaluation of AI companion applications and their evidentiary value. Forensic Science International: Digital Investigation, 56, 302050. doi:10 1016/j fsidi 2026 302050.

Croitoru, F. A., Hiji, A., Hondru, V., Ristea, N.-C., Irofti, P., Popescu, M., Rusu, C., Ionescu, R., Khan, F., & Shah, M. (2026). Deepfake Media Generation and Detection in the Generative AI Era: A Survey and Outlook. ACM Comput. Surv., 58, 387:1–387:35. doi:10 1145/3833867.

Cummings, M. L. (2024). A Taxonomy for AI Hazard Analysis. Journal of Cognitive Engineering and Decision Making, 18, 327–332. doi:10 1177/15553434231224096.

DeCastro-García, N., Muñoz Castañeda, Á. L., Escudero García, D., & Carriegos, M. V. (2019). Efect of the Sampling of a Dataset in the Hyperparameter Optimization Phase over the Eficiency of a Machine Learning Algorithm. Complex ity, 2019, 6278908. doi:10 1155/2019/6278908.

Diaz, O., Kushibar, K., Osuala, R., Linardos, A., Garrucho, L., Igual, L., Radeva, P., Prior, F., Gkontra, P., & Lekadir, K. (2021). Data preparation for artificial intelligence in medical imaging: A comprehensive guide to open-access platforms and tools. Physica Medica, 83, 25–37. doi:10 1016/ j ejmp 2021 02 007.

Dragonas, E., Lambrinoudakis, C., & Nakoutis, P. (2024). Forensic analysis of OpenAI’s ChatGPT mobile application. Forensic Science International: Digital Investigation, 50, 301801. doi:10 1016/j fsidi 2024 301801.

Dunsin, D., Ghanem, M. C., Ouazzane, K., & Vassilev, V. (2024). A comprehensive analysis of the role of artificial intelligence and machine learning in modern digital forensics and incident response. Forensic Science International: Digital Investigation, 48, 301675. doi:10 1016/ j fsidi 2023 301675.

Fritsch, L., Jaber, A., & Yazidi, A. (2022). An Overview of Artificial Intelligence Used in Malware. In E. Zouganeli, A. Yazidi, G. Mello, & P. Lind (Eds.), Nordic Artificial Intelligence Research and Development (pp. 41–51). Cham: Springer International Publishing. doi:10 1007/978-3- 031-17030-0\_4.

Gopinathan, D. V. R. (2025). Software Engineering Practices for AI-Driven Systems: From Development to Deployment (MLOps Perspective). International Journal of Science, Research and Technology, 8, 13493–13500. doi:10 15662/ IJSRAT 2025 0801002.

Grover, S. (2025). ENGINEERING ROBUST AI PROD-UCTS THROUGH CONTINUOUS QUALITY ASSUR-ANCE: A FRAMEWORK FOR TESTING, MONITORING, AND VALIDATION OF ADAPTIVE LIVE LEARNING AI/ML SYSTEMS IN DYNAMIC PRODUCTION ENVI-RONMENTS. International Journal of Applied Mathematics, 38, 1092–1113. doi:10<sub>.</sub>12732/ijam<sub>.</sub>v38i2s<sub>.</sub>710.

Gruber, J., & Hilgert, J.-N. (2026). Foundations for Agentic AI Investigations from the Forensic Analysis of OpenClaw. doi:10 48550/arXiv 2604 05589. arXiv:2604.05589.

Haakman, M., Cruz, L., Huijgens, H., & van Deursen, A. (2021). AI lifecycle models need to be revised. Empirical Software Engineering, 26, 95. doi:10 1007/s10664-021- 09993-1.

Hossain, M. T., Afrin, R., & Biswas, M. A.-A. (2024). A Review on Attacks against Artificial Intelligence (AI) and Their Defence Image Recognition and Generation Machine Learning, Artificial Intelligence. Control Systems and Optimiza tion Letters, 2, 52–59. doi:10 59247/csol v2i1 73.

Jeong, D. (2020). Artificial Intelligence Security Threat, Crime, and Forensics: Taxonomy and Open Issues. IEEE Access, 8, 184560–184574. doi:10 1109/ACCESS 2020 3029280.

Jeong, S., Lee, S., & Park, J. (2025). LangurTrace: Forensic analysis of local LLM applications. Forensic Science International: Digital Investigation, 54, 301987. doi:10 1016/ j fsidi 2025 301987.

John, M. M., Holmström Olsson, H., & Bosch, J. (2021). Architecting AI Deployment: A Systematic Review of Stateof-the-Art and State-of-Practice Literature. In E. Klotins, & K. Wnuk (Eds.), Software Business (pp. 14–29). Cham: Springer International Publishing. doi:10 1007/978-3- 030-67292-8\_2.

King, T. C., Aggarwal, N., Taddeo, M., & Floridi, L. (2020). Artificial Intelligence Crime: An Interdisciplinary Analysis of Foreseeable Threats and Solutions. Science and Engineering Ethics, 26, 89–120. doi:10 1007/s11948-018-00081- 0.

Kurakin, A., Goodfellow, I., & Bengio, S. (2016). Adversarial Machine Learning at Scale. https://arxiv.org/abs/1611.01236v2.

Li, Y., & Goel, S. (2025). Making It Possible for the Audit ing of AI: A Systematic Review of AI Audits and AI Auditability. Information Systems Frontiers, 27, 1121–1151. doi:10 1007/s10796-024-10508-8.

Liao, L., Li, H., Shang, W., & Ma, L. (2022). An Empirical Study of the Impact of Hyperparameter Tuning and Model Optimization on the Performance Properties of Deep Neural Networks. ACM Transactions on Software Engineering and Methodology (TOSEM), 31, 53:1–53:40. doi:10 1145/ 3506695.

Martínez-Plumed, F., Contreras-Ochando, L., Ferri, C., Flach, P., Hernández-Orallo, J., Kull, M., Lachiche, N., & Ramírez-Quintana, M. J. (2017). CASP-DM: Context Aware Standard Process for Data Mining. https://arxiv.org/abs/1709.09003v1.

Mohammed, S., Budach, L., Feuerpfeil, M., Ihde, N., Nathansen, A., Noack, N., Patzlaf, H., Naumann, F., & Harmouch, H. (2025). The efects of data quality on machine learning performance on tabular data. Information Systems, 132, 102549. doi:10 1016/j is 2025 102549.

Nickerson, R. C., Varshney, U., & Muntermann, J. (2013). A method for taxonomy development and its application in information systems. European Journal of Information Systems, 22, 336–359. doi:10 1057/ejis 2012 26.

Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., & Dosovitskiy, A. (2021). Do Vision Transformers See Like Convo lutional Neural Networks? In Advances in Neural Information Processing Systems (pp. 12116–12128). Curran Associates, Inc. volume 34.

Roh, Y., Heo, G., & Whang, S. E. (2021). A Survey on Data Collection for Machine Learning: A Big Data - AI Integration Perspective. IEEE Transactions on Knowledge and Data Engineering, 33, 1328–1347. doi:10 1109/ TKDE 2019 2946162.

Schneider, J., & Breitinger, F. (2023). Towards AI forensics: Did the artificial intelligence system do it? Journal of Information Security and Applications, 76, 103517. doi:10 1016/j jisa 2023 103517.

Shao, Y., Huang, Y., Shen, J., Ma, L., Su, T., & Wan, C. (2025). Are LLMs Correctly Integrated into Software Systems? In 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE) (pp. 1178–1190). doi:10 1109/ICSE55347 2025 00204.

Shorten, C., Khoshgoftaar, T. M., & Furht, B. (2021). Text Data Augmentation for Deep Learning. Journal ofBig Data, 8, 101. doi:10 1186/s40537-021-00492-0.

Sibai, F. N. (2020). AI Crimes: A Classification. In 2020 International Conference on Cyber Security and Protection of Digital Services (Cyber Security) (pp. 1–8). doi:10 1109/ CyberSecurity49315 2020 9138891.

Solanke, A. A. (2022). Explainable digital forensics AI: Towards mitigating distrust in AI-based digital forensics analysis using interpretable models. Forensic Science International: Digital Investigation, 42, 301403. doi:10 1016/ j fsidi 2022 301403.

Soundarya, B. C., & Gururaj, H. L. (2026). Deepfake detection: Critical review of state-of-the-art approaches and future perspectives. Discover Applied Sciences, 8, 201. doi:10 1007/s42452-025-08174-9.

Wang, D., Chen, R., Li, C., & Gu, S. (2025). AI lifecycle from a data-driven perspective: A systematic review. Information Research an international electronic journal, 30, 159–170. doi:10 47989/ir30iConf47560.

Weber, I. (2024). Large Language Models as Software Components: A Taxonomy for LLM-Integrated Applications. doi:10 48550/arXiv 2406 10300. arXiv:2406.10300.

Yang, Y., Chen, X., Xuan, Y., Zhao, Z., Zhao, X., & Li, H. (2026). DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks. Proceedings of the AAAI Conference on Artificial Intelligence, 40, 27711–27718. doi:10 1609/ aaai v40i33 39992.

Yin, Z., Wang, Z., Xu, W., Zhuang, J., Mozumder, P., Smith, A., & Zhang, W. (2025). Digital Forensics in the Age of Large Language Models. doi:10 48550/ arXiv 2504 02963. arXiv:2504.02963.

Zhou, X., He, J., Zhou, W., Chen, H., Tang, Z., Zhao, H., Tong, X., Li, G., Chen, Y., Zhou, J., Sun, Z., Hui, B., Wang, S., He, C., Liu, Z., Zhou, J., & Wu, F. (2025). A Survey of LLM \$\times\$ DATA. doi:10 48550/arXiv 2505 18458. arXiv:2505.18458.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Govny un e unce</td><td rowspan=1 colspan=1>Data ton</td><td rowspan=1 colspan=1>Daaatna Pton</td><td rowspan=1 colspan=1>Mopel Dpent</td><td rowspan=1 colspan=1>Mmote ton</td><td rowspan=1 colspan=1>Deppomment</td><td rowspan=1 colspan=1>Use</td></tr><tr><td rowspan=1 colspan=1>AI Training Forensics</td><td rowspan=1 colspan=1>Training Process ForensicsDataset ForensicsEnvironment Forensics</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>XX</td><td rowspan=1 colspan=1>XX</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>AI Substrate Forensics</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>AI Application Forensics</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>AI Model Forensics</td><td rowspan=1 colspan=1>Model Authentication ForensicsModel Identification/Extraction ForensicsModel Ballistics ForensicsModel Performance ForensicsModel Malware Forensics</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>XXXX</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>XXX</td></tr></table>

Table A.2: The subfields defined by Baggili & Behzadan (2019) and where they fit into FALC.

## Appendix B. Applying FALC to the artifacts discovered in NTSB Highway Investigation HWY18MH010

To test the completeness of FALC we applied it to a real investigation of an AI system. We chose the NTSB Highway Investigation HWY18MH010 as it is a case where the investigators determined the responsibility of both the developer of the AI system (Uber ATG) and the person using it (the safety driver), the case gained some popularity, caused regulatory changes and the NTSB docket with 44 items is publicly available. Table B.3 shows the 30 artifacts mentioned in the docket items and where they fit into FALC.

<table><tr><td>Docket No.</td><td>AI related artifact</td><td>FALC stage</td><td>FALC artifact group</td></tr><tr><td>4</td><td>Sensors used by automation system</td><td>Governance and Quality Assurance Model Integration</td><td>documentation before the model</td></tr><tr><td>4</td><td>Sensor calibration schedule and last calibration</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>4</td><td>Procedure to plan a route</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>4</td><td>ADS engagement procedure and restrictions</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>4</td><td>ADS disengagement procedures and triggers</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>4</td><td>Object detection and classification, and path prediction</td><td>Governance and Quality Assurance</td><td>documentation</td></tr><tr><td>4</td><td>Hazard avoidance and emergency breaking pro- cedures</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>4</td><td>Table 1 (speed measurements, classifications,</td><td>Use</td><td>inputs, model outputs</td></tr><tr><td>4</td><td>and predictions) Crash history of Uber ADS cars</td><td>Governance and Quality Assurance</td><td>monitoring</td></tr><tr><td>17</td><td>Sensor measurements</td><td>Use</td><td>inputs</td></tr><tr><td>18</td><td>safety division and its culture</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>hiring procedures and standards</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Cell phone and no texting policy</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Seat belt policy</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Drug test policy</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Maximum driving time</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Fatigue policy</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Reward and disciplinary programs</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Additional policies</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Vehicle operator training</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Shifts and assigned shift</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>ADV testing procedures</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>VO&#x27;s responsibilities</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>18</td><td>Post crash internal audit</td><td>Governance and Quality Assurance</td><td>audit results</td></tr><tr><td>18</td><td>Post crash external audit</td><td>Governance and Quality Assurance</td><td>audit results</td></tr><tr><td>29</td><td>Intended Human SDV system interactions</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>29</td><td>when the SDS identified the pedestrian and its</td><td>Use</td><td>model outputs</td></tr><tr><td>37</td><td>reactions description of different policies</td><td>Governance and Quality Assurance</td><td>policies and procedures</td></tr><tr><td>42</td><td>description of camera system</td><td>Governance and Quality Assurance</td><td>documentation</td></tr><tr><td>42</td><td>captured video and sensor data</td><td>Use</td><td>inputs</td></tr></table>

Table B.3: AI system related artifacts mentioned in the NTSB Highway Investigation HWY18MH010 Docket and where they fit into FALC. Some artifacts included in this list are only very loosely related to the AI system and its development, yet they are still covered by FALC.