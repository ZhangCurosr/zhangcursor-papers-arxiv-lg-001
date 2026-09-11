# From Protocols to Evidence: Bounded Claims for AI in Service of the Common Good

Nitesh V. Chawla   
University of Notre Dame   
Notre Dame, USA   
nchawla@nd.edu

Paolo Benanti LUISS Guido Carli University Rome, Italy pbenanti@luiss.it

## Abstract

Artificial Intelligence (AI) does more than create a governance problem. It can also reveal where institutions have already failed to provide responsiveness, belonging, care, and accountability. Once deployed, AI becomes an intervention in those conditions. It can repair, compound, substitute for, or conceal the failures it encounters. Responsible AI must therefore evaluate both the system and the institutional rupture into which it is introduced.

The move from principles to protocols is already underway. The EU AI Act, NIST AI RMF, ISO/IEC 42001, and assurance practices translate commitments into roles, requirements, records, oversight, and assessment. The harder questions are what these protocols actually establish, whose power they leave untouched, and where measurement must stop.

Pope Leo XIV’s Magnifica Humanitas provides a broader moral frame centered on dignity, technological power, and the common good. Drawing on that frame, we develop a rupture test that links in stitutional baselines to system evaluation. We distinguish evidencebounded deployment, which limits claims to what has actually been evaluated, from measurement-bounded governance, which records constraints that favorable evidence cannot override. Within those limits, RISE AI provides an architecture for making bounded, evidence-based claims about (R)esponsibility, (I)nclusivity, (S)afety, and (E)mpowerment. Responsible AI requires better engineering, institutional repair, and continued moral and political judgment.

## CCS Concepts

• Social and professional topics → Computing / technology policy; • Computing methodologies → Artificial intelligence.

## Keywords

responsible AI, AI governance, institutional failure, human flourishing, construct validity, political economy, dignity, RISE AI

## ACM Reference Format:

Nitesh V. Chawla and Paolo Benanti. 2026. From Protocols to Evidence: Bounded Claims for AI in Service of the Common Good. In ACM AI Summit 2026 (AI Summit ’26), August 31-September 02, 2026, Atlanta, GA, USA. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3806096.3844885

## 1 Introduction

Artificial intelligence (AI) has become a focus for some of society’s deepest anxieties. AI is said to challenge what it means to be human, weaken human connection, displace human judgment, and demand a new moral vocabulary. These concerns matter, but this framing can grant AI too much causal power and humanity too little historical resilience. If a technology that has emerged at its current scale only recently can fundamentally unsettle the meaning of being human, then the problem may not be the technology alone. Many institutions had already begun to reduce intelligence to performance, education to what can be measured, relationships to transactions, and human worth to productivity.

AI did not create these reductions. It scales them and makes longstanding weaknesses more visible. People often turn to AI because it is available, responsive, patient, and present in ways that many institutions and relationships are not. The desire for recognition, companionship, guidance, and afirmation did not originate with AI. AI has entered spaces where communities are weaker, educational systems are overburdened, workplaces can become transactional, and civic institutions struggle to sustain connection.

AI is therefore not simply a possible rupture. It also reveals conditions that were already present in society. It exposes where institutions have failed to cultivate wisdom, sustain belonging, or provide meaningful accountability. Yet revelation is not the end of the story. Once deployed, AI becomes an intervention within the conditions it reveals. It can repair them, compound them, substitute for weakened human capacities, or conceal them behind improved performance. That makes AI both diagnostic and consequential. It reveals institutional weaknesses while also changing the institutions in which it is deployed.The central question is therefore not only whether an AI system is accurate, safe, or compliant. We must also ask what institutional weakness it is being asked to compensate for, which human relationships or capacities it may displace, and what evidence would show that the deployment has strengthened rather than weakened them.

The challenge in AI ethics is no longer simply how to move from principles to practice. The last decade produced a recognizable shared vocabulary for ethical AI, while leaving substantial variation in how its commitments are interpreted, prioritized, and implemented [6]. That vocabulary has already been translated, although unevenly, into technical standards, educational curricula and training programs, organizational management systems, assurance practices, and binding legal requirements. The unresolved problem is deciding what those commitments mean in practice: which become enforceable requirements, which can support measurable claims, what evidence is suficient, and what remains outside measurement altogether. That operationalization is neither neutral nor linear. Principles and practices are interpreted and reshaped in context [7]. Each translation also embeds choices about whose interpretation counts, which evidence is accepted, and who bears the resulting burdens.

We develop four linked claims. First, AI should be treated as both intervention and revelation. Evaluation must therefore include a rupture test: identify the institutional or relational failure preceding deployment, specify the relevant human and non-AI baseline, and determine whether the AI-mediated system repairs, compounds, substitutes for, or conceals that failure. Second, Magnifica Humanitas broadens the scope of responsible AI beyond the artifact to the institutional, economic, and political order in which it operates. Third, system design and empirical evidence cannot settle questions of dignity, ownership, political economy, or legitimate refusal. Fourth, RISE AI provides a narrower architecture for making bounded claims about responsibility, inclusivity, safety, and empowerment explicit and evidence-based. RISE AI’s purpose is simple: to make clear what is being claimed, what evidence supports the claim, who has authority, and what remains unresolved.

## 2 Human-Centered AI and the Social Order

Magnifica Humanitas and the Social Order. Pope Leo XIV’s Magnifica Humanitas frames AI as one of the res novae of our time, in continuity with the social transformations addressed by Rerum Novarum in 1891 [1, 2]. We treat the encyclical as a moral and anthropological frame, not as a technical specification. It asks us to look beyond the behavior of an AI system to the economic, institutional, and political structures through which technology is conceived, financed, controlled, and used.

The encyclical is not mainly a moral exhortation awaiting an engineering response. Chapter Five places technology within a broader culture of power, the normalization of war, autonomous weapons, and the crisis of multilateralism (Magnifica Humanitas, paras. 188–224). Earlier, the universal destination of goods is extended to patents, algorithms, platforms, technological infrastructure, and data (para. 67). The encyclical observes that control over platforms, infrastructure, data, and compute often lies with ma jor economic actors that set conditions of access and participation (para. 95), and warns that AI can amplify the power of those who already possess resources, expertise, data, and regulatory influence (paras. 106–110) [1]. Its concern is not only whether a system treats an individual fairly, but who owns the infrastructure, sets the agenda, and has the capacity to make or refuse technological futures.

This concern with political economy rests on a deeper claim about the human person. Technological innovations, including AI, are not neutral: they can foster participation and justice or intensify inequality, control, and exclusion (para. 85). Systems reflect the assumptions of those who design and train them (paras. 104, 111), while responsibility extends from developers and deployers to institutions, financiers, regulators, and users (paras. 105–111, 170, 209). Dignity is not one value to be optimized alongside others. It is an inalienable premise governing the legitimacy of institutions and the treatment of every person.

The encyclical nevertheless calls for concrete action. Paragraph 14 connects dignity and the common good to responsible planning, human and social impact assessment, inclusion of vulnerable populations, and digital literacy. Paragraph 156 calls for verifiable measures protecting employment, retraining, and worker participation when AI is introduced. Paragraph 164 calls for consequential algorithmic decisions to be understandable, contestable, and subject to oversight [1]. These passages raise concrete engineering questions about agency, contestation, provenance, and evidence. But they do not reduce the encyclical to an engineering checklist. They leave us with the question at the center of this paper: how far can moral and political commitments be translated into protocols, and what must remain outside those protocols in order to judge them?

From Commitments to Protocols. Operationalization is already underway in law, standards, assurance, and organizational practice. The following chain is a way to make that translation explicit. It does not assume that AI governance remains stuck at the level of principles:

Principle → Design Objective → System Requirement

→ Implementation Mechanism → Evaluation

Protocol → Bounded Evidence Claim.

The chain begins with the rupture test defined above: what institutional or relational failure already exists, what human or non-AI alternative is available, and which human capability is at risk of substitution? The resulting design should then be tested for whether it repairs, compounds, substitutes for, or conceals that failure. This baseline matters especially for empowerment. A claim that a system empowers users is incomplete unless we ask: compared with what alternative, for which users, according to whose definition of agency, and at what cost to others?

Consider human agency in consequential AI-mediated decisions. A design objective may be that afected people retain meaningful control. System requirements may include disclosure of AI use, understandable reasons, alternatives, override and appeal channels, human review with authority, and protection against penalty for contesting a decision. Implementation mechanisms may include review interfaces, versioned decision provenance, escalation workflows, and constraints against irreversible automated action. Evaluation then requires more than a usability test: it may combine comprehension studies, appeal and override logs, review outcomes, subgroup analyses, and audits of whether alternatives are practically available. Only then is a bounded claim possible: for a specified system version, class of users, decisions, channels, and time period, the system provided meaningful control under stated conditions.

This translation changes the unit of responsible AI from the model to the sociotechnical system [9, 10]. An accurate model can participate in an irresponsible system if no actor answers for failure. A benchmark-safe model can become unsafe through workflow integration, user overreliance, absent recourse, or weak institutional capacity. A usable interface can still disempower users who cannot understand, contest, override, or refuse its recommendations.

## 3 Design Patterns and Legal Protocols

Design patterns connect governance objectives to engineering practice. We do not claim that the following patterns are individually new; related governance, process, and product patterns have been catalogued elsewhere [11]. Their purpose here is to connect a governance objective to an implementable mechanism and then specify the evidence needed to determine whether it works in context.

Answerability-by-Design. Consequential systems should assign who answers for which decision at each lifecycle stage through role definitions, provenance, audit logs, escalation, and redress. The rupture test asks whether AI restores institutional answerability or further difuses it across vendors, deployers, and users.

Contestability-by-Design. Afected people need understandable grounds for challenge, usable appeal or refusal pathways, timely human review with authority, and protection against retaliation. Evaluation must test practical use, not merely the presence of an appeal button.

Agency-Preserving Interfaces. Interfaces should expand the capacity to understand options and act on self-determined goals. Useful friction and cognitive forcing functions can reduce overreliance on AI, while meaningful human control requires that people retain the ability and authority to act on their responsibilities [18, 19].

Context-Aware Evaluation. Benchmarks do not travel as univer sal evidence. Evaluation must match the deployment population, workflow, institution, and version. A general score may support a narrow capability claim without supporting responsibility, inclusivity, safety, or empowerment in a particular domain.

Evidence-Bounded Deployment. Deployment claims should not exceed the evidence collected. Evidence gaps, conflicts, and expiry conditions should be recorded explicitly. They should trigger qualification of the claim, additional evaluation, remediation, or withdrawal when necessary.

Across all five patterns, the governing question is the same: does the system repair, compound, substitute for, or conceal the institutional and relational fracture into which it has been introduced? A system can satisfy these technical requirements and still violate a categorical constraint or operate within an illegitimate institutional order.

EU AI Act Requirements and Mechanisms. The EU AI Act is one of the most comprehensive legal attempts to translate high-level commitments into an operational governance chain. For high-risk systems, Articles 9–15 establish requirements concerning risk management, data governance, documentation, logging, information for deployers, human oversight, accuracy, robustness, and cybersecurity. Article 27 requires specified deployers of certain high-risk systems to conduct fundamental-rights impact assessments, while Article 43 establishes conformity-assessment procedures. Articles 72–73 extend governance into post-market monitoring and seriousincident reportingt [5].

The same mechanisms reveal where translation remains incomplete. Oversight requirements do not establish that a reviewer has the time, authority, understanding, and institutional protection needed for meaningful control. Logs do not establish which claim they support or whether an appeal produces correction. Oversight requirements do not establish that a reviewer has the time, authority, understanding, and institutional protection needed for meaningful control. Logs do not establish which claim they support or whether an appeal produces correction. The Act gives responsible AI a legal and institutional framework. But compliance alone does not establish empowerment, institutional repair, or a just distribution of technological power. That is the narrower problem an evidence architecture such as RISE is intended to address.

## 4 Limits of Design and Measurement

A design-centered approach still has two important limits. Measurement cannot resolve either one. Being explicit about those limits is essential if the claims we make are to remain credible.

Political Economy and System Boundaries. The shift from the model to the sociotechnical system is an important advance. But even the deployment context is not the full object of governance.An evaluation architecture may establish that a deployed system preserves meaningful control for its users, ofers practicable contestation, and communicates uncertainty well, while saying nothing about who owns the compute on which it runs, who set the agenda under which it was built, whose labor and resources sustain it, or which asymmetries between institutions and populations it reproduces [8].

Magnifica Humanitas makes these questions part of the object of governance. It treats algorithms, platforms, infrastructure, and data as goods whose concentration can violate their universal destination; identifies private control over data and compute as a source of political and economic asymmetry; and asks who can train and govern systems and who is merely subjected to them (paras. 67, 95, 106–110) [1]. A validity chain can therefore establish that a system is well designed within its deployment context while saying nothing about whether the larger distribution of power is just. Measurement of the artifact is not measurement of the order the artifact serves. Ownership, financing, labor, resource consumption, agenda-setting, and regulatory influence must remain visible fields of governance even when they cannot be reduced to system-performance indicators.

Dignity and Construct Validity. The second failure concerns the status of dignity in the translation chain. Construct validity presupposes an object whose variation can be observed and whose indicators can be more or less adequate to it. Dignity itself is neither a variable nor an indicator. Observable conditions may provide evidence that dignity has been respected or violated, but they do not measure a person’s possession of dignity or determine its weight against competing outcomes. Dignity is what allows us to say that a system can perform well on every selected measure and still treat a person merely as a means.

The distinction is between inherent human status and contingent sociotechnical conditions. Dignity and human worth do not vary with system performance; no indicators can establish, increase, or ofset them. By contrast, conditions relevant to agency, recourse, access, safety, and answerability vary across populations, institutions, workflows, and system versions. RISE does not measure dignity, empowerment, responsibility, inclusion, or safety as latent attributes, nor does it assign a comprehensive moral score. It evaluates whether specified evidence supports bounded claims about variable conditions. Validity attaches to the inference from evidence to claim, not to the normative value itself [15–17].

Table 1: Evidence-bounded claims and measurementbounded constraints.
<table><tr><td></td><td>Evidence-bounded claims</td><td>Measurement-bounded straints</td></tr><tr><td>Question</td><td>What does evidence support here?</td><td>What may not be authorized?</td></tr><tr><td>Object</td><td>Variable conditions: agency, recourse, access, safety, answerability.</td><td>Dignity, legitimate refusal, prohibited uses, non-substitutable relationships.</td></tr><tr><td>Evidence</td><td>Supports, qualifies, defeats, or expires a claim.</td><td>May reveal a breach; cannot override the boundary.</td></tr><tr><td>Authority</td><td>Predeclared protocol and accountable decision process.</td><td>Law, policy, affected communities, in- stitutional mission, moral judgment.</td></tr><tr><td>Decision</td><td>Support, qualify, remediate, retest, or withdraw.</td><td>Prohibit, stop, redesign, or require a human alternative.</td></tr><tr><td>RISE role</td><td>Evaluates and records evidentiary support.</td><td>Records the boundary, basis, author- ity, and revision process.</td></tr></table>

Measurement-Bounded Governance. We therefore propose a counterpart to evidence-bounded deployment: measurement-bounded governance. Just as a governance claim should be bounded to the evidence collected, an evidence profile should carry an explicit declaration of commitments that do not depend on evidence for their force: uses excluded regardless of performance, actions no benchmark result can license, populations whose exclusion cannot be ofset by aggregate gains, and human relationships or capabilities that a deployment may not substitute away.

These commitments are constraints, not constructs to be measured. But they should still be explicit and reviewable. A design record can identify the constraint, its normative or legal basis, the persons or communities it protects, the actor authorized to interpret it, and the process required to revise it. Article 5 of the EU AI Act illustrates this logic by prohibiting specified practices rather than inviting a favorable risk-benefit score to legitimate them [5]. Table 1 formalizes the distinction. RISE evaluates evidence-bounded claims and records measurement-bounded constraints; legitimate legal, political, institutional, or community authorities establish those constraints.

This sharpens the paper’s central claim. AI does not demand better design in place of moral and political judgment. It demands both. Better design is necessary because moral commitments must shape how systems actually behave. Moral and political judgment remains necessary because AI redistributes agency and responsibility, raising questions of authorship, accountability, and power that design alone cannot settle. The engineering and moral-political agendas must move forward together. Technical benchmarks cannot establish moral legitimacy, just as ethical ideals cannot substitute for system design.

## 5 RISE AI Evidence Architecture

RISE addresses a narrower operational question: when institutions make claims about a particular AI system, what evidence supports those claims, and where should those claims stop? It does not seek to translate the encyclical as a whole into technical requirements, nor does it claim authority over the deeper moral questions the encyclical raises.

RISE organizes bounded system claims around four domains: Responsibility, Inclusivity, Safety, and Empowerment. Each begins with a simple question: Who answers for the system? Whose perspectives shape it and whose needs does it serve? Whom does it protect from harm? Who gains or loses agency through it? Empowerment, for example, is not a measurable trait or a general claim of benefit. It asks whether people gain durable capability, practical choice, contestation, and access to human and institutional support relative to an explicit baseline. Each claim passes through a validity chain:

Normative Domain → Bounded System Claim → Indicator → Evidence Source → Bounded Inference.

RISE borrows one important discipline from measurement theory: validity belongs to the inference drawn from evidence, not to the metric or moral value itself [15–17]. RISE is designed to complement, not replace, existing governance frameworks. NIST AI RMF, ISO/IEC 42001, and the EU AI Act supply lifecycle outcomes, management requirements, and legal obligations [3–5]. Responsible AI pattern catalogues supply reusable governance, process, and product practices [11]. Assurance cases structure claims, arguments, and evidence [12]. Model and data documentation, provenance records, and operational logs provide candidate evidence [12–14]. A standard supplies requirements; a pattern supplies a candidate mechanism; an operational system supplies an artifact; RISE records what claim the artifact supports and where that support stops.

The primary artifact is a versioned claim-evidence graph, not a universal governance score. A minimal record links a bounded claim to the system, model, data, and interface version; population and context; accountable owner; indicator and threshold; evidence provenance; limitations; and expiry or change trigger. Evidence can be direct, proxy-based, conflicting, missing, or stale.

Each graph is also linked to a context-and-power record: ownership and control of models, data, compute, and platforms; financing and procurement relationships; labor and resource dependencies; agenda-setting authority; and the distribution of risks and benefits. These are not additional indicators from which a political-economy score is calculated. They make visible the institutional order within which a supported claim is made and may supply constraints that qualify or preclude deployment. RISE cannot determine through measurement whether that institutional order is just. It can, however, make clear that system-level evidence does not establish the legitimacy of that order.

Operational Protocol and Institutional Baseline. RISE can operate as a shared schema and decision protocol layered over existing requirements repositories, registries, observability tools, audit systems, user research, and redress workflows. The process has six stages. Teams first scope the system, population, baseline, institutional rupture, ownership, procurement, infrastructure, and decision authority. They then formulate bounded claims with afected communities, domain experts, and system owners, while preserving disagreements. Before evaluation, they predeclare indicators, methods, thresholds, unacceptable gaps, constraints, and change triggers. They then collect evidence through tests, provenance records, logs, user studies, case audits, appeals, incidents, and recovery processes. Evidence is classified as direct, proxy-based, conflicting, missing, or stale. Finally, deployment or procurement is revisited when the system or its institutional conditions materially change.

Thresholds are not universal: what counts as timely review or suficient comprehension depends on context. RISE instead makes the choice, rationale, and decision authority explicit and contestable. Its claim template is: for this system version, population, context, and period, evidence supports this bounded claim under these conditions; it does not establish these unevaluated propositions; and it expires upon these changes.

The rupture test is comparative, not merely metaphorical. At scoping, the team documents the preexisting failure, available human and non-AI alternatives, institutional capacity, and capability at risk of substitution. Evaluation then compares the AI-mediated system with that baseline and tests both intended repair and plausible displacement. Faster throughput coupled with reduced access to a caseworker, for example, may defeat an empowerment claim even when accuracy improves. A repair claim requires evidence that the relevant human or institutional capability became more available, durable, or answerable; evidence of substitution, burden shifting, or suppressed visibility qualifies or defeats it.

Public-Benefits Example. Suppose an AI system supports publicbenefits eligibility decisions in an institution already marked by slow processing, limited caseworker capacity, and dificult appeals. Faster decisions alone may conceal rather than repair that rupture. Table 2 instantiates one RISE chain for an empowerment claim expressed through meaningful contestability of adverse recommendations.

Table 2: A compact RISE evidence contract for contestability.
<table><tr><td>Element</td><td>Public-benefits instantiation</td></tr><tr><td>Claim</td><td>Applicants can meaningfully contest adverse AI- mediated recommendations.</td></tr><tr><td>Requirements</td><td>AI disclosure; understandable reasons; accessible on- line and offline appeal; authorized human review; non- retaliation; correction of downstream effects.</td></tr><tr><td>Mechanisms</td><td>Decision notice; reason codes; appeal endpoint; review queue; adverse-action pause; correction and recovery workflow.</td></tr><tr><td>Evidence</td><td>Versioned decision, notice, appeal, review, incident, complaint, and recovery logs; accessibility and com-</td></tr><tr><td>Indicators</td><td>prehension studies. Notice comprehension; appeal initiation and comple- tion; abandonment; review latency; reversal; subgroup</td></tr><tr><td>Boundary</td><td>disparities; recovery completeness. Specified system version, jurisdiction, channels, lan-</td></tr><tr><td>Context &amp; power</td><td>guages, populations, decisions, and evaluation period. System and data owner; vendor and cloud dependen- cies; procurement terms; eligibility-policy authority;</td></tr><tr><td>Constraints</td><td>caseworker staffing effects; control of appeal records. No irreversible adverse action without authorized hu- man review; no online-only appeal; no retaliation for contesting a decision; no substitution away of practi-</td></tr></table>

Logs support audit and recovery but do not establish contestability without user studies, case-file audits, subgroup coverage, and human reviewers with practical authority. The resulting claim must state what is supported and what remains unknown, such as phoneonly applicants, additional languages, or another jurisdiction. The context-and-power row prevents system evidence from obscuring ownership or institutional capacity; the constraints row records conditions that favorable aggregate performance cannot ofset.

Cross-Domain Probes. In education, completion and answer accuracy do not establish empowerment; evidence should address learning transfer, unaided performance, student authorship, mentor availability, and the ability to question or refuse guidance. In clinical decision support, predictive performance does not establish sociotechnical safety; evidence must join model evaluation to responsibility allocation, usable override, clinician observation, subgroup outcomes, incidents, correction, and recovery. In both domains, claims remain bounded to the evaluated population, institution, workflow, version, stafing conditions, and period. These probes illustrate portability, not validation.

## 6 Research Roadmap

The next phase should test the architecture rather than elaborate it. We see seven priorities for that next phase. Participatory content validity asks whether afected communities, domain experts, and responsible institutions judge bounded claims to cover what matters and where their definitions diverge. Inter-evaluator reproducibility tests whether independent evaluators classify the same evidence, gaps, and inference boundaries similarly. Consequential validity examines whether profiles reveal unsupported claims or failures missed by benchmarks and compliance review. Change sensitivity asks whether claims are invalidated appropriately when models, data, interfaces, workflows, populations, ownership, procurement, or institutional capacity change. Decision utility tests whether evidence gaps and constraints alter procurement, deployment, remediation, withdrawal, or appeals. Political-economic sensitivity examines whether the context-and-power record exposes dependencies or distributions of risk and benefit that system-level assessment misses and whether those disclosures change decisions. Operational burden and proportionality measures the expertise, time, infrastructure, and cost required, including whether credible use remains possible beyond large organizations.

These programs require field studies with regulators, providers, deployers, workers, and afected populations rather than treating compliance artifacts as self-interpreting. Comparative studies can test whether AI Act logs, impact assessments, oversight measures, conformity procedures, and post-market records support the inferences these groups need. Decision studies should compare choices made with and without a RISE profile; a framework that improves documentation without changing consequential decisions would have limited practical value.

Limitations. The worked trace and cross-domain probes are proof-of-concept specifications, not empirical validation of RISE or an implementation study of the AI Act. Claim formulation and threshold setting still require normative judgment and are shaped by institutional power. Evidence may underrepresent those most afected, while logging and provenance can create privacy and surveillance risks. Political-economic analysis can become superficial if ownership and infrastructure are merely added as fields. A supported bounded claim is not equivalent to moral legitimacy, legal compliance, or respect for dignity. RISE can make evidence, power, and limits visible; it cannot resolve them by measurement alone.

## 7 Conclusion

AI is both intervention and revelation. It does not enter a social vacuum: it exposes failures of responsiveness, belonging, care, and accountability, and then acquires the capacity to repair, compound, substitute for, or conceal them. Responsible AI must therefore evaluate the system and the institutional rupture into which it is introduced.

The movement from principles to protocols is already underway. The EU AI Act, NIST AI RMF, ISO/IEC 42001, assurance practices, and operational systems demonstrate that values and rights can become roles, requirements, records, assessments, and decision gates. The next challenge is not simply to produce more protocols. It is to determine what their evidence actually warrants and what those protocols leave out.

Magnifica Humanitas shows why design cannot be self-suficient. It directs attention beyond the system and its immediate deployment to ownership, infrastructure, financing, labor, political authority, and the distribution of technological power. It also makes a crucial distinction: dignity is not another outcome to be traded against performance. Dignity governs evaluation; it is not produced by it.

RISE AI ofers a practical response within this broader account of dignity, power, and the common good. It makes claim-evidence relations explicit, links them to a context-and-power record, and records constraints established by legitimate authorities. Evidencebounded deployment restricts claims to what has been evaluated; measurement-bounded governance identifies what favorable evidence cannot override. Together, they make engineering more accountable while recognizing that measurement cannot confer moral or political legitimacy.

Responsible AI must ask whether a deployment repairs the fractures it reveals, compounds them, substitutes for the relationships and capacities they have weakened, or conceals them behind improved performance. It must also ask whether the institutional order surrounding that deployment should itself be sustained. The engineering agenda and the moral-political agenda must proceed together. The question is not whether machines will become more human. It is whether humans will build institutions and systems worthy of humanity.

## References

[1] Pope Leo XIV. 2026. Magnifica Humanitas: On Safeguarding the Human Person in the Time of Artificial Intelligence. Vatican.

[2] Pope Leo XIII. 1891. Rerum Novarum: On Capital and Labor. Vatican

[3] Elham Tabassi. 2023 Artificial Intelligence Risk Management Framework (AI RMF 1.0). National Institute of Standards and Technology.. 10.6028/NIST.AI.100-1.

[4] International Organization for Standardization. 2023. ISO/IEC 42001:2023 Information Technology — Artificial intelligence — Management system. ISO.

[5] European Union. 2024. Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence. Oficial Journal ofthe European Union.

[6] Anna Jobin, Marcello Ienca, and Efy Vayena. 2019. The global landscape of AI ethics guidelines. Nature Machine Intelligence 1, 9, 389–399.

[7] Lorenn P. Ruster and Jenny L. Davis. 2025. The gaps that never were: Reconsidering responsible AI’s principle-practice problem. In Proceedings ofthe 2025 ACM Conference on Fairness, Accountability, and Transparency, 350–360.

[8] Kate Crawford. 2021. Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence. Yale University Press.

[9] Andrew D. Selbst, Danah Boyd, Sorelle A. Friedler, Suresh Venkatasubramanian, and Janet Vertesi. 2019. Fairness and abstraction in sociotechnical systems. In Proceedings ofthe Conference on Fairness, Accountability, and Transparency, 59–68.

[10] Inioluwa Deborah Raji, Andrew Smart, Rebecca N. White, Margaret Mitchell, Timnit Gebru, Ben Hutchinson, Jamila Smith-Loud, Daniel Theron, and Parker Barnes. 2020. Closing the AI accountability gap: Defining an end-to-end framework for internal algorithmic auditing. In Proceedings ofthe Conference on Fairness, Accountability, and Transparency, 33–44.

[11] Qinghua Lu, Liming Zhu, Xiwei Xu, Jon Whittle, Didar Zowghi, and Aurelie Jacquet. 2024. Responsible AI Pattern Catalogue: A collection of best practices for AI governance and engineering. ACM Computing Surveys 56, 7, Article 173, 1–35.

[12] Alpay Sabuncuoglu, Christopher Burr, and Carsten Maple. 2025. Justified evidence collection for argument-based AI fairness assurance. In Proceedings ofthe 2025 ACM Conference on Fairness, Accountability, and Transparency, 18–28.

[13] Karl Werder, Balasubramaniam Ramesh, and Sophia Rongen Zhang. 2022. Establishing data provenance for responsible artificial intelligence systems. ACM Transactions on Management Information Systems 13, 2, Article 22, 1–23.

[14] Patrick Loic Foalem, Leuson Da Silva, Foutse Khomh, Heng Li, and Ettore Merlo. 2025. Logging requirement for continuous auditing of responsible machine learning-based applications. Empirical Software Engineering 30, Article 97.

[15] Lee J. Cronbach and Paul E. Meehl. 1955. Construct validity in psychological tests. Psychological Bulletin 52, 4, 281–302

[16] Samuel Messick. 1995. Validity of psychological assessment: Validation of inferences from persons’ responses and performances as scientific inquiry into score meaning. American Psychologist 50, 9, 741–749.

[17] Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and fairness. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, 375–385.

[18] Zana Buçinca, Maja B. Malaya, and Krzysztof Z. Gajos. 2021. To trust or to think: Cognitive forcing functions can reduce overreliance on AI in AI-assisted decision-making. Proceedings ofthe ACM on Human-Computer Interaction 5, CSCW1, 1–21.

[19] Luciano Cavalcante Siebert, Maria Luce Lupetti, Evgeni Aizenberg, Niek Beckers, Arkady Zgonnikov, Herman Veluwenkamp, David Abbink, Elisa Giaccardi, Geert-Jan Houben, Catholijn M. Jonker, Jeroen van den Hoven, Deborah Forster, and Reginald L. Lagendijk. 2023. Meaningful human control: Actionable properties for AI system development. AI and Ethics 3, 1, 241–255.