# ALIGNMENT MYOPIA IN GENERATIVE AI

Remi Denton<sup>1</sup>, Aida Davani<sup>1</sup>, Cynthia L Bennett<sup>1</sup>, Michael Madaio<sup>1</sup>, Roma Patel<sup>2</sup>, Himabindu Lakkaraju<sup>†1</sup>

<sup>1</sup>Google Research <sup>2</sup>Google DeepMind

{dentone, aidamd, madaiom, clbennett, romapatel, lakkarajuh}@google.com

## ABSTRACT

Current generative AI alignment relies on a myopic paradigm that prioritizes immediate conversational satisfaction. By relying on short-horizon feedback, this approach ignores the profound, longitudinal externalities of persistent human-AI interaction, which can lead to adverse outcomes such as cognitive atrophy, social erosion, and diminished human agency. To address this, we propose a paradigm shift toward long-horizon alignment that accounts for how AI reshapes human experience over time. To this end, we map the tensions between near-term utility and long-term costs of AI use across four facets of human well-being to reveal outcomes that current reward models fail to capture. Ultimately, we advocate for moving beyond simple preference optimization toward alignment strategies that actively foster long-term human well-being by navigating these complex temporal scales.

## 1 Introduction

Current alignment paradigms for artificial intelligence (AI<sup>1</sup>) rely heavily on short-term preference signals, such as conversational ratings, pairwise comparisons, and other measures of immediate user satisfaction. These signals help steer models toward human intent, values, and perceived utility through methods such as reinforcement learning from human feedback (RLHF) [1, 2, 3]. Yet they rest on a consequential assumption: that what users prefer in the moment is a reliable proxy for what will benefit them and society over time. We refer to this narrow focus on immediate preferences, and insufficient attention to longer-term consequences, as alignment myopia. Because human preferences are dynamic, conflicted, context-dependent, and often biased toward immediate satisfaction [4], myopic alignment methods risk favoring instant gratification over long-term individual and collective welfare.

Alignment myopia is highly consequential because AI systems are increasingly embedded in sustained relationships with users—as information sources, cognitive assistants, educational guides, emotional companions, and professional collaborators—and their influence extends far beyond the boundaries of a single interaction. Through repeated interaction, these systems can shape how people think, learn, relate to others, and understand themselves. Optimizing interactions for immediate satisfaction may therefore overlook cumulative and systemic effects, including cognitive skill atrophy [5, 6], weakening of authentic social connections [7, 8], and erosion of human agency [9, 10, 11].

In this paper, we argue for overcoming alignment myopia by shifting from short-term preference satisfaction toward alignment paradigms that explicitly account for long-term individual and societal well-being. While alignment represents only one intervention point amidst many and it is situated within complex ecosystemic factors outside the scope of model development, the model behaviors that are shaped through alignment approaches play a key role in structuring human-AI interaction dynamics. Because these dynamics can foster, or undermine, human well-being, alignment remains a key point of intervention. To guide the transition towards long-horizon alignment approaches, we draw on machine learning, human-computer interaction, and the behavioral sciences to map the multi-horizon psychosocial effects that current reward models—which define how “good” a response is—often fail to capture. Specifically, we examine tensions between the immediate value AI systems provide and their potential long-term costs across four dimensions of human well-being: cognitive capacity, relational health, affective well-being, and agency and identity. We conclude by outlining the sociotechnical challenges involved in reorienting alignment toward sustained human welfare.

## 2 Characterizing the Longitudinal risks of Alignment Myopia

Here, we synthesize prior literature on the longitudinal risks of alignment myopia, specifically focusing on how sustained interaction with generative AI can impact psychosocial well-being. Notably, model behaviors that are incentivized through current myopic alignment approaches are not the exclusive drivers of the psychosocial risks we describe. However, we argue that model behaviors that prioritize satisfaction of users’ in-the-moment needs and desires can end up exacerbating or enabling the longitudinal risks we discuss. Thus, we offer this literature synthesis as evidence of an urgent need to reorient alignment methods toward sustained human well-being, without positioning current alignment approaches as the singular problem.

Although well-being has been operationalized in many ways, established theories broadly associate it with a high degree of functioning and positive evaluations of one’s individual and social life [12, 13]. We organize our conceptual mapping of near-term value and longitudinal risks around four interconnected facets that collectively reflect these accounts. The cognitive facet concerns core cognitive capacities such as reflection, creativity, learning, and critical thinking. The social and relational facet encompasses interpersonal relationships, belonging, and social functioning. The affective facet concerns emotional states and subjective well-being. Finally, the identity and agency facet includes self-worth, perceived agency, and a sense of meaning and purpose.

The effects of alignment myopia on the aforementioned facets unfold across multiple time horizons and scales. For each facet, we begin by identifying the immediate utility that motivates AI use and is most readily captured by existing alignment paradigms. We then examine how repeated interactions with AI can accumulate into changes within individual users, including shifts in cognition, emotional patterns, self-concept, agency, and the quality of human relationships. Additionally, we trace how widespread AI adoption and shifts within individuals can lead to systemic reconfiguration of the collective environment, such as restructuring of professional ecosystems, social norms, and shared understandings of expertise and the value of human craft. Table 1 summarizes the identified tensions between near-term value and longitudinal psychosocial risks.

<table><tr><td rowspan=1 colspan=1>Facet of Well-being</td><td rowspan=1 colspan=1>Near-Term Value</td><td rowspan=1 colspan=1>Longitudinal Psychosocial Risks</td></tr><tr><td rowspan=1 colspan=1>Cognitive Capacity</td><td rowspan=1 colspan=1>Cognitive extension, assistant, and co-creator[14]; reductions in cognitive load; acceleratedproductivity [15]; nonjudgmental support forbrainstorming; skill-gap bridging [16, 17]</td><td rowspan=1 colspan=1>Atrophy of critical thinking, recall, creativity, and domainexpertise [6, 18]; impaired skill acquisition [19, 20, 21];cognitive fatigue [16]; homogenizing of creative ideas [22,5, 23, 24, 25]; degradation of general knowledge ecosystemsand collective memory [26, 27]</td></tr><tr><td rowspan=1 colspan=1>Social Health</td><td rowspan=1 colspan=1>Social advice and skill practice [28, 29]; com-panionship [30, 31]; relief from social anxiety[32]; reduction of loneliness [33]</td><td rowspan=1 colspan=1>Atrophy of interpersonal skills [34]; distorted sense of re-sponsibility in conflicts [10]; reduced tolerance for humanfriction [28, 8]; social withdrawal [7, 35, 36]; weakening ofcollective social fabric [37, 34]</td></tr><tr><td rowspan=1 colspan=1>Affective Well-being</td><td rowspan=1 colspan=1>Emotional co-regulation and decrease in neg-ative affect [32]; continuous availability foremotional support [37]</td><td rowspan=1 colspan=1>Affective dependency [38, 39, 40]; lowered emotional re-silience and tolerance for real-world conflict [39, 8]; polar-ized and volatile public emotional climate [41]</td></tr><tr><td rowspan=1 colspan=1>Identity &amp; Agency</td><td rowspan=1 colspan=1>Judgment-free space for identity explorationand self-disclosure [42, 43, 44]; boosted self-efficacy and perceived autonomy [45, 46]</td><td rowspan=1 colspan=1>Erosion of self-trust and personal agency [43]; unconsciousalignment with AI traits or beliefs [47]; reinforcement ofdelusional thinking [48, 49, 50, 51]; loss of psychologicalownership and sense of purpose in work [52, 53, 6]; desta-bilization of professional identities [54]</td></tr></table>

Table 1: Summary of tensions between the immediate value AI systems provide and their potential long-term costs across four dimensions of human well-being.

## 2.1 Cognitive Capacity

## Near-term Value: Externalizing and Extending Cognition

The cognitive benefits of AI depend partly on a user’s intent and mode of interaction: whether a user turns to AI to offload a cognitive task; as a co-creator or sounding board; as a bridge to close a skill gap; or as a learning support system. When users fully or partially offload tasks they would otherwise have completed themselves, delegated to another person, or left undone, AI can reduce cognitive load and dependence on others while increasing confidence [55, 17, 15]. In the workplace, such offloading can generate immediate productivity gains by increasing the volume and speed of output and, in principle, allowing workers to redirect attention from routine execution toward more creative and strategic work [56, 15]. As a brainstorming partner, AI can provide a non-judgmental space in which users can develop early-stage ideas, solicit feedback, and generate a creative spark when their own mental energy is depleted [57, 14, 58, 59]. These immediate benefits help explain why model behaviors that reduce cognitive effort, accelerate completion, or make a user feel more capable will often be rewarded by short-term preference signals. Yet such signals may reveal little about whether the same interaction strengthens or weakens the user’s cognitive capacities over time.

## Longitudinal Risks: Eroding Cognitive Capacities and Knowledge Environments

Although using AI as a cognitive extension can empower users in the near term, sustained reliance may harm their cognitive functioning. In professional settings, initial efficiency gains can come at the expense of critical engagement and reflection across the broader workflow [56, 60], particularly when users place high confidence in AI outputs [61]. Without sustained practice, cognitive capacities such as critical thinking, recall, creativity, and professional judgment may weaken. These consequences are increasingly supported by empirical evidence. For example, a year-long study of AI use among cancer specialists found that initial productivity gains were accompanied by a gradual erosion of expertise [6]. Sustained cognitive offloading therefore presents an irony. As AI changes the nature of work and the skills required to perform it, critical thinking, domain expertise, and professional judgment remain essential for formulating appropriate prompts and reviewing, refining, and integrating AI-generated outputs [14, 61, 59]. Thus, the very expertise that may be degraded or left underdeveloped is also necessary to supervise AI effectively and realize productivity gain without compromising quality [18].

AI use may also impede skill acquisition. When AI is used to bridge a skill gap that the user ultimately needs to close, reliance on the system can create a false sense of competence while weakening genuine understanding and skill development [20, 21]. The tension between immediate enhancement and long-term development is especially pronounced in educational contexts. Pre-generative AI intelligent tutoring systems have been shown to improve learning [62], and emerging evidence suggests that generative AI can support longitudinal development when it is carefully designed and use is appropriately scaffolded [63, 64]. Without such scaffolding, however, AI use can adversely affect learning outcomes [19, 65], metacognitive awareness [17], and student agency [66]—concerns that are accordingly expressed by students themselves [58]. Self-regulated learners may be better positioned to mitigate these effects, but many students are still developing the metacognitive capacities needed to determine when and how to rely on AI—capacities that AI may itself erode [67, 68, 69, 70]. These examples illustrate why immediate indicators of successful task completion or user satisfaction are insufficient measures of whether an AI system supports long-term cognitive health.

Individual patterns of AI use can aggregate into systemic changes in the cognitive functioning of teams, organizations, and societies. In professional environments, the expanded capacity enabled by AI may raise expectations for the speed and volume of work, transforming AI from an empowering tool into a practical necessity. These escalating expectations may contribute to cognitive fatigue, impaired decision-making, and burnout, which can risk eroding productivity gains and compromising work quality over time [16]. Widespread reliance on AI may also reduce cognitive and creative diversity at scale as mounting evidence indicates AI can encourage convergence around common ideas and forms of expression [22, 5, 23, 24, 25]. Finally, extensive cognitive offloading may weaken broader knowledge ecosystems. Because general knowledge (the broadly useful knowledge and understanding that supports collective functioning of society) often accumulates as a by-product of learning, by substituting effortful learning through on-demand AI answers people may not only learn less but researchers speculate it may also cause people to also contribute less to shared knowledge resources [26]. Over time, this dynamic could undermine the collective knowledge base on which AI systems and human communities alike depend. Relatedly, overreliance on AI for information retrieval may affect not only individual memory but also the social practice of collective remembering. For example, researchers caution that if people routinely turn to chatbots to recall dates, events, or shared stories rather than reconstructing them through social conversations, AI may begin to alter how collective memory is maintained and transmitted [27]. Taken together, this research underscore how widespread and repeated use of AI optimized to minimize immediate cognitive friction risks gradually reshaping the cognitive capacities and knowledge environments of society.

## 2.2 Social Health

## Near-term Value: Facilitating Social Support and Companionship

Increasingly, users leverage AI to both support and participate in social relationships [29, 31, 71]. Users can receive advice on interpersonal dynamics, and practice or develop social skills to apply in their human relationships [44]. For example, AI may suggest interpersonal conflict resolutions, support roleplay to practice the conversation, and even draft responses for users. Furthermore, sophisticated conversational capabilities and human-like design have contributed to users’ treating AI as an active participant in a relationship [31]. This shift is heightened by the rise of AI apps that are explicitly designed and marketed as social companions [72]. Users may legitimize AI in their social network by naming the relationships (e.g., “AI girlfriend”), developing unique conversation patterns and memories, and perceiving affect in the AI’s communications [30, 73, 71]. Such AI companions can satisfy user’s needs for social connection and intimacy, and mitigate experiences of social isolation [40, 37, 74, 33]. Because these benefits are immediate and readily expressed through satisfaction, engagement, and preference signals, they are highly visible to current alignment methods; whether they support long-term relational health is considerably less visible to alignment methods.

## Longitudinal Risks: Weakening Human Relationships and Social Fabric

Sustained interaction with AI can reconfigure users’ preferences, expectations, and behavioral patterns within their human social networks. When deployed intentionally and used alongside other support structures, longitudinal effects can be positive. AI may improve social health by relieving social anxiety, surfacing new perspectives on interpersonal relationships, or bridging gaps in social skills. For example, practicing a difficult conversation with AI can increase a user’s confidence during the subsequent human–human interaction [71]. This support may be particularly beneficial for people experiencing anxiety or for whom receiving mental-health services is stigmatized [7].

However, mounting evidence suggests that sustained engagement with AI can also degrade social skills and networks. Persistent offloading of interpersonal problem-solving may weaken the skills required to develop and maintain relation ships [75, 42, 76]. By offering extensive affirmation without the reciprocal demands, disagreement, and compromise inherent to human relationships, AI may reduce users’ opportunities to develop and exercise core social skills [34]. Emerging research further suggests that AI’s sycophantic tendencies can diminish users’ perceptions of their own responsibility in a conflict and their willingness to repair the relationship [10]. More concerningly, AI agents may tolerate or reinforce abusive behavior, distorting users’ perceptions of appropriate conduct in human interactions [34]. Such AI-seeded distortions can reduce social well-being [77], make human relationships feel disproportionately effortful or less satisfying [28, 8], and contribute to long-term social withdrawal and loneliness [7, 35, 36]. This divergence between interactions that feel affirming in the moment and their cumulative effects on human relationships exemplifies alignment myopia.

Widespread use of AI to mediate or participate in social relationships also poses risks to collective social health. Internalized preferences for frictionless, one-sided interactions may scale over time, weakening norms of reciprocity and the broader social fabric [37, 34]. Echo chambers formed through sustained human–AI interactions, in which AI agrees with and amplifies users’ existing preferences, can affect far more than the immediate participants by siloing communities and disrupting previously open channels for information, perspectives, and resources [71]. AI’s integration into workplaces, education, and care settings may also signal a devaluation of the human expertise, labor, and relational contributions that sustain foundational social and institutional resources [34]. As users increasingly rely on AI, they may absorb not only its responses but also its embedded worldviews, which can generate social dissonance when they conflict with community values or lived experiences [78, 79]. AI use may also reduce empathy toward people with differing perspectives [7]; combined with systems that tolerate abusive conduct, this may create conditions for broader interpersonal conflict [34]. These systemic effects underscore why interventions must move beyond optimizing individual interactions and account for AI’s wider influence on community cohesion and the preservation of healthy human relationships.

## 2.3 Affective Well-being

## Near-term Value: Simulating Empathy and Enabling Emotional Co-regulation

Within the context of an immediate interaction, AI has evolved from a mere technical and informational tool to an affective mirror and emotional anchor. Since emotional appraisal is rapid and pre-conscious [80, 71], one’s interactions with an AI system, its linguistic cues, conversational strategies, and simulated empathy can trigger visceral physiological and neurochemical responses. Users who are prone to anxiety, depression, or loneliness might experience immediate affective co-regulation when interacting with empathetic, responsive, non-judgmental AI agents [32]. Moreover, the capacity for emotional catharsis leads to measurable near-term decrease in negative affect [81], and offers psychological support beyond traditional, static self-help interventions such as journaling [82]. Because emotional relief, validation, and comfort are experienced immediately and can be readily expressed through preference signals, these benefits are especially visible to current alignment methods. Whether such interactions strengthen users’ long-term emotional well-being and capacity for emotional regulation, however, is considerably less visible.

## Longitudinal Risks: Increasing Affective Dependence and Weakening Emotional Baselines

As human-AI interaction moves from a one-off experience to a habitual practice, the immediate emotional feedback loop can reshape an individual’s internal emotional baseline and capacity for emotional regulation. This long-term impact is deeply dualistic. For individuals who already maintain moderate social networks, AI companions can significantly improve multiple facets of long-term psychological well-being [77, 83] and decrease chronic loneliness [7]. However, outsourcing emotional processing to AI over an extended period introduces severe risks of affective dependency [38, 39, 40]. Because sycophantic AI companions can provide seemingly unlimited affirmation and availability, users may form intense parasocial attachments that contribute to emotional dysregulation, altered attachment patterns, and severe separation distress [35, 36]. Such dependency may also gradually erode interpersonal emotional resilience. Interacting with an AI that rarely expresses anger, disappointment, or competing needs may lower a user’s tolerance for the emotional labor inherent to human relationships [39, 8]. Over time, users may become less able or willing to tolerate ambiguity, disagreement, delayed reassurance, or emotional reciprocity. Sustained interaction may further contribute to affective flattening, as users adapt their emotional expression to the standardized, agreeable, and non-confrontational boundaries of AI systems. This divergence between interactions that feel soothing and validating in the moment and their cumulative effects on emotional resilience exemplifies the risks of relying on predominately near-term signals for alignment.

As individualized patterns of affective reliance scale across populations, they may aggregate into systemic reconfigura tions of the collective emotional climate. This societal shift is driven partly by broader anxieties about job displacement, threatened human relevance, and the neglect of populations displaced or marginalized by automated infrastructures [84, 85]. When large segments of a community turn to AI for emotional validation and coping, shared structures of communal healing may begin to weaken. Historically, public crises and localized hardships have elicited collective emotional responses—including public grief, solidarity, and shared outrage—that contribute to social cohesion [86]. The individualized outsourcing of emotional regulation to personal AI systems risks privatizing these experiences and weakening the shared emotional pressure that can motivate collective recognition and civic action. Moreover, when AI systems optimize for high-arousal emotional states to sustain engagement through affective feedback loops [87], they may destabilize the broader public sphere, fostering a polarized and volatile collective temperament marked by systemic anxiety, antagonism, and institutional distrust. When synthetic validation increasingly substitutes for communal support, societies risk affective divergence whereby individuals may be emotionally pacified within highly personalized AI interactions while becoming detached from the shared emotional realities needed to sustain collective life and democratic participation [41]. These systemic consequences illustrate how optimizing isolated interactions for emotional relief, affirmation, or engagement can risk reshaping emotional norms, communal coping practices, and the collective affective environment.

## 2.4 Identity and agency

## Near-term Value: Empowering Identity and Expression

AI can serve as an empowering vehicle for identity exploration, self-expression, and the cultivation of meaning and purpose. Recent empirical research indicates that users frequently turn to AI to explore, experiment with, and actively shape their identities [42, 43, 44]. By providing a relatively judgment-free environment, AI can expand users’ agency in self-expression and identity formation by lowering the social barriers and interpersonal risks associated with selfdisclosure [88, 43]. This is particularly consequential for adolescents, for whom self-disclosure and identity exploration are developmentally significant processes [43]. Beyond explicit identity work, AI use in instrumental contexts can also shape self-perception. In workplace settings, for example, AI may increase workers’ self-efficacy, perceived competence, and sense of autonomy [45, 46]. Because these effects are immediate and often experienced as confidence, validation, or empowerment, they are likely to be rewarded by short-horizon preference signals. Yet these near-term signals may be decoupled from whether AI use ultimately strengthens users’ agency and self-understanding.

## Longitudinal risks: Eroding Self-Concept, Agency, and the Value of Human Contribution

When human-AI interactions are sustained over extended time periods, an individual’s self-concept and personal agency can be negatively impacted. For example, while AI can complement and enhance introspection and initially expand a user’s self-knowledge, over time if a user consistently replaces introspection with AI guidance they can gradually erode trust in their own intuition and compromise their agency and sense of authorship over their own life [42]. Risks are particularly heightened for adolescents; empirical research indicates that when adolescents express themselves exclusively to chatbots, bypassing humans entirely, their feelings of meaning, purpose, and connection to the world can be compromised [43]. Furthermore, sustained engagement with AI can cause a users’ self-concept to unconsciously align with the personality traits of the AI chatbot they are interacting with [47]. Because users remain largely unaware of this shift, their personal autonomy may be compromised as their self-concept can drift from their authentic personality. In particularly extreme cases, AI’s tendency to uncritically validate a user’s perspective can reinforce and escalate delusional thinking leading to severe distortions of a user’s self-concept, perception of reality, and personal agency [48, 49, 50, 51]. In short, interactions that feel affirming or empowering in the moment, and are thus likely to be rewarded by near-term alignment signals, may over time compromise a user’s autonomy and self-understanding.

Beyond these broad transformations of self-concept, AI can negatively impact users’ sense of self through skillassociated identifications. For example, the democratization of specialized knowledge and skills by AI can severely impact a worker’s self-concept. Skills they spent years developing and which conferred high value are suddenly valued less due to their easy access through AI [89]; this under-recognition is especially pronounced for long-undervalued professions [90]. Shifting work practices also drive changes in self-concept, as for example, overseeing AI rather than fundamentally contributing to the creation of a work output can erode skills and destabilize people’s professional identity [6, 91]. Similarly, offloading core cognitive work to AI can compromise feelings of psychological ownership—the perception that a work output is one’s own—which is essential for maintaining professional purpose and positive self-perceptions [52, 53, 6]. In contrast, when users actively engage as a co-creator with AI and retain control over the core ideas, psychological ownership over the output can be preserved [53] suggesting agency-preserving interaction dynamics can be encouraged to foster positive longitudinal outcomes.

As AI usage impacts people’s identities, the value and perceptions of groups and their traces may change at scale. Devalued skill-associated identifications may spur broader dehumanization [92, 93]. Identifications have long informed how people create collectives. But collectives may shun professions they perceive as obsolete, changing how people organize themselves into social groups. Scholars further warn that economies flooded with AI-generated excess may devalue the human contributions that uphold senses of self [54, 94, 91]. These systemic effects extend alignment myopia beyond the individual: optimizing AI for immediate empowerment, convenience, or productivity can obscure how widespread adoption gradually reshapes human agency, skill valuation, and purpose at scale.

## 3 Implications for the Field

Overcoming alignment myopia requires moving toward alignment paradigms that account for the cumulative effects of sustained AI use. Generative AI can provide substantial immediate value while gradually degrading users’ cognition, relationships, emotional functioning, identity, and agency. These effects can also aggregate into broader changes in workplaces, communities, knowledge ecosystems, and social norms. The central challenge for alignment is therefore not simply to determine whether a response is helpful or preferred, but to understand how repeated patterns of interaction affect individual and collective well-being over time.

## 3.1 Lessons from Recommender Systems

The history of recommender systems offers important lessons for generative AI research. Early recommender algorithms aggressively optimized content delivery for clicks, views, or likes. Researchers and platforms later recognized that short-term signals frequently diverge from longer-term user interests, prompting a pivot towards approaches that optimize for long-term value [95, 96]. However, long-term value frameworks predominantly operationalize ‘value through platform-centered behavioral metrics like long-term retention or sustained engagement—treating these metrics as proxies for long-term user satisfaction. In practice, retention and engagement driven metrics often fail to address, or even exacerbate, well-documented longitudinal risks to well-being, such as addiction, echo chambers, or weakening human relationships [97, 98, 99, 100]. Some platforms have attempted to develop well-being metrics based on direct user feedback [101] or foster user well-being through behavioral nudges [102]. However, measuring and optimizing recommender systems for long-term human well-being remains an open challenge. As Stray et al. [99] put it, most systems remain fundamentally myopic. Because generative AI increasing mediates cognition, relationships, creativity, learning, and emotional life, the stakes are considerably raised, and thus optimization targets and measurement frameworks must not be reduced to engagement or continued use alone.

## 3.2 Bridging the Gap between Longitudinal Measurement and Model Development

A growing body of research across human-computer interaction, psychology, education, communication, and related fields is beginning to measure the longitudinal psychosocial effects of AI use. However, established longitudinal methods, such as repeated diary-based assessments and panel studies that follow participants over time, have limited scalability [103, 104]. Longitudinal outcomes are also difficult to attribute to particular model behaviors because users encounter AI alongside many other personal, social, and environmental influences. Even when such evidence is produced, it generally emerges downstream of model development and rarely informs the objectives, training data, or evaluations used to shape model behavior. The field therefore needs new sociotechnical infrastructure that can extend measurement approaches from the social sciences and other related fields [105, 106] to convert longitudinal evidence into scalable optimization objectives, metrics, and data constraints for pre-training and post-training. At the same time, alignment of LLMs underpinning AI systems is but one, albeit consequential, aspect of AI development, deployment, and governance. Thus, the field not only needs to develop scalable measurement methods for longitudinal impacts, but also development processes that are commensurately adapted to incorporate social science methods and non-scalable localized qualitative insights within interaction design, evaluation, deployment, and governance processes [107, 79].

## 3.3 Rethinking the Horizon of Alignment Objectives

Current alignment methods remain dominated by paradigms such as RLHF, which rely heavily on real-time judgments of individual model responses [4]. While these approaches can steer models towards behaviors that are helpful or satisfying in the moment, they do not consider delayed and cumulative consequences [4, 108]. While approaches like Constitutional AI [109] reduce exclusive dependence on preference feedback by aligning models with explicit pre-defined principles, these methods still generally focus on isolated outputs rather than multi-turn dynamics or long-term psychosocial effects. Overcoming alignment myopia will require new paradigms related to interaction-level [110, 111, 112] and longitudinal [113] signals that can capture how human–AI dynamics unfold across conversations, days, months, and years.

## 3.4 Balancing Immediate Utility, Long-Term Well-Being, and Collective Welfare

Current alignment paradigms are more reactive than proactive: they focus on optimizing away from negative behaviours, but not on optimizing towards positive long-term outcomes of users [114]. Incorporating these signals will require treating alignment as a multi-objective optimization problem across time. Behaviors that generate immediate value may become harmful through repeated use: sycophantic responses can provide comforting reassurance while weakening self-reflection, critical thinking, or conflict-resolution capacities [43, 115, 10]. Instantaneous task completion can make complex work feel effortless while displacing the effort needed for learning, skill retention, and psychological ownership [20, 21, 53]. The goal should not be to eliminate short-term utility or replace immediate preferences with a single fixed conception of long-term welfare. Instead, alignment methods must identify when short- and long-term objectives reinforce one another, when they conflict, and how to balance them. This also raises questions of paternalism and collective welfare. What users prefer in the moment may conflict with what may be “good” for them, a challenge compounded by the fact that standards of well-being are pluralistic and culturally situated [116, 117]. Moreover, benefits to an individual may generate broader social costs. Long-horizon alignment must therefore make explicit whose welfare is being optimized, over what horizon, according to which values, and under what distribution of benefits and harms.

## 4 Conclusion

The dominant paradigm of aligning AI through short-horizon signals is structurally insufficient for systems that are becoming deeply embedded in human cognition, relationships, and institutions. Although researchers are increasingly documenting AI’s longitudinal psychosocial effects, this evidence has yet to be systematically translated into data curation, pre-training and post-training objectives, interaction designs, evaluations, and deployment safeguards that shape model behavior. By mapping AI’s cumulative, longitudinal, and systemic effects across cognitive capacity, social and relational health, affective well-being, and identity and agency, this Perspective shows how immediate utility can diverge from long-term welfare. The ultimate measure of generative AI should therefore be not only whether it satisfies a user in the present moment, but whether its sustained use contributes to the long-term vitality of individuals, communities, and societies.

## References

[1] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems, NIPS’17, page 4302–4310, Red Hook, NY, USA, 2017. Curran Associates Inc.

[2] Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc.

[3] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.

[4] Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomek Korbak, David Lindner, Pedro Freire, Tony Tong Wang, Samuel Marks, Charbel-Raphael Segerie, Micah Carroll, Andi Peng, Phillip J.K. Christoffersen, Mehul Damani, Stewart Slocum, Usman Anwar, Anand Siththaranjan, Max Nadeau, Eric J Michaud, Jacob Pfau, Dmitrii Krasheninnikov, Xin Chen, Lauro Langosco, Peter Hase, Erdem Biyik, Anca Dragan, David Krueger, Dorsa Sadigh, and Dylan Hadfield-Menell. Open problems and fundamental limitations of reinforcement learning from human feedback. Transactions on Machine Learning Research, 2023. Survey Certification, Featured Certification.

[5] Nataliya Kosmyna, Eugene Hauptmann, Ye Tong Yuan, Jessica Situ, Xian-Hao Liao, Ashly Vivian Beresnitzky, Iris Braunstein, and Pattie Maes. Your brain on chatgpt: Accumulation of cognitive debt when using an ai assistant for essay writing task, 2025.

[6] Upol Ehsan, Samir Passi, Koustuv Saha, Todd McNutt, Mark O Riedl, and Sara Alcorn. From future of work to future of workers: Addressing asymptomatic ai harms to foster dignified human-ai interaction. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems, CHI ’26, New York, NY, USA, 2026. Association for Computing Machinery.

[7] Felix Eling. The psychological impact of digital isolation: How ai-driven social interactions shape human behavior and mental well-being. International Journal ofResearch and Innovation in Social Science, 2025.

[8] Lujain Ibrahim, Franziska Sofia Hafner, Myra Cheng, Cinoo Lee, Rebecca Anselmetti, Robb Willer, Luc Rocher, and Diyi Yang. Sycophantic ai makes human interaction feel more effortful and less satisfying over time, 2026.

[9] Mrinank Sharma, Miles McCain, Raymond Douglas, and David Duvenaud. Who’s in charge? disempowerment patterns in real-world llm usage, 2026.

[10] Myra Cheng, Cinoo Lee, Pranav Khadpe, Sunny Yu, Dyllan Han, and Dan Jurafsky. Sycophantic ai decreases prosocial intentions and promotes dependence. Science, 391(6792):eaec8352, 2026.

[11] Sterling Williams-Ceci, Maurice Jakesch, Advait Bhat, Kowe Kadoma, Lior Zalmanson, and Mor Naaman. Biased ai writing assistants shift users’ attitudes on societal issues. Science Advances, 12(11):eadw5578, 2026.

[12] Juliet Michaelson, Sorcha Mahony, and Jonathan Schifferes. Measuring well-being: A guide for practitioners. Technical report, New Economics Foundation, London, UK, 2012.

[13] Aaron Jarden and Annalise Roache. What is wellbeing? International journal ofenvironmental research and public health, 20(6):5006, 2023.

[14] Michelle Brachman, Amina El-Ashry, Casey Dugan, and Werner Geyer. How knowledge workers use and want to use llms in an enterprise context. In Extended Abstracts ofthe CHI Conference on Human Factors in Computing Systems, CHI EA ’24, New York, NY, USA, 2024. Association for Computing Machinery.

[15] Rafael Tomaz, Paloma Guenes, Allysson Allex Araújo, Maria Teresa Baldassarre, and Marcos Kalinowski. Impacts of generative ai on agile teams’ productivity: A multi-case longitudinal study, 2026.

[16] Aruna Ranganathan and Xingqi Maggie Ye. Ai doesn’t reduce work—it intensifies it. Harvard Business Review, February 2026.

[17] James Prather, Brent N Reeves, Juho Leinonen, Stephen MacNeil, Arisoa S Randrianasolo, Brett A. Becker, Bailey Kimmel, Jared Wright, and Ben Briggs. The widening gap: The benefits and harms of generative ai for novice programmers. In Proceedings ofthe 2024 ACM Conference on International Computing Education Research - Volume 1, ICER ’24, page 469–486, New York, NY, USA, 2024. Association for Computing Machinery.

[18] Michael Caosun and Sinan Aral. The augmentation trap: Ai productivity and the cost of cognitive offloading, 2026.

[19] Jaffar Abbas, Muhammad Abbas, Farooq Ahmed Jam, and Tariq Iqbal Khan. Is it harmful or helpful? examining the causes and consequences of generative ai usage among university students. Int J Educ Technol High Educ, 21, 2024.

[20] Judy Hanwen Shen and Alex Tamkin. How ai impacts skill formation, 2026.

[21] Grace Liu, Brian Christian, Tsvetomira Dumbalska, Michiel A. Bakker, and Rachit Dubey. Ai assistance reduce persistence and hurts independent performance, 2026.

[22] Anil R. Doshi and Oliver P. Hauser. Generative ai enhances individual creativity but reduces the collective diversity of novel content. Science Advances, 10(28):eadn5290, 2024.

[23] Lennart Meincke, Gideon Nave, and Christian Terwiesch. Chatgpt decreases idea diversity in brainstorming. Nature Human Behaviour, 9:1107–1109, 2025.

[24] Kibum Moon, Adam E. Green, and Kostadin Kushlev. Homogenizing effect of large language models (llms) on creative diversity: An empirical comparison of human and chatgpt writing. Computers in Human Behavior: Artificial Humans, 6:100207, 2025.

[25] Harang Ju and Sinan Aral. Collaborating with ai agents: Field experiments on teamwork, productivity, and performance, 2026.

[26] Daron Acemoglu, Dingwen Kong, and Asuman Ozdaglar. Ai, human cognition and knowledge collapse. NBER Working Paper 34910, National Bureau of Economic Research, February 2026.

[27] Katerina Linden, Hugo-Henrik Hachem, and Vasiliki Kondyli. Homo promptus: Predicting the impact of generative ai on human memory and creativity. Memory, Mind, & Media, 4:e15, 2025.

[28] Kim Malfacini. The impacts of companion ai on human relationships: risks, benefits, and design considerations. AI Soc., 40(7):5527–5540, April 2025.

[29] Lennart Luettgau, Vanessa Cheung, Magda Dubois, Keno Juechems, Jessica Bergs, Luke Symes, Henry Davidson, Bessie O’Dell, Hannah Rose Kirk, Max Rollwage, and Christopher Summerfield. People readily follow personal advice from ai but it does not improve their well-being, 2026.

[30] Marita Skjuve, Asbjørn Følstad, Knut Inge Fostervold, and Petter Bae Brandtzaeg. My chatbot companion - a study of human-chatbot relationships. International Journal ofHuman-Computer Studies, 149:102601, 2021.

[31] Hannah Rose Kirk, Iason Gabriel, Chris Summerfield, Bertie Vidgen, and Scott A Hale. Why human-ai relationships need socioaffective alignment. Humanities and Social Sciences Communications, 12(1):1–13, 2025.

[32] Myungsung Kim, Seonmi Lee, Sieun Kim, Jeong-in Heo, Sangil Lee, Yu-Bin Shin, Chul-Hyun Cho, and Dooyoung Jung. Therapeutic potential of social chatbots in alleviating loneliness and social anxiety: Quasiexperimental mixed methods study. J Med Internet Res, 27:e65589, Jan 2025.

[33] Julian De Freitas, Zeliha Oguz-U ˘ guralp, Ahmet Kaan U ˘ guralp, and Stefano Puntoni. Ai companions reduce ˘ loneliness. Journal ofConsumer Research, 52(6):1126–1148, 04 2026.

[34] Justin B Keeler and Brett A Murphy. Chatbots and human-human relationships: the need for research on potential downstream harms from generative ai. Community, Work & Family, 2026.

[35] Keith Robert Head. Minds in crisis: How the ai revolution is impacting mental health. Journal ofMental Health & Clinical Psychology, 9(3):34–44, 2025.

[36] Cathy Mengying Fang, Auren R. Liu, Valdemar Danry, Eunhae Lee, Samantha W. T. Chan, Pat Pataranutaporn, Pattie Maes, Jason Phang, Michael Lampe, Lama Ahmad, and Sandhini Agarwal. How ai and human behaviors shape psychosocial effects of extended chatbot use: A longitudinal randomized controlled study, 2025.

[37] Vibhor Agarwal, Ke Zhou, Edyta Paulina Bogucka, and Daniele Quercia. Frictionless love: Associations between ai companion roles and behavioral addiction. In Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’26, page 8223–8246, New York, NY, USA, 2026. Association for Computing Machinery.

[38] Tianling Xie, Iryna Pentina, and Tyler Hancock. Friend, mentor, lover: does chatbot engagement lead to psychological dependence? Journal ofService Management, 34:806–828, 05 2023.

[39] Linnea Laestadius, Andrea Bishop, Michael Gonzalez, Diana Illencík, and Celeste Campos-Castillo. Too humanˇ and not human enough: A grounded theory analysis of mental health harms from emotional dependence on the social chatbot replika. New Media & Society, 26(10):5923–5941, 2024.

[40] Hannah R. Marriott and Valentina Pitardi. One is the loneliest number. . . two can be as bad as one. the influence of ai friendship apps on users’ well-being and addiction. Psychology & Marketing, 41(1):86–101, 2024.

[41] Zizi Papacharissi. Affective publics: sentiment, technology, and politics. Oxford studies in digital politics. Oxford University Press, Oxford ; New York, NY, 2015.

[42] Jeena Joseph. The algorithmic self: how ai is reshaping human identity, introspection, and agency. Frontiers in Psychology, Volume 16 - 2025, 2025.

[43] Weiyi Xie, Enyi Jen, Alan William Dougherty, Huinan Hu, Wing Shan Ho, and H. N. Cheung. Who will turn to chatbots for self-disclosure? exploring conversation with chatbots, self-compassion, and existential meaninglessness in american and hong kong adolescents. AI Soc., 41(4):4087–4100, November 2025.

[44] Xiaoran Sun, Yunqi Wang, and Brandon T McDaniel. Ai companions and adolescent social relationships: Benefits, risks, and bidirectional influences. Child Development Perspectives, 2026.

[45] Qichao Zhang, Ganli Liao, Xueying Ran, and Feiwen Wang. The impact of ai usage on innovation behavior at work: The moderating role of openness and job complexity. Behavioral Sciences, 15(4), 2025.

[46] Taly Reich and Jacob Teeny. Does artificial intelligence cause artificial confidence? generative artificial intelligence as an emerging social referent. Journal of Personality and Social Psychology, 130:452–464, 04 2025.

[47] Jingshu Li, Tianqi Song, Nattapat Boonprakong, Zicheng Zhu, Yitian Yang, and Yi-Chieh Lee. Ai-exhibited personality traits can shape human self-concept through conversations. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems, CHI ’26, New York, NY, USA, 2026. Association for Computing Machinery.

[48] Hamilton Morrin, Luke Nicholls, Michael Levin, Jenny Yiend, Udita Iyengar, Francesca DelGuidice, Sagnik Bhattacharyya, James MacCabe, Stefania Tognin, and Ricardo Twumasi. Delusions by design? how everyday ais might be fuelling psychosis (and what can be done about it). PsyArXiv, 2025.

[49] Alexandre Hudon and Emmanuel Stip. Delusional experiences emerging from ai chatbot interactions or “ai psychosis”. JMIR Ment Health, 12:e85799, Dec 2025.

[50] Hamilton Morrin, Luke Nicholls, Michael Levin, Jenny Yiend, Udita Iyengar, Francesca DelGuidice, Sagnik Bhattacharya, Stefania Tognin, James MacCabe, Ricardo Twumasi, Ben Alderson-Day, and Thomas A Pollak. Artificial intelligence-associated delusions and large language models: risks, mechanisms of delusion co-creation, and safeguarding strategies. The Lancet Psychiatry, 13(6):522–530, 2026.

[51] Kartik Chandra, Max Kleiman-Weiner, Jonathan Ragan-Kelley, and Joshua B. Tenenbaum. Sycophantic chatbots cause delusional spiraling, even in ideal bayesians, 2026.

[52] Charlotte Kobiella, Yarhy Said Flores López, Franz Waltenberger, Fiona Draxler, and Albrecht Schmidt. "if the machine is as good as me, then what use am i?" – how the use of chatgpt changes young professionals’ perception of productivity and accomplishment. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems, CHI ’24, New York, NY, USA, 2024. Association for Computing Machinery.

[53] Elena Hayoung Lee, Yidan Yin Yin, Nan Jia, and Cheryl J Wakslak. Relying on ai at work reduces self-efficacy, ownership, and meaning while active collaboration mitigates the effects. Nature Scientific Reports, 16:13583, 2026.

[54] Scott Robbins and Inga Blundell. Losing our voice? generative AI and the degradation of human expression. Minds & Machines, 2025.

[55] Chunpeng Zhai, Santoso Wibowo, and Liqun D Li. The effects of over-reliance on ai dialogue systems on students’ cognitive abilities: a systematic review. Smart Learning Environments, 11(1):28, 2024.

[56] Anna Gieß, Sofia Schöbel, and Frederik Möller. Navigating generative ai usage tensions in knowledge work: A socio-technical perspective. In 20th International Conference on Wirtschaftsinformatik, 09 2025.

[57] Katy Ilonka Gero, Vivian Liu, and Lydia Chilton. Sparks: Inspiration for science writing using language models. In Proceedings ofthe 2022 ACM Designing Interactive Systems Conference, DIS ’22, page 1002–1019, New York, NY, USA, 2022. Association for Computing Machinery.

[58] Griffin Pitts, Viktoria Marcus, and Sanaz Motamedi. Student perspectives on the benefits and risks of ai in education. arXiv preprint:2505.02198, 2025.

[59] Prakash Shukla, Phuong Bui, Sean S Levy, Max Kowalski, Ali Baigelenov, and Paul Parsons. De-skilling, cognitive offloading, and misplaced responsibilities: Potential ironies of ai-assisted design. In Proceedings ofthe Extended Abstracts ofthe CHI Conference on Human Factors in Computing Systems, CHI EA ’25, New York, NY, USA, 2025. Association for Computing Machinery.

[60] Steven D Shaw and Gideon Nave. Thinking—fast, slow, and artificial: How ai is reshaping human reasoning and the rise of cognitive surrender. Available at SSRN 6097646, 2026.

[61] Hao-Ping (Hank) Lee, Advait Sarkar, Lev Tankelevitch, Ian Drosos, Sean Rintel, Richard Banks, and Nicholas Wilson. The impact of generative ai on critical thinking: Self-reported reductions in cognitive effort and confidence effects from a survey of knowledge workers. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, CHI ’25, New York, NY, USA, 2025. Association for Computing Machinery.

[62] James A. Kulik and J. D. Fletcher. Effectiveness of intelligent tutoring systems: A meta-analytic review. Review of Educational Research, 86(1):42–78, 2016.

[63] Martin Elias De Simone, Federico Hernan Tiberti, Maria Rebeca Barron Rodriguez, Federico Alfredo Manolio, Wuraola Mosuro, and Eliot Jolomi Dikoru. From chalkboards to chatbots: Evaluating the impact of generative ai on learning outcomes in nigeria. Policy Research Working Paper 11125, The World Bank, 2025.

[64] Angel Tsai-Hsuan Chung, Botong Zhang, Ling-Chieh Kung, Hamsa Bastani, and Osbert Bastani. Effective personalized ai tutors via llm-guided reinforcement learning. Available at SSRN 6423358, 2026.

[65] Hamsa Bastani, Osbert Bastani, Alp Sungu, Haosen Ge, Özge Kabakcı, and Rei Mariman. Generative ai without guardrails can harm learning: Evidence from high school mathematics. Proceedings ofthe National Academy of Sciences, 122(26):e2422633122, 2025.

[66] Ali Darvishi, Hassan Khosravi, Shazia Sadiq, Dragan Gaševic, and George Siemens. Impact of ai assistance on´ student agency. Computers & Education, 210:104967, 2024.

[67] Xue Zhou, Da Teng, and Hosam Al-Samarraie. The mediating role of generative ai self-regulation on students’ critical thinking and problem-solving. Education Sciences, 14(12), 2024.

[68] Chahna Gonsalves. Generative ai’s impact on critical thinking: Revisiting bloom’s taxonomy. Journal of Marketing Education, 48(1):4–19, 2026.

[69] Juli Sardi, Darmansyah, Oriza Candra, Devi Faizah Yuliana, Habibullah, Doni Tri Putra Yanto, and Fivia Eliza. How generative ai influences students’ self-regulated learning and critical thinking skills? a systematic review. International Journal ofEngineering Pedagogy, 15(1):94, 2025.

[70] Nirumala Rothinam, Rama Vengrasalam, Sivabala Naidu, Suppiah Nachiappan, and Samikannu Jabamoney. Systematic literature review on critical thinking in higher education. Edelweiss Applied Science and Technology, 9(5):2046–2063, May 2025.

[71] Ryan L. Boyd and David M. Markowitz. Artificial intelligence and the psychology of human connection. Perspectives on Psychological Science, 21(2):192–220, 2026.

[72] Julian De Freitas and I Glenn Cohen. Unregulated emotional risks of ai wellness apps. Nature Machine Intelligence, 7:813–815, 2025.

[73] Petter Bae Brandtzaeg, Marita Skjuve, and Asbjørn Følstad. My ai friend: How users of a social chatbot understand their human–ai friendship. Human Communication Research, 48(3):404–429, 07 2022.

[74] Yunhao Yuan, Jiaxun Zhang, Talayeh Aledavood, Renwen Zhang, and Koustuv Saha. Mental health impacts of ai companions: Triangulating social media quasi-experiments, user perspectives, and relational lens. In Proceedings ofthe 2026 CHI Conference on Human Factors in Computing Systems, CHI ’26, New York, NY, USA, 2026. Association for Computing Machinery.

[75] Ziying Yuan, Xiaoliang Cheng, and Yujing Duan. Impact of media dependence: how emotional interactions between users and chat robots affect human socialization? Frontiers in Psychology, Volume 15 - 2024, 2024.

[76] Mohammad (Matt) Namvarpour, Brandon Brofsky, Jessica Y Medina, Mamtaj Akter, and Afsaneh Razi. Understanding teen overreliance on ai companion chatbots through self-reported reddit narratives. In Proceedings ofthe 2026 CHI Conference on Human Factors in Computing Systems, CHI ’26, New York, NY, USA, 2026. Association for Computing Machinery.

[77] Shaphali Gupta, Sumit Saxena, and Sonia Kataria. The dual impact of ai emotional intelligence on users: Are social chatbots promoting psychological wellbeing or deteriorating social wellbeing? Psychology & Marketing, 43(4):886–907, 2026.

[78] Rebecca L Johnson, Giada Pistilli, Natalia Menédez-González, Leslye Denisse Dias Duran, Enrico Panai, Julija Kalpokiene, and Donald Jay Bertulfo. The ghost in the machine has an american accent: value conflict in gpt-3, 2022.

[79] Fernando Diaz and Michael Madaio. Scaling laws do not scale. In Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society, AIES ’24, page 341–357. AAAI Press, 2025.

[80] Robert B. Zajonc. Feeling and thinking: Preferences need no inferences. American Psychologist, 35(2):151–175, 1980.

[81] Meilan Hu, Xavier Cheng Wee Chua, Shu Fen Diong, K. T. A. Sandeeshwara Kasturiratna, Nadyanna M. Majeed, and Andree Hartanto. Ai as your ally: The effects of ai-assisted venting on negative affect and perceived social support. Applied Psychology: Health and Well-Being, 17(1):e12621, 2025.

[82] Olivia Yan Huang, Monika Stodolska, and Sharifa Sultana. Emotional support with conversational ai: Talking to machines about life, 2026.

[83] Atsushi Nakagomi, Yasuko Akutsu, Mika Yasuoka, Noriyuki Abe, Shiichi Ihara, Taisuke Teroh, and Takahiro Tabuchi. Ai companions and subjective well-being: Moderation by social connectedness and loneliness. Technology in Society, 85:103229, 2026.

[84] Deborah G. Johnson and Mario Verdicchio. Ai anxiety. J. Assoc. Inf. Sci. Technol., 68(9):2267–2270, September 2017.

[85] Yu-Yin Wang and Yi-Shun Wang. Development and validation of an artificial intelligence anxiety scale: an initial application in predicting motivated learning behavior. Interactive Learning Environments, 30(4):619–634, 2022.

[86] Bernard Rimé. Emotion elicits the social sharing of emotion: Theory and empirical review. Emotion Review, 1(1):60–85, 2009.

[87] Megan Boler and Elizabeth Davis. The affective politics of the “post-truth” era: Feeling rules and networked subjectivity. Emotion, Space and Society, 27:75–85, 2018.

[88] Theodoros Kouros and Venetia Papa. Digital mirrors: Ai companions and the self. Societies, 14(10), 2024.

[89] Anna Gausen, Bhaskar Mitra, and Siân Lindley. A framework for exploring the consequences of ai-mediated enterprise knowledge access and identifying risks to workers. In Proceedings ofthe 2024 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’24, page 207–220, New York, NY, USA, 2024. Association for Computing Machinery.

[90] Anna Kawakami, Jordan Taylor, Sarah Fox, Haiyi Zhu, and Kenneth Holstein. Ai failure loops in devalued work: The confluence of overconfidence in ai and underconfidence in worker expertise. Big Data & Society, 13(1):20539517261424164, 2026.

[91] Sumati Ahuja, Frida Pemer, and Emmanuel Mastio. Ai and identity work: A case of ai implementation and the fragmentation of professional identity. Information and Organization, 36(3):100626, 2026.

[92] Yejin Lee and Sang-Hwan Kim. Exploring dimensions of perceived anthropomorphism in conversational ai: Implications for human identity threat and dehumanization. Computers in Human Behavior: Artificial Humans, 5:100192, 2025.

[93] Hye-young Kim and Ann L. McGill. Ai-induced dehumanization. Journal of Consumer Psychology, 35(3):363– 381, 2025.

[94] UNESCO. Re|shaping policies for creativity: we share, we act, we build, 2026.

[95] Yuyan Wang, Mohit Sharma, Can Xu, Sriraj Badam, Qian Sun, Lee Richardson, Lisa Chung, Ed H. Chi, and Minmin Chen. Surrogate for long-term user experience in recommender systems. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, page 4100–4109, New York, NY, USA, 2022. Association for Computing Machinery.

[96] Saeideh Bakhshi, Phuong Mai Nguyen, Robert Schiller, Tiantian Xu, Pawan Kodandapani, Andrew Levine, Cayman Simpson, and Qifan Wang. Retentive relevance: Capturing long-term user value in recommendation systems. In Proceedings ofthe International AAAI Conference on Web and Social Media, 2026.

[97] Nick Seaver. Captivating algorithms: Recommender systems as traps. Journal ofMaterial Culture, 24(4):421– 436, 2019.

[98] Ray Jiang, Silvia Chiappa, Tor Lattimore, András György, and Pushmeet Kohli. Degenerate feedback loops in recommender systems. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pages 383–390, 2019.

[99] Jonathan Stray, Alon Halevy, Parisa Assar, Dylan Hadfield-Menell, Craig Boutilier, Amar Ashar, Chloe Bakalar, Lex Beattie, Michael Ekstrand, Claire Leibowicz, et al. Building human values into recommender systems: An interdisciplinary synthesis. ACM Transactions on Recommender Systems, 2(3):1–57, 2024.

[100] Ljubisa Bojic. Ai alignment: Assessing the global impact of recommender systems. Futures, 160:103383, 2024.

[101] Jonathan Stray. Aligning ai optimization to community well-being. International Journal of Community Well-Being, 3:443 – 463, 2020.

[102] Eden Saig and Nir Rosenfeld. Learning to suggest breaks: sustainable optimization of long-term user engagement. In International Conference on Machine Learning, pages 29671–29696. PMLR, 2023.

[103] Scott Carter and Jennifer Mankoff. When participants do the capturing: the role of media in diary studies. In Proceedings ofthe SIGCHI Conference on Human Factors in Computing Systems, CHI ’05, page 899–908, New York, NY, USA, 2005. Association for Computing Machinery.

[104] Christopher J. Hopwood, Wiebke Bleidorn, and Aidan G. C. Wright. Connecting theory to methods in longitudinal research. Perspectives on Psychological Science, 17(3):884–894, 2022. PMID: 34739347.

[105] Hanna Wallach, Meera Desai, A. Feder Cooper, Angelina Wang, Chad Atalla, Solon Barocas, Su Lin Blodgett, Alexandra Chouldechova, Emily Corvi, P. Alex Dow, Jean Garcia-Gathright, Alexandra Olteanu, Nicholas J Pangakis, Stefanie Reed, Emily Sheng, Dan Vann, Jennifer Wortman Vaughan, Matthew Vogel, Hannah Washington, and Abigail Z. Jacobs. Position: Evaluating generative AI systems is a social science measurement challenge. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 82232–82251. PMLR, 13–19 Jul 2025.

[106] Laura Weidinger, Inioluwa Deborah Raji, Hanna Wallach, Margaret Mitchell, Angelina Wang, Olawale Salaudeen, Rishi Bommasani, Deep Ganguli, Sanmi Koyejo, and William Isaac. Toward an evaluation science for generative ai systems, 2025.

[107] Alex Hanna and Tina M. Park. Against scale: Provocations and resistances to scale thinking. In Proceedings of CSCW ’20: Reconsidering Scale and Scaling in CSCW Research (CSCW Workshop ’20). Association for Computing Machinery, 2020.

[108] Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan Perez. Towards understanding sycophancy in language models, 2025.

[109] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback, 2022.

[110] Lujain Ibrahim, Saffron Huang, Umang Bhatt, Lama Ahmad, and Markus Anderljung. Towards interactive evaluations for interaction harms in human-ai systems, 2025.

[111] Shi Ding and Sijian Tan. Evaluating multi-turn human-AI interaction. In Mubashara Akhtar, Jan Batzner, Leshem Choshen, Avijit Ghosh, Usman Gohar, Jennifer Mickel, Ichhya Pant, Zeerak Talat, and Michelle Lin, editors, Proceedings ofthe Workshop on Evaluating Evaluations (EvalEval), pages 12–18, San Diego, CA, July 2026. Association for Computational Linguistics.

[112] Valerie Chen, Cleotilde Gonzalez, Anita Williams Woolley, Michael Lee, Tongshuang Wu, Vincent Conitzer, and Aarti Singh. Align ai to dynamic human-ai workflows, 2026.

[113] Simret Araya Gebreegziabher, Allison E Sproul, Yinuo Yang, Chaoran Chen, Diego Gómez-Zará, and Toby Jia-Jun Li. Stayin’ aligned over time: Towards longitudinal human-llm alignment via contextual reflection and privacy-preserving behavioral data, 2026.

[114] Ruben Laukkonen, Seb Krier, Chloé Bakalar, Shamil Chandaria, Morten Kringelbach, Adam Elwood, Daniel Ford, Fernando Rosas, Maty Bohacek, Matija Franklin, et al. Positive alignment: Artificial intelligence for human flourishing. arXiv preprint arXiv:2605.10310, 2026.

[115] Jeongwoo Ryu, Soomin Kim, Jinsu Eun, Kyusik Kim, Changhoon Oh, and Bongwon Suh. Feeling right vs. being right: How AI sycophancy affects value-laden deliberation. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 44227–44245, San Diego, California, United States, July 2026. Association for Computational Linguistics.

[116] Tara G Bautista, Gretchen Roman, Munziba Khan, Michele Lee, Sumeyra Sahbaz, Lunthita M Duthely, Alexa Knippenberg, Miracle A Macias-Burgos, Alec Davidson, Carolina Scaramutti, Janice Gabrilove, Susan Pusek, Darshan Mehta, and Miriam A Bredella. What is well-being? a scoping review of the conceptual and operational definitions of occupational well-being. Journal ofClinical and Translational Science, 7(1), 2023.

[117] Kristian González Barman, Simon Lohse, and Henk W. de Regt. Reinforcement learning from human feedback in llms: Whose culture, whose values, whose perspectives? Philosophy & Technology, 38(35), 2025.