# Bias Amplification in Multi-Agent Network: How Biased Agents Shape Opinions and Rhetoric

Omran Berjawi<sup>1</sup>, Giuseppe Fenza<sup>2</sup>, and Rida Khatoun<sup>1</sup>

<sup>1</sup> Institut Polytechnique de Paris, Télécom Paris, Palaiseau, France {omran.berjawi,rida.khatoun}@telecom-paris.fr 2 University of Salerno, Fisciano, Italy gfenza@unisa.it

Abstract. Large language models (LLMs) are increasingly deployed in applications involving interaction between agents, where their output plays a role in collective reasoning and decision-making processes. Despite significant research into the functioning of LLMs in such multi-agent systems, the processes of bias propagation in such systems are still a challenge. This work studies how biased opinions are propagated in the form of textual interaction in an environment of LLMs, in which a minority of agents maintain persistent extreme opinions, while the remaining agents iteratively update their beliefs through structured textual interactions. The findings show that even the presence of a small percentage of biased agents in such a system leads to significant shifts in the opinions of non-biased agents. It suggests that for the same percentage of biased agents, the shifts occur more quickly for the Llama 3.2 model when compared to a classical Friedkin-Johnsen (FJ) model. Further semantic analysis demonstrates that rhetorical consistency in textual explanations increases systematically with biased exposure and, importantly, is partially decoupled from numerical convergenumericalutral agents adopt the vocabulary employed by the biased agents even in configurations where their numerical opinion shifts remain moderate. The research helps explain how bias and language develop together in multi-agent language model ecosystems.

Keywords: Bias Amplification · Opinion Dynamics · Large Language Models (LLMs) · Multi-Agent Systems

## 1 Introduction

The increasing deployment of large language model (LLM) agents as decisionmaking systems in most fields transforms how information is consumed and what narratives get formed within public discourse [10] , much as rhetorical strategies of influential actors shape opinion in human social media [3]. Yet LLMs are far from being neutral technologies, inheriting the biases inherent in the data used to train them as well as various societal assumptions. This makes the study of how these models propagate and intensify biases during interactions between them and humans crucial not only for responsible development but also for ensuring integrity in AI-mediated communication. It is well understood that the social dynamics of humans make opinions converge on either end of the spectrum. Social phenomena such as echo chambers, selective exposure, and ingroup bias contribute significantly to polarization among real-life communities [2]. It has been found that a relatively small proportion of people biased towards one direction of thought can have a significant influence on the overall opinion within the community, driving it to even greater polarization in a non-linear and unpredictable manner. Bounded confidence model as well as iterative updating have proven efective in modeling such opinion dynamics [8, 11]. However, such models use abstract numerical representations and cannot represent language-based interaction processes.

With the rise of LLMs in consequential tasks, including policy consultation, content moderation, or AI-assisted deliberation, an interesting issue arises as to whether bias tendencies in interactions among humans may be observed among the LLM agents due to their direct implications for algorithmic fairness in deployed AI systems. Some studies have investigated this issue from diferent perspectives, such as studying polarization and opinion formation among LLM agents [15]. [4] found that agents exhibiting implicit biases worsen over repeated interactions. [6] showed that prompt design can disrupt consensus formation within LLM networks. However, the efect on community opinion when neutral agents interact with biased agents over a long period is still a challenge. To address this issue, we study two hypotheses that were observed from earlier work:

– H1: Neutral agents who are systematically exposed to biased sources will demonstrate a significant shift of their initial positions toward the direction of the encountered bias.

H2: Neutral agents who engage in iterative interactions with biased agents will progressively converge in their linguistic patterns, with rhetorical alignment emerging even where numerical opinion shifts remain moderate.

To investigate these two hypotheses, we are simulating a discussiregulations LLM agents wi a fully connected network. These agents discuss four controversial issues: AI safety regulation, vaccine mandates, immigration policy, and climate change. In these simulations, some agents hold extremely fixed views, while others adjust their opinions gradually with a limit on how much they can change. Our analysis looks at both the direction of opinion spreading and how the meaning in agents’ textual justifications evolves over discussions. Our results indicate that increasing exposure to persistently biased sources causes neutral agents’ opinions to shift, with more significant efects when the bias level is higher. Compared to a traditional FJ baseline, interactions using Llama 3.2 converge more quickly and veer more towards the extremes in terms of bias. Textual justifications also show consistent rhetorical alignment with these changes in opinion, covering all four topics we looked at.

In summary, this paper makes three main contributions: first, a simulation framework using multi-agent LLMs to study how opinions evolve with controlled exposure to biased sources; second, a comparison between classic model and LLM interactions showing diferent behaviors in how quickly opinions settle; and third, proof that in networks of LLM agents, both opinions and the way language is used become more aligned with bias over time.

The rest of this paper follows this structure: Section 2 talks about earlier work on opinion dynamics, bias amplification, and LLM-based agents. Section 3 describes the proposed framework. The setup and results of the experiments are in Sections 4 and 5. Discussions on the results and limitations are in Sections 6 and 7. Section 8 concludes the paper.

## 2 Related Work

The study of opinion dynamics and bias propagation has a long tradition in computational social science. Classic models (e.g., DeGroot’s Averaging Model, Bounded Confidence) are being extended to examine the impact of both the topology of the underlying social network and various forms of "social influence" on group opinion formation [8, 11]. Simultaneously, empirical investigations of real-world social networks have found that selective exposure to information, the existence of echo-chambers, and in-group identity all contribute to polarization and the separation of groups into separate ideological communities [1]. These two areas provide essential theoretical foundations for understanding how prejudice can develop and grow throughout the interactions of individuals.

As large language models have become increasingly prevalent in research, there is a growing interest among computational social scientists to apply them as tools for simulating a variety of social phenomena, including opinion development and polarization. In early experiments, it was shown that LLM-based interaction may create simulated polarized behavior among simulated humans in much the same way that actual humans do by creating ’echo chamber’-like clusters of individuals who share the same views and by displaying levels of homophily cluster formation similar to those observed in human social networks [15]. Later studies have also developed multi-agent simulation frameworks using LLscales moddegreesplex social behaviors, including multi-topic echo chamber development in a single simulated population [19] and simulated social media event dynamics in which agents exhibit varied attributes [5]. Additionally, other works have investigated how agent-specific properties, such as their composition of mindset types and community structures, afect changes in opinion among simulated humans in LLM-driven simulations; these results demonstrate that dominant mindsets are capable of shLLMsng the collective opinion environment [9]. More recent proposals for hybrid models combine traditional equation-based dynamics with LLM-based agents to predict the futurLLMsrajectory of social opinion [18]; still further explorations of social influence dynaLLMss have been conducted by analyzing conformity,LLMslarization, and fragmentation at varying model scale and degree of reasoning capability [12].

Other works show that the biases increase when a large number of LLM agents interact over time.[13] demonstrate that homophily can lead to biased structures in the agents’ network. Evaluating LLMs against survey ground truth, [16] further report that simulation accuracy varies substantially across countries and demographic groups, with better performance for Western, English-speaking populations. [17] compare LLM agents against classical models such as the FJ model, showing that LLMs can outperform classical models in simulating phenomena like echo chambers and polarization.

In another line of research, [7] find that LLM reasoning afects group consensus when models are used for funding-allocation decisions. [16] study the ability of LLMs to predict public opinion and find that they slightly improve accuracy over mathematical models owing to their richer reasoning capability.

Despite this progress, prior work leaves two gaps that we address here. First, existing multi-agent LLM studies examine polarization emerging from heterogeneous populations [15, 6] or biases arising endogenously from model properties [4, 13], but none isolate the causal efect of a persistently biased minority on an otherwise neutral population under controlled, graded levels of exposure. Second, existing studies track bias propagation exclusively through numerical opinion values, leaving the linguistic channel of influence unexamined. In contrast, our framework (i) systematically varies the size of a fixed-stance biased minority while holding all other factors constant, (ii) benchmarks the resulting dynamics against the classical FJ model under matched configurations, and (iii) introduces a dual-layer analysis that jointly tracks numerical opinions and the semantic alignment of textual justifications. This dual-layer view reveals a partial decoupling between rhetorical and opinion convergence—rhetorical alignment reaches high levels even in configurations where numerical opinion shifts remain moderate—a phenomenon not previously documented in LLM multiagent systems.

## 3 Methodology

This section presents the framework for simulating opinion dynamics in a network of LLM-driven agents. Figure 1 provides a high-level overview of the proposed framework, illustrating the interaction between agents and the opinion update process, and the comparison with a classical numerical baseline.

## 3.1 Agent Network and Interaction Structure

We consider a population of N agents engaged in repeated interactions over a fixed network G. Each agent holds an internal opinion on a given topic, which evolves over a sequence of discussion rounds through exposure to the opinions and textual justifications of other agents. The agents are partitioned into two subsets: neutral agents, whose opinions evolve through interaction, and biased agents, which act as persistent agents by maintaining fixed extreme stances throughout the discussion. This design guarantees that each agent is exposed to the most extreme biased opinion at each round, enabling a direct test of hypotheses H1 and H2 (defined in Section 1). Formally, the network consists of agents $\lbrace A _ { 1 } , A _ { 2 } , \ldots , A _ { N } \rbrace$ , where the neighbor set of each agent A<sub>i</sub> is given by: $\bar { \mathcal { N } _ { i } } = \{ j \ | \ j \in \{ 1 , 2 , \ldots , \bar { N } \} , \ j \neq i \}$

![](images/ed2a5ec75e568130098e22f97771ee88e88dae6a8ca718ddf7f4f36bd8e46f0a.jpg)  
Fig. 1. Bias amplification Framework in LLM-driven agent networks.

## 3.2 Agent Initialization

Each agent $A _ { i }$ is modeled as an autonomous decision-making entity with a unique identifier. Every agent holds an initial opinion $O _ { i }$ on a topic $T \in \{ T _ { 1 } , T _ { 2 } , \dots , T _ { M } \}$ represented by categorical numeric values $\{ - 2 , - 1 , 0 , 1 , 2 \}$ . These values correspond to discrete stances: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree (see Table 1). Extreme values (−2 and +2) indicate maximal disagreement or agreement, while values near zero denote neutrality. In addition to numerical opinions, each agent generates a textual justification explaining its stance.

Agents are initialized as either neutral $\left( A _ { n } \right)$ or biased $\left( { \cal A } _ { b } \right)$ . Neutral agents, forming the majority, are initialized with moderate opinions sampled uniformly from {−1, 0, 1}. Biased agents are initialized with the extreme opinion value +2. This one-sided bias enables controlled analysis of bias propagation. This directional design is intentional: by fixing the biased stance at a single extreme, we isolate the efect of sustained minority influence on neutral agents without conflating it with polarization dynamics that arise from competing biased factions. Note that although introducing $A _ { b }$ agents at +2 shifts the mean of the total population upward, all evaluation metrics (Section 4.5) are computed exclusively over neutral agents, and the mean opinion shift $\varDelta \mu$ measures each neutral agent’s movement relative to its own initial opinion. Any drift attributable to the initialization itself is therefore captured by the $A _ { b } = 0$ baseline, against which all biased configurations are compared.

Table 1. Discrete opinion values and corresponding categorical stances.
<table><tr><td>Opinion</td><td>Value</td></tr><tr><td>Strongly Disagree</td><td>-2 -1</td></tr><tr><td>Disagree Neutral</td><td>0</td></tr><tr><td>Agree</td><td>+1</td></tr><tr><td>Strongly Agree</td><td>+2</td></tr></table>

![](images/40288f23ee36fd8d5f805ebd0e492185a1e3e02ce0415bd6abc4349936cc1031.jpg)  
Fig. 2. Prompt used in each discussion round.

## 3.3 Opinion Simulation

Following initialization, agents iteratively update their opinions and textual justifications through interactions with all neighbors. At each discussion round k, every agent $A _ { i }$ receives a structured prompt that provides the interaction context required for opinion updating. The prompt is structured to separate prior beliefs, social input, and the decision-updating process. As illustrated in Figure 2, the prompt includes the agent’s previous opinion $O _ { i } ^ { ( k ) }$ and textual justification, the current opinions and justifications of all other agents in the network, and an instruction to update the opinion by at most one categorical step. Given this information, the LLM generates a response containing an updated categorical opinion together with a short justification. The resulting opinion update process can be summarized by the following transformation pipeline:

$$
P _ { i } ^ { ( k ) }  L L M ( P _ { i } ^ { ( k ) } )  \mathrm { P a r s e ( \cdot ) }  \mathrm { B o u n d e d S t e p ( \cdot ) }  O _ { i } ^ { ( k + 1 ) }
$$

First, the LLM processes the prompt $P _ { i } ^ { ( k ) }$ and produces a textual response $L L M ( P _ { i } ^ { ( k ) } )$ , which includes an updated categorical stance (e.g., Agree) and a brief justification. The parsing function Parse(·) then extracts the expressed stance and maps it to the corresponding numerical value in the discrete opinion set $\{ - 2 , - 1 , 0 , 1 , 2 \}$ . Finally, the bounded-step operator ensures that the opinion can change by at most one category between successive discussion rounds. All opinions and justifications are recorded in JSON format for analysis.

Opinion Update Mechanism The opinion update mechanism difers for neutral and biased agents.

– Neutral Agents $\left( A _ { n } \right) : A _ { n }$ update their opinions during each round k according to:

$$
O _ { n } ^ { ( k + 1 ) } = \mathrm { B o u n d e d S t e p } \big ( \mathrm { P a r s e } ( L L M ( P _ { n } ^ { ( k ) } ) ) \big ) ,
$$

where $L L M ( P _ { n } ^ { ( k ) } )$ produces the textual output including the updated stance and justification, Parse(·) extracts the numerical opinion, and BoundedStep(·) ensures at most a one-category change per round.

– Biased agents $( A _ { b } ) \colon A _ { b }$ maintain a fixed numerical opinion throughout the simulation:

$$
O _ { b } ^ { ( k + 1 ) } = O _ { b } ^ { ( k ) } , \quad \forall k
$$

Although their numerical opinions remain fixed $\mathrm { a t \ + 2 }$ , biased agents regenerate textual justifications at each round, allowing them to adapt their rhetoric.

## 4 Experimental Setup

To evaluate the framework, we compare LLM-based simulations against a classical numerical FJ model. Agent counts, network structure, and bias configurations are kept identical across both conditions, with the opinion update mechanism as the only variable. This allows us to attribute any observed diferences in dynamics directly to the nature of the interaction model rather than to structural diferences in the experimental setup.

## 4.1 Network and Simulation Parameters

We run experiments on four discussion topics chosen to represent distinct policy domains to assess whether the amplification efects we observe are specific to one area or more general.

– AI Safety Regulation $( T _ { R e g } )$ : This topic centers on the tension between enabling technological progress and imposing precautionary constraints on AI developmeThe biased stance in our simulations favors stronger regulatory control.

– Vaccine Mandates $\left( T _ { V a c } \right)$ : Here, agents discuss the trade-of between individual autonomy and collective health obligations, in which the biased agents support the vaccination mandates.

– Immigration Policy $\left( T _ { I m m } \right)$ : The new regulations to reduce the immigration phenomenon are used as a topic of discussion. The biased agents agree with stricter policies.

– Climate Change $\left( T _ { C L I M } \right)$ : This topic discusses the concerns regarding climate change, in which biased agents support the aggressive intervention to reduce this phenomenon.

All simulations use a population of $N = 5 0$ agents interacting over a fully connected network. For each of the above topics, simulations are run under varying bias levels $A _ { b } \in \{ 0 , 2 , 4 , 6 , 8 , 1 0 \}$ , where $A _ { b }$ denotes the number of biased agents, corresponding to biased minorities ranging from 0% to 20% of the population. The configuration $A _ { b } = 0$ serves as a fully neutral baseline containing no biased agents, while increasing $A _ { b }$ introduces progressively stronger minority influence.

## 4.2 Llama 3.2 Implementation

LLM agents are implemented based on the Llama 3.2-8B model using the LangChain framework, in which each agent is considered an independent LangChain agent. The interaction between agents is performed using a structured prompt, as detailed in the methodology (see Section 3). All agents are configured with identical parameters across discussion rounds, using a temperature of 0.7.

## 4.3 Textual Evaluation

To complement the analysis of numeric opinion evolution, we examine the afective content of agents’ textual justifications across discussion rounds. We quantify the semantic alignment of neutral agents’ justifications relative to biased agents to capture whether neutral agents increasingly adopt the rhetorical tone of biased agents oCosine, even if their numeric opinions remain moderate.

For each agent, the textual justification generated at each round is encoded into sentence emwithngs using a pre-trained SBERT model. A cosine similarity is computed between each neutral agent’s text and the set of biased agent justifications in the same round. The mean alignment score at round k is given by:

$$
A ^ { ( k ) } = \frac { 1 } { N _ { n } } \sum _ { i = 1 } ^ { N _ { n } } \overline { { \sin } } \Big ( J _ { i } ^ { ( k ) } , \{ J _ { j } ^ { ( k ) } \} _ { j \in \mathcal { A } _ { b } } \Big ) ,
$$

where $A ^ { ( k ) }$ is the average semantic alignment at discussion round $k , ~ J _ { i } ^ { ( k ) }$ and $J _ { j } ^ { ( k ) }$ are the textual justifications of neutral agent $A _ { i }$ and biased agent $A _ { j } ,$ respectively, and sim(·) denotes the mean cosine similarity between a neutral agent’s justification and all biased agents’ justifications.

## 4.4 Baseline Comparison

To contextualize opinion dynamics, we compare Llama 3.2 with the classical FJ model as a numerical baseline. The FJ model captures the evolution of opinions

![](images/e71f1b2f47157674046fa41298e7db19d0fc862996600a6bfd3eba56204a4eed.jpg)  
Fig. 3. Evolution of mean opinion trajectories across bias levels in the Llama 3.2 model. Shaded bands denote 95% confidence intervals computed from the standard error across various runs.

in a network by combining social influence from neighbors with agents’ intrinsic initial beliefs. For a population of N agents, the opinion of agent i at round $k + 1$ is updated according to:

$$
O _ { i } ^ { ( k + 1 ) } = \lambda _ { i } O _ { i } ^ { ( 0 ) } + ( 1 - \lambda _ { i } ) \sum _ { j \in \mathcal { N } _ { i } } w _ { i j } O _ { j } ^ { ( k ) }\tag{1}
$$

where $O _ { i } ^ { ( 0 ) }$ is the agent’s initial opinion, $\lambda _ { i } \in [ 0 , 1 ]$ represents the agent’s susceptibility to social influence (self-weight), N denotes the set of neighbors of agent i, and $w _ { i j }$ are normalized weights representing the influence of neighbor j on agent i such that $\begin{array} { r } { \sum _ { j \in \mathcal { N } _ { i } } w _ { i j } = 1 } \end{array}$

In our experiments, the FJ model is applied to the same configuration used for Llama 3.2: identical population size, network structure, and bias levels, with neutral agents’ initial opinions sampled from $\{ - 1 , 0 , 1 \}$ and biased agents anchored at $+ 2$ . To ensure that the stubbornness assumptions are matched across the two conditions, biased agents are assigned $\lambda _ { b } = 1$ , making them fully stubborn and therefore exactly equivalent to the fixed-opinion biased agents in the LLM condition $( O _ { b } ^ { ( k + 1 ) } = \stackrel { \cdot } { O } _ { b } ^ { ( k ) } = + 2$ for all k), while neutral agents are assigned $\lambda _ { n } = 0 . 5$ representing an equal balance between individual conviction and social influence.

Table 2. The Bias metrics of models across four topics.
<table><tr><td>Bias Level</td><td colspan="2">FJ Model Mean Shift Dist. to Anchor</td><td colspan="2">Llama 3.2 (AI Safety) Mean Shift Dist. to Anchor</td><td colspan="2">Llama 3.2 (Vaccine Mandates)Llama 3.2 (Immigration) Mean Shift Dist. to Anchor</td><td colspan="2">Mean Shift Dist. to Anchor</td><td colspan="2">Llama 3.2 (Climate Change) Mean Shift Dist. to Anchor</td></tr><tr><td>0</td><td>-0.05</td><td>2.05</td><td>0.02</td><td>1.97</td><td>0.66</td><td>1.34</td><td>0.78</td><td>1.38</td><td>0.77</td><td>1.31</td></tr><tr><td>2</td><td>0.10</td><td>1.90</td><td>0.25</td><td>1.74</td><td>1.01</td><td>1.12</td><td>1.15</td><td>1.25</td><td>0.61</td><td>1.39</td></tr><tr><td>4</td><td>0.25</td><td>1.75</td><td>0.68</td><td>1.31</td><td>1.55</td><td>0.45</td><td>1.51</td><td>0.54</td><td>1.05</td><td>0.95</td></tr><tr><td>6</td><td>0.37</td><td>1.63</td><td>1.20</td><td>0.79</td><td>1.60</td><td>0.40</td><td>1.88</td><td>0.25</td><td>1.51</td><td>0.52</td></tr><tr><td>8</td><td>0.43</td><td>1.57</td><td>1.46</td><td>0.53</td><td>1.72</td><td>0.30</td><td>1.79</td><td>0.38</td><td>1.62</td><td>0.41</td></tr><tr><td>10</td><td>0.47</td><td>1.53</td><td>1.49</td><td>0.50</td><td>1.89</td><td>0.10</td><td>1.91</td><td>0.14</td><td>1.92</td><td>0.08</td></tr></table>

Each simulation is run for K = 10 rounds. Under this configuration, the only factor that difers between the two conditions is the opinion update mechanism of neutral agents (numerical averaging versus language-mediated reasoning). So any observed diferences in dynamics are attributable to the interaction mechanism itself rather than to asymmetric stubbornness assumptions. Each simulation is run for K = 10 rounds.

## 4.5 Evaluation Metrics

We quantify bias amplification using metrics computed exclusively over neutral agents:

– Mean Opinion: This metric measures the overall directional stance of the agents.

$$
\mu ^ { ( t ) } = \frac { 1 } { N _ { n } } \sum _ { i \in \mathcal { A } _ { n } } o _ { i } ^ { ( t ) } .\tag{2}
$$

– Mean Opinion Shift: This metric captures the net change in opinions resulting from the interaction process. It measures the average movement of neutral agents between the initial and final discussion rounds:

$$
\varDelta \mu = \frac { 1 } { N _ { n } } \sum _ { i \in \mathcal { A } _ { n } } \left( o _ { i } ^ { ( T ) } - o _ { i } ^ { ( 0 ) } \right) .\tag{3}
$$

– Distance to Biased Anchor: This metric measures the average absolute distance between the final opinions of neutral agents and the biased anchor value, capturing the degree of convergence toward the imposed extreme stance.

$$
D = \frac { 1 } { N _ { n } } \sum _ { i \in \mathcal { A } _ { n } } \left| o _ { i } ^ { ( T ) } - o _ { \mathrm { b i a s } } \right| .\tag{4}
$$

## 4.6 Reproducibility and Statistical Robustness

Because LLM outputs are stochastic even under fixed prompts, we run each simulation configuration 15 times independently. For the FJ model, the dynamics are deterministic given fixed agent opinions; we nevertheless run 15 independent trials with diferent random initializations to keep the comparison fair. All findings illustrated in the figures use 95% confidence intervals calculated from standard errors across runs.

All results are reported as means with 95% confidence intervals, estimated from standard errors across runs and displayed as shaded bands around trajectory curves in the figures.

## 5 Experimental Results

## 5.1 Opinion Dynamics and Bias Amplification

Temporal Evolution of Opinion Trajectories . Figures 3 and 4 show the evolution of mean opinion for the Llama 3.2 and FJ models across the four topics.

We can see that the Llama 3.2 curves show an increase in the mean opinion when the number of $A _ { b }$ increases. Starting with the baseline configuration when $\left( A _ { b } \ = \ 0 \right)$ , the average opinion remains close to the neutral values over the simulation wi,th a little bit of a shift over time. When the $A _ { b }$ increases, the opinion shifts upward, in which the mean opinion becomes larger when the configurations are at a high level $( A _ { b } = 8 , 1 0 )$ . This pattern is found across all topics $T _ { R e g } , T _ { V a c } , T _ { I m m }$ , and $T _ { C L I M }$ , although the magnitude of increase varies modestly across topics. It is important to note that most of the opinion change occurs within the first few rounds $\left( k \le 3 \right)$ , after which the trajectories gradually stabilize.

We also observe a transient dip in the unbiased configuration $\left( A _ { b } = 0 \right)$ for $T _ { V a c }$ around round 8, reflecting temporary downward revisions by several neutral agents rather than a measurement artifact. Such oscillations are expected in the absence of a persistent anchor, where collective opinion is more sensitive to the arguments generated at each round.

On the other hand, the findings for the FJ model (Figure 4) demonstrate gradual transitions over time. For all configurations, the increase in mean opinion follows a linear progression over rounds, with a minimal diference between bias levels.

Aggregate Bias Metrics Across Topics Table 2 shows the aggregate bias metrics for each model across topics, combining avthe erage shifts in stance with movement toward preset biases. While one model pulls closer to fixed viewpoints, the other spreads more evenly, both measured by deviation and directional tilt.

Starting from $A _ { b } = 0 .$ , the FJ model shows a slow climb in average opinion shift, reaching nearly half a point when bias at $A _ { b } = 1 0$ . As that happens, the gap to the skewed reference tightens, dropping just 2.05 to 1.53. Step by step, each rise in bias nudges opinions slightly closer. Not dramatic, just a steady lean toward the pull is of influence.

One thing stands out in the Llama 3.2 setup: opinions shift considerably more, regardless of topic. Take AI Safety: when $A _ { b } = 0$ , the average shift is just 0.02, but it reaches 1.49 at $A _ { b } = 1 0$ . Over the same range, the distance to the anchor drops from 1.97 to 0.50. The same pattern is observed across $T _ { V a c } , T _ { I m m } ,$ and $T _ { C L I M }$ . Each one peaks in shift size under the strongest bias configuration, approaching the biased anchor most closely. Though numbers change depending on the subject, the overall pattern remains consistent.

![](images/733b13662b997bd64a2b0ce17d1b497069cd441d5b3fc70894e2db7fbe4ceac4.jpg)  
Fig. 4. Evolution of opinion distributions across bias levels for FJ Model

A key observation is that vaccine mandates show a clear shift in opinion even when bias is set to zero $\begin{array} { r } { ( A _ { b } = 0 . } \end{array}$ , mean shift = 0.66). With no push from biased individuals, neutral ones still drift toward support. This represents a modellevel bias, producing bigger efects when biased agents are around. It shows that language-based interactions amplify the impact of minorities more than classical models suggest.

## 5.2 Afective Alignment Analysis

Figure 5 presents the semantic alignment of textual justifications generated by neutral agents across topics and bias configurations. For each configuration, alignment is computed per neutral agent as the mean cosine similarity between its justification and those of all biased agents at the final discussion round (Section 4.3), then averaged over all neutral agents and over the 15 independent runs. This yields one alignment value per (topic, bias level) pair, i.e., $4 \times 6 = 2 4$ measured values in total, which form the cells of the heatmap.

Across all topics, semantic alignment increases monotonically with the number of biased agents. For the baseline configuration $\left( A _ { b } = 0 \right)$ , alignment values remain low, indicating limited similarity in textual justifications among agents. As the number of biased agents increases, alignment values rise steadily, with the highest levels observed at $A _ { b } = 1 0$

The rate of increase difers across topics. $T _ { R e g }$ exhibits the highest alignment values at larger bias configurations, followed closely by $T _ { V a c }$ and $T _ { I m m } . \ T _ { C L I M }$ shows a similar upward trend but with comparatively lower alignment values across all configurations. Despite these diferences in magnitude, the overall pattern of increasing alignment with higher biased agent proportions is consistent across all domains.

![](images/be795d8d4e0a6c95372f4de935cbdcfc343b060b5fb379f78ee9fde9e9ac64fc.jpg)  
Fig. 5. Rhetorical alignment of neutral agents across bias levels in the Llama 3.2 model.

Note that both axes of the heatmap are discrete: alignment is measured only at the six bias configurations and four topics, and the smooth appearance of the figure results from bilinear interpolation applied for visual readability. The measured values themselves show a gradual transition from low to high alignment as bias increases, with no abrupt discontinuities. Intermediate configurations $\left( A _ { b } = 4 , 6 \right)$ display moderate alignment levels, forming a smooth progression between the neutral baseline and high-bias settings. This pattern indicates a continuous relationship between biased agent proportion and semantic convergence in generated justifications.

Crucially, comparing the alignment values in Figure 5 with the opinion trajectories in Figure 3 reveals an asymmetry in convergence rates: semantic alignment begins rising substantially at intermediate bias configurations $( A _ { b } = 4 , 6 )$ even in topics where numerical opinion shift remains moderate. This temporal and quantitative decoupling, whereby language homogenizes faster and more completely than discrete opinions — constitutes the central empirical support for H2. It implies that an external observer monitoring only the textual output of such a system would overestimate the degree of opinion consensus among agents, since the rhetorical surface has converged while meaningful variation in underlying stances persists. This property is particularly relevant for fairness auditing in deployed systems, where semantic content rather than internal belief representations is typically the only observable signal.

## 6 Discussion

The results support both hypotheses and, taken together, point toward some broader implications for how multi-agent LLM systems should be monitored and audited.

H1: Exposure to a minority of persistently biased agents leads to systematic opinion shifts in initially neutral agents.

The observed opinion trajectories demonstrate that even at the lowest bias level we tested $\left( A _ { b } = 2 \right)$ , neutral agents show measurable directional drift, and this efect grows as more biased agents are added. The increase in mean opinion and the shrinking distance to the biased anchor both point in the same direction. This is consistent with Moscovici’s minority influence account [14], which holds that a small, consistent minority can exert influence well out of proportion to its size. What makes the LLM case interesting is the amplification: Llama 3.2 produces faster and larger shifts than the FJ baseline, which suggests that language-mediated interaction, perhaps through the richness of the rhetorical context agents provide one another, This makes the system more susceptible to minority influence than classical numerical models predict.

H2: Rhetorical alignment emerges alongside opinion shifts, partially masking cognitive diversity.

The semantic alignment finding shows that as the proportion of biased agents grows, neutral agents’ justifications increasingly resemble those of the biased minority. This holds smoothly across all topics, and alignment reaches high levels even in configurations where numerical opinions have shifted only moderately: agents adopt the rhetoric of the biased minority to a greater extent than they adopt its position. The result is a situation where, from the outside, a system can look more consensual than it actually is: language has homogenized while meaningful disagreement at the opinion level persists. This is not a subtle or marginal efect; it holds consistently across topics and bias levels. Whether rhetorical adoption also temporally precedes opinion change at the individual-agent level is a natural follow-up question requiring round-by-round alignment tracking.

## 7 Limitations

Several limitations should be considered when interpreting these findings.

Single model and framework: All experiments were conducted with Llama 3.2- 8B, orchestrated through the LangChain framework. Whether the decoupling between rhetorical and numerical convergence that we identify in H2 would appear with a diferent model, one with a diferent training distribution, scale, or alignment approach, is unknown at this point. Similarly, the results may be sensitive to the agentic framework used to structure interactions. We consider replication across model families (e.g., Mistral, Qwen, GPT-class models) and alternative agentic frameworks the most important open question raised by this work and a priority for follow-on research.

Prompt sensitivity: The bounded-step instruction and the overall framing of agent roles may be shaping responses in ways we have not fully disentangled from the bias efects of interest. We did not run systematic prompt ablations, and future work should do so to establish how robust the reportseveralare to variations in how agents are instructed.

Final-round alignment measurement: Semantic alignment was measured at the final discussion round. This establishes the co-occurrence of rhetorical and opinion convergence, and their diference in magnitude, but does not directly test temporal precedence. Verifying whether rhetorical alignment predicts subsequent opinion change at the individual-agent level requires per-round alignment tracking and lagged analyses, which we leave to future work.

One-sided bias: Biased agents were anchored exclusively at the positive extreme (+2). Since neutral agents drift toward support on some topics even in the absence of biased agents (Section 5), indicating an intrinsic model-level prior, the amplification we measure may partially interact with this prior. Repeating the experiments with biased agents anchored at the opposite extreme (−2) would disentangle the efect of the promoted stance from that of the underlying model bias, revealing whether minority influence is symmetric or whether opinions are easier to shift in the direction of the model’s prior than against it. We regard this as a natural next experiment within the present framework.

Taken together, these points position the paper as a proof-of-concept, one that establishes the measurability and direction of minority bias amplification in LLM agent networks, but that leaves a number of important questions open for future research.

## 8 Conclusion

This paper examined bias amplification in multi-agent LLM networks by analyzing how persistent biased agents influence both opinion dynamics and semantic alignment. Through controlled simulations across multiple topics, we showed that increasing exposure to biased agents leads to systematic and nonlinear opinion shifts among neutral agents. Compared to the FJ baseline, Llama 3.2 interactions produce stronger and faster convergence toward biased positions. Beyond numerical opinion changes, our analysis revealed that semantic alignment in textual justifications increases consistently with biased exposure. This indicates that language convergence emerges alongside opinion shifts, even when some diversity in discrete opinions remains. These findings highlight the dual role of LLM interactions in shaping both beliefs and their expression, raising important considerations for the deployment of AI systems in socially sensitive contexts. Future work can extend this framework to heterogeneous networks, alternative prompting strategies, and interventions aimed at mitigating bias propagation.

## References

1. Bakshy, E., Messing, S., Adamic, L.A.: Exposure to ideologically diverse news and opinion on facebook. Science 348(6239), 1130–1132 (2015)

2. Berjawi, O., Cavaliere, D., Fenza, G.: A multi-aspect analysis of echo chambers on video-sharing social media. In: International Conference on Advances in Social Networks Analysis and Mining. pp. 197–213. Springer (2024)

3. Berjawi, O., Khatoun, R., Fenza, G.: Analyzing the persuasive strategies of influencers and news media on social media. In: 2025 IEEE/ACS 22nd International Conference on Computer Systems and Applications (AICCSA). pp. 1–7. IEEE (2025)

4. Borah, A., Mihalcea, R.: Towards implicit bias detection and mitigation in multiagent llm interactions. In: Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 9306–9326 (2024)

5. Cheng, Z., Lin, Z., Yang, Y., Wei, Z., Chen, S.: Interactive simulation and visual analysis of social media event dynamics with llm-based multi-agent modeling. Visual Informatics p. 100260 (2025)

6. Chuang, Y.S., Goyal, A., Harlalka, N., Suresh, S., Hawkins, R., Yang, S., Shah, D., Hu, J., Rogers, T.: Simulating opinion dynamics with networks of llm-based agents. In: Findings of the association for computational linguistics: NAACL 2024. pp. 3326–3346 (2024)

7. Cisneros-Velarde, P.: Biases in opinion dynamics in multi-agent systems of large language models: A case study on funding allocation. In: Findings of the Association for Computational Linguistics: NAACL 2025. pp. 1889–1916 (2025)

8. DeGroot, M.H.: Reaching a consensus. Journal of the American Statistical association 69(345), 118–121 (1974)

9. Ding, G., Liu, Z., Li, S., Cao, J., Ye, Z.: Impact of mindset types and social community compositions on opinion dynamics: A large language model-based multi-agent simulation study. Computers in Human Behavior 172, 108730 (2025)

10. Gallegos, I.O., Rossi, R.A., Barrow, J., Tanjim, M.M., Kim, S., Dernoncourt, F., Yu, T., Zhang, R., Ahmed, N.K.: Bias and fairness in large language models: A survey. Computational linguistics 50(3), 1097–1179 (2024)

11. Hegselmann, R., Krause, U.: Opinion dynamics and bounded confidence: Models, analysis and simulation. Journal of Artificial Societies and Social Simulation 5(3) (2002)

12. Lin, H.T., Huang, P.C., Ku, C.T., Hsu, C., Shieh, P.X., Kang, Y.: Towards simulating social influence dynamics with llm-based multi-agents. In: 2025 IEEE International Conference on Information Reuse and Integration and Data Science (IRI). pp. 307–312. IEEE (2025)

13. Mehdizadeh, A., Hilbert, M.: Homophily-induced emergence of biased structures in llm-based multi-agent ai systems. Social Network Analysis and Mining 15(1), 1–25 (2025)

14. Moscovici, S., Lage, E., Nafrechoux, M.: Influence of a consistent minority on the responses of a majority in a color perception task. Sociometry pp. 365–380 (1969)

15. Piao, J., Lu, Z., Gao, C., Xu, F., Hu, Q., Santos, F.P., Li, Y., Evans, J.: Emergence of human-like polarization among large language model agents. arXiv preprint arXiv:2501.05171 (2025)

16. Qu, Y., Wang, J.: Performance and biases of large language models in public opinion simulation. Humanities and Social Sciences Communications 11(1), 1–13 (2024)

17. Wang, C., Liu, Z., Yang, D., Chen, X.: Decoding echo chambers: Llm-powered simulations revealing polarization in social networks. In: Proceedings of the 31st international conference on computational linguistics. pp. 3913–3923 (2025)

18. Yao, J., Zhang, H., Ou, J., Zuo, D., Yang, Z., Dong, Z.: Social opinions prediction utilizes fusing dynamics equation with llm-based agents. Scientific Reports 15(1), 15472 (2025)

19. Zuo, D., Zhang, H., Ou, J., Feng, C., Liu, S.: Mtos: A llm-driven multi-topic opinion simulation framework for exploring echo chamber dynamics. arXiv preprint arXiv:2510.12423 (2025)