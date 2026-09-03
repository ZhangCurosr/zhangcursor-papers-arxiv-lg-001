# The Implications of Linguistic Illegibility for LLM Security

James Mickens Harvard University mickens@g.harvard.edu

## Abstract

LLMs are trained to generate natural language. However, various strands of evidence indicate that an LLM’s externalized linguistic outputs and mechanisticallyextracted linguistic features can be an unreliable lens for understanding internal model computation. We introduce the term linguistic illegibility to broadly refer to scenarios in which an LLM’s externalized or mechanistically-probed language artifacts fail to represent how the model actually thinks. We argue that the specter of linguistic illegibility is unavoidable for LLMs whose internal computations are not directly expressed via language, but rather math over activation spaces (with lossy translations between activation spaces and natural language happening at the bookends). If linguistic illegibility is always possible, then security mechanisms that rely on a model’s linguistic self-reporting (e.g., chain-of-thought monitoring, constitutional self-critique, activation probing for linguistically-defined feature vectors) can never be completely sound; the model sandbox will always need isolation techniques whose guarantees do not depend on reading a model’s linguistic state at all. We argue that observing a model’s outputs using taint tracking is a promising approach for an effective sandbox: regardless of how a model linguistically self-reports, a taint tracking policy can define, a priori, various pieces of system state that should never be influenced by model-produced data. We also discuss several additional sandboxing mechanisms (e.g., robust virtualization, third-party auditing of sandboxing configurations) which collectively provide a critical floor beneath linguistic monitoring, and would have mitigated recent sandbox exploits by frontier models.

## 1 Introduction

Large language models (LLMs) are increasingly entrusted with sensitive data, and increasingly empowered to autonomously interact with the physical world. Two critical security challenges arise. The first involves detection of misaligned intent: how can we detect when LLMs are planning to carry out behavior with harmful consequences? The second challenge is disruption of misaligned action: in situations where misaligned planning cannot be detected and derailed, how can we prevent a model from actually engaging with the external world in malicious ways?

In the classic literature on systems security, the disruption of misaligned action is called “sandboxing.” A sandbox ensures that code which tries to do harm cannot. Classic sandboxing techniques like system call filtering [20, 48], hardware virtualization [47, 72, 75], and network firewalling [32, 42, 64] are now employed in the LLM context too. For example, best practice dictates that an inference-as-aservice deployment should be decomposed using least privilege [61], such that each component (e.g., the inference engine, the RAG database, and the execution environment for model-generated code) is placed in a tightly-restricted container or VM [43, 44].

Detection of misaligned intent is also a goal of classic systems security. In the ideal world, harmful code would announce its goals using natural language that could be analyzed in straightforward ways. However, traditional programs (e.g., web browsers and video games) plan via code and externalize intent via system calls, managed runtime interactions, or other non-lingual means; as a result, intent detection traditionally relied on static code analysis [36, 52, 65] or dynamic observation of environmental interactions [7, 23, 83].

Unlike a traditional program, an LLM uses natural language as a primary interface with humans. Thus, a popular approach for detecting misaligned LLM intent is to treat a model’s own language outputs as a diagnostic signal. Chain-of-thought traces are examined for signs of reward hacking or deception [7, 40]; constitutional self-critique asks a model to evaluate its own outputs against principles formulated in natural language [6, 13]; activation probes are trained to detect linguistically-framed concepts such as truthfulness, harmfulness, or planned deception [10, 50, 51]. These approaches have an obvious intuitive appeal: a model trained to receive natural-language inputs and generate natural-language outputs should have internal behaviors that are legible via linguistic analyses.

However, the robustness of these approaches is tied to whether the linguistically-decodable aspects of an LLM’s state actually reflect the model’s true intent. We use the phrase “linguistically decodable” by way of analogy to the phrase “linearly decodable” that is often used in the mechanistic interpretability literature. A linearly-decodable feature of model cognition is one that can be detected via a linear function applied to a hidden layer’s activations. A linguistically-decodable feature is one that is defined in terms of natural language, and is extracted directly from a model’s natural-language outputs or indirectly via mechanistic probes like sparse autoencoders [9] or logit lenses [57] which map language-level concepts like “happy” or “conflicted” to functions of activation vectors.

So, are an LLM’s linguistically-decodable features guaranteed to reflect a model’s internal computations? A growing body of empirical work says “no,” and we argue that the problem is intrinsic to how LLMs are constructed (§2): there is no way to eliminate the possibility of semantic divergence between a model’s natural-language outputs and its true planning goals. This lingering threat is particularly alarming given the rapid increase in model capabilities. As those capabilities continue to improve, we argue that the AI safety community must pay more attention to isolation mechanisms that do not rely the extraction of natural-language observations from increasingly byzantine models.

In Section 3, we propose some concrete design principles for a systems-level, non-linguistic sandbox for misaligned frontier models. A core idea is to aggressively track how model-generated outputs are used to derive new data and influence critical pieces of system state. Such taint tracking is critical because in many cases we can define, a priori, system state that models should not influence regardless of the model’s ostensible or true planning process. For example, if a model should be isolated from the public internet, there is never a reason that model-influenced bytes should appear in an outbound network packet.

## 2 Linguistic Illegibility: Examples and Security Implications

## 2.1 When Can Illegibility Occur?

Chains of thought: Asking an LLM to explain its thought process leads to performance gains in many settings [39, 79]. This gain is intuitively reasonable: if LLMs internally use linguistic reasoning, then providing the model with a “linguistic scratchpad” should give the model additional opportunities to plan and self-correct. However, empirical work has shown that chains of thought are not necessarily a faithful representation of a model’s true planning process. For example:

• Turpin et al. demonstrated that if the chain-of-thought exemplars provided in system prompts contain a hidden bias (e.g., the correct answer to a multiple choice question is always “(a)”), models often do not mention those biases in chains of thought, and instead rationalize away incorrect answers that were clearly influenced by the bias [74].

• To further test the extent to which chains of thought can be post-hoc rationalizations for model output, Lanham et al. took real prompt+CoT pairs generated by an LLM, edited the CoT by truncating it, inserting a logically incorrect sentence, or correctly paraphrasing the original CoT, and then examined the resulting model output [45]. Model accuracy was only minimally affected by paraphrased CoTs. However, truncated or incorrect CoTs had different effects on different model tasks, sometimes hurting accuracy but sometimes having little impact; for the latter kinds of tasks, model reasoning could not have relied on the linguistic content of CoT tokens.

• When an LLM is trained via reinforcement learning, the model might try to “reward hack” (i.e., maximize its training reward by generating solutions that are misaligned with respect to human-desired goals, but nonetheless maximize the scoring function). Chen et al. found that Claude 3.7 Sonnet verbalized its use of discovered hacks less than 2% of the time in 5 out of 6 evaluation scenarios [14].

• Each token that a model generates is created via hidden-layer computations. Thus, asking a model to give a chain of thought before answering a question will give the model extra computational capacity regardless of whether the resulting CoT tokens are linguistically legible. As an extreme demonstration of this idea, Pfau et al. showed that a model whose chain of thought is just dots $( ^ { 6 6 } . . . ^ { 7 } )$ can nonetheless use the associated compute to solve problems that are intractable for a model with an equivalent architecture but no chain of thought [60].

Other examples of CoTs being linguistically illegible abound [4, 5].

Constitutional self-critique: Constitution-based alignment (CAI) methods use a natural-language document to define what “good model behavior” is; the document guides fine-tuning, with the goal of nudging the model into a more aligned state [6]. The basic CAI approach has two phases:

• A pretrained model M receives a prompt, generates a response, and is requested to critique the response according to alignment principles contained in a natural-language constitution. Examples of such principles might be “Do not harm users” or “Always be polite.” The model revises the response according to the constitution. The <original prompt, refined response> pair is then used to fine-tune M. This process is repeated across many pairs, producing the initial fine-tuned model M<sup>ˆ</sup> .

• M<sup>ˆ</sup> is iteratively provided a prompt and asked to generate two different responses. A separate judge model J ranks the two responses. The results of many such pairwise comparisons are used to train a reward model which is then used to refine M<sup>ˆ</sup> via reinforcement learning.

The potential illegibility of a model’s output tokens threatens the effectiveness of CAI. For example, suppose that the un-fine-tuned M is a “sleeper agent” model that emits harmful code only when a trigger word is in a prompt [34]. In phase one of CAI, when M is asked to critique its response for a triggering prompt, the model might affirm the principle “All code that I emit should be safe” while nonetheless emitting unsafe code. Later, in the second phase of CAI, M<sup>ˆ</sup> can try to reward-hack the evaluator model J, producing responses whose overt linguistic content please the judge but do not reveal M<sup>ˆ</sup> ’s true strategy for task completion [81].

Mechanistic interpretability: The goal of mechanistic interpretability is to observe a model’s internal activation vectors and identify the semantic concepts (or “features”) that are relevant to the model’s planning [54]. From the security perspective, mechanistic interpretability is an attractive foundation for runtime sandboxing: if a monitor can detect that a model is internally ruminating on unsafe features, the model can be halted or nudged towards less harmful features [70, 73].

A common assumption in mechanistic interpretability is the linear representation hypothesis, which states that a model represents features as directions in the activation space [59]. For example, let $\mathbf { h } _ { \ell , \mathbf { t } }$ represent a model’s hidden-layer activations at layer ℓ and forward pass t.

• We could train a linear probe [2] of the form $\mathbf { f } \ = \ s o f t m a x ( \mathbf { W } \mathbf { h } _ { \ell , \mathbf { t } } + \mathbf { b } )$ , where W is a learned matrix, b is a learned vector, and f is a probability vector which captures the likelihood that a particular feature is represented by an activation. For example, Gurnee and Tegmark demonstrated that Llama-2 has linearly-decodable features representing geographic locations for physical places such as landmarks, such that when prompted about a specific place, the longitude and latitude of that place are recoverable [29].

• A sparse autoencoder maps a hidden layer $\mathbf { h } _ { \ell , \mathbf { t } }$ of dimension d to a much larger feature space of dimension $f > d ;$ the encoding looks like $\mathbf { x } _ { } = g ( \mathbf { W _ { e } } \mathbf { h } _ { \ell , \mathbf { t } } + \mathbf { b _ { e } } )$ where g is a nonlinear activation function. The autoencoder then maps x back to the original dimensionality via the decoding projection $\hat { \mathbf { h } } _ { \ell , \mathbf { t } } = \mathbf { W } _ { \mathbf { d } } \mathbf { x } + \mathbf { b } _ { \mathbf { d } }$ . The autoencoder is called “sparse” because the function for training loss not only encourages reconstruction accuracy, but also rewards x values that have few non-zero entries; the hope is that the resulting autoencoder will be able to generate a sparse feature vector for the target model’s $\mathbf { h } _ { \ell , \mathbf { t } }$ . For example, Cunningham et al. built a sparse autoencoder for Pythia models and empirically identified various dimensions in the expanded space that corresponded to natural-language concepts like indirect objects [33].

• A logit lens [57] uses a transformer’s token vocabulary as the feature dictionary, and simply applies the transformer’s final-layer unembedding matrix U to a non-final-layer activation vector $\mathbf { h } _ { \ell , \mathbf { t } }$ . The probability vector $\mathbf { U h } _ { \ell , \mathbf { t } }$ is treated as the likelihood that various features (as named by tokens) are the focal areas of model planning at layer ℓ.

Crucially, all of these mechanistic probes are linguistically anchored in some way, searching for features that are defined in natural language. Logit lenses literally use the token vocabulary as the feature dictionary. With sparse autoencoders, the semantic meanings of dimensions in the feature space are determined by a second LLM tasked to observe how activations in each dimension are correlated with language-level concepts contained by prompts [8]. Even in the linear probe example above, where features ostensibly correspond to physical, numeric concepts like time, those notions are rooted in a model’s general language-based training data.

Because the starting point for a mechanistic probe is language, a probe is sensitive to the linguistic framing used. For example, consider research into how LLMs internally represent the state of an Othello game. Initial work observed that linear probes were less accurate than nonlinear ones at recovering game state, suggesting that LLMs built a nonlinear world model [46]. However, subsequent work [55] found that linear probes can be just as effective if they reflect the LLM’s own perspective on the game— during a particular turn in the Othello game, the LLM does not conceive of a cell as being “empty,” “black,” or “white,” but rather “empty,” “mine,” or “theirs” (where the color associated with “mine” and “theirs” flips during every turn).

From the security perspective, the sensitivity of mechanistic probes to the linguistic biases of the monitor is troubling. Ideal sandbox policies would be crisply specifiable using formal semantics for good and bad behavior.

## 2.2 The Security Implications of Linguistic Illegibility

The bulk of an LLM’s computation is linear and nonlinear math over real-valued activation vectors. Only during input embedding and output unembedding does the model explicitly manipulate reified linguistic constructs. Prior research on feature superpositioning suggests that a model’s internal representation space contains more usable features than the size of the token vocabulary space [9]. Thus, we are confronted with a fundamental mismatch of capacity: there are more unique features than unique tokens, meaning that there must be some features which defy a reasonable single-token description. If we could automatically enumerate such features, we could at least empirically observe which kinds of linguistic inputs most strongly trigger those features. However, the number of possible feature directions in even a moderately-sized LLM is vast [24, 27]. The specter of linguistic illegibility therefore seems to be inescapable, at least with our current understanding of internal model dynamics.

The implication for model sandboxing is not that language-based oversight is worthless. However, such oversight must be layered atop isolation mechanisms which do not inherit the linguistic illegibility of the model itself. Recent sandbox escapes by frontier models [3, 12, 58] further suggest that commodity isolation mechanisms like containerization are woefully inadequate for sandboxing highly-capable models. We propose approaches that we think are more robust in the next section.

## 3 A Stronger Sandbox

## 3.1 Threat Model

We define an agent as an LLM that runs atop an inference engine, with an orchestrator allowing the inference engine to interact with the outside world (e.g., via MCP calls). In our threat model, the primary adversary is the agent itself, so our core security goal is to prevent the agent from externalizing harmful behavior that impacts entities outside the agent’s sandbox. For example, a misaligned model might try to exfiltrate its weights, exchange network traffic with non-allowlisted servers, or modify audit logs associated with its own execution.

Our threat model is directly motivated by recent sandbox escapes of models being evaluated:

• Example 1: An OpenAI agent escaped a network-isolated sandbox by exploiting a zero-day vulnerability in a trusted package-manager server that was accessible to the model and had open Internet access itself [18, 58].

Table 1: The design principles for our proposed sandbox.
<table><tr><td>Design principle</td><td>Mechanism</td><td>Sandbox failure addressed</td></tr><tr><td>Provenance labeling</td><td>Taint tracking, information flow control</td><td>Undetected model influence on sensitive state</td></tr><tr><td>Robust virtualization</td><td>Hardware-enabled TEEs, verifiable microkernels</td><td>Enforcement bypass via privilege escalation</td></tr><tr><td>Egress mediation</td><td>Minimal default-deny proxy w/no agent credentials</td><td>Confused-deputy compromise of trusted infrastructure</td></tr><tr><td>Behavioral anomaly detection</td><td>Statistical monitoring of permitted output streams</td><td>Information laundering (e.g., steganography), behavioral phase shifts</td></tr><tr><td>Configuration attestation</td><td>Pre-launch validation by third parties</td><td>Inconsistency between policy goals and policy implementation</td></tr></table>

• Example 2: Anthropic observed that Mythos could leverage a multi-step browser exploit to escape from the browser’s renderer sandbox and then the OS’s container sandbox [12].

• Example 3: Anthropic separately observed three cases in which models escaped their sandboxes due to configuration errors in the sandbox environment (as opposed to enforcement failures as the other examples we list above). The models then hacked into servers on the open Internet [3].

In this section, we focus on security guarantees that can be provided even if linguistically-rooted oversight fails, so we mostly ignore language-based oversight mechanisms. In practice, we expect (and encourage!) language-based monitors to form an inner layer of the sandbox.

## 3.2 Architectural Overview

Our proposed sandboxing architecture is guided by five principles, as shown in Table 1. We discuss each principle (and its consequences for the sandbox architecture) below, and describe the open research questions. The first principle (provenance labeling) has the most subtle design consequences, so we focus most of our discussion on this principle.

Provenance labeling: Linguistic illegibility challenges our ability to understand a model’s intent. But regardless of whether we can understand model intent, we can still define various system-level interactions that should be forbidden. An important kind of unsafe interaction involves agentic influence over sensitive bytes in memory or in persistent storage devices. For example, an agent that can send data to unauthorized OS-level sinks like file systems and network sockets can inappropriately memorialize data or attack remote endpoints. As another example, if an agent can overwrite intraprocess state (e.g., return addresses) that belong to inference-stack components like package managers or RAG databases, the agent can subvert the control flow of those components [1, 67].

Our proposed sandbox uses taint tracking [25, 56, 66] to taint model outputs and observe how they influence other system state. The goal is to (1) define a set of sinks which should not receive modelinfluenced data, and (2) define introspection points, akin to LSM hooks [80], for which security policies can register interest. An initial implementation of the sandbox would run on commodity hardware and define sinks with respect to OS-level abstractions like processes, files, and network sockets; TaintCheck [56] and TaintDroid [25] are conceptual representatives of the proposed approach. Ultimately, we would want hardware-level tainting support [76, 78] to efficiently understand how agent output influences both OS-level sinks and hardware-level, intra-process state for components residing outside of the narrow agent sandbox (but inside the inference-as-a-service deployment).

Note that the last sentence implies that the full deployment environment runs atop a taint-tracking framework. This requirement is non-trivial but reasonable if one takes the threat of superintelligent misaligned agents seriously; being able to identify unexpected model influence throughout the inference stack is critically important.

The taint sinks defined in the classic literature (e.g., system-call-level abstractions, intra-process control-flow state like return addresses and function pointers) are all relevant in our scenario. However, what should be the taint sources, and how should detainting occur? A naive approach would not support detainting, and would use single-bit taint tags which indicate whether any of the model’s output tokens have influenced the tagged state. The risk of this straightforward approach is overtainting [17, 31, 69] in which the system monotonically accumulates taint (such that the taint tags gradually become less useful to security policies). To avoid these problems, we propose causal tagging ofoutput tokens and auditable detainting. We describe these concepts below.

At a high level, causal tagging allows the sandbox to have many taint tags. Some tags are associated with actors (e.g., users who supply prompts, or different instances of deployed agents); other tags are associated with specific resources like files in a RAG database or outputs from a tool invocation. An output token from an agent receives the union of tags from the relevant actors and resources that the agent’s model consulted to generate the token. For example, some output tokens might be influenced (or not influenced) by the system prompt, the user prompt, the output of a specific tool call, or the contents of a document fetched by RAG. Other tokens might be primarily influenced by a model’s intrinsic biases towards generating certain output patterns. By tagging each output token with the information used by the model to derive that token, we enable rich sandboxing policies, e.g.,

• No agentic output that is tainted by documents from an internal RAG database should leave the internal network: This policy thwarts agent-laundered exfiltration of sensitive corporate information.

• An output sink owned by entity e should only receive agentic output that is tainted by inputs from tools that are also owned by e: For example, if each tool is tagged with the identity of a particular customer, and each output sink is also tagged with the identity of the owning customer, then no customer can receive information derived from data belonging to a different customer.

• A privileged deputy, when provided with tainted inputs, can only perform actions that are allowed by the capabilities of the taint sources: For example, a tool gateway that internally manages upstream API keys should only use a key in response to inputs that were generated by authorized users of that key.

Besides these contextual policies, the sandbox can also support more straightforward categorical policies (e.g., “no model-tainted bytes should ever reach a particular sink”). Attempted policy violations should be logged at a minimum, and (depending on the seriousness of the violation) should be grounds for halting the agent or buffering its outputs for asynchronous analysis before the outputs are potentially detainted and externalized.

The best algorithm for assigning tags to output tokens is an open research question. We are exploring three different approaches, all of which assume that input tokens are pretagged and that tags are floating point numbers between 0.0 and 1.0 representing the strength of a taint source’s influence.

• With attention-based output tagging, the inference engine looks at the transformer’s attention patterns over input tokens. During a forward pass, tags from strongly-attended tokens propagate their taint more aggressively to the new output token.

• Alternatively, the inference engine might calculate the Jacobian or the integrated gradient [71] of the LLM’s output logit with respect to the input tokens, and scale propagation of gradient tags with respect to the magnitude of an input token’s gradient.

• With counterfactual-based output tagging, the impact of a particular input token on an output token (or full response) is determined via an ablation experiment in which the model is rolled back to an earlier snapshot, the token in question is replaced by a filler token, and the model’s forward passes are re-executed to generate a new response [11]. If the original response and the counterfactual one differ significantly (e.g., output different tool calls or contain significantly divergent linguistic content), the ablated input token propagates its tag to the agent’s output.

Attention-based tagging can run inline with a normal forward pass, but will not always provide true causal attribution [35, 62]. Counterfactual-based tagging does, but requires additional forward passes and thus is better suited for batch validation of a buffered multi-token response.

Robust virtualization: Taint propagation, sink monitoring, and policy enforcement must execute in a strictly more-privileged layer than the agent’s own. For example, the agent could run atop sel4 [37], a verified microkernel with provable guarantees of isolation and noninterference. Alternatively (or additionally), agents could be executed within hardware-rooted virtualization layers with information flow tracking and policy decisions implemented by code that hardware prevents the agent from modifying. In either case, the goal is to ensure that an agent which has achieved full privilege escalation within its guest cannot disrupt sandbox machinery.

Egress mediation: All of an agent’s outbound network activity is forced through a minimal network proxy. The proxy rejects all connections by default, only allowing those enabled by a manifest signed by the first-party inference operator and a third-party validator (see our discussion of configuration attestation below). The proxy only acts as a firewall, and contains no additional functionality like package management; the goal is to reduce the proxy’s threat surface.

Behavioral anomaly detection: Taint tracking cannot detect implicit flows [31] in which information is externalized not through tainted bytes, but through a choice of which untainted bytes to externalize; for example, agents may try to launder information through token-distribution biasing [68] or zero-width Unicode formatting [77], or through selective invocation of seemingly innocuous tool. An agent that is not trying to leak information, but simply disrupt internal or remote systems, may also exhibit phase changes in behavior, e.g., suddenly issuing a large number of tool calls during the reconnaissance phase of remote server exploitation. A robust sandbox must therefore perform statistical analysis of the content in ostensibly low-risk output channels, and rate-limit, pause, or subject to greater scrutiny models that are deviating from expected behavioral baselines.

Configuration attestation: Before a model is deployed (whether for training, testing, or real inference serving), the sandbox configuration must be independently verified by an organizationally and infrastructurally separate entity. During the verification period, the verifier has complete access to the computational infrastructure of the first-party agent vendor, allowing the verifier to (for example) run test agents directly on the first-party infrastructure and perform red-teaming on that infrastructure. Once verification is done, the sandboxing hardware and software expect the sandboxing configuration to be co-signed by both the independent third-party and the agent’s first-party vendor. Any changes to that configuration must also be co-signed.

## 3.3 Effectiveness and Limitations

Returning to Section 3.1’s examples of real-life sandbox escapes, we see that our proposed sandbox infrastructure would mitigate or outright prevent those escapes:

• Example 1: Even if the OpenAI agent was able to subvert the package manager, the manager would not have direct access to the public Internet. Instead, the manager would connect to the minimal network proxy; the uncompromised proxy (or if that is compromised, the uncompromized taint-tracking system) would detect that model-influenced bytes were trying to leave the internal network, and raise an alarm. Behavioral analysis would also flag that abnormally large number of tool calls that misaligned agents issued while probing external systems.

• Example 2: Mythos could still escape from the browser-level sandbox, but attempts to escape from the virtualization-level sandbox would fail (e.g., due to the formal security guarantees of sel4, or the inability of Mythos to corrupt isolation mechanisms protected by the hardware itself).

• Example 3: Anthropic’s intended sandbox configuration was correct; the deployed configuration was not. Configuration attestation would detect the mismatch and force Anthropic to fix the problem before deploying the models atop the (now properly-configured) sandbox. Taint tracking would allow the sandbox to prevent model-influenced bytes from traversing sinks connected to the public Internet.

In summary, we think that the proposed sandbox would materially increase the difficulty of subverting isolation boundaries.

That being said, building the proposed sandbox is not trivial. For example, configuration attestation will require careful technical and legal design to ensure the integrity of first-party infrastructure, respect intellectual property, and establish clear liability regimes. The anomaly detection layer is fundamentally statistical, meaning that agents can try to evade it by encoding problematic behavior in ways that nonetheless satisfy the sandbox’s statistical understanding of normal behavior. Guarantees provided by formally verified hypervisors like sel4 are premised on the correctness of the proofs; in turn, proof correctness relies on the trustworthiness of the compiler and the hardware which executes the compiled code. Hardware-level side channels [38, 49] are also a constant concern, although they can be mitigated by reducing or eliminating microarchitectural co-tenancy [30, 63].

Another important caveat is that taint tracking can monitor explicit flows but not implicit ones (§ 3.2). The behavioral anomaly detector can identify some implicit flows, but only in a probabilistic sense. If we replaced taint tracking with information flow control [21, 41, 82, 22], the sandbox could detect such implicit flows; however, monitored applications would have to be (1) written in an IFC language like Jif [53], or (2) decomposed into privileged-separated components involving non-secret workers which send requests to secret-handling workers, who then compute over sensitive data and give the results to a trusted declassifier for subsequent return to non-secret workers [41]. In the short term, rewriting the inference stack to use IFC seems more daunting than porting legacy applications to taint tracking frameworks, but ultimately, an IFC inference stack could enforce policies with much lower runtime overhead—taint tracking essentially requires every assignment operation in a program to be instrumented [26], whereas IFC allows inappropriate data transfers to be caught statically or with simple runtime checks enforceable by the kernel during system calls.

## 4 Related Work

Dynamic taint tracking: This technique has a rich history in the literature. For example, see prior work on whole-system taint tracking for exploit detection [15], control-flow-aware propagation policies to combat overtainting [31], and hardware-assisted taint propagation to decrease runtime overhead [76, 78]. Classic taint tracking examines data flows in programs with imperative def-use updates of discrete program variables; in LLMs, program state is more diffuse, consisting of highdimensional vectors whose intrinsic meaning is less obvious and that are updated by continuouslyvalued mathematical operations.

Information flow control for LLM agents: Several recent systems leverage IFC to improve LLM security. FIDES [16] attaches confidentiality and integrity labels to orchestrator inputs and outputs like tool arguments and results, preventing untrusted inputs from causing prompt injection attacks. FIDES also prevents the explicit leaking of sensitive data, but does not prevent implicit leaks because agents are allowed to make control-flow decisions on secret data. NeuroTaint [11] detects implicit flows involving secret data using a counterfactual probe that resembles the one discussed in Section 3.2. CaMeL [19] handles implicit flows by forcing the model to emit a concrete action plan in the form of Python code; CaMeL executes the plan in a modified Python interpreter that enforces IFC policies. FIDES, NeuroTaint, and CaMeL focus on how untrusted data might trick a trusted model into behaving inappropriately; in contrast, our focus is on sandboxing a full inference deployment where the model itself might be intrinsically misaligned.

AI control: This line of work assumes that a model can be misaligned, and investigates oversight protocols in which the output of the untrusted model is reviewed by a weaker but trusted model (or by a strong, untrusted model deployed atop an evaluation framework which discourages collusion between the two models) [28]. These approaches fundamentally require analysis of a model’s natural-language content, and thus are exposed to validity threats posed by linguistic illegibility.

## 5 Conclusion

Linguistic illegibility is the semantic gap between what a model’s language output reveals and what a model’s internal computation actually does. We argue that linguistic illegibility is always possible with models that do not intrinsically compute over reified language concepts, but instead compute over activation spaces that are lossily initialized from and converted to natural language at the bookends of model computation. If linguistic illegibility is inevitable in at least some situations, then sandboxing methods that are rooted in natural-language analysis (e.g., chain-of-thought analysis, constitutional self-critique, and mechanistic probing) must be supplemented by systems-level mechanisms whose policies are not rooted in natural-language analysis. Furthermore, if the threat model includes the possibility that models are competitive with or surpass human capabilities, the systems-level sandbox must be relentlessly paranoid and incredibly robust: paranoid because the sandbox must track a wide variety of model behaviors, and robust because the sandbox must not be subvertible by a highly-capable model. We think that our proposed design principles are a good starting point for what a paranoid and robust sandbox should look like.

## Acknowledgments

This work was partially supported by funding from Georgetown University’s Center for Security and Advanced Technology, and from Coefficient Giving (formerly known as Open Philanthropy).

## References

[1] Martín Abadi, Mihai Budiu, Úlfar Erlingsson, and Jay Ligatti. Control-Flow Integrity Principles, Implementations, and Applications. ACM Transactions on Information and System Security, 13(1):1–40, 2009.

[2] Guillaume Alain and Yoshua Bengio. Understanding Intermediate Layers Using Linear Classifier Probes. arXiv preprint arXiv:1610.01644, 2016.

[3] Anthropic. Investigating three real-world incidents in our cybersecurity evaluations, July 30, 2026. Anthropic blog post. https://www.anthropic.com/news/ investigating-incidents-cybersecurity-evals.

[4] Ivan Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, and Arthur Conmy. Chain-of-Thought Reasoning In The Wild Is Not Always Faithful. arXiv preprint arXiv:2503.08679, 2025.

[5] Benjamin Arnav, Pablo Bernabeu-Pérez, Nathan Helm-Burger, Tim Kostolansky, Hannes Whittingham, and Mary Phuong. CoT Red-Handed: Stress Testing Chain-of-Thought Monitoring. arXiv preprint arXiv:2505.23575, 2025.

[6] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional AI: Harmlessness from AI Feedback. arXiv preprint arXiv:2212.08073, 2022.

[7] Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. arXiv preprint arXiv:2503.11926, 2025.

[8] Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. Language models can explain neurons in language models, 2023. https://openaipublic.blob.core.windows.net/ neuron-explainer/paper/index.html.

[9] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards Monosemanticity: Decomposing Language Models with Dictionary Learning, 2023. https://transformer-circuits.pub/2023/ monosemantic-features.

[10] Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering Latent Knowledge in Language Models Without Supervision. arXiv preprint arXiv:2212.03827, 2022.

[11] Yuandao Cai, Wensheng Tang, Cheng Wen, and Shengchao Qin. Ghost in the Agent: Redefining Information Flow Tracking for LLM Agents. arXiv preprint arXiv:2604.23374, 2026.

[12] Nicholas Carlini, Newton Cheng, Keane Lucas, Michael Moore, Milad Nasr, Vinay Prabhushankar, Winnie Xiao, Hakeem Angulu, Evyatar Ben Asher, Jackie Bow, Keir Bradwell, Ben Buchanan, David Forsythe, Daniel Freeman, Alex Gaynor, Xinyang Ge, Logan Graham, Kyla Guru, Hasnain Lakhani, Matt McNiece, Mojtaba Mehrara, Renee Nichol, Adnan Pirzada, Sophia Porter, Andreas Terzis, and Kevin Troy. Assessing Claude Mythos Preview’s cybersecurity capabilities, April 7, 2026. Anthropic research post. https://www.anthropic.com/ research/mythos-preview.

[13] Antonio-Gabriel Chacón Menke and Phan Xuan Tan. How Effective Is Constitutional AI in Small LLMs? A Study on DeepSeek-R1 and Its Peers. arXiv preprint arXiv:2503.17365, 2025.

[14] Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Sam Bowman, Jan Leike, Jared Kaplan, and Ethan Perez. Reasoning Models Don’t Always Say What They Think, 2025. https://www-cdn.anthropic.com/ b9ca6db27f02a9ddf0d4fdb51b26432c99a27be0.pdf.

[15] Jim Chow, Ben Pfaff, Tal Garfinkel, Kevin Christopher, and Mendel Rosenblum. Understanding Data Lifetime via Whole System Simulation. In Proceedings of the 13th USENIX Security Symposium, 2004.

[16] Manuel Costa, Boris Kopf, Aashish Kolluri, Andrew Paverd, Mark Russinovich, Ahmed Salem, Shruti Tople, Lukas Wutschitz, and Santiago Zanella-Beguelin. Securing AI Agents with Information-Flow Control. arXiv preprint arXiv:2505.23643, 2025.

[17] Michael Dalton, Hari Kannan, and Christos Kozyrakis. Tainting Is Not Pointless. ACM SIGOPS Operating Systems Review, 44(2):88–92, 2010.

[18] Michael Dalton and Eric Wallace. The OpenAI–Hugging Face Incident. Black Hat USA 2026. https://www.youtube.com/watch?v=87DyyMV0kCY.

[19] Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, and Florian Tramèr. Defeating Prompt Injections by Design. arXiv preprint arXiv:2503.18813, 2025.

[20] Nicholas DeMarinis, Kent Williams-King, Di Jin, Rodrigo Fonseca, and Vasileios P. Kemerlis. sysfilter: Automated System Call Filtering for Commodity Software. In Proceedings of the 23rd International Symposium on Research in Attacks, Intrusions and Defenses, 2020.

[21] Dorothy E. Denning. A Lattice Model of Secure Information Flow. Communications ofthe ACM, 19(5):236–243, 1976.

[22] Petros Efstathopoulos, Maxwell Krohn, Steve VanDeBogart, Cliff Frey, David Ziegler, Eddie Kohler, David Mazières, Frans Kaashoek, and Robert Morris. Labels and Event Processes in the Asbestos Operating System. In Proceedings of the 20th ACM Symposium on Operating Systems Principles, pages 17–30, 2005.

[23] Manuel Egele, Theodoor Scholte, Engin Kirda, and Christopher Kruegel. A Survey on Automated Dynamic Malware-Analysis Techniques and Tools. ACM Computing Surveys, 44(2):6:1– 6:42, 2012.

[24] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy Models of Superposition. arXiv preprint arXiv:2209.10652, 2022.

[25] William Enck, Peter Gilbert, Byung-Gon Chun, Landon P. Cox, Jaeyeon Jung, Patrick McDaniel, and Anmol N. Sheth. TaintDroid: An Information-Flow Tracking System for Realtime Privacy Monitoring on Smartphones. ACM Transactions on Computer Systems, 32(2):1–29, 2014.

[26] John Galea and Daniel Kroening. The Taint Rabbit: Optimizing Generic Taint Analysis with Dynamic Fast Path Generation. In Proceedings of Asia CCS, pages 622–636, 2020.

[27] Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093, 2024.

[28] Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, and Fabien Roger. AI Control: Improving Safety Despite Intentional Subversion. arXiv preprint arXiv:2312.06942, 2023.

[29] Wes Gurnee and Max Tegmark. Language Models Represent Space and Time. In Proceedings ofthe International Conference on Learning Representations, 2024.

[30] Jana Hofmann, Cédric Fournet, Boris Köpf, and Stavros Volos. Gaussian Elimination of Side-Channels: Linear Algebra for Memory Coloring. In Proceedings of CCS, pages 2799–2813, 2024.

[31] Kevin Hough and Jonathan Bell. Conflux: A Practical Approach for Dynamic Taint Tracking with Control-Flow Relationships. ACM Transactions on Software Engineering and Methodology, 31(2):1–43, 2022.

[32] Kevin Hsieh, Mike Wong, Santiago Segarra, Sathiya Kumaran Mani, Trevor Eberl, Anatoliy Panasyuk, Ravi Netravali, Ranveer Chandra, and Srikanth Kandula. NetVigil: Robust and Low-Cost Anomaly Detection for East-West Data Center Security. In Proceedings ofthe 21st USENIX Symposium on Networked Systems Design and Implementation, 2024.

[33] Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, and Lee Sharkey. Sparse Autoencoders Find Highly Interpretable Features in Language Models. In The Twelfth International Conference on Learning Representations, 2024.

[34] Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera Lanham, Daniel M. Ziegler, Tim Maxwell, Newton Cheng, Adam Jermyn, Amanda Askell, Ansh Radhakrishnan, Cem Anil, David Kristjanson Duvenaud, Deep Ganguli, Fazl Barez, Jack Clark, Kamal Ndousse, Kshitij Sachan, Michael Sellitto, Mrinank Sharma, Nova DasSarma, Roger Grosse, Shauna Kravec, Yuntao Bai, Zachary Witten, Marina Favaro, Jan Markus Brauner, Holden Karnofsky, Paul Francis Christiano, Samuel R. Bowman, Logan Graham, Jared Kaplan, Sören Mindermann, Ryan Greenblatt, Buck Shlegeris, Nicholas Schiefer, and Ethan Perez. Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training. arXiv preprint arXiv:2401.05566, 2024.

[35] Sarthak Jain and Byron C. Wallace. Attention Is Not Explanation. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3543–3556, 2019.

[36] Johannes Kinder, Stefan Katzenbeisser, Christian Schallhart, and Helmut Veith. Detecting Malicious Code by Model Checking. In Detection ofIntrusions and Malware, and Vulnerability Assessment, volume 3548 of Lecture Notes in Computer Science, pages 174–187. Springer, 2005.

[37] Gerwin Klein, Kevin Elphinstone, Gernot Heiser, June Andronick, David Cock, David Derrin, Dhammika Elkaduwe, Kai Engelhardt, Rafal Kolanski, Michael Norrish, Thomas Sewell, Harvey Tuch, and Simon Winwood. seL4: Formal Verification of an OS Kernel. In Proceedings ofthe ACM SIGOPS 22nd Symposium on Operating Systems Principles, pages 207–220, 2009.

[38] Paul Kocher, Daniel Genkin, Daniel Gruss, Werner Haas, Mike Hamburg, Moritz Lipp, Stefan Mangard, Thomas Prescher, Michael Schwarz, and Yuval Yarom. Spectre Attacks: Exploiting Speculative Execution. Communications of the ACM, 63(7):93–101, 2020.

[39] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large Language Models are Zero-Shot Reasoners. In Advances in Neural Information Processing Systems, 2022.

[40] Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David Farhi, Ryan Greenblatt, and Dan Hendrycks. Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety. arXiv preprint arXiv:2507.11473, 2025.

[41] Maxwell Krohn, Alexander Yip, David Brodley, M. Frans Kaashoek, Robert Morris, and Ted Wobber. Information Flow Control for Standard OS Abstractions. In Proceedings ofthe 21st ACM SIGOPS Symposium on Operating Systems Principles, pages 321–334, 2007.

[42] Kubernetes. Network Policies. https://kubernetes.io/docs/concepts/ services-networking/network-policies/. Accessed on August 25, 2026.

[43] LangChain. How to Choose the Right Sandbox for AI Agents. https://www.langchain. com/blog/how-to-choose-the-right-sandbox-for-your-agent. Accessed on August 25, 2026.

[44] LangChain. Security Policy. LangChain Docs. https://docs.langchain.com/oss/ python/security-policy. Accessed on August 25, 2026.

[45] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukoši ˙ ut¯ e, Ka-˙ rina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Measuring Faithfulness in Chain-of-Thought Reasoning. arXiv preprint arXiv:2307.13702, 2023.

[46] Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task. arXiv preprint arXiv:2210.13382, 2022.

[47] Zhiyuan Liang, Tianzheng Li, and Enfang Cui. RISC-V Virtualization: Exploring Virtualization in an Open Instruction Set Architecture. In Proceedings of the 2024 5th International Conference on Computing, Networks and Internet of Things, pages 473–477, 2024.

[48] Linux man-pages project. seccomp(2) — Linux Manual Page. https://man7.org/linux/ man-pages/man2/seccomp.2.html. Accessed August 25, 2026.

[49] Moritz Lipp, Michael Schwarz, Daniel Gruss, Thomas Prescher, Werner Haas, Anders Fogh, Jann Horn, Stefan Mangard, Paul Kocher, Daniel Genkin, Yuval Yarom, and Mike Hamburg. Meltdown: Reading Kernel Memory from User Space. In Proceedings of the 27th USENIX Security Symposium, pages 973–990, 2018.

[50] Monte MacDiarmid, Timothy Maxwell, Nicholas Schiefer, Jesse Mu, Jared Kaplan, David Duvenaud, Sam Bowman, Alex Tamkin, Ethan Perez, Mrinank Sharma, Carson Denison, and Evan Hubinger. Simple Probes Can Catch Sleeper Agents. Anthropic Alignment Science research post, 2024. https://www.anthropic.com/research/probes-catch-sleeper-agents.

[51] Samuel Marks and Max Tegmark. The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets. In Proceedings of the Conference on Language Modeling, 2024.

[52] Andreas Moser, Christopher Kruegel, and Engin Kirda. Limits of Static Analysis for Malware Detection. In Proceedings of the 23rd Annual Computer Security Applications Conference, pages 421–430, 2007.

[53] Andrew C. Myers. JFlow: Practical Mostly-Static Information Flow Control. In Proceedings of the 26th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages, pages 228–241, 1999.

[54] Neel Nanda. A Comprehensive Mechanistic Interpretability Explainer & Glossary, 2023. https://www.neelnanda.io/mechanistic-interpretability/glossary.

[55] Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent Linear Representations in World Models of Self-Supervised Sequence Models. arXiv preprint arXiv:2309.00941, 2023.

[56] James Newsome and Dawn Song. Dynamic Taint Analysis for Automatic Detection, Analysis, and Signature Generation of Exploits on Commodity Software. In Proceedings of the Network and Distributed System Security Symposium, 2005.

[57] nostalgebraist. Interpreting GPT: The Logit Lens. LessWrong post, 2020. https://www. lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

[58] OpenAI. OpenAI and Hugging Face partner to address security incident during model evaluation, July 21, 2026. OpenAI blog post. https://openai.com/index/ hugging-face-model-evaluation-security-incident/.

[59] Kiho Park, Yo Joong Choe, and Victor Veitch. The Linear Representation Hypothesis and the Geometry of Large Language Models. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 39643–39666, 2024.

[60] Jacob Pfau, William Merrill, and Samuel R. Bowman. Let’s Think Dot by Dot: Hidden Computation in Transformer Language Models. arXiv preprint arXiv:2404.15758, 2024.

[61] Niels Provos, Markus Friedl, and Peter Honeyman. Preventing Privilege Escalation. In Proceedings of the 12th USENIX Security Symposium, 2003.

[62] Danish Pruthi, Mansi Gupta, Bhuwan Dhingra, Graham Neubig, and Zachary C. Lipton. Learning to Deceive with Attention-Based Explanations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4782–4793, 2020.

[63] Kartik Ramkrishnan, Stephen McCamant, Antonia Zhai, and Pen Chung Yew. Partitioned Tags, Shared Data: Reconciling Strict Cache Isolation with Write-Shared Coherence. arXiv preprint arXiv:2606.12259, 2026.

[64] Karen Scarfone and Paul Hoffman. Guidelines on Firewalls and Firewall Policy. Special Publication 800-41 Revision 1, National Institute of Standards and Technology, 2009.

[65] Suzanna Schmeelk, Junfeng Yang, and Alfred Aho. Android Malware Static Analysis Techniques. In Proceedings of the 10th Annual Cyber and Information Security Research Conference, 2015.

[66] Edward J. Schwartz, Thanassis Avgerinos, and David Brumley. All You Ever Wanted to Know About Dynamic Taint Analysis and Forward Symbolic Execution (but Might Have Been Afraid to Ask). In Proceedings ofthe 2010 IEEE Symposium on Security and Privacy, pages 317–331, 2010.

[67] Hovav Shacham. The Geometry of Innocent Flesh on the Bone: Return-into-libc without Function Calls (on the x86). In Proceedings ofthe 14th ACM Conference on Computer and Communications Security, 2007.

[68] Daniel Silva, Frederic Sala, and Ryan Gabrys. Look Who’s Talking Now: Covert Channels From Biased LLMs, 2024.

[69] Asia Slowinska, Traian Stancescu, and Herbert Bos. Pointless Tainting?: Evaluating the Practicality of Pointer Tainting. In Proceedings of the 4th ACM European Conference on Computer Systems, pages 61–74, 2009.

[70] Alessandro Stolfo, Vidhisha Balachandran, Safoora Yousefi, Eric Horvitz, and Besmira Nushi. Improving Instruction-Following in Language Models through Activation Steering. arXiv preprint arXiv:2410.12877, 2023.

[71] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70, pages 3319–3328, 2017.

[72] Andrew Tichon. Hardware Virtualization: The Foundations of Cloud Computing. Computer, 50(1):65–69, 2017.

[73] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering Language Models With Activation Engineering. arXiv preprint arXiv:2308.10248, 2023.

[74] Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. In Advances in Neural Information Processing Systems, 2023.

[75] Rich Uhlig, Gil Neiger, Dion Rodgers, Amy L. Santoni, Fernando C. M. Martins, Andrew V. Anderson, Steven M. Bennett, Alain K"agi, Felix H. Leung, and Larry Smith. Intel Virtualization Technology. Computer, 38(5):48–56, 2005.

[76] Guru Venkataramani, Ioannis Doudalis, Yan Solihin, and Milos Prvulovic. FlexiTaint: A Programmable Accelerator for Dynamic Taint Propagation. In Proceedings of the 14th International Symposium on High Performance Computer Architecture, pages 173–184, 2008.

[77] Guangnian Wan, Xinyin Ma, Gongfan Fang, and Xinchao Wang. Invisible Safety Threat: Malicious Finetuning for LLM via Steganography. arXiv preprint arXiv:2603.08104, 2026.

[78] Yu Wang, Jinting Wu, Haodong Zheng, Zhenyu Ning, Boyuan He, and Fengwei Zhang. Raft: Hardware-assisted Dynamic Information Flow Tracking for Runtime Protection on RISC-V. In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, pages 595–608, 2023.

[79] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, 2022.

[80] Chris Wright, Crispin Cowan, Stephen Smalley, James Morris, and Greg Kroah-Hartman. Linux Security Modules: General Security Support for the Linux Kernel. In Proceedings of the 11th USENIX Security Symposium, 2002.

[81] Wenkai Yang, Shiqi Shen, Guangyao Shen, Wei Yao, Yong Liu, Gong Zhi, Yankai Lin, and Ji-Rong Wen. Super(ficial)-alignment: Strong Models May Deceive Weak Models in Weak-to-Strong Generalization. In Proceedings of the International Conference on Learning Representations, 2025.

[82] Nickolai Zeldovich, Silas Boyd-Wickizer, Eddie Kohler, and David Mazières. Making Information Flow Explicit in HiStar. In Proceedings of the 7th USENIX Symposium on Operating Systems Design and Implementation, pages 263–278, 2006.

[83] Zhaoqi Zhang, Panpan Qi, and Wei Wang. Dynamic Malware Analysis with Feature Engineering and Feature Learning. arXiv preprint arXiv:1907.07352, 2019.