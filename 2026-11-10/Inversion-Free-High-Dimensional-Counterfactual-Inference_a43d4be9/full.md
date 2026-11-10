# Inversion-Free High-Dimensional Counterfactual Inference

Saketh Vishnubhatla<sup>∗</sup>, Pingchuan Ma<sup>∗</sup>, Shu Wan<sup>∗</sup>, Alimohammad Beigi<sup>∗</sup>, Andre Harrison<sup>†</sup>, Adrienne Raglin<sup>†</sup>, Huan Liu<sup>∗</sup>

<sup>∗</sup>School of Computing and Augmented Intelligence, Arizona State University, Tempe, AZ 85281, USA

{svishnu6, pingchua, swan16, abeigi, huanliu}@asu.edu

<sup>†</sup>DEVCOM Army Research Laboratory, Adelphi, MD 20783, USA

{andre.v.harrison2.civ, adrienne.raglin2.civ}@army.mil

Abstract—Counterfactual inference is the task of answering hypothetical “what-if” queries about a unit, given an observational dataset, modeling assumptions, and a causal graph. This enables a modeler to reason about hypothetical scenarios that cannot otherwise be answered without assumptions or a causal graph. The simpler case involves a causal graph with scalar 1-D variables that denote different attributes in the data-generation process. While the literature is clear in the 1-D case, extending this to settings with high-dimensional nodes (e.g., images) remains challenging. Conventionally, in the literature, methods recover the latent state from the factual image (inversion), generate counterfactual attributes from the graph, and render the counterfactual image from the latent state using the updated attributes. In our work, we investigate the necessity of the inversion step in computing high-dimensional counterfactuals. We first discuss moving from the scalar 1-D case to the higher-dimensional case. Secondly, we discuss the relevance of the inversion step and evaluate an inversion-free counterpart repurposed from the image-editing literature. Finally, on image datasets such as CelebA and ADNI, we benchmark our approach using two measures: namely, effectiveness and identity preservation. While effectiveness is a measure of how faithfully the counterfactual attribute is rendered, identity preservation measures whether the same unit (image) is being rendered, both of which are crucial for high-dimensional counterfactual inference. We present the tradeoffs of using inversion-based and inversion-free approaches in our work.

Index Terms—causal inference, counterfactual reasoning

## I. INTRODUCTION

Answering hypothetical questions about individual cases are common in data analysis and hard to answer without assumptions. Would this patient’s chances of survival looked different had the disease progressed further? This is a counterfactual query, and Pearl’s abduction–action–prediction procedure [1] answers the query when the variables concerned are scalars, under some assumptions that are by now well understood. However, most of the variables of interest are not singledimensional scalars. A brain scan or a photograph carries several attributes at once, many of them causally related, and answering a counterfactual query about such a record means producing the record itself rather than a number. This has become feasible, with generative models capable of rendering an image conditioned on a requested set of attributes.

The pipelines that answer such queries share a common structure. A causal model over the attributes, a generative backbone that renders the image, and between them sits a step that recovers a latent state from the factual image, intended to make the result depict the same unit rather than a fresh image carrying the requested attributes. This state is taken from a learned encoder [2, 3] or from running the backbone backward to a noise map [4–6], and the step is also referred to as “abduction”, taking its name after the procedure of inverting a structural equation the modeler has fitted in the standard 1- D case. The name is warranted when the image mechanism is trained as part of the causal model, and identification in high dimensions has been established on that basis [7], under conditions that are demanding in practice. We investigate the role of this inversion step in computing counterfactuals in the higher dimensional case.

Our contributions are as follows:

• Analysing the issues in moving from 1-D counterfactual inference to higher dimensional counterfactual inference, discussing approaches in the literature and identification arguments involved to tackle the higher dimensional case.

• Presenting the role of inversion step, conventionally used by most works in computing higher-dimensional counterfactuals, and proposing an inversion-free pipeline that does not generate the latent during the edit.

• Testing the inversion-based approaches against our proposed inversion-free approach on image datasets like CelebA, a corpus of celebrity images and ADNI, consisting of brain MRI scans. We provide tradeoffs on two measures namely effectiveness and identity preservation.

## II. RELATED WORK

We classify works on counterfactual inference into three categories: starting with works relevant to traditional counterfactual inference, literature dealing with using deep networks for the task and the body of work relevant to counterfactual queries over images.

a) Counterfactual Inference: Counterfactual queries are posed within the Structural Causal Model framework [1], to answer a query about a specific unit through the abduction– action–prediction procedure and, with the axiomatic characterization of Galles and Pearl [8]. The difficulty lies in abduction, and it is resolved by restricting the functional form of the mechanism, a restriction that the literature has progressively weakened. Additive noise models fix how the exogenous term enters, so that it is recovered as a residual while the dependence on the parents remains nonlinear [9]. Post-nonlinear models relax additivity itself by composing the additive form with an invertible output nonlinearity, which is inverted before the residual is taken [10]. Nasr-Esfahany et al. [11] unify these under bijective generation mechanisms and establish counterfactual identifiability for three causal structures, in the Markovian case relevant here, strict monotonicity in the exogenous argument. The other two cases admit unobserved confounding, identifying the mechanism through an instrumental variable or a backdoor set in place of the Markovian assumption. Recently another approach by Von Kuglegen et al. [12] popularised backtracking counterfactuals. This paradigm allows for the same causal graph while computing the counterfactual, breaking away from the traditional interventionist account.

![](images/2eec4729d3b952fc9f371036eac8592306ad977995b31d7dccf5812aad0ede7f.jpg)  
Fig. 1. The two transports, both given the same counterfactual attributes obtained by propagating the antecedent on age through the structural equations. Left: inversion carries the factual to noise under the source condition and regenerates under the target. Right: the inversion-free map moves the factual directly, b accumulated velocity differences between the two conditions, always staying on the image manifold.

b) Deep Networks for Counterfactual Inference: Scaling causal inference to high-dimensional data has led to the development of Deep Structural Causal Models (DSCMs) [2], which leverage normalizing flows and VAEs to parameterize exogenous noise. Neural Causal Models [13] further extend this by providing frameworks for identification and estimation in complex, non-linear environments. Recent research has specifically addressed the problem of Deep Backtracking [14], proposing methods that ensure counterfactual explanations remain ”causally compliant” by allowing background variables to be updated. Other developments include handling categorical background variables [15] and ensuring end-to-end consistency for counterfactual fairness [16], ensuring that deep estimators do not introduce spurious correlations during the abduction phase.

c) Image Editing: In the visual domain, counterfactual editing seeks to modify specific image attributes while preserving the underlying causal structure. Early generative approaches often conflated causal dynamics with image rendering, but recent work on Augmented SCMs (ASCM) [17] has formalized the separation of mixing functions from causal mechanisms. High-fidelity image counterfactuals have been achieved using probabilistic causal models [3], which utilize latent space abduction to ensure realistic reconstructions. The emergence of diffusion-based models has further pushed the state-of-the-art. Diffusion Causal Models [4] allow for finegrained counterfactual estimation by treating the reverse diffusion process as a causal mechanism. More recently, semantic abduction techniques [5] have improved the alignment between textual prompts and causal interventions, enabling diffusion models to generate images that are not only visually convincing but also causally consistent with complex counterfactual queries. A parallel line in the vision literature avoids the round trip altogether, editing along the generative path [18, 19]. These methods carry no causal model over the attributes, but supply the transports compared here against inversion.

## III. PRELIMINARIES

Lowercase letters denote scalars, lowercase boldface vectors, uppercase boldface matrices, and calligraphic letters sets and tuples that are not random variables. Plain uppercase letters denote random variables and their lowercase counterparts realized values. We write $P ( \cdot )$ for probability distributions and do(·) for Pearl’s intervention operator. The remainder of this section focuses on structural causal models, counterfactuals, the notion of counterfactual identifiability and some basics of flow-based generative models that we later use.

a) Structural Causal Models: A Structural Causal Model (SCM) is a tuple $\mathcal { M } = \langle U , V , \mathcal { F } , P ( U ) \rangle$ , where U is a set of exogenous noise variables with prior $P ( U )$ , V is a set of endogenous variables, and ${ \mathcal F } = \{ f _ { i } \}$ is a set of structural equations $v _ { i } ~ = ~ f _ { i } ( \mathrm { p a } _ { i } , u _ { i } )$ mapping parents and noise to each variable [1]. Each SCM induces a causal diagram $\mathcal { G }$ in which directed edges encode direct causal influence and bidirected edges encode shared exogenous noise. An intervention do $( v _ { i } = v ^ { * } )$ replaces $f _ { i }$ with the constant $v ^ { * }$ , yielding a submodel $\mathcal { M } _ { v _ { i } = v ^ { * } }$ used for interventional and counterfactual queries.

b) Counterfactuals: Counterfactuals reason about a hypothetical world for a specific unit while conditioning on what was actually observed. The hypothesised condition $V _ { i } = v ^ { * }$ is the antecedent. Counterfactuals are obtained as follows:

1) Abduction. Update the noise to the posterior consistent with the evidence, $P ( \mathbf { U } \mid \mathbf { V } = \mathbf { v } )$ . When M is bijective in its noise, this posterior collapses to the point ${ \bf u } =$ ${ \cal F } ^ { - 1 } ( { \bf v } )$

2) Action. Apply $\mathrm { d o } ( V _ { i } ~ = ~ v ^ { * } )$ to obtain the submodel $\mathcal { M } _ { v _ { i } = v ^ { * } }$

3) Prediction. Evaluate the queried variable in $\mathcal { M } _ { v _ { i } = v ^ { * } }$ using the abducted noise u.

c) Counterfactual Identifiability: Two SCMs can agree on the observational distribution and still disagree on a counterfactual, since matching the conditional law of a variable given its parents leaves free how exogenous values are paired with outcomes (Figure 3). A counterfactual query is identifiable when no such disagreement is possible, that is, when every SCM consistent with the observational distribution assigns it the same value [1]. Restricting the class restores this: with each $f _ { i }$ bijective in its exogenous argument abduction returns a point, and under a Markovian diagram with $f _ { i }$ strictly monotone the counterfactual is identified [11].

d) Flow-based generative models: Flow-based generative models learn a distribution mapping samples from a noise distribution to a target image distribution [20]. Flow matching learns to transport noise to data along a prescribed interpolation. Rectified flow [21] prescribes the straight-line interpolation

$$
z _ { t } = ( 1 - t ) x + t \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) ,\tag{1}
$$

so that $t = 0$ is the image and $t \ : = \ : 1$ is noise, a network $V ( z , t , c )$ is trained to predict the velocity $\epsilon - x$ of this path from $\mathbf { z } _ { t }$ alone, given a condition $c .$ Generation integrates $\dot { z } _ { t } =$ $V ( z _ { t } , t , c )$ from $t ~ = ~ 1$ to $t ~ = ~ 0$ starting at a noise draw, running the same field from $t ~ = ~ 0$ upward instead carries an image to the noise state that would have produced it. This second direction is the inversion step, and the round trip invert under $c _ { \mathrm { s r c } }$ , then regenerate under $c _ { \mathrm { t a r } }$ is the mechanism by which flow and diffusion backbones implement abduction and prediction steps at the image level.

![](images/a4356342a68746152ac94bd1e88bcb737e0a36b6014f023e5a19185208eba66a.jpg)  
Fig. 2. Attribute-level diagram and structural equations used as a running example, following Melistas et al. [22]: ApoE-ε4 status ap, sex s, age a, brain volume b and ventricular volume v. The image node is introduced in Section IV-B.

![](images/5d1c8e413b787b0d81949a3d9791050a6ec11a5bb277dc334877d340cdea63f8.jpg)

![](images/9fdc7d1676508d5be15012f7218e39d26772d5749e46a49f4754d1305eb9b14d.jpg)  
Fig. 3. Abduction at fixed parents. Solid curves are $f _ { v } ( a , b , \cdot )$ at the observed brain volume b, dashed curves the same mechanism under $\overset { \prime } { \mathrm { d o } } ( B = b ^ { * } )$ . In (a) the observed v has a single preimage $\hat { u } _ { v } ,$ , and the counterfactual is read off the dashed curve at that point. In (b) the mechanism attains v at two exogenous values, and the two preimages give different counterfactuals.

## IV. UNDERSTANDING COUNTERFACTUAL INFERENCE IN HIGH-DIMENSIONAL SETTINGS

We first consider counterfactual identification in the scalar case, where the assumptions are well understood and abduction reduces to inverting the structural equation. Introducing an image node, we then examine what assumptions are required for identification in higher dimensions, and how it is approached in practice. We close by reviewing how abduction is implemented in the current literature, and by separating two notions the scalar case allows to coincide: abduction and inversion.

## A. Counterfactual Identification in the Scalar 1-D Case

We introduce a running example based on structural brain imaging, to which we return to in Section VI. Genotype, sex and age are roots; brain volume is their common child; ventricular volume responds negatively to brain volume, since the two partition a fixed intracranial space, and to age directly (Fig. 2). The exogenous terms absorb the subject-specific variation the observed attributes leave unexplained, and are what abduction must recover.

![](images/907bc87e8507f2c5cf667e62e3394059e41419b7275c17eb91fab7f6b844bef3.jpg)  
Fig. 4. The two-stage model. The attribute mechanisms and their exogenous variables are those of Fig. 2. All five attributes $\mathbf { p a } _ { \mathbf { \alpha } } = \mathbf { \alpha } ( a p , s , a , b , v )$ are rendered into the image by g (grey edges), together with a high-dimensional exogenous state $u _ { x }$ carrying everything the diagram does not name. The labelling map ℓ recovers the attributes from the image.

Consider a subject with observed attributes and the antecedent $\begin{array} { l } { \mathrm { d o } ( B { \it \Delta \phi } = { \it \Delta \ b } ^ { * } ) } \end{array}$ Genotype, sex and age are nondescendants of $B$ and retain their observed values, so the query reduces to re-evaluating $f _ { v }$ at $( a , b ^ { * } , u _ { v } )$ . Everything therefore depends on recovering $u _ { v }$ from the factual, and the data alone does not determine this. The observational distribution constrains $f _ { v }$ only through the conditional law of V given its parents. Fig. 3 shows two mechanisms that agree on that law and disagree on the counterfactual, because they pair exogenous values with outcomes differently.

Imposing restrictions on the causal models by assuming their functional forms, we can remove the indeterminacy that leads to counterfactual non-identification. If the mechanism is strictly monotone in its exogenous argument at every configuration of the parents, the map $u _ { v } \mapsto$ v is invertible, the posterior collapses to the point $u _ { v } \ = \ f _ { v } ^ { - 1 } ( v ; a , b )$ , and the counterfactual is point-identified [11]. Along with monotonicity, two more conditions are assumed in this setting. The diagram is known, and the model is Markovian, so the exogenous variables are mutually independent.

## B. Counterfactual Inference over a Higher Dimensional Node

We now consider a brain MRI scan that is rendered as a function of brain and ventricular volume at once. The causal mechanism is therefore separated from the rendering mechanism, also advocated by Pan and Bareinboim [17], shown in Fig. 4. We take $g$ to be invertible with respect to the attributes, so a labelling map ℓ recovers them from any image in the support, which is what makes the two volumes readable off a scan at all.

Abduction step can now be split in two phases. Recovering the attribute-level noise $u _ { a p } , \ldots , u _ { v }$ is the scalar problem of Section IV-A, unchanged. Though recovering $u _ { x }$ is different in nature. It carries everything not specified in the attribute graph, the slice depth, the cortical folding pattern, the scanner and its intensity profile, and it is high-dimensional. No mechanism has been specified for it, so there is no fitted function to invert in the sense of Section IV-A. Also monotonicity does not transfer easily: it is a condition on a scalar argument, whereas $g$ takes a latent state of many thousands of dimensions. In practice $u _ { x }$ is whatever a generative backbone returns when run in reverse.

TABLE I  
IMAGE MECHANISMS AND THE EXOGENOUS STATES THEY RECOVER.
<table><tr><td>Method</td><td>Image mechanism</td><td>Exogenous state</td></tr><tr><td>Pawlowski et al. [2]</td><td>VAE, trained</td><td>latent + residual</td></tr><tr><td>De Sousa Ribeiro et al. [3]</td><td>HVAE, trained</td><td>latents + residual</td></tr><tr><td>Melistas et al. [22]</td><td>GAN, trained</td><td>single latent</td></tr><tr><td>Sanchez and Tsaftaris [4]</td><td>diffusion, trained</td><td>uncond. noise map</td></tr><tr><td>Rasal et al. [5]</td><td>diff. AE, trained</td><td>noise map, semantic code, guidance tokens</td></tr><tr><td>De Sousa Ribeiro et al. [7]</td><td>OT-coupled flow</td><td>flow noise map</td></tr><tr><td>Tong et al. [6]</td><td>SD1.5 frozen, adapter trained</td><td>cond. noise map</td></tr></table>

## C. Abduction v.s. Inversion

A counterpart to monotonicity does exist in the higher dimensional case making identification possible in higher dimensions. But the condition must be imposed while the model is trained. De Sousa Ribeiro et al. [7] require the image mechanism to be the map induced by a dynamic optimal transport flow, which generalises monotonicity to $d > 1$ and identifies the counterfactual. Nothing applied at inference time can supply this to a backbone pretrained for generation. Table I places the literature against that standard. Each method factorises the image noise differently, and abduction returns whatever that factorisation names: an encoder latent and a reparameterisation residual for the variational mechanisms [22], a single latent for the adversarial one, and for Rasal et al. [5] a spatial noise map, a semantic code and a sequence of guidance tokens from the same image. Abduction then returns whatever these models declare, so the states they recover are properties of the model rather than of the images.

Definition 4.1 (Abduction): Given a structural causal model M and evidence $e ,$ abduction returns the posterior over the exogenous variables of M consistent with $e .$

Definition 4.2 (Inversion): Given a conditional generator $G ( z , c )$ and an image $x ,$ inversion returns a state z satisfying $G ( z , c ) \approx x .$

Definition 4.1 is relative to a declared model and inherits whatever that model’s assumptions secure, while Definition 4.2 is a numerical procedure that secures nothing on its own. The two coincide when the network is the model’s rendering mechanism, which the trained entries of Table I arrange for and a frozen backbone does not. What remains in the latter case is a recovery step carrying no guarantee, which makes its contribution an empirical question. Section V takes it up.

## V. FLEXIBILITY OF TRANSPORT AND AN INVERSION-FREE PIPELINE

We first separate the attribute mechanism from the transport map that generates the output. We then identify composition, the requirement that a null antecedent return the factual, as the one further property the semantics places on the map itself. Using the freedom these results leave, we construct a pipeline that dispenses with inversion, and close with the properties to be tested empirically.

## A. Separating the Attribute Mechanism from the Transport

Write a pipeline as a pair $( A , T )$ . The attribute mechanism $\mathcal { A }$ takes the factual image x and an antecedent, reads off $\mathbf { p } \mathbf { a } = { \boldsymbol { \ell } } ( { \boldsymbol { x } } )$ , abducts the attribute-level noise and propagates the antecedent through the structural equations to obtain the counterfactual attributes $\mathbf { p } \mathbf { a } ^ { * }$ . The transport $T$ takes the factual image together with a condition $c _ { \mathbf { p } \mathbf { a } ^ { * } }$ built from those attributes, and returns an image realising them. In the running example A is the computation of Section IV-A and $T$ is whatever produces the edited image. The two are specified independently, and every method in Table I pairs some A with some T.

Definition 5.1 (Faithful transport): A transport T is faithful if, for every factual image x in the support and every attribute vector $\mathbf { p } \mathbf { a } ^ { * }$ in the range of A, the image it returns satisfies $\ell ( T ( x , c _ { \mathbf { p a } ^ { * } } ) ) = \mathbf { p a } ^ { * }$

Proposition 5.2 (Invariance of attribute-level metrics under faithful transports): Assume the two-stage model of Section IV-B. Let $T _ { 1 }$ and $T _ { 2 }$ be faithful in the sense of Definition 5.1. Then $( A , T _ { 1 } )$ and $( A , T _ { 2 } )$ induce the same joint distribution of factual and counterfactual attributes, and any functional of that joint distribution is therefore invariant to which of the two is used.

Proof: Fix a factual image x and an antecedent, and let $\mathbf { p } \mathbf { a } = \ell ( x )$ and $\mathbf { p } \mathbf { a } ^ { * }$ be the attributes $\mathcal { A }$ returns. For $j \in \{ 1 , 2 \}$ write $x _ { j } ^ { * } = T _ { j } ( x , c _ { \mathbf { p a } ^ { * } } )$ . The labelling map ℓ is a function of the image alone, so $\ell ( x _ { j } ^ { * } )$ is determined by $\boldsymbol { x } _ { j } ^ { * }$ and does not depend on the trajectory through image space by which it was reached. By Definition 5.1, $\ell ( x _ { j } ^ { * } ) = \mathbf { p a } ^ { * }$ almost surely for $j = 1 , 2$ . The pair $\big ( \ell ( x ) , \ell ( x _ { j } ^ { * } ) \big )$ therefore equals $\mathbf { \left( p a , p a ^ { * } \right) }$ almost surely for both transports, and its distribution over factual images is fixed by the distribution of $\mathbf { \left( p a , p a ^ { * } \right) }$ , which A determines from the factual image and the antecedent alone. The second claim follows, a functional of a distribution being determined by it.

The proposition says that a faithful transport is invisible to anything read off the attributes of the counterfactual image generated. Effectiveness is measured this way, by predicting the intervened attribute from the counterfactual image, so it cannot tell two such transports apart. Section VI nonetheless reports differences in it. The reason is that no transport on a pretrained backbone is faithful in the exact sense of Definition 5.1: a requested attribute is delivered only in part. A gap in effectiveness therefore records how much of the request each transport failed to deliver, and not which of them took the better route to the counterfactual.

## B. Exact Composition Under a Null Antecedent

One property escapes Proposition 5.2. A counterfactual under a null antecedent must return the factual, since setting a variable to the value it already holds changes nothing. This is the composition axiom of Galles and Pearl [8], used as an evaluation metric for image counterfactuals by Monteiro et al. [23], and unlike identity preservation it can be checked without knowing the true rendering mechanism.

Proposition 5.3 (Exactness under a null antecedent): Let the antecedent set the intervened attribute to its observed value, so that $\mathbf { p } \mathbf { a } ^ { * } = \mathbf { p } \mathbf { a }$ . Then the transport of Equation (2) returns the factual latent unchanged, for any step count N, any number of noise draws $M .$ , and any solver whose update is a linear combination of evaluations of the integrand.

Proof: Since $\mathbf { p } \mathbf { a } ^ { * } = \mathbf { p } \mathbf { a }$ , the two conditions coincide, $c _ { \mathrm { t a r } } = c _ { \mathrm { s r c } } ,$ , so $\mathcal { V } ( \cdot , t , c _ { \mathrm { t a r } } )$ and $\mathcal { V } ( \cdot , t , c _ { \mathrm { s r c } } )$ are the same function of state and time. The coupling $\hat { z } _ { t } ^ { \mathrm { t a r } } = z _ { t } + ( \hat { z } _ { t } ^ { \mathrm { s r c } } - x )$ gives $\hat { z } _ { t } ^ { \mathrm { t a r } } = \hat { z } _ { t } ^ { \mathrm { s r c } }$ whenever $z _ { t } = x ,$ , in which case the two fields are evaluated at the same state, time and condition and the integrand of Equation (2) vanishes for every draw $\epsilon ,$ hence in expectation. The trajectory starts at $z _ { t _ { \operatorname* { m a x } } } = x , { \mathrm { i f ~ } } z _ { t } = x$ at the current step the update is zero and $z _ { t } = x$ at the next, so $z _ { t } = x$ throughout.

The round trip has no such guarantee. Inversion and regeneration are exact inverses only in the continuum limit, so at finite $N$ the two integrations accumulate discretisation error that does not cancel. The error shrinks with N but is never zero, and it is paid on every query, including those where no edit is requested. Table IV measures the gap this predicts: under an antecedent restoring the observed value the displacement map returns to 0.0470 in LPIPS, against 0.4345 for the round trip.

## C. An Inversion-Free Counterfactual Pipeline

We now give a pipeline that constructs no exogenous latent state at any point. It has three parts: the attribute mechanism producing $\mathbf { p } \mathbf { a } ^ { * }$ , a conditioning interface turning those attributes into an offset the frozen backbone can read, and the transport realising it (Fig. 1).

1) Conditioning interface: The backbone reads an image as a grid of n patches. A single vector added identically to every patch can request more of a quantity but cannot say where the change belongs, which is the wrong parameterisation for an attribute confined to one anatomical region. We instead give each attribute i its own spatial pattern $M _ { i } \in \mathbb { R } ^ { n \times d }$ , one vector per patch, and add $\sum _ { i } a _ { i } M _ { i }$ to the patch embeddings before the first transformer block, where $a _ { i }$ is the value of attribute i. Each attribute contributes a fixed pattern whose amplitude is its value, so the request carries a location as well as a magnitude. The patterns are zero-initialised and attached at the input, so the pretrained weights are untouched and the frozen model is exactly recovered at initialisation. We write $c _ { \mathbf { p } \mathbf { a } }$ ∗ for the offset built from the counterfactual attributes.

2) Transport: The backbone is a flow model: a network $\mathcal { V } ( z , t , c )$ predicting, along the straight interpolation between an image x and a noise draw ϵ, the velocity carrying one to the other under a condition c. Rather than integrate to noise and back, we adopt the displacement map of Kulikov et al. [18] unmodified and transport the image directly. The source state is the factual image noised along that path, $\hat { z } _ { t } ^ { \mathrm { s r c } } = ( 1 -$ $t ) x + t \epsilon$ , and the target state is the same point carried forward by however far the edit has travelled, $\hat { z } _ { t } ^ { \mathrm { t a r } } = z _ { t } + \left( \hat { z } _ { t } ^ { \mathrm { s r c } } - x \right)$ so the two stay a fixed displacement apart and differ only in the condition at which the field is evaluated. That difference drives the edit,

![](images/69b0769315b65f44b44c3c953267d98ee248420f79b8a9c38913fc0df24aed70.jpg)  
Fig. 5. The attribute-level diagram assumed on CelebA.

$$
d z _ { t } = \mathbb { E } _ { \epsilon } \big [ \mathcal { V } ( \hat { z } _ { t } ^ { \mathrm { { t a r } } } , t , c _ { \mathbf { p a } ^ { * } } ) - \mathcal { V } ( \hat { z } _ { t } ^ { \mathrm { s r c } } , t , c _ { \mathbf { p a } } ) \big ] d t ,\tag{2}
$$

integrated from $t _ { \mathrm { m a x } }$ to 0 with $z _ { t _ { \operatorname* { m a x } } } = x$ and the expectation estimated over M noise draws. The trajectory starts at a real image and moves only by accumulated velocity differences, requiring no inversion, no optimisation and no access to model internals.

## D. Properties Left to Empirical Comparison

Few questions remain open, and they decide the choice between the two families in practice. Whether the counterfactual depicts the same subject lies outside Proposition 5.2, the exogenous state being no attribute of the diagram. How much of a requested change is actually delivered lies outside it too, faithfulness in the sense of Definition 5.1 being an idealisation a real backbone meets only approximately.

A transport that leaves the image untouched attains a perfect preservation score, so identity is informative only where the edit demonstrably occurred. Section VI adopts this convention and reports the comparison on CelebA, on a from-scratch checkpoint isolating the map, and on brain MRI with a continuous attribute.

## VI. EMPIRICAL ANALYSIS

Below we compare inversion and inversion free methods using existing pretrained flow based generative backbones as well as scratch trained models for reference. Later we choose a case brain MRI images where the interventions are over continous attributes to contrast both family of methods (See Figure 1).

## A. Benchmarking Pretrained Flow Based Models on CelebA

1) Setup: CelebA dataset [24] is used in this experiment, a split of 162,770 training/19,962 test samples, with the attributes Young (a), Male $( g )$ , No Beard (br) and Bald (bl) over the causal graph as shown in the Figure 5. Each test image, is queried under the four single-node antecedents $\mathrm { d o } ( v _ { S } = 1 - v _ { S } )$ . The counterfactual attributes are obtained by propagating each through the SCM. Interventions on the roots displace both leaves, interventions on the leaves must leave their non-descendants at the observed values. All methods see an identical query set.

2) Baselines and protocol: Methods are grouped by whether a latent state is recovered at all rather than by architecture. The first group recovers one explicitly: VAE, HVAE and GAN are deep structural causal models using a learned encoder [2, 3], and Causal-Adapter recovers a latent state from a frozen SD1.5 by DDIM inversion [6]. FlowEdit integrates the displacement dynamics of Eq. (2) on a frozen SD3-medium [25] with the counterfactual attributes supplied through the target condition. Effectiveness is the F1 of an anti-causal classifier for the intervened attribute, pooled over the entire test split. Realism is FID [26], which compares generated and real images in the feature space of a pretrained network and is computed once per intervention across all samples. Identity is the LPIPS [27] between each counterfactual and its corresponding factual, a perceptual distance that better tracks human similarity judgments than pixel norms.

3) Analysis: Explicit inversion attains higher effectiveness, inversion-free higher identity preservation. SD3-inv and FlowEdit share the same frozen backbone. They differ only in the transport. The comparison between them therefore isolates the map. Inversion is more effective on three of four attributes. The gap is 0.307 on No Beard and 0.366 on Bald. The inversion-free map stays closer to the factual on all four. On Young it records 0.078 against 0.369. On the same attribute the two are within 0.010 on effectiveness. Causal-Adapter and SD3-inv are the only methods to exceed 0.4 on every attribute. FlowEdit (inversion-free) attains LPIPS between 0.055 and 0.124 on the attributes where the edit is realised, against 0.119–0.246 for Causal-Adapter, and achieves the lowest identity cost on do(Young) and do(No Beard) among methods whose edit took effect.

Bald is the attribute on which FlowEdit fails (0.039), despite attaining the highest F1(a) (0.866). The backbone is frozen and receives the antecedent only through the target condition. So an attribute that it does not already represent densely cannot be requested effectively. Baldness is rare in web-scale pretraining relative to age.

The from-scratch baselines fail for reasons specific to each. The VAE produces blurred reconstructions, with FID exceeding 57 on every antecedent and effectiveness of 0.004 on Bald. The HVAE alters global contrast in place of the attribute, which yields the lowest FID on a and g and the lowest identity cost on g (0.076) while failing the effectiveness criterion on two attributes. The GAN departs from the image manifold under some antecedents and incurs the highest identity cost of any successful edit (0.285 on Bald). The identity values of 0.26–0.36 reported for the VAE and GAN are therefore attributable largely to reconstruction error rather than to the intervention.

## B. Scratch Trained Models

a) Setup: The comparison in Section VI-A varies the backbone, the training corpus and the transport simultaneously, and consequently cannot attribute an observed difference to any one of them. This section holds every factor fixed except the map. All rows of Table III are generated by a single checkpoint, a DiT-B/2 of 131 M parameters trained from scratch for 50k steps by conditional flow matching on SD3 latents under independent coupling. The attribute vector enters through the backbone’s class-conditioning path: one learned embedding per attribute–value pair, summed and added to the timestep embedding, which modulates every block through adaptive layer norm. Conditioning is therefore global rather than spatial. The evaluation on the graph, test split and intervention plan is done as described in Section VI-A. The rows differ only in the procedure by which the counterfactual image is obtained from the factual one.

FlowEdit  
TABLE II  
CELEBA COUNTERFACTUALS, GROUPED BY WHETHER THE EXOGENOUS STATE IS RECOVERED AT ALL. BEST PER COLUMN IN BOLD, RUNNER-UP UNDERLINED. CELLS MARKED <sup>†</sup> ARE EXCLUDED FROM THE RANKING.
<table><tr><td></td><td></td><td colspan="4">Effectiveness F1 ↑</td><td colspan="4">Realism FID ↓</td><td colspan="4">Identity LPIPS ↓</td></tr><tr><td>Method</td><td>Backbone</td><td>a</td><td>g</td><td>br</td><td>bl</td><td>a</td><td>g</td><td>br</td><td>bl</td><td>a</td><td>g</td><td>br</td><td>bl</td></tr><tr><td colspan="10">Explicit inversion — encoder; DDIM for Causal-Adapter, flow-ODE for SD3-inv</td><td colspan="3"></td></tr><tr><td>VAE</td><td>deep-SCM</td><td>0.390</td><td>0.990</td><td>0.284</td><td>0.004</td><td>60.0</td><td>63.4</td><td>59.9</td><td>57.4</td><td>0.283</td><td>0.295</td><td>0.283</td><td>0.284†</td></tr><tr><td>HVAE</td><td>deep-SCM</td><td>0.246</td><td>0.742</td><td>0.476</td><td>0.122</td><td>2.2</td><td>5.9</td><td>25.6</td><td>31.7</td><td>0.026†</td><td>0.076</td><td>0.138</td><td>0.229†</td></tr><tr><td>GAN</td><td>deep-SCM</td><td>0.471</td><td>0.989</td><td>0.273</td><td>0.340</td><td>22.1</td><td>26.6</td><td>23.0</td><td>24.3</td><td>0.256</td><td>0.363</td><td>0.262</td><td>0.285</td></tr><tr><td>Causal-Adapter</td><td>SD1.5 + adapter</td><td>0.530</td><td>0.997</td><td>0.424</td><td>0.497</td><td>8.2</td><td>9.0</td><td>8.0</td><td>17.7</td><td>0.166</td><td>0.246</td><td>0.119</td><td>0.240</td></tr><tr><td>SD3-inv</td><td>SD3 (2B, frozen)</td><td>0.856</td><td>0.937</td><td>0.522</td><td>0.405</td><td>33.1</td><td>47.0</td><td>33.1</td><td>37.4</td><td>0.369</td><td>0.464</td><td>0.351</td><td>0.397</td></tr><tr><td colspan="10">Inversion Free</td><td colspan="3"></td></tr><tr><td>FlowEdit</td><td>SD3 (2B, frozen)</td><td>0.866</td><td>0.927</td><td>0.215</td><td>0.039</td><td>10.8</td><td>9.6</td><td>3.9</td><td>3.5</td><td>0.078</td><td>0.124</td><td>0.055</td><td>0.049†</td></tr></table>

<sup>†</sup> The row’s F1 for that attribute falls in the lowest quartile of the interval between copying the input and resampling from noise, so the intervention effectively did not occur.

Factual  
VAE  
Causal-Adapter  
HVAE  
GAN  
![](images/0573c8e3ef1be22a7bfb701b7358e80dcfedb4d154c062539cad4f945f48dbdf.jpg)  
Fig. 6. CelebA counterfactuals with four antecedents, one method per column.

b) Inference maps: Four maps are compared on a single checkpoint, all at guidance weight ω=2 [28].

Inversion-free. The displacement map integrates the dynamics of Eq. (2) directly from the factual image, estimating the expectation over M=8 noise draws at N=32 steps, at 2NM forward passes per query. Its regularised variant adds the source-consistency term of FlowAlign [19] at $\gamma { = } 0 . 0 1$ , which at each step contracts the accumulated displacement toward the factual rather than allowing it to accumulate unconstrained.

Inversion. The round trip recovers a noise map from the factual under the source condition and regenerates under the target condition, following a single trajectory rather than an expectation, at 2N forward passes. We report it at N=32 and at N=256.

Two bounds frame the comparison rather than competing in it: copying the input, which measures what the classifier assigns to an unedited image, and sampling at the counterfactual parent configuration, which measures the effectiveness attainable once the factual is disregarded entirely. Both are excluded from the ranking. Metrics follow Section VI-A.

c) Analysis: At matched budget the inversion-free map is ahead on all four effectiveness columns and on three of four identity columns, and the round trip only closes the gap by running at N=32, at one eighth of the forward passes. The FlowAlign regulariser moves along the same trade-off rather than improving on it, gaining 0.028 in identity under do(Bald) at a cost of 0.174 in effectiveness.

Identity does separate the two maps, but only on some attributes. Under do(Young) the inversion-free map (no pathreg) records an LPIPS of 0.070 against 0.148 for the round trip, and under do(No Beard) 0.128 against 0.188, in both cases staying closer to the factual. Under do(Bald) the two are indistinguishable, at 0.430 and 0.432. What predicts this is how close the maps sit to the from-noise bound of 0.517: on Bald both are near it, and the prototype collapse described below leaves no identity for any map to preserve, whereas on Young there is room between them and the bound.

TABLE III  
ONE MODEL, SEVERAL INFERENCE MAPS. ALL ROWS SHARE THE SAME SCRATCH-TRAINED DIT-B/2 (131 M, SD3 LATENTS, INDEPENDENT COUPLING), SO NO DIFFERENCE BETWEEN THEM IS ATTRIBUTABLE TO TRAINING. THE FIRST TWO ROWS ARE BOUNDS AND ARE EXCLUDED FROM THE RANKING; BEST PER COLUMN IN BOLD, RUNNER-UP UNDERLINED.
<table><tr><td></td><td></td><td colspan="4">Effectiveness F1 ↑</td><td colspan="4">Identity LPIPS ↓</td></tr><tr><td>Inference map</td><td>Fwd./query</td><td>a</td><td>g</td><td> $b r$ </td><td>bl</td><td>a</td><td>g</td><td> $b r$ </td><td>bl</td></tr><tr><td>copy input</td><td>0</td><td>0.142</td><td>0.349</td><td>0.050</td><td>0.007</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>sample at  ${ p } a ^ { * }$ </td><td>32</td><td>0.569</td><td>0.992</td><td>0.312</td><td>0.700</td><td>0.495</td><td>0.517</td><td>0.508</td><td>0.517</td></tr><tr><td>inversion-free, M=8</td><td>512</td><td>0.674</td><td>0.994</td><td>0.361</td><td>0.724</td><td>0.070</td><td>0.325</td><td>0.128</td><td>0.430</td></tr><tr><td>+ FlowAlign,  $\gamma { = } 0 . 0 1$ </td><td>512</td><td>0.420</td><td>0.968</td><td>0.247</td><td>0.550</td><td>0.033</td><td>0.239</td><td>0.104</td><td>0.402</td></tr><tr><td>inversion,  $N { = } 3 2$ </td><td>64</td><td>0.610</td><td>0.992</td><td>0.314</td><td>0.717</td><td>0.148</td><td>0.343</td><td>0.188</td><td>0.432</td></tr><tr><td>inversion,  $N { = } 2 5 6$ </td><td>512</td><td>0.539</td><td>0.987</td><td>0.297</td><td>0.698</td><td>0.116</td><td>0.335</td><td>0.147</td><td>0.426</td></tr></table>

Identity is $\mathrm { L P I P S } ( x ^ { \prime } , x )$ per intervention. The sample at ${ p } a ^ { * }$ bound never reads the factual image.

![](images/49a8afc761f01b58930d5a4e6a56386f90387a6e254a4299b895bc891cb33b04.jpg)  
Fig. 7. CelebA counterfactuals from the scratch-trained model, one inference map per column.

The from-scratch checkpoint also answers many queries with a similar output regardless of the factual, a failure neither effectiveness nor identity registers: the outputs carry the intervened attribute, and their perceptual distance to the factual is comparable to that of a subject-specific counterfactual.

## C. Case study: Counterfactuals on brain MRIs

1) Setup: Our final setting is over structural brain MRIs from the ADNI dataset [29], a public study of aging and Alzheimer’s disease. We remove non-brain tissue from each scan and align it to a standard anatomical template, as also done in previous work [22], so that a given structure appears at approximately the same location in every image. We retain the ten central axial slices per subject used in prior work on this benchmark and report results at three of these levels. The attribute we intervene on is the volume of the ventricles, the fluid-filled cavities at the center of the brain, which enlarge as surrounding tissue is lost. Two features distinguish this setting from Section VI-A. The attribute is continuous, so an antecedent specifies how much change is requested and not merely its direction. And its effect is confined to a small, anatomically fixed region, so a successful edit must move one part of the image and leave the rest alone.

2) Modeling: The conditioning interface is that of Section V-C, with one attribute pattern per node of the causal graph. Because every scan is aligned to a common template, a given structure occupies approximately the same patches across subjects, so a fixed spatial pattern per attribute is well-defined across the dataset. Without alignment, the same pattern would map to different anatomy in each scan. The model is trained under the flow-matching objective alone, with the attribute dropped for a fixed fraction of samples so that classifier-free guidance is available without further training. One frozen backbone serves both transports, so every difference in Table IV is attributable to the map.

3) Analysis: Effectiveness is reported in the units of the attribute: delivered is the measured change in ventricular volume and realised that change as a fraction of the change requested. Identity is split across three axes. Reconstruction is the LPIPS between the factual and what the map returns before any intervention, isolating the cost the map imposes on the factual itself. Edit magnitude is the LPIPS under $\mathrm { d o ( v e n t \_ v o l \to p 9 5 ) }$ . Do-then-undo applies the antecedent and then restores the observed value, the continuous-attribute analog of the null antecedent of Proposition 5.3.

Neither map wins outright. Inversion delivers about twice the requested change, realising 82.1% of it against 46.8%. But it displaces the factual twice as much before any intervention is applied, a cost paid on every subject, and it fails outright on 15 of 273 subjects where the inversion-free map fails on none. No subject fails under both maps, so these are failures of the transport on particular images rather than hard cases in the data.

Inversion-free (full walk)  
![](images/5c9dd44c2bab521f6fdd3e4a1dc786180303e355f944853331f53ab2846c9287.jpg)  
Fig. 8. do $\cdot ( \mathrm { v e n t \_ v o l } \to p _ { 9 5 } )$ on four held-out subjects, taken at even quantiles of the requested change among stable outputs.

TABLE IV  
COUNTERFACTUAL EDITING ON ADNI, do(vent vol→p<sub>95</sub>). ON THE SAME BACKBONE.
<table><tr><td></td><td>Inversion</td><td>Inversion-free</td></tr><tr><td>Effectiveness Delivered (mm³) ↑ Realised ↑</td><td>+41,459 ± 4,671 82.1%</td><td>+23,634 ± 2,732 46.8%</td></tr><tr><td>Identity (LPIPS)</td><td></td><td></td></tr><tr><td>Reconstruction↓</td><td>0.0508</td><td>0.0223</td></tr><tr><td>Edit magnitude, median (max)</td><td>0.149 (0.618)</td><td>0.062 (0.108)</td></tr><tr><td>Do-then-undo ↓</td><td>0.4345</td><td>0.0470</td></tr><tr><td>Unstable outputs ↓</td><td>15/273</td><td>0/273</td></tr></table>

n = 91 subjects × 3 axial levels; ± is a 95% CI for the mean. Instrument floor 2,066 mm<sup>3</sup>. Unstable is edit $\mathrm { L P I P S } > 0 . 3 0$ , unchanged at 3× median (9/273 against 0/273). Edit magnitude is not ranked.

Composition shows the largest gap. When the antecedent restores the observed value, the two conditions coincide, the integrand of Eq. (2) vanishes, and the inversion-free map returns to 0.0470. The round trip has no such guarantee and records 0.4345. Inversion is useful when the requested edit is to be effective, while the inversion-free counterpart proves to be useful when some parts of the anatomy must remain untouched.

## VII. CONCLUSION

This work examined the role of the inversion step in highdimensional counterfactual inference. Moving from the 1-D setting to the high-dimensional one splits the exogenous state into an attribute-level component and an image-level one. The attribute-level noise is recovered as in the scalar case, under the same identifying assumptions. For the image-level node there is no fitted mechanism to invert, and monotonicity is not available as an identifying assumption. Inversion applied over a backbone recovers a latent state from the factual image, but bears no identification guarantees, making its contribution an empirical question. This leaves room for an inversionfree alternative. On CelebA and ADNI we observe tradeoffs rather than dominance: inversion realises a larger fraction of the requested change, while the inversion-free construction departs less from the factual and returns it exactly under a null antecedent. The choice between them can be governed by whether an application prioritises the magnitude of the edit or the preservation of what was not intervened upon.

Our analysis covers a couple of inversion-free transports, and both work the same way, so a transport built on another principle may land elsewhere on the effectiveness–identity trade-off. We have also set the identification route of [7] aside, since it requires the image mechanism to be trained under an optimal-transport condition that a pretrained backbone does not meet.

As future work, we intend to explore applications where identity preservation is paramount, such as longitudinal medical imaging. We also intend to extend the comparison to complex causal graphs and to antecedents on several nodes at once, where the requested attributes compete for the same region of the image.

[1] J. Pearl, Causality. Cambridge university press, 2009.

[2] N. Pawlowski, D. Coelho de Castro, and B. Glocker, “Deep structural causal models for tractable counterfactual inference,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 857–869.

[3] F. De Sousa Ribeiro, T. Xia, M. Monteiro, N. Pawlowski, and B. Glocker, “High fidelity image counterfactuals with probabilistic causal models,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023.

[4] P. Sanchez and S. A. Tsaftaris, “Diffusion causal models for counterfactual estimation,” in Conference on Causal Learning and Reasoning. PMLR, 2022, pp. 647–668.

[5] R. R. Rasal, A. Kori, F. D. S. Ribeiro, T. Xia, and B. Glocker, “Diffusion counterfactual generation with semantic abduction,” in International Conference on Machine Learning. PMLR, 2025, pp. 51 201–51 228.

[6] L. Tong, Z. Liu, C. Lu, D. Oglic, T. Diethe, P. Teare, S. A. Tsaftaris, and C. Jin, “Causal-adapter: Taming textto-image diffusion for faithful counterfactual generation,” arXiv preprint arXiv:2509.24798, 2025.

[7] F. De Sousa Ribeiro, A. Santhirasekaram, and B. Glocker, “Counterfactual identifiability via dynamic optimal transport,” Advances in Neural Information Processing Systems, vol. 38, pp. 147 855–147 903, 2026.

[8] D. Galles and J. Pearl, “An axiomatic characterization of causal counterfactuals,” Foundations of Science, vol. 3, no. 1, pp. 151–182, 1998.

[9] P. Hoyer, D. Janzing, J. M. Mooij, J. Peters, and B. Scholkopf, “Nonlinear causal discovery with additive¨ noise models,” Advances in neural information processing systems, vol. 21, 2008.

[10] K. Zhang and A. Hyvarinen, “On the identifiability of the post-nonlinear causal model,” arXiv preprint arXiv:1205.2599, 2012.

[11] A. Nasr-Esfahany, M. Alizadeh, and D. Shah, “Counterfactual identifiability of bijective causal models,” in International conference on machine learning. PMLR, 2023, pp. 25 733–25 754.

[12] J. Von Kugelgen, A. Mohamed, and S. Beckers, “Back-¨ tracking counterfactuals,” in Conference on Causal Learning and Reasoning. PMLR, 2023, pp. 177–196.

[13] K. Xia, Y. Pan, and E. Bareinboim, “Neural causal models for counterfactual identification and estimation,” in The Eleventh International Conference on Learning Representations, 2023.

[14] K.-R. Kladny, J. von Kugelgen, B. Sch¨ olkopf, and¨ M. Muehlebach, “Deep backtracking counterfactuals for causally compliant explanations,” Transactions on Machine Learning Research, 2024.

[15] E. De Brouwer, “Deep counterfactual estimation with categorical background variables,” Advances in Neural Information Processing Systems, vol. 35, pp. 35 213– 35 225, 2022.

[16] Y. Ma, V. Melnychuk, D. Frauen, and S. Feuerriegel, “Consistent end-to-end estimation for counterfactual fairness,” in Fifth Conference on Causal Learning and Reasoning, 2026.

[17] Y. Pan and E. Bareinboim, “Counterfactual image editing,” in International Conference on Machine Learning. PMLR, 2024, pp. 39 087–39 101.

[18] V. Kulikov, M. Kleiner, I. Huberman-Spiegelglas, and T. Michaeli, “Flowedit: Inversion-free text-based editing using pre-trained flow models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 19 721–19 730.

[19] J. Kim, Y. Hong, J. Park, and J. C. Ye, “Flowalign: Trajectory-regularized, inversion-free flow-based image editing,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 121 814–121 846.

[20] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[21] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” arXiv preprint arXiv:2209.03003, 2022.

[22] T. Melistas, N. Spyrou, N. Gkouti, P. Sanchez, A. Vlontzos, Y. Panagakis, G. Papanastasiou, and S. A. Tsaftaris, “Benchmarking counterfactual image generation,” Advances in Neural Information Processing Systems, vol. 37, pp. 133 207–133 230, 2024.

[23] M. Monteiro, F. D. S. Ribeiro, N. Pawlowski, D. C. Castro, and B. Glocker, “Measuring axiomatic soundness of counterfactual image models,” arXiv preprint arXiv:2303.01274, 2023.

[24] Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 3730–3738.

[25] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller,¨ H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first international conference on machine learning, 2024.

[26] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.

[27] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in 2018 IEEE/CVF conference on computer vision and pattern recognition. IEEE, 2018, pp. 586–595.

[28] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.

[29] C. R. Jack Jr et al., “The alzheimer’s disease neuroimaging initiative (adni): Mri methods,” Journal of Magnetic Resonance Imaging: An Official Journal of the International Society for Magnetic Resonance in Medicine, vol. 27, no. 4, pp. 685–691, 2008.