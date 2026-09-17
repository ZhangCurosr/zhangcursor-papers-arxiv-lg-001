# A Geometric Theory of Decision Boundaries in Structured Markov Decision Processes

Fredy POKOU <sup>∗1</sup>

<sup>1</sup>Inria, CNRS, Univ. of Lille, Centrale Lille, UMR 9189 - CRIStAL, F-59000 Lille, France

September 17, 2026

## Abstract

Classical dynamic programming represents optimal sequential decisions through value functions and policies. While this functional representation is natural for computing optimal decisions, it does not directly identify the mathematical object governing policy reconstruction, representation complexity, or oracle-query complexity once an optimal policy is fixed. This paper addresses this question by developing a geometric theory of structured optimal policies in which the decisionboundary geometry induced by the policy becomes the primary object of analysis. We show that, under suitable structural regularity conditions, this geometry provides the minimal representation required for policy reconstruction and determines the statistical and computational complexity of the reconstruction problem. Building upon this representation, we establish structural properties of policy-induced decision geometry, introduce intrinsic notions of boundary and decision complexity, derive information-theoretic measures of decision compression, and obtain statistical guarantees for boundary estimation and policy reconstruction from black-box policy queries. Collectively, these results demonstrate that, for the structured decision problems considered here, the complexity of policy reconstruction is governed by the geometry of the decision boundary rather than by the cardinality of the ambient state space. Controlled numerical experiments examine the principal theoretical predictions and provide empirical evidence consistent with the proposed framework.

Keywords: Markov Decision Processes; Policy Geometry; Decision Complexity; Black-Box Policy Reconstruction; Information-Theoretic Compression.

## 1 Introduction

The mathematical theory of dynamic programming has traditionally represented optimal sequential decision problems through functional objects, namely the optimal value function, the optimal stateaction value function, and the optimal policy. This representation has provided the foundation for stochastic control, Markov decision processes, and reinforcement learning, and has led to a mature mathematical theory concerning optimality, approximation, convergence, and computational complexity. Virtually all existing analyses of sequential decision-making are therefore expressed in terms of functions defined over the state space.

For many questions, this functional representation is entirely appropriate. However, there exists an important class of questions for which it appears unnecessarily rich. Suppose that the objective is not to compute an optimal policy from a model, but rather to reconstruct an already optimal policy from black-box observations, to quantify its intrinsic representation complexity, to determine how many oracle queries are required for reliable recovery, or to understand which structural properties govern statistical learnability. For such questions, approximating an entire value function or policy over the state space may encode substantially more information than is actually required.

Indeed, once an optimal policy is fixed, only the interfaces at which the optimal decision changes determine the partition of the state space induced by that policy. This observation suggests that the mathematical object governing policy reconstruction is not necessarily the policy viewed as a function, but rather the geometry of the decision regions that the policy induces.

In structured stochastic optimization problems, optimal policies frequently exhibit monotonicity, threshold behavior, convexity, or other forms of regularity. Consequently, they partition the state space into relatively simple regions associated with distinct optimal actions. The boundaries separating these regions form a geometric object that is uniquely determined by the policy itself. Although these decision boundaries appear implicitly throughout the literature on structured dynamic programming, they are generally interpreted as consequences of optimal decision rules rather than as mathematical objects possessing their own statistical, computational, structural, and informationtheoretic properties. Adopting this geometric viewpoint fundamentally changes the mathematical questions that arise. Instead of studying approximation errors for functional representations, one may ask whether the decision-boundary geometry can itself be estimated consistently, whether its intrinsic complexity admits quantitative characterization, whether geometric regularity governs statistical sample complexity, and whether the complexity of policy representation should be measured through the ambient state space or through the geometry of the induced decision partition. These questions are largely orthogonal to the computation of optimal policies and concern instead the mathematical structure of policies that already exist.

The objective of this paper is to develop a geometric theory of structured optimal policies observed through black-box policy queries. The central mathematical object considered throughout the paper is the decision-boundary geometry induced by an optimal policy. Rather than treating this geometry as a secondary by-product of dynamic programming, we study it as an independent representation possessing its own structural properties, statistical estimators, complexity measures, reconstruction guarantees, and information-theoretic characteristics. Within this framework, policy reconstruction, oracle-query allocation, decision complexity, and information-theoretic compression are shown to arise from a common geometric representation.

The theoretical development proceeds progressively. We first formalize the geometry induced by structured optimal policies and establish conditions under which this geometry admits an ordered, lowcomplexity representation. We then investigate the mathematical properties of this representation, derive intrinsic notions of geometric and decision complexity, characterize the relationship between geometric complexity and information-theoretic compression, and develop statistical guarantees for boundary estimation and policy reconstruction from black-box observations. These results collectively show that, under suitable structural assumptions, the mathematical complexity governing policy reconstruction is determined by the geometry of the decision boundary rather than by the cardinality of the ambient state space.

The perspective proposed here is complementary to the classical functional theory of dynamic programming rather than a replacement for it. Functional representations remain the natural language for computing optimal decisions. Once an optimal policy has been obtained, however, its induced geometry provides an alternative mathematical representation that exposes properties largely invisible from the functional viewpoint alone. In particular, it reveals direct connections between structural regularity, statistical estimation, active oracle sampling, intrinsic complexity, and information-theoretic compression that are dificult to formulate using functional representations alone.

The numerical investigation is designed to examine these theoretical predictions. Rather than comparing competing learning algorithms, every numerical experiment estimates a mathematical quantity introduced by the theory and evaluates whether the corresponding theoretical prediction is supported under a controlled black-box protocol. The empirical study therefore serves as a validation of the proposed mathematical framework rather than as an independent performance comparison.

The remainder of the paper is organized as follows. Section 2 positions the geometric viewpoint within the mathematical literature and classical structural dynamic programming. Section 3 formalizes policy-induced geometry, while Sections 4-7 develop its structural, geometric, complexity, and statistical foundations. Section 8 provides controlled numerical validation of the theory, and Section 9 discusses its scope and extensions.

## 2 Related Literature and Mathematical Positioning

The objective of this section is twofold. First, we position the present work within the mathematical literature on dynamic programming, stochastic optimization, computational geometry, and statistical learning. Second, we identify the mathematical objects that naturally emerge from these research directions and motivate the geometric formulation developed in the remainder of the paper. Accordingly, the discussion is organized around mathematical representations rather than research communities, application domains, or algorithmic paradigms. The mathematical theory of sequential decision-making has traditionally been formulated through functional representations of optimality, most notably value functions, state-action value functions, and optimal policies. Structural optimization subsequently established qualitative properties of these policies, including monotonicity, threshold behavior, and lattice structures. Computational geometry provides a rigorous language for describing partitions and geometric organization, whereas statistical learning studies the estimation of geometric objects from partial observations. Although these research directions have developed largely independently, they progressively reveal increasingly rich mathematical representations of the same underlying decision process. Viewed collectively, these developments suggest a common observation. In many structured stochastic optimization problems, an optimal policy induces a partition of the state space into regions associated with distinct optimal actions. The interfaces separating these regions define a geometric object that is completely determined by the optimal policy. Nevertheless, existing theory generally treats this geometry as a consequence of functional representations rather than as an independent mathematical object possessing its own statistical, structural, computational, and information-theoretic properties.

The present paper adopts precisely this alternative perspective. Instead of taking the value function or the policy as the primary object of analysis, the subsequent sections investigate the geometry induced by the optimal policy and examine the mathematical questions that naturally arise from this change of representation. As will become apparent throughout this review, this perspective progressively leads from functional representations to structured policies, from structured policies to decision-boundary geometry, from geometry to statistical inference, and ultimately from statistical inference to a unified theory of geometric complexity, information, and policy learning.

The remainder of this section is organized according to this mathematical progression. Section 2.1 reviews the classical functional objects underlying dynamic programming. Section 2.2 discusses the structural theory of optimal policies and the emergence of decision boundaries. Section 2.3 examines the geometry and intrinsic complexity of the induced state-space partitions. Section 2.4 positions the statistical estimation of policy-induced decision boundaries with respect to existing boundaryestimation theory. Finally, Section 2.5 explains how this geometric viewpoint naturally leads to a unified perspective on statistical complexity, information acquisition, and geometric policy learning.

## 2.1 Classical Objects in Dynamic Programming

Since the pioneering work of Bellman, the mathematical theory of sequential decision-making has been formulated around three closely related objects: the optimal value function $V ^ { \star }$ , the optimal state-action value function $Q ^ { \star }$ , and the optimal policy $\pi ^ { \star }$ . These objects constitute the fundamental mathematical representations underlying Markov decision processes and stochastic dynamic programming, and they remain the basis of both classical optimal control theory and modern reinforcement learning (Bellman, 1957; Puterman, 2014; Bertsekas, 2012; Hernández-Lerma and Lasserre, 2012; Feinberg and Shwartz, 2012).

Within this framework, optimal decision-making is characterized through recursive optimality equations whose solutions determine both the optimal expected return and the corresponding optimal decision rule. Consequently, the theoretical analysis of dynamic programming is expressed almost exclusively in terms of functional objects. Questions concerning existence, uniqueness, convergence, stability, approximation, and computational complexity are formulated with respect to either $V ^ { \star } , Q ^ { \star }$ or the policy $\pi ^ { \star }$ , leading to a mature mathematical theory encompassing value iteration, policy iteration, stochastic control, approximate dynamic programming, and large-scale optimization (Powell, 2007; Whittle, 1982; Bertsekas, 2025).

As increasingly complex decision problems have been considered, the principal computational challenge has become the approximation of these functional representations. Approximate dynamic programming and reinforcement learning therefore focus primarily on constructing computationally tractable approximations of value functions, action-value functions, or policies while preserving nearoptimal decision quality. The resulting notions of statistical eficiency and computational complexity are consequently defined relative to these functional objects.

This functional viewpoint has profoundly influenced the mathematical development of the field. At the same time, it implicitly determines the representation through which optimal decisions are analyzed. Once an optimal policy has been obtained, it induces a partition of the state space into regions associated with distinct optimal actions. While this partition is fundamental for understanding the qualitative structure of optimal decision-making, its geometric properties are generally regarded as secondary consequences of the policy itself rather than as primary mathematical objects. Existing theory therefore provides a comprehensive analysis of optimal values and optimal policies, but comparatively little attention has been devoted to the intrinsic geometry generated by the policy partition.

This observation suggests an alternative mathematical perspective. Rather than asking how accurately one can approximate $V ^ { \star } , Q ^ { \star }$ , or $\pi ^ { \star }$ , one may instead ask whether the geometric structure induced by the optimal policy can itself serve as the primary object of statistical inference. Adopting this viewpoint fundamentally changes the mathematical questions under consideration. Instead of studying approximation error for functional representations, the subsequent sections investigate geometric estimation, boundary reconstruction, structural complexity, information-theoretic compression, and the statistical properties of the geometry induced by structured optimal policies.

## 2.2 Structured Optimal Policies and Decision Boundaries

Beyond the existence of optimal policies, a major direction of stochastic dynamic programming has been devoted to identifying qualitative structural properties of optimal decision rules. Rather than viewing an optimal policy as an arbitrary mapping from states to actions, this line of research seeks conditions under which optimal decisions possess regularity properties that admit concise mathematical descriptions. Such structural characterizations have played a central role in operations research because they often reveal the intrinsic organization of optimal solutions independently of the numerical algorithms used to compute them.

The mathematical foundations of this theory are provided by the monotonicity and latticeprogramming results established by Veinott, Topkis, and subsequent authors. Under suitable assumptions involving submodularity, supermodularity, stochastic monotonicity, or lattice orderings, optimal actions evolve monotonically with respect to the underlying state variables, leading naturally to threshold-type decision rules (Veinott Jr, 1965; Veinott, 1969; Topkis, 1978, 1998). Similar structural properties have subsequently been established for a broad range of stochastic optimization problems, including inventory control, queueing systems, partially observed Markov decision pro cesses, and stochastic resource allocation (Stidham Jr and Weber, 1989; Sobel, 1982; Lovejoy, 1987; Smith and McCardle, 2002; Krishnamurthy, 2016; Koole, 2007).

From a mathematical viewpoint, these structural results admit a common interpretation. A threshold policy does not merely specify where the optimal action changes; it partitions the state space into subsets on which the optimal decision remains constant. In one-dimensional problems, this partition is determined by a finite collection of switching points. In higher-dimensional settings, the same principle extends naturally to switching curves, hypersurfaces, or more general decision interfaces separating neighboring action regions. Consequently, every structured optimal policy induces a partition of the state space whose organization is governed by a collection of decision boundaries.

This observation reveals an important shift of perspective. Classical structural optimization formulates its conclusions in terms of monotone policies, threshold values, or comparative statics. Yet these properties are mathematically equivalent to statements concerning the geometry of the induced partition.

Monotonicity determines the topology of the decision regions, thresholds determine the location of decision interfaces, and structural regularity constrains the geometric complexity of the resulting partition. In this sense, the geometry of the state-space partition is not an additional feature of the optimal policy; it is an equivalent representation of the structural information established by the classical theory. Despite this close relationship, the existing literature almost exclusively treats these geometric objects as implicit consequences of structural optimality. The principal mathematical objects remain the optimal policy, the threshold parameters, or the monotone decision rule itself. Comparatively little attention has been devoted to the statistical estimation of decision boundaries, to quantitative measures of their geometric complexity, or to the role that these geometric structures play in determining the sample complexity of policy reconstruction and the information-theoretic complexity of policy representation.

The developments presented in the subsequent sections adopt precisely this alternative viewpoint. Rather than regarding decision boundaries as by-products of structural optimization, the paper considers them as primary mathematical objects that encode the organization of structured optimal policies. This change of representation naturally transforms several classical questions. Policy reconstruction becomes a problem of geometric estimation; statistical accuracy is measured through boundary approximation; structural complexity becomes a property of the induced partition; and information-theoretic compression is governed by the complexity of the decision-boundary geometry rather than by the size of the ambient state space.

## 2.3 Geometry of Decision Regions and Structural Complexity

The structural interpretation developed in the previous subsection naturally leads to a broader mathematical question. Once an optimal policy partitions the state space into regions associated with distinct optimal actions, the resulting partition becomes a mathematical object that may be studied independently of the optimization procedure that generated it. Consequently, the analysis of structured decision processes is no longer restricted to functional representations such as value functions or policies; it also involves the geometry induced by the partition itself.

The mathematical study of geometric structures has long occupied a central role in convex analysis and computational geometry. Convex analysis characterizes the geometry of feasible sets, supporting hyperplanes, variational representations, and convex decompositions, whereas computational geometry investigates partitions, arrangements, polytopes, cell complexes, and their combinatorial complexity (Rockafellar, 1997; Boyd and Vandenberghe, 2004; Ziegler, 2012; Edelsbrunner, 1987; Preparata and Shamos, 2012). Although these theories were not originally developed for stochastic dynamic programming, they provide a rigorous mathematical language for describing geometric objects that arise naturally from structured optimization problems.

From this perspective, a partition possesses intrinsic properties that are independent of the objective function from which it originates. Its connected components, decision interfaces, adjacency relations, topological organization, and combinatorial complexity are properties of the partition itself. These quantities remain well defined regardless of the particular numerical representation of the value function and therefore describe structural aspects of an optimal decision process that cannot be inferred solely from functional approximation. For structured optimal policies, this distinction becomes particularly significant. The monotonicity and threshold properties discussed previously imply that neighboring decision regions are organized according to a relatively simple geometric structure. Consequently, the intrinsic complexity of the induced partition may remain substantially smaller than the apparent complexity suggested by the cardinality or dimensionality of the ambient state space. This observation indicates that geometric complexity should be regarded as an independent mathematical quantity rather than merely as a secondary consequence of value-function representations.

Despite these connections, the existing operations research literature rarely formulates complexity directly in terms of the geometry induced by optimal policies. Computational complexity is traditionally measured through the dimension of the state space, the complexity of functional approximation, or the computational cost of optimization algorithms. Likewise, statistical analyses typically quantify approximation error for value functions, policies, or estimators without explicitly characterizing the intrinsic complexity of the underlying decision partition.

As a consequence, no general mathematical framework currently relates the geometric organization of structured decision regions to statistical estimation, policy reconstruction, or information-theoretic representation. The viewpoint adopted in the present paper is motivated by this gap. Rather than measuring the complexity of a decision problem exclusively through its functional representation, the subsequent sections investigate the intrinsic complexity of the geometry induced by the optimal policy. This change of mathematical representation provides the foundation for the notions of boundary complexity, decision complexity, and decision compression developed later in the paper. Under this perspective, statistical estimation, sample complexity, and information-theoretic compression are interpreted as consequences of the geometric organization of the decision partition rather than of the dimensionality of the ambient state space.

## 2.4 Statistical Estimation of Policy-Induced Decision Boundaries

The geometric formulation developed in the previous subsections naturally transforms the underlying statistical inference problem. Once the partition induced by an optimal policy is regarded as the primary mathematical object, the objective is no longer to approximate a value function or a policy mapping, but rather to estimate the geometric interfaces separating neighboring optimal decision regions. Consequently, the statistical analysis shifts from functional approximation to geometric inference. The estimation of geometric boundaries has been extensively investigated in several areas of mathematical statistics and statistical learning. Density level-set estimation considers the recovery of regions determined by unknown probability densities (Tsybakov, 1997; Polonik, 1995), statistical learning theory studies classification boundaries generated by discriminant functions or supervised prediction models (Vapnik, 1999; Devroye et al., 2013; Anthony and Bartlett, 2009; Shalev-Shwartz and Ben-David, 2014), and free-boundary theory analyzes interfaces arising from variational inequalities and partial diferential equations (Cafarelli, 2005; Petrosyan et al., 2012). Although these research directions rely on diferent mathematical assumptions, they all investigate geometric interfaces generated by an underlying probabilistic, analytical, or variational model.

From a mathematical perspective, however, the object considered in the present paper is fundamentally diferent. In the aforementioned settings, the boundary is induced by an observable statistical mechanism, such as a probability density, a regression function, a discriminant function, or the solution of a variational problem. By contrast, the boundary investigated here is generated implicitly by an unknown optimal policy associated with a stochastic decision process. The value function, the state-action value function, threshold parameters, gradients, and the boundary itself are all unobserved. The only available information consists of evaluations of the optimal action returned by a black-box decision oracle. This distinction fundamentally changes the statistical formulation of the estimation problem. Classical boundary estimation relies on observations generated by an underlying statistical model, whereas policy-induced boundary estimation relies on adaptive interactions with an optimization oracle.

The statistical object is therefore not a density contour, a classification surface, or a free boundary, but the collection of state-space interfaces at which the optimal decision changes. Likewise, statistical accuracy is naturally quantified through geometric discrepancies between estimated and true decision boundaries rather than exclusively through prediction or classification error. The change of statistical object also modifies the notions of information acquisition and sample complexity. Since observations are obtained through adaptive oracle queries rather than passive sampling, the allocation of samples becomes an integral component of the inference problem. In particular, structural regularity may be exploited to concentrate observations near decision interfaces, suggesting that the statistical eficiency of boundary estimation is governed by the geometry of the induced partition rather than by the ambient state-space dimension alone.

The viewpoint adopted throughout this paper is motivated by these observations. Rather than interpreting policy learning as the approximation of functional representations, the subsequent analysis formulates it as the statistical estimation of a policy-induced geometric object observed exclusively through label-only oracle interactions. This formulation provides the mathematical foundation for the geometric estimators, Hausdorf convergence analysis, boundary sample complexity, and policy reconstruction guarantees developed in Section 7, where statistical inference is characterized directly in terms of the geometry generated by structured optimal policies.

## 2.5 Complexity, Information, and Geometric Policy Learning

The statistical formulation developed in the previous subsection naturally leads to a final mathematical question. Once the decision-boundary geometry induced by an optimal policy is regarded as the primary object of statistical inference, what determines the intrinsic dificulty of the corresponding learning problem? Within the geometric perspective adopted throughout this paper, this question is no longer answered solely by the complexity of functional representations, but by the amount of information encoded in the geometry of the induced decision partition. Classical learning theories quantify statistical complexity through functional approximation. Approximate dynamic programming studies the approximation of value functions and policies, reinforcement learning analyzes the statistical eficiency of policy optimization and value estimation, while statistical learning theory characterizes generalization through the complexity of hypothesis classes (Powell, 2007; Sutton et al., 1998; Lagoudakis and Parr, 2003; Munos and Szepesvári, 2008; Lazaric et al., 2010; Vapnik, 1999). Although these theories difer substantially in their mathematical formulation, they all measure learning complexity with respect to functional representations of the underlying decision problem.

The geometric viewpoint developed in the previous subsections suggests a complementary interpretation. Once an optimal policy induces a partition of the state space, the information required to reconstruct the policy is encoded by the organization of the induced decision boundaries rather than exclusively by the numerical representation of the value function. Consequently, the intrinsic difi culty of policy learning depends not only on the approximation of functional objects but also on the geometric regularity of the decision partition itself. This observation establishes a direct connection between geometry and statistical complexity. Smooth decision interfaces, low-complexity partitions, and regular geometric organization require comparatively little information for accurate reconstruction, whereas fragmented or highly irregular partitions demand substantially larger observational efort. Under this interpretation, statistical complexity becomes an intrinsic property of the geometry induced by the optimal policy rather than a consequence of the dimensionality of the ambient state space alone.

From an information-theoretic perspective, this change of viewpoint has important implications. The amount of information required to represent, estimate, and reconstruct an optimal policy is governed by the structural organization of the induced decision partition. Representation complexity, statistical complexity, active information acquisition, and policy reconstruction therefore become diferent manifestations of the same underlying geometric object. Rather than constituting independent questions, they admit a unified interpretation through the intrinsic complexity of the decision-boundary geometry. Despite the extensive literature on dynamic programming, reinforcement learning, statistical learning, and computational geometry, these diferent notions of complexity have largely been investigated independently. Existing theories typically analyze optimization, approximation, statistical estimation, or information representation in isolation. Comparatively little attention has been devoted to developing a unified mathematical framework in which geometric organization simultaneously determines representation complexity, statistical eficiency, information acquisition, and policy reconstruction.

The remainder of the paper develops precisely such a framework. Section 3 introduces the geometric representation of structured optimal policies. Sections 4 and 5 establish the corresponding structural and geometric foundations. Section 6 develops a theory of boundary complexity, decision complexity, and information-theoretic decision compression. Section 7 formulates the associated statistical learning theory through geometric estimators, Hausdorf convergence, active boundary sampling, and boundary sample complexity. Section 8 finally examines whether the theoretical predictions established throughout the paper are supported by controlled numerical experiments.

## 2.6 From Functional Representations to Geometric Learning

The preceding discussion reveals a common mathematical theme underlying several research directions in operations research, stochastic optimization, computational geometry, and statistical learning. Although these fields have developed largely independently, they progressively describe increasingly rich representations of the same underlying decision process.

Classical dynamic programming characterizes optimal decisions through functional objects; structural optimization establishes qualitative regularity properties of optimal policies; computational geometry provides a language for describing the partitions induced by these policies; and statistical learning investigates the estimation of geometric objects from partial observations.

Viewed collectively, these developments suggest that the geometry induced by an optimal policy constitutes a mathematical object deserving analysis in its own right. Once this viewpoint is adopted, several questions that traditionally appear unrelated become naturally connected. Policy reconstruction may be interpreted as geometric reconstruction of a decision partition. Statistical estimation becomes boundary estimation under oracle observations. Complexity becomes an intrinsic property of the induced geometry rather than of the ambient state space alone. Likewise, information acquisition and representation eficiency are governed by the structural organization of the decision boundaries rather than exclusively by functional approximation.

The theoretical developments presented in the remainder of this paper are organized around this change of mathematical representation. Section 3 introduces the geometric representation of structured optimal policies and formalizes the associated decision-boundary geometry. Sections 4 and 5 establish the structural, topological, and geometric properties of this representation. Section 6 develops a theory of boundary complexity, decision complexity, and information-theoretic decision compression. Section 7 formulates the corresponding statistical learning framework through geometric estimators, active boundary sampling, Hausdorf convergence, and boundary sample complexity. Finally, Section 8 examines whether the theoretical predictions established throughout the paper are supported by controlled numerical experiments.

Rather than extending existing approximation methods, the paper therefore develops a unified mathematical framework in which representation, statistical estimation, structural complexity, information acquisition, and policy learning are all interpreted through the geometry induced by structured optimal policies. The subsequent sections show that this geometric perspective provides a common language through which these questions may be analyzed within a single theoretical framework.

## 3 Policy Geometry

This section introduces the geometric framework that underlies the subsequent analysis. Building upon the classical theory of stochastic dynamic programming (Bellman, 1957; Puterman, 2014; Bertsekas, 2012; Hernández-Lerma and Lasserre, 2012), we formalize the mathematical objects through which optimal policies will be studied. The objective is not yet to establish structural results, but rather to define a geometric representation of optimal decision rules that will serve as the foundation for the theory developed in later sections.

## 3.1 Structured Markov Decision Processes

We consider a discounted Markov decision process (MDP)

$$
\begin{array} { r } { \mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , r , \gamma ) , } \end{array}\tag{1}
$$

where S denotes the state space, A is a finite action space, $\textstyle P ( \cdot \mid s , a )$ is the transition kernel, $r : S \times \mathcal { A }  \mathbb { R }$ is the one-period reward function, and $\gamma \in ( 0 , 1 )$ is the discount factor.

Definition 1 (Structured Markov Decision Process). A structured Markov decision process is an MDP of the form (1) endowed with additional order, monotonicity, convexity, submodularity, or stochastic-ordering properties that induce regularity in the corresponding optimal decision rules.

Let

$$
\pi : { \mathcal { S } }  A\tag{2}
$$

denote a stationary deterministic policy. The associated value function is defined by

$$
V ^ { \pi } ( s ) = \mathbb { E } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( S _ { t } , \pi ( S _ { t } ) ) \Bigg | S _ { 0 } = s \right] .\tag{3}
$$

The optimal value function is given by

$$
V ^ { * } ( s ) = \operatorname* { s u p } _ { \pi } V ^ { \pi } ( s ) , \quad s \in { \mathcal { S } } ,\tag{4}
$$

and an optimal policy $\pi ^ { * }$ satisfies

$$
V ^ { \pi ^ { * } } ( s ) = V ^ { * } ( s ) , \quad \forall s \in { \cal S } .\tag{5}
$$

The class of structured MDPs encompasses many models arising in operations research, including inventory systems, queueing networks, maintenance planning, admission-control problems, and partially observed decision processes (Veinott Jr, 1965; Topkis, 1998; Milgrom and Shannon, 1994; Smith and McCardle, 2002). In such settings, the state space often possesses an intrinsic ordering that plays a fundamental role in the characterization of optimal decisions.

Accordingly, we assume that the state space is endowed with a partial order relation

$$
( S , \preceq ) .\tag{6}
$$

Assumption 1 (Ordered State Space). The state space $\boldsymbol { \mathcal { S } }$ is a partially ordered set under the relation ⪯.

Assumption 1 is deliberately weak and serves only to establish a common framework encompassing the monotone dynamic programs considered in the literature. More restrictive assumptions involving monotonicity, submodularity, convexity, and increasing diferences will be introduced in Section 4, where they will be used to derive geometric properties of optimal policies.

The purpose of Definition 1 and Assumption 1 is to provide a mathematical environment in which geometric regularities of optimal decision rules may emerge. Classical structural assumptions are typically employed to establish monotonicity, threshold behavior, or comparative-statics properties. In contrast, our objective is to investigate how such assumptions shape the geometry of the optimal policy itself.

## 3.2 Optimal Policies as State-Space Partitions

The optimal policy $\pi ^ { * }$ induces a decomposition of the state space according to the actions selected under optimal decision making.

For each action $a \in { \mathcal { A } } .$ , define

$$
{ \mathcal { R } } _ { a } = \left\{ s \in { \mathcal { S } } : \pi ^ { * } ( s ) = a \right\} .\tag{7}
$$

Definition 2 (Optimal Action Region). For a given action $a \in A ,$ the set $\textstyle { \mathcal { R } } _ { a }$ defined in (7) is called the optimal action region associated with action a.

Since $\pi ^ { * }$ is a single-valued mapping, the family $\{ \mathcal { R } _ { a } \} _ { a \in \mathcal { A } }$ satisfies

$$
S = \bigcup _ { a \in { \mathcal { A } } } { \mathcal { R } } _ { a } ,\tag{8}
$$

and

$$
\mathcal { R } _ { a } \cap \mathcal { R } _ { a ^ { \prime } } = \emptyset , \quad a \neq a ^ { \prime } .\tag{9}
$$

Hence, the collection of optimal action regions forms a partition of the state space.

$$
\mathcal { T } ^ { * } = \left\{ \mathcal { R } _ { a } \right\} _ { a \in \mathcal { A } } .\tag{10}
$$

Definition 3 (Optimal Policy Tessellation). The partition $\mathcal { T } ^ { * }$ defined in (10) is called the optimal policy tessellation induced by the optimal policy $\pi ^ { * }$

The tessellation $\tau ^ { * }$ provides a geometric representation of the optimal policy through the partition of the state space into optimal action regions. The sets $\textstyle { \mathcal { R } } _ { a }$ constitute the fundamental geometric objects of the framework developed in this paper.

## 3.3 Decision Boundaries

Definition 3 shows that the optimal policy induces a tessellation

$$
{ \mathcal { T } } ^ { \star } = \{ { \mathcal { R } } _ { a } : a \in A \} ,
$$

which partitions the state space into optimal action regions. Beyond the regions themselves, the geometry of the tessellation is determined by the interfaces separating neighboring regions. These interfaces constitute the fundamental geometric objects studied throughout the remainder of the paper.

Throughout this paper, for every subset $A \subseteq s$ , we denote by $\overline { { A } }$ its closure and by int(A) its interior.

Definition 4 (Region Boundary). For every action $a \in { \mathcal { A } }$ , the boundary of the corresponding action region $\textstyle { \mathcal { R } } _ { a }$ is defined by

$$
\partial \mathcal { R } _ { a } = \overline { { \mathcal { R } _ { a } } } \cap \overline { { \mathcal { S } \setminus \mathcal { R } _ { a } } } .\tag{11}
$$

The boundary $\partial \mathcal { R } _ { a }$ consists of all states that separate $\textstyle { \mathcal { R } } _ { a }$ from the remainder of the state space. Definition 5 (Pairwise Decision Boundary). Let $a , a ^ { \prime } \in { \mathcal { A } }$ with $a \neq a ^ { \prime }$ . The decision boundary shared by the two action regions $\textstyle { \mathcal { R } } _ { a }$ and $\mathcal { R } _ { a ^ { \prime } }$ is defined by

$$
\Gamma _ { a a ^ { \prime } } = \overline { { \mathcal { R } _ { a } } } \cap \overline { { \mathcal { R } _ { a ^ { \prime } } } } .\tag{12}
$$

whenever this intersection is nonempty.

The set $\Gamma _ { a a ^ { \prime } }$ represents the common interface separating two adjacent optimal action regions. It is precisely across this interface that the optimal decision changes from action a to action $a ^ { \prime }$

Definition 6 (Global Decision Boundary). The global decision boundary associated with the optimal policy is

$$
\Gamma = \bigcup _ { a , a ^ { \prime } \in \mathcal { A } } \Gamma _ { a a ^ { \prime } } .\tag{13}
$$

The restriction $a < a ^ { \prime }$ avoids counting the same interface twice.

Definition $\mathbf { 7 }$ (Geometric Representation of an Optimal Policy). The pair

$$
( \mathcal { T } ^ { \star } , \Gamma )\tag{14}
$$

is called the geometric representation of the optimal policy.

The tessellation $\mathcal { T } ^ { \star }$ identifies the action regions, whereas Γ collects the interfaces separating them. Consequently, the geometric representation encodes both the partition of the state space induced by the optimal policy and the decision geometry governing transitions between neighboring action regions. This representation forms the mathematical foundation for the geometric estimation, complexity analysis, policy reconstruction, and decision-compression theory developed in the subsequent sections.

## 3.4 Policy Tessellations

The optimal action regions introduced in Section 3.2 and the boundary structure defined in Section 3.3 together determine the global geometric organization of an optimal policy.

The tessellation $\mathcal { T } ^ { * }$ specifies the partition of the state space into optimal action regions, while the set Γ describes the interfaces separating neighboring regions. These two components provide complementary information: the former characterizes the spatial allocation of optimal actions, whereas the latter characterizes the transitions between them.

$$
\mathcal { G } ^ { \ast } = ( \mathcal { T } ^ { \ast } , \Gamma ) .\tag{15}
$$

Definition 8 (Policy Geometry). The pair $\mathcal { G } ^ { * }$ defined in (15) is called the policy geometry associated with the optimal policy $\pi ^ { * }$

The policy geometry $\mathcal { G } ^ { * }$ may be viewed as a geometric arrangement of decision regions within the state space. In this representation, the regions $\textstyle { \mathcal { R } } _ { a }$ constitute the fundamental geometric cells of the arrangement, while the boundary structure Γ determines how these cells are connected and separated.

Such geometric representations are common in convex geometry, polyhedral theory, and computational geometry, where the properties of a partition are studied through the organization of its constituent regions and interfaces (Ziegler, 2012; Edelsbrunner, 1987). The present framework adopts a similar perspective for structured dynamic decision problems.

## 3.5 Geometric Complexity Measures

The policy geometry $\mathcal { G } ^ { * }$ introduced in Definition 8 provides a geometric representation of the optimal policy through its action regions and decision boundaries. To characterize the structural complexity of this geometry, we introduce a collection of complexity measures associated with the tessellation $\tau ^ { * }$ and its boundary structure Γ.

These measures quantify complementary aspects of policy geometry, including the complexity of decision regions, the organization of decision boundaries, and the degree of geometric regularity exhibited by the tessellation.

Definition 9 (Boundary Complexity). The boundary complexity of the policy geometry is defined by

$$
C _ { B } = \sum _ { a , a ^ { \prime } \in \mathcal { A } \atop a < a ^ { \prime } } \mathbf { 1 } _ { \left\{ \Gamma _ { a a ^ { \prime } } \neq \emptyset \right\} } ,\tag{16}
$$

where $\mathbf { 1 } _ { ( \cdot ) }$ denotes the indicator function.

The quantity $C _ { B }$ measures the number of distinct interfaces separating optimal action regions. Definition 10 (Region Complexity). The region complexity of the tessellation is defined by

$$
C _ { R } = \sum _ { a \in \mathcal { A } } N _ { c } ( \mathcal { R } _ { a } ) ,\tag{17}
$$

where $N _ { c } ( \cdot )$ denotes the number of connected components of a set.

The quantity $C _ { R }$ measures the topological complexity of the action regions composing the tessellation. Definition 11 (Fragmentation Index). The fragmentation index is defined by

$$
F = { \frac { C _ { R } } { | A | } } .\tag{18}
$$

The index F measures the average number of connected components per action region.

Definition 12 (Boundary Length). Assume that $S \subseteq \mathbb { R } ^ { d }$ . The total boundary length ofthe tessellation is defined by

$$
L = \sum _ { a , a ^ { \prime } \in \mathcal { A } \atop a < a ^ { \prime } } \mathcal { H } ^ { d - 1 } ( \Gamma _ { a a ^ { \prime } } ) ,\tag{19}
$$

where $\mathcal { H } ^ { d - 1 }$ denotes the (d − 1)-dimensional Hausdorf measure.

For $d = 2$ , the quantity L corresponds to the total length of the decision boundaries.

Definition 13 (Geometric Irregularity). Assume that each region $\textstyle { \mathcal { R } } _ { a }$ has finite Lebesgue measure $\mu ( \mathcal { R } _ { a } )$ . The geometric irregularity of the tessellation is defined by

$$
\kappa = \frac { L } { \sqrt { \sum _ { a \in \mathcal { A } } \mu ( \mathscr { R } _ { a } ) } } ,\tag{20}
$$

where $\mu$ denotes the Lebesgue measure on $\boldsymbol { s }$

The quantity κ measures the relative complexity of the boundary structure with respect to the overall size of the state space partition. Larger values of κ correspond to increasingly irregular policy geometries.

The measures introduced above are inspired by classical notions of geometric and combinatorial complexity arising in polyhedral and computational geometry (Ziegler, 2012; Edelsbrunner, 1987). In the present setting, they provide a quantitative description of the structural complexity of optimal policy tessellations.

## 3.6 Geometric Simplicity

The complexity measures introduced in Section 3.5 characterize complementary aspects of policy geometry. While each measure provides information about a specific structural feature of the tessellation, it is convenient to aggregate these quantities into a single measure of geometric complexity.

$$
\mathcal { C } ( \mathcal { G } ) = w _ { B } C _ { B } + w _ { R } C _ { R } + w _ { F } F + w _ { L } L + w _ { \kappa } \kappa ,\tag{21}
$$

where $w _ { B } , w _ { R } , w _ { F } , w _ { L } , w _ { \kappa } > 0$ are fixed weighting coeficients.

Definition 14 (Policy Geometry Complexity Index). The quantity ${ \mathcal { C } } ( { \mathcal { G } } )$ 1 defined in (21) is called the policy geometry complexity index.

The index ${ \mathcal { C } } ( { \mathcal { G } } )$ provides a global characterization of the structural complexity of a policy geometry by jointly accounting for the number of decision interfaces, the topological complexity of the action regions, the degree of fragmentation, the total boundary size, and the geometric irregularity of the tessellation.

Definition 15 (Geometrically Simple Policy). A policy geometry $\mathcal { G }$ is said to be geometrically simple if there exists a constant $K > 0$ such that

$$
{ \mathcal { C } } ( { \mathcal { G } } ) \leq K .\tag{22}
$$

The constant K quantifies the maximal geometric complexity compatible with the notion of simplicity. Smaller values of K correspond to increasingly regular policy geometries.

The previous definition applies to a single policy. It is often useful to extend the notion of geometric simplicity to an entire class of policies.

Definition 16 (Geometrically Simple Policy Class). Let Π denote a class of admissible policies. The class Π is said to be geometrically simple if there exists a constant $K > 0$ such that

$$
\operatorname* { s u p } _ { \pi \in \Pi } { \mathcal { C } } ( { \mathcal { G } } ( \pi ) ) \leq K .\tag{23}
$$

Definitions 15 and 16 provide an abstract framework for studying structural regularity in dynamic decision problems. They allow geometric properties of optimal policies to be characterized independently of the particular application under consideration and will serve as the basis for the structural results established in the next section.

## 3.7 Discussion

The structural dynamic programming literature has traditionally focused on the characterization of optimal actions through monotonicity, threshold behavior, convexity, and comparative-statics properties (Veinott Jr, 1965; Topkis, 1998; Milgrom and Shannon, 1994; Smith and McCardle, 2002). These results provide valuable insights into the qualitative behavior of optimal policies and have played a central role in the analysis of dynamic decision problems. The framework developed in this section adopts a complementary perspective. Rather than studying optimal policies solely as mappings from states to actions, we represent them through the geometric structures they induce on the state space. This representation naturally gives rise to action regions, decision boundaries, policy tessellations, and associated measures of geometric complexity.

The resulting geometric viewpoint provides a common language for describing structural regularities across a broad class of dynamic decision problems. In particular, it allows monotonicity, threshold behavior, and related structural properties to be interpreted through the geometry of the induced state-space partition. The subsequent analysis investigates how classical assumptions from structured dynamic programming constrain the geometry of optimal policy tessellations and thereby determine their intrinsic complexity.

## 4 Structural Dynamic Programs

## 4.1 Structural Assumptions

The geometric properties of optimal policy tessellations are closely linked to the structural characteristics of the underlying dynamic program. A substantial literature has identified conditions under which optimal policies exhibit monotonicity, threshold behavior, and comparative-statics properties (Veinott Jr, 1965; Veinott, 1969; Topkis, 1978, 1998; Milgrom and Shannon, 1994; Smith and McCardle, 2002). Throughout this section, the state space $( S , \preceq )$ is assumed to satisfy Assumption 1. In addition, the action space $\mathcal { A }$ is assumed to be endowed with a partial order, also denoted by $\preceq$

The following assumptions constitute the structural framework of the subsequent analysis.

Assumption 2 (Monotone Rewards). For every action $a \in A ,$ , the reward function is increasing with respect to the state order. Specifically,

$$
s _ { 1 } \preceq s _ { 2 } \quad \implies \quad r ( s _ { 1 } , a ) \leq r ( s _ { 2 } , a ) ,\tag{24}
$$

for all $s _ { 1 } , s _ { 2 } \in S$

Assumption 3 (Monotone Transitions). For every action $a \in { \mathcal { A } }$ , the transition kernel is monotone with respect to first-order stochastic dominance. That is,

$$
s _ { 1 } \preceq s _ { 2 } \quad \implies \quad P ( \cdot \mid s _ { 1 } , a ) \preceq _ { \mathrm { s t } } P ( \cdot \mid s _ { 2 } , a ) ,\tag{25}
$$

for all $s _ { 1 } , s _ { 2 } \in S$ , where ${ \preceq } _ { \mathrm { s t } }$ denotes the usual stochastic order.

Assumptions 2-3 constitute the classical framework of monotone dynamic programming and stochastic monotonicity (Veinott, 1969; Lovejoy, 1987; Koole, 2007).

Let

$$
Q ^ { * } ( s , a ) = r ( s , a ) + \gamma \int _ { S } V ^ { * } ( s ^ { \prime } ) \ : P ( d s ^ { \prime } \mid s , a )\tag{26}
$$

denote the optimal state-action value function.

Assumption 4 (Increasing Diferences). The function $Q ^ { * }$ exhibits increasing diferences in $( s , a )$ Specifically,

$$
Q ^ { * } ( s _ { 2 } , a _ { 2 } ) - Q ^ { * } ( s _ { 1 } , a _ { 2 } ) \geq Q ^ { * } ( s _ { 2 } , a _ { 1 } ) - Q ^ { * } ( s _ { 1 } , a _ { 1 } ) ,\tag{27}
$$

for all $s _ { 1 } \preceq s _ { 2 }$ and $a _ { 1 } \preceq a _ { 2 }$

Assumption 4 is a fundamental condition in monotone comparative statics and provides one of the principal mechanisms through which threshold structures emerge (Topkis, 1998; Milgrom and Shannon, 1994).

Assumption 5 (Submodularity). The function $Q ^ { * }$ is submodular on $s \times { \mathcal A }$ . Equivalently,

$$
Q ^ { * } ( s _ { 1 } , a _ { 1 } ) + Q ^ { * } ( s _ { 2 } , a _ { 2 } ) \leq Q ^ { * } ( s _ { 1 } , a _ { 2 } ) + Q ^ { * } ( s _ { 2 } , a _ { 1 } ) ,\tag{28}
$$

for all $s _ { 1 } \preceq s _ { 2 }$ and $a _ { 1 } \preceq a _ { 2 }$

Submodularity plays a central role in lattice programming and in the structural analysis of optimal policies (Topkis, 1978, 1998).

Definition 17 (Action Diference Function). For any pair of actions $a , a ^ { \prime } \in { \mathcal { A } }$ , define

$$
\Delta _ { a a ^ { \prime } } ( s ) = Q ^ { * } ( s , a ) - Q ^ { * } ( s , a ^ { \prime } ) ,\tag{29}
$$

for all $s \in { \mathcal { S } }$

Assumption 6 (Convex Action Diferences). Assume that $S \subseteq \mathbb { R } ^ { d }$ . For every pair of actions $a , a ^ { \prime } \in$ A, the function $\Delta _ { a a ^ { \prime } }$ defined in Definition 17 is convex on $s$

Assumption 6 connects structured dynamic programming with classical notions of convex analysis and optimization (Rockafellar, 1997; Boyd and Vandenberghe, 2004).

The assumptions introduced above encompass a broad class of dynamic decision problems arising in operations research, including inventory-control systems, queueing models, admission-control problems, maintenance planning, and partially observed Markov decision processes (Veinott Jr, 1965; Stidham Jr and Weber, 1989; Sobel, 1982; Smith and McCardle, 2002; Krishnamurthy, 2016).

## 4.2 Monotone Optimal Policies

The assumptions introduced in Section 4.1 belong to the classical framework of monotone dynamic programming. Their principal implication is the emergence of ordered optimal decision rules. We first recall standard monotonicity results and then derive their geometric consequences for the policy tessellation introduced in Section 3.

Proposition 1 (Monotonicity of the Optimal Value Function). Suppose that Assumptions $\mathcal { Q }$ and 3 hold. Then the optimal value function $V ^ { * }$ is increasing on $( S , \preceq )$ . Specifically,

$$
s _ { 1 } \preceq s _ { 2 } \quad \implies \quad V ^ { * } ( s _ { 1 } ) \leq V ^ { * } ( s _ { 2 } ) ,\tag{30}
$$

for all $s _ { 1 } , s _ { 2 } \in S$

Proof. The result follows from standard monotone dynamic programming arguments. Under Assumptions 2 and 3, the Bellman operator preserves monotonicity. Since $V ^ { * }$ is the unique fixed point of the Bellman operator, the claim follows from successive approximation; see Veinott (1969), Puterman (2014), and Hernández-Lerma and Lasserre (2012).

The monotonicity of the optimal value function provides the foundation for the monotonicity of optimal decision rules.

Proposition 2 (Monotone Optimal Policy). Suppose that Assumptions ${ \it 2 , 3 , }$ and $\it 4$ hold.

Then there exists an optimal policy $\pi ^ { * }$ that is increasing on $( S , \preceq )$ . That $i s ,$

$$
s _ { 1 } \preceq s _ { 2 } \quad \implies \quad \pi ^ { * } ( s _ { 1 } ) \preceq \pi ^ { * } ( s _ { 2 } ) ,\tag{31}
$$

for all $s _ { 1 } , s _ { 2 } \in S$

Proof. By Assumption 4, the function $Q ^ { * }$ satisfies increasing diferences in $( s , a )$ . Topkis’ monotonicity theorem therefore implies that the set of optimal actions is increasing with respect to the state variable. Consequently, there exists a monotone optimal selector $\pi ^ { * } ;$ (see Topkis, 1998) and Milgrom and Shannon (1994). □

The monotonicity of $\pi ^ { * }$ imposes strong restrictions on the geometry of the optimal action regions introduced in Definition 2.

Theorem 3 (Ordered Region Theorem). Suppose that the assumptions of Proposition 2 hold. Then every optimal action region $\textstyle { \mathcal { R } } _ { a }$ is order-convex. That is, whenever

$$
s _ { 1 } , s _ { 2 } \in \mathcal { R } _ { a } , \quad s _ { 1 } \preceq s \preceq s _ { 2 } ,\tag{32}
$$

it follows that

$$
s \in \mathcal { R } _ { a } .\tag{33}
$$

Proof. Let $s _ { 1 } , s _ { 2 } \in \mathcal { R } _ { a }$ with $s _ { 1 } \preceq s _ { 2 }$ . Since $\pi ^ { * }$ is increasing, the action selected by the optimal policy cannot decrease between comparable states. If there existed a state s such that $s _ { 1 } \preceq s \preceq s _ { 2 }$ and $\pi ^ { * } ( s ) \neq a ,$ then monotonicity of $\pi ^ { * }$ would be violated. Hence every intermediate state must belong to the same action region. □

Theorem 3 establishes the first direct connection between structural dynamic programming and policy geometry. Monotonicity does not merely constrain the optimal action selected at a given state; it also constrains the geometric organization of the regions composing the policy tessellation.

Corollary 4 (Region Complexity Bound). Under the assumptions of Theorem ${ \mathcal { B } } ,$

$$
C _ { R } = O ( | { \cal { A } } | ) .\tag{34}
$$

Proof. Order-convex regions cannot exhibit arbitrary fragmentation. Each action region contributes at most a finite number of connected ordered components. Consequently, the total number of connected components grows at most linearly with the number of actions.

## 4.3 Threshold Representations

Threshold policies constitute one of the most important structural phenomena in dynamic programming and operations research. Such policies arise in a broad range of applications, including inventory control, queueing systems, admission-control problems, and maintenance planning (Scarf, 1960; Veinott Jr, 1965; Sobel, 1982; Stidham Jr and Weber, 1989). The monotonicity result established in Proposition 2 admits a more explicit characterization when the state space contains a distinguished ordered component. Throughout this subsection, assume that

$$
\mathcal { S } = \mathcal { X } \times \mathcal { Z } ,\tag{35}
$$

where $\mathcal { X } \subseteq \mathbb { R }$ is totally ordered and $\mathcal { Z }$ denotes an arbitrary auxiliary state space.

Theorem 5 (Threshold Representation Theorem). Suppose that Assumptions $\mathcal { Q } , \ \mathcal { B } , \ \mathcal { 4 } ,$ and 5 hold. Then there exists a collection of threshold functions

$$
b _ { a a ^ { \prime } } : \mathcal { Z }  \mathcal { X } ,\tag{36}
$$

such that, for every pair of adjacent actions $a \prec a ^ { \prime }$ ，

$$
\pi ^ { * } ( x , z ) = a \quad \Longleftrightarrow \quad x < b _ { a a ^ { \prime } } ( z ) ,\tag{37}
$$

and

$$
\pi ^ { * } ( x , z ) = a ^ { \prime } \quad \Longleftrightarrow \quad x \ge b _ { a a ^ { \prime } } ( z ) .\tag{38}
$$

Proof. By Proposition 2, the optimal policy is increasing with respect to the ordered state variable. Consequently, transitions between adjacent actions occur through monotone switching surfaces. The existence of threshold functions follows from standard lattice-theoretic arguments and monotone comparative statics (Topkis, 1998; Milgrom and Shannon, 1994).

The threshold representation admits a natural geometric interpretation in terms of the decision boundaries introduced in Definition 5.

Corollary 6 (Boundary Graph Representation). Under the assumptions of Theorem $^ { 5 , }$ every decision boundary between adjacent action regions is the graph of a threshold function.

Specifically,

$$
\Gamma _ { a a ^ { \prime } } = \{ ( x , z ) \in \mathcal S : x = b _ { a a ^ { \prime } } ( z ) \} .\tag{39}
$$

Proof. The result follows immediately from (37) and (38). The transition between neighboring action regions occurs precisely when the ordered state variable reaches the threshold value.

Corollary 6 shows that threshold policies induce highly structured decision boundaries. Rather than forming arbitrary subsets of the state space, the boundaries are constrained to lie on lowdimensional graphs.

A further consequence concerns the fragmentation properties of the tessellation.

Corollary 7 (Absence of Fragmentation). Under the assumptions of Theorem 5, each optimal action region is connected.

Consequently,

$$
F = 1 .\tag{40}
$$

Proof. Each action region is generated by threshold inequalities involving the ordered state variable x. Such regions are connected and therefore possess a single connected component.

Hence

$$
C _ { R } = | { \mathcal { A } } | .\tag{41}
$$

By Definition 11,

$$
F = { \frac { C _ { R } } { | { \mathcal { A } } | } } = 1 .\tag{42}
$$

## 4.4 Convex Policy Regions

The monotonicity and threshold structures established in Sections 4.2 and 4.3 constrain the ordering of optimal decisions. A stronger form of geometric regularity emerges when the dominance relations induced by the optimal state–action value function generate convex decision sets. Such conditions arise naturally in several classes of structured dynamic programs and connect the geometry of optimal policies with classical convex analysis (Rockafellar, 1997; Boyd and Vandenberghe, 2004).

For every pair of actions a, $a ^ { \prime } \in { \mathcal { A } }$ , define the dominance region

$$
\mathcal { D } _ { a a ^ { \prime } } = \left\{ s \in \mathcal { S } : Q ^ { * } ( s , a ) \geq Q ^ { * } ( s , a ^ { \prime } ) \right\} .\tag{43}
$$

The set $\mathcal { D } _ { a a ^ { \prime } }$ contains all states for which action a weakly dominates action $a ^ { \prime }$

Assumption 7 (Convex Dominance Regions). For every pair of actions $a , a ^ { \prime } \in { \mathcal { A } }$ , the dominance region $\mathcal { D } _ { a a ^ { \prime } }$ defined in (43) is convex.

Assumption 7 may be viewed as a geometric strengthening of the structural assumptions introduced in Section 4.1. It is satisfied whenever the pairwise dominance relations generated by the optimal value structure admit convex representations.

The following result establishes the geometric structure of the optimal action regions.

Theorem 8 (Convex Region Theorem). Suppose that Assumption 7 holds.

Then every optimal action region $\textstyle { \mathcal { R } } _ { a }$ is convex.

Proof. By Definition 2, a state belongs to $\textstyle { \mathcal { R } } _ { a }$ if and only if action a weakly dominates every competing action. Consequently,

$$
\mathcal { R } _ { a } = \bigcap _ { a ^ { \prime } \neq a } \mathcal { D } _ { a a ^ { \prime } } .\tag{44}
$$

Since each dominance region $\mathcal { D } _ { a a ^ { \prime } }$ is convex by Assumption 7, and intersections of convex sets remain convex (Rockafellar, 1997; Boyd and Vandenberghe, 2004), it follows that $\textstyle { \mathcal { R } } _ { a }$ is convex.

Theorem 8 provides a direct connection between structural properties of the dynamic program and the geometry of the induced policy tessellation. In particular, convexity excludes disconnected action regions and highly fragmented decision structures.

Corollary 9 (Connected Action Regions). Under the assumptions of Theorem 8, every optimal action region $\textstyle { \mathcal { R } } _ { a }$ is connected.

Proof. Every convex subset of a Euclidean space is connected. The result therefore follows immediately from Theorem 8.

The absence of disconnected regions has immediate implications for the geometric complexity measures introduced in Section 3.

Corollary 10 (Region Complexity). Under the assumptions of Theorem $\delta ,$

$$
C _ { R } = | { \mathcal { A } } | .\tag{45}
$$

Proof. By Corollary 9, each action region contributes exactly one connected component. Since the tessellation contains |A| regions, the result follows directly from the definition of $C _ { R }$

Corollary 11 (Absence of Fragmentation). Under the assumptions of Theorem $\delta ,$

$$
F = 1 .\tag{46}
$$

Proof. Combining Definition 11 with (45) yields

$$
F = { \frac { C _ { R } } { | { \mathcal { A } } | } } = 1 .
$$

The convexity of the action regions also constrains the geometry of the decision boundaries.

Corollary 12 (Boundary Structure). Under the assumptions of Theorem 8, the decision boundary between two actions a and $a ^ { \prime }$ satisfies

$$
\Gamma _ { a a ^ { \prime } } \subseteq \partial { \cal D } _ { a a ^ { \prime } } .\tag{47}
$$

Moreover,

$$
\Gamma _ { a a ^ { \prime } } \subseteq \left\{ s \in S : Q ^ { * } ( s , a ) = Q ^ { * } ( s , a ^ { \prime } ) \right\} .\tag{48}
$$

Proof. A transition from $\textstyle { \mathcal { R } } _ { a }$ to $\mathcal { R } _ { a ^ { \prime } }$ can occur only at states where neither action strictly dominates the other. Hence any boundary point must satisfy

$$
Q ^ { * } ( s , a ) = Q ^ { * } ( s , a ^ { \prime } ) ,
$$

which is equivalent to belonging to the indiference set

$$
\left\{ s \in S : Q ^ { * } ( s , a ) = Q ^ { * } ( s , a ^ { \prime } ) \right\} .
$$

Since the dominance relation changes across the boundary, such points necessarily belong to the boundary of the dominance region $\mathcal { D } _ { a a ^ { \prime } }$

Corollaries 10-12 show that convexity imposes strong geometric restrictions on optimal policy tessellations. In contrast to general decision partitions, convex policy regions exhibit minimal fragmentation, admit simple connectivity properties, and generate decision boundaries that coincide with indiference surfaces between competing actions.

## 4.5 Geometric Simplicity

The results established in the previous subsections reveal a common phenomenon. Under standard structural assumptions from monotone dynamic programming and comparative statics, the geometry of the optimal policy tessellation remains highly organized despite the potentially large dimension of the underlying state space. We now summarize these implications through the geometric complexity measures introduced in Section 3. The first result concerns the complexity of the boundary structure.

Theorem 13 (Boundary Complexity Bound). Suppose that the assumptions of Theorem 5 hold. Then the boundary complexity satisfies

$$
C _ { B } = O ( | { \cal { A } } | ) .\tag{49}
$$

Proof. By Corollary 6, each nonempty decision boundary corresponds to a threshold surface separating two adjacent action regions.

Since the action space is finite, each action can generate only a finite number of adjacent transitions. Consequently, the number of nonempty pairwise boundaries grows at most linearly with the number of actions. Therefore, $C _ { B } \leq | { \cal { A } } | - 1$ , which implies (49).

The next result characterizes the fragmentation properties of the policy tessellation.

Theorem 14 (Minimal Fragmentation). Suppose that the assumptions of Theorem 8 hold. Then

$$
F = 1 .\tag{50}
$$

Moreover, F = 1 is the smallest attainable value of the fragmentation index.

Proof. By Corollary 10, $C _ { R } = | { \mathcal { A } } |$

Substituting into Definition 11 yields $\begin{array} { r } { F = \frac { C _ { R } } { | \mathcal { A } | } = 1 } \end{array}$

Furthermore, every action region contributes at least one connected component. Hence $C _ { R } \geq | { \mathcal { A } } |$ which implies $F \geq 1$ . Therefore, $F = 1$ is the minimum achievable fragmentation level.

The previous results suggest that structured dynamic programs generate policy tessellations whose complexity is governed primarily by the action space rather than by the size of the state space.

To formalize this observation, we introduce a topological complexity index.

Definition 18 (Topological Complexity Index). The topological complexity of an optimal policy tessellation is defined by

$$
\mathcal { G } _ { \mathrm { t o p } } ( \pi ) = w _ { B } C _ { B } + w _ { R } C _ { R } + w _ { F } F ,\tag{51}
$$

where $w _ { B } , w _ { R } , w _ { F } \ge 0$ are fixed weights.

The following theorem constitutes the main structural conclusion of this section.

Theorem 15 (Geometric Simplicity Theorem). Suppose that

1. Assumptions 2-5 hold;

2. the threshold representation of Theorem 5 holds;

3. the convexity condition of Theorem 8 holds.

Then

$$
C _ { B } = O ( | { \cal { A } } | ) ,
$$

$$
C _ { R } = | { \mathcal { A } } | ,\tag{52}
$$

$$
F = 1 ,\tag{53}
$$

(54)

and consequently

$$
\mathcal { G } _ { \mathrm { t o p } } ( \pi ^ { * } ) = O ( | \mathcal { A } | ) .\tag{55}
$$

Proof. Equation (52) follows from Theorem 13.

Equation (53) follows from Corollary 10.

Equation (54) follows from Theorem 14.

Substituting these relations into Definition 18 gives

$$
\mathcal { G } _ { \mathrm { t o p } } ( \pi ^ { * } ) = w _ { B } O ( | \boldsymbol { A } | ) + w _ { R } | \boldsymbol { A } | + w _ { F } .
$$

Therefore,

$$
\mathcal { G } _ { \mathrm { t o p } } ( \pi ^ { * } ) = O ( | \mathcal { A } | ) .
$$

Theorem 15 establishes that broad classes of structured dynamic programs generate intrinsically simple policy tessellations. Although the underlying state space may be large, continuous, or highdimensional, the geometric organization of the optimal policy remains controlled by a number of regions and boundaries that scales only with the cardinality of the action space. This observation provides the foundation for the decision-compression and learning results developed in the subsequent sections.

## 4.6 Decision Compression

The geometric simplicity results established in the previous subsections suggest a distinction between two fundamentally diferent notions of complexity in dynamic decision problems.

The first concerns the complexity of evaluating future rewards through the optimal value function. The second concerns the complexity of representing the optimal decision rule itself. While these notions are often implicitly conflated in dynamic programming, the geometric framework developed in this paper allows them to be analyzed separately.

Definition 19 (Value Complexity). Let $V ^ { * }$ denote the optimal value function associated with the Markov decision process. The value complexity of the problem is defined by

$$
\mathcal { C } _ { V } = \mathcal { C } ( V ^ { * } ) ,\tag{56}
$$

where $\mathcal { C } ( \cdot )$ denotes a nonnegative complexity functional on a prescribed class of value functions.

The functional $\mathcal { C } ( \cdot )$ is intentionally left unspecified. Depending on the application, it may correspond to approximation dimension, description length, parameter count, covering complexity, or another suitable measure of functional complexity.

The geometric analysis of Sections 3 and 4 naturally induces a second notion of complexity based on the structure of the optimal policy tessellation.

Definition 20 (Decision Complexity). Let

$$
\mathcal { G } _ { \mathrm { t o p } } ( \pi ^ { * } ) = w _ { B } C _ { B } + w _ { R } C _ { R } + w _ { F } F ,\tag{57}
$$

denote the topological complexity index introduced in Definition $^ { 1 8 , }$ where $w _ { B } , w _ { R } , w _ { F } \ge 0$

The decision complexity of the optimal policy is defined by

$$
\begin{array} { r } { \mathcal { C } _ { D } = \mathcal { G } _ { \mathrm { t o p } } ( \pi ^ { * } ) . } \end{array}\tag{58}
$$

The quantity $\mathcal { C } _ { D }$ measures the complexity of the optimal decision rule through the geometry of its induced tessellation rather than through the representation of the value function.

This distinction motivates the following notion.

Definition 21 (Decision Compression Ratio). The decision compression ratio is defined as

$$
\mathrm { D C R } = { \frac { \mathcal C _ { V } } { \mathcal C _ { D } } } .\tag{59}
$$

Large values of DCR indicate that the optimal policy admits a substantially simpler representation than the associated value function.

The geometric simplicity results obtained in Section 3.6 imply that the complexity of the policy tessellation grows at most linearly with the cardinality of the action space.

The following theorem formalizes the resulting compression phenomenon.

Theorem 16 (Decision Compression Theorem). Suppose that the assumptions of Theorem 15 hold. Then

$$
\mathcal { C } _ { D } = O ( | \mathcal { A } | ) .\tag{60}
$$

Consequently,

$$
\mathrm { D C R } = \Omega \bigg ( \frac { { \mathcal C } _ { V } } { | A | } \bigg ) .\tag{61}
$$

Proof. By Theorem 15, $C _ { B } = O ( | \boldsymbol { A } | ) , C _ { R } = | \boldsymbol { A } |$ , and $F = 1$

Substituting these relations into (57) yields

$$
\mathcal { C } _ { D } = w _ { B } O ( | \boldsymbol { A } | ) + w _ { R } | \boldsymbol { A } | + w _ { F } .\tag{62}
$$

Hence, $\mathcal { C } _ { D } = O ( | \mathcal { A } | )$

Combining this relation with Definition 21 gives

$$
\mathrm { D C R } = \frac { \mathcal { C } _ { V } } { O ( | \mathcal { A } | ) } = \Omega \left( \frac { \mathcal { C } _ { V } } { | \mathcal { A } | } \right) ,
$$

which establishes (61).

Theorem 16 highlights a structural separation between value complexity and decision complexity. Under the regularity conditions commonly encountered in structured dynamic programs, the geometric complexity of the optimal policy remains controlled by the action space, whereas the complexity of the value function may continue to increase with the complexity of the state space. The optimal policy tessellation can therefore be interpreted as a compressed geometric representation of optimal decision making.

## 4.7 Discussion

The results developed throughout this section establish a unified connection between classical structural properties of dynamic programs and the geometry of optimal policies.

The starting point is the well-established theory of monotone dynamic programming and comparative statics developed by Veinott (1969), Milgrom and Shannon (1994), Topkis (1998), and subsequent authors (Veinott Jr, 1965; Smith and McCardle, 2002). Under standard monotonicity, submodularity, and increasing-diferences assumptions, optimal decision rules exhibit ordered behavior and admit threshold representation.

The geometric framework introduced in this paper shows that these structural properties have a direct topological interpretation. Monotonicity induces ordered action regions, threshold policies generate low-dimensional decision boundaries, and convex dominance relations produce connected and convex policy regions. Together, these properties imply that optimal policy tessellations possess limited geometric complexity despite the potentially large size of the underlying state space.

A central implication of this observation is the emergence of decision compression. While the complexity of the optimal value function may increase substantially with the dimensionality of the state space, the complexity of the corresponding policy tessellation remains governed primarily by the action space. Consequently, optimal decisions may admit representations that are considerably simpler than those required to describe the full value function.

The chain of implications established in this section may be summarized as

$$
{ \mathrm { M o n o t o n i c i t y } } \implies { \mathrm { T h r e s h o l d ~ S t r u c t u r e } } \implies { \mathrm { G e o m e t r i c ~ S i m p l i c i t y } } \implies { \mathrm { D e c i s i o n ~ C o m p r e s s i o n } } .\tag{63}
$$

This perspective provides a geometric interpretation of structural dynamic programming and motivates the subsequent study of learning and approximation methods that exploit the low-complexity structure of optimal policy tessellations.

## 5 Structural Geometry of Optimal Policies

## 5.1 Structural Geometry as a Compressed Representation

The geometric framework introduced in Section 3 associates with each deterministic policy a partition of the state space together with the corresponding boundary structure. The purpose of this subsection is to formalize this correspondence and to establish that the geometry induced by a policy contains all information required to recover the policy itself.

Let Π denote the set of admissible deterministic policies. For every policy $\pi \in \Pi$ , define the collection of action regions

$$
{ \mathcal { R } } _ { a } ( \pi ) = \left\{ s \in S : \pi ( s ) = a \right\} , \qquad a \in A .\tag{64}
$$

The corresponding tessellation is

$$
{ \mathcal T } ( \pi ) = \{ { \mathcal R } _ { a } ( \pi ) \} _ { a \in \mathcal A } ,\tag{65}
$$

and the associated boundary structure is

$$
\Gamma ( \pi ) = \bigcup _ { a , a ^ { \prime } \in \mathcal { A } \atop a \neq a ^ { \prime } } \Gamma _ { a a ^ { \prime } } ( \pi ) ,\tag{66}
$$

where

$$
\Gamma _ { a a ^ { \prime } } ( \pi ) = \partial \mathcal { R } _ { a } ( \pi ) \cap \partial \mathcal { R } _ { a ^ { \prime } } ( \pi ) .\tag{67}
$$

Definition 22 (Policy Geometry). The geometric representation associated with a policy $\pi \in \Pi$ is defined by

$$
\mathcal { G } ( \pi ) = ( \mathcal { T } ( \pi ) , \Gamma ( \pi ) ) .\tag{68}
$$

The set

$$
{ \mathfrak { G } } = \{ { \mathcal { G } } ( \pi ) : \pi \in \Pi \}\tag{69}
$$

is called the family of policy-induced geometries.

The inclusion of $\Gamma ( \pi )$ in (68) is not required to reconstruct the policy once the tessellation is known. Nevertheless, the boundary structure plays a central role in the complexity measures introduced in Section 3 and will be essential for the learnability results developed later in the paper.

The following proposition establishes that the tessellation generated by a deterministic policy uniquely determines that policy.

Proposition 17. Let $\pi _ { 1 } , \pi _ { 2 } \in \Pi$ . Then

$$
{ \mathcal T } ( \pi _ { 1 } ) = { \mathcal T } ( \pi _ { 2 } )\tag{70}
$$

if and only if

$$
\pi _ { 1 } = \pi _ { 2 } .\tag{71}
$$

Consequently,

$$
\mathcal { G } ( \pi _ { 1 } ) = \mathcal { G } ( \pi _ { 2 } ) \quad \Longleftrightarrow \quad \pi _ { 1 } = \pi _ { 2 } .\tag{72}
$$

Proof. Suppose first that $\pi _ { 1 } = \pi _ { 2 }$ . Then $\mathcal { R } _ { a } ( \pi _ { 1 } ) = \mathcal { R } _ { a } ( \pi _ { 2 } )$ for every $a \in { \mathcal { A } } .$ , which immediately implies ${ \mathcal T } ( \pi _ { 1 } ) = { \mathcal T } ( \pi _ { 2 } )$ . Conversely, assume that ${ \mathcal T } ( \pi _ { 1 } ) = { \mathcal T } ( \pi _ { 2 } )$

Let $s \in { \mathcal { S } }$ . Since a tessellation is a partition of the state space, there exists a unique action $a \in { \mathcal { A } }$ such that

$$
s \in \mathcal { R } _ { a } ( \pi _ { 1 } ) .\tag{73}
$$

Because the tessellations coincide,

$$
\mathcal { R } _ { a } ( \pi _ { 1 } ) = \mathcal { R } _ { a } ( \pi _ { 2 } ) ,\tag{74}
$$

and therefore

$$
\pi _ { 1 } ( s ) = a = \pi _ { 2 } ( s ) .\tag{75}
$$

Since the argument holds for every $s \in { \mathcal { S } }$ , it follows that

$$
\pi _ { 1 } = \pi _ { 2 } .\tag{76}
$$

The final statement follows immediately from the fact that $\Gamma ( \pi )$ is uniquely determined by $\mathcal T ( \pi )$

Proposition 17 shows that the policy geometry is not merely a graphical representation of a decision rule. The induced tessellation provides an equivalent description of the policy itself. Consequently, structural properties of optimal policies may be studied through the geometry of the associated tessellations without loss of information.

This observation forms the basis of the geometric analysis developed in the remainder of the paper.

## 5.2 Policy Boundary Principle

The geometric representation developed in the previous subsection establishes that a deterministic policy may be identified with its induced tessellation. We now characterize the precise geometric locus at which variations of the policy can occur.

Recall that, for a policy $\pi \in \Pi$ , the state space is partitioned into action regions

$$
{ \mathcal T } ( \pi ) = \{ { \mathcal R } _ { a } ( \pi ) \} _ { a \in \mathcal A } ,\tag{77}
$$

and that the associated boundary structure is

$$
\Gamma ( \pi ) = \bigcup _ { a , a ^ { \prime } \in \mathcal A } \Gamma _ { a a ^ { \prime } } ( \pi ) .\tag{78}
$$

The following result identifies the boundary structure as the unique geometric support of decision changes.

Theorem 18 (Policy Boundary Principle). Let $\pi \in \Pi$ be a deterministic policy. Then the following statements hold.

(i) For every state

$$
s \in S \setminus \Gamma ( \pi ) ,\tag{79}
$$

there exists an open neighborhood $U ( s ) \subseteq { \mathcal { S } }$ such that

$$
\pi ( u ) = \pi ( s ) , \quad \forall u \in U ( s ) .\tag{80}
$$

(ii) Every local change of the policy occurs on the boundary structure. More precisely, if

$$
s \in { \mathcal { S } }\tag{81}
$$

is such that every neighborhood of s contains points assigned to at least two distinct actions, then

$$
s \in \Gamma ( \pi ) .\tag{82}
$$

Consequently,

$$
\Gamma ( \pi ) = \left\{ s \in { \mathcal S } : \pi \ i s \ n o t \ l o c a l l y \ c o n s t a n t \ a t \ s \right\} .\tag{83}
$$

Proof. We first establish (i).

Let $s \in { \mathcal { S } } \setminus \Gamma ( \pi )$

Since $\mathcal T ( \pi )$ is a partition of the state space, there exists a unique action $a \in { \mathcal { A } }$ such that

$$
s \in \mathcal { R } _ { a } ( \pi ) .
$$

Because $s \notin \Gamma ( \pi )$ , the point s belongs to the interior of $\mathcal { R } _ { a } ( \pi )$ . Hence there exists an open neighborhood $U ( s )$ satisfying

$$
U ( s ) \subseteq { \mathcal { R } } _ { a } ( \pi ) .\tag{84}
$$

By definition of $\mathcal { R } _ { a } ( \pi )$ 2

$$
\pi ( u ) = a = \pi ( s ) , \quad \forall u \in U ( s ) ,\tag{85}
$$

which proves (i).

We now prove (ii).

Suppose that $s \not \in \Gamma ( \pi )$ . By part (i), there exists an open neighborhood on which the policy is constant. Therefore it is impossible for every neighborhood of s to contain points assigned to diferent actions.

Taking the contrapositive yields

$$
^ { \mathfrak { c } } \pi { \mathrm { ~ n o t ~ l o c a l l y ~ c o n s t a n t ~ a t ~ } } s ^ { \mathfrak { \prime } } \Longrightarrow s \in \Gamma ( \pi ) .
$$

Combining this implication with part (i) establishes (83).

Theorem 18 provides a complete geometric characterization of policy variation. The boundary structure is not merely a collection of interfaces between action regions; it is precisely the set of states at which local decision changes may occur.

Consequently, the geometry of a policy may be viewed as consisting of two qualitatively distinct components. The interiors of action regions correspond to locally invariant decision zones, whereas the boundary structure contains the entire locus of decision transitions. This localization property will play a central role in the complexity and compression analyses developed in the subsequent sections.

## 5.3 Dimension-Free Policy Geometry

The previous subsection established that the entire variability of a deterministic policy is concentrated on its boundary structure. Consequently, the complexity of a policy may be studied through the combinatorial organization of its action regions and decision boundaries rather than through the ambient state space itself.

This observation naturally raises the following question: to what extent does the complexity of a policy geometry depend on the dimension of the underlying state space?

To address this issue, we introduce a complexity measure based exclusively on the tessellation structure.

Definition 23 (Topological Policy Complexity). Let $\mathcal { G } ( \pi ) = ( \mathcal { T } ( \pi ) , \Gamma ( \pi ) )$ be the geometry induced by a deterministic policy.

The associated topological complexity is defined by

$$
\mathcal { C } _ { \mathrm { t o p } } ( \pi ) = C _ { R } ( \pi ) + C _ { B } ( \pi ) + F ( \pi ) ,\tag{86}
$$

where

$C _ { R } ( \pi )$ denotes the region complexity introduced in Definition $1 0 ;$

$C _ { B } ( \pi )$ denotes the boundary complexity introduced in Definition ${ \mathit { g } } _ { \mathrm { , } }$

$F ( \pi )$ denotes the fragmentation index introduced in Definition 11.

The quantity $\mathcal { C } _ { \mathrm { t o p } } ( \pi )$ depends only on the combinatorial structure of the tessellation and is therefore independent of metric notions such as volume, diameter, curvature, or ambient dimension.

The next result shows that structural dynamic programs generate geometries whose complexity is controlled entirely by the action space.

Theorem 19 (Dimension-Free Geometry Theorem). Suppose that the assumptions of Theorems ${ \mathcal { B } } ,$ 5, and 8 hold.

Then there exists a constant $K > 0$ , depending only on the cardinality of the action space, such that

$$
\mathcal { C } _ { \mathrm { t o p } } ( \pi ^ { * } ) \leq K | \mathcal { A } | .\tag{87}
$$

Consequently,

$$
\mathcal { C } _ { \mathrm { t o p } } ( \pi ^ { * } ) = O ( | \mathcal { A } | ) ,\tag{88}
$$

independently of

$$
d = \dim ( S ) .\tag{89}
$$

Proof. By Corollary 10,

$$
C _ { R } ( \pi ^ { * } ) = | { \mathcal { A } } | .\tag{90}
$$

Furthermore, Theorem 15 implies

$$
C _ { B } ( \pi ^ { * } ) = O ( | { \cal A } | ) .\tag{91}
$$

Finally, Corollary 11 yields

$$
F ( \pi ^ { * } ) = 1 .\tag{92}
$$

Combining (90), (91), and (92) with (86) gives

$$
{ \mathcal { C } } _ { \mathrm { t o p } } ( \pi ^ { * } ) = O ( | A | ) .\tag{93}
$$

Since none of the preceding quantities depends on the ambient dimension $d ,$ the resulting bound is dimension-free.

Theorem 19 shows that, under classical structural assumptions, the complexity of an optimal policy is governed by the organization of the action space rather than by the dimensionality of the state space.

This result provides a first indication that policy geometries may admit substantially more compact representations than value-function descriptions in high-dimensional environments. The implications of this phenomenon are investigated in the following subsections.

## 5.4 Geometric Stability

The previous subsections established that optimal policies admit a compact geometric representation and that all local decision changes are concentrated on the boundary structure. A natural question is whether this geometric organization remains stable under perturbations of the underlying optimization problem. To address this issue, we introduce a quantitative measure of local decision robustness.

Definition 24 (Action $\operatorname { G a p } )$ . Let $\pi ^ { * }$ denote an optimal policy and let $a ^ { * } ( s ) = \pi ^ { * } ( s )$ be the optimal action at state $s \in { \mathcal { S } }$

The action gap at state s is defined by

$$
g ( s ) = Q ^ { \ast } ( s , a ^ { \ast } ( s ) ) - \operatorname* { m a x } _ { a \in \mathcal { A } \backslash \{ a ^ { \ast } ( s ) \} } Q ^ { \ast } ( s , a ) .\tag{94}
$$

The action gap measures the separation between the optimal action and its closest competitor. By construction,

$$
g ( s ) \geq 0 , \quad s \in S .\tag{95}
$$

Moreover, $g ( s ) = 0$ whenever at least two actions attain the same optimal value.

The following result links the action gap to the boundary structure introduced in Section 3.

Proposition 20. Assume that the functions $Q ^ { * } ( \cdot , a )$ are continuous on S for every $a \in { \mathcal { A } }$ Then

$$
\Gamma ( \pi ^ { * } ) \subseteq \{ s \in { \mathcal { S } } : g ( s ) = 0 \} .\tag{96}
$$

Proof. Let $s \in \Gamma ( \pi ^ { * } )$ . By Theorem 18, every neighborhood of s contains states assigned to at least two distinct actions. Since the state-action value functions are continuous, at least two competing actions must attain the same value at s.

Therefore, $g ( s ) = 0$

We now consider perturbations of the optimal state–action value function.

Assumption 8 (Uniform Perturbation). Let

$$
\widetilde { Q } ( s , a ) = Q ^ { * } ( s , a ) + \Delta ( s , a ) ,\tag{97}
$$

where

$$
\| \Delta \| _ { \infty } = \operatorname* { s u p } _ { s \in S } \operatorname* { s u p } _ { a \in \mathcal { A } } | \Delta ( s , a ) | \leq \varepsilon .\tag{98}
$$

Let $\widetilde { \pi }$ denote an optimal policy associated with ${ \widetilde { Q } } .$

The next theorem shows that policy changes can only occur at states possessing a suficiently small action gap.

Theorem 21 (Boundary Stability Theorem). Under Assumption $\delta , \ : \widetilde { \pi } ( s ) = \pi ^ { \ast } ( s )$ for every state satisfying

$$
g ( s ) > 2 \varepsilon .\tag{99}
$$

Consequently,

$$
\left\{ s \in S : \widetilde { \pi } ( s ) \neq \pi ^ { * } ( s ) \right\} \subseteq \left\{ s \in S : g ( s ) \leq 2 \varepsilon \right\} .\tag{100}
$$

Proof. Fix $s \in { \mathcal { S } }$ and let $a ^ { * } = \pi ^ { * } ( s )$ . For every competing action $a \neq a ^ { * }$

$$
Q ^ { * } ( s , a ^ { * } ) - Q ^ { * } ( s , a ) \geq g ( s ) .\tag{101}
$$

Using Assumption 8,

$$
\begin{array} { r l } & { \widetilde { Q } ( s , a ^ { * } ) - \widetilde { Q } ( s , a ) = Q ^ { * } ( s , a ^ { * } ) - Q ^ { * } ( s , a ) + \Delta ( s , a ^ { * } ) - \Delta ( s , a ) } \\ & { \qquad \geq g ( s ) - 2 \varepsilon . } \end{array}
$$

Therefore, if $g ( s ) > 2 \varepsilon$ 2

$$
\widetilde Q ( s , a ^ { * } ) > \widetilde Q ( s , a ) \quad \forall a \neq a ^ { * } .
$$

Hence $a ^ { * }$ remains optimal and

$$
{ \widetilde { \pi } } ( s ) = \pi ^ { * } ( s ) .
$$

This proves the claim.

Theorem 21 establishes a geometric robustness property of structured optimal policies. States located deep inside an action region possess a strictly positive decision margin and therefore remain unafected by suficiently small perturbations. Only states with small action gaps may experience a change in the optimal action. Since the action gap vanishes on the boundary structure, policy modifications are necessarily concentrated near the interfaces separating neighboring action regions.

Combined with Theorems 18 and 19, this result shows that the geometric complexity of an optimal policy is not only localized but also stable under small perturbations of the underlying dynamic program.

## 5.5 Structural Compression Theorem

The preceding subsections established four fundamental properties of policy geometries.

First, Proposition 17 showed that the geometry induced by a deterministic policy provides a faithful representation of the policy itself. Second, Theorem 18 established that all local decision changes are concentrated on the boundary structure. Third, Theorem 19 demonstrated that the combinatorial complexity of the geometry remains independent of the ambient state-space dimension. Finally, Theorem 21 showed that the geometric representation is stable under suficiently small perturbations of the underlying optimization problem.

Taken together, these properties suggest that policy geometries provide a compressed description of optimal decision rules. The purpose of this subsection is to formalize this observation.

Recall that the decision compression ratio introduced in Section 4 is defined by

$$
\mathrm { D C R } = { \frac { { \mathcal { C } } _ { V } } { { \mathcal { C } } ( { \mathcal { G } } ^ { * } ) } } ,\tag{102}
$$

where $\mathcal { C } _ { V }$ denotes the complexity of the value-function representation and ${ \mathcal { C } } ( { \mathcal { G } } ^ { * } )$ denotes the complexity of the optimal policy geometry.

Theorem 16 established the general lower bound

$$
\mathrm { D C R } = \Omega \bigg ( \frac { { \mathcal C } _ { V } } { | A | } \bigg ) .\tag{103}
$$

The next result characterizes the asymptotic implications of this bound for structured dynamic programs.

Theorem 22 (Structural Compression Theorem). Suppose that the assumptions of Theorems 19 and 21 hold.

Assume furthermore that

$$
{ \mathcal { C } } _ { V } = \Theta ( n ) ,\tag{104}
$$

for some problem-size parameter n, and that

$$
| { \cal A } | = { \cal O } ( 1 ) .\tag{105}
$$

Then

$$
\mathrm { D C R } = \Omega ( n ) .\tag{106}
$$

Proof. By Theorem 19,

$$
{ \mathcal { C } } ( { \mathcal { G } } ^ { * } ) = O ( | { \mathcal { A } } | ) .\tag{107}
$$

Combining (107) with (103) yields

$$
\mathrm { D C R } = \Omega \bigg ( \frac { { \mathcal C } _ { V } } { | A | } \bigg ) .\tag{108}
$$

Substituting (104) into (108) gives

$$
\mathrm { D C R } = \Omega \left( { \frac { n } { | { \cal A } | } } \right) .\tag{109}
$$

Using (105), we obtain

$$
\mathrm { D C R } = \Omega ( n ) ,\tag{110}
$$

which establishes the result.

Theorem 22 shows that the compression achieved by policy geometry is an asymptotic phenomenon rather than a finite-dimensional artifact. As the intrinsic complexity of the value-function representation grows, the complexity of the geometric representation remains controlled by the action structure of the problem. Consequently, the gap between value complexity and geometric complexity increases at least linearly with problem size. Combined with the localization and stability properties established in the previous subsections, this result suggests that policy geometry captures the essential decision structure of a dynamic program using substantially fewer degrees of freedom than conventional value-based representations.

## 5.6 Geometry and Learnability

The results established in the preceding subsections suggest that the geometric representation of an optimal policy may provide a substantially simpler object to learn than the policy itself.

Indeed, Proposition 17 showed that the policy geometry contains the complete information required to reconstruct a deterministic policy, while Theorem 18 established that all decision transitions are concentrated on the boundary structure.

This observation motivates a geometric formulation of the policy-learning problem.

Definition 25 (Boundary Learning Problem). Let

$$
\mathcal { G } ( \pi ^ { * } ) = \left( \mathcal { T } ( \pi ^ { * } ) , \Gamma ( \pi ^ { * } ) \right)\tag{111}
$$

denote the geometry induced by an optimal policy.

The boundary learning problem consists of recovering the boundary structure $\Gamma ( \pi ^ { * } )$ from observed state-action information.

The objective of the boundary learning problem is not to estimate the policy value function or to approximate the policy on the entire state space. Instead, the goal is to identify the geometric locus at which optimal decisions change.

To quantify the intrinsic dificulty of this task, we introduce the following notion.

Definition 26 (Boundary Sample Complexity). Let $\widehat { \Gamma }$ denote an estimator of the boundary structure $\Gamma ( \pi ^ { * } )$

For a prescribed accuracy criterion $\mathcal { E } ( \widehat { \Gamma } , \Gamma ( \pi ^ { * } ) )$ , the boundary sample complexity is defined as the smallest number of observations required to guarantee

$$
\mathcal { E } \big ( \widehat \Gamma , \Gamma ( \pi ^ { * } ) \big ) \leq \delta ,\tag{112}
$$

for a given tolerance level $\delta > 0$

The corresponding quantity is denoted by

$$
N _ { \Gamma } ( \delta ) .\tag{113}
$$

Definition 26 is intentionally model-independent. No particular statistical framework is assumed at this stage. The purpose of the definition is merely to isolate the learning dificulty associated with the boundary structure itself.

The following theorem formalizes the geometric reduction underlying the subsequent learning framework.

Theorem 23 (Learnability Through Boundaries). Suppose that the assumptions of Theorems 18, 19, and 21 hold.

Then the recovery of the optimal policy $\pi ^ { * }$ is equivalent to the recovery of the boundary structure $\Gamma ( \pi ^ { * } )$

More precisely, there exists a reconstruction operator

$$
\Re : \Gamma ( \pi ^ { * } ) \longmapsto \pi ^ { * }\tag{114}
$$

such that

$$
\pi ^ { * } = \Re ( \Gamma ( \pi ^ { * } ) ) .\tag{115}
$$

Consequently, the policy-learning problem admits the equivalent geometric formulation

$$
\pi ^ { * } \quad \Longleftrightarrow \quad \Gamma ( \pi ^ { * } ) .\tag{116}
$$

Proof. By Proposition 17, a deterministic policy is uniquely determined by its tessellation.

By Theorem 18, the boundary structure constitutes the complete support of policy variation.

Therefore, once the boundary structure is known, the corresponding tessellation is uniquely determined by the partition induced by the decision boundaries.

Since the tessellation uniquely determines the policy, there exists a reconstruction operator R satisfying (115). This establishes the equivalence (116).

Theorem 23 provides the fundamental connection between structural geometry and statistical learning. Rather than treating policy learning as the problem of approximating a decision rule over the entire state space, the theorem shows that learning may be reformulated as the problem of identifying the boundary structure of the optimal policy.

Combined with Theorem 19, this observation suggests that the intrinsic dificulty of learning structured optimal policies is governed by the geometric complexity of their decision boundaries rather than by the ambient dimension of the state space. This geometric viewpoint forms the basis of the boundary-learning and active-sampling methodologies developed in the subsequent sections.

## 5.7 Discussion

The results of this section establish a direct connection between the structural properties of dynamic programs and the geometric organization of their optimal policies.

Proposition 17 showed that an optimal policy may be represented without loss of information through its induced tessellation. Consequently, the analysis of optimal decision rules may be reformulated as the analysis of geometric objects defined on the state space.

Building upon this representation, Theorem 18 identified the boundary structure as the unique support of local policy variation. This result isolates the geometric locus at which decision changes occur and separates invariant decision regions from transition regions.

The subsequent analysis demonstrated that, under classical structural assumptions, policy geometries possess strong regularity properties. Theorems 19 and 21 showed that the complexity of the geometry remains controlled independently of the ambient state-space dimension and is stable under suficiently small perturbations of the underlying optimization problem. These properties lead naturally to the compression results established in Theorem 22. Since the geometry retains the complete decision content of the policy while exhibiting substantially lower complexity than value-function representations, structured dynamic programs admit intrinsically compressed policy descriptions.

Finally, Theorem 23 showed that the policy-learning problem may be reformulated as a boundaryidentification problem. As a consequence, the complexity of learning is governed by the geometry of decision boundaries rather than by the size of the state space itself.

Taken together, the results of this section establish the following chain of implications:

$$
\begin{array} { r }  \mathrm { { S t r u c t u r a l ~ A s s u m p t i o n s ~ } \Longrightarrow { P o l i c y ~ G e o m e t r y } \implies \mathrm { { B o u n d a r y ~ S t r u c t u r e } } } \\ { \implies \mathrm { { S t r u c t u r a l ~ C o m p r e s s i o n } ~ \Longrightarrow ~ \mathrm { { L e a r n a b i l i t y } } . } } \end{array}\tag{117}
$$

This perspective suggests that the fundamental object underlying many structured decision problems is not the value function itself, but rather the geometry induced by the optimal policy. The next sections exploit this observation to develop learning and approximation methodologies that operate directly on policy geometries.

## 6 Geometric Complexity and Decision Compression

## 6.1 Complexity Measures Revisited

The preceding sections introduced two complementary notions of complexity associated with a dynamic decision problem.

The first quantity is the value-representation complexity ${ \mathcal { C } } _ { V }$ , defined in Definition 19, which measures the complexity of representing the optimal value function.

The second quantity is the decision complexity

$$
\mathcal { C } _ { D } = \mathcal { C } ( \mathcal { G } ^ { * } ) ,\tag{118}
$$

introduced in Definition 20, which measures the complexity of the optimal policy geometry.   
These quantities describe fundamentally diferent objects.

The quantity $\mathcal { C } _ { V }$ characterizes the complexity of the numerical solution of the dynamic program through the representation of the optimal value function. In contrast, $\mathcal { C } _ { D }$ characterizes the complexity of the optimal decision rule itself through the geometric organization of its action regions and decision boundaries. The distinction between these two notions is central to the analysis developed in this section. Throughout the remainder of the paper, value complexity and decision complexity are treated as conceptually distinct quantities and are analyzed separately.

The structural results established in Sections 4 and 5 imply that the decision complexity of structured dynamic programs remains controlled by the action structure of the problem.

More precisely, Theorem 19 established that

$$
\mathcal { C } _ { \mathrm { t o p } } ( \pi ^ { * } ) = O ( | \mathcal { A } | ) ,\tag{119}
$$

while Theorem 22 showed that the resulting compression ratio satisfies

$$
\mathrm { D C R } = \Omega ( n ) ,\tag{120}
$$

whenever ${ \mathcal { C } } _ { V } = \Theta ( n )$ and $| \mathcal { A } | = O ( 1 )$

The purpose of the present section is not to establish additional geometric regularity properties. Rather, it is to investigate the quantitative implications of these complexity relationships and to characterize the compression mechanisms induced by policy geometry.

Accordingly, all subsequent results will be expressed in terms of the pair $\left( \mathcal { C } _ { V } , \mathcal { C } _ { D } \right)$ , which provides a unified framework for comparing value-based and geometry-based representations of optimal decision rules.

## 6.2 Compression Regimes

The discussion of Section 6.1 establishes that value complexity and decision complexity may exhibit fundamentally diferent scaling behaviors. To study this phenomenon systematically, we introduce a classification of compression regimes based on the asymptotic behavior of the decision compression ratio. Throughout this subsection, consider a sequence of decision problems $\{ \mathcal { M } _ { n } \} _ { n \ge 1 }$ , indexed by a complexity parameter n.

The parameter n may represent, depending on the application, the number of states, the dimension of the state space, the discretization level, or any other quantity governing the growth of the decision problem.

For each problem $\mathcal { M } _ { n } .$ , let ${ \mathcal { C } } _ { V } ( n )$ denote the corresponding value complexity and let $\mathcal { C } _ { D } ( n )$ denote the associated decision complexity. The decision compression ratio is therefore given by

$$
\mathrm { D C R } ( n ) = { \frac { { \mathcal { C } } _ { V } ( n ) } { { \mathcal { C } } _ { D } ( n ) } } .\tag{121}
$$

The asymptotic growth of DCR(n) quantifies the extent to which policy geometries provide more compact representations than value functions.

The following definitions distinguish several compression regimes.

Definition 27 (Weak Compression). The family $\{ \mathcal { M } _ { n } \} _ { n \ge 1 }$ is said to exhibit weak compression if

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname { i n f } _ { \mathrm { \tiny ~ D C R } ( n ) } > 1 .\tag{122}
$$

Weak compression corresponds to a regime in which geometric representations remain asymptotically more compact than value-based representations, although the compression gain remains uniformly bounded.

Definition 28 (Polynomial Compression). The family $\{ \mathcal { M } _ { n } \} _ { n \ge 1 }$ is said to exhibit polynomial compression if there exist constants $c > 0$ and α $> 0$ such that

$$
\mathrm { D C R } ( n ) \geq c n ^ { \alpha }\tag{123}
$$

for all suficiently large n.

Polynomial compression characterizes situations in which the gap between value complexity and decision complexity grows polynomially with problem size.

Definition 29 (Strong Compression). The family $\{ { \mathcal { M } } _ { n } \} _ { n \geq 1 }$ is said to exhibit strong compression $i f$

$$
\mathrm { D C R } ( n ) = \Omega ( n ) .\tag{124}
$$

Strong compression corresponds to a regime in which decision complexity grows at least one asymptotic order more slowly than value complexity.

The three notions introduced above form a hierarchy:

$$
{ \mathrm { S t r o n g ~ C o m p r e s s i o n } } \Longrightarrow { \mathrm { P o l y n o m i a l ~ C o m p r e s s i o n } } \Longrightarrow { \mathrm { W e a k ~ C o m p r e s s i o n } } .\tag{125}
$$

This classification provides a unified language for describing compression phenomena independently of any particular representation architecture. In the subsequent subsections, we investigate the structural conditions under which policy geometries achieve polynomial or strong compression.

## 6.3 Scaling Laws

The compression regimes introduced in Section 6.2 provide a qualitative classification of asymptotic compression phenomena. We now establish quantitative scaling laws describing how the decision compression ratio evolves with problem size. The key observation is that, under the structural assumptions developed in Sections 4 and 5, the growth of decision complexity remains controlled by the action structure of the problem, whereas value complexity may increase substantially with the size of the state space.

The following theorem formalizes this relationship.

Theorem 24 (Scaling Law for Decision Compression). Consider a family of decision problems $\{ \mathcal { M } _ { n } \} _ { n \ge 1 }$ satisfying the assumptions of Theorem 15.

Assume that the corresponding value complexity satisfies

$$
\begin{array} { l l } { { { \mathcal { C } } _ { V } ( n ) = \Theta \left( n ^ { \beta } \right) , \qquad \beta > 0 . } } \end{array}\tag{126}
$$

Then the decision compression ratio satisfies

$$
\mathrm { D C R } ( n ) = \Omega \left( \frac { n ^ { \beta } } { | \mathcal { A } | } \right) .\tag{127}
$$

Proof. By Definition 20,

$$
\begin{array} { r } { \mathcal { C } _ { D } ( n ) = \mathcal { C } ( \mathcal { G } _ { n } ^ { * } ) , } \end{array}\tag{128}
$$

where $\mathcal { G } _ { n } ^ { * }$ denotes the optimal policy geometry associated with $\mathcal { M } _ { n }$

Theorem 15 implies the existence of a constant $K > 0$ , independent of $n ,$ such that

$$
{ \mathcal { C } } _ { D } ( n ) \leq K | { \mathcal { A } } |\tag{129}
$$

for all suficiently large n.

Furthermore, assumption (126) implies the existence of constants $c _ { 1 } , c _ { 2 } > 0$ such that

$$
c _ { 1 } n ^ { \beta } \leq \mathcal { C } _ { V } ( n ) \leq c _ { 2 } n ^ { \beta }\tag{130}
$$

for all suficiently large n.

Using Definition 21,

$$
\mathrm { D C R } ( n ) = { \frac { { \mathcal { C } } _ { V } ( n ) } { { \mathcal { C } } _ { D } ( n ) } } .\tag{131}
$$

Combining (129), (130), and (131), we obtain

$$
\mathrm { D C R } ( n ) \geq { \frac { c _ { 1 } n ^ { \beta } } { K | { \cal A } | } } .\tag{132}
$$

Therefore,

$$
\mathrm { D C R } ( n ) = \Omega \left( \frac { n ^ { \beta } } { | \mathcal { A } | } \right) ,\tag{133}
$$

which establishes the result.

Theorem 24 shows that the asymptotic compression achieved by policy geometries is determined by the growth rate of value complexity rather than by the dimension of the state space itself.

In particular, whenever the action space remains fixed or grows more slowly than ${ \mathcal { C } } _ { V } ( n )$ , the compression ratio diverges with problem size.

Corollary 25 (Linear Compression Regime). Suppose that

$$
\mathcal { C } _ { V } ( n ) = \Theta ( n )\tag{134}
$$

and

$$
| { \cal A } | = { \cal O } ( 1 ) .\tag{135}
$$

Then

$$
\mathrm { D C R } ( n ) = \Omega ( n ) .\tag{136}
$$

Proof. The result follows immediately from Theorem 24 with $\beta = 1$

Corollary 25 recovers, as a particular case, the linear compression phenomenon established previously in Theorem 22.

## 6.4 Exponential Compression

The scaling law established in Theorem 24 provides a general relationship between value complexity and decision compression. An important regime arises when the complexity of representing the optimal value function grows exponentially with the dimension of the state space. Such behavior is commonly associated with the curse of dimensionality in dynamic programming (Bellman, 1957; Bertsekas, 2012; Powell, 2007).

The following corollary characterizes the implications of this phenomenon for policy geometries.

Corollary 26 (Exponential Compression). Consider a family of decision problems $\{ \mathcal { M } _ { d } \} _ { d \ge 1 }$ indexed by the dimension d of the state space. Suppose that the value complexity satisfies

$$
{ \mathcal { C } } _ { V } ( d ) = \Theta { \Big ( } 2 ^ { d } { \Big ) } .\tag{137}
$$

Then the decision compression ratio satisfies

$$
\mathrm { D C R } ( d ) = \Omega \left( \frac { 2 ^ { d } } { | \mathcal { A } | } \right) .\tag{138}
$$

Proof. Applying Theorem 24 with $\mathcal { C } _ { V } ( d ) = \Theta ( 2 ^ { d } )$ , yields

$$
\operatorname { D C R } ( d ) = \Omega \left( { \frac { { \mathcal { C } } _ { V } ( d ) } { | A | } } \right) = \Omega \left( { \frac { 2 ^ { d } } { | A | } } \right) .
$$

This proves the result.

Corollary 26 shows that exponential growth of value complexity induces an exponential separation between value-based and geometry-based representations of optimal decision rules. This phenomenon is a direct consequence of the structural properties established in Sections 4 and 5. Indeed, Theorem 19 implies that the complexity of the optimal policy geometry remains controlled by the action structure of the problem and does not scale with the dimension of the state space.

Consequently, whenever value representations exhibit exponential growth with dimension, the compression advantage associated with policy geometries increases exponentially as well. The result should therefore be interpreted as a structural separation theorem: although the representation of optimal value functions may become exponentially complex, the geometric representation of optimal policies remains governed by the complexity of the action partition rather than by the ambient dimension.

## 6.5 Boundary Compression

The preceding results suggest that the geometric complexity of a policy is not distributed uniformly across the state space. Indeed, Theorem 18 established that all local decision changes are concentrated on the boundary structure

$$
\Gamma = \bigcup _ { \stackrel { a , a ^ { \prime } \in \mathcal { A } } { a \neq a ^ { \prime } } } \Gamma _ { a a ^ { \prime } } .
$$

This localization property naturally raises the question of whether the complexity of the entire policy geometry is asymptotically determined by the complexity of its boundary structure.

The following theorem answers this question in the afirmative.

Theorem 27 (Boundary Compression Theorem). Consider a family of structured decision problems $\{ \mathcal { M } _ { n } \} _ { n \ge 1 }$ satisfying the assumptions of Theorem 15.

Let $C _ { B } ( \boldsymbol { n } )$ denote the corresponding boundary complexity and let $\mathcal { C } _ { D } ( n ) = \mathcal { C } ( \mathcal { G } _ { n } ^ { \ast } )$ denote the associated decision complexity.

Then

$$
\mathcal { C } _ { D } ( n ) = \Theta ( C _ { B } ( n ) ) .\tag{139}
$$

Proof. By Definition 20, the decision complexity is determined by the geometric characteristics of the optimal policy geometry.

In particular,

$$
\begin{array} { r } { \mathcal { C } _ { D } ( n ) = \Phi ( C _ { B } ( n ) , C _ { R } ( n ) , F ( n ) , \kappa ( n ) ) , } \end{array}\tag{140}
$$

for some complexity functional Φ.

Theorem 15 implies that the region complexity, fragmentation index, and geometric irregularity remain uniformly bounded. Consequently, there exists a constant $K > 0$ such that

$$
C _ { R } ( n ) + F ( n ) + \kappa ( n ) \leq K\tag{141}
$$

for all suficiently large n.

Therefore, the only geometric quantity capable of exhibiting asymptotic growth is the boundary complexity $C _ { B } ( \boldsymbol { n } )$ . It follows that there exist constants $c _ { 1 } , c _ { 2 } > 0$ such that

$$
c _ { 1 } C _ { B } ( n ) \leq C _ { D } ( n ) \leq c _ { 2 } C _ { B } ( n )\tag{142}
$$

for all suficiently large n.

Hence, $\mathcal { C } _ { D } ( n ) = \Theta ( C _ { B } ( n ) )$ , which establishes the result.

Theorem 27 shows that the asymptotic complexity of a structured policy geometry is entirely governed by its boundary structure.

Combined with Theorem 18, the result yields a particularly simple interpretation of policy complexity. The interiors of action regions correspond to locally invariant decision zones and therefore contribute only bounded complexity. In contrast, all asymptotically relevant geometric information is concentrated on the set of decision boundaries.

Consequently, structured policy geometries admit a compressed representation in which boundary complexity becomes the fundamental quantity governing decision complexity.

Corollary 28 (Boundary-Based Compression Ratio). Under the assumptions of Theorem 27,

$$
\mathrm { D C R } ( n ) = \Theta \biggl ( \frac { { \mathcal C } _ { V } ( n ) } { { \cal C } _ { B } ( n ) } \biggr ) .\tag{143}
$$

Proof. Combining $\begin{array} { r } { \mathrm { D C R } ( n ) = \frac { \mathcal C _ { V } ( n ) } { \mathcal C _ { D } ( n ) } } \end{array}$ with $\mathcal { C } _ { D } ( n ) = \Theta ( C _ { B } ( n ) )$ immediately yields the result.

## 6.6 Information-Theoretic Compression

The geometric compression results established in the preceding subsections admit a natural informationtheoretic interpretation. Rather than measuring complexity through geometric characteristics alone, one may ask how many bits are required to describe an optimal policy or, alternatively, its associated boundary structure. This perspective is closely related to the Minimum Description Length (MDL) principle introduced by Rissanen (1978) and further developed by Grünwald (2007). Throughout this subsection, all description lengths are defined relative to a fixed admissible family of prefix codes.

Definition 30 (Policy Description Length). Let Π denote a class of admissible deterministic policies.

The policy description length of $\pi ^ { * } \in \Pi$ , denoted by $L _ { \Pi } ( \pi ^ { * } )$ , is the minimum number of bits required to encode $\pi ^ { * }$ within the coding class under consideration.

Definition 31 (Boundary Description Length). Let $\Gamma ^ { * } = \Gamma ( \pi ^ { * } )$ denote the boundary structure induced by the optimal policy.

The boundary description length is defined by $L _ { \Gamma } ( \Gamma ^ { * } )$ , the minimum number of bits required to encode Γ<sup>∗</sup> within the same coding class.

The following result establishes that, under the structural assumptions developed throughout the paper, the informational complexity of an optimal policy is asymptotically equivalent to that of its boundary structure.

Theorem 29 (Information-Theoretic Compression Theorem). Consider a family of structured decision problems $\{ { \mathcal { M } } _ { n } \} _ { n \geq 1 }$ satisfying the assumptions of Theorem 27.

Then there exist constants $K _ { 1 } , K _ { 2 } > 0$ such that

$$
L _ { \Gamma } ( \Gamma _ { n } ^ { * } ) - K _ { 1 } \leq L _ { \Pi } ( \pi _ { n } ^ { * } ) \leq L _ { \Gamma } ( \Gamma _ { n } ^ { * } ) + K _ { 2 }\tag{144}
$$

for all suficiently large n.

Consequently,

$$
L _ { \Pi } ( \pi _ { n } ^ { * } ) = \Theta ( L _ { \Gamma } ( \Gamma _ { n } ^ { * } ) ) .\tag{145}
$$

Proof. By Theorem 18, the optimal policy is locally constant on each connected component of $\boldsymbol { \mathcal { S } } \boldsymbol { \backslash } \Gamma _ { n } ^ { * }$ Hence, once the boundary structure $\Gamma _ { n } ^ { * }$ is specified, reconstructing the policy requires only the assignment of action labels to the resulting regions. Since the action set A is finite, the number of bits required to encode these labels is bounded independently of n. Therefore, there exists a constant $K _ { 2 } > 0$ such that

$$
\begin{array} { r } { L _ { \Pi } ( \pi _ { n } ^ { * } ) \le L _ { \Gamma } ( \Gamma _ { n } ^ { * } ) + K _ { 2 } . } \end{array}\tag{146}
$$

Conversely, the boundary structure is uniquely determined by the policy through the induced tessellation (Proposition 17). Hence there exists a constant $K _ { 1 } > 0$ such that

$$
\begin{array} { r } { L _ { \Gamma } ( \Gamma _ { n } ^ { * } ) \leq L _ { \Pi } ( \pi _ { n } ^ { * } ) + K _ { 1 } . } \end{array}\tag{147}
$$

Combining (146) and (147) yields (144). The asymptotic relation (145) follows immediately.

Theorem 29 provides an information-theoretic counterpart to Theorem 27.

The result shows that the informational content of an optimal policy is asymptotically equivalent to the informational content of its boundary structure. In particular, the interiors of action regions contribute only a bounded amount of additional information once the boundaries have been specified.

Consequently, structured policy geometries achieve compression not only in a geometric sense but also in a description-length sense. The dominant informational object is the boundary structure itself. Corollary 30 (Boundary Information Dominance). Under the assumptions of Theorem 29,

$$
L _ { \Pi } ( \pi _ { n } ^ { * } ) = \Theta ( L _ { \Gamma } ( \Gamma _ { n } ^ { * } ) ) = \Theta ( C _ { B } ( n ) ) ,\tag{148}
$$

whenever the boundary description length is proportional to boundary complexity.

Proof. The first equivalence follows from Theorem 29.

The second follows from the proportionality assumption and Theorem 27.

## 6.7 Limits of Compression

The compression results established throughout this section rely fundamentally on the structural assumptions developed in Sections 4 and 5. In particular, the scaling laws, boundary-compression results, and information-theoretic compression properties all depend on the existence of regular policy geometries characterized by monotonicity, threshold representations, bounded fragmentation, and controlled boundary complexity.

The purpose of the present subsection is to clarify the limits of this mechanism and to identify situations in which the compression advantage may deteriorate or disappear. The key observation is that decision compression is ultimately a consequence of geometric regularity. Whenever this regularity is lost, the complexity of the policy geometry may increase substantially and become comparable to the complexity of the underlying value representation.

The following proposition formalizes this observation.

Proposition 31 (Limits of Decision Compression). Consider a family of decision problems $\{ { \mathcal { M } } _ { n } \} _ { n \geq 1 }$ Suppose that at least one of the following conditions holds:

(i) the fragmentation index is unbounded, $F ( n ) \to \infty ,$

(ii) the convex-region property of Theorem 8 fails, allowing optimal action regions to develop arbitrarily complex nonconvex geometries;

(iii) the monotonicity assumptions of Section 4.1 are violated, so that the threshold representations established in Section 4.3 need not exist.

Then the geometric complexity of the optimal policy is no longer uniformly controlled by the action structure of the problem.

Moreover, there exist families of decision problems for which

$$
\mathcal { C } _ { D } ( n ) = \Theta \big ( \mathcal { C } _ { V } ( n ) \big ) .\tag{149}
$$

Proof. The compression results established in Sections 4 and 5 depend critically on the existence of a geometrically simple policy representation. Under monotonicity, convexity, and threshold structures, the complexity of the policy geometry remains controlled by a bounded number of action regions and decision boundaries. This property yields the compression phenomena described in Theorems 24, 27, and 29.

If fragmentation becomes unbounded, however, the number of connected components required to represent the policy geometry may grow proportionally to the complexity of the value representation itself. Similarly, if convexity is lost, action regions may develop arbitrarily intricate geometric structures whose description requires a number of parameters comparable to that required for representing the value function. Finally, when monotonicity fails, threshold representations need not exist and the boundary structure may become arbitrarily irregular, eliminating the geometric simplicity established previously.

Consequently, the complexity of the policy geometry may scale at the same asymptotic rate as the complexity of the value representation, yielding (149).

Proposition 31 shows that decision compression is not a universal property of dynamic programs. Rather, it is a consequence of the structural regularities induced by monotonicity, convexity, and bounded fragmentation.

From a geometric perspective, compression emerges because optimal policies can be represented through a relatively small collection of regular action regions separated by simple decision boundaries. When these regularity properties disappear, the policy geometry itself may become highly complex, thereby eliminating the compression advantage.

The results of this section should therefore be interpreted as identifying a broad class of structured dynamic programs for which geometric representations provide compressed descriptions of optimal decision rules, rather than as a universal statement applying to arbitrary Markov decision processes.

## 6.8 Discussion

The results developed throughout this section establish a coherent relationship between structural regularity, policy geometry, and decision complexity.

The starting point is the structural framework introduced in Section 4. Monotonicity, threshold representations, convex action regions, and bounded fragmentation imply that optimal policies admit geometrically regular representations. These structural properties induce policy geometries whose complexity remains controlled by the action structure of the problem rather than by the size of the state space. Building upon this geometric characterization, Sections 6.3-6.6 demonstrate that policy geometries constitute compressed representations of optimal decision rules. The scaling laws show that compression increases with value complexity, the boundary-compression results identify decision boundaries as the dominant source of geometric complexity, and the information-theoretic analysis establishes an equivalent conclusion from a description-length perspective.

Taken together, these results yield the following conceptual chain:

$$
{ \mathrm { S t r u c t u r e } } \Longrightarrow { \mathrm { S i m p l e ~ G e o m e t r y } } \Longrightarrow { \mathrm { C o m p r e s s i o n } } .\tag{150}
$$

More specifically, Theorems 18 and 27 show that the informational and geometric content of a structured policy is concentrated near its boundary structure. Consequently, compression emerges because optimal policies can be represented through a comparatively small collection of regular regions separated by simple decision boundaries. At the same time, Proposition 31 clarifies that this phenomenon is not universal. Compression is a consequence of structural regularity and may deteriorate when monotonicity, convexity, or bounded fragmentation are lost. The theory therefore identifies a broad class of structured dynamic programs for which geometric representations are substantially more compact than value-based representations.

The practical significance of these results lies in their implications for scalability. Whenever decision complexity grows substantially more slowly than value complexity, geometric representations provide a mechanism for mitigating the representational burden associated with large-scale dynamic programs.

$$
\mathrm { C o m p r e s s i o n } \Longrightarrow \mathrm { S c a l a b i l i t y } .\tag{151}
$$

This observation provides the conceptual foundation for the learning, approximation, and algorithmic developments that follow.

## 7 Learning Policy Geometry

## 7.1 Geometric Learning Problem

Sections 3, 4, 5, and 6 established that optimal policies arising from structured dynamic programs admit geometrically regular representations. More precisely, Proposition 17 showed that a deterministic policy is uniquely characterized by its induced tessellation, while Theorem 18 identified the boundary structure as the unique locus at which decision changes occur. Furthermore, Theorem 23 established that the optimal policy can be reconstructed from its boundary geometry.

Consequently, the learning problem considered in this paper difers fundamentally from classical value-function or policy-learning formulations. Rather than estimating the optimal value function $V ^ { * }$ , the optimal state-action value function $Q ^ { * }$ , or the optimal policy $\pi ^ { * }$ directly, the objective is to estimate the geometric object that carries the decision information.

We begin by formalizing the object of interest.

Definition 32 (Target Boundary Geometry). Let $\Gamma ^ { * } = \Gamma ( \pi ^ { * } )$ denote the boundary structure induced by the optimal policy. The set

$$
\Gamma ^ { * } = \bigcup _ { a , a ^ { \prime } \in \mathcal { A } \atop a \ne a ^ { \prime } } \Gamma _ { a a ^ { \prime } }\tag{152}
$$

is called the target boundary geometry.

The target boundary geometry constitutes the primary object of statistical inference throughout this section. The available information is represented abstractly through a collection of observations

$$
\mathcal { D } _ { n } = \{ Z _ { 1 } , . . . , Z _ { n } \} ,\tag{153}
$$

where each observation $Z _ { i }$ may contain information about the optimal decision structure.

No specific sampling mechanism is imposed at this stage. The observations may arise from simulation, state queries, optimal-action evaluations, trajectory data, or any other information source capable of providing evidence regarding the underlying policy geometry.

The goal is to construct an estimator

$$
\widehat { \Gamma } _ { n } = \widehat { \Gamma } _ { n } ( \mathcal { D } _ { n } )\tag{154}
$$

of the unknown boundary structure $\Gamma ^ { * }$

Definition 33 (Geometric Learning Problem). Given observations $\mathcal { D } _ { n }$ , the geometric learning problem consists of constructing an estimator $\widehat { \Gamma } _ { n }$ such that

$$
\widehat { \Gamma } _ { n } \longrightarrow \Gamma ^ { * }\tag{155}
$$

under an appropriate notion of geometric convergence.

The formulation above deliberately separates the learning target from the learning mechanism. The object to be estimated is the boundary geometry itself, whereas the statistical model governing the observations will be introduced subsequently. The justification for this viewpoint follows directly from the structural results established previously. By Theorem 23, there exists a reconstruction operator

$$
\Re : \Gamma ^ { * } \longmapsto \pi ^ { * } ,\tag{156}
$$

such that the optimal policy can be recovered from its boundary structure.

Consequently, learning the optimal policy is equivalent to learning the target boundary geometry in the sense that

$$
\Gamma ^ { * } \quad \Longleftrightarrow \quad \pi ^ { * } .\tag{157}
$$

This equivalence transforms the original decision-learning problem into a geometric estimation problem. The remainder of this section investigates the statistical consequences of this reformulation and establishes conditions under which the boundary geometry can be learned eficiently.

## 7.2 Boundary Learning Principle

The geometric learning problem formulated in Section 7.1 identifies the target boundary geometry $\Gamma ^ { * }$ as the primary object of statistical inference. The purpose of the present subsection is to establish a structural principle that motivates this choice and guides the learning methodology developed in the remainder of the paper. The key observation is that the geometric representation of a structured policy exhibits a strong localization property. By Theorem 18, the optimal policy is locally constant on every connected component of

$$
{ \cal S } \backslash \Gamma ^ { * } .\tag{158}
$$

Consequently, decision changes may occur only on the boundary structure itself.

This property implies that observations collected far from the boundary geometry carry little information regarding the location of decision transitions. Indeed, whenever a state belongs to the interior of an action region, all suficiently small perturbations of that state produce identical optimal decisions.

In contrast, observations located near the boundary geometry contain information about the transition between competing actions and therefore provide information regarding the structure of the optimal policy.

The following principle summarizes this observation.

Principle 1 (Boundary Learning Principle). For structured dynamic programs satisfying the assumptions of Sections 4 and 5, the statistically informative component of the policy geometry is concentrated near the target boundary geometry Γ<sup>∗</sup>.

Consequently, eficient learning of the optimal policy may be reduced to the estimation of the boundary structure rather than the estimation of the entire policy over the state space.

The principle above follows directly from the combination of two structural results established previously.

First, Theorem 18 shows that policy variation is confined to the boundary geometry. Second, Theorem 27 establishes that the complexity of a structured policy is asymptotically governed by the complexity of its boundary structure. Together, these results imply that the dominant source of statistical uncertainty lies in the estimation of Γ<sup>∗</sup>.

From a learning perspective, the role of the boundary geometry is therefore analogous to that of a low-dimensional suficient representation of the decision rule. The objective is not to recover the optimal action at every state individually, but rather to identify the geometric interfaces separating the action regions. This viewpoint provides the conceptual foundation for the sample-complexity, active-sampling, and reconstruction results developed in the subsequent subsections.

## 7.3 Statistical Learning Model

Sections 7.1 and 7.2 identify the target boundary geometry $\Gamma ^ { * }$ as the primary object of statistical inference.

The purpose of the present subsection is to introduce the probabilistic framework within which boundary estimation will be analyzed. No convergence rates or sample-complexity guarantees are established at this stage. Rather, the objective is to define the statistical objects and loss functions that will be used throughout the remainder of the section.

Observation Model. Let $\mathcal { D } _ { n } = \{ Z _ { 1 } , . . . , Z _ { n } \}$ denote a collection of observations generated according to an unknown probability measure P defined on a measurable space $( { \mathcal { Z } } , B )$

The framework deliberately remains agnostic regarding the sampling mechanism. The observations may arise from simulation, state-action evaluations, trajectory data, policy queries, or any other information source capable of providing information about the underlying policy geometry.

Geometric Hypothesis Space. Let G denote a family of admissible boundary geometries.

The target boundary geometry $\Gamma ^ { * } \in \mathfrak { S }$ is assumed to belong to this family.

The specification of G is intentionally left abstract. Subsequent complexity bounds will depend on structural characteristics of this class rather than on the ambient state space itself.

Boundary Estimators. A geometric learning procedure is a measurable mapping

$$
{ \widehat { \Gamma } } _ { n } : { \mathcal { Z } } ^ { n } \longrightarrow \mathfrak { G } ,\tag{159}
$$

which associates to every dataset $\mathcal { D } _ { n }$ an estimated boundary geometry

$$
\widehat { \Gamma } _ { n } = \widehat { \Gamma } _ { n } ( \mathcal { D } _ { n } ) .\tag{160}
$$

The estimator $\widehat { \Gamma } _ { n }$ is therefore a random element of G.

Definition 34 (Hausdorf Distance). Let A, $B \subseteq S$ be nonempty closed subsets.

The Hausdorf distance between A and B is defined by

$$
d _ { H } ( A , B ) = \operatorname* { m a x } \left\{ \operatorname* { s u p } _ { x \in A } \operatorname* { i n f } _ { y \in B } d ( x , y ) , \ : \operatorname* { s u p } _ { y \in B } \operatorname* { i n f } _ { x \in A } d ( x , y ) \right\} ,\tag{161}
$$

where $d ( \cdot , \cdot )$ denotes the metric on $\boldsymbol { \mathcal { S } }$

The Hausdorf distance provides a natural notion of geometric discrepancy because it measures the maximal localization error between two boundary structures.

Definition 35 (Boundary Estimation Error). The estimation error associated with a boundary estimator $\widehat { \Gamma } _ { n }$ is defined $b y$

$$
\mathcal { E } _ { n } = d _ { H } \big ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \big ) .\tag{162}
$$

Definition 36 (Geometric Risk). The geometric risk of a boundary estimator $\widehat { \Gamma } _ { n }$ is

$$
\mathcal { R } _ { G } ( \widehat { \Gamma } _ { n } ) = \mathbb { E } \left[ d _ { H } \big ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \big ) \right] ,\tag{163}
$$

whenever the expectation exists.

The quantity $\mathcal { R } _ { G }$ plays the role of a statistical risk function adapted to geometric estimation. Unlike classical policy-learning criteria, it measures performance directly in terms of the accuracy with which the decision boundaries are recovered.

The framework introduced above deliberately separates three distinct objects:

1. the unknown target geometry $\Gamma ^ { * }$ ;

2. the admissible geometric class $\mathfrak { G } ;$

3. the estimator $\widehat { \Gamma } _ { n }$

This separation will allow the subsequent analysis to relate statistical learnability to geometric complexity. In particular, the sample-complexity results developed below will depend on structural characteristics of the boundary geometry rather than on the cardinality of the state space or the complexity of the value function representation.

## 7.4 Boundary Sample Complexity

The statistical framework introduced in Section 7.3 provides a probabilistic formulation of the geometric learning problem. The purpose of the present subsection is to quantify the amount of information required to estimate the target boundary geometry $\Gamma ^ { * }$ with prescribed accuracy. The key question is whether the statistical dificulty of the learning problem is governed by the size of the state space or by the complexity of the boundary geometry itself. We begin by introducing the corresponding notion of sample complexity.

Definition 37 (Boundary Sample Complexity). Let $\varepsilon > 0$ and $\delta \in ( 0 , 1 )$

The boundary sample complexity is defined as

$$
N _ { \Gamma } ( \varepsilon , \delta ) = \operatorname* { i n f } \left\{ n \geq 1 : \exists \widehat { \Gamma } _ { n } \ s u c h \ t h a t \mathbb { P } \Big ( d _ { H } \big ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \big ) \leq \varepsilon \Big ) \geq 1 - \delta \right\} ,\tag{164}
$$

where $d _ { H }$ denotes the Hausdorf distance introduced in Definition $\ 3 4 .$

The quantity $N _ { \Gamma } ( \varepsilon , \delta )$ represents the smallest number of observations required to localize the target boundary geometry with geometric accuracy ε and confidence level $1 - \delta$ . The structural results established in Sections 5 and 6 suggest that the complexity of the learning problem should be governed by the complexity of the boundary geometry rather than by the ambient state space.

Indeed, Theorem 18 showed that policy variation is entirely concentrated on $\Gamma ^ { * }$ , while Theorem 27 established that the complexity of the optimal policy geometry is asymptotically equivalent to the complexity of its boundary structure.

The following theorem formalizes this observation.

Theorem 32 (Boundary Sample Complexity Principle). Suppose that the assumptions of Sections $\it 4$ and 5 hold. Then the sample complexity of the geometric learning problem is governed by the boundary complexity $C _ { B }$ . More precisely, there exists a nondecreasing function

$$
\Psi : \mathbb { R } _ { + } ^ { 3 }  \mathbb { R } _ { + }\tag{165}
$$

such that

$$
N _ { \Gamma } ( \varepsilon , \delta ) \leq \Psi ( C _ { B } , \varepsilon , \delta ) ,\tag{166}
$$

and the dependence on the state-space cardinality |S| enters only through its influence on $C _ { B }$

Proof. By Theorem 18, the optimal policy is locally invariant away from $\Gamma ^ { * }$ . Consequently, observations collected in the interior of action regions do not contribute to the localization of decision transitions. The informative component of the learning problem is therefore restricted to the boundary structure. Furthermore, Theorem $2 7$ establishes that the efective complexity of the policy geometry is characterized by $C _ { B }$ . Hence any estimator of $\Gamma ^ { * }$ requires information only about a geometric object whose complexity is measured by $C _ { B } ,$ which implies that the corresponding sample complexity is controlled by this quantity rather than by the ambient state space.

Theorem 32 constitutes the first statistical consequence of the geometric framework developed in the previous sections. Its main implication is conceptual rather than quantitative. The result establishes that the relevant notion of complexity for policy learning is the complexity of the decision boundaries and not the complexity of the state space itself. Subsequent sections will exploit this principle to derive more explicit learning guarantees and active-sampling strategies adapted to the geometry of optimal policies.

## 7.5 Active Boundary Sampling

The preceding subsections establish that the statistical complexity of policy learning is governed by the geometry of the decision boundaries. A natural question therefore arises:

Where should observations be collected in order to estimate the target boundary geometry most eficiently?

The purpose of the present subsection is not to introduce a specific learning algorithm, but rather to identify the regions of the state space that contain the largest amount of information about the unknown boundary structure.

Decision Gaps and Boundary Geometry. Recall from Definition 17 that, for every pair of actions $a , a ^ { \prime } \in { \mathcal { A } }$

$$
\Delta _ { a a ^ { \prime } } ( s ) = Q ^ { * } ( s , a ) - Q ^ { * } ( s , a ^ { \prime } ) ,\tag{167}
$$

denotes the corresponding action-diference function.

Furthermore, Corollary 6 established that pairwise decision boundaries admit the representation

$$
\Gamma _ { a a ^ { \prime } } = \left\{ s \in \mathcal { S } : \Delta _ { a a ^ { \prime } } ( s ) = 0 \right\} .\tag{168}
$$

Hence, the geometry of the optimal policy is completely determined by the zero-level sets of the action-diference functions.

The following definition identifies the states that are geometrically close to decision transitions.

Definition 38 (Boundary Neighborhood). Let $\tau > 0$

For each pair of actions a, $a ^ { \prime } \in { \mathcal { A } }$ , define

$$
\begin{array} { r } { \mathcal { B } _ { a a ^ { \prime } } ( \tau ) = \{ s \in \mathcal { S } : | \Delta _ { a a ^ { \prime } } ( s ) | \leq \tau \} . } \end{array}\tag{169}
$$

The global boundary neighborhood is

$$
B ( \tau ) = \bigcup _ { a , a ^ { \prime } \in \mathcal { A } \atop a \neq a ^ { \prime } } B _ { a a ^ { \prime } } ( \tau ) .\tag{170}
$$

By construction,

$$
\begin{array} { r } { \Gamma ^ { \ast } \subseteq B ( \tau ) , \quad \forall \tau > 0 . } \end{array}\tag{171}
$$

Moreover,

$$
\bigcap _ { \tau > 0 } B ( \tau ) = \Gamma ^ { * } .\tag{172}
$$

Thus, $B ( \tau )$ provides a geometric approximation of the target boundary structure.

The next principle identifies the regions of maximal statistical relevance.

Principle 2 (Active Boundary Sampling Principle). Among all states in the state space, those belonging to boundary neighborhoods $B ( \tau )$ contain the largest amount of information regarding the location of the target boundary geometry $\Gamma ^ { * }$

Consequently, observation mechanisms that allocate a larger fraction of samples to $B ( \tau )$ are expected to estimate the policy geometry more eficiently than observation mechanisms based on uniform exploration of the state space.

The intuition follows directly from Theorem 18. Inside the interior of an action region, the optimal policy is locally invariant. Additional observations therefore provide little information regarding the location of decision transitions. In contrast, states satisfying

$$
\left| \Delta _ { a a ^ { \prime } } ( s ) \right| \approx 0\tag{173}
$$

lie near competing action regions. Small perturbations of the state may then alter the identity of the optimal action, making such observations particularly informative for boundary localization.

The principle above should be interpreted as a structural guideline rather than a concrete algorithmic prescription. Its role is to identify the regions of the state space that concentrate the informational content relevant for learning. The statistical consequences of this localization phenomenon are developed in the subsequent subsections, where boundary-focused learning procedures are shown to exploit the geometric compression properties established in Sections 5 and 6.

## 7.6 Policy Reconstruction

The previous subsections formulate policy learning as a geometric estimation problem whose objective is the recovery of the target boundary geometry $\Gamma ^ { * }$

The purpose of the present subsection is to establish the converse link between geometry estimation and policy estimation. The key question is the following:

If the boundary geometry can be estimated accurately, does this sufice to recover the optimal policy?

The answer follows from the structural results established in Sections 5 and 6.

Recall that Theorem 23 established that the optimal policy is uniquely determined by its boundary geometry. Consequently, the estimation of $\Gamma ^ { * }$ naturally induces an estimator of the optimal policy.

Definition 39 (Policy Reconstruction Operator). Let G denote the admissible family of boundary geometries introduced in Section 7.3.

A policy reconstruction operator is a mapping

$$
\Re : { \mathfrak { G } } \longrightarrow \Pi ,\tag{174}
$$

which associates with every admissible boundary geometry $\Gamma \in \mathfrak { s }$ the unique deterministic policy consistent with the induced tessellation.

The existence and uniqueness of R follow from Theorem 23 together with Proposition 17.

Given a boundary estimator ${ \widehat { \Gamma } } _ { n } .$ , the corresponding policy estimator is defined through geometric reconstruction.

Definition 40 (Reconstructed Policy Estimator). Let $\widehat { \Gamma } _ { n } \in \mathfrak { G }$ be a boundary estimator.

The reconstructed policy estimator is

$$
\widehat { \pi } _ { n } = \Re \Big ( \widehat { \Gamma } _ { n } \Big ) .\tag{175}
$$

The next theorem establishes that consistency of boundary estimation implies consistency of policy reconstruction.

Theorem 33 (Policy Reconstruction Principle). Suppose that

$$
d _ { H } \bigl ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \bigr ) \longrightarrow 0 \qquad i n \ p r o b a b i l i t y .\tag{176}
$$

Then the reconstructed policy sequence $\widehat { \pi } _ { n } = \Re \Big ( \widehat { \Gamma } _ { n } \Big )$ converges to the optimal policy $\pi ^ { * }$ in the sense that policy discrepancies can occur only inside neighborhoods whose size vanishes with

$$
d _ { H } \big ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \big ) .\tag{177}
$$

Proof. By Theorem 18, the optimal policy is locally constant away from the boundary geometry.

Therefore, any discrepancy between $\widehat { \pi } _ { n }$ and $\pi ^ { * }$ must arise from inaccuracies in the localization of decision boundaries. Since $d _ { H } \big ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \big ) \to 0$ , the estimated boundaries converge geometrically to the true boundaries. Consequently, the regions in which policy disagreement may occur shrink toward the true boundary geometry. Outside these shrinking neighborhoods, the reconstructed policy coincides with the optimal policy. The claim follows.

Theorem 33 provides the final link in the geometric learning framework. Combined with Theorem 32 and Principle 2, it shows that policy learning can be reduced to three successive steps:

1. estimate the boundary geometry;

2. localize the decision boundaries accurately;

3. reconstruct the policy through the operator $\Re$

Thus, the statistical analysis of policy learning may be conducted entirely through the geometry of decision boundaries.

## 7.7 Learning Guarantees

The preceding subsections establish a complete geometric formulation of the policy-learning problem. More precisely,

1. the target object of inference is the boundary geometry $\Gamma ^ { * }$ ;

2. the statistical complexity of the learning problem is governed by the complexity of this geometry;

3. policy estimators are obtained through the reconstruction operator R.

The purpose of the present subsection is to establish the final link between geometric estimation accuracy and decision accuracy. To do so, we introduce a generic notion of policy disagreement.

Definition 41 (Policy Disagreement Risk). Let $\mu$ be a probability measure on the state space S.

For a policy estimator $\widehat { \pi } _ { n }$ , the policy disagreement risk is defined by

$$
{ \mathcal { R } } _ { \pi } ( { \widehat { \pi } } _ { n } ) = \mu ( \{ s \in S : { \widehat { \pi } } _ { n } ( s ) \neq \pi ^ { * } ( s ) \} ) .\tag{178}
$$

The quantity $\scriptstyle { \mathcal { R } } _ { \pi }$ measures the probability mass of states on which the reconstructed policy disagrees with the optimal policy.

The next theorem establishes that geometric estimation error controls policy error.

Theorem 34 (Geometric Learning Guarantee). Suppose that the assumptions of Theorem 33 hold. Then there exists a nondecreasing function

$$
\Phi : \mathbb { R } _ { + } \to \mathbb { R } _ { + }\tag{179}
$$

satisfying

$$
\Phi ( 0 ) = 0 ,\tag{180}
$$

such that every reconstructed policy estimator

$$
{ \widehat { \pi } } _ { n } = \Re { \left( { \widehat { \Gamma } } _ { n } \right) }\tag{181}
$$

satisfies

$$
\begin{array} { r } { \mathcal { R } _ { \pi } ( \widehat { \pi } _ { n } ) \leq \Phi \Big ( d _ { H } \big ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \big ) \Big ) . } \end{array}\tag{182}
$$

Consequently,

$$
d _ { H } \bigl ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \bigr ) \longrightarrow 0\tag{183}
$$

implies

$$
\mathcal { R } _ { \pi } ( \widehat { \pi } _ { n } ) \longrightarrow 0 .\tag{184}
$$

Proof. By Theorem 18, the optimal policy is locally constant away from the boundary geometry.

Furthermore, Theorem 33 establishes that policy discrepancies can occur only inside neighborhoods whose size is controlled by the Hausdorf distance $d _ { H } \big ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \big )$

Therefore, the disagreement set

$$
\{ s : \widehat { \pi } _ { n } ( s ) \neq \pi ^ { * } ( s ) \}\tag{185}
$$

is contained in a neighborhood of $\Gamma ^ { * }$ whose radius vanishes together with the geometric estimation error. The measure of this neighborhood defines a nondecreasing function Φ satisfying $\Phi ( 0 ) = 0$ which yields (182). Taking the limit as $d _ { H } \big ( \widehat { \Gamma } _ { n } , \Gamma ^ { * } \big ) \to 0$ gives (184).

Theorem 34 constitutes the principal statistical consequence of the geometric learning framework. Combined with Theorem 32, Principle 2, and Theorem 33, it establishes the following chain of implications:

$$
\begin{array} { r } { \mathrm { B o u n d a r y ~ C o m p l e x i t y } \Longrightarrow \mathrm { B o u n d a r y ~ L e a r n a b i l i t y } \Longrightarrow \mathrm { B o u n d a r y ~ E s t i m a t i o n } } \\ { \Longrightarrow \mathrm { P o l i c y ~ R e c o n s t r u c t i o n } \Longrightarrow \mathrm { P o l i c y ~ A c c u r a c y } . } \end{array}
$$

Thus, the learning performance of structured dynamic programs can be analyzed entirely through the geometry of their decision boundaries.

## 7.8 Discussion

The results developed throughout this section provide a geometric interpretation of policy learning for structured dynamic programs. The central insight is that the learning problem inherits the structural regularity of the underlying decision process. Sections 4 and 5 showed that structural properties of the optimal value function induce regular geometric properties of the optimal policy. In particular, policy variation is localized on a comparatively small boundary structure rather than being distributed throughout the entire state space.

This observation fundamentally changes the perspective on policy learning. Instead of viewing the objective as the estimation of a value function or a decision rule defined over the whole state space, the learning problem may be reformulated as the estimation of the target boundary geometry $\Gamma ^ { * }$ . The boundary learning principle, the sample-complexity analysis, the active-sampling framework, and the reconstruction results collectively show that the statistically relevant information is concentrated on the decision boundaries. The resulting chain of implications may be summarized as

$$
\mathrm { S t r u c t u r e } \Longrightarrow \mathrm { G e o m e t r y } \Longrightarrow \Gamma ^ { * } \Longrightarrow \mathrm { L e a r n a b i l i t y } .\tag{186}
$$

The learning consequences of this geometric viewpoint follow directly from the compression results established in Section 6. In particular, Theorem 27 identified the boundary structure as the efective carrier of decision complexity, while Theorem 32 showed that the statistical complexity of learning is governed by this same object. Consequently, the complexity parameter controlling policy learning is not the size of the ambient state space but the complexity of the boundary geometry itself. At a conceptual level, the results suggest the relationship

$$
\begin{array} { r } { \mathcal { C } _ { D } = \Theta ( C _ { B } ) \qquad \Longrightarrow \qquad N _ { \Gamma } = O ( C _ { B } ) , } \end{array}\tag{187}
$$

up to the accuracy and confidence factors appearing in the corresponding statistical guarantees.

Taken together, Sections 5, 6, and 7 establish that the learnability of structured optimal policies is determined by the geometry of their decision boundaries. This conclusion provides the theoretical foundation for the numerical investigations reported in the next section.

## 8 Numerical Validation of Geometric Learning Theory

The purpose of this section is not to establish the empirical superiority of a particular learning algorithm, but rather to assess whether the geometric, statistical, and information-theoretic predictions developed in Sections 3–7 are supported by controlled numerical experiments. Throughout the paper, the theoretical analysis identifies the decision-boundary geometry $\Gamma ^ { \star } = \Gamma ( \pi ^ { \star } )$ , rather than the value function or the optimal policy itself, as the primary object of inference. Consequently, every numerical experiment is designed to estimate a theoretical quantity introduced earlier, such as the boundary estimator ${ \widehat { \Gamma } } _ { n } .$ , the Hausdorf risk $d _ { H } ( \widehat { \Gamma } _ { n } , \Gamma ^ { \star } )$ , the policy disagreement risk $\mathcal { R } _ { \pi } .$ the decision complexity $\mathcal { C } _ { D }$ , or the Decision Compression Ratio (DCR), and to compare the observed behaviour with the corresponding theoretical prediction.

Accordingly, the numerical study is organized as a sequence of empirical validations of the main theoretical results established in the previous sections. Rather than evaluating predictive performance in isolation, each group of experiments investigates a precise mathematical statement. The reconstruction experiments examine the predictions of the Policy Boundary Principle and the Policy Reconstruction Principle; the convergence experiments evaluate the theoretical notion of boundary sample complexity; the compression experiments investigate the scaling laws derived for the Decision Compression Ratio and structural complexity; and the robustness experiments assess the stability properties predicted under smooth perturbations of the decision geometry. The interpretation of every figure and table is therefore explicitly tied to the corresponding theoretical result.

To ensure that the empirical evidence remains directly interpretable from a statistical perspective, all experiments are conducted under a fully controlled black-box setting. The learner has access exclusively to oracle action labels and never observes privileged information such as value functions, action-value functions, gradients, threshold locations, or the true boundary geometry.

Independent random seeds are used to separate the stochastic generation of the oracle geometry from the sampling strategy employed by the learner, ensuring that competing methods are evaluated on identical underlying decision problems. Unless stated otherwise, all reported quantities correspond to averages over thirty independent replications, and uncertainty is quantified by 95% confidence intervals.

Table 1 summarizes the complete experimental protocol, including the construction of the oracle geometries, query budgets, evaluation metrics, perturbation scenarios, scalability settings, and reproducibility outputs. The protocol has been designed so that every numerical result reported in the remainder of this section can be interpreted as empirical evidence supporting or challenging a specific theoretical prediction established in Sections 3-7.

## 8.1 Experimental Protocol

Table 1 reports the complete experimental configuration used throughout the numerical study. The protocol specifies the generation of the black-box oracle, the construction of structured and unstructured policy geometries, the active and uniform query strategies, the scalability scenarios, the perturbation experiments, and the evaluation criteria adopted for all subsequent analyses. Unless explicitly indicated, every figure and every table presented in this section follows exactly this experimental protocol.

Table 1: Experimental configuration and black-box boundary-learning protocol.
<table><tr><td>Component</td><td>Setting</td><td>Description</td></tr><tr><td>Random replications</td><td>30 seeds</td><td>All statistics are computed over independent oracle geometries and independent query-design replications.</td></tr><tr><td>State domain</td><td> $( x , z ) \in [ - 1 , 1 ] ^ { 2 }$ </td><td>Two-dimensional state space used for boundary visualization, Hausdorff evalua- tion, and reconstructed-policy testing.</td></tr><tr><td>Action space</td><td> $| \mathcal { A } | \in \{ 2 , 3 , 4 , 6 , 8 , 1 2 , 1 6 \}$ </td><td>Scalability experiments vary the number of discrete oracle actions; the baseline policy-reconstruction experiment uses |A| = 4.</td></tr><tr><td>Structured oracle</td><td>Smooth monotone tessellations</td><td>The oracle policy is generated from smooth ordered decision boundaries, produc- ing a low-dimensional geometric representation of the optimal policy.</td></tr><tr><td></td><td>Unstructured baselinesTable, checkerboard, random labels</td><td>Unstructured and fragmented label maps are used as falsification benchmarks to test whether boundary learning fails when geometric regularity is removed.</td></tr><tr><td>Perturbation design</td><td>Smooth boundary noise</td><td>Decision boundaries are perturbed by controlled smooth perturbations with am- plitude in [0, 0.10].</td></tr><tr><td>Black-box oracle</td><td> $( x , z ) \mapsto \pi ^ { * } ( x , z )$ </td><td>The learner observes only action labels and never observes value functions, action</td></tr><tr><td>Target object</td><td> $\Gamma ^ { * } = \Gamma ( \pi ^ { * } )$ </td><td>gaps, thresholds, gradients, or true boundary locations. The statistical target is the decision-boundary geometry of the oracle policy, rather than  $V ^ { * } \mathrm { o r } \bar { Q } ^ { * } .$ </td></tr><tr><td>Uniform baseline</td><td>Random state queries</td><td>Labels are collected from uniformly sampled states and boundaries are recon- structed from label transitions.</td></tr><tr><td>Active method</td><td>Boundary-focused adaptive sampling</td><td>Queries are concentrated near estimated action-transition regions, followed by local bracketing and bisection of decision boundaries.</td></tr><tr><td>Budget grid</td><td>102 to 105</td><td>Nominal query budgets are used to compare Hausdorff convergence, policy dis- agreement, and sample complexity.</td></tr><tr><td>Boundary grid</td><td>240 sections</td><td>Resolution used for representing boundary curves, computing Hausdorff error, and reconstructing policies.</td></tr><tr><td>Bisection depth</td><td>At most 22 steps</td><td>Maximum number of label-only bisection steps used to refine each estimated</td></tr><tr><td>Evaluation sample</td><td>100,000 states</td><td>boundary point. Out-of-sample Monte Carlo sample used to estimate reconstructed-policy dis- agreement risk.</td></tr><tr><td>Boundary error</td><td> $d _ { H } ( \widehat { \Gamma } , \Gamma ^ { * } )$ </td><td>Hausdorff-type distance between estimated and oracle boundaries, used only for ex-post evaluation.</td></tr><tr><td>Policy error</td><td> ${ \mathcal { R } } _ { \pi } = \mathbb { P } \{ { \widehat { \pi } } ( S ) \neq \pi ^ { * } ( S ) \}$ </td><td>Out-of-sample disagreement probability between the reconstructed policy and the black-box oracle.</td></tr><tr><td>Compression metrics</td><td> $C _ { D } , \mathrm { D C R } , L _ { \mathrm { t a b l e } } / L _ { \mathrm { s t r u c t } }$ </td><td>Decision complexity, decision-compression ratio, and MDL-style compression gain quantify the structural advantage of boundary representations.</td></tr><tr><td>Generalization test</td><td>Out-of-geometry random tessellations</td><td>The learned reconstruction protocol is evaluated across independently generated smooth geometries not used to tune the method.</td></tr><tr><td>Sample complexity</td><td> $N _ { \Gamma } ( \varepsilon , \delta )$ </td><td>Empirical query budget required to reach dH(Γ, Γ*) ≤ ε with prescribed success</td></tr><tr><td></td><td>Reproducibility outputs CSV, LATEX, PDF, PNG</td><td>frequency. All raw measurements, aggregated tables, and figure inputs are saved for full reproducibility.</td></tr></table>

Notes. The protocol separates the seed defining the oracle geometry from the seed controlling query sampling. Uniform and active methods are therefore evaluated on the same target policies. The active method exploits structural regularity through label-only boundary localization, but does not use privileged access to the value function, gradients, action gaps, or true boundary positions.

## 8.2 Empirical Validation of Boundary Geometry Learning

The first objective of the numerical study is to validate the geometric formulation developed in Sections 5 and 7. The theoretical analysis establishes that the reconstruction problem can be formulated as the estimation of the target boundary geometry $\Gamma ^ { \star } = \Gamma ( \pi ^ { \star } )$ , rather than as the approximation of the value function or of the policy over the entire state space. Under the structural assumptions introduced in Section 5, the statistical behaviour of the reconstructed policy is therefore determined by the accuracy with which the boundary estimator $\widehat { \Gamma }$ approximates $\Gamma ^ { \star }$ , as quantified by the Hausdorf metric $d _ { H } ( \hat { \Gamma } , \Gamma ^ { \star } )$

The experiments reported in this subsection examine three successive theoretical predictions. First, we verify that the oracle policies generated under the experimental protocol indeed admit the low-dimensional boundary representation assumed by the Policy Boundary Principle.

Second, we investigate whether the empirical evolution of the Hausdorf error agrees with the convergence behaviour predicted by the Boundary Sample Complexity theorem. Finally, we estimate the empirical sample complexity $N _ { \Gamma } ( \varepsilon , \delta )$ required to recover the target geometry with prescribed geometric accuracy and confidence.

Unlike conventional empirical evaluations in reinforcement learning, every numerical quantity considered here corresponds directly to an object introduced in the theoretical development. Consequently, the figures and tables presented below should be interpreted as empirical estimates of the theoretical quantities appearing in Sections 5 and 7, rather than as standalone performance benchmarks.

## 8.2.1 Oracle Policy Geometry

The Policy Boundary Principle establishes that the informational content of an optimal policy is completely characterized by its decision-boundary geometry. More precisely, under the structural regularity assumptions introduced in Section 5, the oracle policy induces an ordered tessellation of the state space whose interfaces form the target boundary set $\Gamma ^ { \star }$

Figure 1 displays the oracle policy over the state domain generated according to the protocol described in Section 8.1. Although the learner has access only to oracle action labels, the resulting tessellation exhibits a collection of ordered decision regions separated by smooth transition interfaces. Extracting these interfaces yields the target boundary geometry shown in Figure 2. This geometric object is precisely the statistical target considered throughout the remainder of the paper.

Several observations are consistent with the theoretical assumptions. First, the decision regions are separated by non-intersecting boundary components, satisfying the ordering hypothesis required in the geometric analysis. Second, the complexity of the oracle policy is concentrated on a one-dimensional subset of the state space rather than being distributed throughout the full two-dimensional domain. Consequently, estimating $\Gamma ^ { \star }$ requires recovering only the transition interfaces between adjacent actions, thereby reducing policy reconstruction to a geometric estimation problem. The numerical oracle geometries therefore satisfy the structural assumptions under which the theoretical analysis has been developed.

## 8.2.2 Boundary Estimation Accuracy

Theorem 32 predicts that the statistical accuracy of policy reconstruction is governed by the convergence of the boundary estimator in the Hausdorf metric. In particular, the theory predicts that concentrating oracle queries near the decision interfaces substantially decreases the geometric sample complexity required to estimate Γ<sup>⋆</sup>.

To evaluate this prediction, we estimate the Hausdorf distance $d _ { H } ( \widehat { \Gamma } , \Gamma ^ { \star } )$ for increasing nominal query budgets using both active boundary localization and uniform state-space exploration. The resulting estimates are reported in Figure 3, while the corresponding numerical summaries appear in Table 2.

Table 2: Black-box boundary learning performance.
<table><tr><td rowspan="2">Budget</td><td colspan="2">Hausdorff error  $d _ { H } ( \widehat { \Gamma } , \Gamma ^ { * } )$ </td><td colspan="2">Policy risk  $\scriptstyle { \mathcal { R } } _ { \pi }$ </td><td rowspan="2">Gain</td></tr><tr><td>Active</td><td>Uniform</td><td>Active</td><td>Uniform</td></tr><tr><td>100</td><td> $0 . 0 4 7 \pm 0 . 0 0 5$ </td><td> $0 . 4 1 7 \pm 0 . 0 2 2$ </td><td> $0 . 0 2 5 \pm 0 . 0 0 2$ </td><td> $0 . 2 7 0 \pm 0 . 0 1 5$ </td><td> $\mathcal { R } _ { \pi } ^ { U } / \mathcal { R } _ { \pi } ^ { A }$  10.8×</td></tr><tr><td>200</td><td> $0 . 0 3 1 \pm 0 . 0 0 0 4$ </td><td> $0 . 6 9 9 \pm 0 . 0 7 0$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 0 5$ </td><td> $0 . 3 1 5 \pm 0 . 0 1 9$ </td><td>16.9×</td></tr><tr><td>500</td><td> $0 . 0 1 6 \pm 0 . 0 0 0 4$ </td><td> $0 . 4 1 6 \pm 0 . 0 5 3$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 0 2$ </td><td> $0 . 1 6 2 \pm 0 . 0 0 9$ </td><td>17.7×</td></tr><tr><td>1,000</td><td> $0 . 0 0 8 \pm 0 . 0 0 0 0 4$ </td><td> $0 . 2 1 9 \pm 0 . 0 1 9$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 0 0 9$ </td><td> $0 . 1 1 9 \pm 0 . 0 0 5$ </td><td>26.0×</td></tr><tr><td>2,000</td><td> $0 . 0 0 4 \pm 0 . 0 0 0 1$ </td><td> $0 . 1 1 5 \pm 0 . 0 0 8$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 0 0 6$ </td><td> $0 . 0 5 7 \pm 0 . 0 0 2$ </td><td>25.5×</td></tr><tr><td>5,000</td><td> $0 . 0 0 1 \pm 0 . 0 0 0 0 4$ </td><td> $0 . 0 4 3 \pm 0 . 0 0 3$ </td><td> $5 . 8 { \times } 1 0 ^ { - 4 } \pm 1 . 8 { \times } 1 0 ^ { - 5 }$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 1$ </td><td>30.5×</td></tr><tr><td>10,000</td><td> $4 . 9 { \times } 1 0 ^ { - 4 } \pm 3 . 2 { \times } 1 0 ^ { - 6 }$ </td><td> $0 . 0 2 4 \pm 0 . 0 0 2$ </td><td> $3 . 0 { \times } 1 0 ^ { - 4 } \pm 1 . 0 { \times } 1 0 ^ { - 5 }$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 1$ </td><td>37.4×</td></tr><tr><td>25,000</td><td> $2 . 4 { \times } 1 0 ^ { - 4 } \pm 3 . 6 { \times } 1 0 ^ { - 7 }$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 3$ </td><td> $1 . 7 { \times } 1 0 ^ { - 4 } \pm 9 . 7 { \times } 1 0 ^ { - 6 }$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 1$ </td><td>53.9×</td></tr><tr><td>50,000</td><td> $1 . 2 { \times } 1 0 ^ { - 4 } \pm 5 . 1 { \times } 1 0 ^ { - 8 }$ </td><td> $0 . 0 1 6 \pm 0 . 0 0 3$ </td><td> $7 . 5 { \times } 1 0 ^ { - 5 } \pm 4 . 8 { \times } 1 0 ^ { - 6 }$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td><td>113.2×</td></tr><tr><td>100,000</td><td> $6 . 1 { \times } 1 0 ^ { - 5 } \pm 3 . 4 { \times } 1 0 ^ { - 8 }$ </td><td> $0 . 0 1 5 \pm 0 . 0 0 3$ </td><td> $4 . 0 { \times } 1 0 ^ { - 5 } \pm 3 . 2 { \times } 1 0 ^ { - 6 }$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td><td>209.6×</td></tr></table>

Notes. Entries report mean ± 95% confidence interval over 30 seeds and five structured policy families. “Active” denotes the adaptive black-box boundary bisection method. “Uniform” denotes uniform random sampling over the state space. The gain column reports the policy-risk reduction factor $\mathcal { R } _ { \pi } ^ { U } / \mathcal { R } _ { \pi } ^ { A }$

![](images/6534feb153dd55ec4f251390bfee2d5a093a837f46d3b8d8e7f818eb8c4034f4.jpg)

![](images/7748146031a5e7290a5444fc871c12b9225bca2a05bf99cf76362a1107fa5200.jpg)  
Figure 2: Target boundary geometry Γ<sup>⋆</sup>. The decision boundaries are extracted from the blackbox policy tessellation and identify the loci at which the optimal action changes. This figure illustrates that the informational content of the policy is concentrated on a low-dimensional boundary set rather than distributed uniformly over the state space.

Figure 1: Black-box optimal policy tessellation. The figure displays the action regions induced by the black-box optimal policy over the twodimensional state domain. Although the learner observes only action labels, the induced policy exhibits a low-complexity geometric structure composed of ordered decision regions.  
![](images/aaa7ec2d21b3a23b08cbd7ff076a73585171a668fadcde6bcb806699078fc157.jpg)  
Figure 3: Black-box boundary estimation. The proposed active adaptive bisection procedure achieves substantially smaller Hausdorf boundary error than uniform sampling across all query budgets. The log-log scale shows that active querying progressively refines the target geometry, while uniform sampling reaches a much slower accuracy regime.

The empirical results exhibit two systematic properties. First, the Hausdorf error decreases monotonically as the number of oracle queries increases, consistent with the consistency properties established in Section 7. Second, active boundary localization produces uniformly smaller geometric errors than uniform exploration over the entire range of query budgets considered. Since both methods are evaluated on identical oracle geometries generated from the same random seeds, this improvement can be attributed exclusively to the concentration of sampling efort near the decision-boundary set.

Figure 4(a) investigates the convergence behaviour of the estimator itself. The observed trajectories indicate that the boundary estimator produced by active localization converges substantially faster toward Γ<sup>⋆</sup>, whereas uniform exploration remains limited by the ineficient allocation of oracle evaluations away from the decision interfaces. The accompanying policy disagreement values reported in Table 2 decrease consistently with the geometric error, providing empirical support for the theoretical relationship established in Section 7 between boundary estimation accuracy and policy reconstruction error.

Finally, Figure 4(b) and Table 2 examine the empirical boundary sample complexity $N _ { \Gamma } ( \varepsilon , \delta )$

For every prescribed Hausdorf tolerance, the active estimator reaches the desired geometric accuracy with substantially fewer oracle evaluations than uniform exploration. Moreover, the empirical reduction factors remain stable across confidence levels, indicating that the theoretical notion of geometric sample complexity provides an informative description of the finite-sample behaviour observed in practice. Taken together, these experiments provide consistent empirical evidence supporting the geometric learning framework developed in Sections 5 and 7. The numerical observations agree with the theoretical prediction that the statistical dificulty of black-box policy reconstruction is fundamentally governed by the estimation of the decision-boundary geometry Γ<sup>⋆</sup>, rather than by approximation of the policy over the entire state space.

![](images/60235b291e37fb31dcbd89a08af83b31e94f4943feb8bcbd97701efb8ac515ef.jpg)  
(a) Hausdorf convergence of boundary estimators.

![](images/39851ab27ef10f8d698befd888fd408f2f07ad0cde9aa6488a0242899400bcfc.jpg)  
(b) Empirical sample complexity.  
Figure 4: Boundary-estimation eficiency. Active boundary sampling achieves substantially faster Hausdorf convergence than uniform sampling while requiring fewer samples to reach the same geometric tolerance.

## 8.3 Empirical Validation of Policy Reconstruction

The second group of experiments investigates the reconstruction guarantees established in Sections 5 and 7. Whereas the previous subsection focused exclusively on estimating the target boundary geometry Γ<sup>⋆</sup>, the present experiments examine the second stage of the theoretical framework, namely the reconstruction of the oracle policy from the estimated boundary representation.

The theoretical developments show that, under the structural assumptions introduced in Section 5, the reconstructed policy is entirely determined by the estimated boundary geometry through the reconstruction operator ${ \widehat { \pi } } = \Re ( { \widehat { \Gamma } } )$ , and that the corresponding policy disagreement is controlled by the geometric estimation error measured by $d _ { H } ( \widehat { \Gamma } , \Gamma ^ { \star } )$ .

Consequently, policy reconstruction is analysed as a deterministic consequence of geometric estimation rather than as an independent statistical learning problem. The experiments reported below examine three complementary theoretical predictions. First, we evaluate whether accurate estimation of the decision-boundary geometry indeed induces an accurate reconstruction of the oracle policy. Second, we investigate the empirical relationship between Hausdorf estimation error and policy disagreement predicted by the Geometric Learning Guarantee. Finally, we examine the necessity of the structural assumptions by considering policy classes that intentionally violate the regularity conditions required by the theoretical analysis.

Unlike conventional empirical evaluations in reinforcement learning, every numerical quantity considered in this subsection corresponds directly to an object introduced in Sections 5-7. The reported experiments should therefore be interpreted as empirical estimates of the theoretical reconstruction operator and its associated error bounds, rather than as evaluations of a particular learning algorithm.

## 8.3.1 Boundary-to-Policy Reconstruction

The Policy Reconstruction Principle established in Section 5 states that, under the structural assumptions defining the admissible policy class, the oracle policy is uniquely determined by its decisionboundary geometry. Once an estimator $\widehat { \Gamma }$ of the target boundary set has been constructed, the reconstructed policy $\widehat { \pi } = \Re ( \widehat { \Gamma } )$ is therefore completely specified by the induced partition of the state space. To examine this prediction, we reconstruct the policy associated with each estimated boundary obtained in Section 8.2 and evaluate its disagreement with the oracle policy over an inde pendent Monte Carlo sample. Figure 6 reports the evolution of the empirical policy disagreement $\mathcal { R } _ { \pi } .$ whereas Figure 5 compares the oracle tessellation, the reconstructed partition, and the corresponding disagreement set.

Several observations are consistent with the theoretical reconstruction principle. First, decreasing boundary estimation error is accompanied by a systematic reduction of the policy disagreement probability across the entire range of query budgets considered. Second, the disagreement set remains localized in a narrow neighbourhood of the estimated decision boundaries, indicating that reconstruction errors arise almost exclusively from residual geometric estimation error. Finally, the reconstructed tessellation shown in Figure 5 preserves the ordering and adjacency structure of the oracle partition, illustrating that the reconstruction operator successfully recovers the global policy geometry from local boundary estimates.

These observations provide empirical support for the Policy Reconstruction Principle. In particular, they indicate that accurate estimation of $\Gamma ^ { \star }$ is suficient to recover the corresponding policy partition without requiring direct approximation of the value function, action-value function, or any additional oracle information.

![](images/af65e1f38cfc018ffa1f4dbad9fde14a621a3008155e18b3a45d770585b9130e.jpg)

![](images/8f55ce2e63d32d2d314091daaa97113142090ae0396f0665dbed462cf29b9e66.jpg)

![](images/c6aad58fa75f7d2ce76acc3526813c508c578ae01d0fe944fbd8856e8bf49381.jpg)  
Figure 5: Policy reconstruction from the estimated decision boundary. The oracle policy $\pi ^ { \star }$ , the reconstructed policy $\mathcal { R } ( \widehat { \Gamma } )$ , and the disagreement set show that the learned boundary induces an almost identical policy partition, with very small Hausdorf and policy-disagreement errors.

## 8.3.2 Geometry Controls Policy Risk

The Geometric Learning Guarantee developed in Section 7 predicts that policy disagreement is controlled by the geometric estimation error. More precisely, the theoretical analysis establishes the existence of a monotone function $\Phi$ such that $\mathcal { R } _ { \pi } \leq \Phi \Big ( d _ { H } ( \widehat \Gamma , \Gamma ^ { \star } ) \Big )$ , implying that improvements in boundary estimation necessarily induce improvements in policy reconstruction.

To evaluate this prediction, we jointly estimate the Hausdorf error and the corresponding policy disagreement over all policy families, query budgets, and independent replications. Figure 7 displays the complete collection of empirical observations, while Figure $8 ( \mathrm { a } )$ summarizes the resulting geometric relationship. The empirical observations exhibit a remarkably stable monotone dependence between the two quantities over several orders of magnitude. On the logarithmic scale, the observed relationship is approximately linear, indicating that reductions in Hausdorf estimation error translate proportionally into reductions in policy disagreement.

![](images/0a9ac711905344bd44d67c3c907f1a62ae50d9703de289e1dfaaa2ed695d88a5.jpg)  
Figure 6: Policy reconstruction from black-box queries. The policy reconstructed from the estimated boundary geometry exhibits a rapidly decreasing disagreement risk relative to the oracle policy.

Moreover, the fitted slope reported in Figure 8(a) remains close to the behaviour predicted by the theoretical analysis, with no systematic deviations observed across policy families or sampling budgets. Overall, the numerical evidence is consistent with the Geometric Learning Guarantee developed in Section 7. The experiments indicate that the Hausdorf metric captures the dominant source of reconstruction error and therefore provides an informative geometric surrogate for the statistical behaviour of the reconstructed policy.

![](images/cf52de359e13495d7c25fdb88331ec75d2f5124b471e46d83ffdea6ce5662c1a.jpg)  
Figure 7: Geometric estimation error controls policy error. Each point corresponds to one seed– family-budget configuration. The log–log relationship shows that the policy disagreement risk scales with the Hausdorf boundary estimation error.

## 8.3.3 Failure under Loss of Structural Regularity

The theoretical guarantees established in Sections 5 and 7 rely on structural assumptions describing the geometry of the oracle decision boundaries. In particular, the Policy Boundary Principle assumes that the oracle admits an ordered boundary representation satisfying the regularity conditions introduced in the theoretical analysis. These assumptions are therefore essential components of the reconstruction theory. To investigate their necessity empirically, we apply the same reconstruction protocol to policy classes that intentionally violate the structural assumptions while keeping every other component of the experimental protocol unchanged. Figure 8(b) summarizes the resulting policy disagreement across representative policy classes, whereas Table 3 reports the corresponding quantitative comparisons.

![](images/0e677b5c16d40823329a2da73af7a40293153fb4e1aefe0492a4261b8123d8f8.jpg)  
(a) Policy error versus geometric error.

![](images/57b133793e62ffdc147214d1a4e5b99a4b5f38abb9bb44b369d99df5578f5361.jpg)  
(b) Failure without structural regularity.  
Figure 8: Geometric control of policy error and the role of structural regularity. The near-linear log–log relationship confirms that Hausdorf boundary error controls policy disagreement, whereas unstructured or irregular decision rules lead to large reconstruction risk.

The empirical observations agree with the theoretical prediction. Policy classes satisfying the structural assumptions remain reconstructible with negligible disagreement probability, whereas fragmented and unstructured decision geometries exhibit several orders of magnitude larger reconstruction error despite identical oracle-query budgets. Since the reconstruction procedure and evaluation protocol remain unchanged, the deterioration can be attributed exclusively to the absence of the geometric regularity required by the theoretical analysis. Consequently, the experiments support the interpretation that the assumptions introduced in Section 5 are genuine identifiability conditions for geometric policy reconstruction rather than technical artifacts of the mathematical proofs. The numerical evidence therefore validates not only the positive reconstruction guarantees established by the theory but also the necessity of the structural hypotheses under which these guarantees are derived.

Table 3: Failure-mode validation under loss of structural regularity.
<table><tr><td>Policy class</td><td>Policy risk  $\mathcal { R } _ { \pi }$ </td><td></td><td>Oracle queries Relative degradation</td></tr><tr><td>Structured</td><td> $3 . 9 7 { \times } 1 0 ^ { - 5 } \pm 3 . 1 7 { \times } 1 0 ^ { - 6 }$ </td><td> $2 2 , 2 5 6 \pm 1 9 4$ </td><td>1.0×</td></tr><tr><td>Unstructured</td><td> $1 . 8 5 6 \times 1 0 ^ { - 1 } \pm 4 . 4 3 \times 1 0 ^ { - 4 }$ </td><td> $^ { 2 2 , 1 1 4 \pm 3 }$ </td><td>4,673×</td></tr></table>

Notes. Entries report mean ± 95% confidence interval over 30 seeds. The same black-box reconstruction procedure is applied to structured and unstructured policy classes. The sharp increase in $\mathcal { R } _ { \pi }$ under the unstructured design confirms that the method exploits boundary regularity rather than memorizing policy labels.

## 8.4 Empirical Validation of the Active Boundary Sampling Principle

Sections 8.2 and 8.3 established that the statistical objective of black-box policy learning is the estimation of the decision-boundary geometry $\Gamma ^ { \star }$ , and that the accuracy of the reconstructed policy is governed by the Hausdorf estimation error of this geometric object. These results leave open a complementary question concerning the allocation of oracle evaluations: given that only the boundary geometry carries statistical information for policy reconstruction, where should black-box queries be performed?

Section 7 addresses this question through the Active Boundary Sampling Principle. Rather than treating every state as equally informative, the theory predicts that oracle evaluations should progressively concentrate in neighbourhoods of $\Gamma ^ { \star }$ , since only these regions contribute directly to reducing the geometric estimation error. The experiments reported below investigate whether this prediction is observed empirically.

Unlike the previous subsections, the objective is not to compare competing sampling algorithms. Instead, the experiments examine whether the empirical distribution of oracle queries evolves consistently with the geometric sampling principle implied by the theoretical analysis. Two complementary consequences are considered. First, we investigate the spatial localization of oracle evaluations relative to the target boundary geometry. Second, we evaluate whether all connected components of the decision-boundary representation converge simultaneously, as predicted by the geometric formulation developed in Sections 5 and 7.

## 8.4.1 Localization of Oracle Evaluations

The Active Boundary Sampling Principle predicts that informative oracle evaluations should become increasingly concentrated near the decision-boundary geometry Γ<sup>⋆</sup>. Indeed, away from the boundary, the oracle policy remains locally constant and additional evaluations provide little information regarding the location of the action-transition interfaces. Consequently, the theory predicts that an eficient geometric estimation strategy should allocate progressively fewer evaluations to homogeneous decision regions and increasingly more evaluations to neighbourhoods of the boundary set.

To examine this prediction, we record the spatial distribution of oracle evaluations throughout the reconstruction procedure. Figure 9(a) displays the resulting query locations together with the oracle decision boundaries, while Figure 10 summarizes the corresponding evolution of the relative policy-risk reduction with respect to uniform exploration.

![](images/adbeeb270cb8bdb8474c745cf905e6e33c9ff6193a8da2897b5b4517faee22f4.jpg)  
(a) Spatial localization of active queries.

![](images/47c224c77f6622530831bb8568970e1be56f9f1e949c9d59d4972db663f192d0.jpg)  
(b) Boundary-wise convergence.  
Figure 9: Localization and component-wise stability of active boundary learning. Active queries concentrate near decision interfaces, and all boundary components converge at comparable rates as the query budget increases.

The empirical observations are consistent with the theoretical prediction. As the reconstruction progresses, oracle evaluations become increasingly concentrated around the boundary components composing Γ<sup>⋆</sup>, whereas only a small proportion of queries remain allocated to the interior of homogeneous decision regions. This behaviour agrees with the geometric interpretation of the estimation problem, according to which statistical information is localized near the action-transition interfaces.

The increasing relative risk reduction reported in Figure 10 provides complementary evidence supporting the same conclusion.

Since both sampling strategies are evaluated under identical oracle geometries and identical experimental conditions, the observed improvement is consistent with the theoretical prediction that concentrating measurements near the target geometry yields a more informative estimator than allocating oracle evaluations uniformly over the ambient state space.

Taken together, these experiments provide empirical support for the Active Boundary Sampling Principle introduced in Section 7. Rather than exploring the entire state space uniformly, the empirical sampling distribution progressively adapts to the intrinsic geometric support of the statistical target.

![](images/354c9179e9ac1c84bcc8254c9931bb2d6b4ab6a766b2f1e3f62d15cd54f2f18e.jpg)  
Figure 10: Active boundary sampling eficiency gain. The curve reports the ratio between the disagreement risk of uniform sampling and that of the proposed active boundary method. Values above one indicate a strict improvement over the uniform baseline; the increasing gain shows that active sampling becomes increasingly advantageous as the query budget grows.

## 8.4.2 Component-wise Convergence of the Boundary Geometry

The theoretical analysis is formulated for the complete decision-boundary geometry $\Gamma ^ { \star }$ , which generally consists of several connected boundary components. Consequently, consistency of the boundary estimator requires simultaneous convergence of every component of the estimated geometry rather than accurate reconstruction of only a subset of interfaces. To investigate this prediction, we estimate each connected component independently throughout the reconstruction procedure. Figure 11 compares the reconstructed geometry with the oracle boundary, whereas Figure 9(b) reports the evolution of the component-wise Hausdorf errors as the nominal query budget increases.

Several observations are consistent with the theoretical framework. First, the reconstructed geometry preserves the global topology of the oracle boundary throughout the estimation process. No spurious boundary components, crossings, or changes in ordering are observed. Second, the Hausdorf errors associated with the individual boundary components decrease at comparable rates over the entire range of query budgets considered. Consequently, no single interface dominates the global estimation error. These observations indicate that convergence occurs uniformly over the complete decision-boundary geometry rather than being restricted to isolated regions of the state space. The empirical evidence is therefore consistent with the geometric consistency properties established in Section 7 and supports the interpretation that the estimator converges toward the entire boundary representation $\Gamma ^ { \star }$ , instead of approximating only local fragments of the policy partition.

## 8.5 Empirical Validation of Decision Compression Theory

The previous subsections established that black-box policy learning can be formulated as a geometric estimation problem whose statistical target is the decision-boundary representation $\Gamma ^ { \star }$ . Sections 5 and 7 demonstrated that accurate estimation of this geometric object is suficient for reconstructing the oracle policy and controlling the corresponding policy disagreement risk. The remaining theoretical question concerns the intrinsic complexity of this representation.

Section 6 develops a quantitative theory of decision compression by introducing several geometric complexity measures, including the Decision Compression Ratio (DCR), the decision complexity $\mathcal { C } _ { D }$ , the boundary complexity $\mathcal { C } _ { B }$ , and the topological complexity $\mathcal { C } _ { \mathrm { t o p } }$ . The associated theoretical results predict that these quantities satisfy explicit asymptotic scaling laws, that structured decision geometries admit substantially smaller representations than fragmented policies, and that the intrinsic geometric complexity remains stable as the ambient state space increases. The objective of the present subsection is to investigate whether these theoretical predictions are supported by controlled numerical experiments.

![](images/b64a9f733c58ee85ecc4bd3015614ade093757a1870576309bda6d1898b5b023.jpg)  
Figure 11: Black-box decision boundary reconstruction. Ground-truth boundaries are shown as colored curves, while the reconstructed boundaries are shown in black. The near-perfect overlap demonstrates that the proposed label-only active procedure can recover the decision geometry using only black-box policy queries, without observing the value function, action gaps, gradients, or true boundary locations.

Rather than evaluating compression performance as an engineering criterion, we estimate the mathematical quantities introduced in Section 6 and compare their empirical behaviour with the corresponding theoretical predictions. The experiments therefore constitute numerical validations of the Scaling Law, the Boundary Compression Theorem, the Information-Theoretic Compression Principle, and the Dimension-Free Geometry Theorem.

## 8.5.1 Scaling Law for Decision Compression

Theorem 3 predicts that the Decision Compression Ratio satisfies an asymptotic scaling law governed by the intrinsic geometric complexity of the policy representation. In particular, the theory establishes that the compression ratio grows proportionally with the efective size of the decision problem, with asymptotic exponent equal to one under the structured boundary model developed in Section 6.

To examine this prediction, we estimate the empirical Decision Compression Ratio over progressively larger state spaces while preserving the same underlying geometric structure. Figure 12 reports the resulting scaling behaviour on logarithmic axes, Figure 13 investigates the dependence on intrinsic state-space dimension, and Table 4 summarizes the estimated scaling exponent obtained from the log-log regression

$$
\log ( \mathrm { D C R } ) = \alpha + \beta \log ( | S | ) .
$$

The empirical observations are consistent with the theoretical prediction. The estimated scaling exponent remains statistically indistinguishable from the value predicted by Theorem 3, and the coeficient of determination indicates that the asymptotic model explains nearly all observed variability. Moreover, the exponential behaviour reported in Figure 13 agrees with the theoretical interpretation that explicit policy representations become exponentially more expensive as the intrinsic dimension increases, whereas boundary-based representations preserve their geometric description.

Taken together, these observations provide empirical support for the scaling law established in Section 6 and indicate that the asymptotic behaviour predicted by the theoretical analysis accurately describes the finite-sample regime considered throughout the numerical study.

![](images/b1e8e8a4927c7ca46912f27fb36e683544c4604333548f562adc6dc22f2fcfbe.jpg)

![](images/16c58947802546dbaea1ba27cb5e15c81bc9f6a14088341339e72f0a0b219b5f.jpg)  
Figure 12: Scaling law for decision compression. The empirical decision-compression ratio $\mathrm { D C R } ( n )$ increases proportionally to the size of the state space on logarithmic scales, closely matching the theoretical prediction with scaling exponent $\beta \approx 1$ The near-perfect agreement between theory and experiments validates the asymptotic scaling law established in Section 6 and confirms that the compression gain grows predictably with problem size.  
Figure 13: Exponential growth of the decision-compression ratio. As the intrinsic state-space dimension increases, the decisioncompression ratio follows an exponential trend that is accurately approximated by DCR(d) ∝ $e ^ { 0 . 6 9 d }$ . The empirical measurements are almost indistinguishable from the theoretical prediction, illustrating the exponential separation between explicit policy tables and boundary-based representations.

Table 4: Empirical validation of the scaling law predicted by Theorem 3. The estimated exponent is obtained from the log–log regression $\log ( \mathrm { D C R } ) = \alpha + \beta \log ( | S | )$ . Confidence intervals are computed from ordinary least squares.
<table><tr><td>Model</td><td>Estimated exponent  $\hat { \beta }$ </td><td>Std. Error</td><td>95% CI</td><td> $R ^ { 2 }$ </td></tr><tr><td>Scaling law</td><td>1.002</td><td>0.011</td><td>[0.978, 1.026]</td><td>0.989</td></tr></table>

## 8.5.2 Structural Compression and Information-Theoretic Eficiency

The Boundary Compression Theorem predicts that representation complexity is governed primarily by the intrinsic geometry of the decision boundary rather than by the cardinality of the ambient state space. Consequently, policies admitting regular boundary representations should exhibit substantially smaller description lengths than geometrically fragmented policies. The information-theoretic analysis developed in Section 6 further predicts that this structural organization induces a significant reduction in representation complexity relative to explicit tabular descriptions.

To investigate these predictions, we estimate the empirical relationships between boundary complexity $\mathcal { C } _ { B }$ , decision complexity $\mathcal { C } _ { D }$ , and representation length across both structured and unstructured policy classes. Figure 14 reports the observed dependence between boundary and decision complexity, Figure 15 compares the resulting description lengths with the identity-compression baseline, and Table 5 summarizes the corresponding complexity measures.

Several observations are consistent with the theoretical analysis. First, structured policy classes exhibit an approximately linear relationship between boundary complexity and decision complexity, whereas geometrically fragmented policies display substantially larger representation costs for comparable boundary descriptions. Second, the structured representations remain uniformly below the identity-compression baseline, indicating that the boundary description captures the policy using sub stantially fewer degrees of freedom than explicit state-wise representations. Finally, the complexity measures reported in Table 5 exhibit systematic separation between structured and unstructured geometries across every metric considered. Overall, the empirical evidence agrees with the Boundary Compression Theorem and the information-theoretic interpretation developed in Section 6.

Table 5: Comparison between structured and unstructured decision boundaries. Structured geometries preserve low decision complexity while maintaining exponentially higher compression eficiency.
<table><tr><td>Metric</td><td></td><td>Structured Unstructured Improvement</td><td></td></tr><tr><td>Decision complexity  $C _ { D }$ </td><td>53</td><td>275</td><td>×5.19</td></tr><tr><td>Topological complexity  $C _ { \mathrm { t o p } }$ </td><td>8</td><td>45</td><td>×5.63</td></tr><tr><td>Boundary description length</td><td> $2 . 1 \times 1 0 ^ { 4 }$ </td><td> $9 . 0 \times 1 0 ^ { 4 }$ </td><td>×4.29</td></tr><tr><td>Compression ratio (DCR)</td><td> $3 . 2 \times 1 0 ^ { 3 }$ </td><td>1</td><td>×3200</td></tr></table>

The observed reductions in representation complexity are therefore consistent with the geometric organization of the decision boundary rather than with implementation-specific properties of the reconstruction procedure.

![](images/9f805032e4f4dac6115d62d98525edbbc6f0827cbd493c22507f11b3e6839e71.jpg)

![](images/7b75a319f3d99dc4e41c7f3331b596a6d896cab0d8718cb8b7d0ce41cf86c951.jpg)  
Figure 14: Boundary complexity versus decision complexity. Decision complexity grows approximately linearly with boundary complexity for structured policies, whereas unstructured policies exhibit substantially larger decision complexity for comparable boundary descriptions. This confirms that geometric organization of decision regions dramatically reduces representation complexity while preserving policy behavior.  
Figure 15: Information-theoretic compression proxy. The structured representations remain well below the identity line corresponding to explicit tabular policies, demonstrating a substantial reduction in description length. The gap quantifies the information-theoretic compression achieved through boundary-based policy representations and provides empirical evidence for the compression theorem developed in Section 6.

## 8.5.3 Dimension-Free Geometry

The Dimension-Free Geometry Theorem predicts that the intrinsic topological complexity of structured decision boundaries remains uniformly bounded as the ambient state space increases. In contrast, geometrically fragmented policies are expected to exhibit persistent topological complexity independently of the sampling procedure. The theorem therefore distinguishes intrinsic geometric complexity from the dimensionality of the surrounding state space. To examine this prediction, we estimate the empirical topological complexity $\mathcal { C } _ { \mathrm { t o p } }$ over progressively larger state spaces while preserving the underlying decision geometry. Figure 16 reports the resulting estimates for both structured and fragmented policy classes. The numerical observations are consistent with the theoretical prediction. Across the entire range of problem sizes considered, the structured policy class exhibits essentially constant topological complexity, whereas fragmented policies maintain substantially larger complexity values. No systematic increase of $\mathcal { C } _ { \mathrm { t o p } }$ is observed for the structured geometries as the ambient state space grows, suggesting that the intrinsic boundary representation remains stable despite the increasing size of the decision problem. These observations support the interpretation proposed in Section 6 that geometric complexity is an intrinsic property of the decision-boundary representation rather than a direct consequence of the cardinality of the state space.

The empirical results are therefore consistent with the Dimension-Free Geometry Theorem established by the theoretical analysis.

![](images/40b4a9816fb1a49d5bd58e525c459e01479ed873f10dfe4c4e7a38c1e83ea0d9.jpg)  
Figure 16: Dimension-free geometry versus fragmentation. The topological complexity of structured policies remains essentially constant as the state space increases over several orders of magnitude, whereas fragmented policies consistently exhibit substantially larger complexity.

## 8.6 Robustness of Geometric Learning

The previous subsections established empirical evidence supporting the geometric learning framework developed in Sections 5-7. In particular, the experiments validated the geometric formulation of policy reconstruction, the statistical behaviour of the boundary estimator, the active sampling principle, and the complexity laws governing decision compression. The remaining theoretical question concerns the stability of these geometric quantities under perturbations of the underlying decision boundary.

Section 6 establishes that the geometric representation possesses intrinsic robustness properties. More precisely, the Robustness Theorem predicts that suficiently small perturbations of the target boundary geometry $\Gamma ^ { \star }$ induce only controlled variations of the associated complexity measures, including the decision complexity $\mathcal { C } _ { D }$ , the Decision Compression Ratio, and the information-theoretic description length. Consequently, the theoretical framework predicts that the statistical properties of the reconstructed policy should remain stable under smooth geometric deformations. The experiments reported below investigate these predictions from two complementary perspectives. First, we evaluate whether the complexity measures introduced in Section 6 remain stable under controlled perturbations of the oracle geometry. Second, we examine whether the reconstructed policy preserves its statistical behaviour across independently generated decision geometries, thereby assessing the robustness of the geometric representation beyond the particular realizations used during reconstruction.

Unlike robustness analyses commonly encountered in machine learning, the perturbations considered here are applied directly to the mathematical object $\Gamma ^ { \star }$ , rather than to observations, rewards, or optimization procedures. Consequently, every experiment reported below should be interpreted as an empirical assessment of the structural stability properties established by the theoretical analysis.

## 8.6.1 Robustness to Smooth Boundary Perturbations

The Robustness Theorem established in Section 6 predicts that smooth perturbations of the target boundary geometry induce only controlled variations of the intrinsic complexity measures. In particular, the theory implies that suficiently regular deformations of $\Gamma ^ { \star }$ should preserve the decision complexity, the Decision Compression Ratio, and the associated information-theoretic compression gain. To examine this prediction, we generate a family of perturbed decision boundaries by applying smooth deformations of increasing amplitude to the oracle geometry while preserving the global topology of the decision partition.

For every perturbation level, we estimate the corresponding decision complexity $\mathcal { C } _ { D }$ , the relative Decision Compression Ratio, and the MDL-based compression measure introduced in Section 6. Figure 17 reports the resulting empirical behaviour, while Table 6 summarizes the corresponding numerical values. The empirical observations are consistent with the theoretical prediction. Across the entire perturbation range considered, decision complexity increases only gradually, while the Decision Compression Ratio remains close to its reference value. Similarly, the information-theoretic compression gain exhibits only limited variation despite progressively larger geometric perturbations. Even under the largest perturbation amplitude considered, the observed variations remain modest relative to the corresponding baseline values. Overall, the numerical evidence supports the robustness properties established in Section 6. The experiments indicate that the complexity measures characterizing the geometric representation depend primarily on the global organization of the decision boundary rather than on small local deformations, thereby confirming the structural stability predicted by the theoretical analysis.

![](images/7520b61a0d7f0c1a053cd6a14ee177e883f4bf9d09528e4478246743f80fe727.jpg)  
(a) Structural complexity.

![](images/5632bb4d4e59e20ac84983ceb83cea68ccf36ec9f13a9cbd5afc0b708810bbf6.jpg)  
(b) Compression robustness.

Information-Theoretic Compression under Perturbation  
![](images/0d43b8811b4cc5084be56a064f980baefc346647ea9d881ea967f00d684ac78e.jpg)  
(c) MDL compression gain.  
Figure 17: Robustness under smooth boundary perturbations. Decision complexity remains nearly stable, the decision-compression ratio stays close to the clean-geometry baseline, and the informationtheoretic compression gain remains controlled even under increasing perturbation amplitude.

## 8.6.2 Stability of Policy Reconstruction Across Geometries

The theoretical framework further predicts that the geometric representation should remain statistically meaningful beyond the particular oracle geometry used during reconstruction. If the quantities introduced in Sections 5-7 capture intrinsic properties of the decision boundary, then the resulting complexity measures and reconstruction errors should remain stable across independently generated geometries satisfying the same structural assumptions.

To investigate this prediction, we evaluate the reconstructed boundary representation over independently generated smooth decision geometries that are not used during estimation.

Table 6: Robustness of the proposed decision-compression mechanism under smooth boundary perturbations. Even for the largest perturbation amplitude, decision complexity and information-theoretic compression remain remarkably stable.
<table><tr><td>Perturbation</td><td>Decision Complexity  $C _ { D }$ </td><td>Relative DCR</td><td>MDL Compression  $L _ { \mathrm { t a b l e } } / L _ { \mathrm { s t r u c t } }$ </td><td>Relative variation (%)</td></tr><tr><td>0.000</td><td>19.21</td><td>1.000</td><td>10.39</td><td>0.00</td></tr><tr><td>0.002</td><td>19.21</td><td>1.000</td><td>10.39</td><td>0.02</td></tr><tr><td>0.005</td><td>19.21</td><td>0.999</td><td>10.39</td><td>0.05</td></tr><tr><td>0.010</td><td>19.22</td><td>0.999</td><td>10.38</td><td>0.11</td></tr><tr><td>0.020</td><td>19.23</td><td>0.999</td><td>10.37</td><td>0.24</td></tr><tr><td>0.035</td><td>19.26</td><td>0.997</td><td>10.34</td><td>0.61</td></tr><tr><td>0.050</td><td>19.34</td><td>0.993</td><td>10.28</td><td>1.06</td></tr><tr><td>0.075</td><td>19.55</td><td>0.983</td><td>10.11</td><td>2.05</td></tr><tr><td>0.100</td><td>19.89</td><td>0.967</td><td>9.87</td><td>3.26</td></tr></table>

![](images/eaafc6e315d2453d5391803c98eb5199ffff35e2b273b7539c9105dcf2a7eb68.jpg)  
(a) Complexity–risk Pareto frontier.

![](images/df5c903dd8fa891a2d4471f18284558d1064f72f7d19c109749bd0448a879e51.jpg)  
(b) Out-of-geometry generalization.  
Figure 18: Complexity–risk trade-of and out-of-geometry generalization. Structured active boundary learning lies on a substantially better Pareto frontier and generalizes across random smooth tessellations with much lower policy-disagreement risk than the uniform baseline.

Figure 18 reports the empirical complexity-risk frontier together with the corresponding out-ofgeometry generalization behaviour, whereas Figure 19 summarizes the joint empirical distribution of decision complexity $\mathcal { C } _ { D }$ and policy disagreement risk $\mathcal { R } _ { \pi }$ across the complete collection of randomly generated geometries.

Several observations agree with the theoretical framework. First, the structured geometric representations consistently occupy the low-complexity, low-risk region of the empirical Pareto frontier, whereas unstructured representations remain confined to substantially less favourable regions of the complexity-risk plane. Second, the out-of-geometry experiments exhibit uniformly small policy disagreement across independently generated geometries satisfying the structural assumptions introduced in Section 5. Finally, the joint empirical distribution reveals a clear separation between structured and unstructured policy classes, with little overlap between their respective complexity-risk regimes. Taken together, these experiments provide empirical evidence supporting the robustness properties established by the geometric learning theory. The observed stability across perturbed and independently generated geometries suggests that the statistical behaviour of the reconstructed policy is governed primarily by the intrinsic geometric structure of the decision boundary rather than by particular realizations of the oracle policy.

## 8.7 Empirical Synthesis

The numerical study presented throughout Sections 8.2-8.6 was designed to examine the theoretical predictions established in Sections $5 \mathrm { - } 7$ under a common experimental protocol. Rather than evaluating predictive performance as an end in itself, each experiment estimated a mathematical quantity introduced by the theoretical analysis and investigated whether its empirical behaviour was consistent with the corresponding theorem or theoretical principle.

![](images/34b6522068e6142bb9773a86c5dbaf164a8792a91a252c96e84e224d150b1036.jpg)  
Figure 19: Complexity-risk joint distribution across random geometries. Active boundary sampling concentrates in the low-complexity, low-risk region, whereas uniform sampling remains in a highcomplexity, high-risk regime. The centroids summarize the clear Pareto dominance of structured active learning.

Consequently, the numerical evidence should be interpreted as a sequence of empirical assessments of the geometric learning theory rather than as a benchmark comparison between learning algorithms.

Table 7 summarizes this correspondence. For each major theoretical result, the table identifies the mathematical quantity estimated experimentally, the numerical evidence supporting its empirical behaviour, and the principal conclusion that may reasonably be drawn from the experiments. The table therefore provides a direct correspondence between the theoretical developments of Sections 5-7 and the numerical investigations reported throughout Section 8.

Several general observations emerge from this synthesis.

First, the experiments consistently support the geometric formulation of black-box policy learning proposed in this paper. Across all experimental settings, the decision-boundary geometry $\Gamma ^ { \star }$ behaves as the primary statistical object governing policy reconstruction. The observed evolution of the boundary estimator $\widehat { \Gamma }$ and the associated Hausdorf error agrees with the theoretical interpretation that policy learning may be formulated as a geometric estimation problem. Second, the empirical relationship between Hausdorf estimation error and policy disagreement is consistent with the Geometric Learning Guarantee established in Section 7. The numerical results indicate that improvements in geometric reconstruction are systematically accompanied by reductions in policy disagreement, in agreement with the theoretical bounds linking $d _ { H } ( \widehat { \Gamma } , \Gamma ^ { \star } )$ and $\scriptstyle { \mathcal { R } } _ { \pi }$ . Third, the experiments examining decision compression exhibit empirical behaviour consistent with the complexity theory developed in Section 6. The observed scaling of the Decision Compression Ratio, the dependence of decision complexity on boundary complexity, and the stability of the topological complexity across increasing problem sizes all agree with the qualitative and quantitative predictions established by the corresponding theoretical results. Finally, the perturbation and generalization experiments provide empirical evidence supporting the structural stability of the geometric representation. Moderate smooth deformations of the decision boundary produce only limited changes in the complexity measures introduced in Section 6, while independently generated structured geometries exhibit statistical behaviour consistent with the theoretical framework developed throughout the paper.

Naturally, these experiments do not establish the mathematical validity of the theoretical results, which follows from the proofs presented in Sections 5-7. Their purpose is instead to examine whether the finite-sample behaviour observed under controlled experimental conditions agrees with the theoretical predictions. Within the experimental regime considered in this paper, no systematic empirical contradiction with the proposed geometric learning theory is observed.

Taken together, the numerical evidence provides consistent empirical support for the theoretical framework developed in this paper. The experiments indicate that the principal geometric, statistical, and information-theoretic predictions derived in Sections 5-7 accurately describe the behaviour observed across the controlled black-box policy reconstruction problems considered throughout the numerical study.

Table 7: Empirical validation of the theoretical results established in Sections 4-7. Each theoretical property is associated with its empirical estimator and the corresponding numerical evidence.
<table><tr><td>Theoretical result</td><td>Empirical quantity</td><td>Experimental evidence</td><td>Main conclusion</td></tr><tr><td>Threshold Representation (Theorem 5) Estimated boundary Î</td><td></td><td>Figs. 3, 4 Tables 2,3</td><td>The reconstructed boundaries converge rapidly to the oracle geometry under active sampling.</td></tr><tr><td>Policy Reconstruction Principle (Theo- Policy disagreement Rπ rem 33)</td><td></td><td>Figs. 5, 8</td><td>Accurate reconstruction of the boundary geome- try implies accurate recovery of the oracle policy.</td></tr><tr><td>Active Boundary Sampling Principle Boundary localization and Figs. 9a, 9b (Principle 2)</td><td>query allocation</td><td></td><td>Queries naturally concentrate near decision boundaries, yielding substantially improved sam-</td></tr><tr><td>Scaling Law for Decision Compression Decision Compression Ra- Figs. 12, 16 (Theorem 24)</td><td>tio (DCR)</td><td>Table 4</td><td>ple efficiency. The empirical scaling exponent agrees with the theoretical compression law and remains essen-</td></tr><tr><td>Boundary Compression Theorem (The- Decision complexity CD orem 27)</td><td></td><td>Figs. 14, 15 Table 5</td><td>tially dimension-free. Decision complexity scales with boundary com- plexity rather than the size of the value represen-</td></tr><tr><td>Information-Theoretic (Theorem 29)</td><td>Compression Description length Lπ/LΓ Figs. 15, 17</td><td></td><td>tation. Boundary representations achieve substantial information-theoretic compression while preserv- ing decision accuracy.</td></tr><tr><td>Boundary Stability (Theorem 21)</td><td>Robustness under smooth Figs. 17, 18 perturbations</td><td>Table 6</td><td>Decision compression and policy reconstruction remain stable under moderate geometric pertur- bations.</td></tr><tr><td>Geometric Learning Guarantee (Theo- Joint behaviour of dH and Figs. 8, 19 rem 34)</td><td>Rπ</td><td></td><td>The observed decrease of policy disagreement is consistent with the theoretical geometric learning guarantee.</td></tr></table>

## 9 Conclusion

This paper develops a geometric theory of black-box policy learning. Rather than treating policy reconstruction as a problem of approximating value functions or state–action mappings over highdimensional state spaces, the proposed framework reformulates the problem as one of geometric inference, where the primary object of estimation is the decision-boundary geometry associated with the optimal policy. Under this viewpoint, statistical estimation, computational complexity, oraclequery allocation, and information-theoretic compression become diferent manifestations of a common geometric representation.

Starting from this formulation, the paper establishes a sequence of complementary theoretical results. The decision-boundary geometry is shown to provide a suficient representation of structured optimal policies, and accurate estimation of this geometric object is proved to imply accurate policy reconstruction. The analysis further characterizes the statistical behaviour of boundary estimation through Hausdorf convergence and boundary sample complexity, provides a geometric interpretation of active oracle sampling, and develops a quantitative theory of decision compression based on intrinsic geometric complexity rather than the cardinality of the ambient state space. Collectively, these results identify the geometry of the decision boundary as the central mathematical object governing policy reconstruction in the structured setting considered throughout the paper.

The numerical investigation was designed accordingly. Rather than evaluating predictive performance in isolation, each experiment examined a specific theoretical prediction established in Sections 5-7 by estimating the corresponding mathematical quantity under a controlled black-box protocol. Within the experimental regime considered here, the observed finite-sample behaviour is consistently aligned with the theoretical analysis. The empirical results therefore provide supporting evidence that the proposed geometric framework captures the statistical, computational, and information-theoretic phenomena predicted by the theory. The analysis presented in this paper is intentionally restricted to structured decision geometries satisfying the regularity assumptions introduced in Section 5. Whether analogous geometric principles extend to more general decision processes remains an open mathematical question. In particular, extending the present framework to continuous-action problems, partially observable systems, stochastic boundary evolutions, or strategic multi-agent decision environments will require new theoretical tools.

Equally important is the development of minimax lower bounds, statistical optimality results for geometric boundary estimators, and a deeper understanding of the interplay between geometric regularity and sample complexity.

More broadly, we hope that the perspective developed in this paper contributes to a shift in how black-box policy learning is analysed. The results suggest that, for a broad class of structured decision problems, complexity should not necessarily be understood through the dimensionality of the ambient state space, but through the geometry of the decision boundary itself. From this viewpoint, statistical estimation, computational eficiency, active sampling, and information-theoretic compression are no longer separate phenomena; they arise as diferent consequences of the same underlying geometric structure.

## Competing Interests

The authors declare that they have no competing financial or non-financial interests related to this work.

## Data Availability

This study is based exclusively on simulation experiments. The source code and all scripts required to reproduce the numerical results are available at

https://github.com/phdPokou/A-Geometric-Theory-of-Decision-Boundaries

## Ethics Approval

This study does not involve human participants, personal data, or animals and therefore does not require ethics approval.

## Funding

The authors received no specific funding for this work.

## References

M. Anthony and P. L. Bartlett. Neural network learning: Theoretical foundations. cambridge university press, 2009.

R. Bellman. Dynamic programming: Princeton univ. press. Princeton.[Google Scholar], 1957.

D. Bertsekas. Dynamic programming and optimal control: Volume I, volume 4. Athena scientific, 2012.

D. P. Bertsekas. Neuro-dynamic programming. In Encyclopedia of optimization, pages 1–6. Springer, 2025.

S. Boyd and L. Vandenberghe. Convex optimization. Cambridge university press, 2004.

L. A. Cafarelli. A geometric approach to free boundary problems, volume 68. American Mathematical Soc., 2005.

L. Devroye, L. Györfi, and G. Lugosi. A probabilistic theory of pattern recognition, volume 31. Springer Science & Business Media, 2013.

H. Edelsbrunner. Algorithms in combinatorial geometry, volume 10. Springer Science & Business Media, 1987.

E. A. Feinberg and A. Shwartz. Handbook of Markov decision processes: methods and applications, volume 40. Springer Science & Business Media, 2012.

P. D. Grünwald. The minimum description length principle. MIT press, 2007.

O. Hernández-Lerma and J. B. Lasserre. Discrete-time Markov control processes: basic optimality criteria, volume 30. Springer Science & Business Media, 2012.

G. Koole. Monotonicity in Markov reward and decision chains: Theory and applications. Now Publishers Inc, 2007.

V. Krishnamurthy. Partially observed Markov decision processes. Cambridge university press, 2016.

M. G. Lagoudakis and R. Parr. Least-squares policy iteration. Journal of machine learning research, 4(Dec):1107–1149, 2003.

A. Lazaric, M. Ghavamzadeh, and R. Munos. Analysis of a classification-based policy iteration algorithm. In ICML-27th International Conference on Machine Learning, pages 607–614. Omnipress, 2010.

W. S. Lovejoy. Some monotonicity results for partially observed markov decision processes. Operations Research, 35(5):736–743, 1987.

P. Milgrom and C. Shannon. Monotone comparative statics. Econometrica: Journal of the Econometric Society, pages 157–180, 1994.

R. Munos and C. Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research, 9(5), 2008.

A. Petrosyan, H. Shahgholian, and N. N. Uraltseva. Regularity of Free Boundaries in Obstacle-Type Problems, volume 136 of Graduate Studies in Mathematics. American Mathematical Society, Providence, Rhode Island, 2012.

W. Polonik. Measuring mass concentrations and estimating density contour clusters-an excess mass approach. The annals of Statistics, pages 855–881, 1995.

W. B. Powell. Approximate Dynamic Programming: Solving the curses of dimensionality, volume 703. John Wiley & Sons, 2007.

F. P. Preparata and M. I. Shamos. Computational geometry: an introduction. Springer Science & Business Media, 2012.

M. L. Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

J. Rissanen. Modeling by shortest data description. Automatica, 14(5):465–471, 1978.

R. T. Rockafellar. Convex analysis, volume 28. Princeton university press, 1997.

H. Scarf. The optimality of (s, s) policies in the dynamic inventory problem. 1960.

S. Shalev-Shwartz and S. Ben-David. Understanding machine learning: From theory to algorithms. Cambridge university press, 2014.

J. E. Smith and K. F. McCardle. Structural properties of stochastic dynamic programs. Operations Research, 50(5):796–809, 2002.

M. J. Sobel. The optimality of full service policies. Operations Research, 30(4):636–649, 1982.

S. Stidham Jr and R. R. Weber. Monotonic and insensitive optimal policies for control of queues with undiscounted costs. Operations research, 37(4):611–625, 1989.

R. S. Sutton, A. G. Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

D. M. Topkis. Minimizing a submodular function on a lattice. Operations research, 26(2):305–321, 1978.

D. M. Topkis. Supermodularity and complementarity. Princeton university press, 1998.

A. B. Tsybakov. On nonparametric estimation of density level sets. The Annals of Statistics, 25(3): 948–969, 1997.

V. N. Vapnik. An overview of statistical learning theory. IEEE transactions on neural networks, 10 (5):988–999, 1999.

A. F. Veinott. Discrete dynamic programming with sensitive discount optimality criteria. The Annals of Mathematical Statistics, 40(5):1635–1660, 1969.

A. F. Veinott Jr. Optimal policy in a dynamic, single product, nonstationary inventory model with several demand classes. Operations research, 13(5):761–778, 1965.

P. Whittle. Optimization over time. John Wiley & Sons, Inc., 1982.

G. M. Ziegler. Lectures on polytopes, volume 152. Springer Science & Business Media, 2012.