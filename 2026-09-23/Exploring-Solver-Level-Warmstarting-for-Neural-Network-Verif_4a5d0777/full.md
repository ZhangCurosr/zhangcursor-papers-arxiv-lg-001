# Exploring Solver-Level Warmstarting for Neural Network Verification

Annelot W. Bosman<sup>1</sup>, Minghao Liu<sup>2</sup>, Marta Kwiatkowska<sup>2</sup>, Holger H. Hoos<sup>1,3,4</sup>, and Jan N. van Rijn<sup>1</sup>

<sup>1</sup>Leiden Institute of Advanced Computer Science, Leiden <sup>2</sup>University of Oxford, United Kingdom <sup>3</sup>Chair for AI Methodology, RWTH Aachen University, Germany <sup>4</sup>University of British Columbia, Canada

## Abstract

Neural network verification has become a key tool for providing formal guarantees on the behaviour of neural networks. However, many verification problems remain computationally intractable in the worst case: even for common adversarial robustness specifications, verification is NP-complete. Here, we explore the application of solver-level warmstarting for neural network verification to exploit information from previous solutions. We study the efect on running time as several properties are modified, including perturbation radii, input data and the networks themselves, using a pipeline that is generalisable and potentially adaptable to state-of-the-art verifiers. Our results show that warmstarting can significantly reduce verification time in most cases. Moreover, warmstarting enables the successful verification of instances that could not be solved from scratch within the given time limit.

## 1 Introduction

Recent advances in computational power and the availability of large-scale data have led to the widespread deployment of neural networks in real-world settings [14]. As neural networks are adopted in safety-critical domains, such as autonomous driving [24] and healthcare [23], the demand for formal guarantees in their robustness has become more urgent, since failures can have severe consequences.

Neural network verification aims to determine whether a given neural network satisfies a specified property over a set of inputs, taking the network, input bounds, and property as inputs and returning either a proof that the property holds or a counterexample demonstrating a violation. Most current neural network verifiers treat each verification task in a stand-alone manner. However, in realistic experimental pipelines and in real-world cases, there are many similar verification instances, and it could be beneficial to use these similarities in the verification process. In the case of adversarial robustness property, it has been common knowledge that networks tend to be misled by similar adversarial examples, which are inputs deliberately modified to exploit model vulnerabilities [11, 13, 17, 22], and it has been recently shown that similar networks also have similar safe radii for the same input [3]. Incremental constraint solving has been explored for neural network verification to avoid solving each problem from scratch [6, 10]. Some work improves cross-instance verification eficiency by processing instances with similar perturbation radii in mini-batches [19] or by exploiting similarities and dependencies between the hidden-layer states of diferent inputs [7, 2]. Recent studies also explore relational verification [1], which leverages dependencies among multiple executions of the same network, and incremental verification [20], which exploits similarities between the original and updated networks to reuse previous verification efort. Overall, prior work utilises specific correlations between instances, but is inherently limited in generality; for example, systematic reuse of information across variations in perturbation radii, input data and networks is not supported.

The verification tasks can be encoded as constraint satisfaction problems and solved using mixed-integer linear programming (MILP) solvers. Warmstarting techniques for MILP have been studied extensively in the mathematical optimisation literature [15]. Given a MILP instance with modified parameters or constraints, warmstarting MILP solvers brings gains in eficiency by reusing the information from previous runs on similar instances.

In this paper, we incorporate MILP warmstarting techniques into neural network verification. We focus on reusing previously obtained solver artefacts, such as previous solutions or the information on search tree, to initialise the solver for previously unseen verification instances. This setup enables warmstarting across a wide range of instance variations, largely independently of how the underlying verification property is modified (e.g., modified perturbation radius), although certain scenarios are expected to benefit more than others due to better similarity and reusable lower bounds. Specifically, we combine components from existing verification software with additional interfacing code to construct a pipeline that enables warmstarting at the solver level. We use SYMPHONY [16], an open-source MILP solver that provides direct support for warmstarting techniques, to establish a controlled verification setting and systematically investigate the impact of solver-level warmstarting. However, SYMPHONY has not been benchmarked against state-of-the-art MILP solvers for neural network verification, and thus our baseline should not be considered state-of-the-art. Instead, our goal is to isolate and evaluate the benefits of solver-level warmstarting before integrating these techniques into state-of-the-art verifiers.

The main contributions of this paper are as follows:

• To the best of our knowledge, we are the first to study warmstarting for diferent properties for neural network verification directly at the MILP solver level, in contrast to existing approaches [2, 20] that incorporate warmstarting mechanisms at the verifier level. We provide a systematic analysis of how diferent types of property changes afect the suitability and efectiveness of solver-level warmstarting.

• We investigate multiple warmstarting scenarios and strategies, analysing their impact across diferent property changes. Our experimental results demonstrate that this approach can yield significant performance improvements, both in terms of reduced running time and in the ability to solve verification instances that could not be solved without warmstarting within a given time limit. Importantly, these improvements are observed not only for small property changes, such as changes in the perturbation radius ε, but also for diferent inputs and neural networks.

• We present a proof-of-concept pipeline and empirical evaluation of solver-level warmstarting using the SYMPHONY solver [16], demonstrating its potential benefits and limitations in practice.

Our experimental code and results can be found here: https://github.com/ADA-research/STAI-Solver-Level-Warmstarting-paper.

## 2 Problem Specification

In this section, we first introduce the formalisation of neural networks and the verification tasks at hand. Next, we outline the scenarios we considered in which warmstarting would be beneficial in verification.

## 2.1 Verification of Neural Networks

A deep neural network classifier is a function $f _ { \theta } : \mathbb { R } ^ { n _ { 0 } }  \mathbb { R } ^ { n _ { L } }$ , where $\theta$ denotes the trainable parameters, $n _ { 0 }$ is the input dimension, and $n _ { L }$ is the number of output classes. The network consists of L layers, with layer l containing $n _ { l }$ neurons.

In the following, we consider fully connected neural networks (FCNNs), where the structure of layer l can be formalised as:

$$
x _ { l } = \sigma \left( W _ { l } \cdot x _ { l - 1 } + b _ { l } \right) ,\tag{1}
$$

where $W _ { l } \in \mathbb { R } ^ { n _ { l } \times n _ { l - 1 } }$ and $b _ { l } \in \mathbb { R } ^ { n _ { l } }$ are parameters, and $\sigma$ is an activation function, which in our study we assume to be the widely used ReLU function $\sigma ( x ) =$ max(x, 0).

Formal verification utilises mathematically rigorous methods to determine whether a neural network satisfies a specific property for all inputs within a given input region. Assume a feasible region G and a property $P$ in the form of a Boolean formula over linear inequalities. Following [5], formal verification aims to decide whether

$$
\forall x _ { 0 } \in G : \quad ( x _ { L } = f _ { \theta } ( x _ { 0 } ) ) \implies P ( x _ { L } ) ,\tag{2}
$$

where $x _ { L }$ is the output vector of $f _ { \theta }$ . In this work, we are concerned with the adversarial robustness property, which is a key property for neural networks and has been extensively studied in the literature [12].

Given an original input $x ^ { * }$ with correct label $\lambda ( x ^ { * } )$ , we consider the feasible region is defined by an $L _ { \infty }$ -norm ball centred at $x ^ { * }$ with a radius of $\varepsilon { \mathrm { : } }$

$$
G _ { \varepsilon } ( x ^ { * } ) = \left\{ x _ { 0 } : | x ^ { * } - x _ { 0 } | _ { \infty } \leq \varepsilon \right\} ,\tag{3}
$$

where $\left| x \right| _ { \infty } = \operatorname* { m a x } _ { 1 \leq i \leq n } \left| x _ { i } \right|$ . A neural network $f _ { \theta }$ is said to be robust at $x ^ { * }$ if, and only if, $\forall x _ { 0 } \in G _ { \varepsilon } ( x ^ { * } )$ : arg max<sub>i</sub> $f _ { \theta } ( x _ { 0 } ) [ i ] = \lambda ( x ^ { * } )$

The robustness verification of FCNNs can be formulated as the following mixedinteger linear programming (MILP) problem:

max y

$$
\mathrm { s . t . } \quad y = \operatorname* { m a x } _ { 1 \leq i \leq n _ { L } , i \neq \lambda ( x ^ { * } ) } x _ { L } [ i ] - x _ { L } [ \lambda ( x ^ { * } ) ]\tag{4a}
$$

$$
\hat { x } _ { l } = W _ { l } \cdot x _ { l - 1 } + b _ { l }\tag{4b}
$$

$$
\forall l \in [ 1 , L ]\tag{4c}
$$

$$
x _ { l } = \operatorname* { m a x } \left( \hat { x } _ { l } , 0 \right)
$$

$$
\forall l \in [ 1 , L ]\tag{4d}
$$

$$
x ^ { * } - \varepsilon \leq x _ { 0 } \leq x ^ { * } + \varepsilon ,\tag{4e}
$$

where $x _ { 1 } , \ldots , x _ { L }$ are vectors of real-valued variables. Note that the max functions can be represented exactly with linear constraints using the big-M method [18]. A neural network is robust at $x ^ { * }$ with perturbation radius ε if, and only if, the optimal value of Objective (4a) is less than 0, which corresponds to the satisfaction of the property in Eq. (2).

## 2.2 Scenarios for Warmstarting

In this paper, we always consider two properties: the original property $P _ { x }$ , with an associated solution $A _ { x }$ (whose concrete form we discuss later), and a new property $P _ { z }$ . There are multiple ways in which $P _ { x }$ and $P _ { z }$ may be related to one another, and understanding these relationships is key to determining how $A _ { x }$ can be reused when verifying $P _ { z }$

## 2.2.1 Diferent ε.

A first way in which $P _ { z }$ may difer from $P _ { x }$ is through a change in the perturbation radius. This change is relevant when the feasible region of the verification problem depends on a predefined $L _ { \infty } .$ -norm ball with a radius of ε. Given the same reference input $x ^ { * }$ , we now consider two feasible region sets.

$$
G _ { \varepsilon _ { x } } ( x ^ { * } ) , G _ { \varepsilon _ { z } } ( x ^ { * } ) \quad { \mathrm { w i t h } } \quad \varepsilon _ { z } \neq \varepsilon _ { x } .
$$

This change only has an efect on Constraint (4e), as this encodes the feasible region of the modified input. While this is the simplest type of change we consider, it is particularly relevant for finding the largest safe perturbation radius for an input to a given network. Running time tends to become prohibitively expensive near the decision boundary [3], so reducing running time on previously unseen instances can help obtain a clearer picture of the robustness of a given network.

We consider two warmstarting scenarios. If $\varepsilon _ { z } < \varepsilon _ { x } ,$ then $G _ { \varepsilon _ { z } } ( x ^ { * } ) \subseteq G _ { \varepsilon _ { x } } ( x ^ { * } )$ and the feasible region for $P _ { z }$ is a strict subset of that for $P _ { x }$ . In such cases, all artefacts in solution $A _ { x }$ remain valid, although they could be an over-approximation.

Conversely, if $\varepsilon _ { z } > \varepsilon _ { x }$ , the new perturbation region expands, potentially invalidating some parts of $A _ { x }$ . Cuts and bounds derived under $P _ { x }$ may no longer hold, and branch-and-bound nodes previously ruled out might need to be reopened. This scenario, therefore, provides a natural test of robustness for warmstarting strategies. We note that, in the trivial case where $\varepsilon _ { z } > \varepsilon _ { x }$ and $P _ { x }$ was violated, $P _ { z }$ will also always be violated.

## 2.2.2 Diferent $x ^ { * }$

Another change in property arises when the reference input is replaced by a new input $z ^ { * } \neq x ^ { * }$ . The verification property becomes

$$
{ \cal P } _ { z } : \quad \exists z _ { 0 } \in G _ { \varepsilon } ( z ^ { * } ) \quad \mathrm { s . t . ~ a r g } \operatorname* { m a x } _ { i } f _ { \theta } ( z _ { 0 } ) [ i ] \neq \lambda ( z ^ { * } ) .
$$

This change in property afects Constraints (4c) and $\mathrm { ( 4 e ) }$ ; as a result, some artefacts in $A _ { x } \ ( \mathrm { e . g . }$ , neuron phase assignments, linear relaxations, interval bounds) may no longer be valid without modification.

Nonetheless, $P _ { x }$ and $P _ { z }$ can still be closely related when $z ^ { * }$ lies near $x ^ { * }$ , for example, when they are from the same class. In this case, much of the internal activation pattern may be preserved, and several components of $A _ { x }$ can be meaningfully warmstarted for $P _ { z }$ after appropriate validity checks.

## 2.2.3 Diferent $f _ { \theta } .$

A more substantial change occurs when network parameters or architectures are modified. If the original model is $f _ { \theta }$ and the new model $f _ { \theta ^ { \prime } } ^ { \prime }$ with $\theta ^ { \prime } \neq \theta$ , and $f$ is allowed to be diferent from $f ^ { \prime }$ , the property becomes

$$
\begin{array} { r } { P _ { z } : \quad \exists x _ { 0 } \in G _ { \varepsilon } ( x ^ { * } ) \quad \mathrm { s . t . ~ a r g } \operatorname* { m a x } _ { i } f _ { \theta ^ { \prime } } ^ { \prime } ( x _ { 0 } ) [ i ] \neq \lambda ( x ^ { * } ) . } \end{array}
$$

In this case, Constraint (4c) may be afected, which concerns the vast majority of the encoding; even small perturbations to weights or biases can change activation patterns and thus alter the feasible region of the verification problem.

If the change in parameters is small (e.g., fine-tuning, pruning or lightweight retraining), many artifacts in $A _ { x }$ may remain approximately valid. In such cases, warmstarting remains possible but requires conservative checks or adjustments (e.g., inflating abstract bounds, re-validating cuts). However, if the network structure changes significantly (e.g., new layers, diferent activations), then $A _ { x }$ may no longer be reusable.

## 3 Methodology

In this section, we first introduce the core theory and algorithms of warmstarting in MILP solving. Next, we describe our approach of applying warmstarting techniques in MILP-based neural network verification.

## 3.1 Warmstarting in MILP Solving

Consider a MILP instance

$$
z ( b ) : = \operatorname* { m i n } _ { x \in \Pi ( b ) } c ^ { \top } x , \qquad \Pi ( b ) = \left\{ x \in \mathbb { Z } ^ { p } \times \mathbb { R } ^ { n - p } \ : | \ : A \cdot x = b , x \geq 0 \right\} ,
$$

where x is a vector of n variables, with $p$ of them being integers, and $c , A , b$ are parameters.

To determine the optimal value of $z ( b )$ , a common procedure is the branch-andbound algorithm, which constructs a search tree with node set V by recursively partitioning the feasible region into subproblems. Each node $i \in V$ of the tree corresponds to a restricted feasible region $\Pi _ { i } ( b ) \subseteq \Pi ( b )$ by imposing additional bounds on integer variables. At each node, a lower bound $l _ { i } ( b )$ is obtained by solving the following linear programming (LP) relaxation $l _ { i } ( b ) : = \mathrm { m i n } _ { x \in P _ { i } ( b ) } c ^ { \top } x$ with the simplex algorithm [4], where $P _ { i } ( b )$ is a polyhedral relaxation of $\Pi _ { i } ( b )$ An optimal basis $B _ { i } { \mathrm { : } }$ , consisting of a set of linearly independent columns in A, together with the assignments of the non-basic variables fixed at their bounds, is determined. The global lower bound can be maintained by $L = \operatorname* { m i n } _ { i \in V } l _ { i } ( b )$

Consider a MILP with modified right-hand side d and objective $z ( d )$ . The idea of warmstarting is to solve this instance by using artefacts (e.g., the search tree, lower bounds, and cutting planes) obtained during the solution of $z ( b )$ . In this paper, we introduce the warmstarting techniques proposed by [15], which provide a dual interpretation of branch-and-bound. At a given node i, suppose the LP relaxation admits an optimal basis $B _ { i } ;$ then the optimal LP value can be expressed as an afine function of the right-hand side:

$$
l _ { i } ( d ) = c _ { B _ { i } } ^ { \top } \cdot B _ { i } ^ { - 1 } \cdot d + \beta _ { i } ,
$$

where $c _ { B _ { i } }$ denotes the components of $c$ corresponding to the columns of $B _ { i }$ , and $\beta _ { i }$ represents the constant factors associated with the non-basic variables. This is a valid dual bound for the MILP, since for any $d , \ l _ { i } ( d ) \ \leq \ z ( d )$ . Therefore, the piecewise linear function $F ( d ) = \mathrm { m i n } _ { i \in V } l _ { i } ( d )$ represents a global lower bound of $z ( d )$ and can be used to warmstart the branch-and-bound tree for any new d. Moreover, the pool of global cutting planes is also kept and can be reused for diferent values of d.

We note that, when A is modified, the artefacts that can be reused to solve the new MILP problem are limited. The search tree structure can be kept, but the dual bounds and global cuts may no longer be valid, and the LP relaxation needs to be recomputed. Thus, the efectiveness of warmstarting is reduced compared to modifying the parameters on the right-hand side.

## 3.2 Neural Network Verification with Warmstarting

In this work, we use the SYMPHONY MILP solver [16] to perform warmstarting, since, as far as we are aware, it is the only open-source MILP solver with native warmstarting support<sup>1</sup>. While SYMPHONY has primarily been applied in the context of operations research, it has not, to the best of our knowledge, yet been explored for neural network verification. Importantly, SYMPHONY provides native support for two warmstarting methods that are particularly relevant in our setting: (i) reusing the branch-and-bound search tree (warmstart tree), and (ii) reusing the pool of cutting planes generated for a previous instance.

In order to use SYMPHONY for neural network verification, we need to generate an MPS file containing the MILP formulation of the network and the property to be verified. The complete pipeline of our MPS generation process is depicted in Figure 1.

To generate this formulation, we build on the Marabou verifier [8], which can encode neural network verification problems as MILPs by interfacing with the Gurobi MILP solver. Marabou takes as input a neural network and a corresponding VNN-LIB specification file that encodes the input constraints and the property to be verified. In our workflow, these VNN-LIB files are generated using VERONA [3], which we use as our experiment manager (see Section 4). We modified Marabou to terminate the process after Gurobi exports the corresponding MPS formulation, rather than solving the instance directly. Connecting all these tools to work together smoothly required substantial changes. On the VERONA side, we created an interface for SYMPHONY and Marabou; on the Marabou side, we made sure that MPS files could be retrieved without interfering with the C compiler. For additional details on how we configured and changed these tools, we refer the interested reader to Appendix A.

Modern neural network verifiers represent the result of substantial engineering efort and incorporate a wide range of advanced techniques, including sophisticated branching heuristics, tight relaxations, and parallel solving strategies. While we believe that such state-of-the-art verifiers could also benefit from warmstarting, directly integrating and evaluating warmstarting strategies within these systems would require significant additional engineering efort and may obscure the fundamental efects of warmstarting itself. Instead, we adopt SYMPHONY as a controlled proof-of-concept solver that allows us to systematically study warmstarting behaviour in isolation. This separation of warmstarting from verifier-specific optimisations enables a more transparent analysis of when and why warmstarting is beneficial.

![](images/251dd06f1aa5b77d1b2f8d23f5f37fb6beb71ef112913080a4f5819c181d40c8.jpg)  
Figure 1: The pipeline created to formulate the verification tasks as MPS files that can be used as inputs for the SYMPHONY solver.

## 3.3 Reformulation Algorithm

A practical challenge is that the exported MPS formulation contains indicator constraints, which are not supported by SYMPHONY. To address this issue, we introduce a reformulation procedure that converts indicator constraints into an equivalent MILP formulation, enabling SYMPHONY to read and solve the resulting instance.

The original purpose of the indicator constraints in our MPS formulation is to encode the classification decision of the network, i.e., to enforce that exactly one output class is selected. Listing 1 provides a simplified example illustrating how these indicator constraints appear in the exported MPS file.

The mathematical interpretation of the indicator constraint in Listing 1 is shown in Eq. (5). In this formulation, the output variable for a particular class, $x _ { 1 }$ , is controlled by a binary variable $a _ { 1 }$ . In the full verification problems, multiple classes are present, and an additional constraint ensures that exactly one class is predicted, typically by enforcing a one-hot structure over the binary variables (e.g., $\begin{array} { r } { \sum _ { i \in n _ { L } } a _ { i } = n _ { L } - 1 \big ) } \end{array}$ ). We omit these additional constraints here for simplicity.

$$
x _ { 1 } = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ } } a _ { 1 } = 1 , } \\ { \geq 0 , } & { { \mathrm { i f ~ } } a _ { 1 } = 0 . } \end{array} \right. }\tag{5}
$$

1 ROWS :   
2 L GC0   
3 COLUMNS :   
4 x1 GC0 1   
5 RHS :   
6 RHS GC0 0   
7 BOUNDS :   
8 BV BND1 a1   
9 UP BND1 x1 m0   
10 LO BND1 x1 0   
11 INDICATORS :   
12 IF GC0 a1 1   
13 ENDATA  
Listing 1: Example MPS formulation containing an indicator constraint.

1 ROWS :   
2 L GC0   
3 G N1   
4 COLUMNS :   
5 x1 GC0 1   
6 x1 N1 1   
7 a1 GC0 m0   
8 RHS :   
9 RHS GC0 m0   
10 RHS N1 0   
11 BOUNDS :   
12 BV BND1 a1   
13 UP BND1 x1 m0   
14 LO BND1 x1 0   
15 ENDATA  
Listing 2: Example MPS formulation containing a big-M formulation

We reformulate the indicator constraint using big-M constraints, as shown in Eq. (6) and (7). Moreover, we select M as the upper bound of the corresponding variable, i.e., $M = m _ { i }$ . This choice yields the smallest valid value of M while ensuring that Eq. (6) and (7) remain correct reformulations of Eq. (5). The resulting MPS file after reformulation is shown in Listing 2.

$$
x _ { 1 } + M \cdot a _ { 1 } \leq M\tag{6}
$$

$$
x _ { 1 } \geq 0\tag{7}
$$

Note that the performance of the MILP solver can depend on the tightness of the big-M encoding. Tighter variable bounds can strengthen the linear relaxation and reduce search efort. Since state-of-the-art verifiers typically employ more advanced bound-tightening techniques than those considered here, their MILP formulations may exhibit diferent solving behaviour. The impact of such tighter formulations on the efectiveness of warmstarting remains an interesting direction for future work.

The entire reformulation procedure is shown in Algorithm 1.

Algorithm 1 MPS Reformulation: INDICATORS → big-M constraints   
Require: Input MPS file $F _ { \mathrm { i n } }$ containing an INDICATORS section   
Ensure: Output MPS file $F _ { \mathrm { o u t } }$ containing no indicator constraints   
1: Parse $F _ { \mathrm { i n } }$ into list of lines S   
2: Identify section boundaries in S (e.g., ROWS, COLUMNS, RHS, BOUNDS,   
INDICATORS, ENDATA)   
3: $c \gets 0$   
4: for all indicator constraints of the form IF $( r , a )$ in INDICATORS do   
5: ▷ r: afected row, a: binary indicator variable   
6: Determine the afected continuous variable x associated with row r   
7: by scanning COLUMNS   
8: Extract an upper bound M for x from the BOUNDS section   
9: if no valid upper bound exists then   
10: Set M ← 100 ▷ fallback used in our implementation   
11: end if   
12: RHS adjustment: update RHS entry for r as $b _ { r } \gets b _ { r } + M$   
13: Column insertion: add a coeficient M for variable a in row r   
14: Row type update: set the type of r to L $( \mathrm { i . e . , \leq } )$   
15: Non-negativity constraint:   
16: Create a new row $N _ { c }$ of type G $( \mathrm { i . e . , \geq ) }$   
17: Add RHS entry $b _ { N _ { c } } \gets 0$   
18: Insert coeficient 1 for variable x in row $N _ { c }$   
19: $c \gets c + 1$   
20: end for   
21: Remove the entire INDICATORS section from S   
22: Write the modified lines S to $F _ { \mathrm { o u t } }$

## 4 Setup of Experiments

In this section, we describe the experimental setups, including details on how we selected the source and target instances for warmstarting in our empirical study.

## 4.1 Network and Image Selection

We selected three fully connected neural networks, mnist-net, mnist-net\_256x2, and mnist-net\_256x4, commonly used in the neural network verification literature. These networks are trained on the MNIST dataset [9] and are relatively small, allowing us to keep computational costs manageable while studying the efect of warmstarting.

<table><tr><td>Network</td><td>SAT UNSAT</td><td>ERROR TIMEOUT</td></tr><tr><td>mnist-net</td><td>10 104</td><td>2 144</td></tr><tr><td>mnist-net 256x2</td><td>26</td><td>118 1 138</td></tr><tr><td>mnist-net 256x4</td><td>0</td><td>63 0 192</td></tr></table>

Table 1: Details on the number of verification instances per network, grouped by solver outcome (SAT, UNSAT, ERROR, and TIMEOUT).

We randomly selected 10 test images from the MNIST dataset. Note that not all selected images were used for every network, since we only considered instances that were classified correctly by the corresponding network.

## 4.2 Baseline Instance Generation

To generate baseline results for each image–network combination to which we can compare our method, we first performed a coarse search over the perturbation radius ε. Specifically, we did grid search [3] for $\varepsilon \in [ 0 . 0 0 1 , 0 . 4 ]$ with a step size of 0.02, resulting in 20 candidate values. Next, we refined the search around the transition region between UNSAT and SAT by performing two iterative searches with a step size 0.002.

We refine the boundary from both sides: starting from the largest value for which the property still held, we increased ε, and starting from the smallest value for which the property did not hold, we decreased ε, in both cases using a step size of 0.002 until a timeout is reached. This staged procedure allowed us to obtain a denser set of baseline measurements near the decision boundary while keeping the total computational cost manageable, because each individual verification query was subject to a 1 wall-clock hour timeout, and we observed a substantial number of timeouts – particularly in the region where instances transition from UNSAT to SAT. Table 1 shows the number of generated baseline instances for each solver outcome.

## 4.3 Warmstarting Instance Selection

In this work, we investigate six diferent warm-starting scenarios. Each scenario consists of a source instance and a target instance. The source instance is solved first and results in either a SAT or UNSAT outcome, while the target instance is, in principle, unsolved at the moment warmstarting is applied. In practice, however, we solve all instances independently as well, in order to obtain a baseline running time for each verification task, in order to be able to quantify the potential benefit of warmstarting. Each of these warmstarting scenarios is discussed in Section 2.2. We now describe how we select the diferent instances considered for warmstarting.

For each network–image combination, we consider the set of perturbation radii $\varepsilon$ for which the baseline verification outcome is UNSAT and the set for which the outcome is SAT. If there are at least two UNSAT instances, we generate a warmstart experiment for every unique pair of UNSAT radii $\left( \varepsilon _ { a } , \varepsilon _ { b } \right)$ , such that $\varepsilon _ { a } <$ $\varepsilon _ { b }$ . Each pair defines a source and target instance taken from the corresponding MILP encodings. We repeat the same procedure for the SAT set: if at least two SAT instances exist, we create a warmstart experiment for every unique pair $\left( \varepsilon _ { a } , \varepsilon _ { b } \right)$ such that $\varepsilon _ { a } > \varepsilon _ { b }$

After this, again for each network–image combination, we construct a sorted list of all evaluated perturbation radii ε. For every ε value that corresponds to a timeout, we select a source instance by finding the nearest ε for which a solved baseline result (SAT or UNSAT) is available. We then pair this solved instance with the timeout instance at the original ε, creating a warmstart experiment where the solved instance acts as the source and the timeout instance acts as the target.

For a fixed network $f$ and target image x, we iterate over the available ε values for $( f , x )$ . For each $\varepsilon ,$ we select a source instance from the same network f but a diferent image $x ^ { \prime } \neq x ,$ provided that an instance for $( f , x ^ { \prime } , \varepsilon )$ exists. We then create a warm-started experiment in which the target instance is $( f , x , \varepsilon )$ and the source instance is $( f , x ^ { \prime } , \varepsilon )$ , and we label this experiment as IMAGE.

To construct network warmstart experiments, we instead fix an image x and an ε value, and collect all instances across networks that correspond to $( x , \varepsilon )$ . If at least two diferent networks provide an instance for the same $( x , \varepsilon )$ pair, we form an experiment by pairing the first two such instances (with identical $( x , \varepsilon )$ but diferent networks). We label this experiment as NET.

## 4.4 Execution Environment

All experiments were carried out on a cluster of machines equipped with Intel Xeon Gold 6252 @ 2.10GHz CPUs with 96 cores, 16GB cache size and 252GB RAM, running Ubuntu 24.04 OS.

<table><tr><td>Network</td><td>Category</td><td>#Inst</td><td>Better</td><td>Worse</td><td>Extra</td><td>ERR</td></tr><tr><td rowspan="3">mnist-net</td><td>IMAGE</td><td>79</td><td>36</td><td>12</td><td>12</td><td>0</td></tr><tr><td>ε:U-TO</td><td>30</td><td>30</td><td>0</td><td>30</td><td>0</td></tr><tr><td>ε:U-U</td><td>968</td><td>786</td><td>9</td><td>0</td><td>24</td></tr><tr><td rowspan="4">mnist-net_256x2</td><td>IMAGE</td><td>113</td><td>88</td><td>16</td><td>28</td><td>0</td></tr><tr><td>ε:S-S</td><td>120</td><td>4</td><td>116</td><td>0</td><td>0</td></tr><tr><td>ε:S-TO</td><td>21</td><td>21</td><td>0</td><td>21</td><td>0</td></tr><tr><td>ε:U-U</td><td>648</td><td>564</td><td>12</td><td>0</td><td>35</td></tr><tr><td rowspan="3">mnist-net 256x4</td><td>IMAGE</td><td>39</td><td>26</td><td>2</td><td>12</td><td>0</td></tr><tr><td>ε:U-TO</td><td>140</td><td>140</td><td>0</td><td>140</td><td>0</td></tr><tr><td>ε:U-U</td><td>164</td><td>94</td><td>2</td><td>0</td><td>8</td></tr><tr><td></td><td>NET</td><td>133</td><td>54</td><td>43</td><td>12</td><td>2</td></tr></table>

Table 2: Experimental results obtained from using SYMPHONY for warmstarting neural network verification. The results are split per network and category, where we summarised the category names using S(SAT), U (UN-SAT), and TO(Timeout). We report the number of instances that could not be solved before using warmstarting (Extra); we also report the number of instances for each category that led to an error when using warmstarting (ERR), the total number of instances without considering instances that led to an error (#Inst) and the number of instances that were solved in less running time including extra instances (Better).

## 5 Results

In this section, we present the results of using SYMPHONY to warmstart new MILP formulations that are slight to substantial modifications of previously solved verification instances. The absolute number of evaluated instances, as well as statistics such as the number of warmstart errors and the number of instances with reduced running time, are reported in Table 2.

Moreover, Table 3 reports the running times measured in our experiments.

<table><tr><td>Network</td><td>Category</td><td> $\#$ </td><td>inst Baseline (s) Warm (s)</td><td></td><td>∆t (%)</td></tr><tr><td rowspan="3">mnist-net</td><td>IMAGE</td><td>79</td><td> $1 3 3 3 \pm 1 4 9 1$ </td><td> ${ \bf 2 1 8 } \pm { \bf 4 3 8 }$ </td><td>-83.60</td></tr><tr><td>ε: U-TO</td><td>30</td><td> $3 6 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 8 7 7 \pm 2 5 5 }$ </td><td>-75.60</td></tr><tr><td>ε: U-U</td><td>968</td><td> $5 3 8 \pm 8 5 9$ </td><td> ${ \bf 1 6 } \pm { \bf 7 5 }$ </td><td>-97.00</td></tr><tr><td rowspan="4">mnist-net_256x2</td><td>IMAGE</td><td>113</td><td> $1 6 3 5 \pm 1 6 1 9$ </td><td> ${ \bf 2 9 5 \pm 4 3 5 }$ </td><td>-82.00</td></tr><tr><td>ε: S-S</td><td>120</td><td> $\mathbf { 1 4 2 } \pm \mathbf { 2 0 6 }$ </td><td> $7 3 1 \pm 3 6 5$ </td><td>415.50</td></tr><tr><td>ε: S-TO</td><td>21</td><td> $3 6 0 0 \pm 0 . 0 0$ </td><td> ${ \bf 5 1 2 } \pm { \bf 3 8 8 }$ </td><td>-85.80</td></tr><tr><td>ε: U-U</td><td>648</td><td> $7 3 0 \pm 9 1 2$ </td><td> ${ \bf 8 3 \pm ~ 2 2 7 }$ </td><td>-88.60</td></tr><tr><td rowspan="3">mnist-net_256x4</td><td>IMAGE</td><td>39</td><td> $1 8 0 4 \pm 1 6 3 2$ </td><td> ${ \bf 6 5 \pm 1 4 2 }$ </td><td>-96.40</td></tr><tr><td>ε: U-TO</td><td>140</td><td> $3 6 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 3 4 1 } \pm \mathbf { 3 7 7 }$ </td><td>-90.50</td></tr><tr><td>ε: U-U</td><td>164</td><td> $5 0 3 \pm 6 6 4$ </td><td> ${ \bf 2 1 \pm 8 8 }$ </td><td>-95.80</td></tr><tr><td></td><td>NET</td><td>133</td><td> $9 7 8 \pm 1 2 9 9$ </td><td> $\mathbf { 3 5 0 \ : \pm { \ : 5 0 2 } }$ </td><td>-64.20</td></tr></table>

Table 3: Empirical results obtained from using SYMPHONY for warmstarting neural network verification. The results are shown per network and type of warmstarting (e.g., warmstarting across images, networks and values for ε), where we further split the experiments involving warmstarting across ε-values into variations of S(SAT), U (UNSAT), TO(Timeout). We report the average baseline and warmstarted running time in seconds; we also report the relative change in running time as percentages.

## 5.1 Diferent ε.

We first analyse the categories in which the only diference between the source and target instances is the magnitude of ε. Using warmstarting in the category with source and target instance both resulting in UNSAT results, abbreviated as the UNSAT-UNSAT category, led to a substantial reduction in running time, of up to 97%. In addition, we were able to solve 170 instances that timed out in the baseline experiments. Figure 2a further illustrates this reduction in running time. This category provides great potential in reducing the computational burden of certifying robustness against perturbations.

For the baseline MILP instances, we observe relatively few SAT results; consequently, the number of warmstarted MILP instances with a SAT source instance is small (141), compared to 2021 instances with an UNSAT source instance. Interestingly, most baseline MILP instances that are expected to be SAT result in timeouts.

Although SAT instances are generally considered easier to solve in state-of-the-art neural network verifiers, these verifiers often employ specialised techniques, such as adversarial attacks, to eficiently search for counterexamples. SYMPHONY, as a general-purpose MILP solver, does not incorporate such techniques. Therefore, SYMPHONY may not eficiently locate an adversarial example even when one exists, which may explain the timeouts observed for these instances.

The SAT–SAT category is the only category in which we observe an increase in running time when using warmstarting with SYMPHONY under the default warmstarting configuration of the tool; this can also be observed from Figure 2b. This suggests that warmstarting from a smaller ε-value is not beneficial. However, this observation should be interpreted with caution, as these SAT instances already require relatively short running times. Indeed, in the SAT–TIMEOUT category, warmstarting enables solving target instances that previously timed out.

## 5.2 Diferent $x ^ { * }$

For the category in which the source and target instances difer only in the input image, while sharing the same network and perturbation radius ε (denoted as IMAGE), we observe a substantial potential reduction in running time when using warmstarting: averaged over all networks we have considered, warmstarting leads to a reduction in running time of 85%. This improvement is also evident from Figure 2c, which notably shows a large number of instances that timed out in the baseline experiments but are successfully solved when using warmstarting.

Notably, this reduction is achieved even though changing the input image alters the left-hand side of the MILP formulation, constituting a substantial modification of the optimisation problem. In this work, we do not employ bound propagation techniques, which in state-of-the-art verifiers would further influence neuron bounds across diferent inputs; this is an important consideration when transferring these insights to modern verification frameworks. Nevertheless, our results indicate that warmstarting can be highly efective for this category.

## 5.3 Diferent $f _ { \theta } .$

Although we did not initially expect warmstarting across diferent neural networks to be beneficial, we nevertheless observed an average reduction in running time of 64%. As shown in Figure 2d, this reduction is largely driven by instances that previously timed out in the baseline experiments but can be solved when using warmstarting. For the remaining instances, the points are distributed relatively evenly around the diagonal, indicating that warmstarting benefits some MILP instances based on diferent networks, while increasing running time for others. Identifying which MILP features lead to performance improvements is an interesting direction for future work.

![](images/dc820f9263e85911067a3742ef87c48a4ba59693598f6b1c3d628db0805fa0b2.jpg)  
(a) Warmstart for diferent ε on UNSAT source and UNSAT or TIMEOUT target instances.

![](images/0d7be1dd0380bda4240a368cd963b5771c5606782852920d6ab621d573feb768.jpg)  
(b) Warmstart for diferent ε on SAT source and SAT or TIMEOUT target instances.

![](images/43e3b9b1205ecda322c87adf046ac4c5698f8eb629bc638f0cdfc5238eca6843.jpg)

![](images/2cdd688fb05e1cbbc1ff16d261df724db80ceb21a55f1aa64299e6302b06fe82.jpg)  
(c) Warmstart for the same ε and network, but diferent images as source and target instances.  
(d) Warmstart for the same ε and image, but diferent networks as source and target instances.  
Figure 2: Scatter plots comparing the running times of baseline MILP instances and the corresponding warmstarted instances. Warmstarted instances that were solved normally with UNSAT and SAT source are coloured blue in the top two figures, and in the bottom two figures, we diferentiate the SAT and UNSAT instances by colouring the SAT instances orange. Instances that were previously leading to a timeout error but now could be solved are coloured red, and instances that led to a segmentation error in SYMPHONY occurring are coloured green. All instances below the diagonal line benefit from warmstarting.

## 6 Conclusions and Future Work

In this work, we have studied the efectiveness of solver-level warmstarting for neural network verification using MILP solver SYMPHONY. Our results demonstrate that this form of warmstarting can lead to substantial eficiency gains, particularly for more general instance variations, in which properties difer beyond minor parameter changes. Most notably, for instance pairs in which both the source and target instances are UNSAT (i.e., both are verified to be robust), we observe reductions of up to 97% in average running time, and, across all investigated instance pair categories, warmstarting enables the successful solving of 300 out of 474 instances that previously timed out. At the same time, our analysis highlights important limitations: warmstarting is not beneficial in all cases, and, in 301 out of 1128 individual cases, it even increased running time by on average 364%.

These findings suggest several promising directions for future work. In particular, it would be valuable to investigate how warmstarting, as explored here, can be integrated into state-of-the-art neural network verifiers (e.g., α, β-CROWN [21]), which often rely on more advanced branching strategies, tighter relaxations and parallel solving techniques. Using warmstarting in such verifiers may enable more informed initialisation of the algorithm for new instances, potentially also improving verification eficiency, as demonstrated in our study. In addition, future work could explore which features of MILP instances render warmstarting efective, thus enabling the automatic selection of instances for warmstarting.

## Acknowledgements

This research was partially supported by the ELSA mobility grant, a project funded by the European Union, and by an Alexander-von-Humboldt Professorship in AI held by Holger Hoos. Minghao Liu and Marta Kwiatkowska were supported by the EPSRC Prosperity Partnership FAIR (grant number EP/V056883/1) and ELSA: European Lighthouse on Secure and Safe AI project (grant agreement No. 101070617 under UK guarantee).

## References

[1] Banerjee, D., Singh, G.: Relational DNN verification with cross executional bound refinement. arXiv preprint arXiv:2405.10143 (2024)

[2] Banerjee, D., Xu, C., Singh, G.: Input-relational verification of deep neural networks. Proceedings of the ACM on Programming Languages 8(PLDI), 1–27 (2024)

[3] Bosman, A.W., Berger, A., Hoos, H.H., van Rijn, J.N.: Robustness distributions in neural network verification. Journal of Artificial Intelligence Research 83(20), 1–27 (2025)

[4] Dantzig, G.B.: Maximization of a linear function of variables subject to linear inequalities. Activity analysis of production and allocation 13, 339–347 (1951)

[5] De Palma, A., Bunel, R., Desmaison, A., Dvijotham, K., Kohli, P., Torr, P.H., Kumar, M.P.: Improved branch and bound for neural network verification via lagrangian decomposition. arXiv preprint arXiv:2104.06718 (2021)

[6] Elsaleh, R., Davis, L., Wu, H., Katz, G.: Incremental neural network verification via learned conflicts. arXiv preprint arXiv:2603.12232 (2026)

[7] Fischer, M., Sprecher, C., Dimitrov, D.I., Singh, G., Vechev, M.: Shared certificates for neural network verification. In: International Conference on Computer Aided Verification. pp. 127–148. Springer (2022)

[8] Katz, G., Barrett, C., Dill, D.L., Julian, K., Kochenderfer, M.J.: Reluplex: An eficient SMT solver for verifying deep neural networks. In: Proceedings of the 29th International Conference on Computer Aided Verification (CAV 2017). pp. 97–117 (2017)

[9] Lecun, Y., Bottou, L., Bengio, Y., Hafner, P.: Gradient-based learning applied to document recognition. In: Proceedings of the IEEE. pp. 2278–2324. IEEE (1998)

[10] Liu, M., Lu, C.H., Kwiatkowska, M.: Exact verification of graph neural networks with incremental constraint solving. In: International Symposium on Formal Methods. pp. 641–662. Springer (2026)

[11] Liu, Y., Chen, X., Liu, C., Song, D.: Delving into transferable adversarial examples and black-box attacks. arXiv preprint arXiv:1611.02770 (2016)

[12] Meng, M.H., Bai, G., Teo, S.G., Hou, Z., Xiao, Y., Lin, Y., Dong, J.S.: Adversarial robustness of deep neural networks: A survey from a formal verification perspective. IEEE Transactions on Dependable and Secure Computing (2022)

[13] Naseer, M.M., Khan, S.H., Khan, M.H., Shahbaz Khan, F., Porikli, F.: Crossdomain transferability of adversarial perturbations. Advances in Neural Information Processing Systems 32 (2019)

[14] Naudé, W.: Artificial intelligence: Neither utopian nor apocalyptic impacts soon. Economics of Innovation and New Technology 30(1), 1–23 (2021)

[15] Ralphs, T., Güzelsoy, M.: Duality and warm starting in integer programming. In: The proceedings of the 2006 NSF design, service, and manufacturing grantees and research conference. vol. 40 (2006)

[16] Ralphs, T.K., Güzelsoy, M.: The symphony callable library for mixed integer programming. In: The next wave in computing, optimization, and decision technologies, pp. 61–76. Springer (2005)

[17] Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., Fergus, R.: Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199 (2013)

[18] Tjeng, V., Xiao, K.Y., Tedrake, R.: Evaluating robustness of neural networks with mixed integer programming. In: Proceedings of the 7th International Conference on Learning Representations, ICLR. pp. 1–11 (2019)

[19] Tzour-Shaday, S., Drachsler-Cohen, D.: Mini-batch robustness verification of deep neural networks. Proceedings of the ACM on Programming Languages 9(OOPSLA2), 2786–2814 (2025)

[20] Ugare, S., Banerjee, D., Misailovic, S., Singh, G.: Incremental verification of neural networks. Proceedings of the ACM on Programming Languages 7(PLDI), 1920–1945 (2023)

[21] Wang, S., Zhang, H., Xu, K., Lin, X., Jana, S., Hsieh, C.J., Kolter, J.Z.: Beta-CROWN: Eficient bound propagation with per-neuron split constraints for complete and incomplete neural network verification. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 34 (2021)

[22] Waseda, F., Nishikawa, S., Le, T.N., Nguyen, H.H., Echizen, I.: Closer look at the transferability of adversarial examples: How they fool diferent models diferently. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1360–1368 (2023)

[23] Zhou, S.K., Greenspan, H., Davatzikos, C., Duncan, J.S., Van Ginneken, B., Madabhushi, A., Prince, J.L., Rueckert, D., Summers, R.M.: A review of deep learning in medical imaging: Imaging traits, technology trends, case studies with progress highlights, and future promises. Proceedings of the IEEE 109(5), 820–838 (2021)

[24] Zhu, R.J., Wang, Z., Gilpin, L., Eshraghian, J.: Autonomous driving with spiking neural networks. Advances in Neural Information Processing Systems (2024)

## A Tool Configuration Details

In this work, we combine tools from both the neural network verification domain and the optimisation domain, as described in Section 3.2. In the following, we describe the configuration of each tool used in our experimental pipeline.

First, we use VERONA<sup>2</sup> as our experiment manager. We forked the repository and added a VerificationModule that interfaces with both Marabou<sup>3</sup> and SYM-PHONY<sup>4</sup>.

To generate MILP encodings, we use Marabou through its command-line interface with the ––milp flag. We additionally introduced a command-line option that exports the MPS formulation generated by Gurobi and terminates immediately afterwards. This modification allows us to obtain the MPS files required for our experiments, while avoiding changes that would cause unit tests to fail when building Marabou from source.

Finally, we use SYMPHONY to run both the baseline verification experiments and the warmstarting experiments. In both experiments, we use the default solver configuration.