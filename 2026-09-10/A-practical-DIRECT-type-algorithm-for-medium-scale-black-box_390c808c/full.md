# A practical DIRECT-type algorithm for medium-scale black-box global optimization

Linas Stripinis<sup>a</sup>, Remigijus Paulavičius<sup>a,∗</sup>

<sup>a</sup>Vilnius University Institute of Data Science and Digital Technologies, Akademijos 4, Vilnius, Lithuania, LT-08663

## Abstract

The DIRECT algorithm is a deterministic global optimization method known for its versatility and balanced exploration-exploitation strategy. However, DIRECT-type algorithms are primarily efective for low-dimensional problems and often exhibit slow convergence as dimensionality increases, limiting their applicability to more complex optimization tasks. To address this limitation, this paper introduces X-DTC-GL, a novel DIRECT-type algorithm that incorporates dynamic partitioning and hybridization techniques. The dynamic partitioning approach adaptively refines the search space based on local one-dimensional surrogate models, enabling rapid subdivision of promising hyper-rectangles. The hybridization strategy selectively employs a hill-climbing method to exploit promising regions identified by the surrogate models. Extensive experiments on four diverse benchmark suites demonstrate that X-DTC-GL significantly outperforms existing DIRECT-type baselines, achieving improvements of ∼12% in solvability and ∼27% in solution quality. Performance-profile analyses indicate the fastest convergence on up to ∼40% of instances, the best runtime performance on ∼17% of problems, and competitive overall execution times. By improving performance within the partition-based framework, these advances strengthen the algorithm’s competitiveness in state-of-the-art black-box optimization.

Keywords: Black-box optimization, Global optimization, DIRECT algorithm, Dynamic partitioning, Surrogate models, Hybridization,

## 1. Introduction

This paper discusses a single-objective optimization problem with box constraints:

$$
\operatorname* { m i n } _ { \mathbf { x } \in \mathcal { X } } f ( \mathbf { x } ) .\tag{1}
$$

The problem is to minimize the objective function $f ( \mathbf { x } )$ , where $\mathbf { x } \in \mathbb { R } ^ { n }$ and $f : \mathbb { R } ^ { n }  \mathbb { R }$ , subject to the constraints that x lies within a hyper-cube:

$$
\mathcal { X } = \{ \mathbf { x } \in \mathbb { R } ^ { n } : l _ { j } \leq x _ { j } \leq u _ { j } , j = 1 , \ldots , n \} .\tag{2}
$$

Here, the vectors $1 \in \mathbb { R } ^ { n }$ and $\mathbf { u } \in \mathbb { R } ^ { n }$ represent the lower and upper bounds of the search space for the optimization variable x. The f is Lipschitz continuous and can be non-linear, non-convex, multi-modal, and non-diferentiable. The Lipschitz constant of $f$ is unknown. We assume that f can only be evaluated at any given point in the feasible region X . In other words, we cannot access additional information about the objective function, such as its gradient or Hessian. This is known as Black-Box Optimization (BBO), where the structure of f is unknown, unexploitable, or nonexistent.

Black-box optimization problems, where the objective function is evaluated through simulations, experiments, or complex computations, arise in diverse fields such as engineering design (e.g., airfoil shape optimization), machine learning (e.g., hyperparameter tuning), finance (e.g., portfolio optimization), and drug discovery (e.g., molecular structure identification). Due to the diverse nature of these applications, there is a growing need for eficient and robust BBO algorithms that can handle various problem characteristics.

Numerous techniques are available to solve BBO problems. A recent classification [1] categorizes these algorithms into six classes: exact, hillclimbing, trajectory, population, surrogate, and hybrid. For each class, we cite well-known, canonical, and representative algorithms to illustrate the underlying optimization principles rather than provide an exhaustive list of current state-of-the-art methods.

• Exact methods guarantee finding the global optimum within specified tolerances. Examples include branch-and-bound [2, 3, 4, 5] and DIviding RECTangles (DIRECT) [6, 7, 8] methods.

• Hill-climbing methods focus on greedy exploitation using fast-convergent local optimizers like quasi-Newton methods [9, 10], Nelder–Mead simplex [11], and proximal bundle methods [12].

• Trajectory algorithms systematically explore the search space, such as simulated annealing [13].

• Population-based algorithms operate on a set of potential solutions, including genetic algorithms [14] and particle swarm optimization [15, 16].

• Surrogate models replace expensive evaluations with cheaper ones, reducing computational costs, as exemplified by eficient global optimization [17, 18, 19].

• Hybrid algorithms combine components from diferent classes or use optimization and machine learning to find optimal compositions [20, 21, 22, 23].

This paper focuses on DIRECT-type algorithms, a subclass of exact methods. DIRECT is attractive due to its simplicity, ease of implementation, deterministic nature, global convergence guarantees, and minimal hyperparameter tuning [6]. It efectively balances local and global search (exploitation vs. exploration) by sampling multiple points in each iteration dedicated to both aspects [7, 8]. A recent review [24] highlights the versatility of DIRECT-type algorithms in optimizing various applications, including financial portfolios, transportation systems, engineering designs, energy processes, and medical imaging. Some of the most recent applications we summarized in Table 1. While DIRECT-type methods are most commonly employed for low- to moderate-dimensional black-box optimization problems, the reported applications exhibit diverse computational requirements. For example, TMS electric-field optimization required approximately $2 . 5 \times 1 0 ^ { 3 }$ function evaluations [25], whereas car-following model calibration employed a budget of $1 0 ^ { 4 }$ function evaluations [26].

Recent benchmarking studies [32, 33] of state-of-the-art BBO methods across multiple algorithmic paradigms, including several top-performing algorithms in recent CEC competitions (e.g., EA4eig [34], EBOwithCMAR [35], and HSES [36]), as well as methods ranked among the leading performers like L-SHADE variants [37, 38] indicate that DIRECT-type algorithms remain highly strong competitors under specific operational settings. Their performance is particularly competitive when only moderate evaluation budgets are available (e.g., 5000×n function evaluations) and when addressing low-dimensional optimization problems, where they can remain efective even with considerably larger computational budgets. As the problem dimensionality increases to moderate levels, their competitive performance becomes more specific to particular problem classes, notably optimization problems that are both non-separable and multi-modal. Consequently, establishing a significant performance advancement over the most eficient existing DIRECT-type baselines implicitly enhances their overall competitiveness against the broader spectrum of state-of-the-art black-box optimizers.

Table 1: Recent applications of DIRECT-type algorithms in diverse domains.
<table><tr><td>Ref.</td><td>Application</td><td>Algorithm(s)</td><td>n</td></tr><tr><td>2016 [26]</td><td>Car-following model calibration</td><td>DIRECT-SQP</td><td>3-5</td></tr><tr><td>2023 [27]</td><td>Atomic cluster structure search</td><td>tDIRECT</td><td>11-71</td></tr><tr><td>2023 [28]</td><td>Inverse biosensor parameter</td><td>Aggressive DIRECT</td><td>3</td></tr><tr><td>2025 [25]</td><td>estimation TMS electric-field optimization</td><td>DIRECT, DIRECT-1</td><td>6-12</td></tr><tr><td>2025 [29]</td><td>Biodiesel reaction optimization</td><td>DIRECT-1</td><td>4</td></tr><tr><td>2025 [30]</td><td>Autonomous-driving model</td><td>CSD (stochastic DIRECT)</td><td>2</td></tr><tr><td>2025 [31]</td><td>selection Blisk tool-orientation optimization</td><td>DIRECT</td><td>2</td></tr></table>

In general, DIRECT-type algorithms sufer from slow convergence as the dimensionality increases, which restricts their efectiveness on higher-dimensional optimization problems [7].[7]. This slow convergence stems from the need to repeatedly trisect the minimum-containing hyper-rectangle in each dimension to refine the solution, with the convergence rate worsening linearly as dimensionality increases [7]. Additionally, the algorithm explores other hyper-rectangles in each iteration (“global drag”), further increasing the computational cost [39, 40].

Several approaches have been proposed to address these challenges:

1. Improving the pure DIRECT-type framework: Methods like PLOR [41] and DIRECT-l [42] prioritize exploitation, but this can delay

finding the global minimum [7, 43].

2. Hybridization: Combining DIRECT with hill-climbing techniques [44, 45, 27, 46, 47, 48] can improve convergence, but excessive hill-climbing can lead to high computational costs [27, 46].

3. Involving stochastic heuristics: Techniques like the stochastic “zoom-in” (HD-DIRECT) [40] and block coordinate descent with SQP (ABCD) [44] have been explored, but their efectiveness has been questioned [7].

This article introduces X-DTC-GL, a novel DIRECT-type algorithm with two key enhancements to improve convergence as the problem dimensionality increases:

1. Dynamic partitioning of hyper-rectangles: This approach adaptively refines the search space based on local one-dimensional surrogate models, enabling rapid subdivision of promising regions.

2. Hybridization with hill-climbing: This strategy selectively employs a hill-climbing method to exploit promising regions identified by the surrogate models.

X-DTC-GL constructs local surrogate models within hyper-rectangles to identify promising directions and potential, allowing for swift adjustment of solutions and mitigating the computational cost of global search. The algorithm is tested and validated using four diverse benchmark suites with various problem characteristics.

## 1.1. Main contributions and structure of the paper

This paper makes the following contributions:

1. A comprehensive review of techniques for addressing higher-dimensional problems in DIRECT-type algorithms.

2. The development of X-DTC-GL, a novel and eficient DIRECT-type algorithm designed to mitigate the curse of dimensionality.

3. The release of X-DTC-GL as an open-source resource to ensure reproducibility and reusability of the results.

The paper is structured as follows. Section 2 reviews the original DIRECT algorithm and its extensions for higher-dimensional problems. Section 3 presents the X-DTC-GL algorithm in detail. Section 4 describes the experimental setup and results. Section 5 summarizes the findings and discusses future research directions.

## 2. Literature review

## 2.1. Original DIRECT algorithm

The DIRECT algorithm [6] is a deterministic iterative global optimization algorithm that involves four major steps: initialization, selection, sampling, and Subdivision. Once the initialization step is completed, the algorithm selects Potential Optimal Hyper-rectangles (POHs), samples, evaluates, and trisects them in subsequent iterations. Most DIRECT-type extensions and modifications follow the same algorithmic framework, which is summarized in Algorithm 1, while Figure 1 illustrates the process of the algorithm in its initial three iterations on a two-variable problem. We will briefly review each algorithm step in the following paragraphs.

Algorithm 1: Main steps of DIRECT-type algorithms   
Input:   
f: Objective function;   
X : Decision space;   
OPT: structure with optimization options;   
Output:   
$f _ { k } ^ { \mathrm { m i n } } , \mathbf { c } _ { k } ^ { \mathrm { m i n } }$ : Optimal solution and its corresponding point;   
t (time), k (iterations), m (function evaluations): Performance measures;   
1 Initialization: Normalize X to X<sup>¯</sup>. Evaluate f at the center point c<sup>1</sup>. Set   
$f _ { 1 } ^ { \mathrm { m i n } } \gets f ( \mathbf { c } ^ { 1 } )$ and $c _ { 1 } ^ { \mathrm { m i n } }  \mathbf { c } ^ { 1 }$ . Initialize $t , k \gets 1 , m \gets 1$ , and stopping criteria;   
2 while stopping criteria are not satisfied do   
3 Selection: Identify the set $\mathcal { S } _ { k } \subseteq \mathcal { P } _ { k }$ of POHs;   
4 Sampling: For each $\bar { \mathcal X } _ { k } ^ { j } \in { \mathcal S } _ { k }$ evaluate f at newly sampled points;   
5 Subdivision: Each $\bar { \mathcal X } _ { k } ^ { j } \in { \mathcal S } _ { k }$ subdivide (trisect) along all long sides;   
6 Set k ← k + 1 and update $\mathcal { P } _ { k } , f _ { k } ^ { \operatorname* { m i n } } , \mathbf { c } _ { k } ^ { \operatorname* { m i n } } , m , t ;$   
7 end   
8 Return: $f _ { k } ^ { \mathrm { m i n } } , { \bf c } _ { k } ^ { \mathrm { m i n } }$ , and performance measures (t, k, m).

![](images/cc2633914de7e0bf6f813dbfbe5bcc3114402fcde1908190d3ee6d420853fbe0.jpg)  
Figure 1: Visualization of selection, sampling, and subdivision in DIRECT algorithm on the two-dimensional problem.

Initialization. In the Initialization step, the DIRECT algorithm normalizes the feasible region X to the unit hyper-cube X<sup>¯</sup>. It only refers to the original space X when evaluating the objective function f. During this step, the algorithm assesses the objective function at the midpoint $\mathbf { c } ^ { 1 } \in \bar { \mathcal { X } } _ { 1 } ^ { 1 }$ and initiates the formation of the partition of P. At iteration k, this partition is defined as:

$$
\mathcal { P } _ { k } = \{ \bar { \mathcal { X } } _ { k } ^ { i } : i \in \mathbb { I } _ { k } \} ,\tag{3}
$$

where

$$
\bar { \mathcal { X } } _ { k } ^ { i } = [ \mathrm { { l } } _ { k } ^ { i } , \mathbf { u } _ { k } ^ { i } ] = \{ \mathbf { x } \in \bar { \mathcal { X } } : 0 \leq l _ { k _ { j } } ^ { i } \leq x _ { j } \leq u _ { k _ { j } } ^ { i } \leq 1 , j = 1 , \ldots , n , \forall i \in \mathbb { I } _ { k } \} ,\tag{4}
$$

and $\mathbb { I } _ { k }$ is the index set identifying the current partition $\mathcal { P } _ { k }$ . The subsequent partition, $\mathcal { P } _ { k + 1 }$ , is obtained through the subdivision of the selected POHs

from the current partition $\mathcal { P } _ { k }$

Selection. The Selection procedure at the first iteration is straightforward, as there is only one candidate $\bar { \mathcal { X } } _ { 1 } ^ { 1 } ~ \in ~ \mathcal { P } _ { 1 }$ To identify the POHs in subsequent iterations, the algorithm uses lower-bound estimates of the objective function $f$ over each hyper-rectangle in the current partition, as formalized in Definition 1.

Definition 1 (Potentially optimal hyper-rectangle [6]). Let $\mathbf { c } ^ { i }$ denote the center sample point of hyper-rectangle $\bar { \mathcal { X } } _ { k } ^ { i }$ , and let $\delta _ { k } ^ { i }$ denote its measure. Assume a positive constant $\varepsilon _ { \mathrm { D I R } } > 0$ , and let $f _ { k } ^ { \mathrm { m i n } }$ denote the best objective value obtained up to iteration $k$

A hyper-rectangle $\bar { \mathcal { X } } _ { k } ^ { j }$ , with $j \in \mathbb { I } _ { k }$ , is said to be potentially optimal if there exists a Lipschitz constant $\tilde { L } > 0$ such that

$$
f ( \mathbf { c } ^ { j } ) - \tilde { L } \delta _ { k } ^ { j } \enspace \le \enspace f ( \mathbf { c } ^ { i } ) - \tilde { L } \delta _ { k } ^ { i } , \quad \forall i \in \mathbb { I } _ { k } ,\tag{5}
$$

$$
f ( { \bf c } ^ { j } ) - \tilde { L } \delta _ { k } ^ { j } \ \leq \ fint _ { k } ^ { \operatorname* { m i n } } - \varepsilon _ { \mathrm { D I R } } ,\tag{6}
$$

where the size of hyper-rectangle $\bar { \mathcal { X } } _ { k } ^ { i }$ is defined by

$$
\delta _ { k } ^ { i } = \frac { 1 } { 2 } \Vert \mathbf { u } _ { k } ^ { i } - \mathbf { l } _ { k } ^ { i } \Vert _ { 2 } .\tag{7}
$$

A hyper-rectangle $\bar { \mathcal { X } } _ { k } ^ { j }$ is therefore considered potentially optimal if, for some positive constant $L ,$ its lower Lipschitz bound given by the left-hand side of (5) is not greater than that of any other hyper-rectangle in the current partition $\mathcal { P } _ { k }$ . Condition (6) introduces the parameter $\varepsilon _ { \mathrm { D I R } }$ to avoid excessive refinement near already identified local minima.

Remark 1. The selection of the POH is the most studied and refined step in the literature, indicating its importance. Many diferent strategies have been proposed to improve the original selection, including methods to control the ε<sub>DIR</sub> parameter [49], the selection at various levels [50] and with multiple ε<sub>DIR</sub> parameters [51], using diferent norms in (7) [42], and replacing $f _ { k } ^ { \mathrm { m i n } }$ with the median [52] or average [53] function values. Also, there can be a significant number of tied values within the same hyper-rectangle measure, which issue has been tackled in [42, 45]. Some authors have even integrated completely diferent selection schemes, such as aggressive [54], Pareto [55], and reduced Pareto [41]. Recent studies [56, 57] suggest that the two-step global and local Pareto selection scheme [43] is the most suitable and eficient selection that is currently available.

Partitioning procedure. The following steps in the DIRECT iteration involve sampling and subdivision. These can be thought of as the partitioning procedure of the partition $\mathcal { P } _ { k }$ . This procedure determines which patterns the $\bar { \mathcal X }$ will be subdivided into and where the samples will be taken within these patterns.

Sampling When a POH $( \hat { \mathcal { X } } _ { k } ^ { i } )$ is selected, DIRECT samples points at:

$$
{ \bf c } ^ { i } \pm \bar { d } _ { k } ^ { i } { \bf e } _ { j } , \quad j = \mathbb { J } _ { k } ^ { i } ,\tag{8}
$$

where $j$ belongs to the set $\mathbb { J } _ { k } ^ { i } .$ . These points are sampled along the sides of the POH, with $\bar { d } _ { k } ^ { i }$ equaling one-third of the maximum side length of $\bar { \mathcal { X } } _ { k } ^ { i }$ , and $\mathbf { e } _ { j }$ is the jth Euclidean base-vector. Initially, the proposed DIRECT samples all sides with the maximum side lengths. However, only two additional points must be sampled along the longest side. The number of samples per POH can vary from 2 to 2n, depending on the number of sides that share the maximum length.

Subdivision The division process in DIRECT involves dividing the hyper-rectangle into three equal parts, along the longest sides, in an n-dimensional space. If multiple sides share the same length, the trisection process begins with the sides with the smallest value of $\boldsymbol { w } _ { j } ^ { i }$ and moves to the highest value. The $\boldsymbol { w } _ { j } ^ { i }$ is determined as the minimum value of the function sampled along the dimension $j \colon$

$$
w _ { j } ^ { i } = \operatorname* { m i n } \left\{ f ( \mathbf { c } ^ { i } + \vec { d } _ { k } ^ { \ast } \mathbf { e } _ { j } ) , f ( \mathbf { c } ^ { i } - \vec { d } _ { k } ^ { \ast } \mathbf { e } _ { j } ) \right\} , \quad j \in \mathbb { J } _ { k } ^ { i } .\tag{9}
$$

The centers of the newly created $\mathrm { \Delta ^ { \circ } l e f t { \Sigma } } ^ { \prime }$ and “right” hyper-rectangles take newly sampled points, while the original center point becomes the center of the middle hyper-rectangle. The DIRECT algorithm’s partitioning strategy aims to ensure that the best function values are contained in the largest hyper-rectangles. This strategy helps the algorithm explore points close to favorable function values, emphasizing local search while maintaining global search capabilities.

Remark 2. Partitioning procedures that can be used with the DIRECT algorithm have been the subject of extensive research. Six other partitioning schemes have been proposed for use with DIRECT-type algorithms [45, 57, 58, 59, 60, 61]. While, the original strategy suggested that the hyper-rectangles should be partitioned according to all the longest side lengths. All the newly proposed schemes have restricted this partitioning to only the longest side length. Although various studies [56, 57] show the advantages of one strategy over the other, selecting the appropriate algorithmic composition highly depends on the use case.

Example of the optimization process with DIRECT. Example 1 clarifies the performance issues of the DIRECT and some altered versions that help to mitigate them using a simple linear test problem.

Example 1. The left side of Figure 2, illustrates the performance of the DIRECT algorithm on a simple two-dimensional $( n = 2 )$ linear function:

$$
\operatorname* { m i n } _ { \mathbf { x } \in [ \mathbf { 0 } , \mathbf { 1 } ] } ~ f ( \mathbf { x } ) = 1 + \sum _ { i = 1 } ^ { n } x _ { i }\tag{10}
$$

The global minimum value of (10) is $f ^ { * } = 1$ at the point $\mathbf { x } ^ { * } = ( 0 , 0 )$ For the experiment, we set the function target value within the absolute function value error of $1 0 ^ { - 4 }$ . The original DIRECT algorithm required 613 function evaluations to solve this simple problem.

A recent review paper [7] by the original algorithm’s author demonstrated how DIRECT-type algorithms are afected by convergence and dimensionality issues when finding the solution within varying n, using the same problem (10). To improve the convergence speed, the authors investigated existing variations and suggested the combination (let’s call it I-DTC-IOl) from three studies [6, 42, 45], which reduced the number of function evaluations needed to solve the problem from 613 to 105 (see middle part of Figure 2). The performance could be further improved by substituting the selection of the POH scheme with $I \langle 1 | ,$ which gives equal attention to the exploitation and exploration phases. We incorporated this selection in the suggested framework, let’s call it I-DTC-RPl, and we further can reduce the number of function evaluations needed to solve the problem to 69, as illustrated on the left side of Figure 2. But this is perhaps the cheapest version of the DIRECT-type algorithm that could be designed today.

Open Problem 1. Reducing the exploration capabilities, indeed, can mitigate the curse of dimensionality. However, insuficient global search is a clear risk of spending excessive function evaluations tuning suboptimal solutions, thus delaying finding the global minimum [7, 43].

![](images/4fc5f4569b44e9d73f54b444fad22f002e50835552f824f33ae18de7c164765b.jpg)  
Figure 2: Visualization of three DIRECT-type algorithms design spaces D solving twodimensional linear test problem (10).

## 2.2. Hybrid DIRECT-type algorithms

While diferent practical applications may require specific hybridization strategies [24], in this work, we summarize six popular and useful hybrid DIRECT-type algorithms in Table 2.

Table 2: Summary of existing hybrid DIRECT-type algorithmic frameworks and their combinations
<table><tr><td rowspan=1 colspan=1>Step/Algorithm</td><td rowspan=1 colspan=1>BIRMIN</td><td rowspan=1 colspan=2>DIRECT-rev  glcCluster</td><td rowspan=1 colspan=1>DIRMIN</td><td rowspan=1 colspan=1>tDIRECT</td><td rowspan=1 colspan=1>DIRECT-SQP</td></tr><tr><td rowspan=1 colspan=1>Year and reference</td><td rowspan=1 colspan=1>2020 [47]</td><td rowspan=1 colspan=1>2001 [45]</td><td rowspan=1 colspan=1>2004 [62]</td><td rowspan=1 colspan=1>2010 [46]</td><td rowspan=1 colspan=1>2023 [27]</td><td rowspan=1 colspan=1>2014 [26]</td></tr><tr><td rowspan=1 colspan=1>Selection scheme</td><td rowspan=1 colspan=6>Utilizes Definition 1 to find the set of POHs</td></tr><tr><td rowspan=2 colspan=1>Selectionimprovement</td><td rowspan=1 colspan=3>Breaks Xi ties at each δ</td><td rowspan=2 colspan=3></td></tr><tr><td rowspan=1 colspan=1>globally-biased</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Sampling</td><td rowspan=1 colspan=1>Two pointsonXi diagonal</td><td rowspan=1 colspan=5>Midpoints of each Xi</td></tr><tr><td rowspan=1 colspan=1>Subdivision</td><td rowspan=1 colspan=1>Bisects onelongest side</td><td rowspan=1 colspan=2>Trisects one longest side</td><td rowspan=1 colspan=3>Trisects all longest sides</td></tr><tr><td rowspan=1 colspan=1>Hill-climbingalgorithm</td><td rowspan=1 colspan=1>interiorpoint</td><td rowspan=1 colspan=1>Unspecified</td><td rowspan=1 colspan=1>NPSOL</td><td rowspan=1 colspan=1>NMonNC</td><td rowspan=1 colspan=1>conjugategradient</td><td rowspan=1 colspan=1>SQP</td></tr><tr><td rowspan=2 colspan=1>Runhill-climbersstarting at:</td><td rowspan=2 colspan=2> ${ \bf c } _ { k } ^ { \mathrm { m i n } } ,$ if DIRECT improves $f _ { k } ^ { \mathrm { m i n } }$ </td><td rowspan=2 colspan=1>best x fromeach cluster</td><td rowspan=1 colspan=3>c from every:</td></tr><tr><td rowspan=1 colspan=1> $\bar { \mathcal { X } } _ { k } ^ { i } \in S _ { k }$ </td><td rowspan=1 colspan=1> $\bar { \mathcal { X } } _ { k } ^ { i } \in P _ { k }$ </td><td rowspan=1 colspan=1> $\bar { \mathcal { X } } _ { k } ^ { i } \in S _ { k } , \mathrm { i f }$  $\delta _ { k } ^ { i } \le \delta ^ { \mathrm { l i m i t } }$ </td></tr></table>

Three of the six hybrid algorithms (DIRMIN, tDIRECT, and DIRECT-SQP) use the original DIRECT algorithmic framework in combination with hill-climbing optimizers. Two of these algorithms, DIRMIN and tDIRECT, excessively use hill-climbers, starting from each POH (DIRMIN) or from every sampled point (tDIRECT). The DIRECT-SQP algorithm uses hill-climber more cautiously. Motivated by the fact that the global optima can be located far from the starting points, the authors suggested executing the hill-climber beginning from the POHs only if it reaches some prescribed limit (δ<sup>limit</sup>).

Other researchers have combined hill-climbers and DIRECT more carefully. Other authors modified the selection and partitioning strategies for their approaches. In particular, BIRMIN and DIRECT-rev mostly rely on DIRECT search and use hill-climbers only if there is some improvement in their search. The glcCluster algorithm uses an entirely diferent approach to utilizing hill-climber. The algorithm starts the hill-climber from the best point of each adaptively clustering algorithm-made cluster.

## 3. Description of the Proposed Approach

In this section, we present a description of the newly proposed X-DTC-GL algorithm, which is an extension of the I-DTC-GL.

## 3.1. From I-DTC-GL to X-DTC-GL

## 3.1.1. I-DTC-GL Algorithm

The I-DTC-GL algorithm [56] employs a partitioning strategy that difers slightly from the original DIRECT algorithm. It uses a hyper-rectangular partitioning approach based on 1−Dimensional Trisection, and objective function evaluations are conducted at Center points (I-DTC) [45]. Instead of subdividing all the longest sides of the hyper-rectangle, the algorithm selects only one with the least splits over the entire algorithm search. Additionally, the algorithm utilizes a two-step-based (Global-Local) Pareto selection (GL) scheme [39, 43], which is the most efective for complex problems based on multiple studies [57, 63].

## 3.1.2. Implementing the Dynamic Partitioning Strategy Into X-DTC-GL

When the algorithm begins to iteratively tune the solution, we need to be cautious of the potentially costly global drag that can occur. Instead of relying on a fixed subdivision scheme, the proposed approach adaptively refines a selected POH by constructing a one-dimensional surrogate model along its longest (i.e., least partitioned) side. Based on this model, the predicted minimum is estimated, and the sub-hyper-rectangle containing this predicted minimum is further subdivided if an improvement in $f ^ { \mathrm { m i n } }$ is expected. Unlike the original one-shot partitioning, this process can be repeated, allowing the initially selected hyper-rectangle to be subdivided multiple times through its most promising sub-hyper-rectangles, each time along the longest side, until no further improvement is indicated. This allows the solution region to be progressively tightened, as illustrated later in Figure 4.

To explore the potential in each POH, we investigated two diferent approximations, quadratic and linear, as shown in Figure 3. The left panel

![](images/8839a15532de226955e755ac289228e6ed644389af135afe7d702b9cb6446cd8.jpg)  
Figure 3: Quadratic (left side) and linear (right side) approximation using three-point interpolation in subdivision of the first hyper-rectangle $( \bar { \mathcal { X } } _ { 1 } ^ { 1 } )$ in DIRECT.

of the figure illustrates the quadratic approximation, incorporating all three points sampled subsequent to the subdivision of the $\bar { \mathcal { X } } _ { 1 } ^ { 1 }$ hyper-rectangle along its longest side. Conversely, the right panel depicts the linear approximation utilizing two pairs of points. The search for the minimum is conducted within the bounds of the considered $\bar { \mathcal { X } } _ { 1 } ^ { 1 }$ hyper-rectangle. In both cases, the middle hyper-rectangle encloses the $\hat { f } ^ { \mathrm { m i n } }$ value, with the linear approximation predicting better enhancement in the solution.

The formal definition of dynamic partitioning is given in Definition 2.

Definition 2 (Dynamic partitioning). Let $\bar { \mathcal { X } } _ { k } ^ { i }$ denote the selected POH at iteration k, where i is its index in the current partition $\mathcal { P } _ { k }$ . Its bounds are given by $[ \mathbf { l } _ { k } ^ { i } , \mathbf { u } _ { k } ^ { i } ]$ , and its center is denoted by $\mathbf { c } ^ { i }$

Let m denote the total number of function evaluations performed up to iteration k. Two new sample points, indexed by $m + 1$ and $m + 2$ , are generated along coordinate $j$ at distance $\bar { d } _ { k } ^ { i }$ from the center coordinate $c _ { j } ^ { i } ,$ where $\bar { d } _ { k } ^ { i }$ is equal to one-third of the maximal side length of $\bar { \mathcal { X } } _ { k } ^ { i }$ . Denote the corresponding coordinate values by $c _ { j } ^ { m + 1 }$ and $c _ { j } ^ { m + 2 }$

Let $\hat { f } ^ { \mathrm { m i n } }$ and $\hat { c } ^ { \mathrm { m i n } }$ denote, respectively, the approximate minimum value and its corresponding location within $[ \mathbf { l } _ { k } ^ { i } , \mathbf { u } _ { k } ^ { i } ]$ , obtained by minimizing the one-dimensional model $\hat { f } ( c )$ fitted to the triplet $( c _ { j } ^ { m + 1 } , c _ { j } ^ { i } , c _ { j } ^ { m + 2 } )$

Then, the hyper-rectangle $\bar { \mathcal { X } } _ { k } ^ { h }$ , whose center $\mathbf { c } ^ { h }$ is closest to the approximate minimizer $\hat { c } _ { j } ^ { \mathrm { m i n } }$ , is determined as

$$
\bar { \mathcal { X } } _ { k } ^ { h } = \underset { h \in \{ i , m + 1 , m + 2 \} } { \arg \operatorname* { m i n } } \left( \left| \hat { c } _ { j } ^ { \operatorname* { m i n } } - c _ { j } ^ { h } \right| , f ( \mathbf { c } ^ { h } ) \right) .\tag{11}
$$

If two hyper-rectangles are located at the same distance from the approximate minimizer, the tie is resolved according to their ordering in the candidate set, i.e., the hyper-rectangle with the smallest index is selected. The hyperrectangle $\bar { \mathcal { X } } _ { k } ^ { h }$ is further subdivided if

$$
\hat { f } ^ { \operatorname* { m i n } } < f _ { k } ^ { \operatorname* { m i n } } - \varepsilon _ { \operatorname* { i m p } } \quad \mathrm { a n d } \quad \bar { d } _ { k } ^ { \ast } \geq \varepsilon _ { \mathrm { s i z e } } .\tag{12}
$$

Consequently, Definition 2 facilitates the subdivision of hyper-rectangles in cases where the estimated minima demonstrate an improvement over the current optimal solution, with $\varepsilon _ { \mathrm { i m p } }$ controlling the expected enhancement, and $\varepsilon _ { \mathrm { s i z e } } > 0$ is the minimum side-length threshold (to prevent excessive refinement). Similarly to (6), condition (12) is needed to stop the algorithm from wasting function evaluations by partitioning hyper-rectangles where we can only expect a negligible improvement.

The left and middle parts of Figure 4 in Example 2 illustrate how dynamic partitioning would perform on a simple quadratic function. If the predicted $\hat { f } ^ { \mathrm { m i n } }$ is better than $f _ { k } ^ { \mathrm { m i n } }$ , Definition 2 will continue to subdivide the hyperrectangle containing the predicted minima. We repeat this process while the surrogate predicts a suficiently promising improvement and the triplet step size remains above the user-prescribed limit; otherwise, the dynamic partitioning is terminated.

Example 2. The following example Figure 4 shows the performance of the suggested partitioning scheme utilized on a basic shifted Sphere function $( l e f t )$ and a linear function (middle) described in (10). In this example, we aim to achieve a target function value within an absolute value of $1 0 ^ { - 8 }$ . In both examples, the black points denote evaluated sample locations, whereas the blue points and dashed lines illustrate the approximation employed by the dynamic partitioning mechanism. The X-DTC-GL algorithm successfully identified the solution without necessitating additional global searches. On the right side of the figure, we replicated the experiment performed in [7] using the same linear function (10). As the dimension increases, the number of distinct measure hyper-rectangles increases [39], rendering the isolation of the solution with the required accuracy expensive, even for straightforward instances.

The left side of the figure demonstrates the improved convergence of the proposed partitioning methodology relative to its DIRECT-type counterparts when applied to various dimensions of the linear function. The X-DTC-GL algorithm can isolate the solution in the evaluated straightforward instances with the necessary precision while requiring a minimal number of hyper-rectangular partitions, whereas other algorithms exhibit a significant “global drag” issue. Starting from the one-dimensional case, the original DIRECT and I-DTC-GL algorithms generate approximately 60% more hyper-rectangles than the proposed X-DTC-GL before reaching the convergence threshold. This disparity grows rapidly with dimension. For example, the excess reaches approximately 99.8% for DIRECT already in dimension five, indicating that an increasingly large fraction of the search efort is spent exploring hyper-rectangles that do not contribute to faster convergence. In contrast, the more locally focused I-DTC-IOl and I-DTC-RPl algorithms perform considerably better in this respect. Nevertheless, a substantial amount of search efort remains unproductive, with the corresponding excess reaching approximately 70% and 48%, respectively, for n≥10.

While this example investigates straightforward instances where dynamic partitioning naturally directly leads to the optimum, in complex multi-modal cases, the local models may initially be highly imprecise due to the small number of points and vast intervals used to fit the model. However, as the hyper-rectangles are iteratively reduced in size, the samples become denser, and the approximation might evolve very accurately.

3.1.3. Algorithmic Steps, Hybridization, and Comment on the Convergence Algorithm workflow. Algorithm 2 provides the final pseudo-code for the X-DTC-GL algorithm. The algorithm requires three inputs: the objective function (f), the optimization domain (X ), and algorithmic options defining parameters and stopping criteria, such as the function target value $( f ^ { \mathrm { t a r g e t } } )$ the maximum number of function evaluations $\left( \mathtt { M } _ { \mathrm { m a x } } \right)$ , or the maximum number of iterations $\left( \mathtt { K } _ { \mathrm { m a x } } \right)$

![](images/11431d5abd15eb9e546bfd2c8fd2b08c3dad0263f7e261cc4bff45d9d796319f.jpg)  
Figure 4: The left and middle panels illustrate the performance of the X-DTC-GL algorithm on two straightforward instances: the two-dimensional shifted Sphere function [64] and a linear function described in (10). The right panel compares the performance of X-DTC-GL with relevant DIRECT-type counterparts on the linear function (10) with diferent dimensions (lower curves indicate better performance).

The initialization phase (lines 1–3) largely follows the original DIRECT procedure, except that a hill-climber is executed in line 2 to refine the initial minimum. After the initialization phase (lines 4–20), the algorithm enters the main loop, which continues until at least one of the stopping criteria is satisfied. In line 5, the algorithm determines the set of (POHs), which are subsequently partitioned dynamically in lines 6–18. Depending on the selected variant, the surrogate model used in the dynamic partitioning step (line 11) is either quadratic (Q, default) or linear (L). In the remainder of this paper, the corresponding variants are denoted by $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ and $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { L }$ , respectively. The depth of dynamic partitioning is controlled by $\varepsilon _ { \mathrm { s i z e } }$ and $\varepsilon _ { \mathrm { i m p } } .$ which act as numerical safeguards. By default, they are both set to $1 0 ^ { - 1 6 }$ , slightly above the machine precision limit. Similar safeguards are present in several DIRECT implementations, although they are often used implicitly and not explicitly discussed. Upon completion, the algorithms provide the solution point $( \mathbf { c } _ { k } ^ { \mathrm { { m i n } } } )$ , the objective function value found $( f _ { k } ^ { \operatorname* { m i n } } )$ , and algorithmic performance measures, including the number of function evaluations (m), the number of iterations (k), and time (t).

Incorporating the hill-climber. In addition to the initialization step, the hill-climber may also be invoked during the main iterations. In lines 13–16 of Algorithm 2, a hill-climber is initiated from the center of the hyper-rectangle if: the approximation suggests that this region could potentially ofer a better value; the predicted minimizer lies within the hyper-rectangle; and no local search has previously been performed from this region. All function evaluations performed during the hill-climber search are counted toward the global evaluation counter m. After each local run, m is updated accordingly (lines 15 and 3), ensuring that comparisons with other methods remain fair. The suggested hybridization strategy serves as a middle ground between existing approaches. For instance, algorithms like BIRMIN and DIRECT-rev initiate local searches only when there is significant improvement in $f _ { k } ^ { \mathrm { m i n } }$ , whereas methods like DIRMIN and tDIRECT trigger local searches from every POH or sampled point. In contrast, our approach proactively initiates local searches without waiting for $f _ { k } ^ { \mathrm { m i n } }$ to improve, while also avoiding the execution of local searches in regions deemed unattractive.

```latex
Algorithm 2: Main steps of X-DTC-GL algorithm
Input:
$f \colon$ Objective function
$\mathcal { X } \colon$ Decision space
$O P T ;$ structure with optimization options
Output:
$f _ { k } ^ { \mathrm { m i n } } , { \bf c } _ { k } ^ { \mathrm { m i n } } ;$ : Optimal solution and its corresponding point
t (time), k (iterations), m (function evaluations): Performance measures
1 Initialization: Normalize X to ${ \bar { \mathcal { X } } } .$ Evaluate f at $\mathbf { c } ^ { 1 }$ . Set $f _ { 1 } ^ { \mathrm { m i n } } \gets f ( \mathbf { c } ^ { 1 } )$ and
$\mathbf { c } _ { 1 } ^ { \mathrm { { m i n } } }  \mathbf { c } ^ { 1 }$ and initialize $t , k  1 , m  1 , \lambda  0 , H _ { 1 }  \emptyset .$ , and stopping criteria
2 Run hill-climber starting from $\mathbf { c } ^ { 1 }$
3 Update $f _ { 1 } ^ { \mathrm { m i n } } , { \bf c } _ { 1 } ^ { \mathrm { m i n } }$ , m and set $H _ { k } \gets H _ { k } \cup i .$
4 while $f ^ { \mathrm { t a r g e t } } < f _ { k } ^ { \mathrm { m i n } }$ and $m < M _ { \mathrm { m a x } }$ and $k < K _ { \operatorname* { m a x } }$ do $/ /$ algorithm iterations
5 Selection: Identify the index set $\mathbb { S } _ { k } \subseteq \mathbb { I } _ { k }$ of POHs. $/ /$ selection method [43]
6 foreach $i \in \mathbb { S } _ { k }$ do // loop through all selected POHs
7 repeat $/ /$ dynamic partitioning
8 Find the maximum side length $j$ of $\bar { \mathcal { X } } _ { k } ^ { i }$ which is least split in $\mathcal { P } _ { k }$
9 Sample $\mathbf { c } ^ { i } \pm \bar { d } _ { k } ^ { i } \mathbf { e } _ { j }$ , evaluate $f ,$ and update $f _ { k } ^ { \mathrm { m i n } } , { \bf c } _ { k } ^ { \mathrm { m i n } }$ , and m.
10 Subdivide $\bar { \mathcal { X } } _ { k } ^ { i }$ (trisect) and update $\mathcal { P } _ { k }  \mathcal { P } _ { k } \cup \bar { \mathcal { X } } _ { k } ^ { m - 1 } \cup \bar { \mathcal { X } } _ { k } ^ { m }$
11 Construct surrogate model $\hat { f } ( c )$ and find $\hat { f } ^ { \mathrm { m i n } } , \hat { c } ^ { \mathrm { m i n } }$
12 Update the POH index i using (11).
13 if eq. (12) and $\hat { c } ^ { \mathrm { m i n } } \in ( \mathbf { l } _ { k } ^ { i } , \mathbf { u } _ { k } ^ { i } )$ and $i \notin H _ { k }$ then $/ /$ hybridization
14 Run hill-climber starting from $\mathbf { c } _ { k } ^ { i }$
15 Update $f _ { k } ^ { \mathrm { m i n } } , { \bf c } _ { k } ^ { \mathrm { m i n } }$ , m and set $H _ { k } \gets H _ { k } \cup i .$
16 end
17 until eq. (12) is satisfied
18 end
19 Set k $ k + 1$ and update t.
20 end
21 Return: $f _ { k } ^ { \mathrm { m i n } } , { \bf c } _ { k } ^ { \mathrm { m i n } }$ , and performance measures $( t , k , m )$
```

Comments on the convergence. The convergence properties of DIRECT-type algorithms have been extensively reviewed and investigated in the literature, as can be seen in relevant references [6, 52, 57, 58, 59]. These algorithms typically exhibit a type of convergence known as “everywhere-dense”. Convergence can be ensured by only assuming continuity of the objective function, at least in the vicinity of global minima. Since the selection scheme used in the X-DTC-GL always includes at least one hyper-rectangle from the group of hyper-rectangles with the largest measure $\delta _ { k } ^ { \mathrm { m a x } }$ in the set of POHs, the X-DTC-GL convergence can be proven using the same reasoning as for other DIRECT-type algorithms.

Comments on the implementation. In practical black-box scenarios, the objective function may not be evaluable at all sampled locations due to simulation failures, infeasible configurations, or data sparsity. The proposed X-DTC-GL framework operates on point-wise evaluations and does not require an analytic representation of the objective; however, it assumes that each queried point returns a finite scalar value that can be evaluated relative to other candidates. When evaluations are unavailable at specific locations, standard handling strategies such as rejection, penalization [48, 65], or surrogate-based interpolation from available data should additionally be employed. These mechanisms would allow the algorithm to remain applicable in settings where the objective is defined only on a subset of the feasible region.

## 3.2. Component Analysis, Robustness Assessment, and Limitations

This section evaluates the contribution of the proposed individual mechanisms, investigates the algorithm’s sensitivity to problem perturbations, assesses its performance across diferent problem types, and identifies potential limitations. For this purpose, we employ the BBOB benchmark suite [66] available through the IOHprofiler [67] Python interface. To ensure a robust evaluation, we considered five dimensions $n { \in } \{ 2 , 3 , 5 , 1 0 , 2 0 \}$ and the first five instances of each BBOB function, resulting in 600 total runs per algorithm. All algorithms were evaluated using a budget of $n \times 1 0 ^ { 5 }$ function evaluations, with the goal of achieving an absolute error below $1 0 ^ { - 4 }$ The experiments were conducted in MATLAB R2023a on a system running Microsoft Windows 10, equipped with an 8th-generation Intel Core i7-8750H processor (6 cores) and 16 GB of RAM.

Table 3: Algorithmic components activated in the investigated variants. For each component, the corresponding lines in Algorithm 2 are indicated in the second header row. A plus sign (+) denotes that the corresponding steps are active in a given variant, whereas a minus sign (−) denotes that they are inactive.
<table><tr><td>Algorithm</td><td>Dynamic partitioning 17, 11–12</td><td>Init. local solver LS strategy</td><td>Alg. 2, lines 7, Alg. 2, lines 2–3 Alg. 2, lines 13–</td></tr><tr><td> $\mathrm { I - D T C - G L }$ </td><td></td><td></td><td>16</td></tr><tr><td> $\mathsf { D P } _ { Q } ~ / ~ \mathsf { D P } _ { L }$ </td><td>一 十</td><td>一</td><td>一</td></tr><tr><td> $1 \mathrm { L S - D P } _ { Q } ~ / ~ 1 \mathrm { L S - D P } _ { L }$ </td><td>十</td><td>十</td><td></td></tr><tr><td> $\mathrm { L S } _ { Q } ~ / ~ \mathrm { L S } _ { L }$ </td><td></td><td>十</td><td>十</td></tr><tr><td> $\mathtt { X } \mathrm { - D T C - G L } _ { Q } \ \mathrm { ~ / ~ } \mathtt { X } \mathrm { - D T C - G L } _ { L }$ </td><td>十</td><td>十</td><td>十</td></tr></table>

For the ablation study, we derive several algorithmic variants from the baseline I-DTC-GL. Specifically, we consider the baseline I-DTC-GL and eight extensions obtained by adding individual components, as summarized in Table 3. Each extension is studied in two versions, depending on the surrogate model used in the dynamic partitioning step: a quadratic version (Q) and a linear version (L). Accordingly, we consider: (i) $\mathsf { D P } _ { Q }$ and $\mathsf { D P } _ { L } ,$ which augment the baseline with dynamic partitioning; (ii) $1 \mathrm { L S - D P } _ { Q }$ and $1 \mathrm { L S - D P } _ { L }$ , which further add a single local solver run at initialization; (iii) $\mathbb { L } \mathbb { S } _ { Q }$ and $\mathrm { L S } _ { L }$ , which incorporate the proposed local solver usage strategy without dynamic partitioning; and (iv) the full method, $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ and $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { L }$ , which combine all components.

## 3.2.1. Performance Across Diferent Problem Types

The performance of all nine algorithmic variants is illustrated in Figure 5. Each panel of the figure corresponds to a diferent BBOB problem class and aggregates results across all instances and problem dimensions, while the final panel reports the overall performance aggregated over all problem classes.

A first observation is that dynamic partitioning alone (both variants $\mathsf { D P } _ { Q }$ and $\mathsf { D P } _ { L } )$ provides only limited benefits and may even degrade the performance of the original baseline I-DTC-GL algorithm. This behavior can be explained by the fact that the algorithm initially lacks a suficiently good estimate of $f ^ { \mathrm { m i n } }$ . Consequently, the dynamic partitioning mechanism may prioritize subdivisions that appear promising but do not lead to meaningful improvements. As a result, the algorithm may spend a considerable number of function evaluations exploring subregions before identifying a suficiently strong minima value. However, when the local solver is executed once at initialization $\left( 1 \mathsf { L S } - \mathsf { D P } _ { Q } / 1 \mathsf { L S } - \mathsf { D P } _ { L } \right)$ , the performance improves substantially for most problem classes. The initial local search provides a better estimate of $f ^ { \mathrm { m i n } }$ , which in turn allows the dynamic partitioning mechanism to guide the search more efectively. Although these variants still do not match the performance of the full $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ and $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { L }$ algorithms for moderately and ill-conditioned problems, they consistently outperform the baseline I-DTC-GL on most problem types.

![](images/7a1173c1a06aeedffcc2d66570fe5980754cf051f7639a21ed1f9238241cee12.jpg)  
Figure 5: Data profiles showing the proportion of problem instances solved (higher is better) as a function of the allowable number of function evaluations across five BBOB problem classes and overall.

The variants employing only the local solver usage strategy $\left( \mathrm { L S } _ { Q } \mathrm { ~ / ~ } \mathrm { L S } _ { L } \right)$ are generally competitive with the full $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ and $\mathrm { X - D T C - G L } _ { L }$ methods.

Nevertheless, $\mathrm { L S } _ { Q }$ performs noticeably worse than $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ and $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { L }$ on ill-conditioned problems, while $\mathrm { L S } _ { L }$ ranks among the least competitive methods on multi-modal problems. Overall, $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ achieves the best performance in most cases, solving the largest proportion of problem instances across the considered problem classes.

The introduced modifications do not always provide consistent improvements across all problem types. For highly challenging, weakly structured multi-modal problems, the performance of most variants is similar to that of the baseline I-DTC-GL. In particular, all algorithms solve approximately $6 5 . 6 \% - 7 0 . 4 \%$ of the instances, with the $1 \mathrm { L S - D P } _ { L }$ achieving the highest success rate. A notable limitation is observed for multi-modal problems, where the baseline solves the largest proportion of instances (about 68%), and its curve remains consistently above the other methods, while the second-best method, $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ , solves approximately 67.2%.

## 3.2.2. Sensitivity to Instance Perturbations

To evaluate the algorithms’ sensitivity to changes in the optimization domain and assess whether the proposed modifications afect robustness, we analyzed their performance across the five randomized BBOB instances for each problem. These instances introduce shifts and rotations of the search space, along with changes in the objective function minima values, providing a natural mechanism for evaluating robustness to such perturbations.

Figure 6 summarizes the variability across instances. The left three panels (each for a diferent algorithm) show empirical cumulative distribution functions (ECDFs) of the fraction of precision targets achieved over the evaluation budget, with shaded regions indicating the min–max range across the five instances. The targets were defined using 51 absolute precision levels logarithmically spaced in $1 0 ^ { [ - 4 , 2 ] }$ , similar to the setup used in the COCO benchmarking platform [68]. The shaded regions remain narrow for all three algorithms, indicating minimal variability across instances. This suggests that both the baseline algorithm (I-DTC-GL) and the proposed modifications $\left( \mathtt { X } \mathrm { - D T C } \mathrm { - } \mathtt { G L } _ { Q } \right.$ and $\mathtt { X } \mathrm { - D T C } \mathrm { - } \mathtt { G L } _ { L } \Big )$ are robust and largely insensitive to the introduced perturbations, consistently approaching the desired target precision using a similar number of function evaluations.

The right panel of Figure 6 presents boxplots of the standard deviation of the number of function evaluations required to reach the prescribed target value across the five instances of each problem, aggregated over all problems. For both developed algorithms, the median standard deviation is approximately $5 0 \times n$ function evaluations. In contrast, for the original algorithm (I-DTC-GL), it is about $5 5 0 \times n$ , indicating substantially higher variability across instances. However, the third quartile and higher values are comparable across all three algorithms.

![](images/f0202960cd3a0f8c758594bb8953bcfc80defa57f946228375eb559c6a7d2833.jpg)  
Figure 6: The three plots on the left show ECDFs of the fraction of precision targets reached (higher is better); the filled region indicates the min–max range across the five instances (narrower is better). The plot on the right depicts boxplots of the standard deviation of the number of function evaluations required to reach the target across instances, aggregated over all problems (lower is better).

The few extreme outliers correspond to problems in which some instances require substantially more evaluations than others. Since these outliers appear for all algorithms, they likely reflect inherent variability in the dificulty of specific benchmark instances rather than algorithm instability.

## 3.2.3. Runtime Overhead Assessment

We additionally report wall-clock execution times (t) to quantify the computational overhead introduced by hybridization, surrogate construction, and the adaptive decision mechanism. For this purpose, we compare three algorithms: the original baseline method I-DTC-GL and its two modified variants, $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ and $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { L }$

Figure 7 shows the distribution of total execution time over the 24 BBOB functions and five instances for all considered dimensions. The reported time corresponds to the duration required to reach the prescribed absolute error threshold; in case of failure, it reflects the time needed to exhaust the entire evaluation budget during the search.

![](images/ce0da8e4522cb7f97042141465781490410ab476990052a60a7baf0e07541ba5.jpg)  
Figure 7: Boxplots of execution time t (lower is better) for three algorithms on the 24 BBOB benchmark functions, evaluated over five instances. Each subplot corresponds to a diferent problem dimension $n \in \{ 2 , 3 , 5 , 1 0 , 2 0 \}$

The minimum, first quartile, and median values of both modified variants are consistently lower than those of the original algorithm across all five panels. This follows from the fact that the modifications reduce the number of function evaluations on most problems (as we previously showed); consequently, the overall execution time decreases accordingly.

More informative are the third quartile, the maximum values, and the upper outliers, which are comparable across all panels and algorithms. These statistics largely correspond to runs in which the complete evaluation budget is exhausted. At this level, the total runtime reflects the cost of completing the entire optimization process rather than the cost of early convergence. Importantly, the additional computations introduced do not lead to a noticeable increase in wall-clock time. In dimensions $n \geq 1 0$ , the modified algorithms are even slightly faster. This efect can be attributed to the hybrid structure: local subroutines, inexpensive surrogate-based steps, and dynamic partitioning may consume a portion of the evaluation budget at relatively low computational cost, thereby reducing the number of iterations the core optimization routine can perform. As a result, the overall runtime remains comparable or occasionally lower, despite the added algorithmic components.

## 3.2.4. Limitations of the Proposed Algorithm

The proposed extensions improve performance on many problem classes; however, as demonstrated in Section 3.2.1, their benefits are not universal. The efectiveness of the hybridization strategy and dynamic partitioning depends on specific properties of the objective function. In the following paragraphs, we describe two limitations identified during the study.

Hybridization and dynamic partitioning overhead. It is well known that local solvers do not always perform well [69]. Therefore, in the proposed framework, they may introduce additional computational overhead without improving the solution quality. This may occur with non-smooth functions or in regions with flatness, where gradient information is dificult to approximate eficiently. The left panel of Figure 8 and Example 3 illustrates and explains that hybridization and dynamic partitioning may negatively afect performance.

Example 3. The left panel of Figure 8 shows the convergence plots of three algorithms for the first instance of the Step Ellipsoidal test function in dimension 10. The convergence curve of the X-DTC-GL<sub>Q</sub> algorithm is the slowest. This behavior occurs because dynamic partitioning identifies promising regions and triggers the hill-climber frequently (1902 times). However, the Step Ellipsoidal function consists of many plateaus of diferent sizes, and the gradient is zero almost everywhere except in a small region near the global optimum. As a result, many local-search calls are performed, consuming a substantial number of function evaluations (40289, approximately 21 per call), while none of them produce an improvement.

Dynamic partitioning alone (DP<sub>Q</sub>), without the local solver, converges faster than X-DTC-GL<sub>Q</sub>. However, the baseline I-DTC-GL eventually achieves the fastest convergence. Although the Step Ellipsoidal function globally retains a quadratic ellipsoidal structure, consequently, the DP<sub>Q</sub> variant initially approaches the solution faster because the surrogate model can still capture the global quadratic trend of the landscape. However, once the search enters plateau regions, the surrogate approximation becomes less informative, as many sampled points share identical function values. In such regions, the model cannot reliably predict promising directions, which slows further progress.

![](images/62c0f4521604b0f510eed24ba923ce5b27395901287b49bbefb5f57e6810460a.jpg)  
Figure 8: Convergence plots for the Step Ellipsoidal (left) and Rosenbrock (right) test functions in dimension 10.

Limitations of the surrogate approximation. As shown in Section 3.2.1, dynamic partitioning alone may be inefective and can even degrade the performance of the baseline I-DTC-GL algorithm. For dynamic partitioning to work efectively, a reasonably accurate estimate of the minimum value is needed. Otherwise, if the initial minimum is poor, many selected potentially optimal hyper-rectangles might appear promising, leading the algorithm to partition them excessively, even if they do not contain true minima. To mitigate this issue, a local solver is executed during initialization (Algorithm 2, line 2) to refine the initial minimum before entering the main iteration loop. Example 3 and the right panel of Figure 8 illustrate and explain that providing a better initial value improves the performance of the dynamic partitioning strategy.

Example 4. To illustrate this limitation, we consider the baseline I-DTC-GL, the two dynamic partitioning variants $D P _ { Q }$ and ${ D P } _ { L } ,$ , and modified versions where the target value $( f ^ { \mathrm { t a r g e t } } )$ is used instead of $f _ { k } ^ { \mathrm { m i n } }$ in the condition of Equation (12). The right panel of Figure 8 shows the convergence plots of these variants for the 10-dimensional Rosenbrock function. In higher dimensions, this problem contains a local optimum with a large attraction basin, which causes the baseline I-DTC-GL to fail to reach the global solution. Both dynamic partitioning variants $( D P _ { Q }$ and $D P _ { L } )$ can solve the problem, but they require a large number of function evaluations to reach the target. When the target value $( f ^ { \mathrm { t a r g e t } } )$ is used to control when dynamic partitioning is applied, the performance of both variants improves significantly. In particular, $D P _ { Q } ( f ^ { \mathrm { t a r g e t } } )$ reaches the desired solution using approximately three times fewer function evaluations, while the $D P _ { L } ( f ^ { \mathrm { t a r g e t } } )$ variant requires roughly twice fewer evaluations. This experiment suggests that providing a reliable estimate of the minimum can substantially improve the efectiveness of dynamic partitioning. Note, however, that in practice the value f<sup>∗</sup> is not known and is used here only for experimental analysis.

## 4. Experimental Results and Discussion

## 4.1. Experimental settings

Algorithms and experimental setup. The performance of the two versions of the X-DTC-GL algorithm (with linear $- \mathrm { ~  ~ \cal ~ X ~ } { \bf - } \mathrm { D } \mathrm { T } { \bf C - } { \bf G } { \bf L } _ { L }$ and quadratic – $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ approximations) was evaluated compared to six DIRECT-type algorithms and the Naive Multi-scale Search Optimization (NMSO) algorithm [70], which utilizes a partitioning approach similar to that of DIRECT, and showed excellent performance in GECCO’15 BBO competition. Among the DIRECT-type, we included I-DTC-GL as this algorithm forms the basis for the suggested X-DTC-GL. Furthermore, the evaluation included the emerging tDIRECT algorithm and two highly efective and widely adopted algorithms, DIRMIN and BIRMIN. Furthermore, the analysis encompassed two canonical algorithms, the DIRECT and I-DTC-IOl. The main benchmark experiments were conducted under the same computational setup as the ablation study (Section 3.2). Each algorithm was allowed a budget of $n \times 1 0 ^ { 5 }$ function evalutions and success was declared when the absolute error fell below $1 0 ^ { - 4 }$

To better determine the impact of the DIRECT algorithm in hybrid approaches, in this study we used the same hill-climbing algorithm for all hybrid approaches. The local refinement stage was implemented using MATLAB’s fmincon function with the sqp algorithm option. Bound constraints were handled directly through the lower and upper bound arguments of fmincon. The SQP method was applied with its default configuration [71], with the exception that the maximum evaluation budget per run was established at $n \times 1 0 ^ { 3 }$ for all algorithms, except for the tDIRECT, for which $n \times 1 0 ^ { 2 }$ was recommended in [27]. Function evaluations performed by the SQP procedure were included in the total evaluation counter of the corresponding DIRECT-based algorithm.

Benchmark problems. The recently extended DIRECTGOLib v2.0 benchmark library [64] was used as the basis for testing the considered algorithms. The library encompasses an extensive collection of functions sourced from diverse origins, encompassing classical, emerging (ABS [72] and Layeb [73]), and widely-utilized (BBOB [66] and CEC [74, 75]) functions. These functions exhibit a range of characteristics concerning diferentiability, separability, multi-modality, and flatness.

We followed the settings and transformations of a recent study [32] that utilized 324 box-constrained test functions, investigating scalable ones with dimensions 2, 5, 10, and 20. Each function was evaluated in five diferent instances with distinct random domain shifts, and the last three instances additionally involved random rotations, resulting in a total of 4035 instances. All randomizations were generated using fixed seeds to ensure reproducibility. These transformations also prevent the optimum from coinciding with the origin, ensuring that none of the algorithms can locate the solution through initial sampling alone, which is common in many artificial benchmark problems.

We selected four balanced and representative benchmark suites, each of size 50 instances, using diferent instance selection methods [76, 77, 78]. An overview of the four suites of selected instances, together with brief summaries of their respective selection methodologies, is provided in Table 4. The last column of the table also presents the boxplots of the success rates achieved using the 26 algorithms from the open-sourced data of the recent benchmarking study [32]. This information provides a better understanding of the complexity of each selected benchmark suite. This benchmarking method was chosen to address the challenges associated with analyzing data from inadequately structured benchmark suites, as discussed in [76].

## 4.2. Solution Discovery Eficiency

In this section, we comprehensively analyze the performance eficiency of the algorithms employed to solve four distinct benchmark suites, utilizing both the performance [80] and data [81] profiles. Both of these data evaluation tools only consider instances in which the algorithms delivered the solutions with the required accuracy.

The data profiles for the four benchmark suites are illustrated in Figure 9. The horizontal axis represents the number of function evaluations, whereas the vertical axis denotes the proportion of solved instances. Additionally, the black curve in each subplot outlines the best aggregated performance achieved by any of the algorithms employed in the comparative study. This visualization clarifies the upper bound of performance attainable by the algorithms under consideration.

Table 4: Benchmark suites employed for experimental evaluation of the algorithms.
<table><tr><td rowspan="2">Benchmark</td><td colspan="3">Instance selection method</td><td rowspan="2"># of inst.</td><td colspan="4">Complexity of the</td></tr><tr><td>Ref.</td><td>Space</td><td>Description</td><td></td><td>benchmark suitesc</td><td></td></tr><tr><td>BS1-50</td><td>[77]</td><td>ELAª</td><td>Clusters instances using cosine similarity</td><td>50</td><td>0.0 0.2 0.4 0.6 0.8 1.0</td><td></td><td></td></tr><tr><td>BS2-50</td><td>[78]</td><td>ELAª</td><td>Maximize diversity using Manhattan distance</td><td>50</td><td>0.0 0.2 0.4 0.6 0.8 1.0</td><td></td><td></td></tr><tr><td>BS3-50</td><td>[76]</td><td>APb</td><td>Maximize diversity of algorithms&#x27; runtime</td><td>50</td><td>0.0 0.2 0.4 0.6 0.8 1.0</td><td></td><td>-r</td></tr><tr><td>BS4-50</td><td>[76]</td><td>APb &amp; ELAa</td><td>Maximizes diversity using statistical tests and Euclidean distance</td><td>50</td><td>0.0 0.2 0.4 0.6 0.8 1.0</td><td></td><td></td></tr></table>

<sup>a</sup> Exploratory landscape analysis [79], <sup>b</sup> Algorithm performance, <sup>c</sup> Success rates on the selected instances using 26 algorithms from [32]

The analysis demonstrates that the final success rates of the algorithms in the first two benchmark suites (BS1 and BS2 ) are more significant than in the remaining two benchmark suites. However, when comparing the overall ranking of the algorithms in all four subplots, the results are almost identical, with some minor diferences. In all subplots, the performance of both versions of the X-DTC-GL algorithm consistently stands out, solving a higher number of instances when compared to any other algorithm. Nevertheless, the best aggregated black curve suggests that either version of the X-DTC-GL algorithm does not consistently address instances that other algorithms can solve. Specifically, for BS1, BS2, and BS4, the X-DTC-GL<sub>Q</sub> algorithm failed to solve one to two such instances, while in the BS3 benchmark suite, other competing algorithms successfully addressed six instances that remained unsolved by the X-DTC-GL algorithm. Despite this, the efectiveness of both versions of the X-DTC-GL algorithm is most notable in the BS3 benchmark suite.

While data profiles show a general view of solved instances, performance profiles are specifically designed to compare the performance of specific algorithms to a set of algorithms. Performance profiles assess the overall performance of algorithms using a performance ratio $( r _ { i , a } )$ . This ratio is computed as:

![](images/7de8d8860e151115974a7a993a40aa1850d5ba291fd453c6c0e6acb8311786d0.jpg)  
Figure 9: The data profiles illustrate the proportion of instances successfully solved (higher is better) as a function of the allowable number of function evaluations across four benchmark suites.

$$
r _ { i , a } = \frac { t _ { i , a } } { \operatorname* { m i n } \{ t _ { i , a } : a \in \mathcal { A } \} } ,\tag{13}
$$

where $t _ { i , a } > 0$ is the metric for algorithm a solving instance i, and min $\{ t _ { i , a }$ $a \in \mathcal A \}$ is the best metric for the instance. The performance profile $\left( \rho _ { a } ( \lambda ) \right)$ of an algorithm a is derived from the cumulative distribution function of the performance ratio:

$$
\rho _ { a } ( \lambda ) = \frac { 1 } { | \mathcal { I } | } | \{ i \in \mathcal { I } : r _ { i , a } \leq \lambda \} | , \quad \lambda \geq 1 ,\tag{14}
$$

where |I| is the number of problems. $\rho _ { a } ( \lambda )$ represents the probability that $r _ { i , a }$ for each $i \in \mathcal I$ is within a factor λ of the best ratio.

Performance profiles in Figure 10 enable a comparison of algorithm performance across multiple instances in I. A higher $\rho _ { a } ( \lambda )$ indicates better performance, and $\rho _ { a } ( 1 )$ indicates the fraction of instances where algorithm a achieves the best performance. Based on the performance profiles, it is

![](images/a41d18b82bb8fa5bb77de360888ed7f62e4aefdcea945504e8d66eba031d7c1a.jpg)  
Figure 10: An analysis of the function evaluation metrics for the considered algorithms, utilizing four benchmark suites and employing performance profiles within the interval $\lambda \in [ 1 , 1 0 ]$ (higher curve indicates better performance).

evident that the developed algorithm demonstrates superior outcomes in three out of four benchmark suites. It exhibits the highest eficiency by achieving the most wins and solving the most instances with the smallest number of function evaluations. However, with regard to the number of wins, most algorithms also yield a reasonable number of wins. Notably, the tDIRECT algorithm outperforms both versions of the proposed X-DTC-GL algorithm in the second benchmark suite, most eficiently solving one-third of the instances.

The performance curves of both versions of the X-DTC-GL algorithm consistently surpass those of other algorithms across three subplots, indicating its superior performance relative to the comparison algorithms. Notably, for the BS2 benchmark suite, the $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ and $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { L }$ algorithms required a performance ratio of λ to 1.2 to position its curve above those of the other algorithms. When the performance ratio reaches ten, the $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ algorithm provides solutions within ten times the number of function evaluations compared to the best-performing algorithms in approximately 72.5%, 75%, 47.5% and 50% of the instances for each of the four benchmark suites.

While data profiles provide an aggregated view of the relative performance of the compared algorithms, they do not reveal detailed pairwise relationships between methods on individual problems. To complement these analyses, we present pairwise dominance heatmaps in Figure 11 that summarize how often one algorithm outperforms another across the combined benchmark sets. The left panel counts problems solved by the row algorithm but not by the column algorithm. In contrast, on the left side, it shows how many times the algorithm in the row strictly outperformed the algorithm in the column on the objective error (smaller than $1 0 ^ { - 4 }$ are truncated to $1 0 ^ { - 4 }$ to avoid overemphasizing diferences below the target accuracy threshold). Darker shades correspond to higher win counts.

The left panel shows that the most successful algorithm, $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ failed on only a very few problems that were solved by other algorithms (BIRMIN and $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { L }$ each solved 3 and 4 instances, respectively, whereas $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ failed). In contrast, $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ solved substantially more problems than these methods, namely 29 and 5 instances, respectively.

The right panel further confirms the overall advantage of $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ when considering all problems, including those not solved to the target accuracy. Although $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ is outperformed on 19, 15, and 15 instances by $\begin{array} { r } { \mathtt { X } \mathrm { - D T C \mathrm { - } G L } _ { L } , \quad \mathtt { I } \mathrm { - } \mathtt { D T C } \mathrm { - } \mathtt { G L } . } \end{array}$ and DIRMIN, respectively, the opposite comparison shows a clear advantage for $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ , with 29, 54, and 57 wins over these methods.

## 4.3. Evaluation of Solution Quality Metrics

Since some benchmark suites have many unsolved instances, we utilized the Friedman test [82] to rank the algorithms according to the obtained solutions. The Friedman test was utilized for the varying number of function evaluations, providing insights into the performance of algorithms across diferent budgets. The results of the tests are presented in Figure 12

![](images/5aa1852b5e092c4fa49cf9e34d4b1eb1e60dd42ef3f08ac712a228f4c573c6a6.jpg)  
Figure 11: Pairwise comparison of algorithms. Each heatmap entry (row, column) indicates how many times the row algorithm outperformed the column algorithm (darker shades indicate more wins). The left panel compares the number of solved problems, while the right panel compares error values, with errors below $1 0 ^ { - 4 }$ truncated to $1 0 ^ { - 4 }$

In this context, a lower mean rank indicates a better algorithm performance.

As the algorithms were generally more successful in addressing the first two benchmark suites, the Friedman mean rank values were more closely aligned than in the last two. This is because the algorithms achieved equal ranking on a significant portion of instances in the benchmark suites, thereby reducing the variability in mean ranks. However, with the maximum evaluation budget, the final algorithm ranking is almost identical, with some exceptions between algorithms in second to fourth places.

For a limited budget $\left( \leq n \times 1 0 ^ { 2 } \right)$ , substantial changes in algorithm rankings are observed, particularly afecting the performance of pure DIRECT-type algorithms, which tend to exhibit lower rankings. Notably, the NMSO algorithm performs exceptionally within approximately n×10 function evaluation budgets. However, its rankings are substantially worsening as the budget increases. NMSO consistently secures the fifth position when the evaluation budget reaches the maximum. Conversely, the I-DTC-GL algorithm exhibits an inverse pattern, performing poorly within a small evaluation budget but experiencing a significant improvement in ranking as the budget increases. Across various benchmark suites, I-DTC-GL consistently claims positions ranging from second to fourth best. Finally, our proposed X-DTC-GL consistently achieves the highest ranking regardless of the surrogate model used, starting from the ∼n×50 evaluation budget.

![](images/e2616043764e3141b87696aa45e4cc52055a4eb5d1ab546eafe94de411557d8d.jpg)  
Figure 12: Mean Friedman ranks (lower is better) derived from varying evaluation budgets across four benchmark suites.

## 4.4. Analysis of Execution Time Metrics

To assess the computational eficiency of the considered algorithms, we analyze their execution times (t) across a combined benchmark comprising 200 test functions from four instance sets: BS1-50, BS2-50, BS3-50, and BS4-50. Figure 13 summarizes the results using two complementary visualizations. The left panel shows boxplots of execution times (t) across all instances, while the right panel presents a performance profile based on t for successfully solved instances only. For visualization purposes, execution times are clipped to [10<sup>−2</sup>, 10<sup>4</sup>].

![](images/5cfae5b450eab92a367ccbdba27937055ed1d239530258103a26ab53e666d07b.jpg)

![](images/0882a73844b922556e5fc572e35fecb410ff892a48bf8357bbe8e7e7fee26e09.jpg)  
Figure 13: Execution time t comparison of the algorithms on a combined set of 200 test functions. Execution times t are clipped to $[ 1 \bar { 0 } ^ { - 2 }$ , 10<sup>4</sup>] for visualization. The left panel shows boxplots across all instances (lower is better), while the right panel shows a performance profile of solved instances (higher is better).

The boxplots show that both the minimum and the first quartile values are below 1 seconds for all algorithms, indicating that at least 25% of the test functions can be solved at comparable speeds across all solvers. Furthermore, the median execution times of the two proposed algorithms $\left( \mathrm { X } \mathrm { - D T C } \mathrm { - } \mathrm { G L } _ { Q } , \ \mathrm { X } \mathrm { - D T C } \mathrm { - } \mathrm { G L } _ { L } \right)$ are below one second, meaning that at least 50% of instances are solved within this time. The median execution times of three other algorithms, I-DTC-GL, DIRMIN, and tDIRECT, are within 10 seconds (7.44, 2.39, and 5.40 seconds, respectively), whereas the remaining four algorithms exhibit median values exceeding 200 seconds. The third quartile, maximum values, and upper outliers largely correspond to execution times associated with unsolved instances, where algorithms exhausted the entire evaluation budget. In this context, four algorithms (DIRMIN, tDIRECT, and the two proposed algorithms) exhibit similar performance, while the original baseline algorithm, I-DTC-GL, appears to be slightly slower. For four algorithms (DIRECT, I-DTC-IOl, BIRMIN, and NMSO), execution times t may reach $1 0 ^ { 4 }$ . A possible explanation is their relatively low number of evaluations per iteration, which may lead to an excessive number of iterations. Because each iteration involves several decision-making computations, the algorithm may require substantial time to exhaust the evaluation budget if a solution is not found early.

To assess how quickly the algorithms identify the global optimum relative to one another, execution time was used to generate performance profiles. According to the performance profiles (right panel of Figure 13), NMSO and $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ achieve the highest number of wins (i.e., when $\lambda = 1 )$ , solving approximately 17% of the problems in the shortest time. Another proposed algorithm $\left( \mathtt { X } \mathrm { - D T C } \mathrm { - } \mathtt { G L } _ { L } \right)$ also demonstrated strong performance, solving about 16% of the problems the fastest. However, as the performance ratio increases to $\lambda { = } 2$ , the proposed algorithms achieve the highest performance levels among all methods considered. Overall, $\mathtt { X } \mathrm { - D T C - } \mathtt { G L } _ { Q }$ and $\mathrm { X - D T C - G L } _ { L }$ outperform the other algorithms on at least 60% of the test problems within a performance ratio of six relative to the best-performing algorithm.

## 5. Conclusions and Future Work

This study introduced a novel approach for solving box-constrained BBO problems by enhancing the I-DTC-GL algorithm with dynamic partitioning and hybridization mechanisms. The proposed X-DTC-GL algorithm constructs local surrogate models within subdivided hyper-rectangles and allocates additional search efort to more promising regions, thereby improving both convergence behavior and solution quality. Comprehensive experiments on four benchmark suites based on diferent ISMs demonstrate that the proposed algorithmic variants outperform relevant alternatives in terms of solution quality, evaluation eficiency, and execution time. By establishing a substantial performance advancement over the most eficient existing DIRECT-type baselines, these results implicitly amplify the overall competitiveness of partition-based deterministic optimization within the broader domain of state-of-the-art black-box metaheuristics.

The present study focused on exploiting local monotonicity trends of the objective function and thus partially addressed the concerns raised in [7] regarding the limited use of local trend information in DIRECT. Future research may investigate more information-driven refinement strategies that adapt the subdivision location according to trends observed in function evaluations. Rather than restricting refinement to the geometric center of a selected hyper-rectangle, the search could be directed toward more promising regions, including neighboring hyper-rectangles when surrogate-based indications suggest improvement beyond the current boundaries.

Another promising direction is the investigation of alternative partitioning strategies, such as dividing the longest side into more than three segments (e.g., five, seven, or an adaptive number), which could provide richer sampling for surrogate construction. This, in turn, may enable the use of more flexible surrogate models based on the additional sample points. Additionally, the observed limitations of the algorithm suggest several directions for future research. First, incorporating lower-bound control into the dynamic partitioning strategy may ofer further improvements, as indicated by the preliminary observations. Second, combining DIRECT with alternative local solvers may enhance convergence, particularly for non-smooth or plateau-rich objective functions where diferent derivative-free local search methods may exhibit complementary strengths. An interesting extension would be the development of an adaptive hybrid framework that dynamically selects or switches between local search procedures based on the observed optimization progress and landscape characteristics.

## Funding

The research of L. Stripinis was co-funded by the European Union (project “My First Research Team”, No. 10-092-P-0001, action No. MPK-49).

## Source code and data availability

All algorithms, along with the employed configurations and scripts necessary to replicate the findings of this study, are available in the following GitHub repository:

• https://github.com/blockchain-group/DIRECTGO

The DIRECTGOLib v2.0 test problem library is available as an open-source repository on GitHub and is distributed under the MIT license. In addition, the repository contains the necessary codes for the selection of various instance sets used in this study. The dataset and codes of the ISMs can be accessed through the following channel:

• https://github.com/blockchain-group/DIRECTGOLib

## References

[1] J. Stork, A. E. Eiben, T. Bartz-Beielstein, A new taxonomy of global optimization algorithms, Natural Computing 21 (2022) 219–242. doi:10.1007/s11047-020-09820-4.

[2] S. A. Piyavskii, An algorithm for finding the absolute minimum of a function, Theory of Optimal Solutions 2 (1967) 13–24, in Russian. doi:10.1016/0041-5553(72)90115-2.

[3] B. O. Shubert, A sequential method seeking the global maximum of a function, SIAM Journal on Numerical Analysis 9 (1972) 379–388. doi:10.1137/0709036.

[4] R. Paulavičius, J. Žilinskas, Global optimization using the branch-andbound algorithm with a combination of Lipschitz bounds over simplices, Technological and Economic Development of Economy 15 (2) (2009) 310–325. doi:10.3846/1392-8619.2009.15.310-325.

[5] R. Paulavičius, J. Žilinskas, A. Grothey, Investigation of selection strategies in branch and bound algorithm with simplicial partitions and combination of Lipschitz bounds, Optimization Letters 4 (2) (2010) 173– 183. doi:10.1007/s11590-009-0156-3.

[6] D. R. Jones, C. D. Perttunen, B. E. Stuckman, Lipschitzian optimization without the lipschitz constant, Journal of Optimization Theory and Applications 79 (1993) 157–181. doi:10.1007/BF00941892.

[7] D. R. Jones, J. R. R. A. Martins, The DIRECT algorithm: 25 years later, Journal of Global Optimization 79 (3) (2021) 521–566. doi:10.1007/s10898-020-00952-6.

[8] L. Stripinis, R. Paulavičius, Derivative-free DIRECT-type Global Optimization, SpringerBriefs in Optimization, Springer Cham, 2023. doi:10.1007/978-3-031-46537-6.

[9] R. Fletcher, Practical Methods of Optimization, 2nd Edition, John and Sons Chichester, 1987. doi:10.1097/00000539-200101000-00069.

[10] M. J. Ebadi, A. Fahs, H. Fahs, R. Dehghani, Competitive secant (bfgs) methods based on modified secant relations for

unconstrained optimization, Optimization 72 (7) (2023) 1691–1706. doi:10.1080/02331934.2022.2048381.

[11] J. A. Nelder, R. Mead, A simplex method for function minimization, The Computer Journal 7 (4) (1965) 308–313. doi:10.1093/comjnl/7.4.308.

[12] N. Fadavi, H. Gangammanavar, Active set-based inexact proximal bundle algorithm for stochastic quadratic programming, Computational Optimization and Applications 93 (2026) 489–521. doi:10.1007/s10589- 025-00739-z.

[13] S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, Optimization by simulated annealing, Science 220 (4598) (1983) 671–680. doi:10.1126/science.220.4598.671.

[14] J. Holland, Adaptation in Natural and Artificial Systems, The University of Michigan Press, Ann Arbor, 1975.

[15] J. Kennedy, R. Eberhart, Particle swarm optimization, in: Proceedings of ICNN’95-international conference on neural networks, Vol. 4, IEEE, 1995, pp. 1942–1948.

[16] W.-Y. Fu, Accelerated high-dimensional global optimization: A particle swarm optimizer incorporating homogeneous learning and autophagy mechanisms, Information Sciences 648 (2023) 119573. doi:https://doi.org/10.1016/j.ins.2023.119573.

[17] D. R. Jones, M. Schonlau, W. J. Welch, Eficient Global Optimization of Expensive Black-Box Functions, Journal of Global Optimization 13 (4) (1998) 455–492. doi:10.1023/A:1008306431147.

[18] Y. Zeng, Y. Cheng, J. Liu, An eficient global optimization algorithm for expensive constrained black-box problems by reducing candidate infilling region, Information Sciences 609 (2022) 1641–1669. doi:https://doi.org/10.1016/j.ins.2022.07.162.

[19] T. Bartz-Beielstein, M. Zaeferer, Model-based methods for continuous and discrete global optimization, Applied Soft Computing 55 (2017) 154–167. doi:https://doi.org/10.1016/j.asoc.2017.01.039.

[20] F. Neri, C. Cotta, Memetic algorithms and memetic computing optimization: A literature review, Swarm and Evolutionary Computation 2 (1) (2012) 1–14. doi:10.1016/j.swevo.2011.11.003.

[21] J.-S. Pan, N. Liu, S.-C. Chu, T. Lai, An eficient surrogate-assisted hybrid optimization algorithm for expensive optimization problems, Information Sciences 561 (2021) 304–325. doi:https://doi.org/10.1016/j.ins.2020.11.056.

[22] P. Kerschke, H. H. Hoos, F. Neumann, H. Trautmann, Automated algorithm selection: Survey and perspectives, Evolutionary Computation 27 (1) (2019) 3–45. doi:10.1162/evco\_a\_00242.

[23] H. Dong, B. Song, P. Wang, Z. Dong, Hybrid surrogatebased optimization using space reduction (HSOSR) for expensive black-box functions, Applied Soft Computing 64 (2018) 641–655. doi:https://doi.org/10.1016/j.asoc.2017.12.046.

[24] L. Stripinis, R. Paulavičius, Review and Computational Study on Practicality of Derivative-Free DIRECT-Type Methods, Informatica 36 (1) (2025) 141–174. doi:10.15388/24-INFOR548.

[25] T. Worbs, B. Rumi, K. H. Madsen, A. Thielscher, Realistic electric field characterization of clinically used deformable large tms coils in a large cohort, Brain Stimulation: Basic, Translational, and Clinical Research in Neuromodulation 18 (2025) 1174–1183, doi: 10.1016/j.brs.2025.05.136. doi:10.1016/j.brs.2025.05.136.

[26] L. Li, X. M. Chen, L. Zhang, A global optimization algorithm for trajectory data based car-following model calibration, Transportation Research Part C: Emerging Technologies 68 (2016) 311–332. doi:10.1016/j.trc.2016.04.011.

[27] K. Kanayama, A. Seko, K. Toyoura, Structure search method for atomic clusters based on the dividing rectangles algorithm, PHYSICAL REVIEW E 108 (2023) 035303. doi:10.1103/PhysRevE.108.035303.

[28] I. Dapšys, R. Čiegis, V. Starikovičius, Applying artificial neural networks to solve the inverse problem of evaluating concentrations in multianalyte mixtures from biosensor signals, Nonlinear Analysis: Modelling and Control 29 (1) (2023) 53–70. doi:10.15388/namc.2024.29.33604.

[29] I. B. Hvidsten, K. H. Liland, O. Tomic, J. M. Marchetti, Modeling of biodiesel production using optimization designs from literature: aiming to reduce the laboratory workload, Fuel Processing Technology 275 (2025) 108265. doi:https://doi.org/10.1016/j.fuproc.2025.108265.

[30] Y. Chen, F. Yu, Q. Zhang, M. Pratama, Energy-eficient adaptive perception for autonomous driving via lightweight policy learning and simulation-based optimization, Knowledge-Based Systems 330 (2025) 114514. doi:https://doi.org/10.1016/j.knosys.2025.114514.

[31] R. Chen, X. Tian, H. Du, W. Zhang, Z. Wang, L. Xia, J. Han, K. Wang, Research on generation of toolpaths with smooth tool orientation changes for five-axis machining of blisk based on the rotary axes kinematic features of machine tool, Journal of Manufacturing Processes 152 (2025) 1204–1219. doi:https://doi.org/10.1016/j.jmapro.2025.08.065.

[32] L. Stripinis, J. Kůdela, R. Paulavičius, Benchmarking derivative-free global optimization algorithms under limited dimensions and large evaluation budgets, IEEE Transactions on Evolutionary Computation 29 (1) (2025) 187–204. doi:10.1109/TEVC.2024.3379756.

[33] J. Kůdela, Benchmarking State-of-the-art DIRECT-type Methods on the BBOB Noiseless Testbed, in: Proceedings of the Companion Conference on Genetic and Evolutionary Computation, GECCO’23 Companion, Association for Computing Machinery, New York, NY, USA, 2023, pp. 1620–1627. doi:10.1145/3583133.3596308.

[34] P. Bujok, P. Kolenovsky, Eigen crossover in cooperative model of evolutionary algorithms applied to cec 2022 single objective numerical optimisation, in: 2022 IEEE Congress on Evolutionary Computation (CEC), IEEE, 2022, pp. 1–8.

[35] A. Kumar, R. K. Misra, D. Singh, Improving the local search capability of efective butterfly optimizer using covariance matrix adapted retreat phase, in: 2017 IEEE congress on evolutionary computation (CEC), IEEE, 2017, pp. 1835–1842.

[36] G. Zhang, Y. Shi, Hybrid sampling evolution strategy for solving single objective bound constrained problems, in: 2018 IEEE Congress on Evolutionary Computation (CEC), IEEE, 2018, pp. 1–7.

[37] R. Tanabe, A. S. Fukunaga, Improving the search performance of shade using linear population size reduction, in: 2014 IEEE congress on evolutionary computation (CEC), IEEE, 2014, pp. 1658–1665.

[38] A. A. Hadi, A. W. Mohamed, K. M. Jambi, Single-objective realparameter optimization: Enhanced lshade-spacma algorithm, Heuristics for optimization and learning (2021) 103–121.

[39] L. Stripinis, J. Žilinskas, L. G. Casado, R. Paulavičius, On MATLAB experience in accelerating DIRECT-GLce algorithm for constrained global optimization through dynamic data structures and parallelization, Applied Mathematics and Computation 390 (2021) 125596. doi:10.1016/j.amc.2020.125596.

[40] A. Tavassoli, K. H. Hajikolaei, S. Sadeqi, G. G. Wang, E. Kjeang, Modification of DIRECT for high-dimensional design problems, Engineering Optimization 46 (6) (2014) 810–823. doi:10.1080/0305215X.2013.800057.

[41] J. Mockus, R. Paulavičius, D. Rusakevičius, D. Šešok, J. Žilinskas, Application of Reduced-set Pareto-Lipschitzian Optimization to truss optimization, Journal of Global Optimization 67 (1-2) (2017) 425–450. doi:10.1007/s10898-015-0364-6.

[42] J. M. Gablonsky, C. T. Kelley, A locally-biased form of the DIRECT algorithm, Journal of Global Optimization 21 (1) (2001) 27–37. doi:10.1023/A:1017930332101.

[43] L. Stripinis, R. Paulavičius, J. Žilinskas, Improved scheme for selection of potentially optimal hyper-rectangles in DIRECT, Optimization Letters 12 (2018) 1699–1712. doi:10.1007/s11590-017-1228-4.

[44] Q. Tao, X. Huang, S. Wang, L. Li, Adaptive block coordinate DIRECT algorithm, Journal of Global Optimization 69 (2017) 797–822. doi:10.1007/s10898-017-0541-x.

[45] D. R. Jones, The DIRECT global optimization algorithm, in: C. A. Floudas, P. M. Pardalos (Eds.), The Encyclopedia of Optimization, Kluwer Academic Publishers, Dordrect, 2001, pp. 431–440.

[46] G. Liuzzi, S. Lucidi, V. Piccialli, A DIRECT-based approach exploiting local minimizations for the solution for large-scale global optimization problems, Computational Optimization and Applications 45 (2) (2010) 353–375. doi:10.1007/s10589-008-9217-2.

[47] R. Paulavičius, Y. D. Sergeyev, D. E. Kvasov, J. Žilinskas, Globallybiased BIRECT algorithm with local accelerators for expensive global optimization, Expert Systems with Applications 144 (2020) 113052. doi:10.1016/j.eswa.2019.113052.

[48] L. Stripinis, R. Paulavičius, J. Žilinskas, Penalty functions and twostep selection procedure based DIRECT-type algorithm for constrained global optimization, Structural and Multidisciplinary Optimization 59 (2019) 2155–2175. doi:10.1007/s00158-018-2181-2.

[49] D. Finkel, C. T. Kelley, An adaptive restart implementation of DIRECT, Tech. Rep. CRSC-TR04-30, North Carolina State University. Center for Research in Scientific Computation, online; accessed: 2023-11-08 (2004).

[50] Q. Liu, J. Zeng, G. Yang, MrDIRECT: a multilevel robust DIRECT algorithm for global optimization problems, Journal of Global Optimization 62 (2015) 205–227. doi:10.1007/s10898-014-0241-8.

[51] Q. Liu, G. Yang, Z. Zhang, J. Zeng, Improving the convergence rate of the direct global optimization algorithm, Journal of Global Optimization 67 (2017) 851–872. doi:10.1007/s10898-016-0447-z.

[52] D. E. Finkel, C. T. Kelley, Additive scaling and the DIRECT algorithm, Journal of Global Optimization 36 (4) (2006) 597–608. doi:10.1007/s10898-006-9029-9.

[53] Q. Liu, Linear scaling and the direct algorithm, Journal of Global Optimization 56 (2013) 1233–1245. doi:10.1007/s10898-012-9952-x.

[54] C. A. Baker, L. T. Watson, B. Grossman, W. H. Mason, R. T. Haftka, Parallel global aircraft configuration design space exploration, in: A. Tentner (Ed.), High Performance Computing Symposium 2000, Soc. for Computer Simulation Internat, 2000, pp. 54–66.

[55] J. Mockus, On the Pareto optimality in the context of Lipschitzian optimization, Informatica 22 (4) (2011) 521–536. doi:10.15388/Informatica.2011.340.

[56] L. Stripinis, R. Paulavičius, An empirical study of various candidate selection and partitioning techniques in the DIRECT framework, Journal of Global Optimization 88 (2024) 723–753. doi:10.1007/s10898- 022-01185-5.

[57] L. Stripinis, R. Paulavičius, Lipschitz-inspired HALRECT algorithm for derivative-free global optimization, Journal of Global Optimization 88 (2024) 139–169. doi:10.1007/s10898-023-01296-7.

[58] R. Paulavičius, L. Chiter, J. Žilinskas, Global optimization based on bisection of rectangles, function values at diagonals, and a set of lipschitz constants, Journal of Global Optimization 71 (2018) 5–20. doi:10.1007/s10898-016-0485-6.

[59] Y. D. Sergeyev, D. E. Kvasov, Global search based on eficient diagonal partitions and a set of lipschitz constants, SIAM Journal on Optimization 16 (2006) 910–937. doi:10.1137/040621132.

[60] N. Guessoum, L. Chiter, Diagonal partitioning strategy using bisection of rectangles and a novel sampling scheme, MENDEL 29 (12 2023). doi:10.13164/mendel.2023.2.131.

[61] R. Paulavičius, J. Žilinskas, Simplicial lipschitz optimization without the lipschitz constant, Journal of Global Optimization 59 (2014) 23–40. doi:10.1007/s10898-013-0089-3.

[62] K. Holmström, M. M. Edvall, The TOMLAB Optimization Environment, Springer US, Boston, MA, 2004, pp. 369–376. doi:10.1007/978-1-4613-0215-5\_19.

[63] L. Stripinis, R. Paulavičius, DIRECTGO: A new DIRECTtype MATLAB toolbox for derivative-free global optimization, ACM Transactions on Mathematical Software 48 (dec 2022). doi:10.1145/3559755.

[64] L. Stripinis, J. Kůdela, R. Paulavičius, DIRECTGOLib - DIRECT global optimization test problems library, pre-release v2.0 (2023). URL https://github.com/blockchain-group/DIRECTGOLib

[65] L. Stripinis, R. Paulavičius, A new DIRECT-GLh algorithm for global optimization with hidden constraints, Optimization Letters 15 (2021) 1865–1884. doi:10.1007/s11590-021-01726-z.

[66] N. Hansen, S. Finck, R. Ros, A. Auger, Real-parameter blackbox optimization benchmarking 2009: Noiseless functions definitions, Research Report RR-6829, INRIA (2009). URL https://inria.hal.science/inria-00362633

[67] C. Doerr, H. Wang, F. Ye, S. van Rijn, T. Bäck, Iohprofiler: A benchmarking and profiling tool for iterative optimization heuristics, arXiv e-prints:1810.05281 (oct 2018). arXiv:1810.05281. URL https://arxiv.org/abs/1810.05281

[68] N. Hansen, A. Auger, R. Ros, O. Mersmann, T. Tušar, D. Brockhof, Coco: A platform for comparing continuous optimizers in a black-box setting, Optimization Methods and Software 36 (1) (2021) 114–144.

[69] A. R. Conn, K. Scheinberg, L. N. Vicente, Introduction to Derivative-Free Optimization, SIAM, Philadelphia, PA, 2009. doi:10.1137/1.9780898718768.

[70] A. Al-Dujaili, S. Suresh, A naive multi-scale search algorithm for global optimization problems, Information Sciences 372 (2016) 294–312. doi:https://doi.org/10.1016/j.ins.2016.07.054.

[71] T. M. Inc., Matlab version: 9.14.0 (r2023a) (2023). URL https://www.mathworks.com

[72] J. Kůdela, R. Matousek, New benchmark functions for single-objective optimization based on a zigzag pattern, IEEE Access 10 (2022) 8262– 8278. doi:10.1109/ACCESS.2022.3144067.

[73] L. Abdesslem, New hard benchmark functions for global optimization, mATLAB Central File Exchange. Retrieved February 18, 2022. (2022). URL https://www.mathworks.com/matlabcentral

[74] J. J. Liang, B. Y. Qu, P. N. Suganthan, Problem definitions and evaluation criteria for the cec 2014 special session and competition on single objective real-parameter numerical optimization, Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore 635 (2) (2013).

[75] G. Wu, R. Mallipeddi, P. N. Suganthan, Problem definitions and evaluation criteria for the cec 2017 competition on constrained realparameter optimization, National University of Defense Technology, Changsha, Hunan, PR China and Kyungpook National University, Daegu, South Korea and Nanyang Technological University, Singapore, Technical Report (2017).

[76] L. Stripinis, J. Kůdela, R. Paulavičius, Two novel instance selection methods combining algorithm performance and landscape analysis: A comparative study in continuous optimization, IEEE Transactions on Cybernetics 56 (3) (2026) 1202–1215. doi:10.1109/TCYB.2025.3625095.

[77] G. Cenikj, G. Petelin, C. Doerr, P. Korošec, T. Eftimov, Dynamorep: Trajectory-based population dynamics for classification of blackbox optimization problems, in: Proceedings of the Genetic and Evolutionary Computation Conference, GECCO ’23, Association for Computing Machinery, New York, NY, USA, 2023, pp. 813–821. doi:10.1145/3583131.3590401.

[78] K. Dietrich, D. Vermetten, C. Doerr, P. Kerschke, Impact of training instance selection on automated algorithm selection models for numerical black-box optimization (2024). arXiv:2404.07539.

[79] O. Mersmann, B. Bischl, H. Trautmann, M. Preuss, C. Weihs, G. Rudolph, Exploratory landscape analysis, in: Proceedings of the 13th Annual Conference on Genetic and Evolutionary Computation, GECCO ’11, Association for Computing Machinery, New York, NY, USA, 2011, pp. 829–836. doi:10.1145/2001576.2001690.

[80] E. D. Dolan, J. J. More, Benchmarking optimization software with performance profiles, Mathematical Programming 91 (2002) 201–213. doi:10.1007/s101070100263.

[81] J. J. Moré, S. M. Wild, Benchmarking derivative-free optimization algorithms, SIAM Journal on Optimization 20 (1) (2009) 172–191. doi:10.1137/080724083.

[82] M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (200) (1937) 675–701. doi:10.1080/01621459.1937.10503522.