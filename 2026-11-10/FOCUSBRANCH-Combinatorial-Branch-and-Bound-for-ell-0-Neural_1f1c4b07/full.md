# FOCUSBRANCH: Combinatorial Branch-and-Bound for $\ell _ { 0 }$ Neural Network Robustness Verification

Renwei Deng Yang He Linyi Li<sup>∗</sup> Yuepeng Wang<sup>∗</sup> Simon Fraser University {rda102, yha244, linyi\_li, yuepeng}@sfu.ca

## Abstract

Verifying the local robustness of neural networks under $\ell _ { 0 }$ perturbations is inherently combinatorial, as it must reason over all sparse subsets of ϵ perturbed pixels. This paper presents FOCUSBRANCH, a combinatorial branch-and-bound framework for $\ell _ { 0 }$ robustness verification. The key insight is to make the $\ell _ { 0 } .$ -ball decomposable and branchable using a node abstraction, which consists of a focus pixel set S and a level l, permitting at most ϵ − l perturbed pixels outside S. This abstraction unifies two branching mechanisms: concrete nodes branch through upper shadows to fix additional perturbed locations, while super nodes branch through covering designs to cover many perturbation cases with fewer child nodes. FOCUSBRANCH proves node safety by concretizing linear relaxations over nodeconstrained $\ell _ { 0 }$ -balls and recursively branches on unknown nodes. Rather than relying on a fixed verification plan, it samples nodes across levels to estimate safe rates and costs, then selects a starting level and branching strategy predicted to minimize verification time. Experiments on fully connected and convolutional classifiers for MNIST, Fashion-MNIST, and CIFAR-10 show that FOCUSBRANCH verifies $\ell _ { 0 }$ robustness effectively across architectures and outperforms state-of-the-art verifiers. The code is available at: https://github.com/simpl-group/FocusBranch.

## 1 Introduction

Local robustness provides formal guarantees that a neural network’s prediction is stable under a class of input perturbations [46, 67, 73]. For image classification, robustness to sparse perturbations is especially important, because modifying only a few carefully-chosen pixels can cause misclassification [5, 34, 54]. Therefore, we study local robustness verification against an $\ell _ { 0 }$ -bounded perturbation. In particular, given a neural network, an input image, and a perturbation budget ϵ, we aim to prove that the prediction of the network remains unchanged for every image obtained by modifying at most ϵ pixels. Unlike the widely studied $\ell _ { \infty } { - } \mathrm { n o r m }$ setting [18, 19, 27, 46, 59], ℓ -norm verification presents unique challenges. The verifier must navigate through as large as $\binom { V } { \epsilon }$ sets of perturbed pixels (about $2 . 4 \times 1 0 ^ { 1 2 }$ for a $2 8 \times 2 8$ image at ϵ = 5) while keeping the affine bounds tight enough.

Existing $\ell _ { 0 }$ robustness verifiers offer different solutions. On the one hand, auto\_LiRPA [67] derives a closed-form concretization solution by selecting the top-ϵ most-contributing pixels, which can be integrated with LiRPA-style relaxation propagation pipeline (DeepPoly [46]/CROWN [73]) to compute bounds, but the bounds are too loose to verify under large ϵ. On the other hand, CAL-ZONE [41] and COVERD [43] run branch-and-bound by recursively decomposing the perturbable pixel sets via covering design [39] and verify subproblems with smaller perturbable sets using a box verifier [31]. A more recent verifier COVERD+top-t-GP [42] combines COVERD with auto\_LiRPA’s $\ell _ { 0 }$ concretization (the top-ϵ method) such that those decomposed subproblems are verified over ℓ<sub>0</sub>-balls instead of boxes. Because every subproblem must contain all perturbed pixels in one set, these verifiers are effective when large sets can be verified, so that few sets cover all perturbation cases.

![](images/97937be435c7e6ee3609ccca97ada2fe2087a5faee9faec20febeca3a7b6f117.jpg)  
Figure 1: Overview of FOCUSBRANCH. After choosing $l _ { s }$ and $\mathcal { R } ^ { ( l _ { s } ) }$ by sampling, FOCUSBRANCH runs branch-and-bound over node-constrained $\ell _ { 0 } { \mathrm { - } } \mathbf { b } \mathrm { { a l l s } } .$ , branching via covering designs (super node) or upper shadows (concrete node). Unknown, unbranchable nodes fall back to MILP.

We introduce FOCUSBRANCH, a combinatorial branch-and-bound framework to verify local robustness of neural networks under $\ell _ { 0 }$ perturbations. The key of FOCUSBRANCH is the node abstraction, consisting of a focus pixel set and its level, restricting at most how many perturbed pixels outside focus set. This abstraction naturally captures two different kinds of nodes: super nodes, where the focus set size is larger than the level, and concrete nodes, where the focus set size equals its level. FOCUSBRANCH reduces $\ell _ { 0 }$ robustness verification to proving the robustness of certain nodes. When a node cannot be verified directly by margin lower bounds, we can branch on it to decompose it into several child nodes that are easier to verify. Specifically, concrete branching fixes one additional perturbation location, while super branching uses covering designs [12, 39, 51] to refine [41] the focus set and compactly cover many perturbation cases with fewer generated nodes.

As shown in Figure 1, FOCUSBRANCH does not commit in advance to a fixed verification plan. Instead, it samples nodes across levels, estimates their safe rates and costs, and selects the best starting level and super branching strategy with the smallest predicted running time to initialize the branch-and-bound process. In fact, starting at a small level can substantially reduce the number of nodes to be checked: if a smaller-level node is verified safe, all larger-level nodes it covers can be safely pruned. Restricting to start at level ϵ recovers the decomposition used in prior work [42].

We evaluated FOCUSBRANCH on 13 neural networks for image classification, using 100 images collected from MNIST [22], Fashion-MNIST [66], and 10 images from CIFAR-10 [21]. The results show that FOCUSBRANCH is effective and efficient for $\ell _ { 0 }$ robustness verification. In particular, at $\epsilon = 5$ it verifies 87.2% of the candidate instances on fully connected networks (75 s on average) and 85.9% on convolutional ones (151 s), versus 42.5% and 61.4% for the strongest baseline [42].

Contributions. To summarize, we make the following main contributions in this paper:

• We introduce a node-based combinatorial decomposition of the $\ell _ { 0 }$ -ball, using upper shadows for concrete nodes and covering designs for super nodes to generate node-constrained robustness verification subproblems.

• We derive closed-form concretization solutions over node-constrained $\ell _ { 0 }$ -balls, allowing linearrelaxation-based approaches to bound over nodes without enumerating perturbation subsets.

• We extend the sampling-based planning technique [41, 43] across levels, which estimates safe rates and costs of all-level nodes and selects the best plan predicted to minimize verification time.

• We present FOCUSBRANCH, a branch-and-bound verification framework for $\ell _ { 0 }$ robustness verification. Our evaluation on 13 neural networks and 100 images from various datasets shows that FOCUSBRANCH outperforms state-of-the-art verifiers.

## 2 Preliminaries

Neural networks. Let $f : \mathcal { D } \to \mathbb { R } ^ { N }$ be a neural network for image classification that maps an input $\mathbf { x } \in \mathcal { D }$ to a vector of class scores. Here, the domain D is a box $[ \underline { { \mathbf { x } } } , \mathbf { \overline { { x } } } ] \subseteq \mathbb { R } ^ { V C }$ , where V denotes the number of pixels and C the number of channels. In general, we write $\mathbf { x } = [ \mathbf { x } _ { 1 } ; \ldots ; \mathbf { x } _ { V } ]$ , where each $\mathbf { x } _ { v } \in \mathbb { R } ^ { C }$ corresponds to pixel v. For example, a normalized $2 8 \times 2 8$ grayscale image lies in $[ 0 , 1 ] ^ { 7 8 4 }$ while a normalized $3 2 \times 3 2$ RGB image lies in $[ 0 , 1 ] ^ { 3 0 7 2 }$ . The predicted label of image x by network $f$ is defined as arg $\operatorname* { m a x } _ { j } f _ { j } ( \mathbf { x } )$ , where $f _ { j } ( { \bf x } )$ is the j-th entry of $f ( \mathbf { x } )$ .

Adversarial attacks and perturbation. An adversarial attack considers a nominal input $\mathbf { x } ^ { \star } \in \mathcal { D }$ and seeks a perturbation $\pmb { \delta }$ such that the perturbed input $\mathbf { x } ^ { \star } { + } \delta$ is misclassified, i.e. arg max $f _ { j } ( \mathbf { x } ^ { \star } ) \neq$ arg max<sub>j</sub> $f _ { j } ( \mathbf { x } ^ { \star } + \delta )$ , while remaining within $\mathbf { x } ^ { \star } + \pmb { \delta } \in \mathcal { D }$ . We consider local robustness against few-pixel attacks [30], where the adversary may modify at most ϵ pixels. The integer ϵ is called the perturbation budget and $\mathcal { V } = \{ 1 , \ldots , V \}$ the pixel index set. The perturbation $\pmb { \delta } \in \mathbb { R } ^ { V C }$ satisfies the pixel-wise $\ell _ { 0 }$ constraint $| \{ v \in \mathcal { V } : \delta _ { v } \neq \mathbf { 0 } \} | \le \epsilon$ , where $\pmb { \delta } _ { v } \in \mathbb { R } ^ { C }$ is the perturbation on pixel $v .$ . The space of admissible perturbed inputs $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ defines a pixel-wise ℓ -ball centered at $\mathbf { x } ^ { \star }$ with radius ϵ:

$$
\mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : = \{ \mathbf { x } \in \mathcal { D } \ : \ \left| \{ v \in \mathcal { V } : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \} \right| \leq \epsilon \}\tag{1}
$$

$\ell _ { 0 }$ local robustness and affine bounds. A network $f$ is said to be $\ell _ { 0 }$ locally robust w.r.t. $\mathbf { x } ^ { \star }$ and ϵ if it assigns the same label to all inputs over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ , i.e., for all $\mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { \dot { x } } ^ { \star } )$ , arg max ${ \bf \nabla } _ { j } f _ { j } ( { \bf x } ) = { \bf \nabla }$ arg max<sub>j</sub> ${ \bf \nabla } _ { j } f _ { j } ( { \bf x } ^ { \star } )$ . To verify this property, it suffices to ensure that the predicted class $y ^ { \star } : =$ arg $\mathrm { m a x } _ { j } f _ { j } ( \mathbf { x } ^ { \star } )$ remains dominant over all other classes throughout $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ , i.e.,

$$
\operatorname* { m i n } _ { j \neq y ^ { \star } } \operatorname* { m i n } _ { \mathbf x \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf x ^ { \star } ) } m _ { y ^ { \star } , j } ( \mathbf x ) > 0\tag{2}
$$

where $m _ { y ^ { \star } , j } : = f _ { y ^ { \star } } - f _ { j }$ is the margin function. We adopt Linear Relaxation-based Perturbation Analysis (LiRPA) [67] to compute the concrete lower bounds of these margins. In particular, LiRPA can be applied directly to $m _ { y ^ { \star } , j } ,$ yielding an affine margin lower bound $\underline { m } _ { y ^ { \star } , j } ( { \bf x } )$ such that for all $\mathbf { x } \in { \mathcal { B } } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) , { \underline { { m } } } _ { y ^ { \star } , j } ( \mathbf { x } ) \leq f _ { y ^ { \star } } ( \mathbf { x } ) - f _ { j } ( \mathbf { x } )$ . Thus, a sufficient condition for Eq. (2) is

$$
\operatorname* { m i n } _ { j \neq y ^ { \star } } \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) } \underline { { m } } _ { y ^ { \star } , j } ( \mathbf { x } ) > 0 .\tag{3}
$$

More specifically, for any intermediate affine layer i in the computation graph of $f ,$ let $h ^ { ( i ) } ( \mathbf { x } )$ denote its value. There exist affine lower and upper bounds of $h ^ { ( i ) } ( \mathbf { x } )$ w.r.t. input $\mathbf { x } ,$ namely $\underline { { h } } ^ { ( i ) }$ and $\overline { { h } } ^ { ( i ) }$ , such that for all $\mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) , \ \underline { { h } } ^ { ( i ) } ( \mathbf { x } ) \leq h ^ { ( i ) } ( \mathbf { x } ) \leq \overline { { h } } ^ { ( i ) } ( \mathbf { x } )$ holds. They are computed by back-propagating linear relaxations layer by layer until x [46, 73]. Moreover, $\underline { { h } } ^ { ( i ) }$ and $\overline { { h } } ^ { ( i ) }$ are then concretized over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) \left[ 6 7 \right]$ (see below) to obtain concrete lower and upper bounds of $h ^ { ( i ) } ( \mathbf { x } )$ before over-approximating its immediate nonlinear activation by linear relaxation [46, 73]. Recursively repeating this process forward through every intermediate affine layer until the margins yields $\underline { m } _ { y ^ { \star } , j } .$ Performing one more concretization gives the concrete lower bound needed to evaluate Eq. (3).

Concretization over the full $\ell _ { 0 } { \bf - b a l l . }$ . When a linear relaxation is back-propagated to the input, the resulting affine form $g ( \mathbf { x } ) = \omega ^ { \top } \mathbf { x } + b$ must be optimized over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ . Since $\bar { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ constrains the number of modified pixels rather than their magnitudes, the minimum (resp. maximum) of $g$ is $g ( \mathbf { x } ^ { \star } )$ minus (resp. plus) the sum of the top-ϵ pixel contributions, where a pixel’s contribution is the largest decrease (resp. increase) it can make to g within its range [4, 67], aggregated over channels [42].

Covering-design-based branch-and-bound. A covering design $\mathcal { C } ( S , k , t )$ is a collection of kelement subsets (blocks) of S such that every t-element subset of S is contained in at least one block [11, 12, 39, 51]. Since every set of $\binom { \bar { V } } { \epsilon }$ lies in some block of $\mathcal { C } ( \nu , k , \epsilon )$ , CALZONE [41] verifies $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ by verifying the box $\mathcal { D } _ { B } : = \left\{ \mathbf { x } \in \mathcal { D } : \forall v \notin B . \mathbf { x } _ { v } = \mathbf { x } _ { v } ^ { \star } \right\}$ of every block B with an incomplete box verifier [31]. Starting from $\nu ,$ a block B that fails is refined by $\mathcal { C } ( B , k _ { | B | } , \epsilon )$ at $\epsilon \le k _ { | B | } < | B |$ , recursively, and a failing block of exactly ϵ pixels is solved by MILP [50]. The refinement size $k _ { | B | }$ for each block size |B| is fixed in advance by dynamic programming that minimizes the expected verification time, using safe rates and costs estimated on sampled blocks of each size [41]. Instead of starting from $\bar { \mathit { C } } ( \nu , \bar { k _ { V } } , \epsilon )$ , COVERD [43] introduces covering verification design (CVD), whose blocks may differ in size, chosen among candidate CVDs by estimating the cost over the predicted block-size distributions. COVERD+top-t-GP [42] inherits COVERD and upgrades the box verifier [31] with auto\_LiRPA’s $\ell _ { 0 }$ -ball concretization [67] (Section 2), such that it verifies each block $B$ over its corresponding $\ell _ { 0 }$ -ball $\{ \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : \forall v \notin B . \mathbf { x } _ { v } = \mathbf { x } _ { v } ^ { \star } \}$ , instead of its box $\mathcal { D } _ { B }$

![](images/9c37328a8af22c32c9ab55be028d1847ce6421960055ae276198b3698e965db8.jpg)

(c) Concrete node: upper shadow  
![](images/1ed596114239f0b624f826e25f5c44dd86c025d6d1182d811d292f6ea5f35142.jpg)

![](images/5c3b573ceea084b9bcc7ec0214751e71d8af9855744d79356e1c04de8314ed3e.jpg)  
Figure 2: (a) A node o admits at most $\epsilon - o . l$ perturbed pixels (dots) outside $_ { o . S }$ (purple). (b) Every 2-subset of $_ { o . S }$ lies in some child. (c) Each child adds one pixel v $\notin o . S$ (orange).

## 3 Verification Algorithm

## 3.1 Node Abstraction

Our verification method uses a combinatorial branch-and-bound approach that leverages the node abstraction to verify no $\ell _ { 0 }$ perturbation on the input image within the given budget ϵ can change the prediction. We define a node $o : = ( S , l )$ as a pair of a focus pixel set $s \subseteq \nu$ and its level $l \in \left\{ 0 , \ldots , \epsilon \right\}$ , such that $| S | \geq l .$ . We use $_ { o . S }$ and o.l to specify the $s$ and l elements of a given node $^ { O , }$ respectively. Node o requires no more than $\epsilon - o . l$ pixels are perturbed outside $_ { o . S }$ . Formally, the admissible perturbed input space for $\mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ constrained by a node o can be defined as follows:

$$
\mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) : = \{ \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : | \left\{ v : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \right\} \backslash o . S | \leq \epsilon - o . l \}\tag{4}
$$

Generally, if ϵ pixels are perturbed, node o requires that at least o.l perturbed pixels come from the focus set $_ { o . S }$ . Note that perturbing more than o.l pixels in $_ { o . S }$ is allowed.

Example 1. Figure 2(a) shows the node $o = \left( \left\{ 1 , 3 , 7 , 8 \right\} , 2 \right)$ with $V = 8$ and $\epsilon = 4$ admitting at most $\epsilon - o . l = 2$ perturbed pixels outside $o . S . \mathbf { x } ^ { \prime } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ perturbs {1, 3, 5, 6} with only 5 and 6 lie outside $_ { o . S }$ , whereas $\mathbf { x } ^ { \prime \prime } \notin B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ perturbs $\{ 2 , 4 , 5 , 7 \}$ with more than two pixels lie outside $_ { o . S }$

## 3.2 Verifying Local Robustness using Branch-and-Bound

In order to verify $f$ is locally robust, we first define its local robustness on one node. Similar to $( 2 ) , f$ is said to be locally robust on o (or simply called o is safe), denoted by $\mathrm { S A F E } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o )$ , if $y ^ { \star }$ remains dominant over all other classes throughout $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) , \mathrm { i . e . }$

$$
\mathtt { S A F E } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) : = \operatorname* { m i n } _ { j \neq y ^ { \star } } \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } m _ { y ^ { \star } , j } ( \mathbf { x } ) > 0\tag{5}
$$

To prove node o is safe, we can compute the concrete lower bounds of the margins in Eq. (5) (see Section 3.4). We define the process as a function of nodes (Eq. (6)). Therefore, a sufficient condition for o being safe is $3 0 \mathrm { U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) > 0$ (Lemma E.3). By contrast, a node is said to be unknown if $\mathbf { B } \mathbf { O U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) \leq 0$

$$
\mathbf { B o u N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) : = \operatorname* { m i n } _ { j \neq y ^ { \star } } \operatorname* { m i n } _ { \substack { \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } } \underline { { m } } _ { { y } ^ { \star } , j } ( \mathbf { x } )\tag{6}
$$

We define $\Theta _ { l }$ as the initial node for each level $l ,$ where $\Theta _ { 0 } : = ( \varnothing , 0 )$ and $\Theta _ { l } : = ( \nu , l )$ for $l > 0$ . To prove f is robust over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ , it suffices to prove the safety of any initial node $\Theta _ { l }$ (Lemma E.4). However, directly proving the safety of $\Theta _ { l }$ may be difficult. To address this issue, we use branchand-bound to recursively decompose an unknown node o into child nodes using a branching operator BRANCH(o), which must be sound (Corollary 1): if all nodes in BRANCH(o) are safe, then o is also safe. This decomposes a difficult node-constrained verification problem into smaller subproblems.

Following this methodology, we design the verification procedure shown in Algorithm 1. Given a network $f ,$ a nominal input $\mathbf { x } ^ { \star }$ , and a perturbation budget ϵ, the algorithm first selects a starting level $l _ { s }$ and a super branching strategy $\bar { \mathcal { R } } ^ { ( l _ { s } ) }$ predicted to minimize overall verification time using sampling statistics (Line 2, as detailed in Section 3.5). The algorithm then proceeds with a worklist procedure. Specifically, the worklist O is initialized with an initial node $\Theta _ { l , s }$ (Line 3). For each node $o \in { \mathcal { O } } .$ , the algorithm first checks whether o is safe with BOU $\nabla \mathrm { D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) > 0 .$ . If so, o is pruned and the algorithm proceeds to the next node. Otherwise, o is unknown and is decomposed according to its type (see Section 3.3) and adds the generated nodes into O (Lines 7 – 10). If o is unbranchable (level-ϵ concrete, see Section 3.3), the corresponding safety verification problem is encoded as a Mixed-Integer Linear Programming (MILP) problem [50] in order to search for an adversarial example within the input box induced by ϵ-level concrete o. If such an example $\mathbf { x } ^ { \mathrm { a d v } }$ is found, then the network $f$ is unsafe (Lines 12 – 13). Otherwise, the algorithm continues until O is empty, at which point the safety of $\Theta _ { l _ { s } }$ is proven and the robustness of f over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ is verified (Lemma E.4).

Algorithm 1 Verifying $\ell _ { 0 }$ local robustness for network f over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ using FOCUSBRANCH   
1: procedure FOCUSBRANCH $f , \mathbf { x } ^ { \star } , \epsilon )$   
Input: A classifier $f , \mathrm { a }$ nominal input image $\mathbf { x } ^ { \star }$ , and a perturbation budget ϵ   
Output: Verified if the verification is successful, an adversarial example $\mathbf { x } ^ { \mathrm { a d v } }$ otherwise   
2: $\bar { ( l _ { s } , \mathcal { R } ^ { ( l _ { s } ) } ) } \gets \mathrm { C H o o s E B E S T P L A N } ( f , \mathbf { x } ^ { \star } , \epsilon )$   
3: $\mathcal { O }  \{ \Theta _ { l _ { s } } \}$ ▷ Enqueue an initial node at level $l _ { s }$   
4: while $\supsetneq \varnothing$ do   
5: $o  \mathcal { O } .$ .dequeue $( )$   
6: if BOUND $f , \mathbf { x } ^ { \star } , \epsilon  ( o ) \leq 0$ then   
7: $\mathbf { i f } \left| o . S \right| > \operatorname { \partial } O . l$ then ▷ Branch on unknown super node   
8: $\mathcal { O }  \mathcal { O } \cup \mathop { \mathrm { B R A N C H } } _ { \mathcal { R } ^ { ( l _ { s } ) } [ | o . S | ] } ^ { \operatorname* { s u p } } ( o )$   
9: else if $o . l < \epsilon$ then ▷ Branch on unknown concrete node   
10: $\mathcal { O }  \mathcal { O } \cup \mathbf { B R A N C H } ^ { \mathsf { c o n } } ( o )$   
11: else ▷ Use MILP to verify unbranchable unknown node   
12: $( \mathtt { f e a s i b l e } , \mathbf { x ^ { a d v } } ) \gets \mathtt { M I L P } \left( f , \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) \right)$   
13: if feasible then return $\mathbf { x } ^ { \mathrm { a d v } }$   
14: return Verified

Theorem 1 (Relative soundness). Suppose the MILP solver is sound. Given a network $f ,$ a nominal input $\mathbf { x } ^ { \star }$ , and a maximum number ofperturbed pixels $\epsilon ,$ if FOCUSBRANCH $( f , \mathbf { x } ^ { \star } , \epsilon )$ returns Verified, then $f$ is locally robust with respect to $\mathbf { x } ^ { \star }$ and ϵ.

Theorem 2 (Relative completeness). Suppose the MILP solver is sound and complete. Given a network $f ,$ , a nominal input $\mathbf { x } ^ { \star } { } _ { \mathrm { : } }$ , and a maximum number ofperturbed pixels $\epsilon , i f f$ is locally robust with respect to $\mathbf { x } ^ { \star }$ and $\epsilon ,$ then FOCUSBRANCH $( f , \mathbf { x } ^ { \star } , \epsilon )$ returns Verified.

## 3.3 Branching Policies

The node abstraction enables a combinatorial decomposition of the $\ell _ { 0 }$ -ball beyond explicit perturbable set branching: branching an unknown node o replaces the node-constrained ball $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ by a collection of child node-constrained balls whose union equals exactly their parent’s node-constrained ball (Corollary 2), so proving every child safe proves o safe (Corollary 1).

1. A node o is a super node if its focus set size is larger than its level, i.e., $\left| o . S \right| > o . l .$ . Our branching policy for super nodes decomposes o by applying the covering design refinement (Section 2) to its focus set $o . S \colon$

$$
\mathrm { B R A N C H } _ { k } ^ { \operatorname* { s u p } } ( o ) : = \{ ( S ^ { \prime } , o . l ) : \ S ^ { \prime } \in \mathcal C ( o . S , k , o . l ) \}\tag{7}
$$

Here k is chosen by the strategy planner (see Section $3 . 5 )$ that satisfies $o . l \le k < | o . S |$ . Thus, when $k > o . l$ , one generated child super node can cover many concrete nodes at level o.l.

2. The node o is a concrete node, if the focus set size equals the level, i. $\mathbf { e } . , | o . S | = o . l$ . Our branching policy for concrete nodes decomposes o by the upper shadow of $o . S .$ , i.e., its immediate supersets:

$$
\operatorname { B R A N C H } ^ { \operatorname { c o n } } ( o ) : = \{ ( S ^ { \prime } , o . l + 1 ) : \ S ^ { \prime } \in \{ o . S \cup \{ v \} : v \in \mathcal { V } \setminus o . S \} \}\tag{8}
$$

Intuitively, each child node fixes one additional perturbed location $v \in \mathcal { V } \setminus o . S$ to the focus set, which increases the level by one. In practice, concrete branching is more efficient, and a pruning technique can be applied afterwards (detailed in Appendix F.3). Note that concrete nodes at level ϵ are unbranchable because the level cannot be increased further. Otherwise, we call o branchable $\mathrm { i f } \ | o . S | > o . l \ \mathrm { o r } \ | o . S | = o . l < \epsilon .$

Example 2. Figure 2(b) branches the super node $o = \left( \left\{ 1 , 3 , 7 , 8 \right\} , 2 \right)$ with $k = 3$ . The covering design $\mathcal { C } ( o . S , \bar { 3 , } 2 ) = \{ \{ 1 , 3 , 7 \} , \{ 1 , 3 , 8 \} , \{ 3 , 7 , 8 \} \}$ contains every 2-subset of $_ { o . S }$ in some block, so BRANC $\mathrm { H } _ { 3 } ^ { \mathsf { s u p } } ( o )$ yields three children at level 2, and each child covers three concrete nodes, $\mathrm { e . g . , ( \{ 1 , 3 , \dot { 7 } \} , \dot { 2 } ) }$ covers $( \left\{ 1 , 3 \right\} , 2 ) , ( \left\{ 1 , 7 \right\} , 2 )$ and $( \{ 3 , 7 \} , 2 )$ . Figure 2(c) branches the concrete node $\overset { \cdot } { o } = \left( \left\{ 1 , 3 \right\} , 2 \right)$ , and BR $\overset { \cdot } { \operatorname { A N C H } } ^ { \mathsf { c o n } } ( o )$ yields six level-3 children $( \left\{ 1 , 3 , v \right\} , 3 )$ , one per pixel $v \in \mathcal { V } \setminus o . \dot { S } = \{ 2 , 4 , 5 , 6 , 7 , 8 \}$

Each branching step either strictly shrinks the focus set or strictly increases the level and focus set by one, so the recursion is finite and in the worst case terminates at level-ϵ concrete nodes (Lemmas E.8 and E.10).

![](images/632766cbbc885504b1775f9fc2216a5024cc97f71e1b390dc5240a6830769570.jpg)  
Figure 3: Closed-form node bound with illustrative contributions $\overline { { \Delta } } _ { v }$ . The node bound takes $P \cup Q$ whereas the full $\ell _ { 0 } { \mathrm { - } } \mathbf { b } \mathrm { { a l l } }$ takes the top-ϵ pixels {2, 4, 5, 6}, so the node bound is tighter.

## 3.4 Bounding a Node

A critical step in Algorithm 1 is to prove the safety of each node o with BOUND $\mathbf { \Phi } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) > 0$ early, thereby avoiding unnecessary branching (Line 6). The concrete margin bounds in Eq. (6) are obtained through the same LiRPA-style pipeline as in Section 2, except that each concretization step, for the intermediate affine layers as well as for the margins, is performed over the node-constrained $\ell _ { 0 } { \mathrm { - } } \boldsymbol { \mathrm { b a l l } }$ $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ rather than the full $\ell _ { 0 } { \ - } \mathbf { b } \mathbf { a } \mathbf { l } \mathbf { l } \ B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ . This change preserves soundness, i.e., BO $\mathrm { U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o )$ remains a valid concrete lower bound of the margins over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ (Lemma E.2). What remains is to perform this concretization without enumerating the perturbation subsets that o covers.

We now derive a closed-form solution for concretizing any affine function $g ( \mathbf { x } ) = \omega ^ { \mathsf { T } } \mathbf { x } + b$ over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ , starting with ma $\mathbf { \tilde { x } } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) \ g ( \mathbf { x } )$ . Write $\mathbf { x } = [ \mathbf { x } _ { 1 } ; \ldots ; \mathbf { x } _ { V } ]$ and $\omega = [ \omega _ { 1 } ; \ldots ; \omega _ { V } ]$ , where $\mathbf { x } _ { v } , \boldsymbol { \omega } _ { v } \in \mathbb { R } ^ { C }$ . Then $\begin{array} { r } { g ( \mathbf { x } ) = g ( \mathbf { x } ^ { \star } ) + \sum _ { v = 1 } ^ { V } \Delta _ { v } . } \end{array}$ , where $\begin{array} { r } { \Delta _ { v } : = \omega _ { v } ^ { \top } ( \mathbf { x } _ { v } - \mathbf { x } _ { v } ^ { \star } ) = \sum _ { c = 1 } ^ { C } \omega _ { v , c } ( \mathbf { x } _ { v , c } - \mathbf { x } _ { v , c } ^ { \star } ) } \end{array}$ is the contribution of pixel v that aggregates the contribution of all its channels. Since $g ( \mathbf { x } ^ { \star } )$ is fixed, maximizing $g ( \mathbf { x } )$ over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ reduces to selecting the pixel contributions that can be made largest under the node constraint. For any pixel v, its largest possible contribution can be computed by

$$
\overline { { \Delta } } _ { v } : = \operatorname* { m a x } _ { \substack { \mathbf { x } _ { v } \in [ \mathbf { x } _ { v } , \overline { { \mathbf { x } } } _ { v } ] } } \Delta _ { v } = \sum _ { c = 1 } ^ { C } \omega _ { v , c } ( \mathbf { x } _ { v , c } ^ { \prime } - \mathbf { x } _ { v , c } ^ { \star } ) , \quad \mathbf { x } _ { v , c } ^ { \prime } : = \left\{ \overline { { \mathbf { x } } } _ { v , c } , \quad \omega _ { v , c } \geq 0 , \right.\tag{9}
$$

That is, each channel is pushed to its upper (resp. lower) bound when the coefficient is positive (resp.   
negative) to maximize its contribution before aggregation.

Since $\overline { { \Delta } } _ { v } \geq 0$ , the optimal perturbation uses all budgets, i.e., perturbs exactly ϵ pixels, and the o-constrained ball becomes $\left\{ \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : ~ | \left\{ v : ~ \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \right\} \cap \bar { o } . S | \geq o . l \right\}$ . By Lemma E.1, the maximum of $g ( \mathbf { x } )$ over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ admits the closed-form

$$
\operatorname* { m a x } _ { \substack { \mathbf { x } \in B _ { 0 } ^ { \star } ( \mathbf { x } ^ { \star } , o ) } } g ( \mathbf { x } ) = g ( \mathbf { x } ^ { \star } ) + \operatorname* { m a x } _ { \substack { \mathbf { x } \in B _ { 0 } ^ { \star } ( \mathbf { x } ^ { \star } , o ) } } \sum _ { v = 1 } ^ { V } \Delta _ { v } = g ( \mathbf { x } ^ { \star } ) + \sum _ { \substack { v \in \overline { { A } } ( o , l , o , S ) \cup \overline { { A } } \left( \epsilon - o , l , \gamma \backslash \overline { { A } } \left( o , l , o , S \right) \right) } } \overline { { \Delta } } _ { v } ,\tag{10}
$$

where ${ \overline { { \mathcal { A } } } } ( a , A ) : = \arg { \mathrm { t o p } } - a _ { v \in A } { \overline { { \Delta } } } _ { \iota }$ is a set of a $( a \in \{ 0 , \ldots , \epsilon \} )$ most positively contributed pixels from set $A \subseteq \nu$ . Similarly, the closed-form solution for mi $\begin{array} { r } { \boldsymbol { 1 } _ { \mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } g ( \mathbf { x } ) } \end{array}$ can be derived by Eq. (12) in Appendix D.

Example 3. Figure 3 concretizes g over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ for $o = \left( \left\{ 1 , 3 , 7 , 8 \right\} , 2 \right)$ and $\epsilon = 4$ with illustrative contributions $\bar { \overline { { \Delta } } } _ { v }$ . Let P be the top-o.l pixels of the focus set, $P = \overline { { \mathcal { A } } } ( o . l , o . S ) = \{ 1 , 7 \}$ , and the remaining selected pixels are the top- $\left( \epsilon - o . l \right)$ of $\mathcal { V } \setminus P , \mathrm { i . e . , } \overline { { \mathcal { A } } } ( \epsilon - o . l , \mathcal { V } \setminus P ) = \{ 2 , 4 \}$ , denoted by $Q .$ . By Eq. $( 1 0 ) , \operatorname* { m a x } _ { { \bf x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( { \bf x } ^ { \star } , o ) } g ( { \bf x } ) = g ( { \bf x } ^ { \star } ) + \overline { { \Delta } } _ { 1 } + \overline { { \Delta } } _ { 2 } + \overline { { \Delta } } _ { 4 } + \overline { { \Delta } } _ { 7 }$ . Over the full ball $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ the top-ϵ selection picks $\{ 2 , 4 , 5 , 6 \}$ instead, but all four pixels lie outside $o . S .$ , whereas o admits at most two. Since $\overline { { \Delta } } _ { 5 } + \overline { { \Delta } } _ { 6 } > \overline { { \Delta } } _ { 1 } + \overline { { \Delta } } _ { 7 }$ , the node bound is tighter.

Since $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \subseteq B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ (Lemma E.5), every concretization step performed for a child node $o ^ { \prime }$ is at least as tight as the corresponding step for its parent node o. Tighter intermediate bounds in turn yield tighter relaxations downstream, so in practice BOUND $\begin{array} { r } { f , \mathbf { x } ^ { \star } , \boldsymbol { \epsilon } \bigl ( \bar { O } ^ { \prime } \bigr ) \geq \mathbf { B } \mathrm { O U N D } _ { f , \mathbf { x } ^ { \star } , \boldsymbol { \epsilon } } ( o ) } \end{array}$ . This is exactly what makes branching productive.

Special cases. At level 0, Eq. (10) and Eq. (12) reduce to the top-ϵ selection (Section 2), so FOCUSBRANCH restricted to $\Theta _ { 0 }$ computes auto\_LiRPA’s $\ell _ { 0 }$ bounds [67] (but pixel-wise rather than entry-wise). From $\Theta _ { \epsilon } ,$ a node $( B , \epsilon )$ is exactly the subproblem COVERD+top-t-GP [42] solves for block $B ,$ and super branching is its refinement.

Algorithm 2 Choosing Best Verification Plan   
1: procedure CHOOSEBESTPL $\mathbf { A N } ( f , \mathbf { x } ^ { \star } , \epsilon )$   
Input: A classifier $f ,$ a nominal input image $\mathbf { x } ^ { \star }$ , and a perturbation budget ϵ   
Output: The level $l _ { s }$ to start verification, and a super branching strategy $\mathcal { R } ^ { ( l _ { s } ) }$ at level $l _ { s }$   
2: $( l _ { s } , \mathcal { R } ^ { ( l _ { s } ) } ) \gets ( 0 , \emptyset ) ; \ t _ { m i n } \gets \infty$   
3: $( \hat { \mathbf { P } } , \hat { \mathbf { T } } ) \gets \mathsf { S }$ ampleNodesOfAllShape $( f , \mathbf { x } ^ { \star } , \epsilon )$   
4: for all $l \in \{ 0 , \dot { \ldots } , \epsilon \}$ do   
5: $( \mathcal { R } ^ { ( l ) } , t _ { e s t } ^ { ( l ) } )$ ← GetSuperBranchingStrategyAndEstimate $( \hat { \mathbf { P } } _ { : , l } , \hat { \mathbf { T } } _ { : , l } , l )$   
6: $\begin{array} { r } { t _ { t a i l } \gets \sum _ { i = l + \underline { { 1 } } } ^ { \epsilon } \tilde { \mathbf { t } } _ { i } ; t _ { e s t } \gets t _ { e s t } ^ { ( l ) } + t _ { t a i l } } \end{array}$   
7: if $\ ` t _ { e s t } < t _ { m i n }$ then   
8: $t _ { m i n } \gets t _ { e s t } ; ~ ( l _ { s } , \mathcal { R } ^ { ( l _ { s } ) } ) \gets ( l , \mathcal { R } ^ { ( l ) } )$   
9: return $( l _ { s } , \mathcal { R } ^ { ( l _ { s } ) } )$

## 3.5 Verification Planner

A verification plan is a starting level $l _ { s } \in \{ 0 , \ldots , \epsilon \}$ and a super branching strategy $\mathcal { R } ^ { ( l _ { s } ) } : \mathbb { N }  \mathbb { N }$ mapping a focus-set size to its children’s focus-set size. Algorithm 2 first samples nodes of every focus-set size across levels to estimate safe rates $\hat { \mathbf { P } }$ and mean bounding times (the costs) $\hat { \mathbf { T } } ,$ , indexed by $\left( \left| o . S \right| , o . l \right)$ for a given node o (Line 3). For each level l (Lines 4 – 8), dynamic programming [41] over $\hat { \mathbf { P } } _ { : , l }$ and $\hat { \mathbf { T } } _ { : , l }$ gives $\mathcal { R } ^ { ( l ) }$ and its estimated time $t _ { e s t } ^ { ( l ) }$ . Then the total estimated time $t _ { e s t }$ starting at level l is obtained by adding up $t _ { e s t } ^ { ( l ) }$ and $t _ { t a i l }$ , where $t _ { t a i l }$ is the sum of estimated per-level bounding cost $\tilde { \mathbf { t } } _ { i }$ (detailed in Appendix F.2) from level l + 1 to ϵ (Line 6). The best estimation keeps updating in this way until the loop terminates.

## 4 Evaluation

## 4.1 Experimental Setup

FOCUSBRANCH is implemented using PyTorch [33] and CUDA [32]. Its super branching strategy is partially adapted from the refinement strategy of prior work [41, 43]; its relaxation techniques and backward affine bounds propagation are inspired by DeepPoly [46] and CROWN [73]; and its MILP formulation comes from Tjeng et al. [50], using Gurobi [13] as the solver. The experiments were conducted on a server with AMD EPYC 9454 (Zen 4) @ 2.75 GHz 48-core CPU and four NVIDIA H100 SXM (80 GB memory) GPUs.

We consider three standard image classification datasets with 10 classes each: MNIST [22], which contains 28×28 grayscale handwritten digits; Fashion-MNIST [66], a drop-in MNIST replacement with 28×28 grayscale clothing images; and CIFAR-10 [21], which consists of 32×32 RGB natural images across objects and animals and is generally more challenging due to higher visual variability.

We evaluate FOCUSBRANCH on 13 networks, where 8 are adopted from COVERD [43]: one fully connected MNIST network (bfc1), and seven convolutional neural networks (CNNs) (bcv1–bcv7), consisting of four MNIST networks (bcv1–bcv4), two Fashion-MNIST networks (bcv6 and bcv7), and one CIFAR-10 network (bcv5, originally from ERAN [47]). The remaining five are fully connected MNIST networks (afc1–afc5) adversarially trained under the $\ell _ { 0 }$ setting following the auto\_LiRPA certified defense configuration [55]. The network mlp\_3layer (afc4) is obtained by running the official auto\_LiRPA training script [56]; the other four are derived from pretrained networks. Specifically, afc5 is adapted from bfc1, and afc1–afc3 are adapted from mnist-net\_256x2, mnist-net\_256x4, and mnist-net\_256x6, respectively, from the VNN-COMP 2020 OVAL benchmark [58].

Before verification, we first run adversarial example search on the MNIST and Fashion-MNIST test sets using σ-zero [5], which is effective at finding $\ell _ { 0 } { \cdot } \mathrm { n o r m }$ adversarial examples. This preprocessing step reduces the number of instances that can potentially be verified. For all networks except the CIFAR-10 network bcv5, we record the minimum pixel-wise $\ell _ { 0 }$ budget $\epsilon _ { \mathrm { k n o w n } }$ at which an adversarial example is known for each image. For bcv5, we do not use σ-zero because σ-zero defines the $\ell _ { 0 }$ ball over individual image entries for multi-channel inputs, whereas our verification target is pixel-wise $\ell _ { 0 }$ robustness. Instead, we use the results for adversarial example reported in prior work [42]. Since

Table 1: Main evaluation results. Neurons is the number of neurons in the network, Img is the number of evaluated images, and CC is the number of correctly classified images. Cand. denotes the number of verification candidate instances at the target budget ϵ, after excluding images for which an adversarial example within the budget is already known. R is the number of verified images, R/Cand is the verified rate, and Time reports the mean verification time (in seconds) over each tool’s verified instances. Refer to Figures 4, 7 and 8 for sample-wise comparison.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Network</td><td rowspan="2">Neurons</td><td rowspan="2">Img</td><td rowspan="2">CC €</td><td rowspan="2"></td><td rowspan="2">Cand.</td><td colspan="3">FOCUSBRANCH</td><td colspan="3">CoVERD+top-t-GP</td></tr><tr><td>R↑</td><td>R/Cand.↑</td><td>Time (s)↓</td><td>R↑</td><td>R/Cand.↑</td><td>Time (s).↓</td></tr><tr><td rowspan="9">MNIST</td><td>afc1</td><td>512</td><td>100</td><td>100</td><td>3 4 5 6</td><td>90 72 57 46 94</td><td>90 70 52 29 94</td><td>100.0% 97.2% 91.2% 63.0% 100.0%</td><td>0.3 10.1 17.8 53.7 0.3</td><td>90 67 30 13 94</td><td>100.0% 93.1% 52.6% 28.3% 100.0%</td><td>26.6 165.7 203.3 257.1 37.2</td></tr><tr><td>afc2</td><td>1024</td><td>100</td><td>100 5</td><td></td><td>88 79 68</td><td>87 98.9% 70 88.6% 38 55.9%</td><td></td><td>6.1 48.0 90.9</td><td>80 90.9% 33 41.8% 15 22.1%</td><td></td><td>165.0 126.9 235.8</td></tr><tr><td>afc3</td><td>1536</td><td>100</td><td>100</td><td></td><td>92 84 74 66</td><td>92 84 67 42</td><td>100.0% 100.0% 90.5% 63.6%</td><td>0.5 12.9 34.9 193.1</td><td>92 76 39 15</td><td>90.5% 52.7% 22.7%</td><td>57.3 191.9 308.5 253.7</td></tr><tr><td>afc4</td><td>384</td><td>100</td><td>99</td><td></td><td>89 85 73 61</td><td>89 85 70 49</td><td>100.0% 100.0% 95.9% 80.3%</td><td>&lt;0.1 0.5 5.4 53.5</td><td>89 78 37 19</td><td>91.8% 50.7% 31.1%</td><td>9.5 109.2 90.5 33.1</td></tr><tr><td>afc5</td><td>1000</td><td>100</td><td>100</td><td></td><td>89 80 70 60</td><td>89 80 61 29 91</td><td>100.0% 100.0% 87.1% 48.3% 100.0%</td><td>0.9 30.3 188.4 374.8 4.7</td><td>89 73 31 6 91</td><td>100.0% 91.2% 44.3% 10.0% 100.0%</td><td>109.1 321.3 638.7 741.2 232.2</td></tr><tr><td>bfc1</td><td>1000</td><td>100</td><td>99</td><td>4 5 6</td><td>91 82 68 59 87</td><td>81 47 13 87</td><td>98.8% 69.1% 22.0% 100.0%</td><td>82.3 192.4 401.1 7.5</td><td>67 9 2 87</td><td>81.7% 13.2% 3.4% 100.0%</td><td>554.5 661.0 642.5 60.0</td></tr><tr><td>bcv1</td><td>3604</td><td>100</td><td>98</td><td>5 6</td><td>81 73 64 84</td><td>69 49 84</td><td>94.5% 76.6% 100.0%</td><td>133.8 359.1 8.7</td><td>52 14 84</td><td>71.2% 21.9% 100.0%</td><td>143.5 514.0 644.1 67.3</td></tr><tr><td></td><td></td><td></td><td></td><td>5 6</td><td>67 62 90</td><td>67 55 90 85</td><td>100.0% 88.7% 100.0% 100.0%</td><td>49.4 131.0 12.6 25.2</td><td>64 36 90 84</td><td>95.5% 58.1% 100.0% 98.8%</td><td>269.1 488.8 108.0 238.2</td></tr><tr><td>bcv3</td><td>4504</td><td>100</td><td>99</td><td>5 6</td><td>76 68</td><td>72 41 8</td><td>94.7% 60.3% 88.9%</td><td>185.3 401.9 793.3</td><td>46 9 7</td><td>60.5% 13.2% 77.8%</td><td>676.6 1070.2 1090.5</td></tr><tr><td>bcv6</td><td>3604</td><td></td><td></td><td>3 4</td><td>66 58 49</td><td>66 58 34</td><td>100.0% 100.0% 69.4%</td><td>10.0 57.8 169.3</td><td>66 57 28</td><td>100.0% 98.3% 57.1%</td><td>61.2 197.0 621.1</td></tr><tr><td rowspan="2"></td><td>bcv7</td><td>4504</td><td>100</td><td>86</td><td>3 4</td><td>68 60 54</td><td>68 54 32</td><td>100.0% 90.0% 59.3%</td><td>16.6 134.7 304.8</td><td>68 46 6</td><td>100.0% 76.7% 11.1%</td><td>105.6 474.5 846.4</td></tr><tr><td>bcv5</td><td>4852</td><td>10</td><td>9</td><td>6 3</td><td>47 5</td><td>4 8.5% 4 80.0%</td><td></td><td>602.3 302.0</td><td>0 0.0% 3 60.0%</td><td></td><td>一 一</td></tr></table>

we prioritize verification rather than adversarial example discovery, we skip verification for an image whenever $\epsilon _ { \mathrm { t a r g e t } } \geq \epsilon _ { \mathrm { k n o w n } }$ , where $\boldsymbol { \epsilon } _ { \mathrm { t a r g e t } }$ is the target perturbation budget. Appendix H.2 shows qualitative examples of verified and adversarial images.

## 4.2 Main Results

We evaluate all networks at $\epsilon \in \{ 3 , 4 , 5 , 6 \}$ except bcv4 (by far the largest) and bcv5 (CIFAR-10, where 3 pixels span 9 input features), because their bounds are loose enough that verification relies heavily on MILP even at ϵ = 3. So, as in prior works [41, 42, 43], we evaluate them only at $\epsilon = 3$ (see Appendix H.1). All tools use a timeout of 1800 s (3600 s for bcv4), shorter than the 5 h per instance used by [43]. For bcv5, our machine cannot run COVERD+top-t-GP with 50 MILP instances, so we obtained from their results table [42] and manually timed out those verified but above 1800s. The mean time is not calculated since we run on different machines.

The results are summarized in Table 1. Overall, FOCUSBRANCH verifies 2799 out of 3189 candidate instances, achieving an aggregate verification rate of 87.8%. The method is particularly effective for moderate perturbation budgets: at ϵ = 3 and ϵ = 4, FOCUSBRANCH verifies 99.8% and 98.8% of the candidate instances, respectively. As expected, the verification problem becomes increasingly challenging as ϵ grows. Nevertheless, FOCUSBRANCH still verifies 86.6% of the instances at $\epsilon = 5$ and verifies a non-trivial fraction of instances at $\epsilon = 6$

![](images/6de78cff853c36d2b102fab260f5c88a8e3def209a7410875d734eccb1dbc98c.jpg)

![](images/9a48cf655929bdd4442ad412ce09807d256086bc389eedba221a46982e97e80e.jpg)

![](images/cd5f68b1e500ce202a6ceab1596d7be35fb209adf4cd5fc9501207b95799d37c.jpg)  
Figure 4: Comparison between FOCUSBRANCH, COVERD and COVERD+top-t-GP at $\epsilon = 5$

![](images/b018b5f18a7664e17c9b022c4e7a39ddf3e7bbd227f766cf2655d9f2c8685865.jpg)  
Figure 5: Ablation study on afc5 at $\epsilon = 5 .$ . “w/o policy $1 ^ { \circ }$ disables super-node branching by always starting from $\Theta _ { 0 } ,$ , while $\mathrm { { } ^ { 6 6 } w / o }$ policy $2 ^ { \circ }$ disables concrete-node branching by always starting from $\Theta _ { \epsilon }$

## 4.3 Comparison with Baseline Tools

Next, we compare FOCUSBRANCH with state-of-the-art $\ell _ { 0 }$ robustness verifiers. Since COVERD [43] outperforms CALZONE [41], we only compare FOCUSBRANCH with COVERD and a more recent improvement COVERD+top-t-GP [42]. The latter can be viewed as a special case of FOCUSBRANCH that always starts from the initial node Θ<sub>ϵ</sub> at the largest level ϵ, whereas FOCUSBRANCH selects the starting level per instance from node-sampling statistics. Note that FOCUSBRANCH enables batching, an additional axis of parallelism, while COVERD+top-t-GP cannot. Appendix H.7 further studies the impact of batching on all the networks.

We evaluate all methods on the candidate instances at $\epsilon = 5 ,$ excluding bcv4 and bcv5, since they are only evaluated at $\epsilon = 3$ in the main results. As shown in Figure 4, FOCUSBRANCH significantly outperforms both baselines. Across all instances, FOCUSBRANCH verifies 641 out of 740 instances, compared with 100 by COVERD and 375 by COVERD+top-t-GP. For fully connected networks, FOCUSBRANCH verifies 367 instances, while COVERD verifies only one and COVERD+top-t-GP verifies 179. On CNNs, FOCUSBRANCH also achieves the best result, verifying 274 instances compared with 99 by COVERD and 196 by COVERD+top-t-GP. As Table 2 shows, the planner selects a starting level less than ϵ more often on fully connected networks, so they benefit from both branching policies. On CNNs, the sparse affine representation of convolutional operator (Appendix H.1) prevents this, but CNNs still benefit from batching.

The comparison with MILP-based verification is deferred to Appendix H.4.

## 4.4 Ablation Study

To evaluate the contribution of super-node and concrete-node branching policies, we conduct an ablation study on afc5 at $\epsilon = 5 .$ , where the selected starting levels are relatively balanced between $l _ { s } ~ < ~ \epsilon$ and $l _ { s } ~ = ~ \epsilon$ (Table 2). Specifically, we consider two variants for the study. First, w/o policy 1 disables super-node branching by always starting verification from the initial node $\Theta _ { 0 }$ and skipping the planning. Since $\Theta _ { 0 }$ is a concrete node and concrete branching only generates concrete children (Lemma E.9), this variant performs only concrete branching throughout verification. Second, w/o policy 2 disables concrete-node branching by always starting from $\bar { \Theta } _ { \epsilon }$ . This variant performs only super-node branching until nodes become concrete; since such concrete nodes are already at level ϵ, they are unbranchable and cannot be further decomposed.

As shown in Figure 5, the full FOCUSBRANCH verifies the most instances, solving 61 out of 70 instances compared with 49 for w/o policy 1 and 56 for w/o policy 2. The gap shows that neither branching policy alone is sufficient: concrete-node branching is effective for refining difficult concrete nodes, while super-node branching compactly covers many perturbation cases. By adaptively selecting the starting level and branching strategy, FOCUSBRANCH combines both mechanisms and achieves the best overall verification performance.

## 5 Related Work

Neural network verification [17, 19, 27] provides rigorous certificates for deep neural networks under some specifications, enabling their reliable deployment in safety-critical scenarios [15, 18]. Verification methods can verify a wide range of input specifications, such as robustness against locally $\ell _ { p }$ bounded perturbations [46, 59, 67], semantic distortions [1, 26, 46], data poisoning [25, 61], and label flipping [36]. Typical verification methods use off-the-shelf solvers [50, 52], linear relaxations [10, 62, 67, 73], dual variable optimization [63], branch and bound [18, 59], Lipschitzconstrained layers [14, 53, 69], and randomized smoothing [6, 38, 70].

Among $\ell _ { p }$ norms, the $\ell _ { 0 }$ norm as perturbation bound type is particularly challenging for local robustness verification due to non-convexity and non-differentiability. Existing verification methods tend to modify the inference protocol through systematic or randomized ablation [23, 24], which provides strong certificates but induces significant inference overhead. On the other hand, direct verification of standard neural networks is harder: bound propagation with closed-form $\ell _ { 0 }$ concretization [67] is sound but, on its own, too loose for larger budgets. With α-CROWN [68] enabled, auto\_LiRPA [67] tightens these bounds by optimizing the slopes of the ReLU lower relaxations with gradient ascent, but it still bounds the entire $\ell _ { 0 } .$ -ball as a single problem, without branching. In contrast, CALZONE [41], COVERD [43], and COVERD+top-t-GP [42] run branch-and-bound over covering designs of perturbable pixel sets (see Section 2). For a related $\ell _ { 0 } { \mathrm { - s t r u c t u r e d } }$ specification, Bunel et al. [2] branch by support extension to verify neural compressed-sensing decoders, close in spirit to our concrete branching (Appendix $\mathrm { A } )$ . Our FOCUSBRANCH is a more general branchand-bound-based verification method for ℓ<sub>0</sub>-bounded perturbations that integrates two branching strategies, with node-constrained closed-form concretization, and a strategy planner that chooses a starting level per instance. Since FOCUSBRANCH keeps its relaxations fixed, relaxation optimization is orthogonal to our node abstraction, which only changes the concretization step, and could in principle tighten the bound of every node at the cost of per-node optimization.

More detailed discussion of related work is deferred to Appendix $\mathrm { A } .$

## 6 Conclusion

We introduced FOCUSBRANCH, a combinatorial branch-and-bound framework for verifying local robustness of neural networks against $\ell _ { 0 }$ perturbations. The key insight is a node abstraction that unifies upper-shadow (concrete) and covering-design (super) branching, making the $\ell _ { 0 } .$ -ball decomposable across levels. On a wide range of benchmarks, FOCUSBRANCH verifies substantially more instances than state-of-the-art verifiers, showing that exploiting the combinatorial structure of sparse perturbations is an effective direction for $\ell _ { 0 }$ verification.

## Acknowledgments and Disclosure of Funding

We would like to thank anonymous reviewers for their feedback. This research was conducted with the support of Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grants, Digital Research Alliance of Canada, and Simon Fraser University (SFU) Startup Grants.

We declare no competing interests.

## References

[1] Mislav Balunovic, Maximilian Baader, Gagandeep Singh, Timon Gehr, and Martin Vechev. Certifying geometric robustness of neural networks. Advances in Neural Information Processing Systems, 32, 2019.

[2] Rudy Bunel, Krishnamurthy Dvijotham, M Pawan Kumar, Alessandro De Palma, and Robert Stanforth. Verified neural compressed sensing. arXiv preprint arXiv:2405.04260, 2024.

[3] Yizheng Chen, Shiqi Wang, Yue Qin, Xiaojing Liao, Suman Jana, and David Wagner. Learning security classifiers with verified global robustness properties. In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security, pages 477–494, 2021.

[4] Ping-yeh Chiang, Renkun Ni, Ahmed Abdelkader, Chen Zhu, Christoph Studer, and Tom Goldstein. Certified defenses for adversarial patches. In International Conference on Learning Representations, 2020.

[5] Antonio Emanuele Cinà, Francesco Villani, Maura Pintor, Lea Schönherr, Battista Biggio, and Marcello Pelillo. σ-zero: Gradient-based optimization of ℓ<sub>0</sub>-norm adversarial examples. In International Conference on Learning Representations, 2025.

[6] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In International Conference on Machine Learning, pages 1310–1320. PMLR, 2019.

[7] David R. Cox. The regression analysis of binary sequences. Journal ofthe Royal Statistical Society: Series B (Methodological), 20(2):215–242, 1958.

[8] Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classification. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1625–1634, 2018.

[9] Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, and Yingyan Lin. Patch-Fool: Are vision transformers always robust against adversarial perturbations? In International Conference on Learning Representations, 2022.

[10] Timon Gehr, Matthew Mirman, Dana Drachsler-Cohen, Petar Tsankov, Swarat Chaudhuri, and Martin Vechev. AI2: Safety and robustness certification of neural networks with abstract interpretation. In 2018 IEEE symposium on security and privacy (SP), pages 3–18. IEEE, 2018.

[11] Daniel M. Gordon. The La Jolla Combinatorics Repository. https://ljcr.dmgordon.org/, 2023. Accessed: 2026-05-06.

[12] Daniel M. Gordon, Greg Kuperberg, and Oren Patashnik. New constructions for covering designs. Journal ofCombinatorial Designs, 3(4):269–284, 1995.

[13] Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2025. https://www. gurobi.com/documentation/.

[14] Yujia Huang, Huan Zhang, Yuanyuan Shi, J Zico Kolter, and Anima Anandkumar. Training certifiably robust neural networks with efficient local Lipschitz bounds. Advances in Neural Information Processing Systems, 34:22745–22757, 2021.

[15] Zijian Huang, Wenda Chu, Linyi Li, Chejian Xu, and Bo Li. COMMIT: Certifying robustness of multi-sensor fusion systems against semantic attacks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 17528–17536, 2025.

[16] Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, and Xingxing Wei. DIFFender: Diffusion-based adversarial defense against patch attacks. In European Conference on Computer Vision, pages 130–147. Springer, 2024.

[17] Guy Katz, Clark Barrett, David L. Dill, Kyle Julian, and Mykel J. Kochenderfer. Reluplex: An efficient SMT solver for verifying deep neural networks. In Computer Aided Verification, pages 97–117. Springer, 2017.

[18] Guy Katz, Derek A Huang, Duligur Ibeling, Kyle Julian, Christopher Lazarus, Rachel Lim, Parth Shah, Shantanu Thakoor, Haoze Wu, Aleksandar Zeljic, et al. The Marabou Framework´ for Verification and Analysis of Deep Neural Networks. In Computer Aided Verification, pages 443–452. Springer, 2019.

[19] Konstantin Kaulen, Tobias Ladner, Stanley Bak, Christopher Brix, Hai Duong, Thomas Flinkow, Taylor T Johnson, Lukas Koller, Edoardo Manino, ThanhVu H Nguyen, et al. The 6th International Verification of Neural Networks Competition (VNN-COMP 2025): Summary and Results. arXiv preprint arXiv:2512.19007, 2025.

[20] Donald E. Knuth. The Art of Computer Programming, Volume 4, Fascicle 3: Generating All Combinations and Partitions. Addison-Wesley, 2005.

[21] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[22] Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. The MNIST Database of Handwritten Digits. http://yann.lecun.com/exdb/mnist/, 1998. Accessed: 2026-05-06.

[23] Alexander Levine and Soheil Feizi. (de)randomized smoothing for certifiable defense against patch attacks. Advances in Neural Information Processing Systems, 33:6465–6475, 2020.

[24] Alexander Levine and Soheil Feizi. Robustness certificates for sparse adversarial attacks by randomized ablation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 4585–4593, 2020.

[25] Alexander Levine and Soheil Feizi. Deep partition aggregation: Provable defenses against general poisoning attacks. In International Conference on Learning Representations, 2021.

[26] Linyi Li, Maurice Weber, Xiaojun Xu, Luka Rimanic, Bhavya Kailkhura, Tao Xie, Ce Zhang, and Bo Li. TSS: Transformation-specific smoothing for robustness certification. In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security, pages 535–557, 2021.

[27] Linyi Li, Tao Xie, and Bo Li. SoK: Certified robustness for deep neural networks. In 2023 IEEE symposium on security and privacy (SP), pages 1289–1310. IEEE, 2023.

[28] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.

[29] Peter McCullagh and John A. Nelder. Generalized Linear Models, volume 37 of Monograph on Statistics and Applied Probability. Chapman and Hall, London, 2 edition, 1989.

[30] Apostolos Modas, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. SparseFool: A few pixels make a big difference. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[31] Christoph Müller, François Serre, Gagandeep Singh, Markus Püschel, and Martin Vechev. Scaling polyhedral neural network verification on gpus. In A. Smola, A. Dimakis, and I. Stoica, editors, Proceedings of Machine Learning and Systems, volume 3, pages 733–746, 2021.

[32] NVIDIA Corporation. NVIDIA CUDA Toolkit Documentation, Version 12.2. https://docs. nvidia.com/cuda/archive/12.2.2/, 2023. Accessed: 2026-05-06.

[33] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, 2019.

[34] Jary Pomponi, Simone Scardapane, and Aurelio Uncini. Pixle: a fast and effective black-box attack based on rearranging pixels. In 2022 International Joint Conference on Neural Networks (IJCNN), pages 1–7. IEEE, 2022.

[35] Alexander Robey, Eric Wong, Hamed Hassani, and George J Pappas. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks. arXiv preprint arXiv:2310.03684, 2023.

[36] Elan Rosenfeld, Ezra Winston, Pradeep Ravikumar, and Zico Kolter. Certified robustness to label-flipping attacks via randomized smoothing. In International Conference on Machine Learning, pages 8230–8241. PMLR, 2020.

[37] Frank Ruskey. Combinatorial generation. Working Version, 2003.

[38] Hadi Salman, Jerry Li, Ilya Razenshteyn, Pengchuan Zhang, Huan Zhang, Sebastien Bubeck, and Greg Yang. Provably robust deep learning via adversarially trained smoothed classifiers. Advances in Neural Information Processing Systems, 32, 2019.

[39] J. Schönheim. On coverings. Pacific Journal ofMathematics, 14(4):1405–1411, 1964.

[40] Pranab Kumar Sen. Estimates of the regression coefficient based on Kendall’s tau. Journal of the American Statistical Association, 63(324):1379–1389, 1968.

[41] Yuval Shapira, Eran Avneri, and Dana Drachsler-Cohen. Deep learning robustness verification for few-pixel attacks. Proceedings of the ACM on Programming Languages, 7(OOPSLA1):90, 2023.

[42] Yuval Shapira and Dana Drachsler-Cohen. Tight robustness certification through the convex hull of $\ell _ { 0 }$ attacks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 37913–37922, 2026.

[43] Yuval Shapira, Naor Wiesel, Shahar Shabelman, and Dana Drachsler-Cohen. Boosting fewpixel robustness verification via covering verification designs. In Computer Aided Verification, volume 14682 of Lecture Notes in Computer Science, pages 377–400. Springer, 2024.

[44] Abhijith Sharma, Yijun Bian, Phil Munz, and Apurva Narayan. Adversarial patch attacks and defences in vision-based tasks: A survey. arXiv preprint arXiv:2206.08304, 2022.

[45] Zhouxing Shi, Huan Zhang, Kai-Wei Chang, Minlie Huang, and Cho-Jui Hsieh. Robustness verification for transformers. In International Conference on Learning Representations, 2020.

[46] Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. An abstract domain for certifying neural networks. In Proceedings ofthe ACM on Programming Languages, volume 3, pages 1–30, 2019.

[47] SRI Lab, ETH Zurich. ERAN: ETH robustness analyzer for neural networks. https:// github.com/eth-sri/eran, 2018. Accessed: 2026-11-08.

[48] Guanhong Tao, Shengwei An, Siyuan Cheng, Guangyu Shen, and Xiangyu Zhang. Hard-label black-box universal adversarial patch attack. In 32nd USENIX Security Symposium (USENIX Security 23), pages 697–714, 2023.

[49] Henri Theil. A rank-invariant method of linear and polynomial regression analysis. I, II, III. Proceedings ofthe Koninklijke Nederlandse Akademie van Wetenschappen, 53:386–392, 521–525, 1397–1412, 1950.

[50] Vincent Tjeng, Kai Y. Xiao, and Russ Tedrake. Evaluating robustness of neural networks with mixed integer programming. In International Conference on Learning Representations, 2019.

[51] D. T. Todorov. Combinatorial coverings. PhD thesis, University of Sofia, 1985.

[52] Calvin Tsay, Jan Kronqvist, Alexander Thebelt, and Ruth Misener. Partition-based formulations for mixed-integer optimization of trained ReLU neural networks. Advances in Neural Information Processing Systems, 34:3068–3080, 2021.

[53] Yusuke Tsuzuku, Issei Sato, and Masashi Sugiyama. Lipschitz-margin training: Scalable certifi cation of perturbation invariance for deep neural networks. Advances in Neural Information Processing Systems, 31, 2018.

[54] Danilo Vasconcellos Vargas and Jiawei Su. Understanding the one-pixel attack: Propagation maps and locality analysis. arXiv preprint arXiv:1902.02947, 2019.

[55] Verified-Intelligence. auto\_LiRPA: Examples and certified adversarial defense training. https://github.com/Verified-Intelligence/auto\_LiRPA/blob/master/doc/ src/examples.md#basic-certified-adversarial-defense-training, 2026. Accessed: 2026-05-06.

[56] Verified-Intelligence. auto\_LiRPA: Simple vision training script. https://github.com/ Verified-Intelligence/auto\_LiRPA/blob/master/examples/vision/simple\_ training.py, 2026. Accessed: 2026-05-06.

[57] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, et al. SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17(3):261–272, 2020.

[58] VNN-COMP. VNN-COMP 2020: MNIST OVAL Benchmark Networks. https://github. com/VNN-COMP/vnncomp2020/tree/master/2020/PWL/benchmark/mnist/oval, 2020. Accessed: 2026-05-06.

[59] Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana, Cho-Jui Hsieh, and J Zico Kolter. Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification. Advances in Neural Information Processing Systems, 34:29909– 29921, 2021.

[60] Wenjie Wang, Pengfei Tang, Jian Lou, and Li Xiong. Certified robustness to word substitution attack with differential privacy. In Proceedings ofthe 2021 conference ofthe North American chapter of the association for computational linguistics: human language technologies, pages 1102–1112, 2021.

[61] Maurice Weber, Xiaojun Xu, Bojan Karlaš, Ce Zhang, and Bo Li. RAB: Provable robustness against backdoor attacks. In 2023 IEEE Symposium on Security and Privacy (SP), pages 1311–1328. IEEE, 2023.

[62] Lily Weng, Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Luca Daniel, Duane Boning, and Inderjit Dhillon. Towards fast computation of certified robustness for ReLU networks. In International Conference on Machine Learning, pages 5276–5285. PMLR, 2018.

[63] Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In International Conference on Machine Learning, pages 5286–5295. PMLR, 2018.

[64] Tong Wu, Liang Tong, and Yevgeniy Vorobeychik. Defending against physically realizable attacks on image classification. In International Conference on Learning Representations, 2020.

[65] Chong Xiang, Saeed Mahloujifar, and Prateek Mittal. PatchCleanser: Certifiably robust defense against adversarial patches for any image classifier. In 31st USENIX security symposium (USENIX Security 22), pages 2065–2082, 2022.

[66] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

[67] Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, and Cho-Jui Hsieh. Automatic perturbation analysis for scalable certified robustness and beyond. In Advances in Neural Information Processing Systems, 2020.

[68] Kaidi Xu, Huan Zhang, Shiqi Wang, Yihan Wang, Suman Jana, Xue Lin, and Cho-Jui Hsieh. Fast and complete: Enabling complete neural network verification with rapid and massively parallel incomplete verifiers. In International Conference on Learning Representations, 2021.

[69] Xiaojun Xu, Linyi Li, and Bo Li. LOT: Layer-Wise Orthogonal Training on Improving L2 Certified Robustness. Advances in Neural Information Processing Systems, 35:18904–18915, 2022.

[70] Greg Yang, Tony Duan, J Edward Hu, Hadi Salman, Ilya Razenshteyn, and Jerry Li. Randomized smoothing of all shapes and sizes. In International Conference on Machine Learning, pages 10693–10705. PMLR, 2020.

[71] Mao Ye, Chengyue Gong, and Qiang Liu. SAFER: A structure-free approach for certified robustness to adversarial word substitutions. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3465–3475, 2020.

[72] Cheng Yu, Jiansheng Chen, Youze Xue, Yuyang Liu, Weitao Wan, Jiayu Bao, and Huimin Ma. Defending against universal adversarial patches by clipping feature norms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16434–16442, 2021.

[73] Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient neural network robustness certification with general activation functions. In Advances in Neural Information Processing Systems, volume 31, 2018.

[74] Zhanyuan Zhang, Benson Yuan, Michael McCoyd, and David Wagner. Clipped BagNet: Defending against sticker attacks with clipped bag-of-features. In 2020 IEEE Security and Privacy Workshops (SPW), pages 55–61. IEEE, 2020.

## A Extended Discussion on Related Work

This appendix extends our discussion of related work in Section $5 .$

$\ell _ { 0 }$ -bounded adversary and sparse attacks. In the $\ell _ { 0 }$ -bounded adversarial model, the attacker imposes perturbations to a fixed and small amount of input coordinates (i.e., pixels in the image domain) to manipulate the model predictions while remaining stealthy. The $\ell _ { 0 }$ norm, as a “pseudonorm”, measures the number of changed coordinates rather than the magnitude of change. Hence, in the literature, adversary under $\ell _ { 0 } \cdot$ -bounded constraint is also known as few-pixel or sparse attacks [5, 34, 54]. These attacks usually deploy an effective local search algorithm or continuous approximations combined with gradient descent to find adversarial examples under $\ell _ { 0 } .$ -norm constraints. Their effectiveness demonstrates the internal fragility of neural network robustness.

A closely related type of adversary is the patch attack, where the perturbable coordinates not only have a small $\ell _ { 0 }$ but also need to be spatially concentrated. While imposing more constraints, patch attacks are still powerful while stealthy [9, 44, 48]. Further, the effectiveness of patch attacks in the real physical world, $\mathrm { e . g . }$ ., against autonomous driving systems [8], has been revealed, which necessitates effective defenses against $\ell _ { 0 }$ -bounded attacks and patch attacks.

Comparison with other $\ell _ { 0 }$ -bounded defenses and verification methods. To counteract the $\ell _ { 0 } \cdot$ bounded adversary, empirical defenses are proposed, which strengthen the model robustness against existing attacks [16, 64, 72]. However, empirical defenses could not provide a rigorous guarantee in the presence of future stronger attacks. In contrast, verification methods establish a rigorous guarantee to enable its deployment in safety-critical scenarios.

Among verification methods, one line of research is to alter the inference protocol. Intuitively, when the prediction is derived from majority voting over model predictions on randomly ablated inputs or features, the prediction would be robust against small input perturbations. Using this principle, randomized ablation/smoothing then aggregation [23, 25, 65, 74] enables certified robustness against $\ell _ { 0 }$ adversary and patch adversary. However, they alter the inference procedure and incur a large inference-time overhead.

It is indeed challenging to provide robust certificates for ordinary neural networks, which is the focus of our paper. The challenge primarily lies in the non-convexity and non-differentiability of the $\ell _ { 0 }$ norm. Besides relying on MILP (mixed-integer programming) solver, recent tools on $\ell _ { 0 }$ verification are CALZONE [41] and COVERD [43] and COVERD+top-t-GP [42]. Section 2 describes their covering-design-based branch-and-bound precisely. We view this framework “branch-and-bound” on the covering set. Our tool FOCUSBRANCH is built on this design and is also a branch-andbound based framework. However, our FOCUSBRANCH branches in a new domain, where each node o consists of $\mathbf { a }$ “focus set” $_ { o . S }$ and the soft perturbation budget for the focus set $o . l .$ As a result, our FOCUSBRANCH integrates both covering-design-based branching (super branching, which decomposes the focus set) and upper-shadow-based branching (concrete branching, which extend the focus set). The extended branch-and-bound enables a new dimension of branching design choices and thus significantly improves the verification efficiency while remaining tight, as our analysis and experiments have shown.

$\ell _ { 0 }$ -bounded in broader domains. FOCUSBRANCH and baseline approaches are primarily evaluated in the image domain following the literature [41, 43, 67, 73]. The $\ell _ { 0 }$ adversary is also studied in other modalities such as tabular data [3] and language data [60]. In particular, in the language domain, the $\ell _ { 0 }$ adversary is known as word substitution attacks or suffix jailbreak attacks, where corresponding certified methods are primarily based on majority voting, randomized smoothing, and ablation [35, 60, 71]. Though these methods incur large inference overhead, the other way, direct verification, is still challenging due to the non-linearity and large scales of language models [45]. We leave as future work to further scale the framework to support verification against $\ell _ { 0 } \cdot$ -bounded adversary for language models.

Relation to support-extension branching. Bunel et al. [2] verify that a neural compressed-sensing decoder recovers the support of every signal with a given number of non-zero entries. It branches by extending a fixed support one coordinate at a time in canonical index order, with every skipped coordinate fixed to zero (the “off”-set). Once a support is fully determined, Bunel et al.[2] continue by splitting the input box instead of calling MILP. Their off-set is technically compatible with our node abstraction by excluding the pixels in off-set in the concretization (Eq. (12) and Eq. (10)) steps, which can slightly tighten bounds. However, it can change the branching domain in two ways. First, every node less than level ϵ would carry its own partition of the pixels, which makes nodes more expensive to represent and bound. Second, off-sets make children disjoint and thus rule out cross-parent pruning (Appendix F.3). Specifically, without off-sets, a level-(l + 1) concrete node is dropped as soon as any of its $l + 1$ parents is proven safe, whereas with off-sets only its canonical parent’s safety implies its own.

## B Limitations

While FOCUSBRANCH achieves state-of-the-art efficiency on the evaluated benchmarks, exact $\ell _ { 0 }$ verification remains inherently combinatorial and computationally hard. Therefore, within practical time and memory budgets, FOCUSBRANCH is evaluated on networks with up to thousands of neurons rather than the largest neural networks. This limitation does not affect the soundness of the verification results. To further improve the scalability, a promising path forward is to combine the branch-andbound framework with stronger neural relaxations or tighter bound propagation techniques. We leave the extension to other domains such as language models as future work.

Another limitation is that FOCUSBRANCH is designed primarily as a verifier, so it is not optimized for quickly finding adversarial examples. In particular, FOCUSBRANCH manages nodes in a breadth-first manner to enable batched bounding (Appendix F.3). This design choice substantially improves verification throughput, but it can delay the exploration of deep unbranchable nodes where MILP may quickly expose an adversarial example. In practice, this limitation can be mitigated by running complementary tools to find adversarial examples before verification, as done in our evaluation.

In addition, covering-design-based super branching may introduce duplicated bounding effort, because different child super nodes can cover overlapping perturbation cases. Reducing such redundancy while preserving the efficacy of super-node branching is an important direction for future work.

Finally, the planner used in our experiments is sensitive to timing noise, so repeated runs may verify slightly different sets of borderline instances (Appendices H.7 and H.10).

## C Societal Impacts

As positive societal impacts, local robustness verification of image classifiers can help people better understand and improve the reliability and stability of neural networks. If a network is verified to be robust for a given image under a perturbation budget ϵ, then the classifier’s prediction is guaranteed to remain unchanged for any perturbation involving up to ϵ pixels. Thus, robustness verification has great potential to help protect against adversarial attacks in critical settings such as medical imaging, autonomous driving, and security screening. As for negative societal impacts, we are not aware of any direct harms arising specifically from local robustness verification.

## D Formulas

Here, we derive a closed-form solution to compute the minimum of any affine form $g ( \mathbf { x } ) = \omega ^ { \mathsf { T } } \mathbf { x } + b$ For each pixel, its least possible contribution can be derived as

$$
\Delta _ { v } = \operatorname* { m i n } _ { \substack { \mathbf { x } _ { v } \in [ \mathbf { \bar { x } } _ { v } , \mathbf { \bar { x } } _ { v } ] } } \Delta _ { v } = \sum _ { c = 1 } ^ { C } \omega _ { v , c } ( \mathbf { x } _ { v , c } ^ { \prime } - \mathbf { x } _ { v , c } ^ { \star } ) , \quad \mathrm { w h e r e ~ } \mathbf { x } _ { v , c } ^ { \prime } = \left\{ \begin{array} { l l } { \mathbf { \bar { x } } _ { v , c } , } & { \omega _ { v , c } \leq 0 , } \\ { \mathbf { \bar { x } } _ { v , c } , } & { \omega _ { v , c } > 0 , } \end{array} \right.\tag{11}
$$

and the minimum of $g ( \mathbf { x } )$ over a node-constrained space $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ can be derived as follows:

$$
\operatorname* { m i n } _ { \substack { \mathbf { x } \in B _ { 0 } ^ { \star } ( \mathbf { x } ^ { \star } , o ) } } g ( \mathbf { x } ) = g ( \mathbf { x } ^ { \star } ) + \operatorname* { m i n } _ { \substack { \mathbf { x } \in B _ { 0 } ^ { \star } ( \mathbf { x } ^ { \star } , o ) } } \sum _ { v = 1 } ^ { V } \Delta _ { v } = g ( \mathbf { x } ^ { \star } ) + \sum _ { \substack { v \in A ( o , l , o , S ) \cup A ( \epsilon - o , l , v ) \cup A ( o , l , o , S ) ) } } \Delta _ { v } ,\tag{12}
$$

where $\underline { { \mathcal { A } } } ( a , A ) : = \arg \tan \tt { - } a _ { v \in \cal { A } } - \underline { { \Delta } } _ { v }$ is a set of a (where $a \in \{ 0 , \ldots , \epsilon \} )$ most negatively contributed pixels from set $A \subseteq \nu$

In practice, the evaluation of Eq. (10) and Eq. (12) is CUDA-accelerated.

## E Proofs

Lemma E.1 (Node-constrained concretization). Let $\mathbf { x } ^ { \star } \in \mathcal { D }$ and $\epsilon < V$ . For any affine form $g ( \mathbf { x } ) = \omega ^ { \top } \mathbf { x } + b$ and any node o, the RHS ofEq. (10) evaluates t $\begin{array} { r } { { { \operatorname * { m a x } } _ { { \mathbf { x } } \in { \mathcal { B } } _ { 0 } ^ { \epsilon } ( { \mathbf { x } } ^ { \star } , o ) } } g ( { \mathbf { x } } ) } \end{array}$ ,for any choice ofthe top-a sets when ties occur. Similarly, the RHS ofEq. (12) evaluates to the $\begin{array} { r } { \operatorname* { m i n } _ { \mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } g ( \mathbf { x } ) } \end{array}$

Proof. Write $T ( \mathbf { x } ) : = \{ v : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \}$ and

$$
\mathcal { T } : = \{ T \subseteq \mathcal { V } : | T | \leq \epsilon , | T \backslash o . S | \leq \epsilon - o . l \} ,
$$

so that $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) = \{ \mathbf { x } \in \mathcal { D } : T ( \mathbf { x } ) \in \mathcal { T } \}$ by Eq. (1) and Eq. (4). Note that T is downward closed. Step 0 (decoupling). We claim

$$
\operatorname* { m a x } _ { \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } g ( \mathbf { x } ) \ = \ g ( \mathbf { x } ^ { \star } ) \ + \ \operatorname* { m a x } _ { T \in \mathcal { T } } \sum _ { v \in T } \overline { { \Delta } } _ { v } .\tag{13}
$$

For $\mathbf { \ddot { \Phi } } ^ { 6 6 } \leq \mathbf { \vec { \Phi } } ^ { 9 9 } \mathbf { \vec { \Phi } }$ let $\mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ . Then $\Delta _ { v } \ = \ 0$ for $v \not \in T ( \mathbf { x } )$ and $\Delta _ { v } \ \leq \ \overline { { \Delta } } _ { v }$ for $v \in T ( \mathbf { x } )$ by Eq. (9), so $\begin{array} { r } { g ( \mathbf { x } ) \leq g ( \mathbf { x } ^ { \star } ) + \sum _ { v \in T ( \mathbf { x } ) } \overline { { \Delta } } _ { v } , } \end{array}$ , and $T ( \mathbf { x } ) \in \mathcal { T }$ . For $\stackrel { \bullet \bullet } { \geq } \mathbf { \overrightarrow { : } }$ fix $A \in \mathcal T$ and define x by $\mathbf { x } _ { v } : = \mathbf { x } _ { v } ^ { \prime }$ for $v \in A$ and $\mathbf { x } _ { v } \ : = \ \mathbf { x } _ { v } ^ { \star }$ otherwise, with $\mathbf { x } _ { v } ^ { \prime }$ as in $\operatorname { E q . } \ ( 9 )$ . Then $\textbf { x } \in \ \mathcal { D }$ and ${ \dot { T } } ( \mathbf { x } ) \subseteq A ,$ , so $T ( \mathbf { x } ) \in \mathcal { T }$ by downward closure; in particular |T(x)| ≤ ϵ gives x $\in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ and $| T ( \mathbf { x } ) \setminus o . \mathcal { S } | \leq \epsilon - o . l$ gives $\mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ . Moreover $\Delta _ { v } = \overline { { \Delta } } _ { v }$ for every $v \in A$ and $\Delta _ { v } = 0$ elsewhere, so $\begin{array} { r } { g ( \mathbf { x } ) = g ( \mathbf { x } ^ { \star } ) + \sum _ { v \in A } \overline { { \Delta } } _ { v } } \end{array}$

Step 1 (an optimal set uses the full budget). Since $\tau$ is finite and nonempty, the maximum on the right of $\mathrm { E q . } \left( 1 3 \right)$ is attained; we call any $T \in \mathcal T$ attaining it a maximizer. Since $\mathbf { x } ^ { \star } \in \mathcal { D }$ , the choice $\mathbf { x } _ { v } = \mathbf { x } _ { v } ^ { \star }$ is feasible in Eq. (9), hence

$$
{ \overline { { \Delta } } } _ { v } \ \geq \ 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } v \in \mathcal { V } .\tag{14}
$$

Let $T \in \mathcal T$ with $| T | < \epsilon ;$ we exhibit u $\notin T$ with $T \cup \{ u \} \in \mathcal { T }$ . I $: o . S \ \nsubseteq T$ , pick $u \in o . S \setminus T ;$ then $( T \cup \{ u \} ) \setminus o . { \dot { S } } = T \setminus o . { \dot { S } }$ and the node constraint is unaffected. If $_ { o . S \ \subseteq \ T }$ , then $\left| T \right. \dot { \big \langle } o . { \mathcal { S } } \big | =$ $| T | - \mathbf { \bar { | } } o . S | \leq | T | - o . l < \epsilon - o . l$ using $| o . S | \geq o . l .$ , so any $u \in \mathcal { V } \setminus T$ (which exists because $| T | < \epsilon \le V )$ satisfies $| ( T \cup \{ u \} ) \backslash o . S | \leq | T \backslash o . \dot { S } | + 1 \leq \epsilon { - o . l }$ . In both cases $| T \cup \{ u \} | = | T | + 1 \leq \epsilon$ By Eq. (14) the objective does not decrease, so iterating yields a maximizer of size exactly ϵ. Any $T \subseteq { \bar { \nu } }$ with $| T | = \epsilon$ satisfies $| T \setminus o . S | = \epsilon - | T \cap o . S |$ , whence

$$
\mathrm { f o r } \mathrm { e v e r y } T \subseteq \mathcal { V } \mathrm { ~ w i t h } | T | = \epsilon : \quad | T \setminus o . S | \leq \epsilon - o . l \iff | T \cap o . S | \geq o . l .\tag{15}
$$

Step 2 (exchange into P). Put $P : = \overline { { \mathcal { A } } } ( o . l , o . S )$ , so $| P | = o . l , P \subseteq o . S$ , and $\overline { { \Delta } } _ { p } \geq \overline { { \Delta } } _ { q }$ for all $p \in P$ and $q \in o . S \setminus P .$ . This is the only property of A the argument uses, so ties may be broken arbitrarily. Let $T$ be a maximizer with $| T | = \epsilon$ and suppose $P \not \subseteq { \tilde { T } } ; \operatorname { f i x } p \in P \backslash T$ . Since $p \notin T$ and $P \subseteq o . S$ we have $| T \cap P | \leq | P | - 1 = o \dot { l } - \dot { 1 }$ , and $| T \cap o . { \bar { \cal S } } | \geq o .$ l by Eq. (15), so $\bar { | ( T \cap o . S ) \backslash P | } \geq 1$ ; fix q in this set. Then $\overline { { \Delta } } _ { p } \geq \overline { { \Delta } } _ { q } , \mathrm { s o } T ^ { \prime } : = ( T \setminus \{ q \} ) \cup \{ p \}$ satisfies $\begin{array} { r } { \sum _ { v \in T ^ { \prime } } \overline { { \Delta } } _ { v } = \sum _ { v \in T } \overline { { \Delta } } _ { v } - \overline { { \Delta } } _ { q } + \overline { { \Delta } } _ { p } \geq \sum _ { v \in T } \overline { { \Delta } } _ { v } } \end{array}$ Moreover $| T ^ { \prime } | ^ { \mathit { \prime } } = \epsilon \mathrm { ~ a n d ~ } | T ^ { \prime } \cap o . S | \ = | T \cap o . S | \ \geq \ \tilde { o . l } \ $ , so $T ^ { \prime } \in \dot { \mathcal { T } }$ by Eq. (15) and $T ^ { \prime }$ is again a maximizer. $\operatorname { A s } | T ^ { \prime } \cap P | = | T \cap P | + 1$ and $| T \cap P |$ is bounded by $| { \cal P } | = o . l$ , iterating terminates at a maximizer containing P.

Step 3 (exchange into Q). Put $Q : = \overline { { A } } ( \epsilon - o . l , \mathcal { V } \backslash P )$ , which is well defined since $\epsilon - o . l \le V - o . l =$ $| \mathcal { V } \backslash P |$ . First observe that every $T \supseteq { \dot { P } }$ with $| T | \overset { \cdot } { = } \epsilon$ lies in T: indeed $| T \cap o . S | \geq | P | = o . l$ because $\stackrel { \cdot } { P } \subseteq o . \stackrel { \cdot } { S }$ , so Eq. (15) applies. Hence $T \setminus P$ ranges over all $\left( \epsilon - o . l \right)$ -subsets of $\nu \backslash P$ , unconstrained. Let T be a maximizer with $P \subseteq T , | T | = \epsilon$ , and suppose $T \setminus P \ne Q$ . Since $Q \subseteq \mathcal { V } \setminus P$ we have $Q \backslash T = Q \backslash ( T \backslash P )$ , and as $| T \rangle | P | = | Q | = \epsilon - o . l$ it follows that $| ( T \setminus P ) \setminus Q | = | Q \setminus T | \geq 1$ . Fix $v \in ( T \setminus P ) \setminus Q$ and $w \in Q \backslash T$ ; then $\overline { { \Delta } } _ { w } \geq \overline { { \Delta } } _ { v }$ by the definition of Q, so $T ^ { \prime \prime } : = ( T \setminus \{ v \} ) \cup \{ w \}$ is again a maximizer containing $P ,$ , with $\bar { | { \cal T } ^ { \prime \prime } \cap \bar { Q } | } = | { \cal T } \cap Q | + 1$ . As $| T \cap Q |$ is bounded by $| Q | = \epsilon - o . l .$ , iterating terminates at $T \stackrel { \cdot } { = } P \cup Q$ . Since $Q \subseteq \mathcal { V } \setminus P$ , this union is disjoint and $| P \cup Q | = o . l + ( \epsilon - o . l ) = \epsilon .$ , as required. Substituting into Eq. (13) gives Eq. (10).

Minimum. Apply the above ${ \mathrm { t o } } - g .$ . Its per-pixel contribution is $- \Delta _ { v } .$ , so its best contribution is $\operatorname* { m a x } _ { \mathbf { x } _ { v } } ( - \Delta _ { v } ) = - \underline { { \Delta } } _ { v }$ with $\Delta _ { v }$ as in Eq. (11), and selecting the largest values of $- \underline { { \Delta } } _ { v }$ is exactly the rule defining $\lrcorner$ in Eq. (12). Since mi $\begin{array} { r } { { 1 } _ { \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } g ( \mathbf { x } ) = - \operatorname* { m a x } _ { \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } ( - g ) ( \mathbf { x } ) } \end{array}$ , Eq. (12) follows. □

Lemma E.2 (Validity of node-constrained bounds). Under the assumptions ofLemma E.1, let o be any node. For each $j \neq y ^ { \star }$ , let $\underline { m } _ { u ^ { \star } , i }$ be the affine lower bound ofthe margin $m _ { y ^ { \star } , j }$ computed by LiRPA-style bound propagation $I 4 \bar { 6 } , \bar { 6 } 7 ,$ , 73] in which every concretization step,for all intermediate affine layers andfor the margin itself, is performed over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ via $E q .$ . (10) and Eq. (12). (Thus $\underline { m } _ { y ^ { \star } , j }$ depends on o through the intermediate concrete bounds; as in $E q .$ . (6), we keep this dependence implicit.) Then,for every $j \neq y ^ { \star }$ , we have

$$
\forall \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) . \quad \underline { { m } } _ { y ^ { \star } , j } ( \mathbf { x } ) \leq m _ { y ^ { \star } , j } ( \mathbf { x } ) ,\tag{16}
$$

and consequently

$$
\mathbf { B o u N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) \leq \operatorname* { m i n } _ { j \neq y ^ { \star } } \operatorname* { m i n } _ { \substack { \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } } m _ { y ^ { \star } , j } ( \mathbf { x } ) ,\tag{17}
$$

$i . e .$ , BOUND $f , \mathbf { x } ^ { \star } , \epsilon ^ { \left( O \right) }$ is a valid concrete lower bound ofthe margins over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ . Moreover, the right-hand side of Eq. (12) with $g = \underline { m } _ { y ^ { \star } , j }$ evaluates to exactly mi $1 _ { \mathbf { X } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } \underline { m } _ { y ^ { \star } , j } ( \mathbf { x } )$ , so the closed-form value computed in practice coincides with BO $\mathrm { J N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o )$ as defined in Eq. (6).

Proof. Fix o and write $\mathcal { X } : = \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ . Let ${ h ^ { ( 1 ) } , \dots , h ^ { ( K ) } }$ be the affine layers of $f$ in topological order. For each $i ,$ the bound propagation back-propagates linear relaxations to the input to obtain affine functions $\underline { { h } } ^ { ( i ) }$ and $\overline { { h } } ^ { ( i ) }$ of $\mathbf { x } .$ , and concretizes them entrywise into concrete bounds $\mathbf { \chi } _ { l ^ { ( i ) } }$ and $\mathbf { \boldsymbol { u } } ^ { ( i ) }$ which are then used to linearly relax the nonlinear activation that takes $\it { h ^ { ( i ) } }$ as input. The affine lower bound $\underline { m } _ { y ^ { \star } , j }$ is obtained by the same back-propagation applied directly to the margin $m _ { y ^ { \star } , j }$

Step 1 (soundness oflinear relaxation). Let $q$ be either an affine layer $\it { h ^ { ( i ) } }$ or a margin $m _ { y ^ { \star } , j }$ , and let $\underline { { q } }$ and q be the affine bounds obtained by back-propagating from q (for $q = m _ { y ^ { \star } , j } ,$ only $\underline { { q } } = \underline { { m } } _ { y ^ { \star } , j }$ is needed). The soundness of LiRPA-style bound propagation [46, 67, 73] rests on the following pointwise property: for every input x at which each activation preceding q receives an input within the concrete bounds used for its relaxation, i.e., $l ^ { ( i ^ { \prime } ) } \le h ^ { ( i ^ { \prime } ) } ( { \bf x } ) \overset { - } { \le } u ^ { ( i ^ { \prime } ) }$ for every layer $h ^ { ( \bar { i } ^ { \prime } ) }$ feeding such an activation, we have

$$
q ( \mathbf { x } ) \leq q ( \mathbf { x } ) \leq \overline { { q } } ( \mathbf { x } ) .\tag{18}
$$

Indeed, each linear relaxation sandwiches its activation on the interval given by the concrete bounds of its input, and back-propagating the relaxations through affine layers preserves the inequalities. Note that Eq. (18) does not refer to any input set: the input set enters the pipeline only through the concretization step that produces the concrete bounds, which prior work instantiates for boxes $[ 4 6 ] , \ell _ { p } -$ balls [73], or the full ℓ -ball $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ [67]. Our bounding procedure follows exactly the same pipeline, including all intermediate lower and upper bounds, and differs only in that every concretization is performed over X via Eq. (10) and Eq. (12). It therefore suffices to show that the resulting concrete bounds are valid on X.

Step 2 (intermediate bounds). We show by strong induction on i that $l ^ { ( i ) } \le h ^ { ( i ) } ( \mathbf { x } ) \le \pmb { u } ^ { ( i ) }$ for all $\mathbf { x } \in \mathcal { X }$ . Assume this holds for all $i ^ { \prime } < i$ (vacuously for $i = 1$ , where no activation precedes $h ^ { ( 1 ) }$ and hence $\underline { { { h } } } ^ { ( 1 ) } = \overline { { { h } } } ^ { ( 1 ) } = h ^ { ( 1 ) } )$ . Every activation preceding $\mathbf { \Psi } _ { h } ( i )$ takes as input some $h ^ { ( i ^ { \prime } ) }$ with $i ^ { \prime } < i ,$ so by the induction hypothesis the premise of Eq. (18) holds at every $\mathbf { x } \in \mathcal { X }$ , which gives $\underline { { h } } ^ { ( i ) } ( \mathbf { x } ) \leq h ^ { ( i ) } ( \mathbf { x } ) \leq \overline { { h } } ^ { ( i ) } ( \mathbf { x } )$ for all $\mathbf { x } \in \mathcal { X }$ . Each entry of $\boldsymbol { h } ^ { ( i ) }$ and $\overline { { h } } ^ { ( i ) }$ is an affine form of $\mathbf { x } .$ , and $l ^ { ( i ) } ~ ( \mathrm { r e s p . } ~ \pmb { u } ^ { ( i ) } )$ is obtained by evaluating the right-hand side of Eq. (12) (resp. Eq. (10)) on each entry of $\underline { { { h } } } ^ { ( i ) } ( \mathrm { r e s p . ~ } \overline { { { h } } } ^ { ( i ) } )$ ). By Lemma E.1, these values are exactly the entrywise min $\mathbf { x } ^ { \prime } \in \mathcal { X } \underline { { h } } ^ { ( i ) } ( \mathbf { x } ^ { \prime } )$ and $\mathrm { m a x } _ { \mathbf { x } ^ { \prime } \in \mathcal { X } } \overline { { h } } ^ { ( i ) } ( \mathbf { x } ^ { \prime } )$ , respectively. Hence, for every $\mathbf { x } \in \mathcal { X }$

$$
l ^ { ( i ) } = \operatorname* { m i n } _ { \mathbf { x } ^ { \prime } \in \mathcal { X } } \underline { { h } } ^ { ( i ) } ( \mathbf { x } ^ { \prime } ) \leq \underline { { h } } ^ { ( i ) } ( \mathbf { x } ) \leq h ^ { ( i ) } ( \mathbf { x } ) \leq \overline { { h } } ^ { ( i ) } ( \mathbf { x } ) \leq \operatorname* { m a x } _ { \mathbf { x } ^ { \prime } \in \mathcal { X } } \overline { { h } } ^ { ( i ) } ( \mathbf { x } ^ { \prime } ) = u ^ { ( i ) } .
$$

Step 3 (margin bound). By Step 2, all concrete bounds are valid on $x ,$ , so the premise of Eq. (18) holds for $q = m _ { y ^ { \star } , j }$ at every $\mathbf { x } \in \mathcal { X }$ , which gives Eq. (16) for every $j \neq y ^ { \star }$ . Taking the minimum of both sides of Eq. (16) over $\mathbf { x } \in \mathcal { X }$ and then over $j \neq y ^ { \star }$ yields Eq. (17) by Eq. (6). Finally, since $\underline { m } _ { y ^ { \star } , j }$ is an affine form of x, Lemma E.1 shows that the right-hand side of Eq. (12) with $g = \underline { m } _ { y ^ { \star } , j }$ evaluates to exactly m $\mathrm { i n } _ { \mathbf { x } \in \mathcal { X } } \underline { { m } } _ { y ^ { \star } , j } ( \mathbf { x } )$ , so BOUND ${ \mathbf \xi } ) _ { f , { \mathbf { x } } ^ { \star } , { \epsilon } } ( o )$ is computed in closed-form, without enumerating the perturbation patterns admitted by o.

Note that Eq. (16) is claimed only on $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ : the intermediate bounds, and hence the relaxations and $\underline { m } _ { y ^ { \star } , j }$ , are specific to o and need not remain valid on $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) \setminus B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ . This suffices, because both Eq. (5) and Eq. (6) quantify only over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ □

Lemma E.3 (Bound soundness). For any node $o , i f \mathbf { B } 0 \mathrm { U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) > 0 $ , then $\mathrm { S A F E } _ { f , { \mathbf { x } } ^ { \star } , \epsilon } ( o )$ holds.

Proof. By Lemma E.2,

$$
\operatorname* { m i n } _ { j \neq y ^ { \star } } \operatorname* { m i n } _ { \substack { \mathbf { x } \in \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } } m _ { y ^ { \star } , j } ( \mathbf { x } ) \geq \mathbf { B } \mathrm { O U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) .
$$

Hence, if BOUND $) _ { f , { \bf x } ^ { \star } , \epsilon } ( o ) > 0$ , the LHS is strictly positive, which is exactly $\mathrm { S A F E } _ { f , { \mathbf { x } } ^ { \star } , \epsilon } ( o )$ by Eq. (5). 口

Lemma E.4 (Initial node sufficiency). Recall the definition of the initial node for each level l:

$$
\Theta _ { l } = \left\{ \begin{array} { r } { { \left( \emptyset , 0 \right) , \quad i f l = 0 , } } \\ { { \left( \nu , l \right) , \quad i f l > 0 . } } \end{array} \right.\tag{19}
$$

$I f \mathbf { S A F E } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( \Theta _ { l } )$ holdsfor some $l \in \{ 0 , \ldots , \epsilon \}$ , then f is $\ell _ { 0 }$ locally robust at $\mathbf { x } ^ { \star }$ with budget ϵ.

Proof. We can observe that the only difference between $\operatorname { E q . } \left( 2 \right)$ and $\operatorname { E q . } \left( 5 \right)$ is the input space: $B _ { 0 } ^ { \epsilon } ( { \bf x } ^ { \star } )$ for Eq. (2), while $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ for Eq. (5). Comparing Eq. (4) with $\mathrm { \bar { E q } } . \ ( 1 ) , \ B _ { 0 } ^ { \epsilon } ( \mathbf { \bar { x } } ^ { \star } , \bar { o } )$ enforces a constraint on the set of perturbed pixels in addition to $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ . Thus, we have $\tilde { B _ { 0 } ^ { \epsilon } } ( \mathbf { x } ^ { \star } , \acute { o } ) \subseteq B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ and the equality holds when $o . S \stackrel { - } { = } \mathcal { V }$ or $o . l = 0$ . More precisely, if a node o satisfies $o . S = \mathcal { V }$ or $o . l = 0$ , and $\mathrm { S A F E } _ { f , { \mathbf { x } } ^ { \star } , \epsilon } ( o )$ holds, then Eq. (2) holds and the local robustness of $f$ over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ is verified. By the definition of initial nodes Eq. (19), for any initial node $\Theta _ { l }$ , it satisfies $\Theta _ { l } . \mathcal { S } = \mathcal { V }$ or $\Theta _ { l } . l = 0$ . Therefore, when $\mathbf { S } \mathbf { A F E } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( \Theta _ { l } )$ holds, then Eq. (2) holds and the local robustness of f over $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } )$ is verified. □

Lemma E.5 (Child containment). Given a branchable node $^ { O , }$ we have

$$
\forall o ^ { \prime } \in \mathtt { B R A N C H } ( o ) . \quad \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \subseteq \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) .
$$

$I . e . ,$ every child ball is contained in its parent ball.

Proof. Let us discuss o in two cases.

1. If $o = ( o . S , o . l )$ is a super node, then its child node is derived from $o ^ { \prime } \in \mathrm { B R A N C H } _ { k } ^ { \mathtt { s u p } } ( o )$ By Eq. (7), we have $o ^ { \prime } \overset { ^ { \bullet } } { = } ( o ^ { \prime } . \boldsymbol { S } , o . l )$ where $o ^ { \prime } . S \subset o . S$ and $o . l \le k = | o ^ { \prime } . S | < | o . S |$ . Let S be a non-empty set s.t. $S \overset { \cdot } { = } o . \overset { \triangledown } { S } \setminus \overset { \cdot } { o ^ { \prime } } . \overset { \triangledown } { S }$ . By Eq. (4), we have

$$
\begin{array} { l l l } { \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) } & { = } & { \left\{ \mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : \ | \ \{ v : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \} \setminus o ^ { \prime } . S | \leq \epsilon - o . l \right\} } \\ & { = } & { \left\{ \mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : \ | \ \{ v : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \} \setminus ( o . S \setminus S ) | \leq \epsilon - o . l \right\} } \\ & { \subseteq } & { \left\{ \mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : \ | \ \{ v : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \} \setminus o . S | \leq \epsilon - o . l \right\} } \\ & { = } & { B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } \end{array}
$$

2. If $o = ( o . S , o . l )$ is a concrete node, then its child node is derived from $o ^ { \prime } \in \mathrm { B R A N C H } ^ { \mathsf { c o n } } ( o )$ By Eq. (8), we have $o ^ { \prime } = ( o ^ { \prime } . S , o . l + 1 )$ where $o ^ { \prime } . S = o . S \cup \{ v ^ { \prime } \}$ and $\boldsymbol { v ^ { \prime } } \not \in o . s$ . Further, by Eq. (4), we have

$$
\begin{array} { r l r } { \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) } & { = } & { \{ \mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : | \left\{ v : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \right\} \setminus o ^ { \prime } . S | \leq \epsilon - o ^ { \prime } . l \} } \\ & { = } & { \{ \mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : | \left\{ v : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \right\} \setminus ( o . S \cup \left\{ v ^ { \prime } \right\} ) | \leq \epsilon - ( o . l + 1 ) \} } \\ & { \subseteq } & { \{ \mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } ) : | \left\{ v : \mathbf { x } _ { v } \neq \mathbf { x } _ { v } ^ { \star } \right\} \setminus o . S | \leq \epsilon - o . l \} } \\ & { = } & { B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } \end{array}
$$

where the inclusion holds because, for any set $T$ of pixels, $T \backslash o . S \subseteq ( T \backslash ( o . S \cup \{ v ^ { \prime } \} ) ) \cup \{ v ^ { \prime } \}$ and hence $| T \setminus o . S | \leq | T \setminus ( o . S \cup \{ v ^ { \prime } \} ) | + 1$

By the above two cases, we prove $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \subseteq B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ for BRANCH.

Lemma E.6 (Branching exhaustiveness). For any branchable node $^ { O , }$ the union of child balls generated by BRANCH(o) contains the parent ball:

$$
\bigcup _ { o ^ { \prime } \in \mathrm { B R A N C H } ( o ) } \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \supseteq \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) .\tag{20}
$$

It applies to both branching policies. $H o$ is a super node, then

$$
\bigcup _ { o ^ { \prime } \in \mathrm { B R A N C H } _ { k } ^ { s u p } ( o ) } \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \supseteq \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )\tag{21}
$$

holds; ifo is a concrete node, then

$$
\bigcup _ { o ^ { \prime } \in \mathrm { B R A N C H } ^ { c o n } ( o ) } \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \supseteq \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )\tag{22}
$$

holds.

Proof. Let us discuss o in two cases.

1. For any $\mathbf { x } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ where $o = ( o . S , o . l )$ is a super node, we have $\vert o . { \cal S } \vert > o . l$ . Writing $T = \{ \dot { v } : \mathbf { x } _ { v } \neq \mathbf { \dot { x } } _ { v } ^ { \star } \}$ , by $\operatorname { E q . } \left( 1 \right)$ and Eq. (4), we have $| T | \leq \epsilon$ and $| T \setminus ^ { \cdot } o . \mathcal { S } | \leq \epsilon - o . l$ Let $T _ { i } \subseteq T \cap o . S$ be any subset with $| T _ { i } | = \operatorname* { m i n } ( o . l , | T \cap o . S | )$ . Also, since $T _ { i } \subseteq o . S$ and $| T _ { i } | = \mathrm { m i n } ( o . l , | T \cap o . S | ) \leq o . l < | o . S |$ , there exists $T ^ { \prime } \stackrel { \cdot } { \subseteq } o . S \mathrm { ~ s . t . ~ } | T ^ { \prime } | = o . l$ and $T _ { i } \subseteq T ^ { \dot { \prime } } \subseteq o . S .$

By the definition of covering design, there exists o<sup>′</sup>.S ∈ C(o.S, k, o.l) s.t. $T ^ { \prime } \subseteq o ^ { \prime } . S \subset o . S$ Therefore, we have $T _ { i } \subseteq \breve { T } ^ { \prime } \subseteq \breve { o ^ { \prime } } . S \subset o . S \nonumber$ . Moreover, since ${ \dot { T _ { i } } } \subseteq T$ and $T _ { i } \subseteq o ^ { \prime } . S _ { }$ , we know ${ \cal T } _ { i } \subseteq { \cal T } \cap o ^ { \prime } . s$ and hence $| T _ { i } | = \operatorname* { m i n } ( o . l , | T \cap o . S | ) \leq | T \cap o ^ { \prime } . S |$

Then we discuss $| T \cap o . S |$ in two cases.

(a) $| T \cap o . { \cal S } | \geq o . l .$ Then $| T _ { i } | = \operatorname* { m i n } ( o . l , | T \cap o . S | ) = o . l \leq | T \cap o ^ { \prime } . S | \mathrm { ~ a n d ~ } | T \setminus o ^ { \prime } . S | =$ $\begin{array} { r } { | T | - | T \cap o ^ { \prime } . \displaystyle \mathcal { S } | \leq | T | - o . l \leq \epsilon - o . l . } \end{array}$

(b) $| T \cap o . S | < o . l$ . Then $| T _ { i } | =$ min $\lfloor ( o . l , | T \cap o . S | ) \ : = \ : | T \cap o . S | \ : \leq \ : | T \cap o ^ { \prime } . S |$ Since $T _ { i } \stackrel { \cdot } { = } T \cap o . { \cal S } \subseteq \stackrel { \cdot } { o ^ { \prime } . { \cal S } } .$ , then ${ \vec { T } } \setminus o ^ { \prime } . S \subseteq T ^ { \cdot } \setminus o . S$ . Namely, we can prove $| T \setminus o ^ { \prime } . \mathcal { S } | \leq | T \setminus o . \mathcal { S } | \leq \epsilon - o . l .$

Both cases verify that $| T \setminus o ^ { \prime } . S | \leq \epsilon - o . l .$ . By Eq. (4), we can prove i $\mathbf { \Gamma } _ { \mathbf { X } } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ , then there exists $o ^ { \prime } \in \mathrm { B R A N C H } _ { k } ^ { \mathrm { s u p } } ( o ) \ \mathrm { s . t . } \ \mathbf { x } \in \bar { B } _ { 0 } ^ { \epsilon } ( \mathbf { \dot { x } } ^ { \star } , \bar { o } ^ { \prime } )$ . Hence Eq. (21) holds.

2. If $o = ( o . S , o . l )$ is a concrete node, then its child node is derived from $o ^ { \prime } \in \mathrm { B R A N C H } ^ { \mathsf { c o n } } ( o )$ Let $\mathbf { x } ^ { \prime } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ . By $\operatorname { E q . } ( 4 ) , \mathbf { x } ^ { \prime }$ satisfies $| \left\{ v : \ \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \right\} \setminus o . S | \leq \epsilon - o . l$ . Further, by Eq. (8), we know there exists a child node $\begin{array} { r } { \dot { o ^ { \prime } } \in \mathbb { B R A N C H } ^ { \mathsf { c o n } } ( o ) } \end{array}$ s.t. $o ^ { \prime } . S = o . S \cup \{ v ^ { \prime } \}$ where $v ^ { \prime } \in \mathcal { V } \setminus o . s$

On the one hand, if $\{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \} \subseteq o . S .$ , then $| \left\{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \right\} \setminus o . S | = 0$ . Furthermore, $| \{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \} \setminus ( \setminus \mathcal { S } \cup \{ v ^ { \prime } \} ) | = 0$ where $v ^ { \prime } \in \dot { \mathcal { V } } \backslash o . S .$ . Also, as o is a branchable concrete node, $o . \bar { l } < \epsilon , \bar { \epsilon } - \acute { o . } l > 0 .$ . Then $\epsilon - \left( o . l + 1 \right) \ge 0$ . For $\mathbf { x } ^ { \prime } , | \left\{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \right\} \backslash ( o . S \cup \{ v ^ { \prime } \} ) | \leq$ $\epsilon - \left( o . l + 1 \right)$ still holds. Namely, by Eq. $( 8 ) , \mathbf { x } ^ { \prime } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } )$ holds.

On the other hand, if $\{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \} \subset o . S .$ , then $\{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \} \setminus o . S | > 0$ . That’s to say, there exists $v ^ { \prime } \in \{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \}$ but $\boldsymbol { v ^ { \prime } } \not \in o . s$ . Then $v ^ { \prime } \in \mathcal { V } \setminus \overline { { o . s } }$ and $| \{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \} \ \bar { \ }$ $( o . S \cup \{ v ^ { \prime } \} ) | = | \left\{ v : \mathbf { x } _ { v } ^ { \check { \prime } } \neq \mathbf { x } _ { v } ^ { \check { \star } } \right\} \backslash o . S | - 1$ . Furthermore, since $| \left\{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \right\} \left\backslash o . \mathcal { \bar { S } } \right| \leq$ $\dot { \epsilon } - o . l .$ , then $\{ v : \mathbf { x } _ { v } ^ { \prime } \neq \mathbf { x } _ { v } ^ { \star } \} \setminus \left. \left. o . \dot { S } \cup \left\{ v ^ { \prime } \right\} \right. \right| \leq \epsilon - o . l - 1 = \epsilon - \left( o . l + \check { 1 } \right)$ . By Eq. (8), we can find a child node $o ^ { \prime } = \bar { ( \it o . S \cup \{ v ^ { \prime } \} , o . l + 1 ) } \in$ BRANCH $\mathsf { \tilde { c o n } } ( o ) \mathrm { ~ s . t . ~ } \mathbf { x } ^ { \prime } \in \dot { B _ { 0 } ^ { \epsilon } } ( \bar { \mathbf { x } ^ { \star } } , o ^ { \prime } )$

By the above two cases, we prove if $\mathbf { x } ^ { \prime } \in B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ , then $\begin{array} { r } { \mathbf { x } ^ { \prime } \in \bigcup _ { o ^ { \prime } \in \mathrm { B R a N C H } ( o ) } B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) } \end{array}$ In another words, $\begin{array} { r } { \bigcup _ { o ^ { \prime } \in \mathrm { B R A N C H } ^ { \mathrm { c o n } } ( o ) } \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \supseteq \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) } \end{array}$ holds when o is a concrete node.

By the above two cases, we prove Eq. (21) and Eq. (22), and thus Eq. (20) is proven.

Corollary 1 (Parent safety from children). If all child nodes generated by BRANCH(o) are safe, then the parent node o is safe.

Proof. By Lemma E.6, every input in $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ belongs to at least one child ball. Since all child nodes are safe, all inputs in all child balls are safe. Hence all inputs in $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ are safe. □

Corollary 2 (Branching exactness). For any branchable node $^ { O , }$ the union of all child balls generated by BRANCH(o) equals exactly the parent ball, i.e.,

$$
\bigcup _ { o ^ { \prime } \in \mathrm { B R A N C H } ( o ) } \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) = \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) .\tag{23}
$$

Proof. By Lemma $\mathrm { E . 5 , }$ for any child node $o ^ { \prime } \in$ BRANCH(o), we have $B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \ \subseteq \ B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o )$ Hence

$$
\bigcup _ { o ^ { \prime } \in \mathrm { B R A N C H } ( o ) } \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ^ { \prime } ) \subseteq \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) .
$$

The reverse is proven by Lemma E.6. Combining the two inclusions gives Eq. (23).

Theorem 1 (Relative soundness). Suppose the MILP solver is sound. Given a network $f ,$ a nominal input $\mathbf { x } ^ { \star }$ , and a maximum number ofperturbed pixels $\epsilon , i f$ FOCUSBRANCH $( f , \mathbf { x } ^ { \star } , \dot { \epsilon } )$ returns Verified, then $f$ is locally robust with respect to $\mathbf { x } ^ { \star }$ and ϵ.

Proof. By Lemma $\mathrm { E . 4 } .$ , we know network $f$ is locally robust w.r.t. $\mathbf { x } ^ { \star }$ and ϵ if the initial node $\Theta _ { l }$ is safe, $\mathrm { i . e . , ~ S A F E } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( \Theta _ { l } )$ . However, directly proving Θ is safe is challenging; instead, we prove safety indirectly using decomposition. By Corollary 1, we know that $\Theta _ { l }$ is provably safe if all its child nodes generated by BRANCH are safe. On the one hand, by Lemma E.3, for any child node $o ^ { \prime }$ meeting BOUND $f , \mathbf { x } ^ { \star } , \epsilon  \bigl ( o ^ { \prime } \bigr ) > 0 .$ , it is proven safe as $\mathbf { B O U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o )$ estimates the lower bound of the margin function by Eq. (6) and Eq. (5). On the other hand, the rest child nodes are proven safe if an MILP solver return infeasible for all these nodes and cannot identify any adversarial example. Furthermore, since we assume the MILP solver is sound, it can correctly validate $\mathtt { M I L P } ( f , B _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , \bar { o ^ { \prime } } ) )$ for any child node $o ^ { \prime }$

If FOCUSBRANCH $( f , \mathbf { x } ^ { \star } , \epsilon )$ returns Verified, then O is dequeued to be empty as the MILP solver returns infeasible for all child nodes and BO $\mathrm { U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o )$ directly proves a node is safe. Then $\mathbf { S } \mathbf { A F E } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( \Theta _ { l } )$ is proven true and, therefore, $f$ is locally robust w.r.t. $\mathbf { x } ^ { \star }$ and ϵ by Lemma E.4.

Lemma E.7 (Finite Branching). $H o$ is branchable, then BRANCH(o) generates non-empty, finite child nodes.

1. $H f o = ( o . S , o . l )$ is a super node, $\mathrm { B R A N C H } _ { k } ^ { s u p } ( o )$ contains $| \mathcal { C } ( o .  { S } , k , o . l ) |$ nodes, each child node o<sup>′</sup> is oftheform $( \boldsymbol { o } ^ { \prime } . \boldsymbol { S } , \boldsymbol { o } . \boldsymbol { l } )$ with $o ^ { \prime } . S \stackrel {  } { \subset } o . { \dot { S } }$ and $| o ^ { \prime } . \dot { S } | \overset { \cdot } { = } k _ { \mathrm { \scriptsize ~ \cdot } }$ , where $\quad \sum l \leq k < | o . S |$

2. If o is a concrete node, B $\operatorname { \mathrm { ? A N C H } } ^ { c o n } ( o )$ contains exactly $V - o . l$ nodes, each of the form $\left( o . S \cup \left\{ v \right\} , o . l + 1 \right) w i t h \left| o . S \cup \left\{ v \right\} \right| = o . l + 1$

Proof. Let us discuss o into two cases.

1. If o is a super node, then, according to CALZONE [41] and the definition of covering design $[ 1 2 , \bar { 3 } 9 , 5 1 ] , \mathcal { C } ( o . S , k , o . l )$ in Eq. (7) is a finite non-empty collection of k-element subsets of $_ { o . S }$ where $k < | o . S |$ for super node $o .$

2. If o is a concrete node, then, by Eq. (8), any child node $o ^ { \prime } \in \mathbf { B R A N C H } ^ { \mathsf { c o n } } ( o )$ has one more element $( \mathrm { i . e . , } o ^ { \prime } . S = o . S \cup \{ v \}$ where $v \in \mathcal { V } \setminus o . S )$ and increments its level by one $( \mathrm { i . e . }$ $o ^ { \prime } . l = o . l + 1 )$ for concrete node o. In total, BRA $\mathrm { N C H } ^ { \mathsf { c o n } } ( o )$ generates $V - | o . S |$ child nodes where $o . l = | o . S |$ . That is to say, ${ \mathrm { B R A N C H } } ^ { \mathsf { c o n } } ( o )$ contains exactly $V - \dot { | } o . \dot { S | } = V - o . l$ nodes, each of the form $( o . S \cup \{ v \} , o . l + 1 )$ with $\vert o . S \cup \{ v \} \vert = \stackrel { \cdot } { o . l } + 1$

From the above two cases, we prove the lemma.

Definition 1 (Progress measure). We use $\mu$ to measure the progress of a node o during the process of branching propagation. Specifically, define

$$
\mu ( o ) : = ( \epsilon - o . l , | o . S | ) \in \{ 0 , \ldots , \epsilon \} \times \{ 0 , \ldots , V \} ,
$$

ordered lexicographically: $( a , b ) < _ { \mathrm { l e x } } ( a ^ { \prime } , b ^ { \prime } )$ iff $a < a ^ { \prime } .$ , or $a = a ^ { \prime }$ and $b < b ^ { \prime } .$

Lemma E.8 (Branching progress). For every branchable node o and every child node $o ^ { \prime } \in$ BRANCH(o), we have $\mu ( o ^ { \prime } ) < _ { \mathrm { l e x } } \mu ( o )$ .

Proof. We discuss o into two cases.

1. If o is a super node, then by Eq. (7), o<sup>′</sup>.l = o.l and $| o ^ { \prime } . S | = k < | o . S |$ , so the first components of $\mu ( o ^ { \prime } )$ and $\mu ( o )$ agree and the second strictly decreases.

2. If o is a concrete node, then by Eq. (8), $o ^ { \prime } . l = o . l + 1$ , so the first component strictly decreases (this suffices for $< _ { \mathrm { l e x } }$ even though the second component grows by one).

Both decrease their child nodes’ progress measure.

Lemma E.9 (Concrete propagation). For any branchable concrete node $^ { O , }$ every child node in ${ \mathrm { B R A N C H } } ^ { c o n } ( o )$ is a concrete node.

Proof. By Eq. (8), every child node $o ^ { \prime }$ generated by BRANCH $[ ^ { \mathsf { c o n } } ( o )$ satisfies $o ^ { \prime } . S = o . S \cup \{ v \}$ and $o ^ { \prime } . l \stackrel { \cdot } { = } o . l + \mathrm { 1 }$ . Since the parent node o is concrete, $\mathrm { i . e . , } | o . S | = \partial . l .$ , we have $| o ^ { \prime } . S | = | o . S | \acute { + } 1 =$ $o . l + 1 = o ^ { \prime } . l$ , which proves $o ^ { \prime }$ is a concrete node. □

Lemma E.10 (Finite Propagation). Given $l _ { s } \in \{ 0 , \ldots , \epsilon \}$ , any branching propagation process starting from $\Theta _ { l _ { s } }$ has at most $( \epsilon - l _ { s } ) + ( V - l _ { s } )$ branching depth, which isfinite.

Proof. We discuss in two cases:

1. $\mathrm { I f } l _ { s } \in \{ 1 , \ldots , \epsilon \}$ , then Algorithm 1 starts with $\Theta _ { l _ { s } }$ , whose progress measure $\mu ( \Theta _ { l _ { s } } )$ equals $( \epsilon - l _ { s } , \bar { V } )$ . By Lemma $\bar { \mathrm { E } } . 8 ,$ it strictly decreases along the super branching propagation process, so it has at most $( V - l _ { s } )$ super branching steps away from level- $- l _ { s }$ concrete nodes, whose progress measure is $( \epsilon - l _ { s } , \bar { l } _ { s } )$ . This strictly decreases along concrete branching propagation process according to Lemma E.8, and since the rest of the branching propagation process only includes concrete branching and the search tree consists only concrete nodes (Lemma E.9), level- $- l _ { s }$ concrete nodes have at most $\left( \epsilon - l _ { s } \right)$ concrete branching steps away from finally reaching the unbranchable concrete nodes (the leaves).

2. If $l _ { s } = 0$ , then Algorithm 1 starts with $\Theta _ { 0 } .$ , which is a branchable concrete node, whose progress measure $\mu ( \Theta _ { 0 } )$ equals $( \epsilon , 0 )$ . By Lemma E.9, the whole branching propagation process only includes concrete branching and the search tree consists only concrete nodes. By Lemma E.8 the progress measure strictly decreases along the concrete branching propagation process, so it has at most ϵ concrete-branching steps away from finally reaching the unbranchable concrete nodes (the leaves).

From the above two cases, we prove the lemma.

Theorem 2 (Relative completeness). Suppose the MILP solver is sound and complete. Given a network $f ,$ a nominal input $\mathbf { x } ^ { \star } { } _ { \mathrm { : } }$ , and a maximum number of perturbed pixels ϵ, if f is locally robust with respect to $\mathbf { x } ^ { \star }$ and $\epsilon ,$ then FOCUSBRANCH(f, x<sup>⋆</sup>, ϵ) returns Verified.

Proof. By Lemma E.10, we know Algorithm 1 finally terminates and every unbranchable node at level ϵ is a concrete node. If network f is robust, the MILP solver returns infeasible for each unbranchable node. That is to say, all unbranchable nodes are safe. Moreover, by Corollary 1, the initial node $\Theta _ { l }$ is proven safe as all its child nodes are safe. Namely, FOCUSBRANCH(f, x<sup>⋆</sup>, ϵ) returns Verified.

## F Technical Details and Optimization

## F.1 Super Branching Policy for Initial Nodes

For initial node at level $l > 0 , \mathcal { R } ^ { ( l ) } [ | \Theta _ { l } . S | ] \ ( \mathrm { i . e . , \ } \mathcal { R } ^ { ( l ) } [ V ] )$ is special. We either assign $\mathcal { R } ^ { ( l ) } [ V ]$ to l (especially for small $l , { \mathrm { e . g . } } , l \leq 2 )$ , i.e., generate child nodes with their focus sets coming from $\binom { \mathcal { V } } { l }$ , or pick the best CVD [43] (especially for larger $l , { \mathbf { e . g . } } , l \geq 3 )$ , which outperforms pure covering design [41] (Section 2).

## F.2 Detailed Verification Planning Technique

Given $\hat { \mathbf { P } } , \hat { \mathbf { T } }$ , the number of unknown-parents-only concrete nodes for each level $\tilde { \mathbf { n } } \in \mathbb { R } ^ { ( \epsilon + 1 ) }$ can be estimated as follows:

$$
\tilde { \mathbf { n } } _ { l } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } l = 0 , } \\ { \left[ \binom { V } { l } \cdot \left( 1 - \hat { \mathbf { P } } _ { l - 1 , l - 1 } \right) ^ { 2 } \right] , } & { \mathrm { i f } l > 0 . } \end{array} \right.\tag{24}
$$

Therefore, the per-level estimated duration on bounding those nodes can be computed by $\tilde { \mathbf { t } } _ { l } = \hat { \mathbf { T } } _ { l , l } { \cdot } \tilde { \mathbf { n } } _ { l }$ However, Eq. (24) is not accurate, but empirically works.

We provide a more detailed version of Algorithm 3 for choosing the best verification plan.

Algorithm 3 Choose Best Plan   
1: procedure CHOOSEBESTPLAN(f, x<sup>⋆</sup>, ϵ)   
Input: A classifier $f ,$ , a nominal input image $\mathbf { x } ^ { \star } \in \mathcal { D } ,$ , and a maximum number of perturbed   
pixels ϵ.   
Output: The level $l _ { s }$ to start verification, and a super branching strategy $\mathcal { R } .$   
2: $t _ { m i n } \gets \infty$   
3: $( l _ { s } , \mathcal { R } ) \gets ( 0 , \emptyset )$   
4: (P<sup>ˆ</sup> , T<sup>ˆ</sup> ) ← SampleNodesOfAllShape $( f , \mathbf { x } ^ { \star } , \epsilon )$   
5: for all $l \in \{ 0 , \ldots , \epsilon \}$ do   
6: $\begin{array} { r } { t _ { t a i l } \gets \sum _ { i = l + 1 } ^ { \epsilon } \tilde { \mathbf { t } } _ { i } } \end{array}$   
7: $t _ { \mathsf { c o n } } \gets \hat { \mathbf { T } } _ { l , l } \cdot \left( \begin{array} { l } { V } \\ { l } \end{array} \right) + t _ { t a i l }$   
8: if $\cdot _ { t _ { \tt c o n } } < t _ { m i n }$ then   
9: t<sub>min</sub> ← t<sub>con</sub>   
10: $( l _ { s } , \mathcal { R } [ V ] ) \gets ( l , l )$   
11: $\mathbf { i f } l \in \{ 3 , 4 , 5 , 6 \}$ then ▷ all CVD supported t   
12: $( \mathcal { R } ^ { ( l ) } , \mathcal { G } ^ { ( l ) } , t _ { \mathtt { s u p } } ^ { ( l ) } )$ ← GetRefinementStrategyAndBestCVD $( \hat { \mathbf { P } } _ { : , l } , \hat { \mathbf { T } } _ { : , l } , l )$   
13: i $ { \mathbf { \rho } } :  { t _ { \mathtt { s u p } } } ^ { ( l ) } +  { t _ { t a i l } } < 1$ t<sub>min</sub> then   
14: $t _ { m i n } \gets t _ { \mathsf { s u p } } ^ { ( l ) } + t _ { t a i l }$   
15: $( l _ { s } , \mathcal { R } ) \gets ( l , \mathcal { R } ^ { ( l ) } )$   
16: $\mathcal { R } [ V ]  \mathcal { G } ^ { ( l ) }$   
17: return $( l _ { s } , \mathcal { R } )$

Fitting the safe rates and bounding times. At each level l, the planner samples super nodes across all focus-set sizes $k \geq l$ (Line 4 of Algorithm 3). Let $n _ { k , l }$ denote the number of sampled nodes of size $k ,$ and $s _ { k , l }$ the number proven safe. Directly using the empirical safe rate $s _ { k , l } / n _ { k , l }$ introduces two problems. First, whenever all sampled nodes of size k are verified safe, the empirical rate saturates at 1. The planner then assumes size-k super nodes never require further branching and selects the super branching strategy among such sizes (Line 12) purely by measured bounding time. Because these runtimes are nearly identical, minor timing noise ends up dictating the branching plan, causing run-to-run variance on the exact same instance. Second, due to sampling variance, the raw estimates are not guaranteed to be monotonic in $k ,$ even though larger focus sets admit strictly more perturbations and should be harder to verify.

To address both issues, we smooth the safe-rate estimates at each level l by fitting a logistic curve in log k [7]:

$$
\hat { \mathbf { P } } _ { k , l } = \sigma ( a _ { l } + b _ { l } \log k ) , \qquad b _ { l } \leq 0 ,
$$

where $\sigma$ is the standard logistic function. We estimate $a _ { l }$ and $b _ { l }$ via binomial maximum likelihood over all sampled sizes, with $n _ { k , l }$ trials and success probability $\hat { \mathbf { P } } _ { k , l }$ [29]. The constraint $b _ { l } \leq 0$ ensures the rate is non-increasing in $k ,$ and $\hat { \mathbf { P } } _ { k , l }$ remains strictly in (0, 1) unless all samples at level l are uniformly safe or unsafe. Empirically, the curve matches the samples within binomial noise. For concrete nodes (Eq. (24)), we keep the direct empirical estimates, as each level has only a single concrete focus-set size and thus no curve to fit. Similarly, we model the per-node bounding time linearly as $\hat { \mathbf { T } } _ { k , l } = c _ { l } + d _ { l } k \left( d _ { l } \geq 0 \right)$ using the robust Theil–Sen estimator [40, 49]. To avoid skewing the regression, we exclude the first sampled size (which incurs GPU warm-up overhead) as well as any size that exhausts fewer nodes than the full sampling budget (where per-node latency is dominated by batch overhead). Both models are fitted using SciPy [57].

## F.3 Detailed Concrete Branching and Pruning Technique in Practice

In practice, we perform concrete-node branching and pruning in batches. Rather than being expanded one by one, all unknown concrete nodes at the same level are collected and branched together, then pruned together.

Canonical concrete branching. We refer to the practical concrete branching as the canonical concrete branching. We represent each concrete node by its focus set in canonical increasing order, $\mathrm { i . e . , }$ a level-l node is stored as a tuple $( v _ { 1 } , \ldots , v _ { l } )$ with $v _ { 1 } < \cdots < v _ { l }$ . This allows us to generate children by canonical augmentation [20, 37]: instead of appending every pixel in $\gamma \backslash s ,$ , we append only pixels larger than the last element of the tuple. For a concrete node’s focus set $\mathcal { S } = ( v _ { 1 } , \ldots , v _ { l } )$ the batched concrete split therefore generates

$$
\left\{ \left( v _ { 1 } , \dots , v _ { l } , v \right) : \ v _ { l } < v \leq V \right\} ,
$$

with the convention $v _ { 0 } : = 0$ when $l = 0 .$ . This produces each $( l + 1 )$ -subset exactly once, through its unique canonical predecessor given by its first l elements, and avoids duplicate children that would otherwise arise from independently taking the full upper shadow of each unknown node.

Concretely, let $U _ { l } \in \mathbb { N } ^ { n _ { l } \times l }$ store the $n _ { l }$ unknown concrete nodes at level l in increasing order. For each row ${ \dot { U } } _ { l } [ i ]$ , we compute the number of valid extensions as $V - U _ { l } [ i , l ]$ , where $U _ { l } [ i , l ]$ denotes the last element of the tuple. The implementation then repeats each parent row according to this number and appends the corresponding range of larger indices. This gives a vectorized implementation of canonical concrete branching on GPU, avoiding explicit Python loops over nodes or children.

Pruning concrete child nodes. These generated concrete child nodes are then pruned given the collection of unknown concrete nodes $\mathcal { O } ^ { \mathrm { u n k } }$ at level $l ,$ as shown in Algorithm 4. Specifically, for each child node o<sup>′</sup> at level $l + 1 .$ , we check how many subsets are in $\mathcal { O } ^ { \mathrm { u n k } }$ . If fewer than l + 1 of them are in $\mathcal { O } ^ { \mathrm { u n k } }$ , some parent of $o ^ { \prime }$ is safe, so o<sup>′</sup> is pruned. In practice, this process is CUDA-accelerated.

Algorithm 4 Prune those concrete child nodes with any safe parents   
1: procedure $\operatorname { P R U N E } ( \mathcal { O } ^ { \mathrm { u n k } } , \mathcal { O } ^ { \mathrm { c o n } } )$   
Input: A list of all unknown concrete nodes $\mathcal { O } ^ { \mathrm { u n k } }$ from the current level and a list of child   
concrete nodes $\mathcal { O } ^ { \mathtt { c o n } }$ at the next level (generated by ${ \mathcal { O } } ^ { \mathrm { u n k } } )$   
Output: All concrete nodes $\mathcal { O } ^ { \mathtt { c o n } \prime }$ at the next level without any safe parents, i.e., all their parents   
are from O<sup>unk</sup>.   
2: $\mathcal { O } ^ { \mathsf { c o n } \prime }  \emptyset$   
3: for all $o \in \mathcal { O } ^ { \mathtt { c o n } }$ do   
4: if | $\left\{ o ^ { \prime } \in \mathcal { O } ^ { \mathrm { u n k } } : o ^ { \prime } . \mathcal { S } \subset o . S \right\} | =$ o.l then ▷ $o . l = o ^ { \prime } . l + 1$   
5: $\mathsf { \tilde { O } } ^ { \mathsf { c o n } \prime } \gets \mathsf { O } ^ { \mathsf { c o n } \prime } \cup \{ o \}$   
6: return $\mathcal { O } ^ { \mathtt { c o n } \prime }$

## F.4 Probe and Probe Thickness

At the beginning of the planning, we will first do probing before we do the sampling. Starting from $\Theta _ { 0 }$ we check the worst bound level by level, while keeping track of a fixed number (probe\_thickness) of worst case unknown concrete nodes (i.e., least tight bound that $\mathsf { B O U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) < 0 )$ which will be branched on to generate next level concrete nodes.

The goal is to check on which levels we will have unknown concrete nodes if we choose that level to start verification. If a level $l \in \{ 3 , 4 , 5 , 6 \}$ (all CVD [43] supported t) contains unknown concrete nodes, then FOCUSBRANCH will place an upper bound (we set it to be 15) on the mean focus set size of $\mathcal { R } ^ { ( l ) } [ V ]$ to avoid excessive steps for recursively super branching the parent super nodes of those unknown concrete nodes. When level gets deeper, some unknown super nodes cannot be detected via sampling due to the sheer volume of the combinatorial space if they are few, which results in inaccurate $\hat { \mathbf { P } }$ that can fool the verification planner to select a best geometry (for constructing CVD) where the mean focus set size is very large, and to compute a super branching strategy with dense refinement steps. This can result in an explosion of the redundant child super nodes, making the estimation of the overall verification time varies from the real case.

## F.5 A More Detailed Version for Algorithm 1

Here we show a more detailed version which reflects FOCUSBRANCH in practice. Note that Algorithms 5 and 6 instantiate Algorithm 1 with a level-synchronous worklist, batched bounding, canonical augmentation, and cross-parent pruning.

Algorithm 5 Verify the $\ell _ { 0 }$ local robustness for neural network classifier f   
1: procedure $\operatorname { F O C U S B R A N C H } ( f , \mathbf { x } ^ { \star } , \epsilon )$   
Input: A classifier $f ,$ a nominal input image $\mathbf { x } ^ { \star }$ , and a perturbation budget ϵ   
Output: Verified if the verification is successful, an adversarial example otherwise   
2: $( l _ { s } , \mathcal { R } ^ { ( l _ { s } ) } ) \gets \mathrm { C H o o s E B E S T P L A N } ( f , \mathbf { x } ^ { \star } , \epsilon )$   
3: $\mathcal { \dot { O } }  \{ \Theta _ { l _ { s } } \}$ ▷ Enqueue an initial node at level $l _ { s }$   
4: $\mathcal { O } ^ { \mathrm { u n k } }  \emptyset$   
5: while $\mathcal { O } \neq \emptyset$ do   
6: $o  \mathcal { O } .$ .dequeue $( )$   
7: if $\mathsf { B O U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ) \le 0$ then   
8: $\mathbf { i f } \left| o . S \right| > { o . l }$ then   
9: $\mathcal { O }  \mathcal { O } \cup \mathop { \mathrm { B R A N C H } } _ { \mathcal { R } ^ { ( l _ { s } ) } [ | o . S | ] } ^ { \operatorname* { s u p } } ( o )$   
10: else ▷ Collect unknown concrete nodes at level $l _ { s }$   
11: $\mathcal { O } ^ { \mathrm { u n k } }  \mathcal { O } ^ { \mathrm { u n k } } \cup \{ o \}$   
12: return VERIFYUNKNOWNNODES $f , \mathbf { x } ^ { \star } , \epsilon , O ^ { \mathrm { u n k } } )$

## G Hyper-parameters

The hyper-parameters of FOCUSBRANCH are as follows: the number of GPU workers is set to 4, and the number of MILP workers to 5 (except for bcv5, where we use 50 for consistency with COVERD [43]). The maximum number of sampled super nodes for each k-sized focus set at each level is set to 200 per GPU worker (in total 800), except for bcv4 it’s 100 per GPU worker (in total 400). The upper bound of k is set to 200 (same as COVERD). Super nodes are stored in a batch pool before batched bounding. The batch size for bounding super nodes are set to 800 for fully connected networks and 400 for CNNs (5 for bcv4). The size of the super node pool for each focus-set size are set to 2 times (4 times for bcv4) the super batch size (the batch size to bound super nodes), which is 1,600 for fully connected networks and 800 for CNNs (20 for bcv4). For concrete nodes, the maximum number of samples per level is set to 5,000 per GPU worker (in total 20,000) for fully connected networks and 500 per GPU worker (in total 2,000) for CNNs (in total 400 for bcv4). The concrete batch size is set to 4,000 for fully connected networks and 200 for CNNs (5 for bcv4). The probe thickness (at most how many worse concrete nodes to keep track of at each level during probing; see Appendix F.4) is set to 50 for fully connected and 5 for CNNs. Finally, the timeout is set to 1800 seconds (3600 for bcv4).

```latex
Algorithm 6 Verify the next level concrete nodes without any safe parents
1: procedure VERIFYUNKNOWNNODES $( f , \mathbf { x } ^ { \star } , \epsilon , O ^ { \mathrm { u n k } } )$
Input: A classifier $f ,$ a nominal input $\mathbf { x } ^ { \star } \in \mathcal { D } ,$ a perturbation budget ϵ, and all unknown concrete
nodes $\mathcal { O } ^ { \mathrm { u n k } }$ from the current level.
Output: Verified if the verification is successful, an adversarial example otherwise
2: if ${ \mathcal { O } } ^ { \mathrm { u n k } } = \emptyset$ then return Verified
3: if $\mathcal { O } ^ { \mathrm { u n k } } { } _ { 1 } . l = \epsilon$ then
4: for all $o \in \mathcal { O } ^ { \mathrm { u n k } }$ do
5: $\left( \mathbf { f e a s i b l e } , \mathbf { x } ^ { \mathsf { a d v } } \right) \gets \mathsf { M I L P } \left( \mathbf { \underline { { f } } } , \mathcal { B } _ { 0 } ^ { \epsilon } ( \mathbf { x } ^ { \star } , o ) \right)$
6: if feasible then return $\mathbf { x } ^ { \mathrm { a d v } }$
7: return $\mathtt { V e r i f i e d }$
8: $\begin{array} { r l } { \mathcal { O } ^ { \mathrm { c o n } }  } & { { } \bigcup \ \mathsf { \& } \mathsf { R A N C H } ^ { \mathrm { c o n } } ( o ) } \end{array}$ ▷ Canonical concrete branching applies
$o { \in } \bar { \mathcal { O } } ^ { \mathrm { u n k } }$
9: $\mathcal { O } ^ { \mathrm { c o n } }  \mathrm { P R U N E } ( \mathcal { O } ^ { \mathrm { u n k } } , \mathcal { O } ^ { \mathrm { c o n } } )$ $\triangleright \mathcal { O } _ { 1 } ^ { \mathsf { c o n } } . l = \mathcal { O } ^ { \mathrm { u n k } } \ i . l + 1$
10: $\mathcal { O } ^ { \mathrm { u n k } \prime }  \{ o ^ { \prime } \in \mathcal { O } ^ { \mathrm { c o n } } : \mathrm { B o U N D } _ { f , \mathbf { x } ^ { \star } , \epsilon } ( o ^ { \prime } ) \leq 0 \}$ ▷ $\mathcal { O } ^ { \mathrm { u n k } / } { } _ { 1 } . l = \mathcal { O } ^ { \mathrm { u n k } } { } _ { 1 } . l + 1$
11: return $\operatorname { V E R I F Y U N K N O W N N O D E S } ( f , \mathbf { x } ^ { \star } , \epsilon , \mathcal { O } ^ { \mathrm { u n k \prime } } )$
```

The base seed defaults to 42, which is the same as in COVERD. And we keep the same per instance seed settings, too, where for every instance (at index i), the running seed will be set to $4 2 + i$

Note that if the estimated duration is larger than twice the timeout, we regard the instance as timed out and skip the verification to save time. This heuristic can only ever cost a verified instance, so the reported numbers are conservative.

## H Supplementary Evaluation

## H.1 Details of Benchmarks

The $\ell _ { 0 }$ adversarial training. Adversarially trained networks are frequently used in robustness verification. Some adversarially trained networks in prior work [41, 43] are trained under $\ell _ { \infty }$ settings (against PGD attack [28]). We include networks adversarially retrained under $\ell _ { 0 }$ settings in our experiment, which are generally easier to be verified safe within the time limit by $\ell _ { 0 }$ robustness verifier. Note that the certified adversarial defense training (as referenced above) is only applicable to fully connected networks.

Why the two exception networks. We report bcv4 and bcv5 only at $\epsilon = 3$ . These two networks are challenging for different reasons. The bcv4 network is substantially larger and deeper than the other benchmarks, as reflected by its number of neurons in Table 1. Consequently, its concrete margin lower bounds are looser than those of shallower networks, and FOCUSBRANCH already relies heavily on MILP solving even at $\epsilon = 3$ . The bcv5 network is not significantly larger than the other benchmarks, but it is evaluated on multi-channel CIFAR-10 inputs. Since we consider pixel-wise $\ell _ { 0 }$ perturbations, $\epsilon = 3$ allows modifying three pixels, corresponding to 3 pixels × 3 channels = 9 input features. This also leads to loose bounds and substantial reliance on MILP solving even at $\epsilon = 3 .$ Therefore, we restrict both bcv4 and bcv5 to $\epsilon = 3$ in the evaluation (same as prior work [41, 42, 43]). We use a timeout of 3600 seconds for bcv4 due to its size, and 1800 seconds for all other networks.

Architectural preference in starting level. For fully connected networks, there is a non-negligible fraction of instances that use concrete branching according to Table 2. However, for CNNs, almost all starting levels are ϵ and thus concrete branching is barely used. This is due to the sparsity of the matrix representation of the convolution operator, which results in sparse-weighted affine bounds when the layer-wise relaxation is backpropagated. Concretizing a sparse-weighted affine bound over an input space constrained by nodes whose level is less than ϵ can result in looser concrete bounds than those of non-sparse affine bounds. This is because for a sparse-weighted affine bound, the contribution $\Delta$ is concentrated only on a few pixels (the size of the flattened convolutional kernel), i.e., only a few $\Delta _ { \imath }$ are non-zero. Thus for a node o where $o . l < \epsilon ,$ there is at least one perturbation budget that can select the most-contributing pixels across $\nu ,$ , and it will select the non-zero ones, which is a large proportion compared to the extreme case – perturbing all the pixels. Therefore, the verification planner will always pick ϵ as the starting level since the safe rates of the ϵ-level nodes are significantly higher than other levels’ due to their tighter bounds.

Table 2: Main results of FOCUSBRANCH on pixel-wise $\ell _ { 0 }$ robustness verification. For each benchmark and perturbation budget ϵ, we report the number of correctly classified images, candidate instances, verified instances, timed out instances, mean runtime on verified instances, starting level statistics and number of instances where concrete branching applied.
<table><tr><td>Dataset</td><td>Network</td><td>Neurons</td><td> $\mathrm { I m g }$ </td><td>CC € 3</td><td></td><td>Cand. 90</td><td>R 90</td><td>T/O R/Cand.(%) 0</td><td>100.0%</td><td>Time (s) ls &lt; ∈ 0.3 90</td><td>BRANCHcon 21</td><td> $\overline { { l _ { s } = \epsilon } }$  0</td></tr><tr><td colspan="16"></td></tr><tr><td rowspan="9">MNIST</td><td></td><td></td><td>100</td><td>4 100 5 6 3 4 5</td><td>72 57 46 94 88</td><td></td><td>70 2 52 5 29 17 94 0 87 1</td><td>97.2% 91.2% 63.0% 100.0% 98.9% 88.6%</td><td>10.1 17.8 53.7 0.3 6.1 48.0</td><td>69 52 28 94 87</td><td>28 30 22 14 36</td><td>1 0 1 0 0</td></tr><tr><td>afc2</td><td>1024</td><td>100</td><td>100 6 3</td><td></td><td>79 68 92 84 74</td><td>70 38 92 84 67</td><td>9 30 55.9% 0 100.0%</td><td>90.9 0.5</td><td>70 38 92</td><td>45 26 11 30</td><td>0 0 0</td></tr><tr><td>afc3</td><td>1536</td><td>100</td><td>100 4 5 6 3 4</td><td>66 89 85</td><td>42 89 85</td><td>0 7 24 0 0</td><td>100.0% 90.5% 63.6% 100.0% 100.0%</td><td>12.9 34.9 193.1 &lt;0.1 0.5</td><td>84 67 42 89 85</td><td>43 29 3 20</td><td>0 0 0 0 0</td></tr><tr><td>afc4 afc5</td><td>384 1000</td><td>100 100</td><td>99 5 6 3 100 4</td><td>73 61 89 80 70</td><td>70 49 89 80 61</td><td>3 12 0 0 9</td><td>95.9% 80.3% 100.0% 100.0% 87.1%</td><td>5.4 53.5 0.9 30.3 188.4</td><td>70 49 88 62 23</td><td>34 29 27 35 14</td><td>0 0 1 18</td></tr><tr><td>bfc1</td><td>1000</td><td>100</td><td>5 6 3 4 99 5</td><td>60 91 82 68 59</td><td>29 91 81 47 13</td><td>31 0 1 21 46</td><td>48.3% 100.0% 98.8% 69.1% 22.0%</td><td>374.8 4.7 82.3 192.4 401.1</td><td>6 52 25 8 2</td><td>3 31 18 7 0</td><td>38 23 39 56 39</td></tr><tr><td>bcv1</td><td>3604</td><td>100</td><td>6 3 98 4 5 6</td><td>87 81 73 64</td><td>87 81 69 49</td><td>0 0 4 15</td><td>100.0% 100.0% 94.5% 76.6%</td><td>7.5 13.6 133.8 359.1</td><td>1 1 0 0</td><td>0 0 0 0</td><td>11 86 80 69 49</td></tr><tr><td>bcv2</td><td>3604</td><td>100</td><td>3 94 4 5 6</td><td>84 74 67 62 3 90</td><td>85</td><td>84 0 74 0 67 0 55 7 90 0</td><td>100.0% 100.0% 100.0% 88.7% 100.0%</td><td>8.7 19.6 49.4 131.0 12.6</td><td>4 6 1 1 1</td><td>2 0 0 0 1</td><td>80 68 66 54 89</td></tr><tr><td>bcv3</td><td>4504</td><td>100</td><td>99 4 5 6</td><td>85 76 68 9</td><td>72 41</td><td>0 4 27 1</td><td>100.0% 94.7% 60.3% 88.9%</td><td>25.2 185.3 401.9 793.3</td><td>1 0 0 0</td><td>0 0 0 0</td><td>84 72 41 8</td></tr><tr><td>bcv4 bcv6</td><td>34688</td><td>10</td><td>9 83</td><td>3 3 4 5</td><td>66 58 49</td><td>8 66 58 34</td><td>100.0% 100.0% 15 69.4%</td><td></td><td>10.0 57.8 169.3</td><td></td><td>66 58</td></tr><tr><td>F-MNIST</td><td>bcv7 4504</td><td>100</td><td>86</td><td>6 3 4 5</td><td>45 68 60 54 47</td><td>18 68 54 32 4</td><td>27 0 6 22</td><td>40.0% 100.0% 90.0% 59.3%</td><td>329.1 16.6 134.7 304.8</td><td>0 0 0 0 0</td><td>0 0 0 0 0</td><td>34 18 68 54 32</td></tr><tr><td>CIFAR-10</td><td>bcv5</td><td>4852</td><td>10</td><td>6 9 3</td><td>5</td><td>4</td><td>43 1</td><td>8.5% 80.0%</td><td>602.3 302.0</td><td>0 0</td><td>0 0</td><td>4 4</td></tr></table>

Note. We use a timeout of 3600 seconds for bcv4 and 1800 seconds for all other networks. Img is the number of evaluated images, and CC is the number of correctly classified images. Cand. denotes the number of verification candidate instances at the target budget ϵ, after excluding images for which an adversarial example within the budget is already known from σ-zero, FOCUSBRANCH at smaller budgets, or any prior verifier. R is the number of verified images. $l _ { s } < \epsilon$ ϵ counts verified images whose starting level is smaller than ϵ, while $l _ { s } = \epsilon$ counts verified images whose starting level equals ϵ. $\mathbf { B R A N C H } ^ { \mathsf { c o n } }$ counts, among the $l _ { s } < \epsilon \mathrm { c a s e s } .$ , how many use the concrete branching policy.

## H.2 Qualitative Examples

Figure 6 shows examples where FOCUSBRANCH verifies a correctly classified image at budget ϵ, while an adversarial example exists at ϵ + 1. Adversarial examples are found by the MILP fallback (pixel-wise) and checked by a forward pass.

![](images/2afa98077d81e90a16b72db49334aef2309ddedec6356299421114a2962e5d9d.jpg)  
Figure 6: Qualitative examples. Each pair shows a correctly classified image that FOCUSBRANCH verifies at budget ϵ (left), and an adversarial example at budget ϵ + 1 with the perturbed pixels marked in red (right).

![](images/67bc1579a9e6bf68a0dcb17a918ade9cd317b08f380654dc64e511cf554e1480.jpg)

![](images/bc86258d20052798f701d0e64241a3f5a866ab29b399bb1dc396f0568bff2828.jpg)

![](images/e202cb83f840260088c9a52079bc6753202bad7021053ed0054bd17f56af6a77.jpg)  
Figure 7: Comparison between FOCUSBRANCH, COVERD and COVERD+top-t-GP at ϵ = 3.

## H.3 Detailed Comparison with Baselines

Detailed results of comparison with baselines are shown as cactus plots at ϵ = 3 (Figure 7), 4 (Figure 8) and 6 (Figure 9).

Table 3 shows the sample-wise comparison with COVERD+top-t-GP on bcv4 and bcv5, which are excluded from the aggregated cactus plots. Note that we failed to run COVERD+top-t-GP on bcv5 using the same number of 50 MILP instances as in [42]. Specifically, they ran experiments on a dual AMD EPYC 7713 server with 2TB RAM and eight A100 GPUs. Thus we borrow the statistics from their results, aggregate across seeds, translate verification time from minute-base to second-base, and manually timed out those verified but above 1800s. However, the bcv5 statistics borrowed are not directly comparable with FOCUSBRANCH’s bcv5 statistics, since the two ran on different hardware.

![](images/dcfc4e8d9e7af7040871f4a0cf48f9137b075cc7f6a6d95bacb5d06038a76beb.jpg)

![](images/10b68910be889e1ab22c6b0ea1d3baddbb37504ee1e45c52a57546f8d05322ec.jpg)

![](images/78262abbe9e80d2397e1914ca421f3d69946295adf27f0b6243558b4ee432b4a.jpg)  
Figure 8: Comparison between FOCUSBRANCH, COVERD and COVERD+top- $\cdot t \mathrm { - G P }$ at $\epsilon = 4$

![](images/c3b7dfa91c5f0544e1445f5f20d97d892c87c37f6b741df9dbdda431caa9cc60.jpg)

![](images/38db7e4ab8fad6bbe2ef96c85c4225e9bac78bc296eb30dd38afea281406241b.jpg)

![](images/7638564f7016b5625845710024a199c5885f20cc96bbc99efc0812c1dd888243.jpg)  
Figure 9: Comparison between FOCUSBRANCH, COVERD and COVERD+t $\mathsf { o p } { - } t { - } \mathsf { G P }$ at $\epsilon = 6$

## H.4 Comparison with MILP

We also compared with a pure MILP solution by formulating the pixel-wise $\ell _ { 0 }$ constraint together with the method in [50], using Gurobi [13] as the solver. Note that MILP failed to verify any convolutional instances at $\epsilon \in \{ 3 , 4 , 5 , 6 \}$ , and thus in Table 4, the MILP lines only reflect their results on fully connected networks.

Table 4 shows that FOCUSBRANCH verifies more fully connected instances than MILP while also being faster on average; moreover, unlike MILP, it remains effective on CNNs, substantially increasing the total number of verified instances across architectures.

## H.5 Comparison with auto\_LiRPA

We do not include auto\_LiRPA [67] in this comparison. In our experiments, auto\_LiRPA reports Verified on some instances for which adversarial examples within the target budget are known, as found by MILP, FOCUSBRANCH, COVERD, COVERD+top-t-GP and σ-zero. Therefore, its reported verification results are not directly reliable for this benchmark setting. This unreliability stems from the implementation, not from the underlying method, which is sound. With α-CROWN disabled, the method is exactly the level-0 special case of FOCUSBRANCH (Section 3.4), whose bound is valid by Lemma E.2, and enabling $\alpha { \mathrm { - } } C { \mathrm { R O W N } }$ only selects among valid ReLU relaxations.

In particular, auto\_LiRPA’s $\ell _ { 0 }$ concretization assumes every input entry lies in [0, 1], regardless of the input bounds it is given,<sup>2</sup> whereas our networks can take standardized inputs.

## H.6 Additional Ablation Study

Instead of showing the aggregated view, we study the sample-wise discrepancy as shown in Figure 10.

We also performed ablation study for all fully connected instances at $\epsilon \in \{ 5 , 6 \}$ as shown in Figure 11.

## H.7 Effect of Parallelization

Both COVERD+top-t-GP and FOCUSBRANCH parallelize (on several GPU workers), and batching is a routine GPU practice to increase the level of parallelization, on which we do not claim a contribution, as is disclosed in Section 4.3 precisely for transparency.

Table 3: Sample-wise comparison between FOCUSBRANCH and COVERD+top-t-GP on bcv4 and bcv5. Those incorrectly classified images and images containing adversarial examples within the perturbation budget are marked “-” in the table. The timeout for bcv4 is 3600s and for bcv5 is 1800s. <sup>∗</sup>As we failed to run COVERD+top-t-GP on bcv5 with 50 MILP instances on our machine, this row is taken from the results table in [42] directly, and thus it is not directly comparable.
<table><tr><td>Dataset</td><td>Network</td><td>Method</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td rowspan="2">MNIST</td><td rowspan="2">bcv4</td><td>CoVERD+top-t-GP</td><td>1286.4</td><td>1008.5</td><td>T/0</td><td>1002.5</td><td>1397.5</td><td>T/0</td><td>1248.2</td><td>905.5</td><td></td><td>784.9</td></tr><tr><td>FOCUSBRANCH</td><td>554.2</td><td>448.3</td><td>T/0</td><td>367.6</td><td>611.0</td><td>3145.6</td><td>557.1</td><td>363.3</td><td>-</td><td>299.0</td></tr><tr><td rowspan="2">CIFAR-10</td><td>bcv5</td><td>CoVERD+top-t-GP*</td><td>1608.0</td><td>1540.0</td><td>1698.0</td><td>-</td><td>=</td><td>T/0</td><td>e</td><td>T/0</td><td>=</td><td>=</td></tr><tr><td></td><td>FOCUSBRANCH</td><td>132.3</td><td>133.2</td><td>141.6</td><td>1</td><td>=</td><td>801.1</td><td>=</td><td>T/0</td><td>-</td><td>=</td></tr></table>

Table 4: Comparison between FOCUSBRANCH and an MILP solution that aggregates $\epsilon \in \{ 3 , 4 , 5 , 6 \}$ FOCUSBRANCH <sub>FC</sub> denotes our results aggregating across all fully connected networks while FO-CUSBRANCH denotes our results aggregating across architectures.
<table><tr><td>Method</td><td>#Verified</td><td>Average Time (s)</td></tr><tr><td>MILP</td><td>1245</td><td>58.7</td></tr><tr><td>FOCUSBRANCH FC</td><td>1599</td><td>44.6</td></tr><tr><td>FOCUSBRANCH all</td><td>2799</td><td>67.7</td></tr></table>

![](images/9c7c660ce3674323065f0b55f406f264bc7f3ec3f5c409c327963c3c7d295fb2.jpg)  
Figure 10: Sample-wise ablation study on afc5 at $\epsilon = 5 .$

To isolate the effect of parallelization from that of the algorithm, we rerun the experiment at $\epsilon \in \{ 5 , 6 \}$ for FOCUSBRANCH and COVERD+top-t-GP (except for bcv4 and bcv5 we only run $\epsilon = 3 )$ , with all parallelization disabled (batch-size, workers are set to 1), under the same timeout and hardware with only one GPU. FOCUSBRANCH also uses a super-node pool of size 1. As our sample sizes are set per GPU worker (Appendix G), FOCUSBRANCH draws 4× fewer super-node samples, and we also reduce its concrete-node samples per level to 400 for fully connected networks and 100 for CNNs (from 20000 and 2000), making its planning less reliable. COVERD+top-t-GP would not be affected because it shares the workload with all GPU workers.

As shown in Table 5, turning off parallelization greatly slows down the speed for both methods, as implied by the decreased number of verified instances, which is expected. Without parallelization, both tools verify fewer instances, FOCUSBRANCH more so as it also loses batching, yet FOCUSBRANCH still verifies 432 instances against 352 and is 2.9× faster on those both verify. This advantage comes from the fully connected networks (292 vs. 197 verified, 5.5×), where almost all verified instances start below level ϵ, so one safe node with small level covers many ϵ-level nodes, whereas COVERD+top-t-GP always starts at the last (ϵ-)level and must bound significantly more nodes. On afc1–afc4, FOCUSBRANCH without parallelization even verifies more instances than COVERD+topt-GP with parallelization (268 vs. 201, Table 1). On CNNs, almost all verified instances start at level ϵ, where COVERD+top-t-GP is a special case of FOCUSBRANCH (Section 4.3), and the two tools are close (140 vs. 155 verified, 1.1×); the speedup of FOCUSBRANCH on CNNs with parallelization (11.5× on the same rows) thus comes mainly from batching.

![](images/5845cf7f34ae396dedc9443feca61c957d59044a1ab1b453df9401d41c2d8b06.jpg)

![](images/9448badfb8b42341dbb6004f0652ffe8b310f3963772864ebf6351cc723c0cad.jpg)  
Figure 11: Ablation study on fully connected instances at $\epsilon \in \{ 5 , 6 \}$

Table 5: Main results without parallelization on a single GPU at $\epsilon \in \{ 5 , 6 \}$ , except for bcv4 and bcv5 at ϵ = 3. Neurons is the number of neurons in the network, Img is the number of evaluated images, and CC is the number of correctly classified images. Cand. denotes the number of verification candidate instances at the target budget ϵ, after excluding images for which an adversarial example within the budget is already known (from any run, including the 4-GPU main results, so Cand. is the same as there). R is the number of verified images, R/Cand is the verified rate, and Time reports the mean verification time (in seconds) over each tool’s verified instances. Speedup is how many times faster FOCUSBRANCH is than COVERD+top-t-GP (geometric mean over the instances verified by both tools).

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Network</td><td rowspan="2">Neurons</td><td rowspan="2">Img</td><td rowspan="2">CC</td><td rowspan="2">€</td><td rowspan="2">Cand.</td><td colspan="3">FOCUSBRANCH</td><td colspan="3">CoVERD+top-t-GP</td><td rowspan="2">Speedup↑</td></tr><tr><td>R↑</td><td>R/Cand.↑</td><td>Time (s)↓</td><td>R↑</td><td>R/Cand.↑</td><td>Time (s)↓</td></tr><tr><td rowspan="9">MNIST</td><td>afcl</td><td>512</td><td>100</td><td>100</td><td>5 6</td><td>57 46</td><td>38 20</td><td>66.7% 43.5%</td><td>275.0 136.8</td><td>26 10</td><td>45.6% 21.7%</td><td>116.4 378.0</td><td>4.7× 5.1×</td></tr><tr><td>afc2</td><td>1024</td><td>100</td><td>100</td><td>5 6</td><td>79 68</td><td>37 20</td><td>46.8% 29.4%</td><td>124.2 211.8</td><td>28 13</td><td>35.4% 19.1%</td><td>264.9 649.0</td><td>5.7× 16.7×</td></tr><tr><td>afc3</td><td>1536</td><td>100</td><td>100</td><td>5</td><td>74</td><td>39</td><td>52.7% 28.8%</td><td>103.5 124.8</td><td>32 11</td><td>43.2% 16.7%</td><td>475.8 131.6</td><td>5.4× 1.3×</td></tr><tr><td>afc4</td><td>384</td><td>100</td><td>99</td><td>6 5</td><td>66 73</td><td>19 61</td><td>83.6%</td><td>119.3</td><td>35</td><td>47.9%</td><td>56.3</td><td>6.3×</td></tr><tr><td>afc5</td><td>1000</td><td>100</td><td>100</td><td>6 5</td><td>61 70</td><td>34 15</td><td>55.7% 21.4%</td><td>299.7 458.0</td><td>19 18</td><td>31.1% 25.7%</td><td>124.7 1033.5</td><td>6.7× 5.3×</td></tr><tr><td>bfcl</td><td>1000</td><td>100</td><td>99</td><td>6 5</td><td>60 68</td><td>3 4</td><td>5.0% 5.9% 3.4%</td><td>313.5 602.0</td><td>1 4</td><td>1.7% 5.9%</td><td>&lt;0.1 1206.0</td><td>3.2× 2.5×</td></tr><tr><td>bcv1</td><td>3604</td><td>100</td><td>98</td><td>6 5</td><td>59 73</td><td>2 29 6</td><td>39.7% 9.4%</td><td>1093.5 854.3 1251.9</td><td>0 32</td><td>0.0% 43.8% 9.4%</td><td>一 766.1 1016.2</td><td>一 1.0× 0.8×</td></tr><tr><td>bcv2</td><td>3604</td><td>100</td><td>94</td><td>6 5 6</td><td>64 67 62</td><td>44 17</td><td>65.7% 27.4%</td><td>493.6 1140.6</td><td>6 56 22</td><td>83.6% 35.5%</td><td>686.0 1070.8</td><td>1.2× 0.8×</td></tr><tr><td>bcv3</td><td>4504</td><td>100</td><td>99</td><td>5 6</td><td>76 68</td><td>25 3</td><td>32.9% 4.4%</td><td>873.3 1149.0</td><td>17 3</td><td>22.4% 4.4%</td><td>1004.5 1729.9</td><td>1.9× 1.5×</td></tr><tr><td rowspan="2"></td><td>bcv4</td><td>34688</td><td>10</td><td>9</td><td>3</td><td>9</td><td>7</td><td>77.8%</td><td>1446.4</td><td>2</td><td>22.2%</td><td>3245.1</td><td>4.1×</td></tr><tr><td>bcv6</td><td>3604</td><td>100</td><td>83</td><td>5</td><td>49</td><td>7</td><td>14.3%</td><td>1098.5</td><td>14</td><td>28.6%</td><td>1196.7</td><td>1.0×</td></tr><tr><td rowspan="2">F-MNIST CIFAR-10</td><td></td><td></td><td></td><td></td><td>6 5</td><td>45 54</td><td>1 1</td><td>2.2% 1.9%</td><td>1199.2 722.7</td><td>2 1</td><td>4.4% 1.9%</td><td>1246.6 1013.1</td><td>0.8× 1.4×</td></tr><tr><td>bcv7 bcv5</td><td>4504</td><td>100</td><td>86 9</td><td>6 3</td><td>47 5</td><td>0 0</td><td>0.0% 0.0%</td><td>一</td><td>0 0</td><td>0.0% 0.0%</td><td>一</td><td>一</td></tr></table>

The 32 instances that only COVERD+top-t-GP verifies are borderline, even though FOCUSBRANCH verifies each of them within 2 minutes when parallelization is enabled. The outcome can change from run to run (holds for both tools): on afc5 at $\epsilon = 5 ,$ , a repeated run with the same settings verified 18 instead of 15 instances for FOCUSBRANCH, and 12 instead of 18 for COVERD+top-t-GP, reversing the comparison. We attribute this variance to timing noise in the planner. Repeated runs of FOCUSBRANCH almost always choose the same starting level and the same CVD for the initial node, and bound nodes at about the same speed, so the difference most likely lies in the rest of the super branching strategy. Both tools plan it from sampled bounding times, and when several focus-set sizes are almost equally fast, timing noise decides between them. The fits in Appendix F.2, which these runs predate, make the plan of FOCUSBRANCH deterministic (Appendix H.10).

## H.8 How Often Is MILP Invoked?

Bounding in FOCUSBRANCH is performed by incomplete bound propagation (LiRPA/CROWN-style) with the closed-form concretization over node-constrained $\ell _ { 0 } .$ -balls of Section 3.4 (Eq. (10) and Eq. (12)). MILP is used only as a fallback, on nodes that are simultaneously unbranchable (level-ϵ concrete) and unknown. Table 6 reports how often this fallback is triggered across the full benchmark.

Table 6: Share of verified instances that invoke the MILP solver at least once.
<table><tr><td>€</td><td>Verified</td><td>Instances invoking MILP</td><td>Share</td></tr><tr><td>3</td><td>952</td><td>57</td><td>6.0%</td></tr><tr><td>4</td><td>839</td><td>85</td><td>10.1%</td></tr><tr><td>5</td><td>641</td><td>54</td><td>8.4%</td></tr><tr><td>6</td><td>367</td><td>10</td><td>2.7%</td></tr><tr><td>All</td><td>2799</td><td>206</td><td>7.4%</td></tr></table>

MILP is invoked in 152/1599 (9.5%) of fully connected verified instances and 54/1200 (4.5%) of convolutional ones; the remaining 92.6% of verified instances never call Gurobi, and 12 of the 46 network × ϵ cells invoke it zero times. The speedups we report are therefore not the result of accelerating an MILP solver.

## H.9 Choice of MILP Solver

Like prior $\ell _ { 0 }$ verifiers [41, 42, 43], FOCUSBRANCH solves its fallback problems as MILPs [50] with Gurobi [13]. This is the standard choice for complete verification, and using the same solver as the baselines keeps the comparison fair. Since each fallback problem is posed over the input box of an unbranchable node, a complete verifier such as ${ \alpha , \beta { \mathrm { - } } { \mathrm { C R O W N } } }$ [59] could be used instead. To assess this alternative, we replaced Gurobi with ${ \alpha , \beta { \mathrm { - } } \mathrm { C R O W N } }$ at $\epsilon = 3$ on afc4, the smallest network in our benchmark, keeping all other settings unchanged. Using ${ \alpha , \beta { \mathrm { - } } { \mathrm { C R O W N } } }$ was on average 10× slower than using Gurobi. Our soundness and completeness results require only a sound (and complete) solver, so the fallback solver is interchangeable. Given this slowdown and the consistency with prior work, we keep Gurobi.

## H.10 Reproducibility Note

All experimental results in the main paper were generated using the raw empirical estimates, prior to introducing the regression fits in Appendix F.2. Since the planner in the original version is sensitive to timing noise, re-running those experiments may yield slightly different runtimes and occasionally flip the outcome on borderline instances near the timeout. With the smoothed fits enabled, the planner deterministically selects the exact same plan across repeated runs in our tests. The original version used to generate the reported numbers is tagged v0.9.0 in our code repository. Later versions include the fits.