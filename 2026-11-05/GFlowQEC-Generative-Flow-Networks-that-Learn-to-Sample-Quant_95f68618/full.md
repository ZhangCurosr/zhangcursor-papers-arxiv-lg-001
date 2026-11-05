# GFlowQEC: Generative Flow Networks that Learn to Sample Quantum LDPC Codes<sup>⋆</sup>

Nuno Batista<sup>1</sup>, Gabriel Falcão<sup>2,3</sup>, and Penousal Machado<sup>4</sup>

<sup>1</sup> Department of Informatics Engineering, University of Coimbra, Coimbra, Portugal Instituto de Telecomunicações, Department of Electrical and Computer Engineering, University of Coimbra, Coimbra, Portugal

3 New York University Abu Dhabi, Abu Dhabi, United Arab Emirates 4 University of Coimbra, CISUC/LASI – Centre for Informatics and Systems of the University of Coimbra, Department of Informatics Engineering

nuno.batista@uc.pt, gff9537@nyu.edu, machado@dei.uc.pt

Abstract. Choosing a quantum error-correcting code for a real device means scoring candidates with circuit-level simulations so slow that only a few hundred can be run. So every evaluation counts, and a search is judged by how many good codes it finds per paid evaluation. Quantum LDPC codes built from a few group elements protect many logical qubits, but good ones are rare in a huge space. We introduce GFlowQEC, to our knowledge the first generative flow network (GFlowNet) for quantum code design. It builds a code one group element at a time and learns to propose codes in proportion to a power of their score. We train it once on scores from earlier searches, freeze it, and let it propose lifted-product codes with 180 physical qubits, scored by their distances. Ranking ten of its draws per paid call, it finds more than twice as many new top-scoring codes (inequivalent to any it was trained on) as tuned parallel tempering, even when tempering starts from GFlowQEC’s training codes. It finds more than three and a half times as many as a supervised screen ranking ten random candidates per call. When both rank a hundred, the screen beats GFlowQEC, and does best ranking GFlowQEC’s draws. Soft Qlearning, which shares the GFlowNet’s optimal policy, does as well at one noise bias and better at the other, so the gain does not depend on the GFlowNet’s training loss. Trained proposers can make a budget of expensive evaluations go further when good codes are rare.

Keywords: Quantum error correction · Quantum LDPC codes · Generative flow networks · Code search

⋆ This work is funded by national funds through FCT – Foundation for Science and Technology, I.P., within the scope of the research unit UID/00326 – Centre for Informatics and Systems of the University of Coimbra, https://doi.org/10.54499/ UID/00326/2025. This work is funded by national funds through FCT – Fundação para a Ciência e a Tecnologia, I.P., and, when eligible, co-funded by EU funds under project/support UID/50008/2025 – Instituto de Telecomunicações, with DOI identifier https://doi.org/10.54499/UID/50008/2025.

## 1 Introduction

Quantum low-density parity-check (LDPC) codes can protect many logical qubits in a single block of physical qubits [25,6]. Many of them are specified by a short list of elements of a finite group, and since every list gives a valid code [23,31], designing one amounts to a search over lists. That search is hard for two reasons. The space is enormous: the [[300, 60, 14]] mitten code of Bhardwaj et al. [4] is a list of ten elements of a group with sixty elements, and lists of that length give up to about $3 \times 1 0 ^ { 8 }$ inequivalent codes. And every candidate has to be scored, either by computing its distance or by simulating how often it fails, which takes from about a second to a few minutes per code. A search therefore only ever sees a sliver of the space.

Our motivating setting is choosing a code for a particular device. There, each candidate is scored by a circuit-level simulation, with faulty measurements and a real decoder, which takes minutes to hours per code, so a search scores only hundreds [42,5,4]. Since the best code can also change with the device’s noise and connectivity [5,3], a cheaper score can only shortlist candidates for the expensive simulation, so a search should deliver many good candidates for its budget rather than a single best one [1]. Because methods cannot yet be compared fairly at that price, we compare them where scoring is cheap enough to run each method many times, but we keep the unit of cost that matters on a device: the paid call, that is, one call to the scoring routine (the oracle) for a code the run has not scored before.

Bayesian optimisation, reinforcement learning, filtered uniform sampling and large language models (LLMs) have already been used to search code families of this kind, and Table 1 compares these searches with ours. None of them, however, trains a model to sample codes in proportion to their quality.

Learning can turn a fixed budget into more good codes in two ways. A screen draws random candidates and pays only for the ones a learned model ranks highest, whereas a proposer learns to draw good candidates directly. When good codes are rare, the diference matters: a screen has to look at many random candidates for every one it keeps, while a proposer builds that rejection into its own draws. Generative flow networks (GFlowNets) [1,2] are proposers of this kind. A GFlowNet builds an object step by step and is trained so that, at its optimum, it produces each object with probability proportional to a reward, which means one model yields many distinct good objects where reward maximisation would converge on one [1]. To our knowledge, no GFlowNet has yet been used to design quantum error-correcting codes.

This paper makes two contributions.

– We introduce GFlowQEC, a GFlowNet whose policy fills in the list one group element per step and is trained with trajectory balance [41] to produce each list with probability proportional to a power of its score.

– We test this policy, as a proof of concept, on lifted-product codes with 180 physical qubits, where it is trained once on scores stored from earlier searches, frozen, and compared with tuned parallel tempering, a supervised screen and soft Q-learning at matched budgets of paid calls.

Table 1. Prior code searches compared with this paper. LP stands for lifted product and BB for bivariate bicycle codes, two-block codes built from two sums of elements of $Z _ { l } \times Z _ { m } ,$ the pairs of integers mod l and mod m (Bravyi et al. [5] allow only pairs with one entry zero).
<table><tr><td>Work</td><td>Family</td><td>Score</td><td>Returns</td><td>Matched Exact budget</td><td>dis- tances</td></tr><tr><td>Chengyu et al. [10], Bayesian optimisation</td><td>BB, any weight</td><td>Rate  $k / n$  error rate</td><td>and one code per weighting</td><td>r yes</td><td>no</td></tr><tr><td>Cruz-Benito et al. [11], LLM</td><td>BB</td><td> $k d ^ { 2 } / n$ </td><td>a catalogue</td><td>partly</td><td>some</td></tr><tr><td>Liu and Marquardt [24], LLM</td><td>LP</td><td>Rate k/n and an archive error rate</td><td></td><td>no</td><td>no</td></tr><tr><td>Bhardwaj et al. [4], filtered sampling</td><td>LP, non-abelian</td><td>Distance upper bounds</td><td>one code per group</td><td>no</td><td>some</td></tr><tr><td>Sella et al. [34], reinforcement learning</td><td>BB as defined by Bravyi et al. [5]</td><td>Decoder success rate</td><td>best code so far</td><td>no</td><td>no</td></tr><tr><td>Freire et al. [13], three searches</td><td>Hypergraph product</td><td>Erasure failure rate</td><td>one best code yes</td><td></td><td>not stated</td></tr><tr><td>This paper, GFlowNet policy</td><td>LP, non-abelian</td><td>Failure exponent, Eq. (7)</td><td>codes proposed and ranked by one frozen policy</td><td>yes</td><td>some</td></tr></table>

At 180 physical qubits the frozen policy draws ten candidates for each paid call and pays for the ones it ranks most probable. Counting only codes inequivalent to those it was trained on, it found more than twice as many top-scoring codes per paid call as tuned parallel tempering (tuned PT), 2.4 and 2.8 times as many at the two noise biases we test, where bit flips are ten times likelier than phase flips or the reverse. Started from the same stored codes, tuned PT did not close the gap, so the lead is not just access to the stored data. It also found more than three and a half times as many as a supervised screen built on the same network and trained on the same scores. Because only about one random code in a hundred is top-scoring here, a screen shown ten random codes per paid call rarely sees one. Shown a hundred random codes per paid call, the screen found more than the policy, and the largest count of all came from a screen that ranked the policy’s own draws. Soft Q-learning trained in the same way, which shares the GFlowNet’s optimal policy on our construction [38], did as well at one bias and better at the other, so the gain does not depend on the GFlowNet’s training loss.

A longer version with all settings and further results is available with the code and data at https://doi.org/10.5281/zenodo.23922222.

## 2 Related Work

Code search. Simple baselines are strong in this setting: Freire et al. [13], who compare methods at a matched budget under one oracle as we do, find on hypergraph-product codes that a random walk matches simulated annealing and reinforcement learning. Beyond Table 1, LLM agents search under circuitlevel noise [42] and over balanced-product codes [32]. Reinforcement learning has glued small codes into codes of up to 20 physical qubits with one logical qubit [36] and has found a code on nine physical qubits together with its encoder, with the noise bias as an input [29]. We do not reimplement the LLM searches [11,24,42,32], because their oracles and budgets difer from ours and their models may have seen published codes of these families during pretraining.

GFlowNets. On a tree, where every object has a single construction path, GFlowNet training is equivalent to entropy-regularised reinforcement learning such as soft Q-learning [38]. Conditional GFlowNets take a parameter of the reward as an input, such as the reward’s temperature in Logit-GFN [19] or the noise bias in GFlowQEC. Training a GFlowNet only on stored rewards, with no reward model and no new oracle call, is an established setting [8]. Like the plain baseline of that work, GFlowQEC trains directly on its stored codes, and it adds conditioning on the bias and a deployment that ranks its own draws. GFlowNets have also been applied to quantum circuits: FlowQ-Net [12] builds them gate by gate and lists quantum error-correcting code synthesis among its future applications, and QFlowNet [21] applies a GFlowNet to unitary synthesis.

## 3 Background

## 3.1 Quantum LDPC Codes

An error on a qubit is a bit flip (X), a phase flip (Z) or both (Y ). A Calderbank-Shor-Steane (CSS) code [7,35] is specified by two binary matrices $H _ { X }$ and $H _ { Z }$ each with one column per physical qubit, n in all. Each row is a check, that is, a parity measurement on the qubits where the row has a 1, and the pair of matrices defines a valid code exactly when

$$
H _ { X } H _ { Z } ^ { \mathsf { T } } = 0 \quad ( \mathrm { m o d \ 2 ) . }\tag{1}
$$

The number of logical qubits is

$$
k = n - \operatorname { r a n k } H _ { X } - \operatorname { r a n k } H _ { Z } ,\tag{2}
$$

with the ranks taken over $\mathbb { F } _ { 2 } .$ , the field with two elements (arithmetic mod 2). A pattern of phase flips $e _ { Z } \in \mathbb { F } _ { 2 } ^ { n }$ marks the qubits that are hit, and its weight |e<sub>Z</sub>| is the number of such qubits. The X checks detect the pattern when $H _ { X } e _ { Z } \neq 0$ and the $Z$ checks detect bit flips $e _ { X }$ in the same way. Throughout, the subscript $X$ or $Z$ of $e ,$ d and $p$ names the kind of $\mathrm { { f i p } , }$ so $e _ { Z } , d _ { Z }$ and $p _ { Z }$ all refer to phase flips.

Sums of rows of $H _ { Z }$ , read as phase flips, are the $Z$ stabilizers: they leave the code’s states unchanged and are not seen by the X checks $\left( \mathrm { E q . \ ( 1 ) } \right)$ . Every other undetected pattern changes the encoded information and is a logical operator, and the lightest logical operator of each kind defines the distance,

$$
\begin{array} { r l } & { d _ { Z } = \operatorname* { m i n } \{ \left| e _ { Z } \right| : H _ { X } e _ { Z } = 0 , \ e _ { Z } \notin \mathrm { r o w s p a c e } H _ { Z } \} , } \\ & { d _ { X } = \operatorname* { m i n } \{ \left| e _ { X } \right| : H _ { Z } e _ { X } = 0 , \ e _ { X } \notin \mathrm { r o w s p a c e } H _ { X } \} , } \\ & { \ d = \operatorname* { m i n } ( d _ { X } , d _ { Z } ) , } \end{array}\tag{3}
$$

where rowspace H is the set of sums of rows of H. A code with these parameters is written $[ [ n , k , d ] ]$ . Its Tanner graph has one vertex for each qubit and each check, and an edge wherever a check’s row has a 1. For a quantum LDPC code this graph is sparse [25].

## 3.2 Lifted-Product Codes over a Group

A code in our family is specified by a tuple x of ten elements of a group $G ,$ and every search method in this paper proposes such tuples. We use $G = Z _ { 6 } \times S _ { 3 } ,$ whose elements pair an element of $Z _ { 6 }$ , the integers mod 6 under addition, with one of the six orderings of three objects (the group $S _ { 3 } )$ , combined entry by entry, so $G$ has $| G | = 3 6$ elements. A group is abelian when the order of a product does not matter, and G is not. A sum such as $a = g _ { 1 } + g _ { 2 } + g _ { 3 }$ is a set of group elements. The $| G | \times | G |$ binary matrices $\lambda ( a )$ and $\rho ( a )$ , with rows and columns indexed by the elements of $G _ { \ l }$ , multiply by a on the left and on the right: column v of $\lambda ( a )$ marks the elements $g v _ { ; }$ , and column v of $\rho ( a )$ the elements vg, for every $g$ in a. The two difer only when $G$ is non-abelian. Finally, the sum $a ^ { * }$ replaces every element of a by its inverse, so that $\lambda ( a ^ { * } )$ and $\rho ( a ^ { * } )$ are the transposes of $\lambda ( a )$ and $\rho ( a )$

Our family extends the mitten codes of Bhardwaj et al. [4] (their Definition 4), which are defined over non-abelian groups because over abelian groups their construction has distance at most 6. Ten group elements and the identity 1 form four sums of three distinct elements,

$$
\begin{array} { c c } { { a _ { 1 } = g _ { 1 } + g _ { 2 } + g _ { 3 } , } } & { { a _ { 2 } = 1 + g _ { 4 } + g _ { 5 } , } } \\ { { } } & { { b _ { 1 } = h _ { 1 } + h _ { 2 } + h _ { 3 } , } } & { { b _ { 2 } = 1 + h _ { 4 } + h _ { 5 } . } } \end{array}\tag{4}
$$

A base matrix has sums as its entries and becomes binary when each entry is replaced by its $| G | \times | G |$ matrix. The mitten codes are the lifted product [31,30] of the row $\left( a _ { 1 } a _ { 2 } \right)$ and the column $( b _ { 1 } ^ { * } b _ { 2 } ^ { * } ) ^ { \mathsf { T } }$ , and in our convention the row enters

through λ and the column through $\rho \colon$

$$
\begin{array} { r l } & { H _ { X } = \left( \lambda ( a _ { 1 } ) \quad 0 \quad \lambda ( a _ { 2 } ) \quad 0 \quad \rho ( b _ { 1 } ^ { * } ) \right) , } \\ & { \qquad 0 \quad \lambda ( a _ { 1 } ) \quad 0 \quad \lambda ( a _ { 2 } ) \rho ( b _ { 2 } ^ { * } ) \ d t \ d t ^ { 2 } ) , } \\ & { H _ { Z } = \left( \rho ( b _ { 1 } ) \rho ( b _ { 2 } ) \quad 0 \quad \quad 0 \quad \lambda ( a _ { 1 } ^ { * } ) \right) , } \\ & { \qquad 0 \quad \quad 0 \quad \rho ( b _ { 1 } ) \rho ( b _ { 2 } ) \lambda ( a _ { 2 } ^ { * } ) \ d t \ d t ^ { 2 } , } \end{array}\tag{5}
$$

where 0 is the $| G | \times | G |$ zero matrix. The code has $n = 5 | G | = 1 8 0$ physical qubits, checks of weight 9 and, by Eq. $( 2 ) , k \geq | G |$ . Bhardwaj et al. also require $\lambda ( a _ { 2 } )$ and $\rho ( b _ { 2 } )$ to be invertible, which gives $k = | G |$ . Over $Z _ { 6 } \times S _ { 3 }$ only 123 of the 595 choices of $a _ { 2 }$ , and of $b _ { 2 } .$ , meet this requirement. We drop it, so our family is much larger and contains codes with $k > | G |$ . In this family $d _ { X }$ and $d _ { Z }$ can difer, so the best codes depend on the bias between bit and phase flips.

Once each sum is multiplied by a group element so that it contains the identity, we require the elements of $a _ { 1 }$ and $a _ { 2 }$ to generate $G ,$ that is, every element of G is a product of them and their inverses, and likewise the elements of $b _ { 1 }$ and $b _ { 2 } .$ This discards every code that splits into disconnected copies of a smaller code [23].

## 3.3 GFlowNets

A GFlowNet [1,2] builds an object one piece per step. A neural network with parameters $\theta ,$ called the policy, chooses each piece and is trained to produce each complete object with probability proportional to a positive reward:

$$
p _ { \theta } ( x ) = \frac { R ( x ) } { \sum _ { x ^ { \prime } } R ( x ^ { \prime } ) } ,\tag{6}
$$

where $x ^ { \prime }$ runs over all complete objects. When there are too many objects to list, this total reward cannot be computed, so it is learned as a scalar $Z _ { \theta }$ alongside the network.

## 4 Search Space and Oracle

## 4.1 Equivalent Codes and the Canonical Form

Two CSS codes are equivalent when relabelling the qubits and the checks turns the matrices of one into those of the other, and equivalent codes share $n ,$ , k, $d _ { X }$ and $d _ { Z }$ . Four operations on a tuple produce an equivalent tuple: reordering the elements within a sum, translating a sum (multiplying all its elements by one group element), renaming the elements of G by an automorphism (a oneto-one renaming that respects the multiplication table), and conjugating the g<sub>i</sub> alone or the $h _ { i }$ alone $( g _ { i } \to g g _ { i } g ^ { - 1 }$ for one $g )$ . The canonical form of a tuple undoes only reordering and translation. Two distinct canonical forms can still be equivalent codes, and a graph-isomorphism program decides equivalence exactly on the Tanner graphs. The mirror image of a code swaps $H _ { X }$ and $H _ { Z }$ , so that $d _ { X }$ and $d _ { Z }$ trade places, and we count a code and its mirror image as two codes.

## 4.2 Oracle and Reference Sample

The oracle returns the score $s ( x )$ of the code of a tuple x, with higher scores for better codes. Every search method gets its scores only from the oracle, which first replaces the tuple by its canonical form. A form already scored in the run is answered from memory at no cost, whereas a new form is a paid call, so the cost of a method is the number of paid calls it makes.

The score assumes that each qubit sufers a bit flip with probability $p _ { X }$ and, independently, a phase flip with probability $p _ { Z }$ . The noise bias $\gamma = \log _ { 1 0 } ( p _ { X } / p _ { Z } )$ says which flip is more likely, while the rarer flip is fixed at $1 0 ^ { - 3 }$ . A code of phase-flip distance $d _ { Z }$ corrects every pattern of fewer than $d _ { Z } / 2$ phase flips, so the lightest pattern that defeats it has probability about $p _ { Z } ^ { d _ { Z } / 2 }$ , and the same holds for bit flips with $d _ { X }$ and $p _ { X }$ . The score, which we call the failure exponent, counts the orders of magnitude by which the more likely of these two failures is suppressed:

$$
\begin{array} { c } { { s ( x , \gamma ) = \mathrm { m i n } ( c _ { X } d _ { X } , ~ c _ { Z } d _ { Z } ) , } } \\ { { { } } } \\ { { c _ { X } = \frac { 1 } { 2 } \log _ { 1 0 } { \cfrac { 1 } { p _ { X } } } , ~ c _ { Z } = \frac { 1 } { 2 } \log _ { 1 0 } { \cfrac { 1 } { p _ { Z } } } . } } \end{array}\tag{7}
$$

A score of 10, for example, means that the lightest uncorrectable error has probability about $1 0 ^ { - 1 0 }$ . We report results at $\gamma = \pm 1$ , where the likelier flip has probability $1 0 ^ { - 2 }$ and coeficient 1.0, and the rarer one has coeficient 1.5. We write $s ( x )$ when the bias is fixed.

The threshold for a top-scoring code is the highest score in the uniform reference sample, the 666 of 1,000 uniform draws whose elements generate $G ,$ each with exact distances: 10.0 at $\gamma = \pm 1$ , reached by 6 of the 666. A top-scoring code thus has $d _ { X } \geq 1 0$ and $d _ { Z } \geq 7 \ \mathrm { a t } \ \gamma = + 1 $ , and the reverse at $\gamma = - 1$ . At $\gamma = 0$ , by contrast, 116 of the 666 share the sample’s highest score, so this threshold would not separate methods. The best codes certified by search runs have $d _ { X } = d _ { Z } = 1 0$ , so they are [[180, k, 10]] codes with $k \geq 3 6$

Exact distances come from what we call the SAT ladder, an established use of a satisfiability solver [9]. For $w = 1 , 2 , . . .$ . in turn, the ladder asks whether a logical operator of weight at most w exists, and a question left undecided within its budget stops the ladder at a lower bound. Two such ladders, the exact ladders, give $d _ { X }$ and $d _ { Z }$ . They are expensive, so every new code first gets a cheap and slightly optimistic bound score. It uses $d _ { A }$ and $d _ { B }$ , the distances of the classical codes with check matrices $( \lambda ( a _ { 1 } ) \ | \ \lambda ( a _ { 2 } ) )$ and $( \rho ( b _ { 1 } ) ~ | ~ \rho ( b _ { 2 } ) )$ , computed by ladders of the same kind, the classical ladders. A lightest codeword of the first, placed on two of the five blocks of qubits, is a phase-flip pattern that no X check detects, so $d _ { Z } \leq d _ { A }$ unless it is a stabilizer, and likewise $d _ { X } \leq d _ { B }$ . The bound score is $\operatorname { E q } .$ (7) with $d _ { B }$ and $d _ { A }$ in place of $d _ { X }$ and $d _ { Z } . \mathrm { A t } \gamma = \pm 1$ , the same paid call also runs the exact ladders whenever the bound score reaches the threshold or a classical ladder is left undecided. A code on which they have run is certified, and it counts as top-scoring when the score they prove reaches the threshold. A ladder that stops early, as happens for 0.7 per cent of certified codes, or a classical bound that fails can leave a good code uncounted, but neither can make a bad code count. On one CPU core a bound score takes 0.8 s in the median and 1.5 s in the mean, whereas the exact ladders take about 10 s on a uniform draw and about 2 min on a certified code.

![](images/87c7e4465a8dd5781ffa0ba2825f3fe70e233878f756df71a7d3711ed708ebfc.jpg)  
Fig. 1. GFlowQEC $( n = 1 8 0 )$ . Rows of slots are tuples, with grey slots fixed to the identity, and Tanner graphs are codes. (a) Training, with no oracle call: (1) a stored form x with its score s and a bias $\gamma ,$ which give its reward $R ( x , \gamma )$ , (2) augmented and, with probability $1 / 2 .$ , mirrored by M with $\gamma \mapsto - \gamma$ , (3) read by the transformer (three blocks, causal mask shaded), which takes the bias as a condition z on its start token, (4) the trajectory-balance loss, weighing $R ( x , \gamma )$ against $Z _ { \theta } ( z ) p _ { \theta }$ , and its gradient step (dashed). (b) Deployment: (1) the frozen policy at $\gamma = \pm 1$ , (2) $\mu N$ drawn forms (Sec. 5.4), (3) the N with the highest log $p _ { \theta } ( x \mid z )$ , (4) the oracle’s bound score and SAT certification, (5) the top-scoring codes outside the stored forms.

## 5 GFlowQEC

GFlowQEC’s policy is a causal transformer [39] that builds the tuple $x$ one position, or slot, at a time, and its training target is

$$
p _ { \theta } ( x ) \propto R ( x ) = \operatorname* { m a x } ( s ( x ) , 0 . 0 1 ) ^ { \beta } ,\tag{8}
$$

where $p _ { \theta } ( x )$ is the probability that the policy builds $x ,$ the score $s ( x )$ is clamped at 0.01, and the exponent $\beta > 0$ sets how strongly the policy favours high scores. Fig. 1 shows its training and deployment.

## 5.1 Slot-by-Slot Construction

The tuple has $T = 1 0$ slots, which the policy fills in the order $g _ { 1 } , \ldots , g _ { 5 }$ , then $h _ { 1 } , \ldots , h _ { 5 }$ . We write $x _ { t }$ for the element in slot $t , x _ { < t }$ for the elements before it, and $P _ { F } ( x _ { t } \mid x _ { < t } ; \theta )$ for the policy’s probability of $x _ { t } .$ . Slots $g _ { 1 }$ and $h _ { 1 }$ are fixed to the identity. By translation this loses no code up to equivalence, and it puts the identity in every sum. To keep the elements of each sum distinct, a mask restricts every other slot t to its legal set, which is the group without the identity and without the elements already placed in the same sum,

$$
\mathcal { A } _ { t } ( x _ { < t } ) = G \setminus \left( \{ 1 \} \cup \{ x _ { t ^ { \prime } } : t ^ { \prime } < t , \ t ^ { \prime } \ \mathrm { i n ~ t h e ~ s a m e ~ s u m ~ a s } \ t \} \right) .\tag{9}
$$

A tuple is legal when $g _ { 1 } = h _ { 1 } = 1$ and every other element lies in its legal set, so there are $( ( | G | - 1 ) ( | G | - 2 ) ) ^ { 4 }$ legal tuples. A cheap test, the generation check, decides without an oracle call whether the elements generate G, and a tuple that fails it is never sent to the oracle.

## 5.2 Objective and the Choice of $\beta$

Because a partly filled tuple can be reached only by filling its slots in order, the graph of partial tuples, whose edges each add one element, is a tree. The probability of producing x is then the product of the slot probabilities,

$$
p _ { \theta } ( x ) = \prod _ { t = 1 } ^ { T } P _ { F } ( x _ { t } \mid x _ { < t } ; \theta ) .\tag{10}
$$

Trajectory balance [41] then requires $Z _ { \theta } p _ { \theta } ( x ) = R ( x )$ , and we minimise the squared gap between the logarithms of its two sides,

$$
\mathcal { L } ( \boldsymbol { x } ) = \Big ( \log Z _ { \theta } + \sum _ { t = 1 } ^ { T } \log P _ { F } ( \boldsymbol { x } _ { t } \mid \boldsymbol { x } _ { < t } ; \theta ) - \log R ( \boldsymbol { x } ) \Big ) ^ { 2 } ,\tag{11}
$$

with log the natural logarithm.

We set the exponent to $\beta = 8$ before training by a fixed rule: it is the smallest of 1, 2, 4, 8, 16, 27, 40 and 64 at which the best tenth of the uniform reference sample and its mirror images, ranked by clamped score, holds at least four fifths of their total reward.

## 5.3 Training Data

The policy trains only on scores that 42 earlier runs of several search methods had already paid for, so training itself makes no oracle call. Those runs made 126,000 paid calls covering 116,972 distinct canonical forms, the stored forms, and include the tuned PT runs behind seeds 0 to 2 of Table 2. Each form is stored with the two distances the oracle returned, which give its score at any bias (Eq. (7)): exact distances for the 6,770 certified codes, apart from 45 whose ladder stopped early at a lower bound, and the bounds $d _ { A }$ and $d _ { B }$ for the other 110,202.

During training, each drawn tuple is replaced by a random equivalent tuple x˜ with the same reward, obtained by shufling the two free elements of each sum and conjugating the $g _ { i }$ by one random group element and the $h _ { i }$ by another, never by translation. With probability one half, a training tuple is also replaced by its mirror $M ( x ) = ( h _ { 1 } ^ { - 1 } , \ldots , h _ { 5 } ^ { - 1 } , g _ { 1 } ^ { - 1 } , \ldots , g _ { 5 } ^ { - 1 } )$ , which builds a code equivalent to the mirror image of $x ,$ so it scores at −γ what x scores at $\gamma$ and trains at the opposite bias with the same reward. A single network serves both biases by taking the bias as an input, a condition vector $z ,$ and each bias has its own learned normaliser $Z _ { \theta } ( z )$ . We train one model per random seed, ten in all, at $\gamma = \pm 1$ , the biases at which they are deployed, and then freeze them. The transformer has three blocks of width 128 with four heads (605,863 parameters) and trains with Adam [20] for 315,280 updates of 256 tuples, about 690 passes over the stored forms.

## 5.4 Deployment

The frozen policy proposes codes at $\gamma = + 1 \mathrm { o r } \gamma = - 1$ . It draws until it holds $\mu N$ distinct canonical forms, where $N = 1 { , } 0 0 0$ is the budget of paid calls and $\mu = 1 0$ or 100 is an oversample factor, and discards without an oracle call every draw that fails the generation check or repeats a form. It then ranks the forms by log $p _ { \theta } ( x \mid z )$ , the log-probability of the tuple as drawn, which stands in for the score: at zero loss log $p _ { \theta } ( x \mid z ) = \beta \log \operatorname* { m a x } ( s ( x ) , 0 . 0 1 ) - \log Z _ { \theta } ( z )$ , so at a fixed bias it orders the stored tuples by their stored scores. It pays one oracle call for each of the N highest. Draws are not checked against the stored forms, so a form that is sent may already be stored, and it is still paid for. We chose this ranking after reading earlier runs, in which paying for the draws in the order drawn had trailed an earlier, smaller supervised screen per paid call, in every seed at both biases. Ranking found more than five and a half times as many top-scoring codes as paying in draw order.

## 6 Experimental Setup

## 6.1 Comparison Methods

Every method uses the same oracle, search space and budget, and runs are paired by random seed.

Tuned parallel tempering (tuned PT) [37,14,16] runs eight chains of local search at diferent temperatures, each move redrawing one slot uniformly, with Metropolis acceptance [27] and exchanges between the chains. It uses the better of two temperature ranges on its own runs, 0.25 to 2.0, a choice that favours it. A variant starts its chains at stored forms rather than at uniform draws, and those starting forms never count toward its results.

– The supervised screen uses the policy’s network as a regressor that predicts a code’s score. Unlike the policy, it sees all ten slots at once, and it also reads features that do not change when a sum is translated. It is trained by squared error to predict the scores of the same stored forms, with the policy’s augmentation, batches and seeds. Its variant and training length were tuned on a held-out tenth of the stored forms, a choice the policy did not get, after which it was trained on all of them. When deployed, it ranks ten or a hundred times N uniform random candidates and pays for its top N.

– Soft Q-learning [15] trained ofline uses the same stored scores, network, training steps and deployment rule as GFlowQEC. It shares the GFlowNet’s optimal policy (Sec. 2), so it tests whether the training objective matters.

– MAP-Elites [28], an evolutionary quality-diversity algorithm, keeps the best code found in each bin of a grid of cheap code properties, and we tuned it over two grids, five bins of $d _ { A } - d _ { B }$ alone or crossed with four bins of k.

Bayesian optimisation (BO) fits a Matérn $5 / 2$ Gaussian process to a onehot encoding of the tuple and, after 200 uniform draws, pays for the best of 2,000 uniform candidates by expected improvement. It is not the method of Chengyu et al. [10], who use a learned graph embedding and a diferent decoder, noise rate and code family.

Uniform sampling pays for uniform random tuples that pass the generation check.

We also ran PPO [33] at two entropy weights, MAP-Elites on the twenty-bin grid, and soft Q-learning and Munchausen DQN [40] trained during the search, and each found fewer top-scoring codes than tuned PT in every seed at both biases.

## 6.2 Measures

The main measure is the number of certified top-scoring codes in a run’s first 1,000 paid calls. A paid code is novel when its canonical form is not among the stored forms. For a method trained on or started from the stored forms, only novel codes count, whereas the other methods never saw the stored forms, so all their codes count. Because a novel code can still be equivalent to a stored code or its mirror image, we also give the inequivalent count, which drops such codes and, for every method, counts each equivalence class once.

To compare two methods we use their pooled ratio, which divides their counts, each summed over the seeds both methods ran, with the 95 per cent log-ratio interval of Katz et al. [18]. For counts $\kappa _ { 1 }$ and $\kappa _ { 2 }$ out of $\nu _ { 1 }$ and $\nu _ { 2 }$ paid calls, the interval is the ratio times ${ \bf \Phi } _ { e ^ { \pm 1 . 9 6 \omega } }$ , with $\omega ^ { 2 } = 1 / \kappa _ { 1 } - 1 / \nu _ { 1 } + 1 / \kappa _ { 2 } - 1 / \nu _ { 2 }$ We say that the two are not separated when the interval contains 1. It treats every paid call as independent, which leaves out the variation between seeds and within a search chain, so we also give the number of seeds in which one method leads.

## 7 Results

Table 2 gives the counts, with GFlowQEC and the supervised screen at $\mu = 1 0$ In every paired seed and at both biases, GFlowQEC (below, the policy) found more top-scoring codes than any other method in the table except soft Q-learning trained ofline. Each pair of numbers below is at $\gamma = + 1$ and $\gamma = - 1$ , in that order.

Table 2. Results at 180 physical qubits: certified top-scoring codes in each run’s first 1,000 paid calls, mean over ten seeds rounded to whole codes, at noise bias $\gamma = + 1$ and $\gamma = - 1$ . In the first group of rows only novel codes count. Inequivalent: Sec. 6.2. Ratios in the text pool the seeds both methods ran.
<table><tr><td rowspan="3">Method</td><td>Certified top-scoring codes</td><td>Inequivalent</td></tr><tr><td> $\gamma = + 1$  γ = −1</td><td>γ = +1</td></tr><tr><td></td><td>γ = −1</td></tr><tr><td>Trained on, or started from, the stored forms</td></tr><tr><td>GFlowQEC 313</td><td>317 221 222 59</td></tr><tr><td>Supervised screen</td><td>87 84 62</td></tr><tr><td>Soft Q-learning, trained offline</td><td>317 364 225</td></tr><tr><td>Tuned PT from the stored forms 119</td><td>123 79</td></tr><tr><td>Searching without stored data</td><td></td></tr><tr><td>Tuned PT 97</td><td>84 92 79</td></tr><tr><td>MAP-Elites, five bins</td><td>103 92 88</td></tr><tr><td>Uniform sampling 12</td><td>8 12 8 52</td></tr><tr><td>BO, one-hot encoding 50</td><td>53 49</td></tr></table>

Against tuned PT. On inequivalent codes, tuned PT is the strongest method without the stored data, although MAP-Elites with five bins found slightly more before equivalent codes were merged. The policy found 2.4 [2.2, 2.6] and 2.8 [2.6, 3.0] times as many inequivalent top-scoring codes as tuned PT, and more than twice as many in 19 of its 20 runs. Starting tuned PT from the stored forms does not close the gap, since on the same count the policy still found 2.8 and 2.7 times as many.

Against soft Q-learning. Soft Q-learning trained ofline found 1.01 and 1.15 times the policy’s novel count, not separated at $\gamma = + 1$ and ahead at $\gamma = - 1$ , where the interval [1.10, 1.19] excludes 1. It led in 7 and 8 of the ten seeds.

Against the supervised screen. At $\mu = 1 0$ the policy found 3.6 and 3.8 times as many top-scoring codes as the screen, and led in each of ten seeds. When both oversampled at $\mu = 1 0 0 $ , the screen found more in each of ten seeds, and the policy found 0.45 [0.44, 0.47] and 0.62 [0.61, 0.64] times its count. Still, two controls show that the policy’s strength lies in its draws. Ranking the screen’s uniform candidates at $\mu = 1 0$ instead of its own draws, the policy found a fifth as many codes as from its own draws, and fewer than the screen. Ranking the policy’s draws instead of uniform candidates at $\mu = 1 0 0$ , the screen found 1.27 and 1.30 times its own count, more in every seed, so better draws and better ranking combine.

Origin of the codes. At $\gamma = + 1$ the cap $d _ { X } \leq d _ { B }$ depends only on the $h _ { i }$ so we count two top-scoring codes, novel or not, as one neighbourhood when their $h _ { i }$ agree up to conjugating every element by one group element, and at $\gamma = - 1$ the $g _ { i }$ play this role. In each of the policy’s runs, 87 to 93 per cent of its neighbourhoods also hold a stored top-scoring code, so the policy mostly finds new partners for good halves that the stored data already held.

Cost. The policy’s paid calls ran the exact ladders 3.5 and 4.2 times as often as tuned PT’s, and these ladders dominate the cost of a paid call here. On seeds 0 to 2, per CPU second of exact ladders, the policy and tuned PT found topscoring codes at the same rate, within five per cent at both biases, and on the inequivalent counts tuned PT was ahead. On seeds 3 to 9 the two ran on diferent machines, so we do not compare their CPU seconds.

## 8 Discussion

A trained proposer is most likely to pay of when four conditions hold together: good codes are too rare to find by sampling, no quantity that is cheap to compute from the parity checks ranks them, each call to the oracle is expensive, and the cost of training is shared across many searches. At 180 physical qubits, good codes are not rare enough, which is likely why a screen shown a hundred random codes per paid call overtakes the policy. But the number of logical qubits k, the obvious cheap quantity, does not single out the top-scoring codes: sorting the stored scores by k would raise their top-scoring share by at most 1.4 times. Of the four conditions, only the cost of a call is known to grow, both with the size of the code and under circuit-level noise. And searches already run at 600 to 1,500 physical qubits [24], past the few hundred physical qubits where exhaustive catalogues stop [22].

## Limitations.

The gain is in how many good codes a budget buys, not in the best code found. We show no code better than those found by Bayesian optimisation, reinforcement learning or LLM-driven searches. Nor does a good code at this size need a trained policy, since uniform sampling found about ten topscoring codes per 1,000 paid calls (Table 2).

– The lead is counted per paid call, with nothing charged for the stored data (the 126,000 paid calls of Sec. 5.3, against 1,000 per run) or for the policy’s training, and it disappears per CPU second of exact ladders (Sec. 7).

The score is computed from distances alone (Eq. (7)), with no decoder. We have not measured whether circuit-level noise, with faulty check measurements and a real decoder, reorders the codes.

– We use the policy as a proposer that ranks its own draws and have not tested whether it samples in proportion to its reward.

## 9 Conclusion

Trained once on stored scores and used frozen at 180 physical qubits, GFlowQEC found more than twice as many top-scoring codes per paid call as tuned PT, counting only codes inequivalent to its training data. The gain lies in the candidates, not the training loss, since soft Q-learning did as well or better and a screen did best when it ranked the policy’s draws.

Future work. The next step toward choosing a code for a device is to re-score the top-scoring codes at 180 physical qubits under circuit-level noise and see whether their ranking survives. The proposer would then be graded at that cost, against the method of Chengyu et al. [10] at the budget that method is built for, a few hundred calls to an expensive oracle. A policy conditioned on the measured noise through multi-objective GFlowNets [17] could propose a new set of good codes after each calibration, without running a new search.

The setting we aim for is a quantum compiler that chooses the code itself. A strategic plan for neutral-atom quantum computing [26] calls for the compiler to select codes from the application and the hardware specification, and names generative neural networks among the methods that could find codes suited to a device’s noise and operational constraints. GFlowQEC fits this role as a conditional generator, in which one policy would answer every request. The hard constraints of a request, such as the device’s connectivity, the check weight or a floor on k, would be built into the mask of Eq. (9), and its noise model would enter through the condition z.

A single policy could also search across code families if each action chose one level of a construction in turn: the family, the group, the base matrices and the sums that fill them. Liu and Marquardt [24] let a language model mutate a three-level grammar of this kind, whereas ours would be one trained policy choosing every level.

Acknowledgments. The authors proposed most of the ideas, designed and directed the experiments and carried out the literature review. The large language models Claude Fable 5.1, Claude Opus 5.5 and Claude Sonnet 5.5 (Anthropic) wrote almost all of the code and helped summarise the literature, analyse the results, and write and edit the text, always under the authors’ guidance. The authors reviewed the text and take full responsibility for its content.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Bengio, E., Jain, M., Korablyov, M., Precup, D., Bengio, Y.: Flow network based generative models for non-iterative diverse candidate generation. In: NeurIPS. vol. 34, pp. 27381–27394 (2021)

2. Bengio, Y., Lahlou, S., Deleu, T., Hu, E.J., Tiwari, M., Bengio, E.: GFlowNet foundations. Journal of Machine Learning Research 24(210), 1–55 (2023)

3. Berthusen, N., Devulapalli, D., Schoute, E., Childs, A.M., Gullans, M.J., Gorshkov, A.V., Gottesman, D.: Toward a 2D local implementation of quantum low-density parity-check codes. PRX Quantum 6, 010306 (2025). doi:10.1103/PRXQuantum.6. 010306

4. Bhardwaj, A., Ma, M., Meister, N., King, R., Bluvstein, D., Preskill, J., Cain, M., Xu, Q., Huang, H.Y.: High-rate qLDPC processors. arXiv:2607.28795 (2026)

5. Bravyi, S., Cross, A.W., Gambetta, J.M., Maslov, D., Rall, P., Yoder, T.J.: Highthreshold and low-overhead fault-tolerant quantum memory. Nature 627(8005), 778–782 (2024). doi:10.1038/s41586-024-07107-7

6. Breuckmann, N.P., Eberhardt, J.N.: Quantum low-density parity-check codes. PRX Quantum 2(4), 040101 (2021). doi:10.1103/PRXQuantum.2.040101

7. Calderbank, A.R., Shor, P.W.: Good quantum error-correcting codes exist. Phys. Rev. A 54(2), 1098–1105 (1996). doi:10.1103/PhysRevA.54.1098

8. Chen, R., Wang, X., Hu, R., Li, Z., Huang, L.: Beyond the proxy: Trajectorydistilled guidance for ofline GFlowNet training. In: ICML (2026)

9. Chen, Y.F., Jafari, S.M.R., Lai, C.Y.: SAT, MaxSAT, and SMT for QLDPC distance computation: A large-scale empirical study. arXiv:2606.12445 (2026)

10. Chengyu, Y., Meister, R., Carty, C., Lin, S.K., Bondesan, R.: Bayesian optimization for quantum error-correcting code discovery. arXiv:2601.18562 (2026)

11. Cruz-Benito, J., Cross, A.W., Kremer, D., Faro, I.: Evolutionary discovery of bivariate bicycle codes with LLM-guided search. arXiv:2606.02418 (2026)

12. Dai, J., Rizvi-Martel, M., Rabusseau, G.: FlowQ-Net: A generative framework for automated quantum circuit design. arXiv:2510.26688 (2025)

13. Freire, B.C.A., Delfosse, N., Leverrier, A.: Optimizing hypergraph product codes with random walks, simulated annealing and reinforcement learning. arXiv:2501.09622 (2025), IEEE ISIT

14. Geyer, C.J.: Markov chain Monte Carlo maximum likelihood. In: Computing Science and Statistics: 23rd Symposium on the Interface. pp. 156–163 (1991)

15. Haarnoja, T., Tang, H., Abbeel, P., Levine, S.: Reinforcement learning with deep energy-based policies. In: ICML (2017)

16. Hukushima, K., Nemoto, K.: Exchange Monte Carlo method and application to spin glass simulations. J. Phys. Soc. Jpn. 65, 1604 (1996). doi:10.1143/JPSJ.65. 1604

17. Jain, M., Raparthy, S.C., Hernandez-Garcia, A., Rector-Brooks, J., Bengio, Y., Miret, S., Bengio, E.: Multi-objective GFlowNets. In: ICML (2023)

18. Katz, D., Baptista, J., Azen, S.P., Pike, M.C.: Obtaining confidence intervals for the risk ratio in cohort studies. Biometrics 34, 469–474 (1978)

19. Kim, M., Ko, J., Yun, T., Zhang, D., Pan, L., Kim, W., Park, J., Bengio, E., Bengio, Y.: Learning to scale logits for temperature-conditional GFlowNets. In: ICML (2024)

20. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: ICLR (2015)

21. Koo, I., Cha, H., Lee, J.: QFlowNet: Fast, diverse, and eficient unitary synthesis with generative flow networks. In: IEEE QCNC (2026)

22. Liang, Z., Liu, K., Song, H., Chen, Y.A.: Generalized toric codes on twisted tori for quantum error correction. PRX Quantum 6, 020357 (2025). doi:10.1103/ rmy6-9n89

23. Lin, H.K., Pryadko, L.P.: Quantum two-block group algebra codes. Phys. Rev. A 109(2), 022407 (2024). doi:10.1103/PhysRevA.109.022407

24. Liu, Z., Marquardt, F.: Large-language-model discovery of quantum LDPC codes through structured concept evolution. arXiv:2606.24808 (2026)

25. MacKay, D.J.C., Mitchison, G., McFadden, P.L.: Sparse-graph codes for quantum error correction. IEEE Trans. Inf. Theory 50(10), 2315–2330 (2004). doi:10.1109/ TIT.2004.834737

26. Menssen, A., Wang, T.T., Gullans, M., et al.: Strategic plan for neutral atom quantum computation. arXiv:2607.21554 (2026)

27. Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H., Teller, E.: Equation of state calculations by fast computing machines. J. Chem. Phys. 21, 1087 (1953). doi:10.1063/1.1699114

28. Mouret, J.B., Clune, J.: Illuminating search spaces by mapping elites. arXiv:1504.04909 (2015)

29. Olle, J., Zen, R., Puviani, M., Marquardt, F.: Simultaneous discovery of quantum error correction codes and encoders with a noise-aware reinforcement learning agent. npj Quantum Information 10, 126 (2024). doi:10.1038/ s41534-024-00920-y

30. Panteleev, P., Kalachev, G.: Asymptotically good quantum and locally testable classical LDPC codes. In: STOC 2022. pp. 375–388 (2022). doi:10.1145/3519935. 3520017

31. Panteleev, P., Kalachev, G.: Quantum LDPC codes with almost linear minimum distance. IEEE Trans. Inf. Theory 68(1), 213–229 (2022). doi:10.1109/TIT.2021. 3119384

32. Qian, D., Li, T.: Multi-agent discovery of practical quantum LDPC codes. arXiv:2608.08996 (2026)

33. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv:1707.06347 (2017)

34. Sella, O.S., Pinsler, R., Heinis, T.: Designing quantum error correcting codes to fit decoders via reinforcement learning. arXiv:2608.15754 (2026)

35. Steane, A.M.: Multiple-particle interference and quantum error correction. Proc. R. Soc. London, Ser. A 452(1954), 2551–2577 (1996). doi:10.1098/rspa.1996.0136

36. Su, V.P., Cao, C., Hu, H.Y., Yanay, Y., Tahan, C., Swingle, B.: Discovery of optimal quantum error correcting codes via reinforcement learning. Phys. Rev. Applied 23, 034048 (2025). doi:10.1103/PhysRevApplied.23.034048

37. Swendsen, R.H., Wang, J.S.: Replica Monte Carlo simulation of spin-glasses. Phys. Rev. Lett. 57, 2607 (1986). doi:10.1103/PhysRevLett.57.2607

38. Tiapkin, D., Morozov, N., Naumov, A., Vetrov, D.: Generative flow networks as entropy-regularized RL. In: AISTATS (2024)

39. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017)

40. Vieillard, N., Pietquin, O., Geist, M.: Munchausen reinforcement learning. In: NeurIPS (2020)

41. Whitammer, E.S., Jain, M., Bengio, E., Sun, C., Bengio, Y.: Trajectory balance: Improved credit assignment in GFlowNets. In: NeurIPS. vol. 35, pp. 5955–5967 (2022). doi:10.52202/068431-0431

42. Yan, G., Li, S., Ma, P., Zhang, Q., Ma, P., Wang, J., Hsieh, M.H., Du, Y.: OmniQEC: discovering practical quantum error-correcting codes by an AI scientist. arXiv:2607.25865 (2026)