# CP-MLPs: A Tensor-Rank Theory of Tied and Untied MLP Blocks

Md Rifat Arefin Farzaneh Heidari Irina Rish Guillaume Rabusseau Mila – Quebec AI Institute Université de Montréal Montréal, QC, Canada rifat.arefin@mila.quebec

## Abstract

Transformer feed-forward blocks combine several design choices: squared activations, gating, and tied versus untied parameter sharing, whose separate contributions to expressivity remain poorly understood. We introduce CP-MLPs, a tensor framework in which each multiplicative unit contributes a rank-1 interaction with three roles: detecting an input feature, reading a value, and writing to the output. Tied gated blocks such as ReLU<sup>2</sup> force the detector and value roles to share the same direction; untied gated blocks such as ReGLU and SwiGLU let them differ. We show that this distinction yields exact, degree-wise tensor-rank separations: untied detector-value units are strictly more width-efficient for indefinite quadratic interactions and role-asymmetric monomials, while tied units are structurally matched to symmetric pure-power targets. The width advantage of untying therefore depends on the target structure. Experiments show loss plateaus below the exact-representation width, but these plateaus alone do not establish a positive approximation gap: arbitrarily accurate approximation is governed by border rank. In trained gated MLPs, untied blocks retain an advantage in selected comparisons at matched input-projection budget. Code is available at https://github.com/rarefin/CP-MLPs.

## 1 Introduction

Feed-forward layers account for roughly two-thirds of the parameters in a transformer [Vaswani et al., 2017] language model [Geva et al., 2021], and a wide range of designs have been proposed for them. Common variants include squared activations such as ReLU<sup>2</sup> [So et al., 2021], gated blocks such as ReGLU and SwiGLU [Shazeer, 2020], and tied variants of SwiGLU. Bilinear MLPs are less common as training blocks in modern LLMs, but they provide a clean quadratic base case and have received attention in the interpretability literature [Pearce et al., 2025]. These architectures differ not only in their gating nonlinearity, but also in whether the same direction is used both to detect a feature and to read out its value. In this work we ask when untying these two roles improves expressivity, in the sense of reducing the number of channels required to represent a given interaction.

ReLU<sup>2</sup> and tied SwiGLU use the same direction for the detector and value. Untied blocks such as bilinear MLPs, ReGLU, and SwiGLU allow the two to differ. This decouples two design choices that are typically conflated: the choice of gate, and whether the detector and value branches are tied.

The comparison is naturally made after input symmetrization. A polynomial function evaluates its coefficient tensor on repeated copies of the same input, so only the input-symmetric part of the tensor contributes to the function it represents [Comon et al., 2008]. Tied and untied blocks therefore parameterize the same space of input-symmetric tensors, but through different families of atoms.

Untying helps when the target interaction is better matched to an untied detector-value atom than to a tied pure-power atom.

Our analysis gives exact width separations at fixed polynomial degree. At degree two, untied bilinear atoms can represent certain indefinite quadratic interactions with strictly fewer channels than tied atoms. At higher degree, role-asymmetric monomials are realized by a single untied detector-value atom but require multiple tied pure-power atoms. For smooth gates such as SwiGLU, these statements transfer degree by degree to the Taylor coefficient tensors. These degree-wise results give necessary, but not generally sufficient, width bounds for the full analytic function. The point is not that tying is uniformly worse: tied atoms are well suited to symmetric pure-power targets, and untied atoms to interactions that separate detection from value.

Our coefficient-space experiments distinguish exact rank, border rank, and numerical recovery (Appendices D.9–D.11). A positive approximation gap requires width below border rank, the minimum width permitting arbitrarily accurate approximation. Below exact rank but at or above border rank, the error infimum is zero despite possible numerical loss plateaus. Both families represent pure-power targets at width one, so differences there concern optimization. The untied advantage also appears in selected trained gated-MLP comparisons under parameter matching.

## Contributions.

1. A unified CP-MLP framework. We express bilinear, $\mathrm { R e L U ^ { 2 } }$ , tied SwiGLU, ReGLU, and SwiGLU blocks in a shared atom-based tensor notation with detector, value, and output roles.

2. Exact atom-rank separations. We prove that untied detector-value atoms achieve strictly lower width on indefinite quadratic interactions and role-asymmetric monomials, while tied atoms remain well matched to pure-power targets.

3. Empirical link between exact rank, border rank, and loss. We distinguish positive approximation gaps from zero-infimum limits and optimization failures, and evaluate tied and untied gated MLPs under width and parameter-budget controls.

## 2 Related Work

Transformer MLP variants and interpretations. Transformer feed-forward designs have diversified considerably. GLU-style activations were introduced for language modeling by Dauphin et al. [2017], and variants such as ReGLU and SwiGLU were later shown to improve Transformer feed-forward layers [Shazeer, 2020]. Squared-ReLU activations emerged as effective Transformer primitives through architecture-search studies [So et al., 2021], building on a longer line of activationfunction design that includes Swish/SiLU [Ramachandran et al., 2017]. In parallel, interpretability work has argued that Transformer feed-forward layers behave like structured key-value memories rather than generic nonlinear post-processing [Geva et al., 2021]. These works motivate studying MLP blocks as structured computational modules, but do not directly isolate the expressivity effect of tying versus untying detector and value branches, which is the focus of this paper.

Tensor, polynomial, and Waring-rank views. Our framework uses canonical polyadic (CP) tensor decomposition, where tensors are written as sums of rank-1 outer products [Kolda and Bader, 2009]. Recent work on bilinear MLPs also exploits tensor structure to analyze feed-forward blocks, observing that bilinear layers admit a third-order tensor representation [Pearce et al., 2025]. The Π-Net family of Chrysos et al. [2020, 2022] parameterizes polynomial expansions of the input directly through tensor decompositions, demonstrating that such polynomial blocks are competitive with standard activations on a range of tasks. Polynomial networks have also been studied with algebraic and tensor tools, for example through the geometry of polynomial function classes [Kileel et al., 2019]. Our work differs in that we do not propose polynomial networks as an architecture; we use polynomial structure as an analytical lens for comparing existing MLP variants.

For the higher-degree separation we rely on classical Waring rank: a scalar tied degree-k channel is a pure power $( \ell ^ { \top } x ) ^ { k }$ , so tied atom rank coincides with Waring rank. The monomial Waring-rank formula of Carlini et al. [2011], together with the real-rank result of Carlini et al. [2017], implies that $x _ { 1 } ^ { k - 1 }$ x<sub>2</sub> requires k tied atoms but only one untied detector-value atom. This exact-rank separation differs from the approximation threshold: the same monomial has tied border rank two [Coma and Seiguer, 2011]. Our contribution is to connect these rank facts to practical MLP atom families, showing that tied and untied blocks draw on different atoms inside the same input-symmetric tensor space.

![](images/60679813b471d91c7ca1931a64c91196912aef59525dcefd84c9bb6131fcf51c.jpg)  
Figure 1: Leg diagram of a single CP-MLP channel. The channel computes $h _ { a } ( x ) = g ( u _ { a } ^ { \top } x ) \left( v _ { a } ^ { \top } x \right)$ and writes the result through $w _ { a }$ . At each active polynomial degree, the channel contributes one rank-1 CP atom with output, detector, and value legs.

## 3 The CP-MLP framework

We use a common tensor notation for the multiplicative feed-forward blocks studied in this paper. Bilinear blocks multiply two linear projections, GLU-style blocks gate one projection by another, and square-style blocks reuse the same projection twice. In every case, each hidden channel produces a product of projected input directions, and the output matrix sums these channel-wise products. At each active degree, the raw coefficient tensor has the structure of a canonical polyadic (CP) decomposition [Kolda and Bader, 2009], and we call the resulting representation a $C P - M L P$

The framework separates two design choices: the gate, which selects the polynomial degrees that appear in a local expansion, and the tying pattern, which selects whether the detector and value directions of a channel are shared. The expressivity analysis in Section 4 isolates the effect of the second choice.

Definition 3.1 (CP-MLP block). Fix a scalar gate $g : \mathbb { R }  \mathbb { R }$ and let $G : \mathbb { R } ^ { m }  \mathbb { R } ^ { m }$ be its coordinatewise extension, $G ( z ) ~ = ~ \bigl ( g ( z _ { 1 } ) , \ldots , g ( z _ { m } ) \bigr ) ^ { \mid }$ Given weights $U , V \in \mathbb { R } ^ { m \times d }$ (rows $u _ { a } ^ { \top } , v _ { a } ^ { \top } )$ ) and $W \in \mathbb { R } ^ { p \times m }$ (columns $w _ { a } ) ,$ a width-m CP-MLP block is

$$
F ( x ) = W { \bigl ( } G ( U x ) \odot V x { \bigr ) } = \sum _ { a = 1 } ^ { m } w _ { a } g ( u _ { a } ^ { \top } x ) ( v _ { a } ^ { \top } x ) .
$$

We call $u _ { a }$ the detector direction, $v _ { a }$ the value direction, and $w _ { a }$ the output direction ofchannel a. A tied block constrains $V = U ,$ ; an untied block parameterizes U and V separately, allowing equality as a special case. We consider bias-free blocks throughout.

The detector direction selects the scalar feature passed through the gate, while the value direction supplies the linear signal that the gate modulates. Tying forces both roles onto the same direction; untying lets them differ. Figure 1 shows the corresponding three-leg view of a single channel.

## 3.1 Degree-wise atoms

Assume the gate is analytic at the origin, $\begin{array} { r } { g ( z ) = \sum _ { j \geq 0 } c _ { j } z ^ { j } } \end{array}$ (piecewise-analytic gates such as ReLU are treated region-wise; see Table 1). Substituting $z = u _ { a } ^ { \top }$ x and multiplying by $v _ { a } ^ { \top } x ,$ , one channel contributes $\begin{array} { r } { \bar { \sum _ { j \geq 0 } c _ { j } ( u _ { a } ^ { \top } x ) ^ { j } ( v _ { a } ^ { \top } x ) } } \end{array}$ ; reindexing by total degree $k = j + 1$ , the degree-k term carries the coefficient $c _ { k - 1 }$ . Writing $T [ x , \dots , x ] \in \mathbb { R } ^ { p }$ for the contraction of the k input modes of $T \in \mathbb { R } ^ { p } \otimes ( \mathbb { R } ^ { d } ) ^ { \otimes k }$ against x (output mode free; conventions in Appendix A.1), the degree-k component of the block is $F ^ { ( k ) } ( x ) = T ^ { ( k ) } [ x , \dots , x ]$ , with raw coefficient tensor

$$
T ^ { ( k ) } = c _ { k - 1 } \sum _ { a = 1 } ^ { m } w _ { a } \otimes u _ { a } ^ { \otimes ( k - 1 ) } \otimes v _ { a } \in \mathbb { R } ^ { p } \otimes ( \mathbb { R } ^ { d } ) ^ { \otimes k } .
$$

Each active degree thus admits $\mathbf { a } \mathrm { C P }$ decomposition with at most m rank-one terms, one per channel.   
We define $\bar { \cal S } ( g \bar { \cal ) }$ for the polynomial signature of the gate, i.e. the set of active degrees.

Table 1: Feed-forward blocks as CP-MLPs. The gate controls which degrees are active through its polynomial signature; the tying pattern controls whether detector and value directions are shared.
<table><tr><td>Block</td><td> $\mathbf { G a t e } \ g ( z )$ </td><td></td><td>Tying Signature  $S ( g )$ </td><td>Effective atom</td></tr><tr><td>Bilinear [Pearce et al., 2025]</td><td>z</td><td>untied</td><td>{2}</td><td> $w \otimes \mathrm { S y m } ( u \otimes v )$ </td></tr><tr><td>ReGLU [Shazeer, 2020]</td><td>ReLU(z)</td><td>untied</td><td>{2}†</td><td>region-wise  $w \otimes \operatorname { S y m } ( u \otimes v )$ </td></tr><tr><td>SwiGLU [Shazeer, 2020]</td><td> $\mathrm { S i L U } ( z ) = z \sigma ( z )$ </td><td>untied</td><td> $\{ 2 , 3 , 5 , 7 , \hdots \}$ </td><td> $w \otimes \mathrm { S y m } ( u ^ { \otimes ( k - 1 ) } \otimes v )$ </td></tr><tr><td>ReL  $\boldsymbol { { \mathrm { ~ U } } ^ { 2 } }$  [So et al., 2021]</td><td> $\operatorname { R e L U } ( z )$ </td><td>tied</td><td>{2}†</td><td> $\operatorname { r e g i o n - w i s e } w \otimes u ^ { \otimes 2 }$ </td></tr><tr><td>Tied SwiGLU</td><td> $\mathrm { S i L U } ( z ) = z \sigma ( z )$ </td><td>tied</td><td> $\{ 2 , 3 , 5 , 7 , \hdots \}$ </td><td> $w \otimes u ^ { \otimes k }$ </td></tr></table>

<sup>†</sup>Region-wise: ReLU is piecewise linear, so the channel is a degree-2 polynomial inside each fixed activation region.

Definition 3.2 (Polynomial signature). The polynomial signature of g is $S ( g ) : = \{ k \geq 1 : c _ { k - 1 } \neq$ 0}.

The gate selects which degrees are active through $S ( g )$ ; the tying pattern selects the atom used at each active degree. When $v _ { a } = u _ { a }$ , the raw degree-k atom collapses to $w _ { a } \otimes u _ { a } ^ { \otimes k }$ ; when $v _ { a }$ is independent of $u _ { a }$ , it retains the detector-value form $w _ { a } \otimes u _ { a } ^ { \otimes ( k - 1 ) } \otimes v _ { a }$

## 3.2 Effective atoms after symmetrization

The tensor above distinguishes detector and value slots, but the function it represents evaluates the tensor on repeated copies of the same input, so input-slot order is not observable. We write $\mathrm { S y m } ( T )$ for the symmetrization of $T _ { \ast }$ , the average of $T$ over all permutations of its input modes (with the output mode, when present, left fixed), then

$$
T [ x , \dots , x ] = \operatorname { S y m } ( T ) [ x , \dots , x ] .
$$

The proof is immediate: every permutation of T receives the same vector x in every input slot. A full statement, together with the quadratic matrix specialization, is given in Appendix B.1.

After symmetrization, each architecture yields a family of input-symmetric atoms; an untied atom can have tensor rank greater than one.

Definition 3.3 (Effective tied and untied CP-MLP atoms). The degree-k effective untied atom is

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { u n t i e d } } ^ { ( k ) } ( w , u , v ) : = w \otimes \mathrm { S y m } \big ( u ^ { \otimes ( k - 1 ) } \otimes v \big ) \in \mathbb { R } ^ { p } \otimes \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } ) , } \end{array}
$$

where $\mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ denotes the space of symmetric order-k tensors (Definition B.1). The degree-k effective tied atom is the specialization $v = u \colon$

$$
\mathcal A _ { \mathrm { t i e d } } ^ { ( k ) } ( w , u ) : = w \otimes u ^ { \otimes k } \in \mathbb R ^ { p } \otimes \mathrm { S y m } ^ { k } ( \mathbb R ^ { d } ) .
$$

Tied and untied blocks therefore parameterize tensors in the same symmetric space $\mathbb { R } ^ { p } \otimes \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ but draw on different atom families: tied blocks use pure-power atoms $w \otimes u ^ { \otimes k }$ , while untied blocks use symmetrized detector-value atoms $w \otimes \mathrm { S y m } ( \bar { u } ^ { \otimes ( k - 1 ) } \otimes v )$ . Setting $v = u$ recovers the tied family as a special case. The benefit of untying is not that antisymmetric components become visible, but that the symmetrized untied family can represent certain symmetric tensors with strictly fewer atoms.

Table 1 places the blocks studied in this paper along the two axes of the framework: the gate and the tying pattern. Bilinear, ReGLU, and SwiGLU differ only in their gates and share the untied detector-value atom structure. ReLU<sup>2</sup> and tied SwiGLU use tied atoms, with the same direction serving both the detector and value roles. Tied SwiGLU sets $v = u$ while retaining the SiLU gate, giving SiL $\mathrm { U } ( z ) z = z ^ { 2 } \sigma ( z )$

The next section asks how many atoms of each family are required to represent a given target tensor.

## 4 Expressivity via CP-MLP atom rank

The atom families of Section 3.2 suggest a natural notion of width: the smallest number of atoms needed to express a given coefficient tensor. Fixing an active degree $k ,$ each CP-MLP channel contributes one effective atom in $\mathcal { T } _ { k } : = \mathbb { R } ^ { p } \otimes \operatorname { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ , so the question we want to answer is how many such atoms are required to represent a target $T \in \mathcal { T } _ { k }$ . We call this minimum the $C P - M L P$ atom rank; since each atom corresponds to one channel at degree k, it is the degree-wise width required to represent a coefficient tensor.

For polynomial gates, $T ^ { ( k ) }$ is the ordinary degree-k coefficient tensor; for analytic gates, it is the degree-k Taylor coefficient tensor, satisfying $\bar { D } ^ { k } F ( 0 ) = k ! \operatorname { S y m } ( T ^ { ( k ) } )$ . Throughout this section k denotes an active degree of the gate, and any nonzero scalar Taylor coefficient is absorbed into the output factor w, so the rank comparisons below depend only on the channel atom family.

## 4.1 Atom-rank definitions

Definition 4.1 (Tied CP-MLP atom rank). For $T \in \mathcal { T } _ { k }$ , the tied CP-MLP atom rank $r _ { \mathrm { t i e d } } ( T )$ is the smallest integer m such that $\begin{array} { r } { T = \sum _ { a = 1 } ^ { m } w _ { a } \otimes u _ { a } ^ { \otimes k } } \end{array}$ $w _ { a } \in \mathbb { R } ^ { p } , \quad u _ { a } \in \mathbb { R } ^ { d }$

Definition 4.2 (Untied CP-MLP atom rank). For $T \in \mathcal { T } _ { k }$ , the untied CP-MLP atom rank $r _ { \mathrm { u n t i e d } } ( T )$ is the smallest integer m such that $\begin{array} { r } { T = \sum _ { a = 1 } ^ { m } w _ { a } \otimes \mathrm { S y m } \big ( u _ { a } ^ { \otimes ( k - 1 ) } \otimes v _ { a } \big ) } \end{array}$ ， $w _ { a } \in \mathbb { R } ^ { p } , \quad u _ { a } , v _ { a } \in$ $\mathbb { R } ^ { d }$

Both ranks are minimum-width quantities at a fixed degree, counting the fewest atoms required to express a coefficient tensor in the corresponding family. Since a tied atom is recovered from an untied atom by setting $v _ { a } = u _ { a } ,$ the untied family contains the tied family, which immediately implies the following inequality.

Proposition 4.3 (Untied atoms contain tied atoms). For every $T \in { \mathcal { T } } _ { k } , r _ { \mathrm { u n t i e d } } ( T ) \leq r _ { \mathrm { t i e d } } ( T )$

Proof. Any tied representation $\begin{array} { r } { T = \sum _ { a } w _ { a } \otimes u _ { a } ^ { \otimes k } } \end{array}$ is also an untied representation with $\ v _ { a } = u _ { a }$ since $u _ { a } ^ { \otimes k } = \mathrm { S y m } ( u _ { a } ^ { \otimes ( k - 1 ) } \otimes u _ { a } )$ . Minimizing over tied representations yields the claim. □

The remainder of the section asks when this inequality is strict. We first study degree two, where the coefficient tensor is a symmetric matrix, and then prove a higher-degree separation using a role-asymmetric monomial.

Remark 4.4 (Scalar targets suffice). For a scalar target $S \in \mathrm { { S y m } } ^ { k } ( \mathbb { R } ^ { d } )$ , embedding $T = e _ { o } \otimes S$ gives $r _ { \mathrm { t i e d } } ( T ) = r _ { \mathrm { t i e d } } ( S )$ and $r _ { \mathrm { u n t i e d } } ( T ) = r _ { \mathrm { u n t i e d } } ( S )$ : output projection cannot increase atom rank, and any scalar representation lifts by tensoring with $e _ { o } .$ . Full details are in Appendix C.1. The separations below are stated for scalar targets.

## 4.2 Degree-two bilinear atoms

We first specialize to degree 2, where a scalar coefficient tensor is a symmetric matrix and the atom-rank question reduces to representing quadratic forms

$$
\begin{array} { r } { f _ { M } ( \boldsymbol { x } ) = \boldsymbol { x } ^ { \top } M \boldsymbol { x } , \qquad M = M ^ { \top } \in \mathbb { R } ^ { d \times d } . } \end{array}
$$

A tied atom contributes the signed square $\alpha ( u ^ { \top } x ) ^ { 2 }$ , corresponding to the rank-1 symmetric matrix $\alpha u u ^ { \top }$ . An untied atom contributes a product of two linear forms, $\alpha ( u ^ { \top } x ) ( v ^ { \top } x )$ , which after symmetrization becomes

$$
\begin{array} { r } { \frac { \alpha } { 2 } ( u v ^ { \top } + v u ^ { \top } ) . } \end{array}
$$

A tied atom therefore contributes one quadratic direction, while an untied atom can couple two. Our first result records this coarse rank comparison.

Theorem 4.5 (Coarse bilinear rank bounds). Let $M \in \mathbb { R } ^ { d \times d }$ be symmetric. IfM is represented by a tied bilinear block of width m, then ran $\mathrm { k } ( M ) \leq$ min(m, d). If M is represented by an untied bilinear block ofwidth m, then rank $( M ) \leq \operatorname* { m i n } ( 2 m , d )$ . Both bounds are tight.

Proofsketch. Each tied unit contributes the rank-1 symmetric matrix $\boldsymbol { u _ { a } } \boldsymbol { u _ { a } ^ { \intercal } }$ , so a width-m tied block produces matrices of rank at most mi $_ 1 ( m , d )$ . For untied blocks, only the symmetric part of $u _ { a } v _ { a } ^ { \top }$ contributes; the resulting matrix $\begin{array} { r } { \frac { 1 } { 2 } ( u _ { a } v _ { a } ^ { \top } + v _ { a } u _ { a } ^ { \top } ) } \end{array}$ has image in span $\{ u _ { a } , v _ { a } \}$ and therefore rank at most two, and summing m such terms gives the bound. Both bounds are tight by choosing independent coordinate directions. □

Theorem 4.5 shows that an untied atom can carry twice the rank of a tied atom, but rank alone does not determine which matrices a given width can represent. The rank-2 gain has a specific sign structure. From the difference-of-squares identity

$$
( u ^ { \top } x ) ( v ^ { \top } x ) = \textstyle { \frac { 1 } { 4 } } { \big ( } ( u + v ) ^ { \top } x { \big ) } ^ { 2 } - \textstyle { \frac { 1 } { 4 } } { \big ( } ( u - v ) ^ { \top } x { \big ) } ^ { 2 } ,
$$

a non-collinear untied atom is a difference of two squares. It can pair one positive with one negative quadratic direction—representing interactions like $\dot { a } ^ { 2 } - b ^ { 2 } = ( a \dot { + } b ) ( a - \dot { b } ) \dot { - }$ —but cannot compress two same-sign directions such as $\overline { { a ^ { 2 } + b ^ { 2 } } }$ into a single atom. The exact width is therefore controlled not by rank, but by how many positive and negative quadratic directions the target carries.

Let $n _ { + } ( M )$ and $n _ { - } ( M )$ denote the numbers of positive and negative eigenvalues of M.

Theorem 4.6 (Exact bilinear atom ranks). For every symmetric matrix $M \in \mathbb { R } ^ { d \times d }$

$$
r _ { \mathrm { t i e d } } ( M ) = \mathrm { r a n k } ( M ) = n _ { + } ( M ) + n _ { - } ( M ) ,
$$

whereas

$$
r _ { \mathrm { u n t i e d } } ( M ) = \operatorname* { m a x } \{ n _ { + } ( M ) , n _ { - } ( M ) \} .
$$

Proofsketch. For tied atoms, each channel contributes one signed rank-1 square, so it needs one atom per nonzero eigendirection, and the spectral decomposition achieves this number. For untied atoms, the difference-of-squares identity expresses any width-m representation as $M = A A ^ { \top } - B B ^ { \top }$ with $A , B \in \mathbb { R } ^ { d \times m }$ . On $\ker ( A ^ { \top } ) , \dot { x } ^ { \top } \dot { M } x = - \| \dot { B } ^ { \top } x \| ^ { 2 } \leq 0$ , so M cannot be positive definite on a subspace of dimension greater than m; the analogous argument on ker $( B ^ { \top } )$ gives the same bound for negative directions, so $m \ge \operatorname* { m a x } \{ n _ { + } ( M ) , n _ { - } ( M ) \}$ . The matching upper bound is achieved by pairing opposite-sign eigendirections one atom at a time, with unpaired directions handled by the tied special cases $u = \pm v$ . Full proofs of Theorems 4.5 and 4.6 are in Appendix C.2. □

Corollary 4.7 (When untied bilinear atoms help). $I f M \succeq 0 o r M \preceq 0 ,$

$$
r _ { \mathrm { t i e d } } ( M ) = r _ { \mathrm { u n t i e d } } ( M ) = \mathrm { r a n k } ( M ) .
$$

If M is indefinite,

$$
r _ { \mathrm { t i e d } } ( M ) - r _ { \mathrm { u n t i e d } } ( M ) = \operatorname* { m i n } \{ n _ { + } ( M ) , n _ { - } ( M ) \} .
$$

Untied atoms are therefore most width-efficient on indefinite quadratic interactions; they offer no advantage on purely same-sign quadratic structure.

Connection to hard-gated blocks. The same degree-two atom gap appears regionwise in hard-gated CP-MLPs. On a fixed activation pattern $\mu \in \ \{ 0 , 1 \} ^ { m }$ , ReGLU realizes $Q _ { \mu , o } ~ =$ ${ \begin{array} { r } { \sum _ { a = 1 } ^ { m } \mu _ { a } W _ { o a } { \frac { u _ { a } v _ { a } ^ { \top } + v _ { a } u _ { a } ^ { \top } } { 2 } } } \end{array} }$ for each output coordinate o, whereas $\mathrm { R e L U ^ { 2 } }$ realizes $\begin{array} { r l } { Q _ { \mu , o } } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { a } \mu _ { a } W _ { o a } u _ { a } u _ { a } ^ { \top } } \end{array}$ . Inside each fixed region, ReGLU has access to untied rank-2 atoms with oppositesign structure, while $\mathrm { R e L U ^ { 2 } }$ is restricted to tied rank-1 atoms. Region switching adds further piecewise expressivity; Theorem 4.6 isolates the within-region atom difference.

## 4.3 Higher-degree monomial separation

The degree-two analysis showed that untied atoms can be more width-efficient when the target contains opposite-sign quadratic structure. At higher degree, a different kind of separation emerges that is driven not by sign, but by role asymmetry: one direction is repeated many times in the detector slot, while another direction appears only once in the value slot.

For $k \geq 2$ , consider the binary monomial $F _ { k } ( x _ { 1 } , x _ { 2 } ) = x _ { 1 } ^ { k - 1 } x _ { 2 }$ . This target has the exact shape of a single untied atom: taking $u = e _ { 1 }$ and $v = e _ { 2 }$ gives $( u ^ { \top } x ) ^ { \bar { k } - 1 } ( v ^ { \top } x ) = x _ { 1 } ^ { k - 1 } x _ { 2 }$ , so one untied atom suffices. A tied degree-k atom, by contrast, has the pure-power form $( \ell ^ { \top } x ) ^ { k }$ , so the tied atom rank of $F _ { k }$ is exactly its Waring rank.

Definition 4.8 (Waring rank). Let $\mathbb { F } \in \{ \mathbb { R } , \mathbb { C } \}$ , and let $f : \mathbb { F } ^ { d }  \mathbb { F }$ be a homogeneous polynomial of degree k. The Waring rank off over F, denoted $\operatorname { W r } _ { \mathbb { F } } ( { \dot { f } } )$ , is the smallest integer m such that

$$
f ( x ) = \sum _ { a = 1 } ^ { m } c _ { a } \ell _ { a } ( x ) ^ { k } ,
$$

where $c _ { a } \in \mathbb { F }$ and each $\ell _ { a }$ is a linear form over $\mathbb { F } .$

For scalar tied CP-MLPs, the real Waring rank coincides with the tied atom rank: any tied representation $\begin{array} { r } { f ( x ) \ = \ \sum _ { a } { c _ { a } ( u _ { a } ^ { \top } x ) ^ { k } } } \end{array}$ with $\mathbf { \bar { \boldsymbol { u } } } _ { a } ~ \in ~ \mathbb { R } ^ { d }$ is exactly a real Waring decomposition, so $r _ { \mathrm { t i e d } } ( f ) \ = \ \operatorname { W r } _ { \mathbb { R } } ( f )$ for every scalar homogeneous degree-k polynomial. This is the standard correspondence between symmetric tensor rank and polynomial Waring rank [Comon et al., 2008]. The following classical formula computes the right-hand side for monomials.

Proposition 4.9 (Waring rank of monomials; Carlini et al., 2011, 2012, 2017). Let $\mu ( x ) \ =$ $x _ { 1 } ^ { \alpha _ { 1 } } \cdot \cdot \cdot x _ { n } ^ { \alpha _ { n } }$ be a monomial of degree $k = \textstyle \sum _ { i } \alpha _ { i } ,$ with sorted exponents $1 \leq \alpha _ { 1 } \leq \cdot \cdot \cdot \leq \alpha _ { n } .$ Then

$$
\operatorname { W r } _ { \mathbb { C } } ( \mu ) = { \frac { 1 } { \alpha _ { 1 } + 1 } } \prod _ { i = 1 } ^ { n } ( \alpha _ { i } + 1 ) = \prod _ { i = 2 } ^ { n } ( \alpha _ { i } + 1 ) ,
$$

and $\operatorname { W r } _ { \mathbb { R } } ( \mu ) = \operatorname { W r } _ { \mathbb { C } } ( \mu )$ whenever $\alpha _ { 1 } = 1$

The complex formula is [Carlini et al., 2012, Prop. 3.1]; the real-complex equality for $\alpha _ { 1 } = 1$ follows from Carlini et al. [2017]. For $F _ { k } = x _ { 1 } ^ { k - 1 } x _ { 2 }$ the sorted exponent vector is $( 1 , k - 1 )$ , so

$$
\operatorname { W r } _ { \mathbb { R } } ( F _ { k } ) = \operatorname { W r } _ { \mathbb { C } } ( F _ { k } ) = ( k - 1 ) + 1 = k .
$$

Theorem 4.10 (Degree-k monomial separation). For every $k \geq 2 ,$

$$
r _ { \mathrm { u n t i e d } } ( F _ { k } ) = 1 , \quad r _ { \mathrm { t i e d } } ( F _ { k } ) = k .
$$

A single untied atom therefore represents a degree-k coefficient tensor that requires k tied atoms.

Proof. We bound the two ranks separately.

Untied rank. Setting $u = e _ { 1 }$ and $v = e _ { 2 } , ( u ^ { \top } x ) ^ { k - 1 } ( v ^ { \top } x ) = x _ { 1 } ^ { k - 1 } x _ { 2 } = F _ { k } ( x )$ , so a single untied atom realizes $F _ { k }$ and $r _ { \mathrm { u n t i e d } } ( F _ { k } ) \leq 1$ . Since $F _ { k } \neq 0 .$ , it cannot be represented with zero atoms, $\mathrm { g i v i n g } r _ { \mathrm { u n t i e d } } ( F _ { k } ) = 1$

Tied rank. Any scalar tied degree-k representation $\begin{array} { r } { F _ { k } ( x ) = \sum _ { a } c _ { a } \ell _ { a } ( x ) ^ { k } } \end{array}$ is a real Waring decomposition, so $r _ { \mathrm { t i e d } } ( F _ { k } ) = \mathrm { W r } _ { \mathbb { R } } ( \bar { F } _ { k } ) = k$ by Proposition 4.9. □

Exact representation versus approximation. The real border rank $\underline { { r } } _ { \mathsf { f } } ( T )$ is the smallest width permitting arbitrarily accurate approximation of T by real atoms from family f. In coefficient norm, a positive approximation gap occurs exactly when $m < \underline { { r } } _ { \mathsf { f } } ( T )$ . For scalar quadratics, both width classes are closed, so exact and border ranks coincide (Appendix C.4). For $F _ { k } = x _ { 1 } ^ { k - 1 } x _ { 2 }$ , however, $\underline { { r } } _ { \mathrm { t i e d } } ( F _ { k } ) = 2 .$ , whereas $r _ { \mathrm { t i e d } } ( F _ { k } ) = k .$ Thus, at $2 \leq m < k$ , the tied error infimum is zero but unattained. Under any finite bound on $\begin{array} { r } { \sum _ { a } \| w _ { a } \| \| u _ { a } \| ^ { k } } \end{array}$ , a positive tied approximation gap remains below the exact rank. Appendix C.3 gives the approximation arguments; Appendix C.5 extends the separation to sums of independent interaction blocks.

The degree-wise expansion gives necessary width bounds for exact representation of the full analytic function. If a bias-free width-m block with gate g represents an analytic target F near the origin, then

$$
m \geq \operatorname* { s u p } _ { k \in S ( g ) } r _ { \mathsf { f } } \big ( F ^ { ( k ) } \big ) , \qquad \mathsf { f } \in \{ \mathrm { t i e d , u n t i e d } \} ,
$$

where $F ^ { ( k ) }$ is the degree-k homogeneous component. Exact representation also requires $F ( 0 ) = 0$ and vanishing components at inactive degrees. These conditions are not generally sufficient: all degrees share the same channels, and the gate fixes their coefficient ratios. Appendix C.6 gives the proof and an explicit example where the full required width exceeds every individual degree-wise rank.

## 5 Experiments

We test whether the atom-rank separations from Section 4 appear as observable width and loss effects. Indefinite quadratics realize the degree-two gap (Section 5.1); role-asymmetric monomials realize the higher-degree gap (Section 5.2); pure-power targets show that tying can be the better match (Section 5.3); and gated MLPs show how the separation carries over to actual gates, including under a matched input-projection budget (Section 5.4). Optimization, hyperparameters, and budget checks are in Appendix D.

![](images/6b897b68a0cd0ea0445477cc6ed596898b6ac9713f6e6a6c95c40773692144da.jpg)

![](images/8741f37b6cf309f3dd2d21eefa9fa4646d89586de3da1d9c4ba30d1b4e589bad.jpg)  
Figure 2: Quadratic targets with $( n _ { + } , n _ { - } )$ positive and negative directions. $L e f t { \mathrm { : } }$ required widths match the predictions $n _ { + } + n _ { - }$ (tied) and max $\{ n _ { + } , n _ { - } \}$ (untied). $R i g h t \cdot$ loss stays near the target variance below the predicted width and reaches machine precision at the predicted width.

## 5.1 Indefinite quadratic targets

Theorem 4.6 predicts that untied bilinear atoms are more width-efficient than tied atoms only when the target carries both positive and negative directions. To test this, we construct

$$
Q = \mathrm { d i a g } ( { \bf 1 } ^ { n _ { + } } , - { \bf 1 } ^ { n _ { - } } , { \bf 0 } ^ { d - n _ { + } - n _ { - } } ) , \qquad y ( x ) = x ^ { \top } Q x ,
$$

with $d = 8 ,$ and sweep six choices of $( n _ { + } , n _ { - } )$ covering definite, balanced indefinite, and imbalanced indefinite targets.

Figure 2 confirms the exact threshold: tied blocks require width $n _ { + } + n _ { - }$ , untied blocks require width max $\{ n _ { + } , n _ { - } \}$ . At one width below the prediction, the loss stays near the target variance; at the predicted width, the loss drops to machine precision. Definite targets show no tied/untied gap, while indefinite targets show the predicted gap min $\{ n _ { + } , n _ { - } \}$

## 5.2 Role-asymmetric monomial targets

We next test the higher-degree separation of Theorem 4.10. For

$$
F _ { k } ( x _ { 1 } , x _ { 2 } ) = x _ { 1 } ^ { k - 1 } x _ { 2 } , \qquad k \in \{ 2 , 3 , 4 , 5 \} ,
$$

the theory predicts untied atom rank 1 and tied atom rank k. We fit the target directly in monomial coefficient space and sweep width $m = 1 , \ldots , k + 2 ;$ coefficient-space fitting removes sampling noise and isolates the rank effect.

Figure 3 shows the predicted separation at width one: untied atoms reach machine precision at $m = 1$ for every degree, while tied atoms keep a clearly positive loss. For $2 \leq m < k _ { \cdot }$ , the tied losses are small but nonzero (about $1 0 ^ { - 7 } \mathrm { t o } 1 0 ^ { - 9 }$ , with one optimizer outlier at $k = 4 , m = 2 )$ . They drop to machine precision at $m = k$ for $k \leq 3$ and at $m = k + 1$ for $k = 4 ;$ for $k = 5$ they stay between about $1 0 ^ { - 7 }$ and $5 \times 1 0 ^ { - 1 0 }$ across the tested widths. Below its atom rank the target lies outside the tied model class, but for $2 \leq m < k$ it lies in the class’s closure: the border Waring rank of ${ x _ { 1 } ^ { k - 1 } x _ { 2 } }$ is two [Comas and Seiguer, 2011], via the limit in Equation (1) with $\ell = x _ { 1 }$ and $h = x _ { 2 }$ . The plateau therefore reflects the ill-conditioning of border-rank approximations, whose coefficients must diverge as the residual shrinks, rather than a positive distance to the model class; at $m = 1$ the floor is genuine, since the set of symmetric rank-at-most-one tensors is closed. The untied advantage is specific to single-value-role targets; products with multiple independent value roles require more atoms (Appendix D.7).

## 5.3 Pure powers versus role-asymmetric targets

The previous experiment shows a strong untied advantage on role-asymmetric monomials, but tying is not always worse. At width $m = 1$ , we compare tied and untied models on two targets:

$$
x _ { 1 } ^ { k } \qquad \mathrm { a n d } \qquad x _ { 1 } ^ { k - 1 } x _ { 2 } .
$$

The first is a pure power and is realized exactly by one tied atom; the second is role-asymmetric and is realized exactly by one untied atom. Figure 4 shows the expected reversal: on pure powers, tied

![](images/88b698370475468aad6a6fa3992a47ab9a94b5482dddfe1784464874059dba90.jpg)  
Figure 3: Coefficient-space width sweep for $F _ { k } = x _ { 1 } ^ { k - 1 } x _ { 2 } , k \in \{ 2 , 3 , 4 , 5 \}$ . Untied atoms fit at width $m = 1$ . Tied atoms keep a positive loss at $m = 1 ;$ for $2 \leq m < k$ their losses are small but nonzero, as expected from border rank two, and they reach machine precision at $m = k$ for $k \leq 3$ Median over two seeds; bands show min–max.

![](images/2518fc855b50e08b9faf9165e56b022629f224e1efed340d09a934037ae45a8b.jpg)  
Figure 4: Target structure changes the low-width winner. $\mathbf { A } \mathbf { t } m = 1$ , tied atoms fit pure powers $x _ { 1 } ^ { k } .$ while untied atoms fit role-asymmetric monomials ${ x _ { 1 } ^ { k - 1 } x _ { 2 } }$ . Bands show min and max over seeds.

models reach machine precision while untied models must align the redundant value direction with the detector; on role-asymmetric targets, untied models fit at width one while tied models cannot. The low-width winner therefore depends on target structure.

## 5.4 Separation in gated MLPs

The coefficient-space experiments above isolate the atom-rank effect; we now test actual gated MLPs on the same witness $F _ { k } = x _ { 1 } ^ { k - 1 } x _ { 2 }$ . We compare tied ReL $\boldsymbol { \mathrm { J ^ { 2 } } }$ , tied SwiGLU $( { \mathrm { S i L U } } ( z ) z )$ untied ReGLU, and untied SwiGLU, using Gaussian inputs with $d = 4 , k \in \{ 2 , 3 \}$ , and widths $m \in \{ 1 , 2 , 4 \}$

Figure 5 shows that the exact polynomial threshold becomes a stable qualitative ordering for gated networks. $\mathrm { A t } k = 2$ , untied ReGLU and SwiGLU reach near-machine precision at $m = 2 ,$ , and tied $\mathrm { R e L U } ^ { 2 } \mathrm { a t } m = 4 .$ . These widths are twice the atom ranks 1 and 2. Both gates satisfy $g ( z ) - g ( - z ) = z$ and $g ( z ) z + g ( - z ) ( - z ) = z ^ { 2 }$ , so each polynomial atom can be built from two gated channels. By the second identity, tied SwiGLU can also represent $x _ { 1 } x _ { 2 }$ exactly at $m = 4$ . Its fits here stay near $\mathrm { i 0 ^ { - 3 } }$ whereas the same model reaches $5 . 5 \times 1 0 ^ { - 1 3 }$ at width four in the matched control of Appendix $_ { \mathrm { { D . 8 } , } }$ so this gap is an optimization failure. $\mathrm { A t } k = 3$ , no architecture reaches machine precision. ReLU-gated channels are positively homogeneous of degree two and cannot represent a cubic; a SiLU channel that matches the cubic term also contributes quadratic and higher-degree terms, which finite width cannot cancel exactly. Untied SwiGLU is the best-performing model, and for each gate the untied block lies below its tied counterpart at widths $m \geq 2$

A control comparing untied width m against tied width $2 m$ , which matches input-projection parameters, gives a degree-dependent picture: at $k = 2$ , tied SwiGLU at width four and untied SwiGLU at width two reach comparable losses, while at $k = 3$ untied SwiGLU at width four is about ten times more accurate than tied SwiGLU at width eight. The figure and numbers are in Appendix D.8.

![](images/7a1e4e41693ffeb48d70a3be73dd27c4f2fb941e3aabb78e880c42597ad3b175.jpg)  
Figure 5: Width versus loss for actual gates on ${ x _ { 1 } ^ { k - 1 } x _ { 2 } }$ . Left: for $k = 2 .$ , untied ReGLU and SwiGLU reach near-machine precision at smaller width than tied $\mathrm { R e L U } ^ { 2 }$ and tied SwiGLU. Right: for $k = 3 .$ the exact rank threshold becomes a qualitative ordering; untied SwiGLU is best, and each untied block stays below its same-gate tied counterpart for $m \geq 2 .$ . Median over three seeds; bands show min–max.

## 6 Conclusion

We introduced CP-MLPs, a tensor framework for comparing tied and untied multiplicative feedforward blocks. The framework separates the gate from the choice of whether detector and value directions are shared. Our theory shows that this choice changes width efficiency. Untied atoms can represent indefinite quadratic interactions and role-asymmetric monomials with fewer channels, while tied atoms are naturally matched to pure-power targets. For smooth gates such as SwiGLU, these statements apply degree by degree to Taylor coefficient tensors. Experiments confirm the predicted quadratic width thresholds and the width-one untied fits, show tied losses below the exact monomial rank that are consistent with border rank, and demonstrate that the low-width advantage changes with target structure.

Selected gated-MLP comparisons retain an untied advantage under input-projection matching, while others attain comparable errors. These results suggest that the success of GLU-style feed-forward blocks is not only an activation effect: separating detector and value directions changes which interactions are width-efficient. Several open directions remain. Our higher-degree separation is degree-wise rather than a global statement for the full analytic channel, and our experiments use synthetic polynomial targets rather than language-modeling tasks; bridging both gaps, by extending the theory to richer multi-leg atoms and by testing these effects in full Transformer language models, is a natural next step.

## Acknowledgments

We acknowledge the support from the Canada CIFAR AI Chair Program and from the Canada Excellence Research Chairs Program. The research was enabled in part by computational resources provided by the Digital Research Alliance of Canada and Mila-Quebec AI Institute.

## References

Hirotachi Abo and Nick Vannieuwenhoven. Most secant varieties of tangential varieties to Veronese varieties are nondefective. Transactions ofthe American Mathematical Society, 370(1):393–420, 2018. URL https://arxiv.org/abs/1510.02029.

James Alexander and André Hirschowitz. Polynomial interpolation in several variables. Journal of Algebraic Geometry, 4(2):201–222, 1995.

Enrico Carlini, Maria Virginia Catalisano, and Anthony V Geramita. The solution to Waring’s problem for monomials. arXiv preprint arXiv:1110.0745, 2011.

Enrico Carlini, Maria Virginia Catalisano, and Anthony V. Geramita. The solution to the Waring problem for monomials and the sum of coprime monomials. Journal of Algebra, 370:5–14, 2012. doi: 10.1016/j.jalgebra.2012.07.028.

Enrico Carlini, Mario Kummer, Alessandro Oneto, and Emanuele Ventura. On the real rank of monomials. Mathematische Zeitschrift, 286(1):571–577, 2017.

Grigorios G Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Yannis Panagakis, Jiankang Deng, and Stefanos Zafeiriou. Π-nets: Deep polynomial neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7325–7335, 2020.

Grigorios G Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Jiankang Deng, Yannis Panagakis, and Stefanos Zafeiriou. Deep polynomial neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8):4021–4034, 2022.

Gonzalo Comas and Malena Seiguer. On the rank of a binary form. Foundations of Computational Mathematics, 11(1):65–78, 2011. doi: 10.1007/s10208-010-9077-x.

Pierre Comon, Gene Golub, Lek-Heng Lim, and Bernard Mourrain. Symmetric tensors and symmetric tensor rank. SIAM Journal on Matrix Analysis and Applications, 30(3):1254–1279, 2008.

Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In International Conference on Machine Learning, pages 933–941. PMLR, 2017.

Vin de Silva and Lek-Heng Lim. Tensor rank and the ill-posedness of the best low-rank approximation problem. SIAM Journal on Matrix Analysis and Applications, 30(3):1084–1127, 2008. URL https://arxiv.org/abs/math/0607647.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495, 2021.

Joe Kileel, Matthew Trager, and Joan Bruna. On the expressive power of deep polynomial neural networks. In Advances in Neural Information Processing Systems, volume 32, 2019.

Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. SIAM Review, 51(3): 455–500, 2009.

Michael T Pearce, Thomas Dooms, Alice Rigg, Jose M Oramas, and Lee Sharkey. Bilinear MLPs enable weight-based mechanistic interpretability. In International Conference on Learning Representations, 2025.

Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2017.

Noam Shazeer. GLU variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

David So, Wojciech Manke, Hanxiao Liu, Zihang Dai, Noam Shazeer, and Quoc V Le. Searching´ for efficient transformers for language modeling. In Advances in Neural Information Processing Systems, volume 34, pages 6010–6022, 2021.

Nick Vannieuwenhoven. A condition number for the tensor rank decomposition. Linear Algebra and its Applications, 535:35–86, 2017. URL https://arxiv.org/abs/1604.00052.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

## A Preliminaries

This appendix collects the tensor and CP-decomposition notation used throughout the paper.

## A.1 Tensors and CP decompositions

Let $T \in \mathbb { R } ^ { d _ { 1 } \times \cdots \times d _ { n } }$ be an order-n tensor and let $x _ { i } \in \mathbb { R } ^ { d _ { i } }$ . We write the multilinear contraction of T with vectors $x _ { 1 } , \ldots , x _ { n }$ as

$$
T [ x _ { 1 } , \ldots , x _ { n } ] : = T \times _ { 1 } x _ { 1 } \times _ { 2 } \cdot \cdot \cdot \times _ { n } x _ { n } ,
$$

where $\times _ { i }$ denotes contraction along mode i:

$$
( T \times _ { i } x _ { i } ) _ { j _ { 1 } , \dots , j _ { i - 1 } , j _ { i + 1 } , \dots , j _ { n } } = \sum _ { k = 1 } ^ { d _ { i } } T _ { j _ { 1 } , \dots , j _ { i - 1 } , k , j _ { i + 1 } , \dots , j _ { n } } ( x _ { i } ) _ { k } .
$$

Equivalently,

$$
T [ x _ { 1 } , \ldots , x _ { n } ] = \sum _ { j _ { 1 } = 1 } ^ { d _ { 1 } } \cdots \sum _ { j _ { n } = 1 } ^ { d _ { n } } T _ { j _ { 1 } , \ldots , j _ { n } } ( x _ { 1 } ) _ { j _ { 1 } } \cdots ( x _ { n } ) _ { j _ { n } } .
$$

When all modes are contracted with vectors, the result is a scalar. If some modes are left uncontracted, the same formula applies to the remaining modes only; in particular, contracting the k input modes of $T \in \mathbb { R } ^ { p } \otimes ( \mathbb { R } ^ { d } ) ^ { \dot { \otimes } \dot { k } }$ while leaving the output mode free gives $T [ x _ { 1 } , \dots , x _ { k } ] \in \mathbb { R } ^ { p }$ , the convention used in Section 3.1. We write $e _ { i }$ for the i-th standard basis vector.

A canonical polyadic (CP) decomposition represents a tensor as a sum of rank-one outer products:

$$
T = \sum _ { a = 1 } ^ { r } \lambda _ { a } u _ { a } ^ { ( 1 ) } \otimes u _ { a } ^ { ( 2 ) } \otimes \cdot \cdot \cdot \otimes u _ { a } ^ { ( n ) } ,
$$

where $u _ { a } ^ { ( i ) } \in \mathbb { R } ^ { d _ { i } }$ and $\lambda _ { a } \in \mathbb { R }$ . Under contraction, each rank-one term factorizes:

$$
T [ x _ { 1 } , \ldots , x _ { n } ] = \sum _ { a = 1 } ^ { r } \lambda _ { a } \prod _ { i = 1 } ^ { n } \langle u _ { a } ^ { ( i ) } , x _ { i } \rangle .
$$

Thus a CP decomposition expresses a multilinear map as a sum of separable products of linear forms. This is the tensor structure used throughout the paper. For example, a single CP-MLP channel at degree k contributes a rank-one raw tensor of the form

$$
w _ { a } \otimes u _ { a } ^ { \otimes ( k - 1 ) } \otimes v _ { a } ,
$$

whose diagonal evaluation gives

$$
\bigl ( w _ { a } \otimes u _ { a } ^ { \otimes ( k - 1 ) } \otimes v _ { a } \bigr ) [ x , \ldots , x ] = w _ { a } ( u _ { a } ^ { \top } x ) ^ { k - 1 } ( v _ { a } ^ { \top } x ) .
$$

A width-m CP-MLP sums m such channel-wise rank-one contributions.

Symmetrization. For any tensor $T \in ( \mathbb { R } ^ { d } ) ^ { \otimes k }$ , define

$$
\operatorname { S y m } ( T ) = { \frac { 1 } { k ! } } \sum _ { \pi \in S _ { k } } \pi \cdot T .
$$

Symmetric CP rank. A symmetric tensor T has symmetric rank r if r is the smallest integer such that

$$
T = \sum _ { a = 1 } ^ { r } \lambda _ { a } u _ { a } ^ { \otimes k } .
$$

Matricization lower bound. For any tensor $T ,$ the rank of any flattening provides a lower bound on its CP rank [Kolda and Bader, 2009].

## B Additional framework details

## B.1 Symmetrization details

Let $S _ { k }$ denote the symmetric group on k elements. For a permutation $\pi \in S _ { k }$ and a tensor $T \in$ $\mathbb { R } ^ { p } \otimes { \overset { \cdot \cdot } { ( } } \mathbb { R } ^ { d } ) ^ { \otimes k }$ , write $\pi \cdot T$ for the tensor obtained by permuting the k input modes according to π, leaving the output mode fixed.

Definition B.1 (Input-symmetric tensor). A tensor

$$
T \in \mathbb { R } ^ { p } \otimes ( \mathbb { R } ^ { d } ) ^ { \otimes k }
$$

is input-symmetric $i f \pi \cdot T = T$ for every $\pi \in S _ { k }$ . The space of input-symmetric tensors is denoted

$$
\mathbb { R } ^ { p } \otimes \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } ) .
$$

Definition B.2 (Input symmetrization). The input symmetrization of $T \in \mathbb { R } ^ { p } \otimes ( \mathbb { R } ^ { d } ) ^ { \otimes k }$ is

$$
\operatorname { S y m } ( T ) : = { \frac { 1 } { k ! } } \sum _ { \pi \in S _ { k } } \pi \cdot T .
$$

Lemma B.3 (Diagonal evaluation sees only the input-symmetric part). For every

$$
T \in \mathbb { R } ^ { p } \otimes ( \mathbb { R } ^ { d } ) ^ { \otimes k }
$$

and every $\boldsymbol { x } \in \mathbb { R } ^ { d } ,$

$$
T [ x , \dots , x ] = \operatorname { S y m } ( T ) [ x , \dots , x ] .
$$

Proof. For any $\pi \in S _ { k }$ , all input slots of $\pi \cdot T$ receive the same vector x. Hence

$$
( \pi \cdot T ) [ x , \ldots , x ] = T [ x , \ldots , x ] .
$$

Averaging over all $\pi \in S _ { k }$ gives the claim.

Lemma B.4 (Quadratic diagonal evaluation). For every $A \in \mathbb { R } ^ { d \times d }$ and every $\boldsymbol { x } \in \mathbb { R } ^ { d } ,$

$$
x ^ { \top } A x = x ^ { \top } { \frac { A + A ^ { \top } } { 2 } } x .
$$

Proof. Let

$$
S = { \frac { 1 } { 2 } } ( A - A ^ { \top } ) .
$$

Then $S ^ { \top } = - S$ . Since $x ^ { \top } S x$ is a scalar,

$$
x ^ { \top } S x = ( x ^ { \top } S x ) ^ { \top } = x ^ { \top } S ^ { \top } x = - x ^ { \top } S x .
$$

Therefore $x ^ { \top } S x = 0$ . The result follows by writing

$$
A = { \frac { 1 } { 2 } } ( A + A ^ { \top } ) + S .
$$

Derivative tensors. For an analytic scalar function, the degree-k Taylor coefficient tensor is equivalently the k-th derivative tensor up to the factor k!. In particular, for a homogeneous tensor polynomial

$$
f _ { T } ( x ) = T [ x , \ldots , x ] ,
$$

we have

$$
D ^ { k } f _ { T } ( 0 ) = k ! \operatorname { S y m } ( T ) .
$$

Thus the derivative tensor also records only the input-symmetric part of the raw coefficient tensor.   
This is the differential version of Lemma B.3.

## C Expressivity proofs

## C.1 Scalar-output reduction

Lemma C.1 (Output projection cannot increase channel rank). For every $T \in \mathcal { T } _ { k }$ and every $\eta \in$ (R<sup>p</sup>)<sup>∗</sup>,

$$
r _ { \mathrm { t i e d } } \big ( ( \eta \otimes I ) T \big ) \leq r _ { \mathrm { t i e d } } ( T ) , \qquad r _ { \mathrm { u n t i e d } } \big ( ( \eta \otimes I ) T \big ) \leq r _ { \mathrm { u n t i e d } } ( T ) .
$$

Proof. Suppose first that T has a tied representation with m channels:

$$
T = \sum _ { a = 1 } ^ { m } w _ { a } \otimes u _ { a } ^ { \otimes k } .
$$

Applying η to the output mode gives

$$
( \eta \otimes I ) T = \sum _ { a = 1 } ^ { m } \eta ( w _ { a } ) u _ { a } ^ { \otimes k } .
$$

This is a scalar tied representation with at most m atoms; atoms with $\eta ( w _ { a } ) = 0$ simply drop out.   
Minimizing over tied representations of T gives the first inequality.

The detector–value case is identical. If

$$
T = \sum _ { a = 1 } ^ { m } w _ { a } \otimes \mathrm { S y m } \big ( u _ { a } ^ { \otimes ( k - 1 ) } \otimes v _ { a } \big ) ,
$$

then

$$
( \eta \otimes I ) T = \sum _ { a = 1 } ^ { m } \eta ( w _ { a } ) \mathrm { { \ S y m } } \big ( u _ { a } ^ { \otimes ( k - 1 ) } \otimes v _ { a } \big ) ,
$$

which is a scalar detector–value representation with at most m atoms.

For the special embedding of a scalar target into one output coordinate, the ranks are exactly preserved. Corollary C.2 (Scalar separations embed into vector targets). Let $S \in \mathrm { { S y m } } ^ { k } ( \mathbb { R } ^ { d } )$ , and let

$$
T = e _ { o } \otimes S \in \mathbb { R } ^ { p } \otimes \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } ) .
$$

Then

$$
r _ { \mathrm { t i e d } } ( T ) = r _ { \mathrm { t i e d } } ( S ) , \qquad r _ { \mathrm { u n t i e d } } ( T ) = r _ { \mathrm { u n t i e d } } ( S ) .
$$

Consequently, any scalar tied–detector–value separation transfers to the vector-valued CP-MLP setting.

Proof. The lower bounds follow from Lemma C.1 by taking $\eta ( y ) = y _ { o } ,$ , since

$$
( \eta \otimes I ) T = S .
$$

Thus

$$
r _ { \mathrm { t i e d } } ( S ) \leq r _ { \mathrm { t i e d } } ( T ) , \qquad r _ { \mathrm { u n t i e d } } ( S ) \leq r _ { \mathrm { u n t i e d } } ( T ) .
$$

For the reverse inequalities, take any scalar representation of S and multiply each atom by the output basis vector $e _ { o } .$ This gives a vector-valued representation of $T = e _ { o } \otimes S$ with the same number of channels. Hence the ranks are equal. □

## C.2 Bilinear atom-rank proofs

ProofofTheorem 4.5. We prove the tied and untied rank bounds separately.

Tied blocks. A tied bilinear block of width m represents a matrix of the form

$$
M = \sum _ { a = 1 } ^ { m } \alpha _ { a } u _ { a } u _ { a } ^ { \top } .
$$

Each term $\alpha _ { a } u _ { a } u _ { a } ^ { \top }$ has rank at most one. Therefore

$$
\operatorname { r a n k } ( M ) \leq \sum _ { a = 1 } ^ { m } \operatorname { r a n k } ( \alpha _ { a } u _ { a } u _ { a } ^ { \top } ) \leq m .
$$

Since $M \in \mathbb { R } ^ { d \times d }$ , also

$$
\operatorname { r a n k } ( M ) \leq d .
$$

Thus a tied block satisfies

$$
\operatorname { r a n k } ( M ) \leq \operatorname* { m i n } ( m , d ) .
$$

This bound is tight. Let $r = \operatorname* { m i n } ( m , d )$ . Choosing

$$
u _ { a } = e _ { a } , \qquad \alpha _ { a } = 1 , \qquad a = 1 , \ldots , r ,
$$

gives

$$
\boldsymbol { M } = \sum _ { a = 1 } ^ { r } \boldsymbol { e } _ { a } \boldsymbol { e } _ { a } ^ { \intercal } ,
$$

which has rank $r = \operatorname* { m i n } ( m , d )$

Untied blocks. An untied bilinear block of width m represents a matrix of the form

$$
M = \sum _ { a = 1 } ^ { m } M _ { a } , \qquad M _ { a } = \frac { \alpha _ { a } } { 2 } \left( { u } _ { a } { v } _ { a } ^ { \top } + { v } _ { a } { u } _ { a } ^ { \top } \right) .
$$

For any $z \in \mathbb { R } ^ { d }$

$$
M _ { a } z = \frac { \alpha _ { a } } { 2 } \left( u _ { a } ( v _ { a } ^ { \top } z ) + v _ { a } ( u _ { a } ^ { \top } z ) \right) \in \mathrm { s p a n } \{ u _ { a } , v _ { a } \} .
$$

Therefore

$$
\mathrm { r a n k } ( M _ { a } ) \leq 2 .
$$

Using subadditivity of rank,

$$
\operatorname { r a n k } ( M ) \leq \sum _ { a = 1 } ^ { m } \operatorname { r a n k } ( M _ { a } ) \leq 2 m .
$$

Since $M \in \mathbb { R } ^ { d \times d }$

$$
\operatorname { r a n k } ( M ) \leq d .
$$

Hence

$$
\operatorname { r a n k } ( M ) \leq \operatorname* { m i n } ( 2 m , d ) .
$$

This bound is tight. Let $r = \operatorname* { m i n } ( 2 m , d )$

If $r = 2 s$ is even, choose

$$
u _ { a } = e _ { 2 a - 1 } , \qquad v _ { a } = e _ { 2 a } , \qquad \alpha _ { a } = 1 , \qquad a = 1 , \ldots , s .
$$

Then

$$
M = \frac { 1 } { 2 } \sum _ { a = 1 } ^ { s } \left( e _ { 2 a - 1 } e _ { 2 a } ^ { \top } + e _ { 2 a } e _ { 2 a - 1 } ^ { \top } \right) .
$$

This is a direct sum of s independent $2 \times 2$ off-diagonal blocks, each of rank two. Therefore rank $\mathbf { \partial } ( M ) = 2 s = r$

If $r = 2 s + 1$ is odd, use the same s off-diagonal blocks and add one diagonal atom by taking

$$
u _ { s + 1 } = v _ { s + 1 } = e _ { 2 s + 1 } , \qquad \alpha _ { s + 1 } = 1 .
$$

This adds one independent rank-one block, so the resulting matrix has rank $2 s + 1 = r$ . Thus the untied rank bound is tight. □

Proof of Theorem 4.6. We prove the tied and untied atom ranks separately.

Tied atom rank. By Theorem 4.5, any tied representation of M with m atoms satisfies

$$
\operatorname { r a n k } ( M ) \leq m .
$$

Therefore

$$
r _ { \mathrm { t i e d } } ( M ) \geq \mathrm { r a n k } ( M ) .
$$

For the reverse inequality, use the spectral decomposition of the symmetric matrix M:

$$
M = \sum _ { i = 1 } ^ { n _ { + } ( M ) } \lambda _ { i } q _ { i } q _ { i } ^ { \top } - \sum _ { j = 1 } ^ { n _ { - } ( M ) } \mu _ { j } p _ { j } p _ { j } ^ { \top } , \qquad \lambda _ { i } , \mu _ { j } > 0 .
$$

This is a tied representation using one signed rank-one atom for each nonzero eigendirection. Hence

$$
r _ { \mathrm { t i e d } } ( M ) \leq n _ { + } ( M ) + n _ { - } ( M ) = \mathrm { r a n k } ( M ) .
$$

Combining the two inequalities gives

$$
r _ { \mathrm { t i e d } } ( M ) = \mathrm { r a n k } ( M ) = n _ { + } ( M ) + n _ { - } ( M ) .
$$

Untied atom rank: lower bound. An untied atom satisfies the identity

$$
\frac { \alpha } { 2 } \left( u v ^ { \top } + v u ^ { \top } \right) = \frac { \alpha } { 4 } ( u + v ) ( u + v ) ^ { \top } - \frac { \alpha } { 4 } ( u - v ) ( u - v ) ^ { \top } .
$$

Therefore any width-m untied representation can be written as

$$
M = A A ^ { \top } - B B ^ { \top }
$$

for some $A , B \in \mathbb { R } ^ { d \times m }$ , after absorbing scalar signs and magnitudes into the columns of A and B. We claim that this implies

$$
n _ { + } ( M ) \leq m , \qquad n _ { - } ( M ) \leq m .
$$

Indeed, on ker $( A ^ { \top } )$ ,

$$
x ^ { \top } M x = x ^ { \top } ( A A ^ { \top } - B B ^ { \top } ) x = \| A ^ { \top } x \| ^ { 2 } - \| B ^ { \top } x \| ^ { 2 } = - \| B ^ { \top } x \| ^ { 2 } \leq 0 .
$$

If $n _ { + } ( M ) > \mathrm { r a n k } ( A )$ , a subspace P of dimension $n _ { + } ( M )$ on which M is positive definite would satisfy dim $\imath ( P \cap$ ker $A ^ { \top } ) \geq n _ { + } ( M ) - \operatorname { r a n k } ( A ) > 0$ , contradicting the inequality above; hence $n _ { + } ( M ) \leq \mathrm { r a n k } ( A ) \leq m$ . Similarly, on $\ker ( B ^ { \top } )$

$$
x ^ { \top } M x = \| A ^ { \top } x \| ^ { 2 } \geq 0 ,
$$

so M cannot be negative definite on a subspace of dimension larger than rank $( B ) \leq m$ . Hence

$$
n _ { - } ( M ) \leq m .
$$

Therefore every untied representation satisfies

$$
m \geq \operatorname* { m a x } \{ n _ { + } ( M ) , n _ { - } ( M ) \} .
$$

So

$$
r _ { \mathrm { u n t i e d } } ( M ) \geq \operatorname* { m a x } \{ n _ { + } ( M ) , n _ { - } ( M ) \} .
$$

Untied atom rank: upper bound. It remains to construct an untied representation with exactly max $\{ n _ { + } ( M ) , n _ { - } ( M ) \bar  \}$ atoms. Use the spectral decomposition

$$
M = \sum _ { i = 1 } ^ { n _ { + } ( M ) } \lambda _ { i } q _ { i } q _ { i } ^ { \top } - \sum _ { j = 1 } ^ { n _ { - } ( M ) } \mu _ { j } p _ { j } p _ { j } ^ { \top } , \qquad \lambda _ { i } , \mu _ { j } > 0 .
$$

Pair positive and negative eigendirections as much as possible. For a paired block

$$
\lambda q q ^ { \top } - \mu p p ^ { \top } , \qquad \lambda , \mu > 0 ,
$$

define

$$
u = \sqrt { \lambda } q + \sqrt { \mu } p , \qquad v = \sqrt { \lambda } q - \sqrt { \mu } p .
$$

Then

$$
\frac { 1 } { 2 } \left( u v ^ { \top } + v u ^ { \top } \right) = \lambda q q ^ { \top } - \mu p p ^ { \top } .
$$

Thus one untied atom represents one positive and one negative eigendirection together.

If any positive eigendirections remain unpaired, represent each $\lambda q q ^ { \top }$ by the tied special case

$$
u = v = { \sqrt { \lambda } } q .
$$

If any negative eigendirections remain unpaired, represent each $- \mu p p ^ { \intercal }$ by

$$
u = \sqrt { \mu } p , \qquad v = - \sqrt { \mu } p .
$$

Therefore the total number of atoms used is

$$
\operatorname* { m a x } \{ n _ { + } ( M ) , n _ { - } ( M ) \} .
$$

This proves

$$
r _ { \mathrm { u n t i e d } } ( M ) \leq \operatorname* { m a x } \{ n _ { + } ( M ) , n _ { - } ( M ) \} .
$$

Combining the lower and upper bounds gives

$$
r _ { \mathrm { u n t i e d } } ( M ) = \operatorname* { m a x } \{ n _ { + } ( M ) , n _ { - } ( M ) \} .
$$

## C.3 Exact rank, border rank, and bounded approximation

Fix $k \geq 2$ and $\textsf { f } \in$ {tied, untied}. All parameters and closures are real. We use the coefficient-tensor Frobenius norm; equivalent norms give the same closure. Define

$$
\mathcal { C } _ { \mathsf { f } , m } ^ { ( k ) } = \{ T : r _ { \mathsf { f } } ( T ) \leq m \} , \qquad \underline { { r } } _ { \mathsf { f } } ( T ) = \operatorname* { m i n } \{ m : T \in \overline { { \mathcal { C } _ { \mathsf { f } , m } ^ { ( k ) } } } \} ,
$$

and

$$
e _ { \mathsf { f } , m } ( T ) = \operatorname* { i n f } _ { S \in { \mathcal C } _ { \mathsf { f } , m } ^ { ( k ) } } \| T - S \| _ { F } = \mathrm { d i s t } ( T , { \mathcal C } _ { \mathsf { f } , m } ^ { ( k ) } ) .
$$

The error is positive for $m < \underline { { r } } _ { \mathsf { f } } ( T )$ , zero but unattained for $\underline { { r } } _ { \mathsf { f } } ( T ) \leq m < r _ { \mathsf { f } } ( T )$ , and zero and attained for $m \geq r _ { \mathsf { f } } ( T )$ . A positive infimum alone does not guarantee a best approximation [de Silva and Lim, 2008].

Coefficient sizes. For output $w _ { a } \in \mathbb { R } ^ { p }$ , define

$$
A _ { a } ^ { t } = \| w _ { a } \| \| u _ { a } \| ^ { k } , \qquad A _ { a } ^ { u } = \| w _ { a } \| \| u _ { a } \| ^ { k - 1 } \| v _ { a } \| .
$$

These sizes are invariant under factor rescalings that preserve the raw atom; for scalar output, $\| w _ { a } \| = | w _ { a } | .$ . Let $\mathscr { C } _ { \boldsymbol { \mathsf { f } } , m } ^ { ( k ) } ( B )$ contain tensors with a representation using at most m atoms and total size $\textstyle \sum _ { a } A _ { a } ^ { \mathsf { f } } \leq B .$

Proposition C.3 (Bounded compactness and forced size divergence). For finite $B \geq 0 ,$ , both classes $\mathscr { C } _ { \boldsymbol { \mathsf { f } } , m } ^ { ( k ) } ( B )$ are compact. Thus

$$
\delta _ { \mathsf { f } , m } ( B ; T ) = \operatorname* { m i n } _ { S \in { \mathcal { C } _ { \mathsf { f } , m } ^ { ( k ) } ( B ) } } \| T - S \| _ { F }
$$

exists, is positive when $m < \mathfrak { r } _ { \mathfrak { f } } ( T )$ , and converges to $e _ { \mathsf { f } , m } ( T )$ as $B  \infty$ . Every fixed-width sequence converging to a target outside the exact class has $A _ { \mathrm { m a x } }  \infty$

Proof. Normalize the input directions and absorb their norms into the output factors, padding zero atoms as needed. The tied parameter set becomes

$$
\begin{array} { r } { \{ ( \widetilde { w } _ { 1 } , \ldots , \widetilde { w } _ { m } ) : \sum _ { a } \| \widetilde { w } _ { a } \| \le B \} \times ( \mathbb { S } ^ { d - 1 } ) ^ { m } . } \end{array}
$$

It is compact, as is its image under the continuous tensor-sum map. The untied case uses two unit directions per atom. Compactness gives the positive minimum outside the class. The budgeted classes increase to the unrestricted class, so their distances decrease to $e _ { \mathsf { f } , m } ( T )$ . If $A _ { \mathrm { m a x } }$ did not tend to infinity, a subsequence would have total size at most $m C$ for some finite $C ;$ compactness would place its limit in the exact class, a contradiction. □

Proposition C.4 (Separation under a common coefficient-size budget). $I f r _ { \mathrm { u n t i e d } } ( T ) \leq m < r _ { \mathrm { t i e d } } ( T )$ there is afinite $B _ { 0 }$ such that every $B \geq B _ { 0 }$ permits exact untied width-m representation, whereas $\delta _ { \mathrm { t i e d } , m } ( \tilde { B ; } T ) > 0 .$ . As $B  \infty$ , this tied distance has a positive limit $i f m < \underline { { r } } _ { \mathrm { t i e d } } ( T )$ and tends to zero otherwise.

Proof. Take $B _ { 0 }$ to be the total size of an exact untied representation. Apply Proposition C.3 and the definition of border rank to the tied class. □

The monomial limit. For independent linear forms $\ell , h$

$$
P _ { \varepsilon } = \frac { ( \ell + \varepsilon h ) ^ { k } - ( \ell - \varepsilon h ) ^ { k } } { 2 k \varepsilon } \longrightarrow \ell ^ { k - 1 } h .\tag{1}
$$

Hence the tied border rank is at most two. It is at least two because the first-derivative span of $\ell ^ { k - 1 } h$ has dimension two, while a pure power has dimension at most one, and this matrix-rank bound survives closure. Thus $\underline { { r } } _ { \mathrm { t i e d } } ( F _ { k } ) = 2$ [Comas and Seiguer, 2011]. For $k \geq 3 .$ , binomial expansion gives error $O ( \varepsilon ^ { 2 } )$ and $A _ { \mathrm { m a x } } = \Theta ( | \varepsilon | ^ { - 1 } )$ ), hence squared error $O ( A _ { \operatorname* { m a x } } ^ { - 4 } )$ along this path. This gives an upper bound; optimality is not established. For $k = 2$ , the identity is exact for nonzero ε.

## C.4 Closed quadratic classes

Proposition C.5 (Scalar quadratic closedness). For scalar degree-two targets and every width $m ,$ both tied and untied classes are closed. Their exact and real border ranks coincide.

Proof. By Theorem 4.6, the tied class consists of symmetric matrices of rank at most $m ,$ , a closed condition defined by vanishing $( m + 1 )$ )-minors. The untied class is

$$
\{ M = M ^ { \top } : n _ { + } ( M ) \leq m , n _ { - } ( M ) \leq m \} ,
$$

which is closed by continuity of ordered eigenvalues. Distance from a target outside either class is positive and attained: a minimizing sequence can be restricted to a bounded ball, whose intersection with the class is compact. □

For $x \sim \mathcal { N } ( 0 , I _ { d } )$ and symmetric $H ,$

$$
\mathbb { E } [ ( x ^ { \top } H x ) ^ { 2 } ] = 2 \| H \| _ { F } ^ { 2 } + ( \mathrm { t r } H ) ^ { 2 } .
$$

A positive coefficient-norm gap therefore implies a positive optimal population MSE. An observed finite-sample residual can also reflect sampling and optimization. For $\hat { Q } = \mathrm { d i a g } ( 1 , 1 , - 1 , - 1 )$ , tied rank-m spectral truncation gives relative squared Frobenius error $( 4 - m ) / 4$ for $0 \leq m \leq 4 ;$ the untied error is $1 / 2$ at $m = 1$ and zero at $m = 2$

## C.5 Universal comparison and independent interaction blocks

Proposition C.6 (Universal exact- and border-rank comparison). For every degree-k coefficient tensor T, with $k \geq 2$

$$
r _ { \mathrm { u n t i e d } } ( T ) \leq r _ { \mathrm { t i e d } } ( T ) \leq k r _ { \mathrm { u n t i e d } } ( T ) , \qquad \underline { { r } } _ { \mathrm { u n t i e d } } ( T ) \leq \underline { { r } } _ { \mathrm { t i e d } } ( T ) \leq 2 \underline { { r } } _ { \mathrm { u n t i e d } } ( T ) .
$$

Both upper constants are attained by $F _ { k } .$

Proof. Containment gives the lower bounds, including after closure. For independent $u , v ,$ a change of coordinates maps $( u ^ { \top } x ) ^ { k - 1 } ( v ^ { \top } x )$ to $F _ { k }$ , which has a real k-power decomposition by Theorem 4.10; collinear directions require at most one power. Applying these decompositions to each atom, with its output vector, proves the exact upper bound. Equation (1) puts each untied atom in the closure of the tied width-two class. Summing and taking closure gives the border upper bound. The ranks of $F _ { k }$ establish sharpness. □

Theorem C.7 (Independent interaction blocks). Let $k \geq 2 , s \geq 1 , d \geq 2 s _ { \ast }$ , and $c _ { i } \in \mathbb { R } \setminus \{ 0 \}$ . For linearly independent real forms $\ell _ { 1 } , h _ { 1 } , \ldots , \ell _ { s } , h _ { s } ,$ , define

$$
F _ { k , s } ( x ) = \sum _ { i = 1 } ^ { s } c _ { i } \ell _ { i } ( x ) ^ { k - 1 } h _ { i } ( x ) .
$$

Then

$$
\big ( r _ { \mathrm { u n t i e d } } , r _ { \mathrm { t i e d } } , \underline { { r } } _ { \mathrm { u n t i e d } } , \underline { { r } } _ { \mathrm { t i e d } } \big ) ( F _ { k , s } ) = ( s , k s , s , 2 s ) .
$$

Proof. Use the $2 s$ independent forms as coordinates. The first derivatives span the $2 s$ distinct monomials $x _ { 2 i - 1 } ^ { k - 2 } x _ { 2 i }$ and $x _ { 2 i - 1 } ^ { k - 1 }$ , including when $k = 2$ . Each tied atom contributes at most one derivative direction and each untied atom at most two. Since these matrix-rank bounds survive closure,

$$
\begin{array} { r } { \underline { { r } } _ { \mathrm { t i e d } } \geq 2 s , \qquad \underline { { r } } _ { \mathrm { u n t i e d } } \geq s . } \end{array}
$$

One untied atom per summand gives r<sub>untied</sub> $\leq s ;$ Equation (1) gives $\underline { { r } } _ { \mathrm { t i e d } } \leq 2 s :$ ; and real k-power decompositions give $r _ { \mathrm { t i e d } } \leq k s$

For $k \geq 3 .$ , absorb $c _ { i }$ into $x _ { 2 i }$ and apply the additivity theorem for pairwise coprime monomials [Carlini et al., 2012, Theorem 3.2]: the complex Waring rank is ks, so the real rank is at least ks. Extra coordinates cannot lower rank, since restricting them to zero preserves a decomposition. For $k = 2$ , each block has one positive and one negative eigenvalue, and Theorem 4.6 gives tied rank $2 s = k s$ □

Thus the tied approximation gap is positive at $s \leq m < 2 s$ , while the untied family is exact. At $2 s \leq m < k s$ , the tied infimum is zero but unattained.

## C.6 Necessary width bounds for analytic channels

Proposition C.8 (Degree-wise necessity). Let g be analytic near zero and let F be represented near zero by a bias-free width-m CP-MLP infamily f. Then $\dot { F } ( 0 ) = 0 ;$ , inactive Taylor components vanish, and

$$
m \geq \operatorname* { s u p } _ { k \in { \cal S } ( g ) } r _ { \mathsf { f } } ( F ^ { ( k ) } ) .
$$

Proof. Equality near zero implies equality of Taylor coefficients. Each active degree is a sum of at most m atoms, with its nonzero gate coefficient absorbed into the output factors; inactive components vanish. These conditions are necessary, but parameters shared across degrees prevent choosing the decompositions independently. □

Proposition C.9 (Degree-wise rank one with SiLU-channel width two). Thefunction

$$
G ( x _ { 1 } , x _ { 2 } ) = \big [ \mathrm { S i L U } ( x _ { 1 } ) + \mathrm { S i L U } ( 2 x _ { 1 } ) \big ] x _ { 2 }
$$

has untied atom rank one at every active Taylor degree, but its exact untied SiLU-channel width is two.

Proof. The displayed representation uses two channels, and each active component is a nonzero multiple of $x _ { 1 } ^ { k - 1 } x _ { 2 }$ . To exclude width one, expand

$$
\begin{array} { r } { \mathrm { S i L U } ( z ) = \frac { 1 } { 2 } z + \frac { 1 } { 4 } z ^ { 2 } - \frac { 1 } { 4 8 } z ^ { 4 } + O ( z ^ { 6 } ) . } \end{array}
$$

For $A \mathrm { S i L U } ( s x _ { 1 } ) x _ { 2 }$ , the coefficients $b _ { 2 } , b _ { 3 } , b _ { 5 }$ of $x _ { 1 } x _ { 2 } , x _ { 1 } ^ { 2 } x _ { 2 } , x _ { 1 } ^ { 4 } x _ { 2 }$ satisfy

$$
b _ { 2 } = A s / 2 , \qquad b _ { 3 } = A s ^ { 2 } / 4 , \qquad b _ { 5 } = - A s ^ { 4 } / 4 8 , \qquad \frac { b _ { 2 } ^ { 2 } b _ { 5 } } { b _ { 3 } ^ { 3 } } = - \frac { 1 } { 3 }
$$

when $b _ { 3 } \neq 0 .$ . For $G ,$ they are $3 / 2 , 5 / 4 , - 1 7 / 4 8$ , giving −51/125 ̸= −1/3.

Any width-one representation matching the quadratic term must have detector and value forms proportional to $x _ { 1 } , x _ { 2 }$ in some order, by unique factorization. The cubic term fixes the detector along $x _ { 1 }$ and the value along $x _ { 2 }$ , so the invariant excludes all width-one representations. □

The obstruction is the gate’s fixed coefficient ratios across degrees. Small held-out function error alone does not bound every Taylor coefficient.

## D Experimental details

This section documents the experimental methodology in sufficient detail for an independent reimplementation. Code that reproduces every figure and table is available at https://github.com/ rarefin/CP-MLPs.

## D.1 Computational requirements

All experiments run on a standard CPU; no GPU or specialized hardware is required. The experiments of Section 5 and Appendices D.7–D.8 each complete in about one minute, about five minutes in total on a single CPU core. The additional experiments of Appendices D.9–D.13 take about two to three hours in total on one core. Memory usage stays below 1 GB throughout.

The experiments use standard scientific Python: numpy for linear algebra, scipy.optimize.minimize (L-BFGS-B method) for nonlinear least-squares fitting, and matplotlib for figure generation. The experiments of Appendices D.10, D.11, and D.13 also use PyTorch on the CPU, for automatic differentiation and for Adam.

## D.2 Data generation

The experiments fit two kinds of targets: polynomial coefficient tensors (in coefficient space) and scalarfunctions ofGaussian inputs (in function space). We describe both.

Coefficient-space targets. A degree-k polynomial in d variables corresponds to a coefficient vector $T \in \mathbb { R } ^ { D }$ in the basis of degree-k monomials, where $D = { \binom { d + k - 1 } { k } }$ . For $d = 2$ we use the basis indexed by the exponent of $x _ { 2 } ,$ , so index j corresponds to the monomial $x _ { 1 } ^ { k - j } x _ { 2 } ^ { j }$ . The role-asymmetric witness $F _ { k } = x _ { 1 } ^ { k - 1 } x _ { 2 }$ of Theorem 4.10 therefore has a single nonzero entry at index $j = 1 ;$ the pure-power target $x _ { 1 } ^ { k }$ has its single nonzero entry at $j = 0$

The model’s prediction in coefficient space is the coefficient vector of its parameterized polynomial, computed analytically from the architecture’s atom form (Section D.3). The loss is the squared $\ell _ { 2 }$ distance between the predicted coefficient vector and the target.

Function-space targets. For experiments using actual Gaussian inputs (Sections 5.3 and 5.4), we sample inputs $X \in \overline { { \mathbb { R } } } ^ { N \times d }$ with each row drawn i.i.d. from $\mathcal { N } ( 0 , I _ { d } )$ . The sample size N is given in Table 2. Targets are evaluated pointwise: for $y ( x ) = x _ { 1 } ^ { k - 1 } x _ { 2 }$ , we set $y _ { i } = X _ { i , 1 } ^ { k - 1 } X _ { i , 2 } ;$ for the indefinite quadratic experiment, $y _ { i } = X _ { i } ^ { \top } Q X _ { i }$ <sub>i</sub> with $Q = \mathrm { d i a g } ( { \bf 1 } ^ { n _ { + } } , - { \bf 1 } ^ { n _ { - } } , { \bf 0 } ^ { d - n _ { + } - n _ { - } } )$ . The loss is the mean squared error $\textstyle { \frac { 1 } { N } } \sum _ { i } ( { \hat { y } } _ { i } - y _ { i } ) ^ { 2 }$

Random seeds. Each experiment uses random seeds that are deterministic functions of configuration values. For example, the quadratic experiment uses seed $= \mathrm { ~ s ~ } * \ 1 3 7 \ + \ \mathrm { n \mathrm { \mathrm { \mathrm { . } \mathrm { p l u s } \ * \ 1 7 \ + \ \mathrm { n \mathrm { . } } } } }$ \_minus for seed index s and signature $( n _ { + } , n _ { - } ) ;$ the width-sweep experiment uses seed $= \textbf { s } * \ 3 7 \ + \textbf { k } * \ 1 1$ , shared across the widths of a sweep. Fixed seeds make every run reproducible with the same library versions.

## D.3 Model implementations

Each model is parameterized by a flat parameter vector $\boldsymbol { \theta } \in \mathbb { R } ^ { P }$ that the optimizer treats as the only variable. The total parameter count $P$ depends on the architecture and the width $m$

Function-space architectures. Given input $X \in \mathbb { R } ^ { N \times d }$ and width $m _ { : }$ all architectures follow the CP-MLP structure of Definition 3.1: project inputs through one or two linear maps, apply an elementwise nonlinearity, multiply per channel, and combine via scalar output coefficients $c \in \mathbb { R } ^ { m }$ (the scalar specialization of the output direction $w _ { a } )$

For tied polynomial atoms with detector matrix $U \in \mathbb { R } ^ { m \times d }$ and coefficients c:

$$
\boldsymbol { \hat { y } _ { i } } = \sum _ { a = 1 } ^ { m } c _ { a } \left( \boldsymbol { u } _ { a } ^ { \top } \boldsymbol { x } _ { i } \right) ^ { k } , \qquad \boldsymbol { \theta } = ( \operatorname { v e c } ( \boldsymbol { U } ) , \boldsymbol { c } ) \in \mathbb { R } ^ { m ( d + 1 ) } .
$$

For untied polynomial atoms with detector U, value $V \in \mathbb { R } ^ { m \times d }$ , and coefficients c:

$$
\hat { y } _ { i } = \sum _ { a = 1 } ^ { m } c _ { a } \big ( u _ { a } ^ { \top } x _ { i } \big ) ^ { k - 1 } ( v _ { a } ^ { \top } x _ { i } ) , \qquad \theta = \big ( \mathrm { v e c } ( U ) , \mathrm { v e c } ( V ) , c \big ) \in \mathbb { R } ^ { m ( 2 d + 1 ) } .
$$

For the four gated architectures in Section 5.4, replace the polynomial atoms with their gated analogues from Table 1:

$$
\begin{array} { r l r } { \mathrm { t i e d ~ R e L U ^ { 2 } : } \quad \hat { y } = \displaystyle \sum _ { a } c _ { a } \mathrm { R e L U } ( u _ { a } ^ { \top } x ) ^ { 2 } , } & { \mathrm { ~ u n t i e d ~ R e G L U : } \quad \hat { y } = \displaystyle \sum _ { a } c _ { a } \mathrm { R e L U } ( u _ { a } ^ { \top } x ) ( v _ { a } ^ { \top } x ) , } \\ { \mathrm { t i e d ~ S w i G L U : } \quad \hat { y } = \displaystyle \sum _ { a } c _ { a } \mathrm { S i L U } ( u _ { a } ^ { \top } x ) ( u _ { a } ^ { \top } x ) , } & { \mathrm { ~ u n t i e d ~ S w i G L U : } \quad \hat { y } = \displaystyle \sum _ { a } c _ { a } \mathrm { S i L U } ( u _ { a } ^ { \top } x ) ( v _ { a } ^ { \top } x ) . } \end{array}
$$

with $\mathrm { S i L U } ( z ) = z \sigma ( z )$ and $\sigma ( z ) = 1 / ( 1 + e ^ { - z } )$ . Tied gated architectures have parameter count $m ( d + 1 ) ;$ untied gated architectures have $n ( 2 d + 1 )$ . The matched-parameter control of Section 5.4 pairs untied width m against tied width $2 m ,$ equalizing input-projection parameters at 2md.

Coefficient-space implementations. For coefficient-space experiments we compute the coefficient vector of each parameterized polynomial directly, avoiding sampling. For tied atoms, the multinomial theorem gives

$$
( u ^ { \top } x ) ^ { k } = \sum _ { \mid \alpha \mid = k } { \binom { k } { \alpha } } u ^ { \alpha } x ^ { \alpha } , \qquad { \binom { k } { \alpha } } = { \frac { k ! } { \alpha _ { 1 } ! \cdot \cdot \cdot \alpha _ { d } ! } } .
$$

For untied detector-value atoms,

$$
( u ^ { \top } x ) ^ { k - 1 } ( v ^ { \top } x ) = \left( \sum _ { | \beta | = k - 1 } { \binom { k - 1 } { \beta } } u ^ { \beta } x ^ { \beta } \right) \sum _ { j = 1 } ^ { d } v _ { j } x _ { j } ,
$$

which expands into a degree-k multi-index sum. The total predicted coefficient vector is $\textstyle \sum _ { a } c _ { a }$ times the per-atom coefficient vector, and the loss is the squared $\ell _ { 2 }$ distance from the target.

## D.4 Optimization protocol

The experiments of Section 5 and Appendices D.7–D.8 minimize the squared loss using scipy.optimize.minimize with method L-BFGS-B. The iteration limit and tolerances are: indefinite quadratics 3000, $\mathtt { : t o l } = 1 0 ^ { - 2 2 } , \mathtt { g t o l } = 1 0 ^ { - 1 5 }$ ; width sweep 2000, $1 0 ^ { - 2 2 } , 1 0 ^ { - 1 8 } ;$ pure-vsmixed 5000, $1 0 ^ { - 2 0 } , 1 0 ^ { - 1 5 } ;$ gated MLPs $1 5 0 0 , 1 0 ^ { - 1 8 } , \dot { 1 } 0 ^ { - 1 2 }$ ; matched control 1500, $1 0 ^ { - 2 0 ^ { \circ } } , 1 0 ^ { - 1 2 }$ scope boundary 800, $1 0 ^ { - 2 2 } , 1 0 ^ { - 1 5 }$ . Appendices $\mathrm { D } . 9 \mathrm { - D } . 1 3$ describe their own settings. For each random seed, we run multiple restarts with parameter initialization $\theta _ { 0 } \sim \mathcal { N } ( 0 , 0 . 2 5 )$ (applied independently to each entry of θ). We record the best-of-restarts loss per seed and aggregate (median, min, max) across seeds.

## D.5 Per-experiment configurations

Table 2: Hyperparameters for each experiment.
<table><tr><td>Experiment</td><td>Section</td><td>Input dim d</td><td>Sample size</td><td>Seeds</td><td>Restarts</td></tr><tr><td>Indefinite quadratic targets</td><td>§5.1</td><td>8</td><td>256</td><td>3</td><td>8</td></tr><tr><td>Role-asymmetric monomial targets</td><td>§5.2</td><td>2</td><td>coef. space</td><td>2</td><td>5</td></tr><tr><td>Pure-vs-mixed reversal</td><td>§5.3</td><td>4</td><td>256</td><td>10</td><td>5</td></tr><tr><td>Gated MLPs</td><td>§5.4</td><td>4</td><td>200</td><td>3</td><td>3</td></tr><tr><td>Matched-parameter control (Gated MLPs)</td><td>§5.4</td><td>4</td><td>200</td><td>3</td><td>3</td></tr><tr><td>Scope boundary</td><td>§D.7</td><td>3</td><td>coef. space</td><td>2</td><td>4*</td></tr></table>

<sup>∗</sup>Twelve restarts for the untied model at m = 1.

For coefficient-space experiments, fits operate directly on the $\binom { d + k - 1 } { k }$ monomial coefficients of the target polynomial, eliminating sampling and function-evaluation noise. For experiments with finite samples, inputs are drawn from $\mathcal { N } ( \bar { 0 } , I _ { d } )$ once per seed and held fixed across restarts within that seed.

![](images/36de686ad5952becc8f9b9859dc371b41e4234afbe018cb889ce76f194ef0fc2.jpg)  
Figure 6: Scope boundary at $k = 3 .$ . Untied atoms are rank one on the role-asymmetric witness $F _ { d v } = x _ { 1 } ^ { k - 1 } x _ { 2 }$ but not on the square-free product $F _ { s f } = x _ { 1 } x _ { 2 } \cdot \cdot \cdot x _ { k }$ ; the multi-leg oracle is rank one on both.

Table 3: Scope boundary at $k = 3 .$ . Median best-of-restarts MSE over two seeds.
<table><tr><td>Target</td><td>Architecture</td><td> $m = 1$ </td><td> $m = 2$ </td><td> $m = 3$ </td><td> $m = 4$ </td></tr><tr><td> $x _ { 1 } ^ { k - 1 } x _ { 2 } ~ \mathrm { ( r o l e \mathrm { - } a s y m . ) }$ </td><td>tied untied (det.-val.)</td><td> $4 . 0 \times 1 0 ^ { - 1 }$   $< 1 0 ^ { - 1 5 }$ </td><td> $2 . 0 \times 1 0 ^ { - 8 }$   $6 . 3 \times 1 0 ^ { - 1 5 }$ </td><td> $2 . 3 \times 1 0 ^ { - 1 0 }$   $2 . 0 \times 1 0 ^ { - 1 5 }$ </td><td> $1 . 2 \times 1 0 ^ { - 1 3 }$   $< 1 0 ^ { - 1 5 }$ </td></tr><tr><td></td><td>multi-leg oracle</td><td> $1 . 2 \times { { 1 0 } ^ { - 1 3 } }$ </td><td> $6 . 4 \times { 1 0 } ^ { - 1 4 }$ </td><td> $< 1 0 ^ { - 1 5 }$ </td><td> $< 1 0 ^ { - 1 5 }$ </td></tr><tr><td> $x _ { 1 } x _ { 2 } \cdots x _ { k }$ </td><td></td><td> $6 . 1 \times 1 0 ^ { - 1 }$ </td><td> $3 . 3 \times 1 0 ^ { - 1 }$ </td><td> $1 . 4 \times 1 0 ^ { - 1 }$ </td><td> $1 . 2 \times 1 0 ^ { - 1 5 }$ </td></tr><tr><td>(square-free)</td><td>tied</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>untied (det.-val.)</td><td> $3 . 3 \times 1 0 ^ { - 1 }$ </td><td> $< 1 0 ^ { - 1 5 }$ </td><td> $< 1 0 ^ { - 1 5 }$ </td><td> $< 1 0 ^ { - 1 5 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>multi-leg oracle</td><td> $< 1 0 ^ { - 1 5 }$ </td><td> $< 1 0 ^ { - 1 5 }$ </td><td> $< 1 0 ^ { - 1 5 }$ </td><td> $< 1 0 ^ { - 1 5 }$ </td></tr></table>

## D.6 Convergence criterion

We report best-of-restarts MSE without applying a separate convergence threshold. A configuration is described as fitting the target if the residual reaches machine precision $\mathrm { ( M S E < 1 0 ^ { - 8 } ) }$ and as failing if the residual stays above $1 0 ^ { - 1 }$ . The intermediate regime is interpreted as a partial fit due to optimizer-landscape difficulty rather than a representational limitation, and is discussed for the $k = 5$ case in Section 5.2.

To distinguish the plateau of Section 5.2 from a slow-optimization artifact, we re-ran the failed configurations with 5× the L-BFGS iteration budget. The residual stayed within 10% of its plateau value across all configurations tested, confirming that the plateau is robust to optimization budget.

## D.7 Scope of the role-asymmetric separation

Theorem 4.10 shows that one untied detector-value atom suffices for the role-asymmetric witness $F _ { d v } = x _ { 1 } ^ { k - 1 } x _ { 2 }$ , where one direction acts as a repeated detector and one direction supplies the single value. This appendix tests the boundary of that statement: targets with multiple independent value roles do not reduce to a single untied atom.

Concretely, at $k = 3$ we compare tied, untied, and a multi-leg oracle (the smallest atom family rich enough to express any product of k linear forms) on the role-asymmetric witness $F _ { d v }$ and on the square-free product

$$
F _ { s f } = x _ { 1 } x _ { 2 } \cdot \cdot \cdot x _ { k } ,
$$

in which each of the k variables appears exactly once.

Figure 6 and Table 3 confirm three regimes. On $F _ { d v }$ , untied atoms and the multi-leg oracle both fit at $m = 1$ . Tied atoms keep a positive loss at $m = 1$ , reach small but nonzero losses at $m = 2$ and $m = 3$ (about $1 0 ^ { - 8 }$ and $1 0 ^ { - 1 0 }$ , consistent with border rank two), and reach about $1 0 ^ { - 1 3 }$ at $m = 4$ On $F _ { s f }$ , the multi-leg oracle fits at $m = 1$ , untied requires $m = 2 ,$ , and tied requires m $\geq 2 ^ { k - 1 }$ matching its Waring rank (by Proposition 4.9). The untied advantage is therefore specialized to targets with a single value role and does not extend to arbitrary products.

![](images/dcb8e6b112f52e0214d01646b6b1d3afb8dbf5ab5d6c41c0816e3c17075498bd.jpg)  
Figure 7: Matched input-projection control on $x _ { 1 } ^ { k - 1 } x _ { 2 } \colon$ untied width m (dashed) versus tied width 2m (solid). Tied SwiGLU uses $\mathrm { S i L U } ( z ) z$ . Median over three seeds; bands show min–max. Total parameter counts are $m ( 2 d + 1 )$ and $2 \dot { m ( d + 1 ) }$ , respectively.

Table 4: Coefficient-size diagnostics for $F _ { 3 } = x _ { 1 } ^ { 2 } x _ { 2 }$ . Exact and border ranks determine the regimes; losses and sizes describe the candidate fits.
<table><tr><td>Family</td><td>m</td><td>Regime</td><td>Best loss</td><td>Best  $A _ { \mathrm { m a x } }$ </td><td>Near-best  $A _ { \mathrm { m a x } }$ </td><td>κ</td></tr><tr><td>Untied</td><td>1</td><td>Exact control</td><td> $1 . 6 \times 1 0 ^ { - 2 9 }$ </td><td>1.00</td><td>1.00</td><td>1.7</td></tr><tr><td>Tied</td><td>1</td><td>Positive infimum</td><td>0.5556</td><td>0.385</td><td>0.385</td><td>1.5</td></tr><tr><td>Tied</td><td>2</td><td>Zero, unattained</td><td> $5 . 4 \times { 1 0 } ^ { - 1 2 }$ </td><td>83.17</td><td>83.17</td><td> $4 . 4 \times 1 0 ^ { 7 }$ </td></tr><tr><td>Tied</td><td>3</td><td>Exact width</td><td> $1 . 5 \times 1 0 ^ { - 2 8 }$ </td><td>0.471</td><td>0.471</td><td>n.i.</td></tr></table>

## D.8 Matched input-projection control in gated MLPs

Untied architectures use two input projections per channel. We compare untied width m with tied width $2 m .$ , matching input-projection counts at 2md. The tied model has m additional scalar-output parameters.

Figure 7 shows that the ordering depends on degree and width. At $k = 2 ,$ , tied SwiGLU at width four and untied SwiGLU at width two attain comparable median losses, $5 . 5 \times 1 0 ^ { - 1 3 }$ and $4 . 3 \times 1 0 ^ { - 1 3 }$ respectively. $\mathrm { A t } \ k = 3 .$ , untied width four reaches $4 . 0 \times 1 0 ^ { - 4 }$ , compared with $4 . 4 \times 1 0 ^ { - 3 }$ for tied width eight, an approximately 10.8-fold difference. These fixed-budget fits show a selected untied advantage, not a uniform separation or minimum-width certificate.

## D.9 Border-rank diagnostics

The additional coefficient-space experiments use the apolar norm. $\begin{array} { r } { \operatorname { I f } f ( x ) = \sum _ { | \alpha | = k } f _ { \alpha } x ^ { \alpha } } \end{array}$ , then

$$
\| f \| _ { \mathrm { a p } } ^ { 2 } = \sum _ { | \alpha | = k } { \frac { \alpha ! } { k ! } } f _ { \alpha } ^ { 2 } .
$$

This is the Frobenius norm of the symmetric coefficient tensor and satisfies $\| ( u ^ { \top } x ) ^ { k } \| _ { \mathrm { a p } } = \| u \| ^ { k }$ Reported losses are relative squared errors $\| \widehat { f } - f \| _ { \mathrm { a p } } ^ { 2 } / \| f \| _ { \mathrm { a p } } ^ { 2 }$ . Unlike the unweighted monomialcoordinate norm used in Section 5.2, this convention accounts for multinomial multiplicities. The two norms have the same closure, but numerical losses need not agree.

For the size diagnostic, each configuration has 27 candidate fits: 12 unregularized initializations and five at each of three penalties on $\textstyle \sum _ { a } A _ { a } ^ { 2 }$ . We record the smallest loss and the smallest $A _ { \mathrm { m a x } }$ among candidates with loss at most 1.02 times that minimum. At the selected near-best candidate, we also report the intrinsic condition number $\kappa = 1 / \sigma _ { \operatorname* { m i n } } ( \mathcal { T } )$ for the normalized Terracini/Jacobian matrix after removing scaling directions [Vannieuwenhoven, 2017]. A small singular value indicates sensitivity of decomposition parameters; it is not a certificate of global optimality. The notation “n.i.” means that local nonuniqueness makes this diagnostic uninformative.

Table 5: The same diagnostic for $F _ { 3 , 2 } = x _ { 1 } ^ { 2 } x _ { 2 } + x _ { 3 } ^ { 2 } x _ { 4 }$ . Its tied border width is four and tied exact width is six.
<table><tr><td>Family</td><td>m</td><td>Regime</td><td>Best loss</td><td>Best  $A _ { \mathrm { m a x } }$ </td><td>Near-best  $A _ { \mathrm { m a x } }$ </td><td>κ</td></tr><tr><td>Untied</td><td>2</td><td>Exact control</td><td> $6 . 4 \times 1 0 ^ { - 1 9 }$ </td><td>1.00</td><td>1.00</td><td>1.7</td></tr><tr><td>Tied</td><td>2</td><td>Positive infimum</td><td>0.5000</td><td>46.95</td><td>2.27</td><td> $8 . 6 \times 1 0 ^ { 2 }$ </td></tr><tr><td>Tied</td><td>3</td><td>Positive infimum</td><td> $\phantom { - } 0 . 2 7 7 8$ </td><td>66.30</td><td>2.27</td><td> $8 . 6 \times 1 0 ^ { 2 }$ </td></tr><tr><td>Tied</td><td>4</td><td>Zero, unattained</td><td> $1 . 1 \times 1 0 ^ { - 1 0 }$ </td><td>47.74</td><td>47.736</td><td> $8 . 3 \times 1 0 ^ { 6 }$ </td></tr><tr><td>Tied</td><td>5</td><td>Zero, unattained</td><td> $9 . 3 \times 1 0 ^ { - 1 1 }$ </td><td>34.40</td><td>34.405</td><td> $1 . 9 \times 1 0 ^ { 1 3 }$ </td></tr><tr><td>Tied</td><td>6</td><td>Exact width</td><td> $8 . 3 \times 1 0 ^ { - 1 6 }$ </td><td>3.72</td><td>3.72</td><td>n.i.</td></tr></table>

Table 6: Recovery fractions for configurations with a width-three generating witness. Witness losses are approximately $1 0 ^ { - 3 2 }$ . Widths four and five pad the representation with additional channels.
<table><tr><td></td><td></td><td colspan="2">Tied width</td><td colspan="3">Untied width</td><td rowspan="2">Best untied loss at width 3</td></tr><tr><td>k</td><td>3</td><td>4</td><td>5</td><td>3</td><td>4</td><td>5</td></tr><tr><td>3</td><td>0.33</td><td>0.32</td><td>0.70</td><td>0.80</td><td>0.92</td><td>1.00</td><td> $1 . 2 \times 1 0 ^ { - 2 0 }$ </td></tr><tr><td>4</td><td>0.42</td><td>0.67</td><td>0.75</td><td>0.17</td><td>0.65</td><td>0.88</td><td> $2 . 4 \times 1 0 ^ { - 1 8 }$ </td></tr><tr><td>5</td><td>0.77</td><td>0.80</td><td>0.85</td><td>0.00</td><td>0.02</td><td>0.00</td><td> $1 . 7 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>6</td><td>0.08</td><td>0.20</td><td>0.22</td><td>0.00</td><td>0.00</td><td>0.00</td><td> $8 . 1 \times 1 0 ^ { - 9 }$ </td></tr></table>

Table 4 follows the three regimes for the original monomial. At tied width one, the positive floor is attained because the class is closed. At width two, the smallest near-best candidate has large size; this is consistent with the forced-divergence statement for any sequence whose error tends to zero. A finite collection of large candidates does not exclude a smaller-size candidate that the optimizer missed. At exact width three, a bounded near-zero fit is recovered.

For the independent-block target, Table 5 places the positive-to-zero infimum transition at width four, as predicted by Theorem C.7. The large best-fit size at width two does not contradict the positive gap: this candidate approximates one block through a degeneration while leaving the other unfitted, at loss 0.5. Its condition number is $7 . 9 \times 1 0 ^ { 6 }$ , whereas a candidate within 2% of the same loss has size 2.27. The theorem identifies the regime; the diagnostics describe how individual fits approach it.

As a closed-class control, tied fits to $Q = \mathrm { d i a g } ( 1 , 1 , - 1 , - 1 )$ yield relative coefficient errors 0.7500, 0.5000, and 0.2500 at widths one, two, and three, matching $( 4 - m ) / 4$ . Penalty sweeps along the monomial degeneration give log–log slopes $- 4 . 0 0 3 , - 4 . 0 0 2$ , and −4.002 for squared error versus $A _ { \mathrm { m a x } }$ at $k = 3 , 4 , 5 ,$ , consistent with the constructive $O ( A _ { \operatorname* { m a x } } ^ { - 4 } )$ path in Equation (1). These slopes do not prove an optimal trade-off.

## D.10 Recovery of known representations

To separate optimization from expressivity, we retain generating parameters for targets constructed from three atoms. These witnesses establish width-three representability, not minimal rank three. Fitting uses float64 L-BFGS-B with automatic differentiation, $d = 3 ,$ and 60 restarts per configuration. Recovery means relative apolar loss below $1 0 ^ { - 1 0 }$ . Table 6 reports recovery fractions for the respective target constructions in each family; it should not be read as a comparison on identical targets.

No untied width-three restart reaches the tolerance at $k = 5 \mathrm { o r } k = 6$ , despite the generating witnesses. At $k = 5 ,$ widening yields one recovery in 60 attempts at width four and none at width five. Thus the tested extra widths do not reliably resolve the optimization difficulty. In a separate structured example, $F _ { 4 , 2 }$ has exact untied rank two, but eight restarts all return loss 0.25. A 40-restart run recovers the representation ten times, with best loss $5 . 4 \times 1 0 ^ { - 2 0 }$ and median 0.25. These failures cannot be used as rank lower bounds.

## D.11 Dense random targets and mixtures of degrees

For scalar homogeneous targets with d variables and $k \geq 3$ , let $N = { \binom { d + k - 1 } { k } }$ . Outside the known defective cases, complex-generic ranks are

$$
r _ { \mathrm { t i e d } } ^ { \mathrm { g e n } } = \lceil N / d \rceil , \qquad r _ { \mathrm { u n t i e d } } ^ { \mathrm { g e n } } = \lceil N / ( 2 d - 1 ) \rceil
$$

Table 7: Dense random targets: observed recovery widths and complex-generic predictions. “Above” marks recovery only above the prediction and does not certify a larger real rank.
<table><tr><td> $d$ </td><td>k</td><td>Target</td><td>N</td><td>Tied obs./pred.</td><td>Untied obs./pred.</td><td>Obs. ratio</td><td>Above prediction</td></tr><tr><td>2</td><td>5</td><td>0</td><td>6</td><td>3/3</td><td>3/2</td><td>1.00</td><td>Untied</td></tr><tr><td>2</td><td>5</td><td>1</td><td>6</td><td>4/3</td><td>2/2</td><td>2.00</td><td>Tied</td></tr><tr><td>2</td><td>6</td><td>0</td><td>7</td><td>4/4</td><td>3/3</td><td>1.33</td><td></td></tr><tr><td>2</td><td>6</td><td>1</td><td>7</td><td>5/4</td><td>3/3</td><td>1.67</td><td>Tied</td></tr><tr><td>3</td><td>3</td><td>0</td><td>10</td><td>4/4</td><td>3/3</td><td>1.33</td><td></td></tr><tr><td>3</td><td>3</td><td>1</td><td>10</td><td>4/4</td><td>3/3</td><td>1.33</td><td></td></tr><tr><td>3</td><td>4</td><td>0</td><td>15</td><td>6/6</td><td>3/3</td><td>2.00</td><td></td></tr><tr><td>3</td><td>4</td><td>1</td><td>15</td><td>7/6</td><td>4/3</td><td>1.75</td><td>Both</td></tr><tr><td>4</td><td>3</td><td>0</td><td>20</td><td>6/5</td><td>4/4</td><td>1.50</td><td>Tied</td></tr><tr><td>4</td><td>3</td><td>1</td><td>20</td><td>6/5</td><td>4/4</td><td>1.50</td><td>Tied</td></tr><tr><td>4</td><td>4</td><td>0</td><td>35</td><td>11/10</td><td>6/5</td><td>1.83</td><td>Both</td></tr><tr><td>4</td><td>4</td><td>1</td><td>35</td><td>10/10</td><td>6/5</td><td>1.67</td><td>Untied</td></tr><tr><td>5</td><td>3</td><td>0</td><td>35</td><td>8/8</td><td>5/5</td><td>1.60</td><td></td></tr><tr><td>5</td><td>3</td><td>1</td><td>35</td><td>8/8</td><td>5/5</td><td>1.60</td><td></td></tr></table>

Table 8: Exploratory degree-mixture fits. Each entry is recovered width (dimension-count benchmark). Paired entries correspond to two independently sampled targets.
<table><tr><td> $d$ </td><td> $K$ </td><td> $N _ { \mathrm { t o t } }$ </td><td>Tied</td><td>Untied</td></tr><tr><td>2</td><td>{2, 3, 4, 5}</td><td>18</td><td>5 (4), 4 (4)</td><td>4 (3), 3 (3)</td></tr><tr><td>3</td><td>{2,3}</td><td>16</td><td>5 (4), 4 (4)</td><td>4 (3), 3 (3)</td></tr><tr><td>3</td><td>{2,3,4}</td><td>31</td><td>7(7)</td><td>5 (5)</td></tr><tr><td>4</td><td>{2,3}</td><td>30</td><td>7(6), 7 (6)</td><td>4(4), 5(4)</td></tr></table>

by the Alexander–Hirschowitz theorem and the tangential-Veronese result [Alexander and Hirschowitz, 1995, Abo and Vannieuwenhoven, 2018]. A defective case is one in which the dimension count underestimates the required generic width. For the tied family the count increases by one at $( d , k ) = ( 3 , 4 ) , ( 4 , 4 ) \bar { , } ( 5 , 3 ) \bar { , } ( 5 , 4 )$ ; for the untied family it increases by one at $( d , k ) \overset { \cdot } { = } ( 3 , 3 ) , ( 4 , 3 ) , ( 5 , 3 )$

We sample coefficients independently from a Gaussian in an apolar-orthonormal basis and fix these predictions before fitting. The observed recovery width is the smallest tested width reaching relative loss below $1 0 ^ { - 1 0 }$ in 32 restarts. The complex lower threshold applies almost surely to these continuously sampled real targets; it need not be an attainable real width. A real near-zero fit at that width supplies numerical recovery evidence. A fit only at the next width leaves higher real rank, an ill-conditioned limit, and optimizer failure unresolved.

Table 7 recovers several nontrivial corrected counts. The ternary cubic requires an untied prediction of three, rather than the naive two. At $( d , k ) = ( 5 , 3 )$ , both predictions increase, to tied width eight and untied width five; both are recovered on two targets, with best losses at least $2 . 5 \times 1 0 ^ { - 4 }$ one channel below and below $1 0 ^ { - 1 5 }$ at the predictions. For tied (3, 4) target zero, losses below and at width six are $1 . 4 7 \times 1 0 ^ { - 2 }$ and $9 . 8 6 \times \dot { 1 0 } ^ { - 1 9 }$ . Observed tied/untied width ratios range from one to two; the factor-k separation of the monomial is a structured example rather than a generic prediction.

Mixtures of degrees. We also fit independent random components at degrees $K ,$ , with shared directions but a free coefficient for each channel and degree. With $\begin{array} { r } { N _ { \mathrm { t o t } } = \sum _ { k \in K } \binom { d + k - 1 } { k } } \end{array}$ , parameter counting gives the reference widths

$$
m _ { t } ^ { \mathrm { c o u n t } } = \left\lceil \frac { N _ { \mathrm { t o t } } } { d - 1 + | K | } \right\rceil , \qquad m _ { u } ^ { \mathrm { c o u n t } } = \left\lceil \frac { N _ { \mathrm { t o t } } } { 2 d - 2 + | K | } \right\rceil .
$$

These are dimension-count benchmarks, not proved generic-rank formulas for the shared-direction model. The relaxation also differs from a fixed analytic gate because its coefficients are free across degrees.

Table 9: Trained-block distillation. Ratios are tied/untied errors; values above one favor untied. Ratios are retained at their reported precision and can differ slightly from ratios of rounded losses.
<table><tr><td>Teacher</td><td>m</td><td>Untied m</td><td>Tied m</td><td>Ratio</td><td>Tied 1.5m</td><td>Ratio</td></tr><tr><td>Untied</td><td>4</td><td> $4 . 0 2 \times 1 0 ^ { - 1 }$ </td><td> $3 . 9 6 \times 1 0 ^ { - 1 }$ </td><td>0.99</td><td> $2 . 2 9 \times 1 0 ^ { - 1 }$ </td><td>0.57</td></tr><tr><td>Untied</td><td>8</td><td> $1 . 0 3 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 6 \times 1 0 ^ { - 1 }$ </td><td>1.13</td><td> $8 . 7 0 \times 1 0 ^ { - 4 }$ </td><td>0.01</td></tr><tr><td>Untied</td><td>10</td><td> $1 . 3 7 \times 1 0 ^ { - 4 }$ </td><td> $7 . 5 1 \times 1 0 ^ { - 2 }$ </td><td>549</td><td> $1 . 0 9 \times 1 0 ^ { - 3 }$ </td><td>7.94</td></tr><tr><td>Untied</td><td>12</td><td> $8 . 1 1 \times 1 0 ^ { - 6 }$ </td><td> $8 . 7 0 \times 1 0 ^ { - 4 }$ </td><td>107</td><td> $6 . 1 1 \times 1 0 ^ { - 4 }$ </td><td>75.3</td></tr><tr><td>Tied</td><td>8</td><td> $2 . 8 2 \times 1 0 ^ { - 1 }$ </td><td> $2 . 9 9 \times 1 0 ^ { - 1 }$ </td><td>1.06</td><td> $1 . 6 4 \times 1 0 ^ { - 1 }$ </td><td>0.58</td></tr><tr><td>Tied</td><td>12</td><td> $9 . 3 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 4 \times 1 0 ^ { - 1 }$ </td><td>1.76</td><td> $6 . 4 7 \times 1 0 ^ { - 2 }$ </td><td>0.70</td></tr><tr><td>Tanh</td><td>8</td><td> $3 . 4 5 \times 1 0 ^ { - 1 }$ </td><td> $3 . 4 4 \times 1 0 ^ { - 1 }$ </td><td>1.00</td><td> $2 . 5 1 \times 1 0 ^ { - 1 }$ </td><td>0.73</td></tr><tr><td>Tanh</td><td>12</td><td> $1 . 8 1 \times 1 0 ^ { - 1 }$ </td><td> $2 . 5 1 \times 1 0 ^ { - 1 }$ </td><td>1.38</td><td> $1 . 5 3 \times 1 0 ^ { - 1 }$ </td><td>0.84</td></tr></table>

The untied recovered width is smaller in all seven comparisons in Table 8. The mixture with the largest coefficient dimension recovers both counting benchmarks. Recovery above a benchmark is unresolved and is not attributed to a specific real rank or optimization mechanism.

## D.12 Distillation of trained gated blocks

A frozen random tanh network $\mathbb { R } ^ { 8 } \to \mathbb { R } ^ { 8 }$ with hidden width 64 supplies a common target. We train three teachers with 192 parameters each, excluding biases: untied SwiGLU width eight, tied SwiGLU width twelve, and a tanh MLP width twelve. Their training relative MSEs on the common target are 0.295, 0.172, and 0.112, respectively. Tied SwiGLU uses SiLU(z)z, preserving the untied model’s gate while imposing $v = u$

For each teacher and $m \in \{ 2 , 3 , 4 , 6 , 8 , 1 0 , 1 2 \}$ , we fit untied width m, tied width m, and approximately parameter-matched tied width 1.5m (rounded when necessary). With input and output dimension eight, an untied channel has 24 parameters and a tied channel 16. Counts therefore match exactly for the even base widths in Table 9; odd-width rounding is approximate. Inputs follow $\mathcal { N } ( 0 , \dot { I } _ { 8 } )$ , with 4,000 training and 4,000 held-out samples. L-BFGS runs for up to 6,000 iterations. The study contains $3 \times 7 \times \bar { 3 } \times 3 = 1 8 9$ fits; the table reports median relative held-out MSE over three seeds at the retained representative widths. Recovery uses a descriptive threshold of $1 0 ^ { - 3 }$

On the untied teacher, the first equal-width median recovery is at width ten for untied students and twelve for tied students. Under parameter matching, tied width twelve already reaches the criterion at base width eight. Untied students achieve lower errors once they enter their recovery regime, but the tied family also recovers the target. Across all widths, geometric-mean tied/untied ratios are 1.12 at equal width and 0.67 under matching for the tied teacher, and 1.06 and 0.78 for the tanh teacher. These controls do not show a consistent untied advantage.

At width ten on the untied teacher, one of three untied seeds fails to recover, with losses $1 . 4 \times 1 0 ^ { - 4 }$ $1 . 2 \times 1 0 ^ { - 5 }$ , and $7 . 5 \times 1 0 ^ { - 2 } ;$ ; all three recover at width twelve. A separate six-restart diagnostic at untied width eight obtains best loss $2 . 7 5 \times 1 0 ^ { - 3 }$ , with other losses 0.07–0.15. The teacher parameters themselves give an exact width-eight representation, so these residuals are optimization failures. This experiment tests recovery and precision for imperfect trained teachers; it does not establish independent architectural width lower bounds.

Norm continuation. For tied SwiGLU width twelve fitted to the untied width-eight teacher, the best of three paths with a relaxing norm penalty reaches held-out relative MSE $3 . 0 \times \bar { 1 } 0 ^ { - 4 }$ at squared parameter norm $\| \theta \| ^ { 2 }$ ≈ 180, approximately five times its initial value. One path stalls near $7 \times 1 0 ^ { - 2 }$ No divergence is observed over the tested range. The result concerns finite-error approximation under a fixed parameterization; it proves neither closedness nor exact representability and does not bound each Taylor coefficient separately. This raw squared parameter norm is distinct from the homogeneous atom sizes used in the coefficient-space diagnostics.

## D.13 Adam-trained gated scalar regression

We additionally compare tied and untied SiLU-gated scalar blocks at $d = 8$ on two four-channel teachers. The asymmetric teacher is $\begin{array} { r } { \sum _ { j } a _ { j } \operatorname { S i L U } ( \bar { p _ { j } ^ { } } x ) ( q _ { j } ^ { \top } x ) } \end{array}$ with independent standard Gaussian directions $p _ { j } , q _ { j }$ and weights ${ a } _ { j } ;$ the tied teacher sets $q _ { j } = p _ { j }$ . Here the tied student is the matchedgate specialization SiL $\mathrm { J } ( u ^ { \top } x ) ( u ^ { \top } x )$ . Training uses $2 0 { , } 0 0 0$ examples, 5,000 held-out examples, Adam with learning rate $3 \times 1 \dot { 0 } ^ { - 3 }$ , batch size 256, and a cosine schedule for 4,000 steps, and four seeds. Table 10 reports median relative test MSE.

Table 10: Adam-trained scalar regression with approximately matched total parameters. Counts are untied/tied; the tied models have slightly larger budgets. Bold marks the smaller error within each teacher and row.
<table><tr><td>Parameters Untied/tied</td><td>Asymmetric teacher Untied</td><td>Tied Untied</td><td>Tied teacher Tied</td></tr><tr><td>17/18</td><td>0.3162</td><td>0.3808</td><td>0.1899</td></tr><tr><td>34/36</td><td></td><td>0.2580 0.1049</td><td>0.0014</td></tr><tr><td>51/54</td><td>0.0843</td><td>0.0178 0.0045</td><td>0.0013</td></tr><tr><td>68/72</td><td>0.0322 0.0188</td><td>0.0387 0.0370 0.0010</td><td>0.0006</td></tr><tr><td>102/108</td><td>0.0088</td><td>0.0052 0.0009</td><td>0.0006</td></tr><tr><td>136/144</td><td>0.0062</td><td>0.0024 0.0008</td><td>0.0004</td></tr></table>

The untied student has lower error on the asymmetric teacher at the intermediate budgets of 34–68 parameters, whereas the tied student is more accurate at the smallest budget and at the two largest budgets. On the tied teacher, the tied student is more accurate at every budget. Under approximately matched total parameters, the untied advantage is therefore limited to a range of budgets on the asymmetric teacher.