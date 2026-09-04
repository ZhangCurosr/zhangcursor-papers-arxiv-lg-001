# Relative Prime Factorization and Finite-State Presentations under Fixed Finite-Monoid Observation

Takayuki Kuriyama Independent Researcher, Tokyo, Japan growup.kuriyama@gmail.com

September 2, 2026

## Abstract

Let $L \subseteq \Sigma ^ { * }$ and fix a morphism $h : \Sigma ^ { * } \to M$ into a finite monoid. We study exact factorization and canonical presentation in the relative syntactic congruence $\theta _ { L , h } : = \equiv _ { L } \cap$ ker h. Our main result separates unique factorization from finite direct presentation. We construct an explicit, exhaustively computer-checked 36-element quotient in which every live non-unit relative class has a unique exact prime factorization, while the valid prime-return rules contain the infinite family

$$
[ a b ]  [ a ] [ b a a a ] ^ { m } [ b ] \qquad ( m \geq 0 ) .
$$

Hence unique factorization does not imply the finite relative presentation property (FRP), even for a finite quotient. The witness uses the trivial language $\Sigma ^ { + }$ , showing that the obstruction is already a finite-monoid phenomenon. We then lift the same defect into the nonregular context-free language

$$
L ^ { \operatorname { w r a p } } = \{ c ^ { n } w d ^ { n } : n \geq 0 , \ w \in \{ a , b \} ^ { + } \} .
$$

For a finite observer extending the 36-element one, this pair has an infinite relative quotient, exactly 52 relative primes, global exact unique factorization, FSRP, and failure of FRP. Thus the finite-state separation is not an artifact of a finite quotient or of a disjoint-alphabet embedding.

We isolate the obstruction as tail under-saturation and introduce the finite-state relative presentation property (FSRP), in which the canonical valid right-hand-side languages are represented by their finite residual controllers. We prove FRP ⊊ FSRP and give a structural decomposition of direct-rule infinitude. For h-substitutable context-free languages with finite relative prime spectrum, every correct prime-return language is context-free; the unresolved step is whether factor-minimalization can force a nonregular valid-return language. In particular, we are not aware of any finite-prime ¬FSRP example.

On a rigid branch, prime-target left-division determinism (PTLD) restores Clark-style cancellation phenomena and implies unique exact factorization, tail exactness, tail determinism, and a quadratic valid-rule bound. A five-prime nonregular deterministic context-free example with a finite group observer satisfies PTLD while lying outside every fixed k, ℓ- substitutable class of Yoshinaka. Finally, for fixed h we give a strong positive-data learner for the canonical PTLD presentation with polynomial-time hypothesis updates and an explicit finite characteristic sample, and a more general limit reconstruction of the canonical FSRP controller from any weakly behaviorally correct CFG-valued positive-data learner.

## 1 Introduction

Distributional learning of context-free languages is based on the idea that substrings that occur in the same environments can be treated alike. Clark and Eyraud formalized this through syntactic distributions and substitutability, while Yoshinaka refined the comparison relation by bounded left and right contexts [1, 2]. The fixed finite-monoid framework replaces bounded windows by a specified finite algebraic observation $h : \Sigma ^ { * } \to M :$ two strings are compared only when they have the same h-type, and observed overlap then forces equality of full two-sided distributions.

The present paper studies a diferent but closely related question. Once the relative syntactic congruence $\theta _ { L , h } = \equiv _ { L }$ ∩ ker h has been fixed, what intrinsic multiplicative structure do its classes possess, and when does that structure admit a finite canonical grammar presentation?

Clark’s strong-learning construction for substitutable languages introduced prime syntactic congruence classes and valid productions. In that setting, every nonzero nonunit class has a unique prime factorization and only finitely many valid productions occur when the prime spectrum is finite [3]. The relative setting is more delicate. The finite observation h refines syntactic classes and destroys some of the cancellation phenomena used in Clark’s proof. Exact factorization may become nonunique, and even a finite relative quotient need not imply a finite direct valid-rule basis.

The answer begins with a separation. Relative unique prime factorization (UF) need not produce a finite direct relative-prime presentation (FRP), even when the relative quotient itself is finite. The failure is not factorization ambiguity: in our 36-element witness every live class has exactly one exact prime factorization, while valid prime-return paths have unbounded undersaturated tails. A useful feature of the separation is that it is already visible at the finite-monoid level. The target language of the flagship witness is deliberately chosen to be the trivial language $\Sigma ^ { + }$ . Consequently its nonempty syntactic structure contributes no complexity: on nonempty words the relative congruence is simply the kernel of the observer. Thus the failure is caused by the gap between exact fiber products and multiplication in a finite observer quotient, not by a complicated context-free target language. A matching-wrapper construction later shows that the same under-saturation mechanism persists intrinsically inside a nonregular contextfree language with infinite relative quotient: the wrapper contributes the unbounded syntactic geometry, while the middle 36-element dynamics still forces infinitely many valid prime returns. Finite-state relative presentation (FSRP) preserves these infinite structural families through finite residual controllers. Thus the central line of the paper is

$$
\mathrm { U F } \not \Rightarrow \mathrm { F R P } \longrightarrow \mathrm { u n d e r - s a t u r a t i o n } \longrightarrow \mathrm { F S R P } .
$$

Three levels must therefore be distinguished:

(i) exact prime factorization, where “exact” means equality of sets of strings, not merely equality in a quotient;

(ii) correct and valid prime returns, which concern only multiplication in the relative quotient;

(iii) finite-state presentability, which allows an infinite direct rule family to be represented by a canonical finite-state controller.

This separation is also consonant with the syntactic-concept viewpoint, where CFG nonterminals and productions are interpreted through sets of strings and inclusion/product structure rather than only individual quotient elements [4].

Main contributions. The four principal results are as follows.

1. An explicit, exhaustively computer-checked 36-element quotient has fifteen relative primes and unique exact factorization for all 35 live non-unit classes, yet it admits the infinite valid family $[ a b ]  [ a ] [ b a a a ] ^ { m } [ b ]$ . Hence $\mathrm { U F } \not \Rightarrow \mathrm { F R P }$ , and a separate finite $\mathrm { F R P / n o n – U F }$ witness shows that UF and FRP are incomparable.

2. FRP is strictly contained in FSRP. Beyond the finite 36-element witness, the nonregular context-free language $L ^ { \mathrm { w r a p } } = \{ c ^ { n } w d ^ { n } : n \geq 0 , \ w \in \{ a , b \} ^ { + } \}$ has infinite relative quotient, exactly 52 relative primes, global exact UF, FSRP, and not FRP. Thus finite-state compression is genuinely needed even when the ambient relative structure is infinite. It remains open whether an h-substitutable context-free pair with finite prime spectrum can fail FSRP itself.

3. The saturation-defect decomposition isolates the source of direct-rule infinitude. Saturated valid tails have a finite quantitative bound; every failure of FRP localizes to an infinite family of under-saturated tails in one semantic residual class.

4. PTLD—uniqueness of right division into prime observer targets—restores Clark-style path rigidity. It implies UF, valid-tail exactness, valid-tail determinism, and the $q ^ { 2 }$ direct-rule bound, while remaining strictly weaker than factor cancellation.

Two consequences complete the picture. First, $L _ { 0 } = \{ a ^ { n } b ^ { n } : n \geq 0 \} ^ { * }$ shows that PTLD is not necessary, whereas a five-prime nonregular deterministic context-free language with a $C _ { 2 } \times C _ { 2 }$ observer satisfies PTLD but is not $k ,$ ℓ-substitutable for any finite $k , \ell .$ Thus fixed finite-monoid typing goes strictly beyond the entire bounded-context hierarchy, not merely beyond ordinary substitutability. Second, for each fixed h the PTLD branch admits strong reconstruction of the direct canonical grammar with polynomial-time hypothesis updates and an explicit finite characteristic sample. On the lexically anchored PTLD subbranch the cut-separation radius vanishes, yielding polynomial characteristic data in explicit canonical grammar size; the same five-prime separation witness belongs to this subbranch. Arbitrary compact CFG size cannot control canonical thickness polynomially, so transfer to external grammar representations requires additional restrictions. The FSRP branch admits computable strong reconstruction of the canonical residual controller from any weakly behaviorally correct CFG-valued positive-data learner. Prime existence, finite residual splitting, and minimality of the valid basis provide the structural infrastructure for these results rather than separate headline claims.

## 2 Related work and positioning

The point of departure is Clark’s canonical prime grammar for substitutable context-free languages, where syntactic congruence classes are factored into primes and valid productions provide a canonical strong-learning target [3]. Yoshinaka’s k, ℓ-substitutability weakens the ordinary condition by protecting the substituted middle with fixed boundary words $u \in \Sigma ^ { k }$ and $v \in \Sigma ^ { \ell }$ if two strings $u y _ { 1 } v$ and uy<sub>2</sub>v occur in one common outer context, then they must remain interchangeable in every outer context [2]. The case $k = \ell = 0$ is ordinary substitutability. Clark’s later SCL-based constructions pursue a diferent finite-presentation principle: semantic prime sequences are ordered by inclusion, and finite canonical grammars are obtained by selecting maximal sequences under Noetherian-type hypotheses [5]. In Clark’s ordinary substitutable setting, unique factorization, tail rigidity, and finite valid-rule presentation occur together; finite monoid refinement separates these phenomena. Our FSRP construction keeps the structurally valid prime sequences themselves and, when they are infinite, compresses the family by residual languages rather than replacing it with semantic maxima.

<table><tr><td></td><td>Yoshinaka 2008</td><td>Clark 2013</td><td>Clark 2015</td><td>This paper</td></tr><tr><td>substitution guard</td><td>fixed k, l boundary words</td><td>ordinary shared-context condition</td><td>semantic SCL structure</td><td>fixed finite-monoid type h</td></tr><tr><td>structural atoms</td><td>semantic carrier guarded substrings</td><td>syntactic classes primes</td><td>SCL closed sets SCL primes and irreducibles</td><td>relative syntactic classes relative primes</td></tr><tr><td>finiteness mechanism strong target</td><td>bounded-context substitutability language identification</td><td>substitutable rigidity prime CFG</td><td>Noetherian maxima SCL grammar</td><td>FRP or residual compression direct CFG or residual controller</td></tr></table>

The bounded-context guard and the fixed-monoid guard are genuinely diferent restrictions. Section 15 gives a single nonregular deterministic context-free language $L ^ { \star }$ that is h-substitutable and PTLD for a four-element group observer, yet is not k, ℓ-substitutable for any finite pair (k, ℓ). Thus the fixed-h PTLD branch is not contained in the union of Yoshinaka’s finite-window classes.

The present fixed-h condition specializes the author’s earlier framework of relation substitutability [20]. The first version of that arXiv record introduced the recognizable-relation condition in 2014; its current version develops the fixed finite-monoid weak-learning framework. Taking the auxiliary recognizable equivalence to be ker h gives the same substitution-safety condition. The extra role of the fixed morphism here is structural: quotient multiplication and observer-type division support exact relative factorization, PTLD, residual splitting, and the FRP/FSRP hierarchy. The novelty claimed here is therefore not fixed-h substitutability itself, but exact factorization in $\theta _ { L , h }$ , PTLD, the FRP/FSRP and under-saturation theory, and the resulting strong structural targets.

Extended CFGs and regular right-hand sides. Grammars in which the right-hand side of a production is described by a regular language are classical, appearing under such names as extended context-free grammars and regular-right-part grammars [10, 11, 12]. Accordingly, the fact that a regular family of right-hand sides can be compiled into an ordinary CFG is not new. FSRP uses this classical expressive mechanism in a diferent role. For each intrinsic relative prime P, the right-hand-side language ${ \mathrm { V a l } } _ { P }$ is determined canonically by quotient return and primeirreducibility. FSRP is precisely the condition that these structurally determined languages are regular. The residual controller is then their canonical minimal deterministic partial realization; adjoining the empty residual as a dead state recovers the usual complete realization. The examples below establish that this finite-state level is strictly broader than a finite direct rule list, including for a nonregular context-free target with infinite relative quotient, but they do not establish that FSRP itself can fail under the context-free finite-prime hypotheses studied here. Thus the contribution is not extended-CFG expressiveness itself, but the relative factorization theory that determines which right-hand-side languages must be represented, explains why a finite direct list of rules may fail, and isolates regularity of their factor-minimal part as a separate structural question.

Prime decomposition of languages and codes. Prime decomposition under language concatenation has an independent history in formal-language theory; see, for example, Han et al. [13] for decomposition of regular languages. Our notion is more constrained: the factors are not ar bitrary languages but live classes of a fixed relative syntactic congruence, and exact setwise factorization is studied simultaneously with multiplication and return paths in the associated quotient. The factor-free property of the global valid-rule language also places one aspect of the construction in classical code theory. We use “infix code” only in this standard combinatorial sense; for broader background on codes and automata see Berstel, Perrin, and Reutenauer [14].

Grammatical inference. The learning section uses the characteristic-sample viewpoint of polynomial grammatical inference introduced by de la Higuera [15]. It is also related to Clark’s congruence-based CFG learner with a minimally adequate teacher [16], Yoshinaka’s multidimen sional substitutability for MCFGs [17], and the subsequent distributional learning work of Clark and Yoshinaka [18]. The present strong-learning contribution difers in its target: the learner is required to recover the canonical relative-prime presentation determined by the fixed observer.

Residual finite-state automata already provide canonical automata whose states are residual languages [6]. We do not claim residual automata as a new construction. Here they are applied to the canonical languages ${ \mathrm { V a l } } _ { P }$ of valid relative-prime right-hand sides, so that an infinite structural branching family is retained with its prime skeleton rather than replaced by an arbitrary DFA description of the terminal language.

Coste and Nicolas develop a reduction-based canonical form for local substitutable languages and show polynomial identification in time and thick data [8]. Their reductions may compete and yield incomparable reduced alternatives. Our valid-rule reduction has a related structural flavor, but the relative setting can admit infinitely many irreducible valid alternatives; FSRP replaces finite enumeration of those alternatives by finite-state residual compression. Our primeirreducibility means the absence of a prime-valued proper contiguous quotient block, rather than uniqueness of a reduction normal form in a parsing graph.

The residual-automaton construction and the Gold-style finite-automaton enumeration used later are standard ingredients. The new claims concern the relative structural languages they represent, the UF–FRP separation, the under-saturation characterization, and the canonical structural-learning consequences.

## 3 Relative syntactic classes

Let $L \subseteq \Sigma ^ { * }$ be a language. The two-sided distribution of $x \in \Sigma ^ { * }$ is $\mathcal { D } _ { L } ( x ) \ : = \ \{ ( u , v ) \ \in $ $\Sigma ^ { * } \times \Sigma ^ { * } : u x v \in L \}$ . The syntactic congruence is $x \ { \equiv } _ { L } \ y \iff \ { \mathcal { D } } _ { L } ( x ) \ = \ { \mathcal { D } } _ { L } ( y )$ . Fix a monoid morphism $h : \Sigma ^ { * } \to M$ . Unless stated otherwise M is finite. For later use, write Fact $( L ) : = \{ x \in \Sigma ^ { * } : { \mathcal { D } } _ { L } ( x ) \neq \emptyset \}$ for the set of factors of L.

Definition 3.1 (h-substitutability). The language L is h-substitutable if, for all $x , y \in \Sigma ^ { * }$ $h ( x ) = h ( y )$ and $\mathcal { D } _ { L } ( x ) \cap \mathcal { D } _ { L } ( y ) \neq \emptyset$ imply $x \equiv _ { L } y$ . Equivalently, among words of the same $h { - } t y p e$ , sharing one accepting two-sided context forces equality of the full syntactic distribution.

This is the fixed-morphism specialization of relation-substitutability obtained by taking the auxiliary recognizable equivalence to be ker h [20].

Definition 3.2 (Relative syntactic congruence). Define $\theta _ { L , h } : = \equiv _ { L }$ ∩ ker h. Thus $x \equiv _ { L , h } y \iff$ $x \equiv _ { L } y$ and $h ( x ) = h ( y )$ . Write $[ x ] _ { L , h }$ , or simply [x], for the corresponding class.

Lemma 3.3 (Type–context collapse). If L is h-substitutable, $h ( x ) = h ( y )$ , and $\mathcal { D } _ { L } ( x ) \cap \mathcal { D } _ { L } ( y ) \neq$ ∅, then $x \theta _ { L , h } y$ . In particular, $i f$ uxv, uyv $\in L$ and $h ( x ) = h ( y )$ , then $x \theta _ { L , h } y$

Proof. Immediate from the definitions of h-substitutability and $\theta _ { L , h }$

Since both $\equiv _ { L }$ and ker h are monoid congruences, so is $\theta _ { L , h }$

Definition 3.4 (Live and dead classes). A relative class X is live $i f { \mathcal { D } } _ { L } ( x ) \neq \emptyset$ for $x \in X$ , and dead otherwise.

Unlike the ordinary syntactic congruence, the relative congruence may have several dead classes, because dead words with diferent h-values cannot be identified. For each $m \in M$ there is at most one dead class of $h \mathrm { - t y p e \ } m$ , hence the number of dead relative classes is at most $| M |$

Definition 3.5 (Unit separation). We say that $( L , h )$ is unit-separated $i f$

$$
[ \varepsilon ] _ { L , h } = \{ \varepsilon \} .\tag{US}
$$

Unit separation is exactly what is needed if lexical terminals are to be represented by non-unit prime classes. It should be distinguished from cancellativity of the observation monoid.

## 3.1 Setwise product versus quotient product

If X, $Y \subseteq \Sigma ^ { * }$ are relative classes, define their exact setwise product by $X \cdot Y : = \{ x y : x \in X , y \in$ $Y \}$ . Since $\theta _ { L , h }$ is a congruence, the quotient product $X * Y : = [ x y ] _ { L , h }$ is well defined for any $x \in X , y \in Y$ . Always $X \cdot Y \subseteq X * Y$ , but equality need not hold.

This distinction is fundamental. Relative primality is defined using exact setwise equality, whereas Clark-style correct productions only require the quotient product to return to a prime class. In ordinary syntactic notation Clark likewise uses [XY] for the ambient congruence class containing the set product $X Y$ , and correct productions are based on the ambient class rather than equality of the set product [3].

## 4 Relative primes and exact decomposition

Definition 4.1 (Relative prime). A live relative class P is prime if it is non-unit and whenever $P = X \cdot Y \ f o r$ relative classes $X , Y$ , one of $X , Y$ is the unit class. A live non-unit class that is not prime is composite.

For a nonempty class X, write $\ell ( X ) : = \operatorname* { m i n } \{ | w | : w \in X \}$

Lemma 4.2 (Shortest-word geometry of exact products). Suppose $X = X _ { 1 } \cdots X _ { k }$ is an exact product of nonempty relative classes. Then $\begin{array} { r } { \ell ( X ) = \sum _ { i = 1 } ^ { k } \ell ( X _ { i } ) } \end{array}$ . Moreover, $i f \ w \ \in \ X$ has $| w | = \ell ( X )$ , then $w = w _ { 1 } \cdot \cdot \cdot w _ { k }$ for some $w _ { i } \in X _ { i }$ with $| w _ { i } | = \ell ( X _ { i } )$ for every i. In particular, every exact prime factorization of X occurs among the finitely many prime-labelled cuts of any shortest word of X.

Proof. Every product word has length at least $\textstyle \sum _ { i } \ell ( X _ { i } )$ , while concatenating shortest representatives attains that value. Hence $\begin{array} { r } { \ell ( X ) = \sum _ { i } \ell ( X _ { i } ) } \end{array}$ . Any factorization of a shortest $w \in X$ through the exact product must attain equality in each component, which proves the second assertion. □

Corollary 4.3 (Finite candidate principle). Every exact prime factorization of a live class X occurs as a prime-labelled cut of one fixed shortest word of $X$ . In particular, every such factorization has length at most $\ell ( X )$ , and there are at most $2 ^ { \ell ( X ) }$ <sup>−1</sup> cut candidates when $\ell ( X ) \geq$ 1.

Theorem 4.4 (Existence of exact relative prime decompositions). Every live non-unit relative class is an exact setwise product of one or more relative primes.

Proof. Induct on $\ell ( X )$ . If X is prime there is nothing to prove. Otherwise $X = Y \cdot Z$ with non-unit relative classes $Y , Z$ . Since X is live, so are $Y$ and Z: if $y z$ occurs as a factor of a word of $L ,$ then both y and z occur as factors. By the preceding lemma, $\ell ( X ) = \ell ( Y ) + \ell ( Z )$ and non-unitness gives $1 \leq \ell ( Y ) , \ell ( Z ) < \ell ( X )$ . Apply the induction hypothesis to $Y$ and Z and concatenate their prime decompositions. □

Remark 4.5. No substitutability, cancellation, or finiteness assumption is used in this existence theorem. This parallels the existence half of Clark’s prime-factorization lemma but is valid at the level of an arbitrary L-compatible monoid congruence.

## 5 Correct, pleonastic, and valid productions

Let $\mathcal { P }$ be the set of live relative primes. For a nonempty prime sequence $\alpha = P _ { 1 } \cdot \cdot \cdot P _ { k } \in \mathcal { P } ^ { + }$ write

$$
\overline { { \alpha } } : = P _ { 1 } \cdot \hdots \cdot P _ { k } \subseteq \Sigma ^ { * } , \qquad [ \overline { { \alpha } } ] : = P _ { 1 } \ast \cdot \cdot \cdot \ast P _ { k } .
$$

For the empty prime sequence we set

$$
\begin{array} { r } { \overline { { \varepsilon } } : = \{ \varepsilon \} , \qquad [ \overline { { \varepsilon } } ] : = [ \varepsilon ] _ { L , h } . } \end{array}
$$

Definition 5.1 (Correct branching production). For $k \geq 2$ , a rule $P  P _ { 1 } \cdots P _ { k }$ is correct $i f$ $P \in { \mathcal { P } }$ and $[ \overline { { \alpha } } ] = P .$ Equivalently, ${ \overline { { \alpha } } } \subseteq P$

This is the precise relative analogue of Clark’s correct branching production $[ \bar { \alpha } ]  \alpha \ [ 3 ]$

Definition 5.2 (Pleonastic and valid). A prime sequence $\alpha = P _ { 1 } \cdots P _ { k }$ is pleonastic if it has a proper contiguous block $\beta = P _ { i } \cdots P _ { j }$ 2 $j - i + 1 \geq 2$ , whose quotient product $[ \overline { { \beta } } ]$ is prime. A correct production is valid if its right-hand side is not pleonastic.

For each prime $P ,$ put Val<sub>P</sub> $: = \{ \alpha \in { \mathcal { P } } ^ { \geq 2 } : P \to \alpha$ is valid} and Val $\begin{array} { r } { \mathrm { ~ : = ~ } \bigcup _ { P \in \mathcal { P } } \mathrm { V a l } _ { P } } \end{array}$ . We identify a production $P  \alpha$ with the pair $( P , \alpha )$ and write $\mathcal { V } : = \{ ( P , \alpha ) : P \in \mathcal { P }$ , $\alpha \in { \mathrm { V a l } } _ { P } \}$ Thus, when $\mathcal { P }$ is finite, $\begin{array} { r } { | \mathcal { V } | = \sum _ { P \in \mathcal { P } } | \operatorname { V a l } _ { P } | } \end{array}$

Lemma 5.3 (Valid-rule reduction). Every correct production $P  \alpha$ is derivable using valid productions only: $P \Rightarrow ^ { * } \alpha$

Proof. Induct on |α|. If the rule is valid, apply it directly. Otherwise write $\alpha = \gamma \beta \delta$ where $\beta$ is a proper contiguous block of length at least two and $Q : = [ \overline { { { \beta } } } ]$ is prime. Since quotient multiplication is associative and ${ \overline { { \beta } } } \subseteq Q$ , both $P  \gamma Q \delta , \qquad Q  \beta$ are correct, and both right-hand sides are shorter than α. Apply the induction hypothesis to both rules. □

The proof is the same compression principle as Clark’s reduction of correct rules to valid rules, but does not use uniqueness of prime factorization.

## 5.1 Structural minimality of valid rules

Definition 5.4 (Structurally complete correct prime system). A set R of correct productions over $\mathcal { P }$ is structurally complete $i f$ every correct production $P \to \alpha$ satisfies $P \Rightarrow _ { R } ^ { * } \alpha$

Theorem 5.5 (Minimal valid-basis theorem). The set V of all valid productions is the least structurally complete correct prime-production system.

Proof. The valid-rule reduction lemma shows that V is structurally complete. Conversely, let $R$ be structurally complete and let $P \to \alpha$ be valid. Consider an R-derivation tree from $P$ to the prime word $\alpha .$ Any internal non-root prime node $Q$ spans a contiguous block $\beta$ of the prime frontier. By induction down the derivation subtree rooted at $Q ,$ correctness of its productions implies $[ \overline { { { \beta } } } ] = Q$ . Prime symbols here are grammar nonterminals; lexical leaves are terminal symbols and are not counted as internal prime nodes. $\operatorname { I f } | \beta | \geq 2$ , then $\beta$ is a proper prime-valued contiguous block of $\alpha ,$ contradicting validity. Hence no nontrivial internal branching node can occur below the root. The root must therefore use the rule $P  \alpha$ directly, so $P \to \alpha \in R$ Thus $\nu \subseteq R$ for every structurally complete correct system $R .$ □

Corollary 5.6. The valid right-hand-side language Val is factor-free: no valid word is a proper contiguous factor of another valid word.

Proof. If a valid word properly contained another valid word as a contiguous factor, that factor would have prime quotient and length at least two, making the larger word pleonastic. □

In code-theoretic terminology, the global valid-rule language Val is therefore an infix code [7].

## 6 Canonical generation from a finite direct basis

We first formulate the construction for an arbitrary monoid congruence $\theta \subseteq \subsetequnderline { { = } } _ { L }$ . We write $\mathrm { F R P } ( L , \theta )$ and $\rho ( L , \theta )$ for the general congruence-level notions; when $\theta = \theta _ { L , h }$ we abbreviate them by $\mathrm { F R P } ( L , h )$ and $\rho ( L , h )$ . The same convention will be used below for FSRP and its state complexity.

Definition 6.1 (Accepting classes). An L-compatible congruence class X is accepting $i f X \subseteq L$ Let $\operatorname { A c c } _ { \theta } ( L ) : = \{ X : X \subseteq L \}$

Because $\theta \subseteq \equiv _ { L }$ , each class is either entirely contained in L or disjoint from L.

Definition 6.2 (Finite relative presentation property). The pair $( L , \theta )$ has the finite relative presentation property $( F R P ) \ i f \ | { \mathcal { P } } | < \infty \qquad a n d \qquad | { \mathcal { V } } | < \infty .$

When $| \mathcal { P } | < \infty$ , define the validity radius $\rho ( L , \theta ) : = \operatorname* { s u p } \{ | \alpha | : \alpha \in \mathrm { V a l } _ { P }$ for some $P \in \mathcal P \}$ Since the prime alphabet is finite, $\mathrm { F R P } ( L , \theta ) \iff \rho ( L , \theta ) < \infty$

## Theorem 6.3 (Finite direct relative-prime presentation). Suppose

(i) $\theta \subseteq \equiv _ { L } ,$

(ii) $[ \varepsilon ] _ { \theta } = \{ \varepsilon \}$ 2

(iii) $\operatorname { A c c } _ { \theta } ( L )$ is finite;

(iv) (L, θ) satisfies FRP.

Then there is a finite CFG $G _ { L , \theta } ^ { \mathrm { r p } }$ such that for every live prime P, $L ( G _ { L , \theta } ^ { \mathrm { r p } } , P ) = P ;$ and $L ( G _ { L , \theta } ^ { \mathrm { r p } } ) =$ L.

Proof. Use the live primes as nonterminals, together with a fresh start symbol. Include all valid branching productions and each lexical rule $[ a ] _ { \theta } \to a$ for every letter whose class is live. Unit separation implies that a live letter class is non-unit, and the length-one argument shows it is prime.

For each accepting non-unit class X, include start rules for its exact prime decompositions. If $\varepsilon \in L$ , also include the start rule $S  \varepsilon$ . Corollary 4.3 shows that each accepting class has only finitely many exact prime decompositions, so the start-rule set is finite.

Soundness follows because every branching rule is correct. For completeness, take $w =$ $a _ { 1 } \cdots a _ { n } \in P$ . If $n = 1$ , then $P = [ a _ { 1 } ] _ { \theta }$ and the lexical rule

$$
P  a _ { 1 }
$$

derives w directly. Assume now $n \geq 2$ . Then $P  [ a _ { 1 } ] _ { \theta } \cdot \cdot \cdot [ a _ { n } ] _ { \theta }$ is a correct branching production. By valid-rule reduction,

$$
P \Rightarrow ^ { * } [ a _ { 1 } ] _ { \theta } \cdot \cdot \cdot [ a _ { n } ] _ { \theta } \Rightarrow ^ { * } w .
$$

Thus each prime nonterminal generates exactly its class. The start rules then generate exactly the accepting classes, hence L. □

## 6.1 The role of fixed-h substitutability

For the fixed observation $h : \Sigma ^ { * } \to M$ , define $L _ { m } : = L \cap h ^ { - 1 } ( m )$ . If L is h-substitutable and $x , y \ \in \ L _ { m } ,$ then x and y have the common context $( \varepsilon , \varepsilon )$ and the same $h { \mathrm { - t y p e } } ,$ hence $x \equiv _ { L } y$ . Therefore each nonempty $L _ { m }$ is a single $\theta _ { L , h ^ { - } } \mathrm { c l a s s }$ . Consequently $| \operatorname { A c c } _ { \theta _ { L , h } } ( L ) | \leq | M |$ Thus the internal prime calculus above does not need substitutability; in the fixed-h theorem, substitutability supplies a finite bound on the accepting relative classes.

## 7 Semantic residuals, prime-target division, and Clark rigidity

Clark’s ordinary substitutable theory combines unique prime factorization with a stronger tailexactness phenomenon. Under finite-monoid refinement these efects separate, and the correct intermediate object is the semantic residual. Throughout this section,

$$
\theta : = \theta _ { L , h } .
$$

Accordingly, all relative classes and all residual sets $\operatorname { R e s } _ { \theta } ( A , P )$ below refer to the fixed-h relative syntactic congruence.

For relative classes $A , P ,$ define $A \backslash P : = \{ x \in \Sigma ^ { * } : A x \subseteq P \}$ . This set is θ-saturated. Let

$$
\operatorname { R e s } _ { \theta } ( A , P ) : = \{ R : R { \mathrm { ~ i s ~ a ~ l i v e ~ r e l a t i v e ~ c l a s s ~ a n d ~ } } R \subseteq A \backslash P \} .
$$

Theorem 7.1 (Finite residual splitting). Assume L is h-substitutable and M is finite, and let A, P be live relative classes. For every $m \in \cal { M } , \mathsf { \Gamma } ( A \backslash P ) \cap h ^ { - 1 } ( m )$ is empty or a single relative class. Consequently $\operatorname { R e s } _ { \theta } ( A , P ) | \leq | M |$ . More sharply, $i f a = h ( A )$ and $p = h ( P )$ , then $| \mathrm { R e s } _ { \theta } ( A , P ) | \leq | \{ m \in \mathsf { T } _ { L , h } : a m = p \} |$ , where $\mathsf { T } _ { L , h } = h ( \mathrm { F a c t } ( L ) )$ .

Proof. Take $x , y \in A \backslash P$ with $h ( x ) = h ( y )$ and choose $a _ { 0 } \in A$ . Then $a _ { 0 } x , a _ { 0 } y \in P$ . Liveness of P supplies an outer context in which both occur, so x and y share a live context after the common left factor $a _ { 0 }$ . Lemma 3.3 gives $x \theta y$ . Finally, every $R \in \operatorname { R e s } _ { \theta } ( A , P )$ is live, so its observer type $m = h ( R )$ belongs to ${ \mathsf { T } } _ { L , h }$ . Since $A R \subseteq P$ , every such type also satisfies $h ( A ) m = h ( P )$ , which gives the sharper bound. □

Definition 7.2 (Valid-tail exactness). A valid production $N \to \gamma \alpha , \qquad \gamma , \alpha \neq \varepsilon$ , has an exact tail if $\dot { \alpha } = [ \overline { { \alpha } } ] _ { \theta }$ . The pair $( L , h )$ has valid-tail exactness (VTE) if every such sufix is exact. We write $V T E _ { 1 }$ for the weaker condition requiring exactness only after the first right-hand prime, i.e. for every valid $P  A \alpha$

Only $\mathrm { V T E _ { 1 } }$ is needed for the direct-presentation and rule-recovery consequences below; full VTE is retained because PTLD yields the stronger Clark-style sufix statement.

Definition 7.3 (Valid-tail determinism). We say that $( L , h )$ has valid-tail determinism (VTD) if, whenever $N  A \alpha , \qquad N  A \beta$ are valid productions with the same left-hand prime N and the same first right-hand prime A, then $\alpha = \beta$

Definition 7.4 (Prime-target left-division determinism). Let $\mathsf T _ { L , h } : = h ( \mathrm { F a c t } ( L ) )$ . Since $\varepsilon \in$ Fact(L) whenever $L \neq \emptyset .$ , one has $1 \in \mathsf { T } _ { L , h }$ . We say that $( L , h )$ satisfies prime-target leftdivision determinism $( P T L D )$ if for every $q , r , r ^ { \prime } \in \mathsf { T } _ { L , h }$ and every relative prime $P , ~ q r ~ =$ $\begin{array} { r } { h ( P ) = q r ^ { \prime } \quad \Longrightarrow \quad r = r ^ { \prime } . } \end{array}$

Factor-left-cancellation on ${ \mathsf { T } } _ { L , h }$ implies PTLD, but PTLD asks for cancellation only in equations whose target is a prime observer type.

Remark 7.5 (Groups as a suficient source of PTLD). If M is a finite group, then PTLD holds automatically: $q r = p = q r ^ { \prime }$ implies $r = q ^ { - 1 } p = r ^ { \prime }$ . Groups are thus a convenient source of examples, although they are not necessary; Example 7.25 below uses a non-group observer.

Remark 7.6 (Action-category interpretation). PTLD is naturally a thinness condition in the right action category R(M), whose objects are elements ofM and whose hom-set is $\mathscr { R } ( M ) ( q , p ) =$ $\{ r \in M : q r = p \}$ . When the relevant factor types fill M, PTLD says exactly that every prime observer target $p = h ( P )$ is subterminal: $| \mathcal { R } ( M ) ( q , p ) | \leq 1$ for every $q .$ This is stronger than triviality of the local stabilizer $\{ r : p r = p \}$ because parallel arrows may also occur across strict Green R-drops. This one-sided action category should not be confused with the standard two-sided kernel category of semigroup theory [19]; the viewpoint is interpretive rather than an additional hypothesis.

Standing hypothesis (R).

Within this section, (R) abbreviates the assumptions that L is h-substitutable, $( L , h )$ is unitseparated, and PTLD holds. Outside this section we restate these three assumptions explicitly.

Lemma 7.7 (Prime-target remainder collapse). Under (R), let P be a relative prime, let $u \theta u ^ { \prime } ,$ and suppose ux, $u ^ { \prime } y \in P$ . Then $x \theta y$

Proof. Because P is live, all factors $u , u ^ { \prime } , x , y$ occurring in the displayed prime words are live factors; hence their observer types belong to ${ \sf T } _ { L , h }$ . Since $u \theta u ^ { \prime }$ , PTLD applied to

$$
h ( u ) h ( x ) = h ( P ) = h ( u ^ { \prime } ) h ( y ) = h ( u ) h ( y )
$$

gives $h ( x ) = h ( y )$ . Choose $( \ell , r )$ with $\ell u x r \in L$ . Since $u x \theta u ^ { \prime } y ,$ also $\ell u ^ { \prime } y r \in L$ , and since $u \equiv _ { L } u ^ { \prime }$ , also $\ell u y r \in L$ . Thus x and y share the context $( \ell u , r )$ and have the same observer type; Lemma 3.3 gives xθy. □

Lemma 7.8 (Prime left-quotient rigidity). Under (R), let P be a relative prime and $u \in \Sigma ^ { * }$ Then

$$
u ^ { - 1 } P : = \{ v : u v \in P \}
$$

is empty or exactly one relative class. Moreover,

$$
u ^ { - 1 } P = [ \varepsilon ] _ { \theta } \quad \Longleftrightarrow \quad u \in P .
$$

Proof. If uv, $u w \in P$ , Lemma 7.7 with the common prefix u gives vθw. Thus a nonempty left quotient is contained in one relative class. It is also θ-saturated: if $v \theta v ^ { \prime }$ and $u v \in P$ , then $u v \theta u v ^ { \prime }$ and hence $u v ^ { \prime } \in P .$ . Therefore it is exactly that class. Finally, $\varepsilon \in u ^ { - 1 } P \mathrm { ~ i f f ~ } u \in P ;$ when this holds, the unique class containing ε is $[ \varepsilon ] _ { \theta } = \{ \varepsilon \}$ by unit separation. □

Corollary 7.9 (Prime prefix-freeness). Under (R), no word of a relative prime is a proper prefix of another word of the same prime.

Proof. If $x , x c \in P$ , then $c \in x ^ { - 1 } P = [ \varepsilon ] _ { \theta }$ by Lemma 7.8. Unit separation gives $c = \varepsilon .$ □

Lemma 7.10 (Relative prime-prefix escape). Under (R), if X is a relative prime and Y is a distinct live non-unit relative class, then some $x \in X$ does not begin with an element of Y.

Proof. Assume contrariwise that every $x \in X$ has a factorization $x \ : = \ : y v$ with $y \in Y$ . Fix one such factorization $x _ { 0 } = y _ { 0 } v _ { 0 }$ . For any other $x = y v$ , we have $y \theta y _ { 0 }$ and $y v , y _ { 0 } v _ { 0 } \in X$ , so Lemma 7.7 gives vθv<sub>0</sub>. Hence every remainder lies in one relative class R. Congruence now gives the exact equality $X = Y \cdot R$ . If R is non-unit this contradicts primality of $X ;$ if R is the unit class, unit separation gives $R = \{ \varepsilon \}$ and hence $X = Y$ , again a contradiction. □

Lemma 7.11 (Relative prime-prefix interception). Under (R), let $\alpha = A _ { 1 } \cdot \cdot \cdot A _ { m }$ and $\beta =$ $B _ { 1 } \cdots B _ { n }$ be nonempty sequences of relative primes. $I f \overline { { \alpha } } \supseteq \overline { { \beta } }$ , then there is j, $1 \leq j \leq n$ , such that

$$
A _ { 1 } \supseteq { \overline { { B _ { 1 } \cdots B _ { j } } } } .
$$

Proof. If $A _ { 1 } = B _ { 1 }$ , take $j = 1$ . Otherwise choose $b _ { 1 } \in B _ { 1 }$ by Lemma 7.10 so that $b _ { 1 }$ does not begin with an $A _ { \mathrm { 1 - W O r d } }$ . Extend $b _ { 1 }$ to a word of $\overline { { \beta } }$ and factor that word through α. Its first $A _ { \mathrm { { 1 } \mathrm { { - f a c t o r } } } }$ must extend strictly beyond $b _ { 1 }$ , so $R _ { 1 } : = b _ { 1 } ^ { - 1 } A _ { 1 }$ is nonempty. By Lemma 7.8, $R _ { 1 }$ is a single relative class.

Inductively, suppose $b _ { i } \in B _ { i }$ have been chosen through position t and $R _ { t } : = ( b _ { 1 } \cdot \cdot \cdot b _ { t } ) ^ { - 1 } A _ { 1 }$ is a nonempty relative class. If $R _ { t } = [ \varepsilon ] _ { \theta }$ , then $b _ { 1 } \cdot \cdot \cdot b _ { t } \in A _ { 1 }$ , and congruence gives ${ \overline { { B _ { 1 } \cdots B _ { t } } } } \subseteq A _ { 1 }$ ， so take $j = t$ . If $t < n$ and $R _ { t } = B _ { t + 1 }$ , then congruence similarly gives ${ \overline { { B _ { 1 } \cdot \cdot \cdot B _ { t + 1 } } } } \subseteq A _ { 1 }$ , and take $j = t + 1$

Otherwise $t < n$ and $R _ { t }$ is a live non-unit class distinct from $B _ { t + 1 }$ . By Lemma 7.10, choose $b _ { t + 1 } \in B _ { t + 1 }$ that does not begin with an $R _ { t } \mathrm { - w o r d }$ . Extend $b _ { 1 } \cdots b _ { t + 1 }$ to a word of $\overline { { \beta } }$ and factor it through α. The first $A _ { 1 } { \mathrm { - f a c t o r } }$ cannot end before $b _ { 1 } \cdots b _ { t } \colon$ it would be a proper prefix of an existing A<sub>1</sub>-word extending that prefix, contradicting Corollary 7.9. It cannot end at $b _ { 1 } \cdot \cdot \cdot b _ { t }$ since then $R _ { t } = [ \varepsilon ] _ { \theta }$ . Nor can it end inside $b _ { t + 1 }$ , because then it would have the form $b _ { 1 } \cdots b _ { t } s$ with $s \in R _ { t }$ a prefix of $b _ { t + 1 }$ , contrary to the choice of $b _ { t + 1 }$ . Thus it extends strictly beyond $b _ { 1 } \cdots b _ { t + 1 }$ , so $R _ { t + 1 }$ is again nonempty.

If the construction reached $t = n$ without one of the terminating cases, the complete word $b _ { 1 } \cdot \cdot \cdot b _ { n } \in { \overline { { \beta } } }$ would have to possess an A -factor extending beyond its end, impossible. Hence a required j exists. □

Lemma 7.12 (Shortest-word left division). Under (R), let X be a relative prime and let x be a shortest word in X. For every set $S \subseteq \Sigma ^ { * }$ ，

$$
x ^ { - 1 } ( X \cdot S ) = S .
$$

Proof. The inclusion $S \subseteq x ^ { - 1 } ( X \cdot S )$ is immediate. Conversely, if $x v = x ^ { \prime } s$ with $x ^ { \prime } \in X$ and $s \in S$ , then x and $x ^ { \prime }$ are comparable prefixes of the same word. Minimality of x gives $| x ^ { \prime } | \geq | x |$ 2 while Corollary 7.9 forbids x from being a proper prefix of $x ^ { \prime }$ . Hence $x = x ^ { \prime }$ and $v = s \in S$ □

Corollary 7.13 (Prime-left inclusion cancellation). Under (R), if X is prime and $X { \overline { { \alpha } } } \subseteq X { \overline { { \beta } } }$ then $\overline { { \alpha } } \subseteq \overline { { \beta } }$

Proof. Apply $x ^ { - 1 }$ for a shortest $x \in X$ and use Lemma 7.12.

Corollary 7.14 (Prime-left equality cancellation). Under (R), $X { \overline { { \alpha } } } = X { \overline { { \beta } } }$ implies $\overline { { \alpha } } = \overline { { \beta } }$

Corollary 7.15 (Prime-left saturation). Under (R), if X is prime, C is a live relative class, and $C = X { \overline { { \alpha } } }$ holds exactly, then $\overline { { \alpha } } = [ \overline { { \alpha } } ] _ { \theta }$

Proof. Let x be a shortest word in X. Lemma 7.12 gives $\overline { { \alpha } } = x ^ { - 1 } C$ . Since C is a θ-class, it is θ-saturated, and the left quotient of a θ-saturated set by a fixed word is again θ-saturated. The nonempty set α is contained in the single class $[ { \overline { { \alpha } } } ] _ { \theta } .$ , so saturation forces equality. This also covers $\alpha = \varepsilon .$ , using $\overline { { \varepsilon } } = \{ \varepsilon \}$ and unit separation. □

Corollary 7.16 (Peeling). Under (R), let X be prime and let C be a live relative class. If $C = X { \overline { { \alpha } } }$ holds exactly and $X { \overline { { \beta } } } \subseteq C$ , then α is an exact relative class and ${ \overline { { \beta } } } \subseteq { \overline { { \alpha } } }$

Proof. Exactness is Corollary 7.15, and the inclusion is Corollary 7.13.

Definition 7.17 (Prime-irreducible sequence). A prime sequence $\beta ~ = ~ B _ { 1 } \cdot \cdot \cdot B _ { m }$ is primeirreducible if no contiguous block of length at least two has prime quotient product; that $i s ,$ for every $1 \leq i < j \leq m , [ \overline { { B _ { i } \cdot \cdot \cdot B _ { j } } } ] _ { \theta }$ is not a relative prime.

Lemma 7.18 (Prime contraction). Let $\beta$ be a prime sequence and replace a contiguous block of length at least two whose quotient is a prime Q by Q, obtaining $\beta ^ { \prime }$ . Then ${ \overline { { \beta } } } \subseteq { \overline { { \beta ^ { \prime } } } } \subseteq [ { \overline { { \beta } } } ] _ { \theta }$ and $[ \overline { { \beta ^ { \prime } } } ] _ { \theta } = [ \overline { { \beta } } ] _ { \theta }$ . Hence iterated contractions terminate in a prime-irreducible sequence with the same quotient class; $i f { \overline { { \beta } } }$ is already that whole class, every stage remains exact.

Proof. Replacing the block enlarges its setwise product to its quotient prime without changing quotient multiplication. Each contraction strictly decreases sequence length, so iteration terminates. □

Theorem 7.19 (Irreducible-path exactness). Assume (R). Let $C = { \overline { { A _ { 1 } \cdots A _ { k } } } }$ be any exact prime factorization of a live relative class C. If ${ \overline { { B _ { 1 } \cdots B _ { m } } } } \subseteq C$ and $B _ { 1 } \cdots B _ { m }$ is prime-irreducible, then $m = k$ and $B _ { i } = A _ { i }$ for every i. In particular, ${ \overline { { B _ { 1 } \cdots B _ { m } } } } = C$

Proof. Induct on k. If $k = 1$ , then $C = A _ { 1 }$ is prime. If $m \geq 2 .$ , the whole sequence $B _ { 1 } \cdots B _ { m }$ has prime quotient $C ,$ contradicting prime-irreducibility. Hence $m = 1$ , and the nonempty inclusion $B _ { 1 } \subseteq A _ { 1 }$ gives $B _ { 1 } = A _ { 1 }$

Now let $k \geq 2$ . By Lemma 7.11, some $j \geq 1$ satisfies ${ \overline { { B _ { 1 } \cdots B _ { j } } } } \subseteq A _ { 1 }$ . If $j \geq 2$ , that block has prime quotient $A _ { 1 }$ , contradicting prime-irreducibility. Hence $j = 1$ and $B _ { 1 } = A _ { 1 }$ . The case $m = 1$ is impossible, because then the prime class $B _ { 1 } = A _ { 1 }$ would be contained in the relative class C, forcing $C = A _ { 1 }$ , contrary to the nontrivial exact factorization of C. Thus both tails are nonempty. Corollary 7.16 gives ${ \overline { { B _ { 2 } \cdot \cdot \cdot B _ { m } } } } \subseteq { \overline { { A _ { 2 } \cdot \cdot \cdot A _ { k } } } }$ , and the right-hand side is an exact relative class. The induction hypothesis applies to the two tails. □

Corollary 7.20 (Exact prime factorizations are irreducible). Under (R), every exact prime factorization is prime-irreducible.

Proof. If an exact factorization were reducible, Lemma 7.18 would produce a strictly shorter prime-irreducible exact factorization of the same class. Theorem 7.19, applied to the original exact factorization and that terminal sequence, would force them to be identical, a contradiction.

Theorem 7.21 (Prime-target division factorization theorem). Assume that L is h-substitutable, unit-separated, and satisfies PTLD. Then every live non-unit relative class has a unique exact factorization into relative primes.

Proof. Existence is Theorem 4.4. If $C = \overline { { \alpha } } = \overline { { \beta } }$ are two exact prime factorizations, Corollary 7.20 makes $\beta$ prime-irreducible. Theorem 7.19, applied to the exact factorization α and the contained sequence $\beta ,$ gives $\alpha = \beta$ □

Theorem 7.22 (PTLD implies tail exactness and valid-tail determinism). Assume that L is h-substitutable, unit-separated, and satisfies PTLD. Then VTE holds, and hence $V T E _ { 1 }$ holds. If the relative prime spectrum has size $q < \infty$ , then VTD holds and $| \nu | \le q ^ { 2 }$

Proof. Let $N  \gamma \alpha$ be valid, with $\gamma , \alpha \neq \varepsilon$ . Every contiguous block of α of length at least two is a proper contiguous block of the full right-hand side γα. Hence validity makes α primeirreducible. Put $T ^ { \mathrm { t a i l } } : = [ \overline { { \alpha } } ] _ { \theta }$ . This class is live and non-unit, and Theorem 4.4 gives it an exact prime factorization. Since $\overline { { \alpha } } \subseteq T ^ { \mathrm { t a i l } }$ , Theorem 7.19 shows that α is that exact factorization. Thus $\overline { { \alpha } } = T ^ { \mathrm { t a i l } }$ , proving VTE.

For VTD, suppose $N  A \alpha$ and $N  A \beta$ are valid. VTE makes $R _ { \alpha } : = \overline { { \alpha } }$ and $R _ { \beta } : = \overline { { \beta } }$ exact live classes. For any $a \in A , r \in R _ { \alpha }$ , and $r ^ { \prime } \in R _ { \beta }$ , correctness gives $a r , a r ^ { \prime } \in N ;$ ; Lemma 7.7 therefore gives $r \theta r ^ { \prime }$ . Hence $R _ { \alpha } = R _ { \beta }$ , and unique factorization gives $\alpha = \beta$ . Thus there is at most one valid rule for each ordered pair $( N , A )$ , so $| \nu | \le q ^ { 2 }$ □

Corollary 7.23 (Prime interval DAG). Under (R), let C be a live non-unit relative class and let $w \in C$ . Form the directed acyclic graph with vertices $0 , \ldots , | w |$ , placing an edge $i  j$ exactly when the substring w $[ i : j ]$ belongs to a relative prime class. Then every minimum-edge path from 0 to |w| reads the unique exact prime factorization of C.

Proof. Any path determines a prime sequence whose setwise product is contained in $C ,$ because the chosen substrings concatenate to $w \in C$ and quotient multiplication returns the class C. A minimum-edge path cannot contain a length-at-least-two contiguous subpath with prime quotient: the corresponding whole interval would itself be a prime edge and would shorten the path. Theorem 7.19 therefore applies. □

Theorem 7.24 (Prime-sequence refinement). Assume (R). Let $C = { \overline { { A _ { 1 } \cdots A _ { k } } } }$ be the exact prime factorization of a live class. $I f \beta = B _ { 1 } \cdot \cdot \cdot B _ { m }$ is any prime sequence with ${ \overline { { \beta } } } \subseteq C .$ then the direct valid-rule grammar of Section 5 derives $A _ { 1 } \cdot \cdot \cdot A _ { k } \Rightarrow ^ { * } B _ { 1 } \cdot \cdot \cdot B _ { m }$

Proof. By Lemma 7.18, contract $\beta$ to a prime-irreducible sequence with quotient class C and setwise product contained in C. Theorem 7.19 identifies the terminal sequence with $A _ { 1 } \cdots A _ { k }$ Reversing the contractions gives correct prime expansions, and the Valid-Rule Reduction Lemma replaces each by valid productions. □

Example 7.25 (PTLD is strictly weaker than factor cancellation). Let $M = \{ 1 , a , 0 \}$ with $a ^ { 2 } = 0$ and 0 absorbing. Put $\Sigma = \{ x \} , h ( x ) = a$ , and $L = \{ x ^ { 2 } \}$ . The only live relative prime is $X = \{ x \}$ with type a. Equations $q r = a$ have at most one right solution for each q, so $P T L D$ holds. But $a \cdot a = 0 = a \cdot 0$ a ̸= 0, so factor-left-cancellation fails.

The noncancellative language $L _ { 0 }$ below shows that UF does not imply VTD or PTLD. The 36-element quotient witness in Section 12 will show the stronger separation $\mathrm { U F \neq F R P }$

## 8 Bounded-defect localization

The next mechanism yields unique exact factorization without cancellation.

Definition 8.1 (Bounded concatenation-defect statistic). A function $\nu : \Sigma ^ { * } \to \mathbb { N }$ has bounded concatenation defect ∆ if

$$
\nu ( x ) + \nu ( y ) \leq \nu ( x y ) \leq \nu ( x ) + \nu ( y ) + \Delta\tag{BD}
$$

for all x, $y \in \Sigma ^ { * }$

Iterating (BD) gives

$$
\sum _ { i = 1 } ^ { k } \nu ( x _ { i } ) \leq \nu ( x _ { 1 } \cdot \cdot \cdot x _ { k } ) \leq \sum _ { i = 1 } ^ { k } \nu ( x _ { i } ) + ( k - 1 ) \Delta .\tag{1}
$$

For a relative class X define its defect spectrum ${ \mathrm { S p e c } } _ { \nu } ( X ) : = \{ \nu ( w ) : w \in X \} , \qquad \underline { { \nu } } ( X ) : =$ min ${ \mathrm { S p e c } } _ { \nu } ( X )$ . Call a prime P neutral if $\mathrm { S p e c } _ { \nu } ( P ) = \{ 0 \}$ and positive if $\underline { { \nu } } ( P ) \geq 1$

Theorem 8.2 (Single-carrier lemma). Assume every relative prime is neutral or positive. Let X be a relative class such that $\underline { { \nu } } ( X ) = 1$ and ${ \mathrm { S p e c } } _ { \nu } ( X )$ is unbounded. Then every exact prime factorization of X contains exactly one positive prime. Moreover, that positive prime has unbounded defect spectrum.

Proof. If a factorization $X = P _ { 1 } \cdots P _ { k }$ contains no positive prime, then every factor has defect zero, so (1) gives a uniform bound $( k - 1 ) \Delta$ on ${ \mathrm { S p e c } } _ { \nu } ( X )$ , contradiction. If it contains at least two positive primes, the lower bound in (1) gives $\nu ( w ) \geq 2$ for every product word, contradicting $\underline { { \nu } } ( X ) = 1$ . Hence there is exactly one positive prime. If its spectrum were bounded, then (1) would again bound the spectrum of the entire product. □

For a fixed finite word z, the occurrence count $\nu _ { z } ( w ) = \# z ( w )$ satisfies (BD) with $\Delta = | z | - 1$ since only occurrences crossing the concatenation boundary can be newly created. The example below uses $z = b a$ and $\Delta = 1$

## 9 The nonregular example $L _ { 0 } = \{ a ^ { n } b ^ { n } : n \geq 0 \} ^ { * }$

Put $K _ { n } : = a ^ { n } b ^ { n } ( n \geq 1 ) , K : = \{ K _ { n } : n \geq 1 \} , L _ { 0 } : = K ^ { * }$ . For $w ~ \in ~ \{ a , b \} ^ { * }$ let fst $( w ) , \mathrm { l s t } ( w ) \in \{ \bot , a , b \}$ be the first and last letters, and let ba(w) record whether w contains the factor ba. We use the finite morphism $h ( w ) : = ( \mathrm { f s t } ( w ) , \mathrm { l s t } ( w ) , \mathrm { b a } ( w ) )$ , with the usual concatenation product on first/last symbols and the ba-bit. Moreover,

$$
h ( \varepsilon ) = ( \bot , \bot , 0 ) ,
$$

whereas every nonempty word has first and last symbols in $\{ a , b \}$ . Hence $h ^ { - 1 } ( h ( \varepsilon ) ) \ : = \ : \{ \varepsilon \}$ so $\left[ \varepsilon \right] _ { L _ { 0 } , h } = \left\{ \varepsilon \right\}$ and $( L _ { 0 } , h )$ is unit-separated. We prove h-substitutability here, together with the relative-class, factorization, and valid-rule analysis, so that this boundary example is selfcontained.

All claims about $L _ { 0 }$ used later are proved analytically in this section; no finite computation is used.

We shall repeatedly use the involution $\iota ( w ) : = \sigma ( w ^ { \mathrm { R } } )$ , where $w ^ { \mathrm { R } }$ is reversal and $\sigma ( a ) = b$ $\sigma ( b ) = a$ . It preserves $L _ { 0 }$ , reverses concatenation, and preserves the relative congruence. Hence every statement about a left fringe has a right-fringe dual.

For a word $x = x _ { 1 } \cdot \cdot \cdot x _ { n }$ , write $\beta _ { x } ( i ) : = | x _ { 1 } \cdot \cdot \cdot x _ { i } | _ { a } - | x _ { 1 } \cdot \cdot \cdot x _ { i } | _ { b }$

Lemma 9.1 (Block-height characterization). A word $w \in \{ a , b \} ^ { * }$ belongs to L<sub>0</sub> if and only if (i) $\beta _ { w } ( i ) \geq 0$ for every prefix and $\beta _ { w } ( | w | ) = 0 ;$

(ii) every occurrence $w _ { i } w _ { i + 1 } =$ ba satisfies $\beta _ { w } ( i ) = 0$

Consequently, if a live factor contains ba, all of its internal ba events occur at one common relative height.

Proof. Every block $K _ { n } = a ^ { n } b ^ { n }$ starts at height zero, stays nonnegative, and returns to zero; the only ba factors of a product of such blocks are block boundaries. This proves necessity. Conversely, split w at all zero-height positions. Between two consecutive such positions the relative height is positive at every proper prefix. Condition (ii) forbids ba inside such a segment, so the segment lies in $a ^ { * } b ^ { * }$ . Its total balance is zero, hence it is some $K _ { n }$ . Thus $w \in K ^ { * } = L _ { 0 }$ For the final statement, embed the live factor x in uxv $\in L _ { 0 }$ . Every internal ba of x must occur at absolute height zero, hence at the same relative height $- \beta _ { u } ( | u | )$ inside x. □

Proposition 9.2 (h-substitutability of $L _ { 0 } )$ . The language L<sub>0</sub> is h-substitutable.

Proof. Write $\beta ( x ) = | x | _ { a } - | x | _ { b }$ . If a live factor x contains ba, Lemma 9.1 shows that all its internal ba events occur at one relative height; denote it by $c ( x )$ . If x contains no $b a .$ , put $c ( x ) = \perp$

We first record a consequence of the height characterization. For every live factor x, its twosided distribution is determined by the triple $\left( h ( x ) , \beta ( x ) , c ( x ) \right)$ . Indeed, for arbitrary $\ell , r$ , the criterion for $\ell x r \in L _ { 0 }$ consists of total balance zero, nonnegativity of all prefixes, and height zero at every ba boundary. The total-balance contribution of x is $\beta ( x )$ . If $c ( x ) = \bot$ , then $x \in a ^ { * } b ^ { * }$ and its least relative prefix height is determined by $\beta ( x )$ together with its first and last symbols. If $c ( x ) \neq \perp$ , all internal ba events occur at height $c ( x )$ and the least relative prefix height is $c ( x )$ ; the possible ba events across the two external boundaries are determined by the first and last symbols recorded by $h ( x )$ . Hence every clause of the characterization depends only on the displayed triple and on $\ell , r$

Now suppose that $h ( x ) = h ( y )$ and that uxv, uyv $\in { \cal L } _ { 0 }$ . Total balance gives $\beta ( x ) = \beta ( y )$ The ba-bit in h says that either neither factor contains ba or both do. In the latter case, acceptance of the two displayed words forces every internal ba event to absolute height zero, whence $c ( x ) = - \beta ( u ) = c ( y )$ . Thus x and y have the same determining triple and therefore the same two-sided distribution. This is exactly h-substitutability. □

## 9.1 Complete normal forms for live relative classes

For $r , s \geq 1$ and $d \in \mathbb { Z }$ put $A _ { r } : = \{ a ^ { r } \} , \qquad B _ { s } : = \{ b ^ { s } \} , C _ { d } : = \{ a ^ { i } b ^ { j } : i , j \geq 1 , \ i - j = d \}$ . For ba-containing factors define

$$
X _ { c , e } : = b ^ { c } L _ { 0 } a ^ { e } ~ ( c , e \geq 1 ) ,
$$

$$
Y _ { c , e } : = b ^ { c } L _ { 0 } a ^ { e } K ( c \geq 1 , e \geq 0 ) ,
$$

$$
Z _ { c , e } : = K b ^ { c } L _ { 0 } a ^ { e } ~ ( c \geq 0 , e \geq 1 ) ,\tag{2}
$$

$$
W _ { c , e } : = K b ^ { c } L _ { 0 } a ^ { e } K \quad ( c , e \geq 0 ) .
$$

Lemma 9.3 (Live-class normal forms). The live relative classes of $( L _ { 0 } , h )$ are exactly

$$
[ \mathcal { E } ] _ { \theta } , \qquad A _ { r } \ ( r \geq 1 ) , \qquad B _ { s } \ ( s \geq 1 ) , \qquad C _ { d } \ ( d \in \mathbb { Z } ) ,
$$

together with the four families in (2). Every word in one of $X _ { c , e } , Y _ { c , e } , Z _ { c , e } , W _ { c , e }$ has total balance $e - c _ { \mathrm { { i } } }$ , and every internal occurrence of ba occurs at relative $h e i g h t - c$

Proof. Let $x$ be a live factor and choose an occurrence of $x$ inside a word $K _ { n _ { 1 } } \cdot \cdot \cdot K _ { n _ { t } } \in L _ { 0 }$ . If x contains no ba, then $x \in a ^ { * } b ^ { * }$ . Thus $x$ is either a pure power $a ^ { r }$ , a pure power $b ^ { s } ,$ or $a ^ { i } b ^ { j }$ with $i , j \geq 1$ . The first two cases give $A _ { r }$ and $B _ { s }$ . In the third case the relative class is indexed by the balance $d = i - j$ , giving $C _ { d }$

Suppose now that x contains ba. At its left edge, either the occurrence starts in the descending b-part of a block, contributing $b ^ { c }$ with $c \geq 1$ , or it starts in the ascending a-part of a block. In the latter case the remainder of that block has the form $a ^ { r } b ^ { n } = K _ { r } b ^ { n - r }$ , so the left fringe is $K b ^ { c }$ with $c \geq 0$ . Dually, at the right edge the occurrence either ends in an ascending a-part, giving $a ^ { e }$ with $e \geq 1$ , or in a descending b-part, in which case $a ^ { n } b ^ { r } = a ^ { n - r } K _ { \ i }$ <sub>r</sub> and the right fringe is $a ^ { e } K$ with $e \geq 0$ . All complete blocks strictly between the two fringes form an arbitrary word of $L _ { 0 }$ . This gives exactly the four forms in (2).

It remains to verify that the displayed sets are precisely relative classes. For $C _ { d } ,$ all words have the same h-type. If $d \geq 0$ , then $x b ^ { d } \in K \subseteq L _ { 0 } \qquad ( x \in C _ { d } )$ , while for $d < 0 , a ^ { - d } x \in K$ Hence all members of $C _ { d }$ share an accepting context and are syntactically congruent by $h -$ substitutability. The same context distinguishes $C _ { d }$ from $C _ { d ^ { \prime } }$ when $d ^ { \prime } \neq d .$ Likewise, $A _ { r }$ and $A _ { r ^ { \prime } }$ are distinguished by the right context $b ^ { r }$ , and $B _ { s } , B _ { s ^ { \prime } }$ by the left context $a ^ { s }$

For each of the four ba-families, every member shares the context $( a ^ { c } , b ^ { e } )$ . Indeed, for $z \in L _ { 0 }$ and $K _ { n } \in K$ 2

$$
\begin{array} { c } { { a ^ { c } ( b ^ { c } z a ^ { e } ) b ^ { e } = K _ { c } z K _ { e } , } } \\ { { { } } } \\ { { a ^ { c } ( b ^ { c } z a ^ { e } K _ { n } ) b ^ { e } = K _ { c } z K _ { e + n } , } } \\ { { { } } } \\ { { a ^ { c } ( K _ { n } b ^ { c } z a ^ { e } ) b ^ { e } = K _ { c + n } z K _ { e } , } } \end{array}
$$

and the fourth case combines the last two identities. Thus every set in (2) lies in one relative class.

The first/last/ba observation separates the four families from one another. Within one fixed family, every ba event occurs after a relative balance drop of exactly $^ { c , }$ hence at height $- c ,$ and every word has total balance $e - c$ . If a word from parameters $( c ^ { \prime } , e ^ { \prime } )$ were accepted in the context $( a ^ { c } , b ^ { e } )$ , its first internal ba event would occur at absolute height $c - c ^ { \prime }$ and therefore, by Lemma 9.1, at height zero. Hence $c = c ^ { \prime }$ . The total balance of the completed word then forces $e = e ^ { \prime }$ . Thus diferent parameters give diferent relative classes, completing the classification.

## 9.2 Seven primes and exact fringe identities

Set

$$
\begin{array} { l l l } { { A : = A _ { 1 } = \{ a \} , } } & { { B : = B _ { 1 } = \{ b \} , } } & { { C : = C _ { 0 } = K , } } \\ { { D : = X _ { 1 , 1 } = b L _ { 0 } a , } } & { { E : = Y _ { 1 , 0 } = b L _ { 0 } K = b K ^ { + } , } } & { { } } \\ { { F : = Z _ { 0 , 1 } = K L _ { 0 } a = K ^ { + } a , } } & { { G : = W _ { 0 , 0 } = K L _ { 0 } K = K ^ { \geq 2 } . } } & { { } } \end{array}
$$

The involution satisfies $\iota ( A ) = B , \ \iota ( B ) = A , \ \iota ( E ) = F , \ \iota ( F ) = E$ , while $C , D , G$ are fixed.

Lemma 9.4 (Exact fringe identities). The no-ba classes satisfy $A _ { r } = A ^ { r } , \qquad B _ { s } = B ^ { s }$ , and

$$
\begin{array} { r } { C _ { d } = \left\{ \begin{array} { l l } { A ^ { d } C , } & { d > 0 , } \\ { C , } & { d = 0 , } \\ { C B ^ { - d } , } & { d < 0 . } \end{array} \right. } \end{array}\tag{3}
$$

The ba-classes satisfy the exact setwise identities

$$
X _ { c , e } = B ^ { c - 1 } D A ^ { e - 1 } ,\tag{4}
$$

$$
Y _ { c , e } = \left\{ \begin{array} { l l } { B ^ { c - 1 } E , } & { e = 0 , } \\ { B ^ { c - 1 } D A ^ { e - 1 } C , } & { e \geq 1 , } \end{array} \right.\tag{5}
$$

$$
\begin{array} { r } { Z _ { c , e } = \left\{ \begin{array} { l l } { F A ^ { e - 1 } , } & { c = 0 , } \\ { C B ^ { c - 1 } D A ^ { e - 1 } , } & { c \geq 1 , } \end{array} \right. } \end{array}\tag{6}
$$

and

$$
W _ { c , e } = \left\{ \begin{array} { l l } { G , } & { c = e = 0 , } \\ { F A ^ { e - 1 } C , } & { c = 0 , e \geq 1 , } \\ { C B ^ { c - 1 } E , } & { c \geq 1 , e = 0 , } \\ { C B ^ { c - 1 } D A ^ { e - 1 } C , } & { c , e \geq 1 . } \end{array} \right.\tag{7}
$$

Proof. Each identity follows by multiplying the defining sets. For example, $B ^ { c - 1 } D A ^ { e - 1 } =$ $b ^ { c - 1 } ( b L _ { 0 } a ) a ^ { e - 1 } = b ^ { c } L _ { 0 } a ^ { e } = X _ { c , e }$ , and $C B ^ { c - 1 } D A ^ { e - 1 } C = K b ^ { c } L _ { 0 } a ^ { e } K = W _ { c , e }$ . The remaining cases are identical one-line calculations; the identities for $C _ { d }$ use $A ^ { d } K = \{ a ^ { n + d } b ^ { n } : n \geq 1 \}$ and its right-hand dual. □

Theorem 9.5 (Prime classification for $L _ { 0 } )$ . The live relative prime spectrum is exactly $\mathcal { P } _ { L _ { 0 } , h } =$ $\{ A , B , C , D , E , F , G \}$

Proof. Lemma 9.4 makes every listed live class other than $A , B , C , D , E , F , G$ an exact product of at least two non-unit classes. It remains to prove that these seven classes are prime. The singleton classes A and B are immediate.

For the remaining five classes we use Lemma 4.2. The shortest word of C is $a b .$ . Its only nontrivial cut gives $\boldsymbol { A } \cdot \boldsymbol { B } = \{ a b \} \neq \boldsymbol { C }$ (for instance $a a b b \in C )$ , so C is prime. The shortest word of D is ba, and $\boldsymbol { B } \cdot \boldsymbol { A } = \{ b \boldsymbol { a } \} \neq D$ (for instance baba $\in D )$

The shortest word of $E$ is bab. Its two cuts give $B \cdot C = b K \subsetneq b K ^ { + } = E$ and $D \cdot B =$ $b L _ { 0 } a b \subsetneq b K ^ { + } = E$ . For example babab witnesses strictness of the first inclusion and baabb witnesses strictness of the second. Hence E is prime.

Since $\iota ( E ) = F$ and ι preserves exact decomposability, primality of F follows from primality of E. Finally, the shortest word of G is abab. Its three cuts give $A \cdot E = a b K ^ { + } \subsetneq K ^ { \geq 2 }$ $C \cdot C = K ^ { 2 } \subsetneq K ^ { \geq 2 } , \qquad F \cdot B = K ^ { + } a b \subsetneq K ^ { \geq 2 }$ . These inclusions are strict: for example $K _ { 2 } K _ { 1 } = a a b b a b \in G \backslash A E$ ， $K _ { 1 } K _ { 1 } K _ { 1 } = a b a b a b \in G \backslash { C C }$ , and $K _ { 1 } K _ { 2 } = a b a a b b \in G \setminus F B$ . No nontrivial cut of a shortest word can therefore induce an exact binary product, so G is prime.

## 9.3 Unique factorization without cancellation

Let $\nu ( w ) : = \# b a ( w )$ . Then $\nu ( x y ) = \nu ( x ) + \nu ( y ) + [ \mathrm { l s t } ( x ) = b \ \& \ \mathrm { f s t } ( y ) = a ] .$ so $\Delta \ : = \ : 1$ in (BD). The primes A, B, C are neutral, while $\operatorname { S p e c } _ { \nu } ( D ) = \operatorname { S p e c } _ { \nu } ( E ) = \operatorname { S p e c } _ { \nu } ( F ) = \operatorname { S p e c } _ { \nu } ( G ) =$ $\{ 1 , 2 , 3 , \ldots \}$ . Every class in (2) has the same unbounded positive spectrum with minimum one. Hence Theorem 8.2 implies that every exact factorization of a ba-class contains exactly one of $D , E , F , G$

Lemma 9.6 (Neutral-fringe lemma). A sequence over $\{ A , B , C \}$ whose factor boundaries introduce no new ba has the form $A ^ { p } C ^ { \epsilon } B ^ { q } , \qquad p , q \geq 0 , \quad \epsilon \in \{ 0 , 1 \}$

Proof. A B cannot be followed by A or C, because both begin with a. A C cannot be followed by A or by another C, because $C$ ends with b. Hence all $A \mathrm { { } ^ { \circ } s }$ precede the optional $C ,$ and all $B ^ { \prime } \mathrm { s }$ follow it; at most one $C$ can occur. □

Lemma 9.7 (Carrier-fringe rigidity). Let $T$ be one of the classes in (2), and suppose $T =$ $P _ { 1 } \cdots P _ { k }$ is an exact prime factorization. Then its unique positive carrier and its neutral fringes are exactly those displayed in $( 4 ) - ( 7 )$

Proof. Because T contains words with exactly one ba, no boundary between adjacent prime factors may itself be a ba boundary: otherwise every word of the exact product would contain at least the carrier event and the additional boundary event. Thus Lemma 9.6 applies to the neutral factors on either side of the unique carrier.

The classes D, E have all their internal ba events at relative height −1, whereas $F , G$ have them at height 0. A neutral prefix changes that height only by its total balance. The target class in (2) has event height −c. If the carrier is F or G, the no-ba condition forces every preceding neutral factor to be an $A ;$ ; the event-height equation then reads $p = - c .$ so $p = c = 0$ . Thus $F , G$ can occur only when $c = 0$ , with no neutral prefix.

If the carrier is D or E, a no-ba neutral prefix has the form $A ^ { p } C ^ { \epsilon } B ^ { q }$ . When the target begins with $b ,$ necessarily $p = \epsilon = 0$ . When it begins with $^ { a , }$ exact setwise equality must realize an arbitrary leading block $K _ { n } \ ( n \geq 1 )$ . A fixed string of $A \mathrm { { } s }$ and $B "$ s cannot supply this free block; the only possible neutral source is one copy of $C = K$ . Moreover $p > 0$ would force every initial a-run to have length at least $p + 1$ , missing the target words whose free leading block is $K _ { 1 }$ Hence $p = 0$ and $\epsilon = 1$ . In either case the event-height equation is $- q - 1 = - c ,$ so $q = c - 1$ Therefore the left fringe is empty when $c = 0 .$ , is $B ^ { c - 1 }$ for an initial-b class, and is $C B ^ { c - 1 }$ when a free leading K is present.

The right-fringe statement is the ι-dual of the preceding left-fringe argument. Thus, if the carrier is $E$ or $G$ , exactness forces $e = 0$ and no extra neutral sufix; if the carrier is D or $F$ the sufix is $A ^ { e - 1 }$ when the target ends in $^ { a , }$ and $A ^ { e - 1 } C$ when the target has a free trailing $K$

Consequently the carrier is determined by whether c and e vanish: $( c > 0 , e > 0 ) : D , \quad ( c >$ $0 , e = 0 ) : E , \quad ( c = 0 , e > 0 ) : F , \quad ( c = e = 0 ) : G$ , and the surrounding neutral factors are exactly those in $( 4 ) - ( 7 )$ □

Theorem 9.8 (Noncancellative unique relative factorization). Every live non-unit relative class of $L _ { 0 }$ has a unique exact prime factorization, namely the decompositions in (3)–(7).

Proof. For a singleton class $A _ { r }$ or $B _ { s } ,$ every exact prime factor must have the same one-letter orientation, so shortest-length additivity forces $A _ { r } = A ^ { r }$ and $B _ { s } = B ^ { s }$

Now let $C _ { d }$ be a mixed no-ba class. Any exact factorization uses only neutral primes, and no factor boundary may create ba. By Lemma 9.6 its prime sequence has the form $A ^ { p } C ^ { \epsilon } B ^ { q }$ Since $C _ { d }$ is infinite, $\epsilon = 1 \AA$ ; otherwise the product of the singleton classes A and B would itself be a singleton. Balance gives $p - q \ = \ d .$ Moreover $\ell ( C _ { d } ) = | d | + 2$ and Lemma 4.2 gives $p + q + 2 = | d | + 2$ . Hence $p = \operatorname* { m a x } ( d , 0 ) , \qquad q = \operatorname* { m a x } ( - d , 0 )$ , which is exactly (3).

For a ba-class, Theorem 8.2 gives exactly one positive prime and Lemma 9.7 forces the carrier and both neutral fringes uniquely. The identities are exact by Lemma 9.4. □

The observer monoid is nevertheless not factor-cancellative: the first/last/ba type of a equals that of aa, so $h ( a ) h ( \varepsilon ) = h ( a ) = h ( a ) h ( a ) , \qquad h ( \varepsilon ) \neq h ( a )$ . Thus cancellation is suficient for the Clark-style rigidity mechanism but is not necessary for unique exact relative factorization.

## 9.4 The thirty-four valid productions

For a prime P define interface bits $c ( P ) : = [ \mathrm { f s t } ( P ) = b ] , \qquad e ( P ) : = [ \mathrm { l s t } ( P ) = a ]$ . Their values, together with the balance $d ( P ) = e ( P ) - c ( P )$ , are:

<table><tr><td>Prime</td><td>shortest representative</td><td> $c ( P )$   $e ( P )$ </td><td> $d ( P )$ </td></tr><tr><td>A</td><td>a</td><td>0 1</td><td>+1</td></tr><tr><td>B</td><td>b</td><td>1 0</td><td>-1</td></tr><tr><td>C</td><td>ab</td><td>0 0</td><td>0</td></tr><tr><td>D</td><td>ba</td><td>1 1</td><td>0</td></tr><tr><td>E</td><td>bab</td><td>1 0</td><td>-1</td></tr><tr><td>F</td><td>aba</td><td>0 1</td><td>+1</td></tr><tr><td>G</td><td>abab</td><td>0 0</td><td>0</td></tr></table>

Lemma 9.9 (Binary interface lemma). For primes $P , Q \in \mathcal { P } _ { L _ { 0 } , h } , \ [ P Q ] _ { \theta }$ is prime if and only if $e ( P ) = c ( Q )$

Proof. Take shortest representatives of P and Q. Lemma 9.3 determines the quotient class and Theorem 9.5 its primality. The prime cases are exactly the nine pairs with $e ( P ) = c ( Q ) = 1$ 2 namely $P \in \{ A , D , F \}$ and $Q \in \{ B , D , E \}$ , and the sixteen pairs with $e ( P ) = c ( Q ) = 0$ □

Every correct length-two production is automatically valid, so Lemma 9.9 gives exactly twenty-five binary valid productions. To control longer rules we use the common event height of the ba-classes.

Lemma 9.10 (Internal-C lemma). Let $\alpha = P _ { 1 } \cdot \cdot \cdot P _ { n } , \qquad n \geq 3$ , be a prime sequence whose quotient product is a prime. If every adjacent pair has nonprime quotient product, then some internal symbol $P _ { i } \ ( 1 < i < n )$ is C.

Proof. Write $c _ { i } = c ( P _ { i } ) , e _ { i } = e ( P _ { i } )$ , and $d _ { i } = e _ { i } - c _ { i }$ . By the binary interface lemma, adjacent nonprimality gives $c _ { i + 1 } = 1 - e _ { i } \qquad ( 1 \leq i < n )$ . Let $\begin{array} { r } { s _ { i - 1 } : = \sum _ { i < i } d _ { j } , \qquad \eta _ { i } : = s _ { i - 1 } - c _ { i } } \end{array}$ . The quantity $\eta _ { i }$ is the whole-word relative height of the internal ba event carried by $P _ { i }$ when such an event is present. By the prime table above, the balance of every prime R is $e ( R ) - c ( R )$ Then $\eta _ { i + 1 } - \eta _ { i } = d _ { i } - c _ { i + 1 } + c _ { i } = 2 e _ { i } - 1 \in \{ - 1 , + 1 \}$ . The first symbol of the quotient prime is the first symbol of $P _ { 1 }$ and its last symbol is the last symbol of $P _ { n }$ . Correctness therefore gives $\textstyle \sum _ { i = 1 } ^ { n } d _ { i } = e _ { n } - c _ { 1 }$ . Equivalently, $\eta _ { n } = \eta _ { 1 }$ . Thus $\eta _ { 1 } , \ldots , \eta _ { n }$ is a nontrivial closed walk with unit steps.

If the quotient prime is one of $A , B , C .$ , then no $P _ { i }$ can be a carrier, because a carrier already has ba = 1 and the mismatching interfaces create no new ba boundary. If the quotient prime is one of $D , E , F , G ,$ , every carrier $P _ { i }$ in the sequence has all of its internal ba events at whole-word relative height $\eta _ { i }$ . The target prime has all such events at height $\eta _ { 1 } = - c _ { 1 }$ , so every carrier must occur at a position with $\eta _ { i } = \eta _ { 1 }$

The closed walk cannot go below its initial level. Otherwise an internal strict local minimum would have incoming step −1 and outgoing step +1, hence $c _ { i } = e _ { i } = 1$ . The only such prime is $D ,$ a carrier at a level diferent from $\eta _ { 1 }$ , contradiction (and in the neutral target case carriers are impossible altogether). Therefore the walk has a positive excursion. At an internal strict local maximum, the incoming step is +1 and the outgoing step is −1, so $c _ { i } = e _ { i } = 0$ . Hence $P _ { i } \in \{ C , G \}$ . The level is strictly above $\eta _ { 1 }$ , so $G$ cannot occur there as a carrier; in the neutral case it is excluded for the same reason as above. Thus $P _ { i } = C$ □

Theorem 9.11 (FRP for $L _ { 0 } )$ . For the first/last/ba observation, $| \mathcal { P } _ { L _ { 0 } , h } | = 7 , \qquad | \mathcal { V } _ { L _ { 0 } , h } | =$ $3 4 , \qquad \rho ( L _ { 0 } , h ) = 3$

Proof. A valid right-hand side of length at least three cannot contain a prime-valued adjacent pair, since such a pair would be a proper pleonastic block. Hence all adjacent interfaces mismatch, and Lemma 9.10 supplies an internal C. If $P _ { i - 1 } C P _ { i + 1 }$ surrounds such an occurrence, mismatch on the left forces $e ( P _ { i - 1 } ) = 1$ , so $P _ { i - 1 } \in \{ A , D , F \}$ , while mismatch on the right forces $c ( P _ { i + 1 } ) = 1$ , so $P _ { i + 1 } \in \{ B , D , E \}$ . The nine possible triples in $\{ A , D , F \} C \{ B , D , E \}$ all return to primes by the normal forms; they are exactly the ternary rules displayed in the complete grammar below. Hence a right-hand side of length greater than three contains a proper prime-returning triple and is pleonastic, while length three yields exactly those nine valid rules. Together with the twenty-five binary rules this gives $| \mathcal { V } | = 2 5 + 9 = 3 4$ and $\rho = 3$ □

The thirty-four branching rules are therefore

$$
C \to A B \mid A C B ,
$$

$$
D \to B A \mid B F \mid D D \mid E A \mid E F \mid D C D ,
$$

$$
E  B C \mid B G \mid D B \mid D E \mid E C \mid E G \mid D C B \mid D C E ,
$$

$$
F  A D \mid C A \mid C F \mid F D \mid G A \mid G F \mid A C D \mid F C D ,
$$

$$
G  A E \mid C C \mid C G \mid F B \mid F E \mid G C \mid G G \mid A C E \mid F C B \mid F C E .
$$

Together with $A \to a , B \to b$ , and $S \to \varepsilon \mid C \mid G ,$ , these form the direct relative-prime grammar for $L _ { 0 }$

Proposition 9.12 (Tail saturation for $L _ { 0 } )$ . Every valid production in the direct relative-prime grammar for $L _ { 0 }$ is tail-saturated.

Proof. Every binary valid production is automatically tail-saturated. In the nine ternary rules, the sufix after the first prime is one of $C B , \qquad C D , \qquad C E$ . The normal forms give the exact identities $C \cdot B = C _ { - 1 } , \qquad C \cdot D = Z _ { 1 , 1 } , \qquad C \cdot E = W _ { 1 , 0 }$ . Thus every ternary tail is also exact. □

Remark 9.13 (UF does not imply VTD). The two valid rules $D  B A , \qquad D  B F$ have the same left-hand prime and the same first right-hand prime but diferent tails. Hence $L _ { 0 }$ has exact UF and tail saturation while failing VTD and $P T L D$

## 10 Finite-state relative presentation

## 10.1 Definition and finite-quotient regularity

FRP demands a finite list of valid rules. For finite grammar construction this is stronger than necessary: an infinite valid family can be represented by a finite-state controller.

For each prime P, define ${ \mathrm { C o r r } } _ { P } : = \{ \alpha \in { \mathcal { P } } ^ { \geq 2 } : [ \overline { { \alpha } } ] = P \} , \qquad { \mathrm { C o r r } } : = \bigcup _ { P } { \mathrm { C o r r } } _ { P }$ . Then the pleonastic words are exactly

$$
\operatorname { P l e o n } = { \mathcal { P } } ^ { + } \operatorname { C o r r } { \mathcal { P } } ^ { * } \cup { \mathcal { P } } ^ { * } \operatorname { C o r r } { \mathcal { P } } ^ { + } ,\tag{8}
$$

and

$$
\mathrm { V a l } _ { P } = \mathrm { C o r r } _ { P } \backslash \mathrm { P l e o n } .
$$

Definition 10.1 (Finite-state relative presentation property). A pair $( L , \theta )$ has the finite-state relative presentation property $\left( F S R P \right) \ i f \left| \mathcal { P } \right| < \infty$ and every Val $_ P \subseteq { \mathcal { P } } ^ { \geq 2 }$ is regular. We write $\mathrm { F S R P } ( L , \theta )$ for this property and abbreviate it by $\mathrm { F S R P } ( L , h )$ when $\theta = \theta _ { L , h }$

Clearly $\mathrm { F R P } \implies \mathrm { F S R P }$ . The converse fails by the 36-element finite-quotient example in Section 12.

Theorem 10.2 (Finite quotient implies FSRP). $I f S = \Sigma ^ { * } / \theta$ is finite, then FSRP holds.

Proof. The evaluation morphism $\mu : { \mathcal { P } } ^ { * } \to S$ has finite codomain. Hence Corr $_ P = \mu ^ { - 1 } ( P ) \cap \mathcal { P } ^ { \ge 2 }$ is regular for every prime $P ,$ so Corr is regular. Equation (8) and closure of regular languages under concatenation, union, and diference imply regularity of every $\mathrm { V a l } _ { P }$ □

Proposition 10.3 (Context-freeness before factor-minimalization). Let L be context-free and h-substitutable, and assume that the relative prime spectrum $\mathcal { P }$ is finite. Then every live relative class is context-free, every correct-return language Corr is context-free, and therefore Corr and Pleon are context-free.

Proof. Let $C = [ u ] _ { L , h }$ be a live relative class and choose an accepting context $( x , y )$ with $x u y \in L$ By h-substitutability, a word v lies in C exactly when it has the same observer type as u and occurs in this same accepting context. Hence

$$
C = h ^ { - 1 } ( h ( u ) ) \cap x ^ { - 1 } L y ^ { - 1 } ,
$$

where $x ^ { - 1 } L y ^ { - 1 } = \{ v \ : \ x v y \in L \}$ . The second language is context-free because context-free languages are closed under left and right quotient by fixed words, while $h ^ { - 1 } ( h ( u ) )$ is regular because h has finite codomain. Thus $C$ is context-free.

For each prime $Q \in { \mathcal { P } }$ , choose a representative word $\omega ( Q ) \in Q$ and extend $Q \mapsto \omega ( Q )$ to a morphism $\phi : { \mathcal { P } } ^ { * } \to \Sigma ^ { * }$ . For every prime sequence $\alpha _ { \mathrm { { i } } }$ , the word $\phi ( \alpha )$ lies in the quotient product class [α]. Consequently

$$
\mathrm { C o r r } _ { P } = \phi ^ { - 1 } ( P ) \cap \mathcal { P } ^ { \geq 2 } .
$$

Since P is context-free, inverse homomorphism and intersection with the regular language $\mathcal { P } ^ { \ge 2 }$ show that Corr<sub>P</sub> is context-free. Finiteness of $\mathcal { P }$ gives context-freeness of ${ \mathrm { C o r r } } = \bigcup _ { P }$ Corr<sub>P</sub>, and Equation (8) then gives context-freeness of Pleon by closure under concatenation with regular languages and finite union. □

Remark 10.4 (The unresolved FSRP boundary). Proposition 10.3 localizes the remaining issue. Under the context-free, h-substitutable, finite-prime hypotheses, nonregularity cannot arise already at the level of correct returns: each Corr<sub>P</sub> and the global pleonastic language are contextfree. FSRP asks whether the factor-minimal diference

$$
\mathrm { V a l } _ { P } = \mathrm { C o r r } _ { P } \setminus \mathrm { P l e o n }
$$

is regular. Context-free languages are not closed under diference, so the proposition alone gives no regularity conclusion. At present, we are not aware of an example in this setting for which some Val<sub>P</sub> is nonregular; all explicit finite-prime examples in this paper satisfy FSRP. Thus the possibility that FSRP follows automatically from context-freeness, h-substitutability, and finite relative prime spectrum is not ruled out here.

The preceding finite-quotient regularity theorem is abstract: it follows immediately from quotient evaluation. The following automaton is retained for the stronger algorithmic tasks of deciding FRP, computing the validity radius, and extracting pumping certificates.

## 10.2 Prime-avoidance automata for finite quotients

When the relative quotient $S : = \Sigma ^ { * } / \theta$ is finite, valid RHSs can be recognized by a finite automaton intrinsic to quotient multiplication.

For a nonempty prime word $x = P _ { 1 } \cdots P _ { j }$ define its total product $t ( x ) : = P _ { 1 } * \cdot \cdot \cdot * P _ { j }$ and the set of products of proper nonempty sufixes beginning after the first symbol, $U ( x ) : = { \big . }$ $\{ P _ { i } * \cdot \cdot \cdot * P _ { j } : 2 \leq i \leq j \}$ . For $j = 1$ , set $U ( x ) = \emptyset$ . A state is a pair $( t , U )$

Appending a prime Q changes the sufix profile to $U ^ { \prime } = \{ Q \} \cup \{ u * Q : u \in U \}$ . A safe transition $( t , U ) \stackrel { Q } { \longrightarrow } ( t * Q , U ^ { \prime } )$ is permitted if

$$
t \ast Q \notin \mathcal { P } , \qquad \{ u \ast Q : u \in U \} \cap \mathcal { P } = \emptyset .\tag{9}
$$

A final edge labelled Q is permitted if

$$
t \ast Q \in \mathcal { P } , \qquad \{ u \ast Q : u \in U \} \cap \mathcal { P } = \emptyset .\tag{10}
$$

The initial states are $( P , \emptyset )$ for $P \in { \mathcal { P } }$

Theorem 10.5 (Prime-avoidance automaton). A prime word $P _ { 1 } \cdot \cdot \cdot P _ { k } , k \ge 2$ , is a valid RHS if starting from $( P _ { 1 } , \emptyset )$ , the symbols $P _ { 2 } , \ldots , P _ { k - 1 }$ can be read along safe transitions and $P _ { k }$ along a final edge.

Proof. Inductively, the state records the quotient products of every sufix that could become a proper contiguous block ending at the current position. Condition (9) prevents both the whole current prefix and every newly completed proper sufix of length at least two from being primevalued. Condition (10) changes only the whole word requirement: at the last symbol the total product must return to a prime, while every proper sufix must remain nonprime. Earlier proper intervals were already ruled out when their right endpoints were processed. □

The automaton recognizes the global valid language Val. To isolate $\mathrm { V a l } _ { P }$ , retain only final edges whose total quotient product is the target prime P.

The state set has size at most $| S | 2 ^ { | S | }$ , so it is finite whenever $S$ is finite.

Theorem 10.6 (Productive-cycle criterion). Assume S is finite. Restrict the safe-transition graph to states reachable from an initial state and from which a final edge is reachable. Then $\mathrm { F R P } \iff$ this productive safe graph is acyclic.

Proof. A productive cycle can be iterated arbitrarily many times before taking a path to a final edge, giving arbitrarily long valid RHSs. Conversely, if the productive graph is acyclic, every productive safe path has bounded length, and the finite prime alphabet then yields only finitely many valid words. □

Corollary 10.7. Suppose the finite relative quotient is given efectively by its multiplication table together with the finite set of live prime classes. Then FRP is decidable from this finite data, and the validity radius is computable.

Proof. Construct the finite prime-avoidance automaton from the multiplication table and the distinguished prime subset, and test the productive subgraph for a directed cycle. In the acyclic case its longest productive path gives the validity radius. □

Corollary 10.8 (Pumping certificate). If the finite-quotient pair fails FRP, there exist prime words u, v, w with $v \neq \varepsilon$ such that $u v ^ { n } w$ is valid for every $n \geq 0$

Proof. Take a path into a productive cycle, one nonempty traversal of the cycle, and a path from it to a final edge; their labels give $u , v , w$ □

Section 12 gives an explicit one-state productive loop for the family $P _ { a b }  P _ { a } P _ { q } ^ { m } P _ { b }$

## 10.3 Canonical residual controller

For each prime P and prefix $u \in \mathcal { P } ^ { * }$ , take the left quotient $u ^ { - 1 } \mathrm { V a l } _ { P } : = \{ v : u v \in \mathrm { V a l } _ { P } \}$ . Let

$$
\mathcal { Q } _ { L , \theta } : = \{ u ^ { - 1 } \mathrm { V a l } _ { P } : P \in \mathcal { P } , \ u \in \mathcal { P } ^ { * } , \ u ^ { - 1 } \mathrm { V a l } _ { P } \neq \emptyset \} .
$$

Definition 10.9 (Presentation state complexity). Define $\kappa ( L , \theta ) : = | \mathcal { Q } _ { L , \theta } | \in \mathbb { N } \cup \{ \infty \}$ . In the fixed-h case write $\mathcal { Q } _ { L , h } : = \mathcal { Q } _ { L , \theta _ { L , h } }$ and $\kappa ( L , h ) : = \kappa ( L , \theta _ { L , h } )$

Theorem 10.10 (Residual characterization of FSRP). Assuming $| \mathcal { P } | < \infty$ 2

$$
\mathrm { F S R P } ( L , \theta ) \iff \kappa ( L , \theta ) < \infty .
$$

Moreover, the nonempty residual languages above form the canonical minimal deterministic partial multi-start controller for the family $( \mathrm { V a l } _ { P } ) _ { P \in \mathcal { P } }$ . Adjoining the empty residual ∅ as a single dead state gives the corresponding canonical complete controller.

Proof. This is the Myhill–Nerode theorem applied simultaneously to the finite family of regular languages $\mathrm { V a l } _ { P } .$ , taking residual languages themselves as states. In the usual complete realization, every empty left quotient is represented by the single dead state $\alpha ;$ deleting that state and the transitions entering it yields exactly the partial controller displayed above. Minimality and canonicity therefore follow from equality of residual languages, not merely from state renaming. □

Definition 10.11 (Canonical finite-state relative-prime presentation). For an FSRP pair $( L , h )$ define

$$
\mathfrak { G } _ { L , h } ^ { \mathrm { f s } } : = \big ( \mathcal { P } , \mathrm { L e x } , \mathrm { S t a r t } , \mathcal { Q } _ { L , h } , \delta , \mathrm { F i n } \big ) ,
$$

where $\operatorname { L e x } ( P ) = P \cap \Sigma$ , Start records the exact prime decompositions of the accepting relative classes (and the empty start alternative when $\varepsilon \in L ) , \mathcal { Q } _ { L , h }$ is the shared set of nonempty residual languages defined above, $\delta ( R , A ) = A ^ { - 1 } R$ whenever this quotient is nonempty, and Fin records the residual states containing ε. This tuple is canonical because its controller states are residual languages themselves rather than arbitrarily named automaton states.

## 10.4 Compilation to an ordinary finite CFG

Assume unit separation, finitely many accepting classes, and FSRP. For each residual state $R \in \mathcal { Q } _ { L , h }$ introduce an auxiliary nonterminal $C _ { R } .$ . If $A ^ { - 1 } R = R ^ { \prime } \neq \emptyset$ , include $C _ { R }  A C _ { R ^ { \prime } }$ . If $\varepsilon \in A ^ { - 1 } R$ , also include the terminating controller step $C _ { R }  A$ . For each prime $P$ with nonempty ${ \mathrm { V a l } } _ { P }$ , include $P  C _ { \mathrm { V a l } _ { P } }$ . Add the lexical and start rules as in the direct construction.

Theorem 10.12 (Finite-state canonical grammar theorem). Under unit separation, finitely many accepting classes, finite prime spectrum, and $F S R P ,$ the residual-controller construction yields a finite CFG $G _ { L , h } ^ { \mathrm { r e s } }$ satisfying $L ( G _ { L , h } ^ { \mathrm { r e s } } , P ) = P$ for every live prime $P ,$ , and therefore $L ( G _ { L , h } ^ { \mathrm { r e s } } ) = L$

Proof. The controller realizes exactly the valid right-hand sides. Hence soundness and completeness reduce to the same arguments as for the direct grammar, using valid-rule reduction. Finiteness follows from $| \mathcal { P } | < \infty$ and $\kappa ( L , h ) < \infty$ □

Theorem 10.13 (Prime-skeleton preservation). Contract every maximal chain of residualcontroller nonterminals in a parse tree of $G _ { L , h } ^ { \mathrm { r e s } }$ . The resulting tree is a parse tree of the direct grammar containing all valid productions. Conversely, every direct valid-rule parse has a unique expansion through the deterministic residual controller. Hence contraction induces a bijection between residual-controller parse trees and direct prime-skeleton parse trees.

Proof. A controller chain beginning with $P  C _ { \mathrm { V a l } _ { P } }$ and emitting a prime word $\alpha = A _ { 1 } \cdot \cdot \cdot A _ { k }$ is forced step by step by the deterministic residual transition $R \mapsto A ^ { - 1 } R$ . It terminates exactly when the residual reached after reading α contains ε, equivalently when $\alpha \in { \mathrm { V a l } } _ { P }$ . Contracting the chain therefore yields precisely the direct valid production $P  \alpha$ . Conversely, for every $\alpha \in \mathrm { V a l } _ { P }$ , the same deterministic residual transitions give a unique controller chain from $C _ { \mathrm { V a l } _ { P } }$ to a terminating state. Applying this independently at every branching node gives mutually inverse expansion and contraction maps on parse trees. □

Thus FSRP is not merely weak language compression: it is a finite-state subdivision of the same intrinsic prime branching structure. This is the sense in which FSRP remains suitable for a strong/canonical structural target, paralleling Clark’s motivation for canonical grammars [3].

## 11 Tail saturation and presentation-defect complexity

The semantic residual theorem shows that only finitely many relative tail classes can occur after a fixed pair of primes. The remaining source of direct-rule infinitude is failure of a valid tail to saturate its own relative class.

Definition 11.1 (Saturated and under-saturated valid rules). Write a valid rule as $P $ Aα, $P , A \in { \mathcal { P } }$ ， $\alpha \in \mathcal { P } ^ { + }$ . It is tail-saturated if $\overline { { \alpha } } = [ \overline { { \alpha } } ] _ { \theta }$ , and tail-under-saturated otherwise. Let Sat $\dot { } P$ and Unsat<sub>P</sub> be the corresponding right-hand-side languages for left-hand prime $P .$

Thus $\mathrm { V T E _ { 1 } }$ is equivalent to Unsat ${ \mathrm { \Sigma } } ) _ { P } ~ = ~ { \emptyset }$ for every prime P. Every binary valid rule is automatically saturated, because its tail consists of one prime class.

For finite prime spectrum, put

$$
\begin{array} { r } { q : = | \mathcal { P } | , \qquad r _ { \mathrm { r e s } } ( L , h ) : = \operatorname* { m a x } \Bigl ( \{ | \operatorname { R e s } _ { \theta } ( A , P ) | : A , P \in \mathcal { P } \} \cup \{ 0 \} \Bigr ) , } \end{array}
$$

and let

$$
{ \mathcal { R } } _ { \mathrm { r e l } } : = \bigcup _ { A , P \in { \mathcal { P } } } { \mathrm { R e s } } _ { \theta } ( A , P ) .
$$

For $R \in \mathcal { R } _ { \mathrm { r e l } }$ define

$$
\begin{array} { r } { \mathrm { E x } ( R ) : = \{ \alpha \in \mathscr { P } ^ { + } : \overline { { \alpha } } = R \} , \qquad m _ { \mathrm { e x } } ( L , h ) : = \operatorname* { m a x } \bigl ( \{ | \mathrm { E x } ( R ) | : R \in \mathscr { R } _ { \mathrm { r e l } } \} \cup \{ 0 \} \bigr ) . } \end{array}
$$

The quantity $m _ { \mathrm { e x } } ( L , h )$ is finite even without UF by Corollary 4.3.

Theorem 11.2 (Saturated-basis bound). Assume L is h-substitutable and $| { \mathcal { P } } | = q < \infty$ . Then the set of saturated valid rules is finite and $| \mathcal { V } _ { \mathrm { s a t } } | \leq q ^ { 2 } r _ { \mathrm { r e s } } ( L , h ) m _ { \mathrm { e x } } ( L , h )$

Proof. For fixed $P , A ,$ , a saturated tail is an exact prime decomposition of some $R \in \operatorname { R e s } _ { \theta } ( A , P )$ There are at most $r _ { \mathrm { r e s } } ( L , h )$ such residual classes and at most $m _ { \mathrm { e x } } ( L , h )$ exact decompositions of each. Sum over the $q ^ { 2 }$ choices of $( P , A )$ □

Define the under-saturation radius and defect residual complexity by $\rho _ { \partial } ( L , h ) : = \operatorname* { s u p } \{ | \alpha | :$ $\alpha \in { \mathrm { U n s a t } } _ { P }$ for some $P \}$ and

$$
\kappa _ { \partial } ( L , h ) : = \left| \left\{ u ^ { - 1 } \operatorname { U n s a t } _ { P } : P \in \mathcal { P } , u \in \mathcal { P } ^ { * } , u ^ { - 1 } \operatorname { U n s a t } _ { P } \neq \emptyset \right\} \right| .
$$

Corollary 11.3 (Saturation–defect decomposition). Under the same hypotheses,

$$
\mathrm { V T E } _ { 1 } \longleftrightarrow \mathrm { U n s a t } _ { P } = \emptyset f o r e v e r y P ,
$$

$$
\mathrm { F R P } \iff \forall P , \ | \operatorname { U n s a t } _ { P } | < \infty \iff \rho _ { \partial } ( L , h ) < \infty ,
$$

$$
\mathrm { F S R P } \iff \forall P , \ \mathrm { U n s a t } _ { P } \ i s \ r e g u l a r \iff \kappa _ { \partial } ( L , h ) < \infty .
$$

Moreover $\rho ( L , h ) < \infty \iff \rho _ { \partial } ( L , h ) < \infty$

Proof. For every $P , \mathrm { V a l } _ { P } = \mathrm { S a t } _ { P } \ \dot { \cup } \ \mathrm { U n s a t } _ { P }$ and the saturated part is finite by Theorem 11.2. Over a finite prime alphabet, finiteness is equivalent to bounded word length, and finite unions and diferences preserve regularity. The first equivalence is the definition of $\mathrm { V T E _ { 1 } }$ □

Theorem 11.4 (Single-residual localization of FRP failure). $I f \left| \mathcal { P } \right| < \infty$ , L is h-substitutable, and FRP fails, then there are fixed primes $P , A$ and a fixed live relative class $R \subseteq A \backslash P$ together with infinitely many pairwise distinct valid rules $P  A \alpha _ { n }$ such that $[ { \overline { { \alpha _ { n } } } } ] _ { \theta } = R$ and ${ \overline { { \alpha _ { n } } } } \subsetneq R$ The lengths $\left| \alpha _ { n } \right|$ are unbounded along a subsequence.

Proof. Only finitely many triples $( P , A , R )$ are possible by finite prime spectrum and finite residual splitting. Since the saturated part is finite, infinitely many valid rules must be undersaturated. Pigeonhole localization gives one fixed triple; bounded length over a finite prime alphabet would give only finitely many words. □

This motivates the relative presentation complexity profile $\Pi ( L , h ) : = ( q , r _ { \mathrm { r e s } } , m _ { \mathrm { e x } } ; \rho _ { \partial } , \kappa _ { \partial } )$ The first three coordinates measure static relative geometry; the last two measure the dynamic structural defect. Under PTLD and finite prime spectrum, Theorem 7.1 and unique exact factorization give

$$
r _ { \mathrm { r e s } } \le 1 , \qquad m _ { \mathrm { e x } } \le 1 , \qquad \rho _ { \partial } = \kappa _ { \partial } = 0 .
$$

Thus the static residual and exact-factor multiplicities collapse to at most one, while the dynamic under-saturation defect disappears entirely.

If FSRP holds but FRP fails, some Unsat $\dot { } _ { , P }$ is an infinite regular language, so the ordinary pumping lemma gives an infinite family $u v ^ { n }$ w of under-saturated valid words. By the preceding theorem, an infinite subsequence can be localized to one residual class R. When the relative quotient is finite, Corollary 10.8 refines this to an explicit automaton-lasso certificate.

## 12 Unique factorization without finite direct presentation: a 36-element quotient witness

We now show directly that exact-factor ambiguity is not responsible for failure of FRP. We deliberately choose the trivial target language $L = \Sigma ^ { + }$ , so that the separation cannot be attributed to complicated language syntax: all nontrivial behavior is induced by the finite observer. Let $Q _ { 0 } = \{ 0 , 1 , 2 , 3 \}$ and write transformations as tuples of their values on $Q _ { 0 }$ . Put $\tau _ { a } = ( 1 , 2 , 0 , 0 )$ and ${ \tau _ { b } } = ( 1 , 3 , 1 , 0 )$ , and let $M _ { 0 } : = \langle \tau _ { a } , \tau _ { b } \rangle \leq T _ { 4 }$ . Finite enumeration gives $| M _ { 0 } | = 6 4$ . Define $h _ { 0 } : \{ a , b \} ^ { * } \to M _ { 0 }$ by $h _ { 0 } ( a ) = \tau _ { a }$ and $h _ { 0 } ( b ) = \tau _ { b }$ . Every nonempty product has rank at most three whereas the identity has rank four, so $h _ { 0 } ^ { - 1 } ( 1 ) = \{ \varepsilon \}$

Let ${ \sim } _ { 0 }$ be the least monoid congruence on $M _ { 0 }$ containing $h _ { 0 } ( a b ) \sim _ { 0 } h _ { 0 } ( a b a a a b )$ , and put $M _ { 3 6 } : = M _ { 0 } / { \sim } _ { 0 }$ and $h : = \pi _ { 0 } \circ h _ { 0 }$ . Again take $L : = \{ a , b \} ^ { + }$ . Finite congruence closure gives $\vert M _ { 3 6 } \vert = 3 6$ , and the identity class has no nonempty preimage, so $h ^ { - 1 } ( 1 ) = \{ \varepsilon \}$ . The following witness should therefore be read first as a finite monoid phenomenon and only secondarily as a language example. Choosing $L = \Sigma ^ { + }$ removes all nontrivial syntactic distinctions between nonempty words and isolates the obstruction entirely in the finite observer quotient. Indeed, since $L = \Sigma ^ { + }$ , all nonempty words are syntactically congruent and $\theta _ { L , h } = \ker h$ on nonempty words. Thus the 35 nonidentity quotient classes are exactly the live non-unit relative classes.

Computer-assisted verification. The monoid generators, quotient congruence, and infinitefamily identities are explicit. The remaining finite counts and quotient-minimality assertions are checked exhaustively by the reproducible verifier described in Appendix A.

Theorem 12.1 (Finite structure of the 36-element witness). For the pair $( L , h )$ above, unit separation and h-substitutability hold, the relative prime spectrum has fifteen elements, and every one of the thirty-five live non-unit relative classes has a unique exact relative prime factorization.

Proof. Unit separation follows from $h ^ { - 1 } ( 1 ) = \{ \varepsilon \}$ . Since $L = \Sigma ^ { + }$ , all nonempty words are syntactically congruent. If two equal-h-type words share an accepting context, then either both are nonempty, in which case they are syntactically congruent, or both are empty because $h ^ { - 1 } ( 1 ) = \{ \varepsilon \}$ . Thus L is h-substitutable.

The remaining finite assertions are verified exhaustively. Exact binary fiber-product equality $_ \mathrm { y }$ ields the fifteen relative primes

$$
\begin{array} { c } { \mathcal { P } = \{ a , b , a b , b a , a b a , b a a , b a b , a a a b , a b a a , a b a b , b a a a a , } \\ { b a b a , a a a b a , b a b a a , a a a b a a a \} , } \end{array}
$$

where each word denotes its relative class. Every quotient class has a shortest representative of length at most eight. By Corollary 4.3, cuts of these shortest representatives contain every exact prime factorization candidate. Exhaustive regular-language equality testing leaves exactly one candidate for each of the thirty-five live non-unit classes; Appendix A records the complete certificate. □

For the infinite valid family, put $P _ { a } = [ a ] _ { \theta } , \quad P _ { q } = [ b a a a ] _ { \theta } , \quad P _ { b } = [ b ] _ { \theta } , \quad P _ { a b } = [ a b ] _ { \theta }$ . All four are among the fifteen relative primes.

Proposition 12.2 (Infinite valid family). For every $m \geq 0$

$$
\boxed { P _ { a b }  P _ { a } P _ { q } ^ { m } P _ { b } }
$$

is valid.

Finite quotient certificate. Let $\alpha = h ( a ) , \qquad q = h ( b a a a ) , \qquad \beta = h ( b ) , \qquad t = h ( a b )$ . The finite multiplication table satisfies

$$
q ^ { 2 } = q ^ { 3 } , \qquad \alpha q = \alpha q ^ { 2 } , \qquad q \beta = q ^ { 2 } \beta ,
$$

and

$$
\alpha \beta = t , \qquad \alpha q \beta = t , \qquad \alpha q ^ { 2 } \beta = t .
$$

Hence $\alpha q ^ { m } \beta = t$ for every $m \geq 0$ , proving correctness of the displayed family. The same three stabilization identities make the proper intervals uniform in m: every proper block of length at least two has quotient value of one of the forms αq, $q \beta ,$ , or $q ^ { 2 }$ . For $m = 1$ only the first two types occur, while all three occur for $m \geq 2$ . Thus, over all $m$ , the proper interval values lie in just three relative classes, represented by

$$
[ a b a a a ] _ { \theta } , \qquad [ b a a a b ] _ { \theta } , \qquad [ b a a a b a a a ] _ { \theta } .
$$

Each is exact composite; finite set-product verification gives, for example,

$$
[ a b a a a ] _ { \theta } = [ a b a a ] _ { \theta } \cdot [ a ] _ { \theta } ,
$$

$$
[ b a a a b ] _ { \theta } = [ b ] _ { \theta } \cdot [ a a a b ] _ { \theta } ,
$$

and

$$
[ b a a a b a a a a ] _ { \theta } = [ b a a a b a a ] _ { \theta } \cdot [ a ] _ { \theta } .
$$

Thus no proper contiguous block has prime quotient product, and every rule in the displayed family is non-pleonastic. □

Corollary 12.3 (UF–FRP separation). Unique exact relative prime factorization does not imply $F R P ,$ even under finite relative quotient and finite relative prime spectrum.

Proof. Combine Theorem 12.1 with the preceding infinite family of valid productions. □

The same family gives a particularly transparent separation between UF and tail saturation. For every $m \geq 1$ ，

$$
\underbrace { P _ { q } * \cdot \cdot \cdot * P _ { q } } _ { m { \mathrm { ~ c o p i e s } } } * P _ { b } = H : = [ b a a a b ] _ { \theta } ,
$$

where the notation on the left denotes quotient multiplication of the prime classes. By global UF the class H has the unique exact decomposition

$$
\boxed { H = P _ { b } P _ { r } , \qquad P _ { r } = [ a a a b ] _ { \theta } . }
$$

But the valid tails themselves under-saturate $H \colon$

$$
\begin{array} { r } { \boxed { \overline { { P _ { q } ^ { m } \bar { P _ { b } } } } \subsetneq H \qquad ( m \geq 1 ) . } } \end{array}
$$

Consequently this one finite quotient proves simultaneously $\mathrm { U F } \not \Rightarrow \mathrm { V T E } _ { 1 } , \qquad \mathrm { U F } \not \Rightarrow \mathrm { F R P }$ . Since the relative quotient is finite, FSRP holds automatically; thus the witness has $m _ { \mathrm { e x } } = 1$ $\rho _ { \partial } =$ $\infty , \quad \kappa _ { \partial } < \infty$ . It therefore separates exact-factor ambiguity from under-saturation defect in the strongest possible way: exact ambiguity is zero, while the presentation defect has unbounded length.

Proposition 12.4 (Minimality among quotients of the witness). The monoid $M _ { 3 6 }$ has twenty monoid congruences. Every proper congruence quotient, with the induced observation on $L ,$ has FRP. The nineteen proper quotient sizes are 11, 9, 8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1. Thus no proper quotient of this particular 36-element observer continues to witness $U F { \mathrel { + } } { \neg } F R P$ . This is a quotient-minimality statement for $M _ { 3 6 }$ , not a claim that 36 is the absolute minimum among all finite monoids.

Proof. This is the exhaustive quotient audit in Appendix A, item $8 .$

## 13 A coupled nonregular FSRP–FRP separation

The finite witness of Section 12 proves that FSRP can compress an infinite valid-rule family, but FSRP there is automatic from finiteness of the relative quotient. We now show that the same defect persists when the full relative quotient is infinite and the target language is intrinsically nonregular.

Let

$$
\Gamma = \{ a , b , c , d \} , \qquad L ^ { \mathrm { w r a p } } : = \{ c ^ { n } w d ^ { n } : n \geq 0 , \ w \in \{ a , b \} ^ { + } \} .
$$

Let $h _ { 3 6 } : \{ a , b \} ^ { * } \to M _ { 3 6 }$ be the morphism from Section 12. Extend it to $\widehat { h } _ { 3 6 } : \Gamma ^ { * } \to M _ { 3 6 }$ by $\widehat { h } _ { 3 6 } ( c ) = \widehat { h } _ { 3 6 } ( d ) = 1$ . Let $E = \{ 0 , 1 \} ^ { 2 }$ with coordinatewise maximum as multiplication, and put

$$
\chi ( w ) = \big ( [ c \mathrm { o c c u r s } \mathrm { i n } w ] , [ d \mathrm { o c c u r s } \mathrm { i n } w ] \big ) \in E .
$$

Then

$$
h ^ { \mathrm { w r a p } } ( w ) : = ( \widehat { h } _ { 3 6 } ( w ) , \chi ( w ) ) : \Gamma ^ { * } \to M _ { 3 6 } \times E
$$

is a finite-monoid morphism.

For $s \in M _ { 3 6 } \setminus \{ 1 \}$ define the nonempty middle fiber

$$
\mathsf { M } _ { s } : = \{ u \in \{ a , b \} ^ { + } : h _ { 3 6 } ( u ) = s \} .
$$

For $i , j \geq 1$ and $r \in \mathbb { Z }$ put

$$
\begin{array} { r l r } & { C _ { i } : = \{ c ^ { i } \} , } & { D _ { j } : = \{ d ^ { j } \} , } \\ & { L _ { i , s } : = c ^ { i } \mathsf { M } _ { s } , } & { R _ { j , s } : = \mathsf { M } _ { s } d ^ { j } , } \\ & { W _ { r , s } : = \{ c ^ { i } u d ^ { j } : i , j \geq 1 , \ i - j = r , \ u \in \mathsf { M } _ { s } \} . } \end{array}
$$

Write $C : = C _ { 1 } , D : = D _ { 1 }$ , and $W _ { s } : = W _ { 0 , s }$

Lemma 13.1 (Wrapped live-class normal forms). The live relative classes of $\left( L ^ { \mathrm { w r a p } } , h ^ { \mathrm { w r a p } } \right)$ are exactly

$$
[ \varepsilon ] , C _ { i } , D _ { j } , \mathsf { M } _ { s } , L _ { i , s } , R _ { j , s } , W _ { r , s } ,
$$

with the parameters above. In particular, the relative quotient is infinite. Moreover, the pair is unit-separated and $h ^ { \mathrm { w r a p } }$ -substitutable.

Proof. Every factor of a word $c ^ { n } w d ^ { n }$ has one of the forms

$$
\varepsilon , \quad c ^ { i } , \quad d ^ { j } , \quad u , \quad c ^ { i } u , \quad u d ^ { j } , \quad c ^ { i } u d ^ { j } ,
$$

where $u \in \{ a , b \} ^ { + }$ and the wrapper exponents that occur are positive. The two occurrence bits in χ separate pure middle, left-wrapper, right-wrapper, and two-sided-wrapper factors; the first coordinate separates pure wrapper factors from factors containing a nonempty middle word because $h _ { 3 6 } ^ { - 1 } ( 1 ) = \{ \varepsilon \}$

All nonempty middle words have the same ordinary syntactic distribution in $L ^ { \mathrm { w r a p } }$ : only their position inside the arbitrary nonempty middle word matters, not their $a / b$ content. Hence $h _ { 3 6 }$ splits them precisely into the classes $\mathsf { M } _ { s }$ . For a left-wrapper factor $c ^ { i } u$ , any accepting completion has the form $c ^ { p } ( c ^ { i } u ) v d ^ { q }$ with $v \in \{ a , b \} ^ { * }$ and necessarily $p + i = q ;$ thus its distribution is determined by i, and $h _ { 3 6 } ( u ) = s$ gives $L _ { i , s }$ . The right-wrapper case is dual. If a factor already contains both wrappers, an accepting completion has the form $c ^ { p } ( c ^ { i } u d ^ { j } ) d ^ { q }$ , and membership is equivalent to $p + i = q + j$ . Hence its distribution is determined by $i - j$ , giving $W _ { r , s }$

Conversely, the same equations distinguish diferent values of the displayed parameters. For example, $c ^ { i } a d ^ { i } \in L ^ { \mathrm { w r a p } }$ while $c ^ { j } a d ^ { i } \notin L ^ { \mathrm { w r a p } }$ for $j \neq i$ , so the classes $C _ { i }$ are pairwise distinct; the other parameters are separated similarly. This also shows that the relative quotient is infinite.

Unit separation follows because a nonempty word either contains c or $d ,$ in which case $\chi$ is nonzero, or lies in $\{ a , b \} ^ { + }$ , in which case its $h _ { 3 6 } \mathrm { - v a l u e }$ is nonidentity. Finally, if two ${ \mathrm { e q u a l } } { \mathrm { - } } h ^ { { \mathrm { w r a p } } } .$ type factors share an accepting context, the occurrence bits put them in the same shape above. The common completion equation forces equality of the exponent for pure $c \mathrm { - \ o r }$ pure d-factors, equality of i in the left-wrapper case, equality of $j$ in the right-wrapper case, and equality of $i - j$ in the two-sided case; pure middle factors already have equal syntactic distribution. Thus they lie in the same displayed relative class, proving $h ^ { \mathrm { w r a p } }$ -substitutability. □

Lemma 13.2 (Middle-subsystem preservation). For $u , v \in \{ a , b \} ^ { + }$

$$
u \theta _ { L ^ { \mathrm { w r a p } } , h ^ { \mathrm { w r a p } } } \ v \quad \Longleftrightarrow \quad h _ { 3 6 } ( u ) = h _ { 3 6 } ( v ) .
$$

Consequently the live middle classes, their exact set products, the middle relative primes, and the correct and valid prime sequences contained wholly in the middle alphabet coincide with those of the 36-element witness.

Proof. For middle words the wrapper-occurrence coordinates are both zero, so equality of $h ^ { \mathrm { w r a p } } .$ type is exactly equality of $h _ { \mathrm { 3 6 } - \mathrm { t y p e } }$ . Every two nonempty middle words have the same syntactic distribution in $L ^ { \mathrm { w r a p } }$ : a context can accept such a factor only by supplying equal numbers of outer $c \mathrm { { s } }$ and $d \mathrm { { s } }$ , independently of the internal $a / b$ word. Thus the relative congruence restricted to $\{ a , b \} ^ { + }$ is precisely ker $h _ { 3 6 }$ . Exact products and quotient products therefore agree with the finite witness, and primali $\operatorname { t y } ,$ correctness, and pleonasticity are preserved under this identification. □

Let

$$
\mathsf { P } _ { 3 6 } : = \{ s \in M _ { 3 6 } \setminus \{ 1 \} : \mathsf { M } _ { s } \mathrm { ~ i s ~ a ~ r e l a t i v e ~ p r i m e } \} .
$$

By Theorem 12.1, $| \mathsf { P } _ { 3 6 } | = 1 5$

Theorem 13.3 (Prime spectrum and exact UF of the wrapped pair). The relative prime spectrum $o f \left( L ^ { \mathrm { w r a p } } , h ^ { \mathrm { w r a p } } \right)$ is exactly

$$
\begin{array} { r } { \overline { { \mathcal { P } } } _ { \mathrm { w r a p } } = \{ C , D \} \cup \{ \mathsf { M } _ { p } : p \in \mathsf { P } _ { 3 6 } \} \cup \{ W _ { s } : s \in M _ { 3 6 } \setminus \{ 1 \} \} . } \end{array}
$$

Hence $| \mathcal { P } _ { \mathrm { w r a p } } | = 2 + 1 5 + 3 5 = 5 2$ . Every live non-unit relative class has a unique exact prime factorization.

Proof. The normal forms satisfy the exact identities

$$
C _ { i } = C ^ { i } , \qquad D _ { j } = D ^ { j } , \qquad L _ { i , s } = C ^ { i } { \sf M } _ { s } , \qquad R _ { j , s } = { \sf M } _ { s } D ^ { j } ,
$$

and

$$
W _ { r , s } = \left\{ \begin{array} { l l } { C ^ { r } W _ { s } , } & { r > 0 , } \\ { W _ { s } , } & { r = 0 , } \\ { W _ { s } D ^ { - r } , } & { r < 0 . } \end{array} \right.
$$

Thus only $C , D ,$ the middle fibers, and the balanced wrapped classes $W _ { s }$ can be prime. By Lemma 13.2, the prime middle fibers are exactly the fifteen $\mathsf { M } _ { p }$ with $p \in \mathsf { P } _ { 3 6 }$

It remains to prove that every $W _ { s }$ is prime. Let $u _ { s }$ be a shortest word of $\mathsf { M } _ { s }$ . Then $\mathit { c u } _ { s } d$ is a shortest word of $W _ { s }$ . By Lemma 4.2, any nontrivial exact binary factorization would occur at a cut of this word. The cut immediately after c gives

$$
C \cdot R _ { 1 , s } = c \mathsf { M } _ { s } d \subsetneq W _ { s } ,
$$

and the cut immediately before d gives the dual strict inclusion. A cut $u _ { s } = x y$ inside the middle gives

$$
L _ { 1 , h _ { 3 6 } ( x ) } \cdot R _ { 1 , h _ { 3 6 } ( y ) } \subseteq c \{ a , b \} ^ { + } d \subsetneq W _ { s } .
$$

All three types of products contain only words with one left and one right wrapper symbol, whereas $c ^ { 2 } u _ { s } d ^ { 2 } \in W _ { s }$ . Hence no shortest-word cut is exact, and $W _ { s }$ is prime.

For uniqueness, the middle classes inherit the unique exact prime factorizations of the 36- element witness. In a factorization of $L _ { i , s } ,$ no prime containing d can occur, and a middle prime cannot precede a $C$ without producing an $a / b$ symbol before a later $c ;$ therefore the factorization is $C ^ { i }$ followed by the unique middle factorization of $\mathsf { M } _ { s }$ . The right-wrapper case is dual.

Finally consider $W _ { r , s }$ . Its wrapper depth is unbounded inside one relative class. A product using only $C , D ,$ , and middle primes has fixed wrapper counts, so every exact factorization must contain a wrapped prime. Two wrapped primes would place a d before a later c, producing only dead words, so there is exactly one. A middle prime cannot occur before that wrapped prime, since it would place an $a / b$ symbol before a later $c ;$ nor can one occur after it, since it would place an $a / b$ symbol after an earlier d. Hence every exact factorization has the form $C ^ { p } W _ { t } D ^ { q }$ Exact equality with $W _ { r , s }$ forces $t = s$ and, by comparing the least possible left and right wrapper depths,

$$
( p , q ) = \left\{ { \begin{array} { l l } { ( r , 0 ) , } & { r > 0 , } \\ { ( 0 , 0 ) , } & { r = 0 , } \\ { ( 0 , - r ) , } & { r < 0 . } \end{array} } \right.
$$

This proves global exact UF.

We now describe the valid-rule languages directly. Let

$$
\mathcal { A } : = \{ \mathsf { M } _ { p } : p \in \mathsf { P } _ { 3 6 } \}
$$

be the middle prime alphabet, and let $\mu : { \mathcal { A } } ^ { * } \to M _ { 3 6 }$ be the evaluation morphism. Define

$$
\mathrm { C o r r } ^ { \mathrm { m i d } } : = \{ \beta \in \mathcal { A } ^ { \geq 2 } : \mu ( \beta ) \in \mathsf { P } _ { 3 6 } \}
$$

and, for $s \neq 1$

$$
\mid _ { s } : = \mu ^ { - 1 } ( s ) \cap \mathcal { A } ^ { + } \setminus \mathcal { A } ^ { * } \mathrm { C o r r } ^ { \mathrm { m i d } } \mathcal { A } ^ { * } .
$$

Both languages are regular, since $\mu$ has finite codomain.

Theorem 13.4 (Intrinsic infinite-quotient FSRP–FRP separation). The pair $\left( L ^ { \mathrm { w r a p } } , h ^ { \mathrm { w r a p } } \right)$ is a nonregular context-free, unit-separated, $h ^ { \mathrm { w r a p } }$ -substitutable pair with infinite relative quotient and finite prime spectrum. It has global exact $U F$ and satisfies FSRP but not FRP. More precisely,

$$
\mathrm { V a l } _ { C } = \mathrm { V a l } _ { D } = \emptyset ,
$$

the valid-return language of each middle prime is exactly the corresponding valid-return language of the 36-element witness, and for every $s \neq 1$ 2

$$
\boxed { \mathrm { V a l } _ { W _ { s } } = \{ C W _ { s } D \} \cup C \mathsf { l } _ { s } D . }
$$

Proof. The grammar

$$
S \to c S d \mid A , \qquad A \to a A \mid b A \mid a \mid b
$$

shows that $L ^ { \mathrm { w r a p } }$ is context-free. It is nonregular because

$$
L ^ { \mathrm { w r a p } } \cap c ^ { * } a d ^ { * } = \{ c ^ { n } a d ^ { n } : n \geq 0 \} .
$$

The remaining structural assertions up to finite prime spectrum and UF are Lemma 13.1 and Theorem 13.3.

There is no correct branching rule with target $C$ or $D ,$ , so their valid languages are empty. A correct sequence returning to a middle prime cannot contain $C , D ,$ or any $W _ { s }$ , because the $c / d$ occurrence bits would then be nonzero. Lemma 13.2 therefore identifies its valid-return language with the corresponding language of the 36-element witness, which is regular by Theorem 10.2.

Fix $s \neq 1$ . A prime sequence whose quotient product is $W _ { s }$ has exactly one of two forms. If it contains a wrapped prime, it can contain exactly one: two would create a dc boundary. Shape then forces

$$
C ^ { k } W _ { s } D ^ { k } \qquad ( k \geq 1 ) .
$$

If it contains no wrapped prime, shape forces

$$
C ^ { k } \alpha D ^ { k } , \qquad k \geq 1 , \quad \alpha \in { \cal { A } } ^ { + } , \quad \mu ( \alpha ) = s .
$$

For $k \geq 2$ , either form is pleonastic because deleting one outer $C$ and one outer D leaves a proper contiguous block whose quotient class is the prime $W _ { s } .$ . Thus validity forces $k = 1$

The word $C W _ { s } D$ is valid: its two proper blocks of length two have quotient classes $W _ { 1 , s }$ and $W _ { - 1 , s }$ , both composite by Theorem 13.3. For a sequence $C \alpha D$ , any proper block meeting C but not D has a left-wrapper quotient class and is composite, and dually for a block meeting D but not C. Hence the only possible prime-valued proper blocks lie entirely inside α. Such a block is prime exactly when it belongs to $\mathrm { C o r r ^ { m i d } }$ . Therefore $C \alpha D$ is valid exactly when $\alpha \in \mathsf { I } _ { s } .$ , proving the displayed formula for $\mathrm { V a l } _ { W _ { s } }$ . All valid languages are thus regular, so FSRP holds.

Finally put

$$
\mathsf { M } _ { a } = [ a ] , \qquad \mathsf { M } _ { q } = [ b a a a ] , \qquad \mathsf { M } _ { b } = [ b ] , \qquad \mathsf { M } _ { a b } = [ a b ]
$$

for the corresponding middle prime classes. Quotient multiplication and prime-valued proper intervals inside the middle subsystem are unchanged from Section 12. Hence

$$
\mathsf { M } _ { a b } \to \mathsf { M } _ { a } \mathsf { M } _ { q } ^ { m } \mathsf { M } _ { b } \qquad ( m \geq 0 )
$$

is valid for every $m .$ . Thus FRP fails. The same middle tails retain the strict under-saturation from the finite witness, so $\mathrm { V T E _ { 1 } }$ fails as well. □

The theorem separates FRP from FSRP without appealing to finiteness of the relative quotient. The nonregular wrapper and the defective middle return system are not disjoint components: the new wrapped primes $W _ { s }$ have valid languages obtained by inserting the finite-state middle irreducibles $\mid _ { s }$ between the wrapper primes $C$ and $D .$

Example 13.5 (A finite FRP language without UF). Let $L _ { f } = \{ a b c d , a p c d , b x \}$ . Take the threeelement monoid $M = \{ 1 , s , 0 \}$ with identity 1, zero 0, and $s ^ { 2 } = s _ { ; }$ , and define $h ( a ) = h ( c ) = 0$ $h ( b ) = s .$ , and $h ( p ) = h ( d ) = h ( x ) = 1$ . Among distinct live factors of the same h-type, the only pairs sharing a context are ab/ap, $b c / p c _ { ; }$ , abc/apc, bcd/pcd, and abcd/apcd; each pair has the same syntactic distribution. Hence $L _ { f }$ is h-substitutable. The relative classes $X = \{ a b c , a p c \}$ $Y \ = \ \{ a b , a p \} , \ Z \ = \ \{ c \} , \ A \ = \ \{ a \}$ , and $B ~ = ~ \{ b c , p c \}$ satisfy the two distinct exact prime factorizations $X = Y \cdot Z = A \cdot B$ . By Corollary $4 . 3 ,$ the shortest representatives of A, Z, Y, B expose all possible nontrivial exact cuts; these cuts show that all four classes are prime. Thus UF fails. Since $L _ { f }$ is finite, let N be the maximum length of a live factor. For any correct prime right-hand side of length $k ,$ concatenating shortest representatives yields a live factor of length at least $k ,$ so $k \leq N$ . Over the finite prime alphabet there are therefore only finitely many valid rules, and FRP holds.

Corollary 12.3 and Example 13.5 show that UF and FRP are incomparable.

## 14 Disjoint-alphabet sums as an ambient comparison

Let $( L _ { 1 } , h _ { 1 } )$ and $( L _ { 2 } , h _ { 2 } )$ be fixed-observation pairs over disjoint alphabets $\Sigma _ { 1 } , \Sigma _ { 2 }$ . Adjoin absorbing zeros $\perp _ { i }$ to the monoids $M _ { i } .$ , extend each morphism by sending letters of the other alphabet to $\perp _ { i } ,$ and set $h = ( \bar { h } _ { 1 } , \bar { h } _ { 2 } ) : ( \Sigma _ { 1 } \cup \Sigma _ { 2 } ) ^ { * }  M _ { 1 } ^ { \perp } \times M _ { 2 } ^ { \perp }$ . Put $L : = L _ { 1 } \cup L _ { 2 }$

Theorem 14.1 (Disjoint-alphabet sum). If each $( L _ { i } , h _ { i } )$ is $h _ { i }$ -substitutable and unit-separated, then (L, h) is h-substitutable and unit-separated. Its live relative classes and relative primes are the disjoint unions of the corresponding component classes and primes. Moreover, for a component prime $P _ { i }$ ${ \mathrm { V a l } } _ { P } ^ { L } = { \mathrm { V a l } } _ { P } ^ { L _ { i } }$ . Consequently $\mathrm { F R P } ( L , h ) \iff \mathrm { F R P } ( L _ { 1 } , h _ { 1 } ) \land \mathrm { F R P } ( L _ { 2 } , h _ { 2 } )$ 2 and similarly for FSRP. Exact UF is also componentwise: the sum has UF if each component has UF.

Proof. Every nonempty live factor lies in exactly one component alphabet; mixed-alphabet words are dead. The absorbing coordinates of h separate the two components, and the relative congruence restricted to $\Sigma _ { i } ^ { * }$ is exactly that of $( L _ { i } , h _ { i } )$ . Thus live classes, h-substitutability, and unit separation are componentwise.

Any exact factorization of a live component class uses classes from that same component, so primality, exact factorization, and UF are componentwise. The same separation applies to quotient returns: a prime sequence mixing components has a mixed representative and cannot return to a live prime. Hence ${ \mathrm { V a l } } _ { P } ^ { L } = { \mathrm { V a l } } _ { P } ^ { \breve { L } _ { i } }$ for each component prime $P ,$ proving the FRP and FSRP equivalences. □

As a comparison, take $L _ { 0 }$ from Section 9 and an alphabet-renamed copy of the 36-element witness. Their disjoint sum $L ^ { \ddagger } : = L _ { 0 } \cup \{ c , d \} ^ { + }$ is a nonregular context-free $\mathrm { U F + F S R P + \neg F R P }$ example by the theorem above. This construction is only ambient: the nonregularity and the under-saturation defect remain in separate components. The wrapped language of Section 13 is stronger, since its infinite quotient and finite-state defective return dynamics occur in one coupled relative-prime system.

## 15 Strong structural reconstruction from positive data

The strong PTLD theorem below is stated for $L \subseteq \Sigma ^ { + }$ so that the learner need not carry a separate empty-word start case. The relative-prime and presentation theory developed in the preceding sections is formulated over $\Sigma ^ { * }$ and does not rely on this notational restriction.

The structural results above suggest two distinct learning targets. Under PTLD the canonical object is the finite direct relative-prime grammar. Under the weaker FSRP condition, the canonical object is instead the finite-state residual controller for the valid-rule languages. This section gives a direct positive-data strong learner for the PTLD branch and a limit-canonicalization theorem for the FSRP branch.

We use the strong Gold viewpoint of Clark [3]: a learner succeeds strongly when its hypotheses eventually stabilize to a canonical representation of the target, not merely to an arbitrary grammar generating the same language. In the PTLD branch, each update is polynomial in the current sample size, and the target admits an explicit finite characteristic sample. Quantitative word-length bounds below are stated using a finite cut-separation parameter in addition to canonical size and thickness.

Definition 15.1 (Positive text). A positive text for L is an infinite sequence $T = ( w _ { 0 } , w _ { 1 } , \dots )$ of elements of L in which every word of L occurs at least once. Write $T [ n ]$ for its length-n prefix and content $\left( T [ n ] \right)$ for the finite set of words observed so far.

Definition 15.2 (Weak behavioral correctness). A CFG-valued learner W is weakly behaviorally correct on L if for every positive text T for L there is N such that $L ( \mathcal { W } ( T [ n ] ) ) ~ =$ $\begin{array} { r l r } { L } & { { } } & { ( n \geq N ) } \end{array}$ . The hypotheses themselves need not stabilize.

Definition 15.3 (Strong Gold identification). Let $\mathrm { C a n } ( L , h )$ be a fixed canonical representation of the pair (L, h). A learner A strongly identifies $\mathrm { C a n } ( L , h )$ in the Gold sense if for every positive text T for L there is N such that $\begin{array} { r } { A ( T [ n ] ) = \mathrm { C a n } ( L , h ) \qquad ( n \geq N ) } \end{array}$ , where equality may be replaced by isomorphism when canonical names have not been fixed.

Definition 15.4 (Characteristic sample). For a learner A and a canonical target $R = \operatorname { C a n } ( L , h )$ a finite set $\mathcal { C } S \subseteq L$ is a characteristic sample if every finite sample D with ${ \mathcal { C } } S \subseteq D \subseteq L$ forces $\mathcal { A } ( D ) = R$ . For finite D, write $\begin{array} { r } { \| D \| : = \sum _ { w \in D } | w | } \end{array}$ . Polynomial update time is measured as a polynomial in $\| D \|$

## 15.1 The canonical PTLD target

Assume throughout this subsection that L is h-substitutable, unit-separated, has finitely many relative primes, and satisfies PTLD. Let $\mathcal { P } = \{ P _ { 1 } , \ldots , P _ { q } \}$ }. By Theorems 7.21 and 7.22, exact factorization is unique, VTE and VTD hold, and $| \nu | \leq q ^ { 2 }$

Let $G _ { L , h } ^ { \star }$ be the direct relative-prime grammar whose prime nonterminals are the live relative primes, whose branching rules are exactly the valid productions, whose lexical rules are $P \to a$ whenever $a \in P \cap \Sigma$ , and whose start rules use the unique exact prime factorization of each accepting relative class. We canonically name every prime by its shortlex least word $\omega ( P ) : =$ $\mathrm { m i n _ { s h o r t l e x } } P .$

For any sentential form $\alpha \in ( \mathcal { P } \cup \Sigma ) ^ { * }$ of $G _ { L , h } ^ { \star } .$ , let $\ell ( \boldsymbol { \alpha } ) : = \operatorname* { m i n } \{ | w | : \boldsymbol { \alpha } \Rightarrow ^ { * } w \}$ . Define the structural thickness $\tau ( G ^ { \star } ) : = \operatorname* { m a x } _ { A  \alpha \in G ^ { \star } } \ell ( \alpha )$ . Write simply τ when the target is clear. Since every prime is non-unit, $| \omega ( P ) | \le \tau$ for each prime appearing on a right-hand side or start factorization.

Proposition 15.5 (Trimness of the canonical PTLD grammar). Every live relative prime is reachable and productive in $G _ { L , h } ^ { \star }$ ; hence $G _ { L , h } ^ { \star }$ is trim after deletion of vacuous start structure.

Proof. Productivity is Theorem 6.3. For reachability, choose $x \in P$ and a context ℓxr $\in ~ L$ Factor the live classes [ℓ] and [r] exactly into primes and concatenate those factors with P. This prime sequence has quotient equal to the accepting class [ℓxr] and therefore has setwise product contained in that class. By Theorem 7.24, the exact start factorization of [ℓxr] derives this sequence. Expanding the left and right factors to ℓ and r yields a derivation $S \Rightarrow ^ { * } \ell P r$ □

Lemma 15.6 (Bounded canonical context). For every prime P there is a terminal context $( \ell _ { P } , r _ { P } )$ such that $S \Rightarrow ^ { * } \ell _ { P } P r _ { P }$ and $| \ell _ { P } | + | r _ { P } | \leq q \tau$

Proof. Choose a terminal context of minimum total length and inspect a root-to-P path in a corresponding partial derivation. If a prime nonterminal occurred twice on that path, stop at the earlier occurrence and expand all of-path siblings to shortest terminal yields. Every branching rule has at least two non-unit prime symbols, and every non-unit prime has shortest yield at least one, so removing the repeated spine segment removes a positive terminal contribution and produces a strictly shorter context around the same prime. Hence the prime spine is simple and contains at most q prime nodes. Including the start step, there are at most $q$ of-spine contributions, each bounded by τ by the definition of thickness. □

## 15.2 An h-guarded substring learner

For a finite positive sample $D \subseteq L$ , let $\operatorname { S u b } ( D )$ be the set of nonempty substrings of words in $D$ Define a finite CFG $W _ { h } ( D )$ with a fresh start symbol $S _ { D }$ , nonterminals $\langle u \rangle$ for $u \in { \mathrm { S u b } } ( D )$ , start rules $S _ { D }  \langle w \rangle$ for every $w \in D$ , and rules $\langle u v \rangle \to \langle u \rangle \langle v \rangle$ , whenever u, v, uv ∈ Sub(D), $\langle a \rangle \to a$ for observed letters, and symmetric substitution rules $\langle u \rangle  \langle v \rangle$ whenever $h ( u ) = h ( v )$ and there is a sample context $( \ell , r )$ with ℓur, $\ell v r \in D$ . Here ↔ abbreviates the two unit productions in opposite directions. This is the ordinary substring/substitution construction guarded by equality of the fixed observer type.

Theorem 15.7 (Guarded weak soundness). For every positive sample $D \subseteq L$ and every $u \in$ Sub(D), $L ( W _ { h } ( D ) , \langle u \rangle ) \subseteq [ u ] _ { L , h }$ . Consequently $L ( W _ { h } ( D ) ) \subseteq L$

Proof. Map ⟨u⟩ to the true relative class [u]. Binary rules are sound because $\theta _ { L , h }$ is a congruence. Every guarded substitution preserves the true relative class by Lemma 3.3. Induction on derivations proves the first inclusion. Every start class contains a sample word in $L$ and is therefore an accepting relative class, which proves the second. □

The preceding theorem gives a useful one-sided invariant: the learner may undergenerate before its characteristic data arrive, but it never creates a string outside the target language.

## 15.3 Finite signatures and exact observed relative classes

For each $u \in { \mathrm { S u b } } ( D )$ fix one observed occurrence $c _ { D } ( u ) = ( \ell _ { u } , r _ { u } ) , \qquad \ell _ { u } u r _ { u } \in D$ . For any CFG W define the finite signature

$$
\sigma _ { D , W } ( u ) : = \left( h ( u ) , \left( \mathbf { 1 } [ \ell _ { x } u r _ { x } \in L ( W ) ] \right) _ { x \in \mathrm { S u b } ( D ) } \right) .
$$

Lemma 15.8 (No false relative merges). For $W = W _ { h } ( D ) , \sigma _ { D , W } ( u ) = \sigma _ { D , W } ( v ) \Longrightarrow u \theta _ { L , h } v$ . If in addition $L ( W ) = L$ , then for all observed substrings $\sigma _ { D , W } ( u ) = \sigma _ { D , W } ( v ) \iff u \theta _ { L , h } v .$

Proof. Equality of signatures gives $h ( u ) = h ( v )$ . The coordinate indexed by u satisfies $\ell _ { u } u r _ { u } \in$ $D \subseteq L ( W )$ . Hence signature equality implies $\ell _ { u } v r _ { u } \in L ( W ) \subseteq L$ by Theorem 15.7. Thus u and v have equal observer type and a common L-context, so Lemma 3.3 gives uθv. If $L ( W ) = L$ the converse follows directly from equality of full syntactic distributions and observer types.

Thus, after the weak core becomes complete, the true restriction of $\theta _ { L , h }$ to all observed substrings is computable by finitely many ordinary CFG membership tests.

## 15.4 Finite certificates for relative primality

For a prime P and a nontrivial cut

$$
\omega ( P ) = u v , \qquad u , v \neq \varepsilon ,
$$

put

$$
Y _ { P , u } : = [ u ] _ { \theta } , \qquad Z _ { P , v } : = [ v ] _ { \theta } .
$$

Since $\omega ( P ) \in P$ , congruence gives

$$
Y _ { P , u } \cdot Z _ { P , v } \subseteq P .
$$

Both factors are live and non-unit. Hence primality of P implies

$$
Y _ { P , u } \cdot Z _ { P , v } \subsetneq P .
$$

Define the prime cut-separation radius

$$
\begin{array} { r } { \eta _ { \mathrm { c u t } } ( L , h ) : = \operatorname* { m a x } \left\{ \operatorname* { m i n } \{ | x | : x \in P \setminus ( Y _ { P , u } \cdot Z _ { P , v } ) \} : \begin{array} { l } { P \in \mathcal { P } , } \\ { \omega ( P ) = u v , ~ u , v \neq \varepsilon } \end{array} \right\} , } \end{array}
$$

with $\eta _ { \mathrm { c u t } } ( L , h ) = 0$ if no canonical prime word has a nontrivial cut.

Under PTLD the set diference in this definition has a more transparent prefix interpretation.

Lemma 15.9 (Cut separation is prime-prefix escape). Assume that L is h-substitutable, unitseparated, and satisfies PTLD. Let P be a relative prime and let $\omega ( P ) = u v$ be a nontrivial cut. Put $Y = [ u ] _ { \theta }$ and $Z = [ v ] _ { \theta }$ . Then

$$
P \cap Y \Sigma ^ { * } = Y \cdot Z .
$$

Consequently

$$
\operatorname* { m i n } \{ | x | : x \in P \setminus ( Y \cdot Z ) \} = \operatorname* { m i n } \{ | x | : x \in P , \ x \notin Y \Sigma ^ { * } \} .
$$

Thus, on the PTLD branch, $\eta _ { \mathrm { c u t } }$ is exactly a quantitative prime-prefix escape radius for the finitely many cuts of the canonical prime words.

Proof. The inclusion $Y { \cdot } Z \subseteq P$ follows from congruence and $u v \in P$ . Conversely, let x $\in P \cap Y \Sigma ^ { * }$ say $x = y r$ with $y \in Y$ . Since uθy and both uv and yr lie in the prime class $P ,$ Lemma 7.7 gives vθr. Hence $r \in Z$ and $x \in Y \cdot Z$ . This proves the equality, and the minimum-length identity is immediate. □

Lemma 15.10 (Finite prime cut separation). Assume finite relative prime spectrum and unit separation. Then $\eta _ { \mathrm { c u t } } ( L , h ) < \infty$ . For every prime P and every nontrivial cut $\omega ( P ) = u v$ , there is a word

$$
x _ { P , u , v } \in P \setminus \left( [ u ] _ { \theta } \cdot [ v ] _ { \theta } \right)
$$

with

$$
| x _ { P , u , v } | \leq \eta _ { \mathrm { c u t } } ( L , h ) .
$$

Consequently, if $( \ell _ { P } , r _ { P } )$ is a live context for P, then

$$
\ell _ { P } x _ { P , u , v } r _ { P } \in L
$$

is a positive certificate refuting the false exact split $P = [ u ] _ { \theta } \cdot [ v ] _ { \theta }$

Proof. For every displayed cut, [u]<sub>θ</sub> and [v]<sub>θ</sub> are live non-unit relative classes. If

$$
P = [ u ] _ { \theta } \cdot [ v ] _ { \theta } ,
$$

then P would be composite, contrary to primality. Thus the displayed set diference is nonempty. There are finitely many primes and finitely many cuts of each canonical prime word, so the maximum of the finitely many minimum witness lengths is finite. □

Corollary 15.11 (Finite prime certification). Suppose the observed relative partition is exact. If, for every true prime $P ,$ the sample contains $\omega ( P )$ and the cut-separation witness $\ell _ { P } x _ { P , u , v } r _ { P }$ for every nontrivial cut $\omega ( P ) = u v ,$ then the cut test of Corollary 4.3 identifies exactly the true relative primes among the observed classes.

Proof. For a true prime, every candidate split exposed by a cut of $\omega ( P )$ is refuted by its corresponding cut-separation witness. Conversely, if a live class X is composite, write $X = Y \cdot Z$ exactly with $Y , Z$ live non-unit classes. For every $w \in X$ , exact equality gives a factorization $w = y z$ with $y \in Y$ and $z \in Z$ . Hence the corresponding cut candidate is genuine and no positive word of X can refute it. Therefore a composite class is never certified as prime. □

## 15.5 Polynomial recovery of exact factors and valid rules

Once the observed partition and prime set are correct, Corollary 7.23 turns factor recovery into a finite shortest-path problem. For an observed word $w = a _ { 1 } \cdot \cdot \cdot a _ { n }$ , form its relative prime interval $D A G$ with vertices $0 , \ldots , n$ and an edge $i  j$ whenever the substring $a _ { i + 1 } \cdot \cdot \cdot a _ { j }$ belongs to an observed prime class. A minimum-edge path reads the exact prime factorization of [w].

To recover valid productions, enumerate triples $( N , A , Q )$ of an observed target prime N, a first prime A, and an observed live relative class Q. Let $\alpha ( Q ) = \operatorname { p f } ( Q )$ be the exact prime factorization returned by the interval DAG. Test the candidate $N \to A \alpha ( Q )$ for quotient correctness and reject it if any proper contiguous block of length at least two has prime quotient. All tests reduce to the finite observed partition and CFG membership in $W _ { h } ( D )$

Lemma 15.12 (Rule-recovery completeness under PTLD). After the observed partition and prime set are correct, the preceding triple procedure returns all and only the valid productions whose witnesses occur in the sample. If every target valid production has its canonical witness in the sample, the recovered branching-rule set is exactly V.

Proof. Soundness is by the explicit correctness and non-pleonasticity tests. For completeness, let $N  A \beta$ be target-valid. VTE makes $\beta$ the exact prime factorization of its tail class $Q = [ { \overline { { \beta } } } ]$ A canonical rule witness exposes a representative of $Q$ as a substring. Corollary 7.23 therefore returns $\alpha ( Q ) = \beta$ , so the triple $( N , A , Q )$ regenerates the target rule. □

## 15.6 Characteristic sample

For every prime P, among the contexts of minimum total length satisfying Lemma 15.6, fix the lexicographically least pair $( \ell _ { P } , r _ { P } )$ . This tie-break makes the learner deterministic; once the relative partition is exact, any live context for P induces the same class test. Define four finite families.

Anchors. $\mathcal { C } S _ { \mathrm { a n c } } : = \{ \ell _ { P } \omega ( P ) r _ { P } : P \in \mathcal { P } \}$

Rule witnesses. For every valid rule $P  \alpha , \mathcal { C } S _ { \mathrm { r u l e } } : = \{ \ell _ { P } \omega ( \alpha ) r _ { P } : P  \alpha \in \mathcal { V } \}$ , where $\omega ( \alpha )$ concatenates the canonical shortest representatives of the prime symbols in $\alpha$

Cut-separation witnesses. $\mathcal { C } S _ { \mathrm { c u t } } : = \{ \ell _ { P } x _ { P , u , v } r _ { P } : P \in \mathcal { P } , \omega ( P ) = u v , u , v \neq \varepsilon \}$

Lexical and start witnesses. For every live terminal a, include one sentence $\ell _ { [ a ] } a r _ { [ a ] } ;$ this ensures recovery of all lexical alternatives even when several letters lie in the same relative prime. For each accepting relative class X, include the canonical word obtained by concatenating the $\omega ( P )$ along its unique exact prime factorization. Let the union of these two finite families be $\mathcal { C } S _ { \mathrm { l e x / s t a r t } } .$

Set

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { C S } ^ { \mathrm { P T L D } } ( L , h ) : = \mathcal { C } S _ { \mathrm { a n c } } \cup \mathcal { C } S _ { \mathrm { r u l e } } \cup \mathcal { C } S _ { \mathrm { c u t } } \cup \mathcal { C } S _ { \mathrm { l e x / s t a r t } } . } \end{array} } \end{array}
$$

Lemma 15.13 (Characteristic weak completeness). $I f D \subseteq L$ contains $\mathcal { C } S _ { \mathrm { a n c } } \cup \mathcal { C } S _ { \mathrm { r u l e } } \cup \mathcal { C } S _ { \mathrm { l e x / s t a r t } } ,$ then $L ( W _ { h } ( D ) ) = L$

Proof. Soundness is Theorem 15.7. For the converse, compare an anchor $\ell _ { P } \omega ( P ) r _ { P }$ with the rule witness $\ell _ { P } \omega ( \alpha ) r _ { P }$ for a valid rule $P \to \alpha$ . Their central factors have the same $h \mathrm { - t y p e }$ and the same sample context, so $W _ { h } ( D )$ contains the guarded substitution between their substring nonterminals. Binary splitting then simulates the canonical production $P  \alpha$ . Lexical witnesses provide every terminal alternative. Finally, each start witness is itself a positive sentence whose substring nonterminal splits into the unique exact prime factorization of its accepting class. Hence $W _ { h } ( D )$ simulates every derivation of $G _ { L , h } ^ { \star }$ and generates all of $L .$ □

The size of the characteristic set is explicit. PTLD gives $| \nu | \leq q ^ { 2 }$ , every canonical prime word has length at most $\tau ,$ and there are at most |M| accepting relative classes. There are at most

$$
\sum _ { P \in \mathcal { P } } ( | \omega ( P ) | - 1 ) \leq q ( \tau - 1 )
$$

cut-separation witnesses. Therefore

$$
| { \mathcal { C } } S ^ { \mathrm { P T L D } } | \leq q ^ { 2 } + q \tau + | \Sigma | + | M | .
$$

Anchors and rule witnesses have length at most $( q + 1 ) \tau$ , a cut-separation witness has length at most $q \tau + \eta _ { \mathrm { c u t } } ( L , h )$ , lexical witnesses have length at most $q \tau + 1$ , and start witnesses have length at most τ. Consequently

$$
\operatorname* { m a x } _ { w \in \mathcal { C } \mathcal { S } ^ { \operatorname { P T L D } } } \vert w \vert \leq \operatorname* { m a x } \big \{ ( q + 1 ) \tau , \ q \tau + \eta _ { \mathrm { c u t } } ( L , h ) , \ q \tau + 1 \big \} .
$$

Thus the target has an explicit finite characteristic sample. Quantitatively, for fixed h and fixed alphabet, its total data size is polynomial in $q , \tau$ , and $\eta _ { \mathrm { c u t } } ( L , h )$

## 15.7 Relative-ASGOLD

The name follows Clark’s ASGOLD strong learner [3]. Relative-ASGOLD is its fixed-h guarded analogue: the observable partition is refined by the fixed monoid type, while the reconstruction target is the canonical relative-prime grammar.

Primitive tests. Once the weak core is complete and the canonical context $( \ell _ { P } , r _ { P } )$ of an observed prime P is present, membership of an arbitrary word z in P is tested by

$$
\mathrm { C l a s s T e s t } _ { D } ( z , P ) \iff h ( z ) = h ( \omega ( P ) ) { \mathrm { ~ a n d ~ } } \ell _ { P } z r _ { P } \in L ( W _ { h } ( D ) ) .
$$

For an observed candidate class X with canonical representative w, the prime test is finite. For each nontrivial cut $w = u v$ , let $Y = [ u ]$ and $Z = [ v ]$ in the observed partition. An observed word $x \in X$ refutes the candidate exact split $X = Y \cdot Z$ exactly when no cut $x = x _ { 1 } x _ { 2 }$ satisfies both ClassTest $_ D ( x _ { 1 } , Y )$ and ClassTest $_ { D } ( x _ { 2 } , Z )$ . All such cuts are among observed substrings, so this is a finite search; ${ \mathrm { I s P r i m e } } _ { D } ( X )$ accepts precisely when every canonical cut has an observed refuter. The cut-separation part of the characteristic sample guarantees those refuters for every true prime.

Using the same primitive, the canonicalizer implements $\mathrm { P r i m e F a c t } _ { D }$ by a minimum path in the relative prime interval DAG, $\mathrm { C o r r e c t } _ { D } ( P , \alpha )$ by testing the quotient class of a representative of $\alpha ,$ and $\mathrm { V a l i d } _ { D } ( P , \alpha )$ by additionally testing all proper contiguous blocks of length at least two. Thus every structural test used below reduces to finitely many h-computations and CFGmembership tests.

The strong learner takes a finite positive sample D and performs the following finite operations:

1. construct the guarded weak grammar $W _ { h } ( D )$ ;

2. partition observed substrings by the signatures $\sigma _ { D , W _ { h } \left( D \right) } ;$

3. choose the shortlex least observed representative of each class;

4. apply the finite observed-refuter cut test above, justified by Corollary 15.11, to select sample primes;

5. recover prime decompositions by minimum paths in the relative prime interval DAGs;

6. enumerate the $( N , A , Q )$ triples and keep exactly the correct, non-pleonastic rules;

7. recover observed lexical rules and accepting start factorizations;

8. output the resulting grammar using canonical shortlex labels.

Call this learner Relative-ASGOLD.

Theorem 15.14 (PTLD strong structural reconstruction). Fix a finite monoid morphism h : $\Sigma ^ { * }  M$ . Let $L \subseteq \Sigma ^ { + }$ be an h-substitutable context-free language satisfying unit separation, finite relative prime spectrum, and PTLD. Then Relative-ASGOLD identifies the canonical direct relative-prime grammar $G _ { L , h } ^ { \star }$ strongly from positive strings alone.

Moreover, each hypothesis update is computable in time polynomial in the size of the current finite sample. The target admits an explicit finite characteristic sample; its cardinality is at most $q ^ { 2 } + q \tau + | \Sigma | + | M | .$ , and its maximum word length is bounded by

$$
\begin{array} { r } { \operatorname* { m a x } \bigl \{ ( q + 1 ) \tau , \ q \tau + \eta _ { \mathrm { c u t } } ( L , h ) , \ q \tau + 1 \bigr \} . } \end{array}
$$

Proof. Let D contain $\mathcal { C } S ^ { \mathrm { P T L D } }$ . Lemma 15.13 gives $L ( W _ { h } ( D ) ) = L .$ , so Lemma 15.8 makes the observed relative partition exact. Corollary 15.11 then recovers exactly the true observed primes. All target primes are present because their anchors are present. Corollary 7.23 recovers exact prime factorizations, and Lemma 15.12 recovers exactly the valid rules. Lexical and start witnesses recover the remaining canonical structure. Canonical shortlex names are already present and cannot later change, so every positive supersample of D produces literally the same grammar $G _ { L , h } ^ { \star }$

For complexity, let $N = \| D \|$ and put $s = | \operatorname { S u b } ( D ) |$ . Then $s \le N ( N + 1 ) / 2 = O ( N ^ { 2 } )$ . The grammar $W _ { h } ( D )$ has s substring nonterminals; its binary split rules are bounded by $O ( s N )$ while its guarded substitution rules are bounded by $O ( s ^ { 2 } )$ , so its total size is polynomial in N. Signature construction uses at most $s ^ { 2 } = O ( N ^ { 4 } )$ CFG-membership tests, each on a word of length $O ( N )$ in a grammar of polynomial size. The observed-refuter prime tests inspect at most $O ( s N )$ canonical cuts and only polynomially many observed cuts and class tests. Every relative prime interval DAG has $O ( N )$ vertices and $O ( N ^ { 2 } )$ possible edges, so minimum-path factor recovery is polynomial. Finally, the rule stage considers at most $s ^ { 3 } = O ( N ^ { 6 } )$ triples $( N , A , Q )$ of observed classes, with correctness and non-pleonasticity checked by polynomially many class and interval tests. Since ordinary CFG membership is polynomial in grammar and input size, every stage of one hypothesis update is polynomial in N. The characteristic-sample bounds were established above. □

Remark 15.15 (Relation to the fixed-h weak learner). Theorem 15.14 does not require the general fixed-h weak learner as a black box: the guarded substring grammar supplies the weak core directly on the PTLD subclass. The earlier fixed-h reconstruction theorem remains more general, while the present theorem trades the algebraic PTLD restriction for convergence to a canonical structural representation.

## 15.8 A polynomial-data PTLD subbranch and the thickness obstruction

The only uncontrolled length parameter in Theorem 15.14 is the cut-separation radius. There is, however, a natural PTLD subbranch on which it vanishes identically.

Definition 15.16 (Lexically anchored PTLD). A PTLD pair with finite relative prime spectrum is lexically anchored if every relative prime P contains at least one terminal letter, that is,

$$
P \cap \Sigma \neq \emptyset \qquad ( P \in { \mathcal { P } } ) .
$$

Corollary 15.17 (Polynomial time and data on the lexically anchored branch). Fix the finite alphabet Σ and finite observer h. On the class of unit-separated, h-substitutable, lexically anchored PTLD languages with finite relative prime spectrum, Relative-ASGOLD has polynomial update time and polynomial characteristic data in the size of the explicit canonical direct relative-prime grammar.

Proof. If every prime contains a letter, then its canonical shortlex word has length one. Hence no canonical prime word has a nontrivial cut and

$$
\eta _ { \mathrm { c u t } } ( L , h ) = 0 .
$$

Moreover, for a prime sequence $\alpha = P _ { 1 } \cdot \cdot \cdot P _ { m }$ one has $\ell ( \alpha ) = m$ , so the structural thickness $\tau$ is just the largest right-hand-side length of the explicit canonical grammar. Theorem 15.14 therefore gives

$$
| { \mathcal { C } } S ^ { \mathrm { P T L D } } | \leq q ^ { 2 } + q \tau + | \Sigma | + | M |
$$

and

$$
\operatorname* { m a x } _ { w \in { \cal C S } ^ { \mathrm { P T L D } } } | w | \leq ( q + 1 ) \tau .
$$

For fixed Σ and h, both $q$ and $\tau$ are bounded by the explicit encoding size of the canonical grammar. Thus the total characteristic data size is polynomial in that representation size. Polynomial update time is already part of Theorem 15.14. □

This does not imply a polynomial transfer from an arbitrary compact CFG presentation to the canonical PTLD representation. The obstruction is already present for finite unary languages.

Proposition 15.18 (Exponential thickness under compact CFG presentation). There is a family of unit-separated substitutable PTLD languages having $q = 1$ and $\eta _ { \mathrm { c u t } } = 0$ that admit CFGs of size $O ( n )$ but whose canonical PTLD thickness is $2 ^ { n }$

Proof. Let

$$
L _ { n } = \{ a ^ { 2 ^ { n } } \}
$$

over the unary alphabet, with the trivial one-element group observer. The live factors are $a ^ { i }$ for $0 \leq i \leq 2 ^ { n }$ . If $a ^ { i }$ and $a ^ { j }$ share an accepting context $( a ^ { r } , a ^ { s } )$ , then $r + i + s = 2 ^ { n } = r + j + s $ , so $i = j ;$ hence $L _ { n }$ is substitutable. The empty class is the singleton {ε}, and PTLD is automatic for the trivial group.

Every live non-unit relative class is the singleton $\{ a ^ { i } \}$ , and

$$
\{ a ^ { i } \} = \{ a \} ^ { i } .
$$

Thus the only relative prime is $P = \{ a \}$ , so $q = 1$ and $\eta _ { \mathrm { c u t } } = 0$ . The accepting class has exact start factorization $P ^ { 2 ^ { n } }$ , whence the canonical direct grammar has

$$
\tau = 2 ^ { n } .
$$

On the other hand $L _ { n }$ has an acyclic doubling grammar of size $O ( n )$ :

$$
A _ { 0 }  a , \qquad A _ { i + 1 }  A _ { i } A _ { i } \quad ( 0 \leq i < n ) , \qquad S  A _ { n } .
$$

Therefore no polynomial in the size of an arbitrary compact generating CFG can bound the canonical PTLD thickness in general. □

The proposition isolates the remaining representation issue: the general PTLD theorem is parameterized by canonical thickness, just as compact CFGs can hide exponentially long shortest yields. Polynomial transfer from an arbitrary generating grammar therefore requires additional restrictions on that presentation or on the language class.

## 15.9 A PTLD language beyond every finite k, ℓ-substitutable class

Example 15.19 (A nonregular PTLD language beyond all finite context windows). Over $\Sigma =$ $\{ a , b , x , y , z \}$ put

$$
L ^ { \star } = \{ a ^ { n } x b ^ { n } : n \geq 0 \} \cup \{ a ^ { n } y b ^ { n } : n \geq 0 \} \cup \{ z a ^ { n } x b ^ { n } : n \geq 0 \} .
$$

Let the observer take values in the finite group $C _ { 2 } \times C _ { 2 }$ and be defined on letters by

$$
h ( a ) = h ( b ) = h ( x ) = ( 0 , 0 ) , \qquad h ( y ) = ( 1 , 0 ) , \qquad h ( z ) = ( 0 , 1 ) .
$$

The language is deterministic context-free: a pushdown automaton optionally records the initial z, pushes the a-run, reads the unique center x or $y ,$ and pops on the b-run, rejecting the y-branch when the initial marker was z. It is nonregular because

$$
L ^ { \star } \cap a ^ { * } x b ^ { * } = \{ a ^ { n } x b ^ { n } : n \geq 0 \} .
$$

We first compare the example directly with Yoshinaka’s bounded-context hierarchy. Recall that L is k, ℓ-substitutable when, for every $u \in \Sigma ^ { k }$ and $v \in \Sigma ^ { \ell }$ , two nonempty guarded strings uy<sub>1</sub>v and $u y _ { 2 } v$ that occur in one common outer context are interchangeable in every outer context $[ { \mathcal { Q } } ] .$

Proposition 15.20 (Outside every finite $k , \ell$ window). For every finite $k , \ell \geq 0$ , the language $L ^ { \star }$ is not k, ℓ-substitutable.

Proof. Fix $k , \ell$ and choose $n \geq \operatorname* { m a x } \{ k , \ell \}$ . Put

$$
u = a ^ { k } , \qquad v = b ^ { \ell } ,
$$

$$
y _ { 1 } = a ^ { n - k } x b ^ { n - \ell } , \qquad y _ { 2 } = a ^ { n - k } y b ^ { n - \ell } .
$$

Then

$$
u y _ { 1 } v = a ^ { n } x b ^ { n } \in L ^ { \star } , \qquad u y _ { 2 } v = a ^ { n } y b ^ { n } \in L ^ { \star } ,
$$

so the two guarded strings occur in the common empty outer context. But the outer left context z separates them:

$$
z u y _ { 1 } v = z a ^ { n } x b ^ { n } \in L ^ { \star } , \qquad z u y _ { 2 } v = z a ^ { n } y b ^ { n } \notin L ^ { \star } .
$$

Hence the $k , \ell$ substitution condition fails. Since $k , \ell$ were arbitrary, one language lies outside every finite level of the hierarchy. □

The live syntactic classes admit simple normal forms. Put

$$
U = \{ \varepsilon \} , \quad \quad A _ { i } = \{ a ^ { i } \} ( i \geq 1 ) , \quad \quad B _ { j } = \{ b ^ { j } \} ( j \geq 1 ) ,
$$

$$
H _ { i } = \{ z a ^ { i } \} ~ ( i \geq 0 ) ,
$$

and, for d $\in \mathbb { Z }$

$$
X _ { d } = \{ a ^ { i } x b ^ { j } : i , j \geq 0 , \ i - j = d \} , \qquad Y _ { d } = \{ a ^ { i } y b ^ { j } : i , j \geq 0 , \ i - j = d \} .
$$

Finally, for $d \geq 0$ , put

$$
T _ { d } = \{ z a ^ { j + d } x b ^ { j } : j \geq 0 \} .
$$

Every live factor is in exactly one displayed class. Indeed, a factor either lies wholly in an $a { - } r u n _ { \mathrm { ; } }$ wholly in a b-run, is a prefix beginning with the unique z, or contains the unique center x or y; a factor containing both z and x must be a prefix through the center, which gives $T _ { d }$ with $d \geq 0$

The displayed sets are precisely the syntactic classes. For $X _ { d . }$ , an accepting context has the form

$$
\begin{array} { r } { ( a ^ { p } , b ^ { p + d } ) \quad o r \quad ( z a ^ { p } , b ^ { p + d } ) , \qquad p \ge \operatorname* { m a x } \{ 0 , - d \} , } \end{array}
$$

whereas for $Y _ { d }$ only the first family is available. Thus each of these classes depends only on d. For $T _ { d }$ the two-sided distribution is the singleton

$$
{ \mathcal { D } } _ { L ^ { \star } } ( T _ { d } ) = \{ ( \varepsilon , b ^ { d } ) \} .
$$

The singleton families $A _ { i } , B _ { j } , H _ { i }$ are distinguished by their required completion lengths. These descriptions also separate the displayed families from one another.

They make h-substitutability transparent. The observer types are

$$
h ( U ) = h ( A _ { i } ) = h ( B _ { j } ) = h ( X _ { d } ) = ( 0 , 0 ) ,
$$

$$
h ( Y _ { d } ) = ( 1 , 0 ) , \qquad h ( H _ { i } ) = h ( T _ { d } ) = ( 0 , 1 ) .
$$

There is no live class of type (1, 1). Among classes of type (0, 0), distinct $A _ { i }$ ’s or $B _ { j }$ ’s require diferent completion lengths; an A -context has the unique center to the right of the factor, a $B _ { j } – c o n t e x t$ has it to the $l e f t ,$ and an $X _ { d } - c o n t e x t$ has the center inside the factor, so these three shapes cannot share an accepting context across families. The empty class U cannot share an accepting context with a nonempty class of the same type: inserting $A _ { i }$ or $B _ { j }$ destroys the balance of a completed sentence, while inserting an $X _ { d }$ introduces a second center. Among the $Y _ { d }$ ’s a common accepting context forces the same imbalance d. Among type (0, 1) classes, an $H _ { i }$ requires the center in its right context while a $T _ { d }$ already contains the center; hence the two families have disjoint distributions, and within each family the completion length fixes the index. Therefore equal h-type plus one common accepting context always forces equality of the syntactic class. Thus $L ^ { \star }$ is h-substitutable. The same normal forms give $\left[ \varepsilon \right] _ { L ^ { \star } , h } = U = \{ \varepsilon \}$ , so unit separation holds.

Because $C _ { 2 } \times C _ { 2 }$ is a group, PTLD holds: from $q r = p = q r ^ { \prime }$ one cancels q on the left and obtains $r = r ^ { \prime }$ . The relative prime spectrum has exactly five elements,

$$
A : = A _ { 1 } = \{ a \} , \qquad B : = B _ { 1 } = \{ b \} , \qquad Z : = H _ { 0 } = \{ z \} , \qquad X : = X _ { 0 } , \qquad Y : = Y _ { 0 } .
$$

The five displayed classes are prime because each has a one-letter shortest representative. Every other live non-unit class has one of the exact decompositions

$$
A _ { i } = A ^ { i } , \qquad B _ { j } = B ^ { j } , \qquad H _ { i } = Z A ^ { i } ,
$$

$$
X _ { d } = \left\{ \begin{array} { l l } { { A ^ { d } X , } } & { { d > 0 , } } \\ { { X , } } & { { d = 0 , } } \\ { { X B ^ { - d } , } } & { { d < 0 , } } \end{array} \right. \quad Y _ { d } = \left\{ \begin{array} { l l } { { A ^ { d } Y , } } & { { d > 0 , } } \\ { { Y , } } & { { d = 0 , } } \\ { { Y B ^ { - d } , } } & { { d < 0 , } } \end{array} \right.
$$

and

$$
T _ { d } = Z A ^ { d } X \qquad ( d \geq 0 ) .
$$

Hence $q = 5 ;$ in particular the example has finite relative prime spectrum despite lying outside every finite k, ℓ window.

The normal forms also determine the valid branching rules. A correct branching return to X must have the form

$$
X  A ^ { m } X B ^ { m } \qquad ( m \geq 1 ) ,
$$

and a correct branching return to Y must have the form

$$
Y  A ^ { m } Y B ^ { m } \qquad ( m \geq 1 ) .
$$

No other prime target admits a branching return. When $m > 1$ , the right-hand side contains the proper prime-valued block AXB or AYB, respectively. Thus the only valid branching rules are

$$
X  A X B , \qquad Y  A Y B .
$$

The canonical grammar is therefore

$$
\begin{array} { l } { { S  X \mid Y \mid Z X , } } \\ { { X  A X B \mid x , } } \\ { { Y  A Y B \mid y , } } \\ { { A  a , B  b , Z  z . } } \end{array}
$$

Here $q \ = \ 5$ and $\tau \ = \ 3$ . Every canonical prime word has length one, so $\mathcal { C } S _ { \mathrm { c u t } } ~ = ~ \mathcal { O }$ and $\eta _ { \mathrm { c u t } } ( L ^ { \star } , h ) = 0$ . Since $| \Sigma | ~ = ~ 5$ and $| C _ { 2 } \times C _ { 2 } | = 4$ , Theorem 15.14 gives the explicit (not necessarily minimal) estimates

$$
| { \mathcal { C } } { \mathcal { S } } ^ { \mathrm { P T L D } } | \leq 5 ^ { 2 } + 5 \cdot 3 + 5 + 4 = 4 9
$$

and

$$
\operatorname* { m a x } _ { w \in \mathcal { C } ^ { \mathrm { { P T L D } } } } | w | \leq ( 5 + 1 ) 3 = 1 8 .
$$

All five primes are lexically anchored. Hence Corollary 15.17 applies: this same nonregular witness lies in a PTLD subbranch with polynomial update time and polynomial characteristic data in canonical grammar size. Thus the polynomial-data strong-learning branch itself reaches outside the entire finite k, ℓ-substitutable hierarchy, while retaining a five-prime recursive canonical presentation.

## 15.10 Finite-state strong reconstruction beyond FRP

The direct learner above cannot cover the 36-element witness, because that pair has infinitely many valid productions. FSRP nevertheless supplies a canonical finite-state structural target through the minimal residual controllers of Section 10.

Theorem 15.21 (Weak-to-strong finite-state lifting). Fix $h : \Sigma ^ { * } \to M$ . Consider a class of hsubstitutable context-free languages satisfying unit separation, finite relative prime spectrum, and FSRP. Suppose the class has a weakly behaviorally correct positive-data learner whose hypotheses are CFGs. Then there is a positive-data learner that identifies in the strong Gold sense the canonical finite-state relative-prime presentation, and hence its fixed canonical CFG compilation.

We first isolate the efective facts used by the lifting construction.

Lemma 15.22 (Efective relative-class oracle from a correct CFG). Let G be a CFG with $L ( G ) = L$ , where L is h-substitutable. Then factorhood and membership in every live relative class are decidable uniformly from G and h.

Proof. For a word u, factorhood is equivalent to $L ( G ) \cap \Sigma ^ { * } u \Sigma ^ { * } \neq \emptyset$ . The intersection of a CFL with a regular language is context-free and CFL emptiness is decidable, so factorhood is decidable. If u is live, enumerate pairs $( \ell , r )$ until $\ell u r \in L ( G )$ ; such a pair exists and CFG membership is decidable. Fix one such context. Then, for every v, $v \in [ u ] _ { L , h } \iff h ( v ) =$ $h ( u )$ and $\ell v r \in L ( G )$ . The forward implication follows from $u \equiv _ { L } v$ . Conversely, if the righthand side holds, u and v have the same h-type and share the accepting context (ℓ, r), so hsubstitutability gives $u \equiv _ { L }$ v and hence $u \theta _ { L , h } v$ □

Lemma 15.23 (Prime-spectrum stabilization). Assume the relative prime spectrum is finite. Using the oracle of Lemma 15.22, there is a computable stagewise procedure whose finite prime hypotheses eventually stabilize to exactly $\mathcal { P } _ { L , h }$

Proof. Enumerate live relative classes by their shortlex least representatives. By Corollary 4.3, every exact binary factorization of a live non-unit class X occurs at one of the finitely many cuts of its shortest representative x. For a cut $x = u v$ , let $Y = [ u ]$ and $Z = [ v ]$ . Congruence gives $Y Z \subseteq X$ . If equality fails, enumerate words of X until a counterexample in $X \setminus Y Z$ is found; membership in $Y Z$ is decidable by testing the finitely many cuts of the candidate word with the relative-class oracle. Thus every false decomposition candidate is eventually refuted.

At stage $s ,$ let $\mathcal { P } _ { s }$ consist of those classes among the first s enumerated live non-unit classes for which every nontrivial cut candidate of the shortest representative has already been refuted by stage $s ;$ unresolved classes are simply omitted from $\mathcal { P } _ { s }$ . If X is composite, at least one genuine exact cut is never refuted, so X never enters $\mathcal { P } _ { s }$ . If X is prime, all of its finitely many cut candidates are false and hence are eventually refuted, after which X belongs permanently to every subsequent $\mathcal { P } _ { s }$ . Since the true relative prime spectrum is finite, all true primes are enumerated and certified by some common finite stage. Therefore $\mathcal { P } _ { s }$ eventually stabilizes exactly to $\mathcal { P } _ { L , h }$ □

Lemma 15.24 (Efective validity after prime stabilization). $A f t e r$ the prime alphabet has stabilized, membership in each language Val<sub>P</sub> is decidable uniformly.

Proof. For a prime sequence $\alpha = P _ { 1 } \cdot \cdot \cdot P _ { k } ,$ concatenate fixed canonical representatives of the $P _ { i }$ . The relative-class oracle decides the quotient product of every contiguous block. Hence it decides whether the whole sequence returns to $P$ and whether any proper contiguous block of length at least two returns to a prime. These are exactly correctness and non-pleonasticity, so membership in ${ \mathrm { V a l } } _ { P }$ is decidable. □

Lemma 15.25 (Canonical DFA convergence). Let $R \subseteq { \mathcal { P } } ^ { * }$ be regular and suppose membership in R is decidable. There is a computable sequence of normalized DFAs that eventually stabilizes to the canonical minimal DFA of R.

Proof. At stage t, enumerate normalized complete DFAs with at most t states and retain those agreeing with R on all words of length at most t. If no such DFA exists, output a fixed normalized one-state default DFA for that stage. Otherwise choose first by minimum number of states and then by a fixed canonical encoding. Let the minimal DFA of R have s states. There are only finitely many normalized DFAs with at most s states. Every non-equivalent one difers from R on a finite word; let b be the maximum length of a shortest distinguishing word among these finitely many competitors. For $t \geq \operatorname* { m a x } \{ s , b \}$ every smaller or same-size wrong DFA is eliminated, and the chosen DFA is exactly the canonical minimal DFA. It remains so thereafter. □

Lemma 15.26 (Stabilization of the lexical and start structure). Under unit separation and finite relative prime spectrum, the lexical component and the start component of $\mathfrak { G } _ { L , h } ^ { \mathrm { f s } }$ are limitcomputable from a correct $C F G$

Proof. There are finitely many letters, so $\operatorname { L e x } ( P ) = P \cap \Sigma$ is determined by finitely many classmembership tests. For each $m \in M$ , the language $L \cap h ^ { - 1 } ( m )$ is context-free and emptiness is decidable. If nonempty, h-substitutability and the empty context imply that it is a single accepting relative class, so there are at most $| M |$ such classes.

For an accepting non-unit class X, choose a shortest representative x. Corollary 4.3 gives a finite list of all exact prime-factorization candidates directly from the cuts of x. For each candidate $\alpha ,$ congruence gives ${ \bar { \alpha } } \subseteq X$ whenever its quotient product is X; if equality fails, enumerate words of X until a counterexample in $X \backslash \bar { \alpha }$ appears. Thus every false candidate is eventually eliminated and all exact start decompositions stabilize. $\mathrm { I f } ~ \varepsilon \in ~ L$ , the empty start alternative is detected directly and recorded separately. □

Proof of Theorem 15.21. Let W be the assumed weakly behaviorally correct learner and let $T$ be a positive text for a target L. At stage t, apply the preceding canonicalization procedures with finite search budget t to the CFG $G _ { t } ~ = ~ \mathcal { W } ( T [ t ] )$ and output the resulting finite-state relative-prime presentation.

By weak behavioral correctness there is $t _ { 0 }$ such that $L ( G _ { t } ) = L$ for every $t \geq t _ { 0 }$ . From that point onward Lemma 15.22 supplies the exact relative-class predicate, and Lemma 15.23 makes the finite prime alphabet stabilize. Lemma 15.24 then gives exact membership predicates for all ${ \mathrm { V a l } } _ { P }$ . Since the target has FSRP, each ${ \mathrm { V a l } } _ { P }$ is regular, so Lemma 15.25 makes every canonical minimal DFA stabilize. Equality of residual languages represented by the stabilized DFAs is decidable by standard DFA equivalence; identifying equal residuals across the finitely many start languages ${ \mathrm { V a l } } _ { P } .$ and then discarding any occurrence of the empty residual ∅, therefore yields exactly the shared nonempty partial residual-state controller $\mathcal { Q } _ { L , h }$ of Definition 10.11. Lemma 15.26 simultaneously stabilizes the lexical and start components.

All choices are canonical—shortlex representatives, normalized DFA encodings, and residual languages themselves as controller states—and depend only on $( L , h )$ once the weak grammar is language-correct. Hence there is a stage after which the whole tuple ${ \mathfrak { G } } _ { L , h } ^ { \mathrm { f s } }$ is output literally unchanged. The fixed compilation of Section 10 therefore also stabilizes to its canonical finite CFG. This is strong Gold identification. □

Remark 15.27. Theorem 15.21 is a computable limit result, not a polynomial-time claim. The state bound of the target residual controller is not assumed known, and the DFA search is deliberately enumeration-based. Its contribution is the canonical relative structural target being reconstructed, not the standard finite-automaton enumeration argument. Theorem 15.14 is the eficient algebraically rigid subcase; the 36-element UF–FRP witness lies outside that direct branch but inside the finite-state branch.

## 16 Current hierarchy and separations

Throughout the following diagram, L is assumed h-substitutable and unit-separated. Implications involving FRP and the quadratic rule bound additionally assume a finite relative prime spectrum.

The established implication structure is now

$$
\mathrm { f a c t o r ~ c a n c e l l a t i o n } \Longrightarrow \mathrm { P T L D } \Longrightarrow \mathrm { U F } ,
$$

$$
\begin{array} { r } { \begin{array} { c } { \mathrm { P T L D } \Longrightarrow \mathrm { V T E } \Longrightarrow \mathrm { V T E } _ { 1 } \Longrightarrow \mathrm { F R P } \Longrightarrow \mathrm { F S R P } , } \\ { \mathrm { P T L D } \Longrightarrow \mathrm { V T D } . } \end{array} } \end{array}
$$

Thus VTE and VTD are independent consequences of $\mathrm { P T L D } ;$ no implication between VTE and VTD is asserted here. The structural arrows out of PTLD are Theorems 7.21 and 7.22. The important non-implication $\mathrm { U F } \not \Rightarrow \mathrm { F R P }$ is Corollary 12.3; Example 13.5 gives the reverse non-implication, so UF and FRP are incomparable.

Theorems 7.21 and 7.22 give

$$
\mathrm { P T L D } \Longrightarrow \mathrm { U F } + \mathrm { V T E } + \mathrm { V T D } + \mathrm { F R P } , \qquad | \mathcal { V } | \leq q ^ { 2 } .
$$

The two nonregular direct-presentation examples clarify that PTLD is suficient but not necessary:
<table><tr><td></td><td>nonregular CFL</td><td>h-substitutable</td><td>PTLD</td><td>UF</td><td>FRP</td></tr><tr><td> $L _ { 0 }$ </td><td>yes</td><td>yes</td><td>no</td><td>yes</td><td>yes</td></tr><tr><td> $L ^ { \star }$ </td><td>yes</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td></tr></table>

<table><tr><td>Example</td><td>finite quotient</td><td>PTLD</td><td>exact UF</td><td> $\mathrm { V T E _ { 1 } }$ </td><td>presentation</td></tr><tr><td> $\mathbf { \Pi } _ { L ^ { \star } } ^ { L _ { 0 } } = \{ a ^ { n } b ^ { n } \} ^ { * }$ </td><td>no</td><td>no</td><td>yes</td><td>yes</td><td>FRP, ρ = 3</td></tr><tr><td></td><td>no</td><td>yes</td><td>yes</td><td>yes</td><td>FRP, ρ = 3</td></tr><tr><td>finite FRP/non-UF witness</td><td>finite</td><td>no</td><td>no</td><td>not needed</td><td>FRP</td></tr><tr><td>36-element quotient witness</td><td>yes</td><td>no</td><td>yes</td><td>no</td><td>FSRP, not FRP</td></tr><tr><td> $L ^ { \mathrm { w r a p } }$ </td><td>no</td><td>no</td><td>yes</td><td>no</td><td>nonregular CFL; FSRP, not FRP</td></tr></table>

The profile $\Pi ( L , h ) = ( q , r _ { \mathrm { r e s } } , m _ { \mathrm { e x } } ; \rho _ { \partial } , \kappa _ { \partial } )$ separates prime spectrum, residual splitting, exactfactor multiplicity, and under-saturation complexity. Section 12 realizes the sharp combination $m _ { \mathrm { e x } } = 1$ with $\rho _ { \partial } = \infty$ , showing that exact-factor ambiguity and saturation defect are independent axes.

The learning hierarchy mirrors the presentation hierarchy but is not identical to it. On the PTLD branch, Theorem 15.14 gives direct canonical SGOLD reconstruction with polynomialtime updates and an explicit finite characteristic sample; Lemma 15.9 identifies its remaining quantitative parameter with prime-prefix escape. The lexically anchored PTLD subbranch has $\eta _ { \mathrm { c u t } } = 0$ and therefore polynomial characteristic data in explicit canonical grammar size, while Proposition 15.18 shows that no such bound can be transferred uniformly from arbitrary compact CFG size. On the broader FSRP branch, Theorem 15.21 gives computable strong reconstruction of the minimal finite-state controller whenever a weakly behaviorally correct CFG learner is available. The 36-element witness and the infinite-quotient wrapped witness lie in the second branch but not the first, while Example 15.19 lies in the polynomial-data PTLD subbranch and is outside every finite k, ℓ-substitutable class.

## 17 Open problems and next steps

Question 17.1 (Absolute minimization of the UF–FRP separation). The 36-element witness of Section 12 is quotient-minimal within its own monoid: every proper quotient has FRP. Is there nevertheless a diferent finite monoid of size below 36 admitting a unit-separated h-substitutable pair with exact UF but without FRP? The present computation does not establish an absolute lower bound.

Question 17.2 (Existence of non-FSRP finite-prime examples). Does there exist a context-free fixed-h pair (L, h) that is h-substitutable and unit-separated, has finite relative prime spectrum, but fails FSRP? Equivalently, can some canonical valid right-hand-side language Val<sub>P</sub> be nonregular under these hypotheses? We are not aware of such an example. The wrapped witness of Section 13 shows that FSRP can be nontrivial even for a nonregular language with infinite relative quotient, but it still satisfies FSRP. Thus the present results do not rule out the possibility that FSRP holds automatically for all context-free fixed-h pairs with finite relative prime spectrum.

Question 17.3 (FSRP versus context-free controller presentation). If non-FSRP finite-prime examples exist, can one find one for which every Val is context-free but some Val is nonregular? Such an example would separate finite-state structural presentation from finite context-freecontroller presentation. Proposition 10.3 shows that the corresponding correct-return languages Corr<sub>P</sub> are already context-free; the extra dificulty lies entirely in factor-minimalization.

Question 17.4 (Eficient learning beyond PTLD). Theorem 15.21 learns the FSRP residual controller only by a general limit enumeration of finite automata. Which semantic conditions, weaker than PTLD but stronger than bare FSRP, provide an efective a priori bound or a polynomial procedure for recovering the minimal controller? In particular, can bounded residual splitting, bounded exact-factor multiplicity, or a bounded defect-state parameter replace primetarget thinness in an eficient strong learner?

Question 17.5 (Polynomial prime-prefix escape). Lemma 15.9 shows that under PTLD the remaining cut-separation parameter is exactly a shortest prime-prefix escape length. Can $\eta _ { \mathrm { c u t } } ( L , h )$ be bounded by a polynomial in q and τ? Equivalently, do the finitely many false prefix classes exposed by cuts of canonical prime words always admit polynomial-length escape witnesses? Corollary 15.17 answers this positively with $\eta _ { \mathrm { c u t } } = 0$ on the lexically anchored branch.

Question 17.6 (Representation-size transfer under restricted presentations). Proposition 15.18 rules out any general polynomial bound on canonical thickness in the size of an arbitrary compact

CFG, even for finite unary PTLD languages. For which familiar restricted CFG subclasses can $q , \ \tau ,$ , and $\eta _ { \mathrm { c u t } } ( L , h )$ nevertheless be bounded polynomially in the size of the given generating grammar? A positive answer for a natural linear fixed-h subclass would connect the general weak polynomial bounds of the fixed-h reconstruction paper directly to polynomial strong structural learning.

Question 17.7 (Multidimensional extension). What is the correct analogue of exact relative prime factorization, semantic residual splitting, and under-saturation controllers for multidimensional syntactic congruence and bounded-fan-out MCFGs? Yoshinaka and Clark’s tuple congruence and non-permuting linear regular functions provide the natural compositional substrate [9].

## 18 Conclusion

The central structural separation is

$$
\mathrm { U F } \not \Rightarrow \mathrm { F R P \qquad ~ a n d \qquad } \mathrm { F R P } \subsetneq \mathrm { F S R P } .
$$

The 36-element quotient witness has globally unique exact factorization but infinitely many valid prime returns, while the wrapped language $L ^ { \mathrm { w r a p } }$ lifts the same phenomenon to a nonregular context-free language with infinite relative quotient and finite prime spectrum. Together they show that exact-factor ambiguity, quotient infinitude, and direct-rule infinitude are genuinely diferent axes; FSRP captures the finite-state presentation level that remains when a finite direct rule list fails.

PTLD isolates a rigid branch of this theory. It forces unique exact factorization, validtail exactness and determinism, and the quadratic rule bound. The five-prime language $L ^ { \star }$ shows that this branch reaches beyond every finite k, ℓ-substitutable class. On the learning side, Relative-ASGOLD gives canonical strong reconstruction with polynomial-time updates and an explicit finite characteristic sample; its remaining general data-length parameter is the primeprefix escape radius, which vanishes on the lexically anchored branch containing L<sup>⋆</sup>. The unary doubling family shows separately that arbitrary compact CFGs may hide exponential canonical thickness, so representation-size transfer requires additional restrictions.

The main unresolved structural boundary lies outside FSRP: we are not aware of a contextfree fixed-h finite-prime example with a nonregular valid-return language. It also remains open whether PTLD prime-prefix escape is polynomially bounded in canonical parameters. Natural multidimensional and bounded-fan-out extensions remain to be developed.

## Code and data availability

The computational artifacts supporting the 36-element witness are publicly available in a fixed GitHub snapshot at commit e38a9b09fbba7933777d7a00d7de4a7d77c4a916:

$$
{ \begin{array} { r l } & { { \mathrm { h t t p s : / / g i t h u b . ~ c o m / g r o w u p k u r i y a m a - h u b / l e a n . } } \ { \mathrm { c f g . ~ p r o j e c t / t r e e / e 3 8 a 9 b o p t b b a 7 9 3 } } } \\ & { 3 7 7 7 \% 7 { \mathrm { a } } { \mathrm { m o d } } \ { \mathrm { 7 d } } { \mathrm { e } } 4 { \mathrm { a } } 7 { \mathrm { d } } 7 { \mathrm { 7 c } } { \mathrm { a } } { \mathrm { a } } 9 1 6 / { \mathrm { L e a n c f g P r o j e c t / R e l a t i v e - F a c t o r i z a t i o n } } } \end{array} }
$$

The snapshot contains the standard-library-only verifier verify 36.py and the deterministic machine-readable certificate certificate 36.json. Appendix A describes the theorem-facing checks.

## A Reproducible verification of the 36-element UF–FRP witness

The fixed snapshot cited in the Code and data availability statement contains the theoremfacing verifier verify 36.py and the corresponding deterministic machine-readable certificate certificate 36.json.

The file verify 36.py is a standard-library-only verifier for all theorem-facing finite claims in Section 12. From the directory containing the two files, running

python3 veri $\mathrm { f y } _ { - } 3 6 . { \tt p y } \ \mathrm { \Sigma } ^ { -- } \mathrm { j }$ son certificate\_36.generated.json

reconstructs the 64-element transformation monoid, computes the least monoid congruence ∼<sub>0</sub> generated by $h _ { 0 } ( a b ) \sim _ { 0 } h _ { 0 } ( a b a a a b )$ , and then performs the following independent finite checks.

1. The quotient has 36 elements and the identity congruence block contains only the identity transformation, hence $h ^ { - 1 } ( 1 ) = \{ \varepsilon \}$

2. All 35 nonidentity quotient classes are reachable by nonempty words, and every quotient element has a shortest representative of length at most eight.

3. Exact binary fiber-product equality identifies precisely the fifteen relative primes listed in Theorem 12.1. The equality test is a finite product search comparing the DFA for the target positive fiber with the NFA for the concatenation of the positive factor fibers; a mismatch state is exactly a witness in their symmetric diference.

4. Lemma 4.2 reduces exact factorization to cuts of shortest representatives. There are 222 prime-labeled candidates. The regular-language equality checker rejects 187 candidates and stores a shortest counterexample for each; exactly one candidate survives for each of the 35 live non-unit classes.

5. The quotient multiplication satisfies $q ^ { 2 } = q ^ { 3 }$ for $q \ = \ h ( b a a a )$ and, with $\alpha = h ( a )$ and $\beta = h ( b )$ , also verifies $\alpha q = \alpha q ^ { 2 }$ and $q \beta = q ^ { 2 } \beta$ , together with $\begin{array} { r } { h ( a ) h ( b ) \ : = \ : h ( a ) q h ( b ) \ : = \ : } \end{array}$ $h ( a ) q ^ { 2 } h ( b ) = h ( a b )$ . The proper interval classes occurring in $P _ { a } P _ { q } ^ { m } P _ { b }$ are exactly [abaaa], [baaab], and [baaabaaa], and the verifier checks the three exact composite identities displayed in Section 12.

6. The tail class is $H = [ b a a b ]$ , its unique exact factorization is $[ b ] [ a a a b ]$ , and the setwise product $\overline { { P _ { q } ^ { m } P _ { b } } }$ is a strict subset of H for every $m \geq 1$ . The certificate records explicit counterexamples for the periodic cases.

7. The prime-avoidance automaton has a productive lasso. After the prefix $P _ { a } P _ { q } ^ { 2 }$ it reaches the state whose whole-prefix class is [abaaa]; its proper-sufix classes are [baaa] and [baaabaaa]. Another [baaa] is a self-loop, while the final symbol [b] returns to the prime [ab].

8. Exhaustive congruence closure yields exactly twenty monoid congruences on $M _ { 3 6 }$ . The nineteen proper quotient sizes are $1 1 , 9 , 8 , 7 , 7 , 6 , 6 , 5 , 5 , 5 , 4 , 4 , 3 , 3 , 2 , 2 , 2 , 1$ . For each induced observer quotient the verifier recomputes the live relative classes and relative primes and checks that the productive part of the prime-avoidance graph is acyclic. Hence every proper quotient has FRP. When a proper observer quotient maps nonempty words to its identity, the verifier keeps the syntactic unit {ε} separate from the positive identity fiber, as required by $\theta = \equiv _ { L } \cap$ ker h.

The committed certificate 36.json contains the 35 unique exact factorizations, all 187 rejected shortest-cut candidates and their counterexamples, the three composite interval certificates, the prime-avoidance lasso data, and the audit record for every proper congruence quotient. The generated certificate is deterministic and can therefore be compared byte-for-byte with the committed certificate 36.json in the fixed repository snapshot, for example by running

## References

[1] Alexander Clark and R´emi Eyraud. Polynomial identification in the limit of substitutable context-free languages. Journal of Machine Learning Research, 8:1725–1745, 2007.

[2] Ryo Yoshinaka. Identification in the limit of k, l-substitutable context-free languages. In Grammatical Inference: Algorithms and Applications, Lecture Notes in Artificial Intelligence 5278, pages 266–279, Springer, 2008. doi:10.1007/978-3-540-88009-7 21.

[3] Alexander Clark. Learning Trees from Strings: A Strong Learning Algorithm for some Context-Free Grammars. Journal of Machine Learning Research, 14:3537–3559, 2013.

[4] Alexander Clark. The syntactic concept lattice: Another algebraic theory of the context-free languages? Journal of Logic and Computation, 25(5):1203–1229, 2015. doi:10.1093/logcom/ext037.

[5] Alexander Clark. Canonical Context-Free Grammars and Strong Learning: Two Approaches. In Proceedings of the 14th Meeting on the Mathematics of Language (MoL 2015), pages 99–111, 2015. doi:10.3115/v1/W15-2309.

[6] Fran¸cois Denis, Aur´elien Lemay, and Alain Terlutte. Residual finite state automata. Fundamenta Informaticae, 51(4):339–368, 2002.

[7] Masami Ito, Helmut J¨urgensen, H. J. Shyr, and Gabriel Thierrin. Outfix and infix codes and related classes of languages. Journal of Computer and System Sciences, 43(3):484–508, 1991.

[8] Fran¸cois Coste and Jacques Nicolas. Learning local substitutable context-free languages from positive examples in polynomial time and data by reduction. In Proceedings of the 14th International Conference on Grammatical Inference, Proceedings of Machine Learning Research, volume 93, pages 155–168, 2019.

[9] Ryo Yoshinaka and Alexander Clark. Polynomial time learning of some multiple contextfree languages with a minimally adequate teacher. In Formal Grammar 2010 and 2011, Lecture Notes in Computer Science 7395, pages 192–207, Springer, 2012. doi:10.1007/978- 3-642-32024-8 13.

[10] Ole Lehrmann Madsen and Bent Bruun Kristensen. LR-parsing of extended context-free grammars. Acta Informatica, 7:61–73, 1976. doi:10.1007/BF00265221.

[11] Wilf R. LaLonde. Regular right part grammars and their parsers. Communications of the ACM, 20(10):731–741, 1977.

[12] Wilf R. LaLonde. Constructing LR parsers for regular right part grammars. Acta Informatica, 11:177–193, 1979.

[13] Yo-Sub Han, Arto Salomaa, Kai Salomaa, Derick Wood, and Sheng Yu. On the existence of prime decompositions. Theoretical Computer Science, 376(1–2):60–69, 2007. doi:10.1016/j.tcs.2007.01.013.

[14] Jean Berstel, Dominique Perrin, and Christophe Reutenauer. Codes and Automata. Cambridge University Press, 2009.

[15] Colin de la Higuera. Characteristic sets for polynomial grammatical inference. Machine Learning, 27:125–138, 1997.

[16] Alexander Clark. Distributional learning of some context-free languages with a minimally adequate teacher. In Grammatical Inference: Theoretical Results and Applications, pages 24–37, Springer, 2010. doi:10.1007/978-3-642-15488-1 4.

[17] Ryo Yoshinaka. Eficient learning of multiple context-free languages with multidimensional substitutability from positive data. Theoretical Computer Science, 412(19):1821–1831, 2011. doi:10.1016/j.tcs.2010.12.058.

[18] Alexander Clark and Ryo Yoshinaka. Distributional learning of parallel multiple contextfree grammars. Machine Learning, 96(1–2):5–31, 2014. doi:10.1007/s10994-013-5403-2.

[19] Bret Tilson. Categories as algebra: An essential ingredient in the theory of monoids. Journal of Pure and Applied Algebra, 48:83–198, 1987.

[20] Takayuki Kuriyama. Distributional Learning of Context-Free Languages under Fixed Finite-Monoid Typing. arXiv:1409.6247v4 [cs.FL], 2026.