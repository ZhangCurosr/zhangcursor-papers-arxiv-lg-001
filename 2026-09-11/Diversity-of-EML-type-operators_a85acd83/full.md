# Diversity of EML-type operators

A. ODRZYWOLEK

Institute of Theoretical Physics, Jagiellonian University, Łojasiewicza 11, 30-348 Kraków, Poland

Received September 11, 2026

The discovery of the EML operator, sufficient to evaluate the standard explicit purely transcendental elementary functions, has led to considerable interest and discussion across multiple scientific disciplines. However, most authors have focused on the binary EML itself, while numerous similar variants with slightly different properties are now known. This article attempts to close this gap by enumerating and classifying them. We also take this opportunity to clarify common misconceptions related to the EML operator. The principal goal, symbolic regression within an architecture as close as possible to proven neural networks which combine matrix multiplication with a single univariate non-linear activation function, remains beyond reach. Instead, we propose a Möbius layer, with rational functions replacing matrix operations, and showcase the recently discovered activation function eml $\mathbf { \alpha } ( \mathbf { x } , 1 / \mathbf { x } )$ , which allows exp(x) and ln(x) to be recovered separately, and hence all elementary functions to be evaluated within a rational generalization of the neural network.

PACS numbers: 02.30.-f, 02.70.Wz, 07.05.Mh

## 1. Introduction and follow-up studies

The EML operator

$$
\operatorname { e m l } ( x , y ) = \exp \left( x \right) - \ln \left( y \right)\tag{1}
$$

has been proposed in [35]. It was verified extensively both numerically (including in arbitrary precision) and with a Computer Algebra System (Wolfram Mathematica). The core claim is that using the EML operator, together with input variables and the special distinguished constant 1, one can derive formulas which reconstruct all standard scientific calculator operations. These include the usual arithmetic $( + , - , \times , \div )$ , exp and ln, binary exponentiation and logarithm, integers and rationals, constants like π, e and i and all trigonometric/hyperbolic functions with their inverses. These formulas work in their respective real domains, except at the endpoints in some cases. For trigonometric functions and $\pi ,$ the EML operator must use complex intermediate values. This is done entirely within the principal logarithm branch (the fundamental strip of the complex plane). The shortest formulas are obtained using the extended real axis, ln $0 = - \infty$ and $e ^ { - \infty } = 0$ . Since then, formalized Lean4 proofs [33] have appeared supporting the reconstruction, with qualifications discussed in Subsect. 2.2. In what follows we assume the validity of the main result of [35]. The operator now has its own entry in MathWorld [53].

The surprisingly simple properties of Eq. (1) sparked fierce internet discussions1 about its possible applications and its relation to the logical Sheffer stroke [42] (NAND/NOR binary operators) which enables the computation of all Boolean functions. Indeed, the context-free grammar is extremely simple

$$
S \to 1 \mid x \mid \operatorname { e m l } ( S , S )\tag{2}
$$

and is isomorphic to binary trees, one of the best-studied structures in computer science.

One possible application of EML operators and their relatives is symbolic regression (SR). SR attempts to fit data with free-form expressions composed of arbitrary functions and operations. Usually, this requires mixing discrete and continuous approaches [13]. First, genetic algorithms generate multiparameter candidate formulas. Then, the free parameters are optimized using e.g. gradient descent. EML, eq. (1), in principle allows the discrete step to be skipped entirely by using the master formula [35] encoding a full binary expression tree of depth d

$$
T _ { \mathrm { e m l } , d } \left( x ; \{ a _ { 1 } , \ldots , a _ { k } \} , \{ b _ { 1 } , \ldots , b _ { k } \} , \{ c _ { 1 } , \ldots , c _ { k } \} \right) , k = 2 ^ { d + 1 } - 2 ,\tag{3}
$$

intentionally written in a form resembling the Gauss hypergeometric ${ } _ { 2 } F _ { 1 }$ function. $\mathrm { A }$ proof of concept for gradient-based symbolic regression has already been presented in [35]. However, depth and symbolic recovery rates were very limited. In the follow-up study [21], three depth-3 architectures were compared on over 12,000 training runs; recovery of the same target ranged from 0% to 100% depending on the architecture and training protocol, and balanced (non-chain) trees were never recovered. In these experiments, the optimization landscape, not expressivity, is therefore the limiting factor. This is precisely where we hope EML variants (Sect. 3, Table 1) might help.

EML trees were used as building blocks for simultaneous function and antiderivative discovery [5], as response modules in reduced models of biological dynamics [17], and as interpretable edge mechanisms in causal structure learning [3]. A recent theorem also establishes universal approximation by EML trees in $W ^ { k , \infty }$ , together with a practical fitting procedure [19]. Independent implementations now exist in Rust, Python, and JavaScript [16].

The rest of the article is organized as follows. In Sect. 2 we address some questions which arose after the EML discovery was made public. Sect. 3 showcases the known variants of the EML operator. In Sect. 4 we propose a new architecture upgrading neural networks, and the article is concluded with a short summary.

## 2. Relevant mathematical background and common misconceptions

Months of intense discussion of the EML operator have resulted in a much deeper understanding of its relation to prior mathematical knowledge. While EML and some of its variants, were discovered by an exhaustive bruteforce numerical sieve, more explainable approaches now exist [47]. Some new formulas are far more complex than a direct search using limited compute could reach.

In the following subsections we take the opportunity to explain common misconceptions and nitty-gritty details, which resurfaced during discussions of the EML operator.

## 2.1. All elementary functions?

Some confusion comes from the not universally accepted notion of elementary functions. The meaning of this phrase is different not only between scientific disciplines, but also between languages and countries. For example, in Poland, math teaching clearly distinguishes between explicit and implicit ones (pl. funkcje jawne/uwikłane). While in [35] unambiguous enumerative definition of what is meant by elementary functions, i.e. explicit finite expressions composed using standard scientific calculator buttons, was provided, some confusion among unprepared readers arises. Our definition is the same as in complex analysis textbooks, see e.g. [26, Chapter 5, Elementary Functions]; see also [32, p. 746]. However, some readers of [35] didn't follow further than reading the preprint title, not to mention its abstract or full text, and just a few reached the Supplementary Information. The purely mathematical definition is used mainly in differential algebra, and arose in the context of integration in finite terms. This is a historically famous problem, second only to the solution of cubics/quartics/quintics in importance for shaping the modern notion of what we mean by elementary functions. The well-known example providing a rationale for the mathematical definition of elementary function is the symbolic integration of rational functions such as $\textstyle \int { 1 } / ( x ^ { 4 } + { \mathrm { { 1 } } } ) d x$ . The classic algorithm for that is at the core of the STEM teaching curriculum, and uses decomposition of the rational integrand into the partial fractions. To achieve this, the roots of the denominator are required to be known. If we insisted that they must be found in the form of radicals, then a generic rational function with fifth or higher degree in the denominator would be not be symbolically integrable in finite terms. Therefore, solutions to all polynomial equations are adjoined (using differential algebra terminology). To the great surprise of many STEM practitioners outside pure math, solutions to e.g. $x ^ { 5 } - x + z = 0$ with unknown x and parameter z are therefore (e.g. the 2nd real root) elementary functions f(z) of the variable $z$ in the above sense, despite being explicit only after using a hypergeometric function which does not reduce to exp-log expressions [7]

$$
\begin{array} { r } { f ( z ) = z _ { 4 } F _ { 3 } \left( \{ \frac { 1 } { 5 } , \frac { 2 } { 5 } , \frac { 3 } { 5 } , \frac { 4 } { 5 } \} ; \{ \frac { 1 } { 2 } , \frac { 3 } { 4 } , \frac { 5 } { 4 } \} ; 3 1 2 5 z ^ { 4 } / 2 5 6 \right) . } \end{array}
$$

So if one interpreted title of [35] in a narrow differential-algebraic strictly mathematical sense, then EML expression would provide e.g. constructive exp-log solution $f ( a , b , c )$ to Hilbert's 13th problem, which involves roots of the 7-th order polynomial $x ^ { 7 } + a x ^ { 3 } + b x ^ { 2 } + c x + 1 = 0$ with 3 parameters $a , b , c .$ This is, of course, not what [35] claims, and the algebraic-function version (cf. Subsect. 2.7) of the Hilbert's 13th remains open.

The above didn't stop some authors from proving that EML is not able to express $\because _ { \mathrm { A L L } } ,  \mathrm { { r } }$ elementary functions, despite the difference in definitions of this phrase. An example of a constant which cannot be expressed in terms of the EML operator and 1 is Chaitin's constant [10]. However, it is also not computable. Another blog post discusses the roots of the generic quintic [44], which again is not an EML-class expression. What is expressible by EML are actually Chow's [12] elementary numbers. Carney presents the equality of these sets as Proposition 1 in [10].

As a bottom line I would like to stress that choosing the title for the EML paper [35] was not an easy choice. To include all relevant information in the title, it would have to be ridiculously long and complicated, e.g.

A single binary operator for computing exp-log functions on the extended real line, almost everywhere, under IEEE 754 semantics, with algebraic adjunctions replaced by Chow's elementary numbers, via complex-valued intermediate values, from the constant 1?

or similar. This would be consistent with the modern struggle to attract the attention of potential readers, who only skim titles, rarely read the abstract, and do not even attempt to read the full text, not to mention the Supplementary Information. However, information about the existence of the EML operator might be important to the large community of machine learning and electronic engineers, bioinformaticians, theoretical computer science experts and educators. A title narrowly aimed at mathematicians would mislead the core audience. On the other hand, a casual title, e.g., Two-button calculator would suggest recreational math, without any applications in sight. For most STEM educated people around the world, elementary functions = scientific calculator operations. Period.

## 2.2. Complex domain and branches

Another common misconception related to the EML operator is the use of complex intermediate values to generate real functions. While this sounds natural for theoretical physicists, e.g. quantum mechanics operates exactly that way (complex wave function, real measurements), more mathematically aligned people were confused: what is the domain of the (1)? If one presumes it is the natural real domain $\mathbb { R } \times \mathbb { R } ^ { + } \to \mathbb { R }$ , the entire reconstruction becomes impossible. As proved by Hardy in [24] one cannot obtain sin x from exp, ln and arithmetic within the real domain, because it oscillates at infinity [23, Chap. III, Sect. 2, Theorem on p. 18]. If not real, then maybe a complex mapping $\mathbb { C } ^ { 2 } \to \mathbb { C }$ of EML arguments is enough for the full reconstruction of the elementary function chain? However, while exp(x) is well-defined on the entire complex plane, ln(y) is infinitely-valued. It is a true function only when defined on its Riemann surface, except at (complex) zero. In practice, eml $( x , y )$ is defined on $\mathbb { C } \times ( \mathbb { C } \backslash \{ 0 \} )$ using the principal branch of the complex Log. Moreover, all functions to be reconstructed by EML are real, $\mathbb { R } ^ { n } \to \mathbb { R }$ Therefore, complex inputs to $\operatorname { e m l } ( x , y )$ appear only at internal expression nodes. Both inputs and outputs of EML-expressions are real, while inputs are additionally restricted to be within the real domain of the reconstructed expression, with the possible exception of domain endpoints [33].

The main surprise of [35] preprint result is that the above do work. In principle, one would expect that for some big EML expression tree internal values escape fundamental strip, and entire reconstruction breaks. Initially, a lot of attempts were made to find such a counterexample. All of them appeared to be bugs in EML compiler, or even Wolfram Mathematica FullSimplify procedure failures claiming false formulas to be true. Formal Lean4 results are available in [33], although some trigonometric statements use input-dependent expressions or real/imaginary-part projections.

Note that question whether similar reconstruction for $\mathbb { C } \to \mathbb { C }$ elementary functions, even reduced to principal branch, is possible, is separate and unanswered question. If it is possible, it likely would require operator much different from (1). Reconstruction of complex-valued multivalued elementary functions in similar manner looks impossible task, as they are defined on distinct Riemann surfaces.

## 2.3. Use of extended real axis and logarithm at zero

The original construction used in [35] uses a Bourbaki-style extended domain, including infinite points. In particular, the reconstruction relies on the formulas

$$
e ^ { - \infty } = 0 , \qquad \ln 0 = - \infty .
$$

This is confusing for many STEM practitioners, for whom the logarithm at zero is undefined, and ±∞ are not valid arguments or values. Moreover, since EML operates in the complex domain, one must be careful whether we are using complex infinity, directed infinity, or real infinities. In practice, the above poses no problem, as we use EML expression within Computer Algebra Systems like Mathematica, which handle the above natively. The same applies to IEEE754 numerical floating-point computations, which define and handle infinities automatically. Therefore, depending on the level of error handling, EML expressions are easily evaluated in many programming languages. But not in all of them. For example, Python's and Maple's standard logarithm routine raises an error at zero. This motivated the creation of clean EML compiler, which never touches the troublesome ln(0), at the expense of much longer expressions. So far this looks doable. In particular, Lean4 proofs must avoid the complex Log(0), because, due to the requirement for Log to be a total function (full C domain, including zero) "junk" value $\mathrm { L o g } ( 0 ) \mathrm { { = } } 0$ is imposed. This breaks reconstructions that use the extended value at zero, and [33] must avoid it.

## 2.4. EML as NAND/NOR equivalent for continuous math

As A. Rieu pointed out [40], the history is somewhat more intricate than the modern identification of the Sheffer stroke with NAND suggests. Peirce discovered the functional completeness of both NAND and NOR around 1880, although his result remained unpublished, and Stamm published both forms in 1911. Sheffer's 1913 paper selected the NOR interpretation (“neither-nor"), whereas Nicod's 1917 treatment adopted the NAND interpretation now conventionally associated with the stroke [48, 42, 34]. Their later significance for digital computation rests especially on Shannon's identification of Boolean algebra with relay and switching-circuit synthesis [41].

In some sense the situation is similar to EML. While (1) was discovered first, at least five equivalents are now known, cf. Table 1. The notion of "sole sufficient operator" in both cases (NAND, EML) in fact refers to multiple cases: NAND/NOR vs EML/EDL/LDE/PLI/PLM. The analogy between digital NAND and continuous EML is quite close. One difference is the requirement to use the constant 1 for EML to work, while NAND does not require this. But in practice, all circuits use external 0/1 inputs anyway

NAND can compute any Boolean function and approximate any other function. The same is true for EML. It can compute any elementary function, which can in turn approximate any functions, using polynomials (series, Chebyshev) rational functions (Pade), Fourier series, integration (Gaussian, double-exponential), ODEs (Runge-Kutta methods) and last but not least neural networks. All aforementioned methods have proved useful in science.

The problem of generating EML values (output) is technical, not fundamental. In the late XIX and early XX century no one had any idea how to implement NAND/NOR in a massive, efficient way. Babbage's Analytical Engine [9] was purely mechanical, Shannon [41] discussed electromechanical switches. Only in the 60s, first in the Apollo Guidance Computer, did we start using integrated circuit NOR and later NAND TTL gates. We hope the engineering and adoption of EML-type operators will be much faster, in either digital or analog form.

## 2.5. Use of the brute-force search

Exhaustive computational searches have a long and successful history in the discovery of mathematical counterexamples. For example, Lander and Parkin used a direct search to disprove Euler's sum-of-powers conjecture by finding $2 7 ^ { 5 } + 8 4 ^ { 5 } + 1 1 0 ^ { 5 } + 1 3 3 ^ { 5 } = 1 4 4 ^ { 5 }$ [30]. Another example is the exhaustive enumeration by Dokovic showing that Williamson matrices of order 35 do not exist, thereby disproving the Williamson conjecture [15]. Computational searches have also been used to formulate new conjectures [8]. Here we used them to select arithmetic operators with given properties. Therefore, enumeration was used recursively. First, we enumerated candidate operators, then all formulas composed of them. A common misconception related to [35] is that this was all that was done. In fact, the exhaustive search was meant to provide candidate operators. EML itself, once marked as first possible successful candidate, was subject to very extensive numerical and symbolic hand-crafted verification.

By design, the fast numerical sieve, based on numerical constant recognition, might return false-positives. This can happen due to numerical roundoff errors or floating-point tautologies. We recall that floating-point numbers, while intended to approximate reals, in fact are dyadic rationals. Their set is huge, but finite, and calculations are discrete, not continuous. Our procedure works as follows. First, we generate some elementary formula which is representing two-input operator, similar to (1). Then we want to know, if some combination of this operators is equivalent to standard mathematical operation, like square root or multiplication. Instead of slow and error prone symbolic simplification, we attempt to establish floating-point identity at some point. This cannot be value like, say $x = - 2 / 3 , y = \pi$ because both of these numbers are already elementary. We need some truly transcendental constant. The best situation would be if we know that it is at least irrational, but for most convenient cases, like Khinchin $K \simeq 2 . 6 8 5 4 5$ Glaisher $A \simeq 1 . 2 8 2 4 3$ , Euler gamma $\gamma \simeq 0 . 5 7 7 2 1 6$ or Catalan $C = 0 . 9 1 5 9 6 6$ constant there are no relevant proofs2. Later we proceed as if these constants were truly outside the exp-log class. Once we compute numerically a high-precision value, e.g. √A ≈ 1.1324429915455447052953341751903 or K × C ≈ 2.4597816377901852095513549038665, we can run constant recognition software [46] on the result. If the fast numerical search provide an equivalent candidate formula using e.g. eml(x, y) and 1 only, which agree to some assumed error of a few ULPs with the expression sought, then additional checks (arbitrary precision, symbolic, multi-point) are executed. If they pass, the formula is marked a valid candidate, and we proceed to the next case [35, 33].

If the entire set of operations which define our "elementary functions" is reconstructed in the above way, the result is subject to detailed, hand-crafted symbolic and numeric verification. For details, see [35].

What might be surprising is that the above procedure was able to discover anything. One might call it luck, however it was the culmination of very long research on brute-force methods in symbolic regression, started as early as 20093. The idea was to use a mix of random/genetic and exhaustive search to sweep unsolved problems at scale. In practice it was limited to occasional solving algebraic equations, evaluating definite integrals, solving ODEs or identifying numerical constants from various sources of experimental mathematics or physics. To our unpleasant surprise, for a decade this programme led only to the re-discovery of already known results in different forms. See for example [27]. This unreasonable effectiveness of human science in solving problems was a surprise. It looked like everything that could be found, was already found. No gaps in knowledge, no missing solutions, no forgotten old problems. In fact, the EML discovery was the first major success of the above philosophy. In some sense, finally it was demonstrated to be right. It is conceivable that throwing substantial computational power and programming effort at some unsolved scientific issues could work in the above way.

Now, in 2026, deep learning neural networks in the form of LLM chatbots (ChatGPT, Grok, Gemini) and programming agents (Codex, Claude Code) are overshadowing any other attempts in machine learning. In the last few weeks, torrents of counterexamples4 to decades-old old conjectures (mensis mirabilis) generated by those AI systems have left the world's top-class mathematicians, including Fields medalists, in shock. The broader debate is reflected in the earlier Leiden Declaration [1]. Some of these conjectures are iconic in modern math, and many geniuses attempted solving them. In retrospective, some of these conjectures (e.g. the Jacobian conjecture, DGG, Maxwell) could have been disproved many years ago by brute force search, but no one tried hard enough. Things are happening too fast now to even think about possible consequences. But it looks like bitter lesson [49| i.e. throwing an insane amount of computing power and data at any optimization problem, is working nearly as good as the most sophisticated dedicated "intelligent" solutions, including human science. This in turn raises the major question of the early XXI century: what is intelligence?

Brute-force exhaustive enumerative search seems at first glance to be the opposite of intelligence. The iconic use case of a student who patiently substitute subsequent integers⁵ into a homework equation in an attempt to solve it is an example of nonsense work. We expect them to use some algorithm instead. But where did the algorithm come from? We can enumerate candidate algorithms as well. Once found, it can be used forever, and possible enhancing future brute-force search for other objects. This looks more like developmental biology than scientific progress. The major question of our times is therefore not what intelligence is, but where genuinely new knowledge comes from. In our opinion, it is always brute-force search in disguise. Justification and explanation for discovery is usually provided post-factum In this light, there is no surprise that modern AI systems are said to be unable to create new ideas beyond mixing and application of what was in the training data. The fact that "trivial" counterexamples to famous conjectures are now found by AI is then evidence of the negligence of the scientific community in the valuation of brute-force methods rather than a symptoms of emerging AGI (Artificial General Intelligence) or ASI (Artificial Super Intelligence).

This leaves the intriguing possibility, that redirecting the world's compute into exhaustive search instead of AI training might be the most efficient use of it. Notable examples of such searches of the "mathematical universe" [52] include Wolfram's Physics Project [55] and Taelin's discrete program search on the HVM [50].

## 2.6. Relation to Turing machine and universal computation

A frequent question is how EML relates to established computational models like the Turing machine. EML real-function formalism is closer in spirit to analog computations [54, Chap. 12, Sec. 4, note (c), "Continuous computation," p. 1128], in which the exponential function is a primitive object. Even integers must be constructed from it. A list of the first 100 integers expressed in pure EML form is given in [38]. A more striking example is the representation of π. Its decimal expansion π = 3.14159 ... has been computed to 314 trillion digits, turning it into a big data object [39]. In contrast, EML gives the following exact inite expression [11]:

```prolog
eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(1,eml(eml(1,eml(1,
eml(eml(1,1),1))),1)),eml(eml(eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,
eml(1,eml(eml(1,eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),eml(1,1))),
1))),1))),1)),eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),
eml(eml(1,eml(eml(1,eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),
eml(eml(1,eml(eml(1,eml(eml(1,eml(eml(1,1),1)),eml(eml(eml(1,eml(eml(1,
eml(1,eml(eml(1,1),1))),1)),eml(1,1)),1))),1)),1)),1)),1)),1)),1)),1),1),
1))),1))),1)),eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),
eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),
1))),1)),eml(1,1))),1))),1)),1)),1)),1).
```

This is confusing for many computer scientists, who reason in the following way. First, we need a computationally intensive binary algorithm for the evaluation of $e ^ { x }$ , then we define (1) to compute $\cdot \cdot \cdot e ^ { x }$ again. This looks circular. It makes no sense until we have a method to evaluate $e ^ { x }$ directly. Physics, however, provides exponentials directly in numerous processes: light attenuation, exponential growth, RC circuit, radioactivity $T _ { 1 / 2 } = \tau \ln 2 ,$ statistic $e ^ { - \frac { E } { k _ { B } T } }$ , entropy $S = k _ { B } \log W$ , cosmological inflation $e ^ { H _ { \infty } t }$ , the Gaussian curve $e ^ { - x ^ { 2 } }$ , quantum phase $e ^ { i \varphi } | \psi \rangle$ etc. Evidently, the existing digital architecture is unsuitable for EML computations. An FPGA implementation may be useful in specific conditions [51], namely limited resources, e.g. on micro satellites, but still must use a standard discrete algorithm to compute $e ^ { x }$ . For efficiency, we need some unconventional, nonvon Neumann architecture. A step in the right direction is provided $\mathrm { e . g . }$ by [29].

## 2.7. Prior knowledge

While the discovery of a simple analytical single binary operator for the computations of elementary function was a true surprise, and the majority⁶ before April 2026 would convincingly state that this was impossible, some hints of its existence were hidden in plain sight long before that.

Mathematicians attempted to answer related questions. In rational function theory it was probably known [22] that rational functions can be generated over the field of rational numbers Q using a single operator

$$
x \# y \equiv { \frac { 1 } { x - y } }\tag{4}
$$

plus an unclear number of terminal constants. We found that this is indeed possible using both 0 and 1 as distinguished constants7. In this sense, for rational functions with integer coefficients, e.g., $1 1 ( x ^ { 2 } + x + 3 ) / ( x ^ { 4 } - 4 )$ , the "hash operator" (4) is indeed an analogue of the EML operator. However, it is not clear how to extend (4) to exp-log functions in any other way than by extending the hash operator with exp(x) and ln(x) themselves. But if we allow for this, then plain subtraction is already good enough, cf. Table 1 or Calc 2 from Table 2 in [35]. Moreover, no distinguished terminal constant is required. The statement that the "hash" operator is equivalent to EML is simply false.

Another piece of knowledge missed by [35] is Hua's identity [28], valid for any division ring, hence also for real/complex numbers. For real/complex numbers, Hua's identity allows one to express multiplication using addition/subtraction and inverse

$$
a b a = a - \left[ a ^ { - 1 } + \left( b ^ { - 1 } - a \right) ^ { - 1 } \right] ^ { - 1 } .\tag{5}
$$

The Kolmogorov-style RPN complexity of the above formula is too large for exhaustive search, unless we allow for the use of the reduced mass $u | | v =$ $( u ^ { - 1 } + v ^ { - 1 } ) ^ { \cdot }$ -1 (parallel sum) as a primitive arithmetic operation. Then the reciprocal can be computed using $a ^ { - 1 } = 1 - [ 1 | | ( a - 1 ) ]$ , identity (5) gives multiplication, and the EML reduction of the operator count using the constant 1 sounds less mysterious.

In the abstract theory of binary operators on arbitrary sets, a construction reducing any number of binary operators to a single one is known [20]. Here we present an example construction. The reader is encouraged to compare it to the elegance of the EML operator. In the following, we reduce the four basic operations $( + , - , \times , / )$ to a single "star" operator using four distinguished integer constants: $- 1 , - 2 , - 3 , - 4$ . First, we create four "compactified" copies of the real line

$$
q _ { k } ( x ) = 1 0 + k + { \frac { 1 } { 2 } } + { \frac { \arctan x } { \pi } } .
$$

The inverse operation is, of course

$$
Q _ { k } ( z ) = \tan { [ \pi \left( z - 1 0 - k - 1 / 2 \right) ] } .
$$

Now we define a single operator as

$$
x * y = \left\{ \begin{array} { l l } { q _ { 0 } ( y ) } & { \mathrm { f o r ~ } x = - 1 , } \\ { q _ { 1 } ( y ) } & { \mathrm { f o r ~ } x = - 2 , } \\ { q _ { 2 } ( y ) } & { \mathrm { f o r ~ } x = - 3 , } \\ { q _ { 3 } ( y ) } & { \mathrm { f o r ~ } x = - 4 , } \\ { Q _ { 0 } ( x ) + y } & { \mathrm { f o r ~ } 1 0 < x < 1 1 , } \\ { Q _ { 1 } ( x ) - y } & { \mathrm { f o r ~ } 1 1 < x < 1 2 , } \\ { Q _ { 2 } ( x ) \times y } & { \mathrm { f o r ~ } 1 2 < x < 1 3 , } \\ { Q _ { 3 } ( x ) / y } & { \mathrm { f o r ~ } 1 3 < x < 1 4 . } \end{array} \right.\tag{6}
$$

It is a simple exercise to verify that we have

$$
\begin{array} { r } { x + y = [ ( - 1 ) * x ] * y , } \\ { x - y = [ ( - 2 ) * x ] * y , } \\ { x \times y = [ ( - 3 ) * x ] * y , } \\ { \frac { x } { y } = [ ( - 4 ) * x ] * y . } \end{array}
$$

Because $[ ( - 1 ) * ( - 1 ) ] * ( - 1 ) = - 2 { \mathrm { ~ e t c } } .$ , one can reduce the number of terminal constants to one. Moreover, we have verified in Mathematica that another implementation of the [20] idea leads to a single "diamond" operator working without any constants, see Appendix A.

While the procedure (6) works for binary operators only, since arbitrary arity reduces to binary by a classical theorem of Sierpiński [43], one can extend it to unary exp-log using dummy operators, $\mathrm { e . g . , p w r } ( x , y ) = \exp ( x )$ and $\operatorname { l g } ( x , y ) = \ln x$ . This affirmatively answers the question raised by [35] in the set-theoretic reading. But not for analytic operators [47].

In this sense, [20] anticipated the existence of a single operator. But a Goldstern-type operator is essentially a set-theoretic multiplexer using a lookup table (see Appendix A). It is more similar to the Kolmogorov-Arnold bypass of Hilbert's 13th problem, allowing for the reduction of any continuous function of $n$ variables to 2n + 1 univariate ones using an $f -$ dependent "lookup function" and addition, in its modern form [45]

$$
f ( x _ { 1 } , \dots , x _ { n } ) = \sum _ { q = 1 } ^ { 2 n + 1 } g \left( \sum _ { p = 1 } ^ { n } \lambda _ { p } \phi _ { q } ( x _ { p } ) \right) .
$$

Neither is useful in practice due to pathological mathematical properties. Function $g ( x )$ is f-dependent, only continuous, and highly irregular even for analytic $f ,$ and the inner functions cannot be chosen smooth. This is in contrast to EML-type operators, which are simple analytical formulas.

## 3. Diversity of operators

## 3.1. EML variants

Since the original discovery of (1), it was quickly realized that at least two similar close cousins exist. The first is EDL (Exp-Divide-Log)

$$
\operatorname { e d l } ( x , y ) = { \frac { e ^ { x } } { \ln y } } ,\tag{7}
$$

paired with the Euler number e. Another one is -EML

$$
- \operatorname { e m l } ( y , x ) = \ln { ( x ) } - \exp ( y ) ,\tag{8}
$$

paired with $- \infty$ . Once a faster sieve implementation became available, it become clear that the above three are not, like initially believed, related by some Möbius transform. Two new variants of the EDL, LDE (Log-Divide-Exp) were found

$$
\operatorname { l d e } ( x , y ) = { \frac { \ln x } { e ^ { y } } } ,\tag{9}
$$

one working with 0, other with 1.

Other variants are, PLI (Power-Log-Inverse)

$$
\mathrm { p l i } ( x , y ) = ( \ln x ) ^ { 1 / y } ,\tag{10}
$$

and PLM (Power-Log-Minus)

$$
\operatorname { p l m } ( x , y ) = ( \ln x ) ^ { - y } ,\tag{11}
$$

both with the constant 1.

Related constructions of arithmetic operations through bijections occur in non-Newtonian calculus [14]. In [47], Stachowiak proposed a general scheme for generating EML-type operators. Here we repeat his definitions

Table 1. Known EML-type operators in Stachowiak's form.
<table><tr><td>Name</td><td> $M ( x , y )$ </td><td>ê  $c = f ( \hat { e } )$ </td><td></td><td>f</td><td> $f ^ { - 1 }$ </td></tr><tr><td>EML</td><td> $x - y$ </td><td>0</td><td>1</td><td>exp x</td><td>ln x</td></tr><tr><td>-EML</td><td> $x - y$ </td><td>0</td><td>-∞</td><td>ln x</td><td>exp x</td></tr><tr><td>EDL</td><td>x yal</td><td>1</td><td>e</td><td>exp x</td><td>ln x</td></tr><tr><td>LDE</td><td></td><td>1</td><td>0</td><td>ln x</td><td>exp x</td></tr><tr><td>PLI</td><td> $x ^ { ( 1 / \ln y ) }$ </td><td>e</td><td>1</td><td>ln x</td><td>exp x</td></tr><tr><td>PLM</td><td> $x ^ { ( 1 / \ln y ) }$ </td><td>e</td><td>1</td><td>1 ln x</td><td> $e ^ { 1 / x }$ </td></tr></table>

$$
S ( x , y ) = M \left( f ( x ) , f ^ { - 1 } ( y ) \right) ,\tag{12}
$$

$$
\exists _ { \hat { e } } M ( x , { \hat { e } } ) = x ,\tag{13a}
$$

$$
M ( x , x ) = { \hat { e } } ,\tag{13b}
$$

$$
M ( x , M ( y , z ) ) = M ( z , M ( y , x ) ) .\tag{13c}
$$

where $S ( x , y )$ is an EML-type binary operator, f(x) − generating function, $M ( x , y )$ - non-commutative operator, ê – neutral element, $c = f ( { \hat { e } } ) - \mathrm { d i s } .$ tinguished constant. Noteworthy, all known operators can be cast in Stachowiak's form, cf. Table 1. It is however an open problem what kind of pair $M , f ,$ which defines an EML-type operator, is allowed. Stachowiak discussed the example of $f ( x ) = \cos x$ , but was unable to complete the entire elementary function chain. Our brute-force search depth was too shallow to improve significantly on that. However, using prosthaphaeresis, one can compute multiplication using two constants, $c = 1$ and $a = \operatorname { a r c c o s } ( 1 / 4 )$ with the same $S = \cos x - \operatorname { a r c c o s } y$ . Reciprocal/division is still unreachable, though.

The operator $M ( x , y )$ is in all cases from Table 1 the inverse of a Bennett's symmetric (commutative) hyperoperation [6]

$$
F _ { n } ( a , b ) = \exp ^ { n } ( \ln ^ { n } a + \ln ^ { n } b )
$$

of increasing order:

$n = 0$ addition/subtraction $F _ { 0 } ( x , y ) = x + y  x - y$ (1st order hyperoperation, i.e. repeated zeration/successor)

$n = 1$ multiplication/division $\begin{array} { r } { F _ { 1 } ( x , y ) = x \times y \to \frac { x } { y } } \end{array}$ (2nd order hyperoperation, i.e. repeated addition)

n = 2 symmetrized power $F _ { 2 } ( x , y ) = x ^ { \ln y }  x ^ { 1 / \ln y }$ (commutative analog of the 3rd order hyperoperation, i.e. repeated multiplication)

We must conclude that while the framework proposed by [47] gives some order and insight into the structure of EML-type operators, we still lack a general understanding, and exhaustive or fine-tuned search is still the only viable method to find them.

## 3.2. Ternary variants

The requirement for a distinguished constant at the inputs of EML and its variants is troublesome for the implementation of the master formula (3) and its use for machine learning or symbolic regression. However, the use of this form of the "switch" seems unavoidable [22, 20] (see also Subsect. 2.7). A workaround is to use a ternary operator instead. So far, two [35] were found:

$$
T _ { 1 } ( x , y , z ) = { \frac { e ^ { x } } { e ^ { y } } } \times { \frac { \ln x } { \ln z } } ,\tag{14a}
$$

$$
T _ { 2 } ( x , y , z ) = { \frac { e ^ { x } } { e ^ { y } } } \times { \frac { \ln z } { \ln x } } .\tag{14b}
$$

Noteworthy, $T _ { i } ( x , x , x ) = 1$ , and no distinguished constant is required to generate all expressions. Unlike the interpolating ternary of [47], (14) involve no case distinction.

## 4. Möbius layer neural networks

All classic neural networks use a single univariate real activation function. The essential question is whether we can find a similar function, which not only approximates data, but is able to generate elementary expressions in exact form as well. Such a construction could connect trainable networks with symbolic regression. Therefore, the answer to the above question is very important for progress in machine learning.

The EML operator (1) might be viewed as some variant of 2-input complex-valued activation function, in analogy to sigmoid or ReLU in other variants of computational networks. For EML, the network graph is a parameterized binary tree. As demonstrated in [35| training such a tree is difficult, and so far has been demonstrated in a limited capacity. Its unique property, however, is the ability to express all elementary functions, which have proved useful in STEM and science over the last centuries. On the opposite side we have machine learning, in the form of deep neural networks. While networks with sigmoid or ReLU activations cannot express functions like sin x exactly in the full real domain, they have proven (both in theory and practice) the ability to approximate any of them. But the decisive property favoring the latter in applications is a working optimization procedure (e.g. Adam, stochastic gradient method) which can be extended to industrial-scale networks with trillions of parameters as of 2026, with no "wall" in sight preventing further increase.

Therefore the natural follow-up question, urgent after the discovery of the EML, is whether some of its variants will exhibit the scaling properties of modern deep neural networks, keeping "symbolic" capabilities of the EMLtrees. Due to the aforementioned theorem by Hardy (see Subsect. 2.2), a neural network with a real exp-log activation is unable to compute e.g. sin x. What is the minimal modification then, which would enable exact elementary functions in deep learning, keeping a convenient optimization landscape? Below we provide some hints as to which direction the search should proceed.

A typical neural network rewritten in EML-style language is a "calculator" with basic arithmetic operators $( + , - , \times )$ ("matrix multiplication") plus a single univariate non-linear activation function, e.g. the logistic sigmoid

$$
S ( x ) = \frac { 1 } { 1 + e ^ { - x } } ,\tag{15}
$$

known in physics as the Fermi-Dirac distribution. Noteworthy, the set of allowed operations does not include division. We now attempt to modify the standard neural network, shown in Fig. 1, top panel, to enable exact elementary functions.

To achieve this goal, we employ three modifications: (i) replace the activation function, (ii) use complex numbers internally, (iii) add division to the allowed arithmetic operations. The new (complex valued) activation function is

$$
F ( z ) = e ^ { z } + \ln z \equiv { \mathrm { e m l } } \left( z , z ^ { - 1 } \right) .\tag{16}
$$

The crucial identity

$$
e ^ { z } = \frac { F ( 3 z ) - F ( z ) - \ln 3 } { F ( 2 z ) - F ( z ) - \ln 2 } - 1\tag{17}
$$

allows us to recover the exponential function by standard algebraic simplification. The logarithm is then simply ln $z = F ( z ) - e ^ { z }$ Once we have both exp and ln, together with subtraction, we have (1), and from [35] we know that this is enough to evaluate all elementary functions. The constants used in (17), namely 1,2, 3, ln 2, ln 3 are needed as written but are likely redundant. The complexity of (17) in RPN form is K=23, far beyond direct enumeration reach, and it is not known whether it is the simplest possible construction of this kind. Nevertheless, compositions of rational operations and F can in principle generate exact analytical formulas; Fig. 1, bottom panel, shows the basic layer. The matrix multiplication layer, including the output map, must be replaced by a rational function layer to allow the division required by (17), and complex numbers are required to use the EML-style reduction. The activation function is now (16). The question whether such a network can indeed be optimized efficiently is beyond scope of this note.

(a) Standard neural network  
![](images/bd51e30a2629a159671c33bea7923ac49c4679256522b907c678fdbbacf18cbc.jpg)

(b) Möbius network  
![](images/9e4e53d6a2d85eb46cf9875d423dc644f066560a7a49da73f2bd452097061c4e.jpg)  
Fig. 1. Comparison of the standard neural architecture (top) and the proposed generalized "Möbius" network (bottom). Both include constant inputs supplying the biases. In the proposed network, the affine input and output maps become ratios of affine forms, and S is replaced by F. Each panel shows one layer with four activation units.

## 5. Conclusions

The article discusses common misconceptions, previous knowledge, followup studies and emerging directions which came after the discovery of the EML operator [35]. What was missed by readers is that the EML itself, given by formula (1) is probably only the first known member of a large family of operators and functions, which could open new ways of evaluating elementary functions, enhancing the abilities of machine learning. So far, no systematic theory has been created or exhaustive search done to reveal all its variants. One of the undiscovered variants could possibly enable exact analytical functions in an architecture nearly identical to existing neural network, keeping standard optimization procedures. In the optimistic variant, the only modification would be complex weights in place of real ones, and a new non-linear activation possibly resembling (16) in an algebraic way. If we are lucky, the new activation will work in the full real domain, keeping the asymptotics of the ramp function (ReLU), i.e. zero for $x  - \infty$ and x for $x  + \infty$ All EML-type nonlinearity would remain near zero and in imaginary direction. A natural candidate with the above properties is some combination of elementary hyperbolic functions. The use of complex numbers might not be necessary, as they can be traded for their matrix representation $\bar { i }  ( \begin{array} { c } { { 0 - 1 } } \\ { { 1 } } \end{array} )$ used in e.g. recent articles on quantum mechanics [25] without complex numbers [4]. However, if rational functions are indeed required to achieve the stated goal, it would be of no surprise to practitioners, as e.g. Pade approximation has been known to be a superior method for a long time.

## Acknowledgments

I would like to thank Henrik Klagges for the invitation to the TNG Big Techday conference in Munich, and the Faculty of Mathematics and Computer Science of the Jagiellonian University for support in the form of a Maple license. Computational resources were partially provided by Google Cloud Research Credits and the Polish National Science Centre MAESTRO Grant No. 2017/26/A/ST2/00530.

## REFERENCES

[1] Jarod Alper, Michael Barany, Alain Chavarri Villarello, Sander Dahmen, Walter Dean, Karthik Ganapathy, Michael Harris, David Holmes, Mateja Jamnik, Steven Kelk, Bryna Kra, Ursula Martin, Bartosz Naskrecki, Rodrigo Ochigame, Jim Portegies, and Johannes Schmitt. The Leiden Declaration on Artificial Intelligence and Mathematics, June 2026. Endorsed by the In-

ternational Mathematical Union; featured endorsers include P. Scholze and T. Tao. Over 3,700 signatories as of 2 September 2026.

[2] Philip Arathoon, Gavin Ball, and Matthew D. Kvalheim. The Maxwell conjecture is false. arXiv:2607.27197, 2026.

[3] Sota Asanuma. Eml-cd: Causal mechanism recovery via eml symbolic trees in structure learning, 2026.

[4] Pedro Barrios Hita, Anton Trushechkin, Hermann Kampermann, Michael Epping, and Dagmar Bruß. Quantum mechanics based on real numbers: A consistent description. Physical Review Letters, 136(24):240202, 2026.

[5] Reda Belaiche. Additive atomic forests for symbolic function and antiderivative discovery, 2026.

[6] Albert A. Bennett. Note on an operation of the third grade. Annals of Mathematics, 17(2):74–75, 1915.

[7] Frits Beukers. Hypergeometric functions, how special are they? Notices of the American Mathematical Society, 61(1):48–56, January 2014.

[8] Jonathan M. Borwein and David H. Bailey. Mathematics by Experiment: Plausible Reasoning in the 21st Century. A K Peters, Wellesley, MA, 2 edition, 2008.

[9] Allan G. Bromley. Charles babbage's analytical engine, 1838. Annals of the History of Computing, 4(3):196–217, 1982.

[10] Mark Carney. Inexpressibility in exp-minus-log, 2026.

[11] Michael Carvalho. Creating pi from a single binary operator and the constant 1. https://www.mapleprimes.com/posts/235284-Creating-Pi-From-A-Single-Binary-Operator, July 2026. Posted July 22, 2026; accessed August 28, 2026.

[12] Timothy Y. Chow. What is a closed-form number? The American Mathematical Monthly, 106(5):440–448, 1999.

[13] Miles Cranmer, MilesCranmerBot, Dhananjay Ashok, tttc3, William Booth-Clibborn, Ward K Harold, wkharold, Johann Brehmer, William Booth-Clibborn, Mark Kittisopikul, Dilum Aluthge, Saurav Maheshkar, Shah Mahdi Hasan, Anubhav Kamal, Arthur Grundner, BrotherHa, Christos Pliakos, Daniel Eduardo Conde Villatoro, DeepSource Bot, Hao Guo, Ho Fung Tsoi, Hongyu Wang, Ilya Orson, Digvijay (Jay) Wadekar, Leonardo Voltolini, and LionessOfCintra. astroautomata/pysr: v2.0.0-beta.2, August 2026.

[14] Marek Czachor. Waves along fractal coastlines: From fractal arithmetic to wave equations. Acta Physica Polonica B, 50(4):813–831, 2019.

[15] Dragomir Z. Dokovic. Williamson matrices of order 4n for n = 33, 35, 39. Discrete Mathematics, 115(1–3):267–271, 1993.

[16] Independent implementations of the EML operator. OxiEML (Rust): https://github.com/cool-japan/oxieml;eml(Python):https: //github.com/Inknyto/eml; monogate (JavaScript): https://github.com/ agent-maestro/monogate; emlvm: https://github.com/nullwiz/emlvm. Accessed 29 August 2026.

[17] Amir Erez. Non-Monotone Response Modules and Cascades from the EML Operator for Reduced Models of Biological Dynamics. arXiv e-prints, page arXiv:2605.02972, May 2026.

[18] Shuhong Gao. Counterexamples to the Jacobian conjecture in dimensions greater than two. arXiv:2608.00222, 2026.

[19] Joe Germany, Elie Abdo, and Joseph Bakarji. Eml trees are universal approximators. arXiv:2606.23179, 2026.

[20] Martin Goldstern. A single binary function is enough. In Johannes Czermak, Gerhard Dorfer, Günther Eigenthaler, Winfried Bernward Müller, and Johannes Schoißengeier, editors, Contributions to General Algebra 20: Proceedings of the Salzburg Workshop 2011 on General Algebra, pages 35-37, Klagenfurt, Austria, 2012. Verlag Johannes Heyn.

[21] Chakshu Gupta and Theodore J. LaGrow. Architecture-Induced Recoverability Bias in Differentiable Symbolic Regression. arXiv e-prints, page arXiv:2604.23256, April 2026.

[22] Joel David Hamkins. Can we unify addition and multiplication into one binary operation? to what extent can we find universal binary operations? Math-Overflow, March 2011. Question 57465; accessed 28 August 2026.

[23] G. H. Hardy. Orders of Infinity: The 'Infinitärcalcül' of Paul du Bois-Reymond. Number 12 in Cambridge Tracts in Mathematics and Mathematical Physics. Cambridge University Press, Cambridge, 1910.

[24] G. H. Hardy. Properties of logarithmico-exponential functions. Proceedings of the London Mathematical Society, 10(1):54–90, 1912.

[25] Timothée Hoffreumon and Mischa P. Woods. Quantum theory does not need complex numbers. axXiv, 2025.

[26] Russell W. Howell and John H. Mathews. Complex analysis. https://complexanalysis.org/web/root-1-2.html, 2025. Online textbook, last updated August 22, 2025; accessed August 28, 2026.

[27] Andrzej Odrzywolek (https://math.stackexchange.com/users/307694/andrzej odrzywolek). Integral $\begin{array} { r } { \int _ { - 1 } ^ { 1 } \frac { 1 } { x } \sqrt { \frac { 1 + x } { 1 - x } } \ln \left( \frac { 2 x ^ { 2 } + 2 x + 1 } { 2 x ^ { 2 } - 2 x + 1 } \right) } \end{array}$ dx. Mathematics Stack Exchange. URL:https://math.stackexchange.com/q/1625336 (version: 2016-01- 24).

[28] Loo-Keng Hua. On the automorphisms of a sfield. Proceedings of the National Academy of Sciences of the United States of America, 35(7):386–389, 1949.

[29] Andraž Jelinčič, Owen Lockwood, Akhil Garlapati, Peter Schillinger, Isaac L. Chuang, Guillaume Verdon, and Trevor McCourt. An efficient probabilistic hardware architecture for diffusion-like models. npj Unconventional Computing, 3:30, 2026.

[30] L. J. Lander and T. R. Parkin. Counterexample to euler's conjecture on sums of like powers. Bulletin of the American Mathematical Society, 72(6):1079, 1966.

[31] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, and Max Tegmark. KAN: Kolmogorov-

Arnold networks. In International Conference on Learning Representations (ICLR), 2025.

[32] Vladimir Mityushev and Sergei Rogosin. On relations between elliptic and elementary functions. Acta Physica Polonica B Proceedings Supplement 13(4):745–751, 2020.

[33] B. Naskrecki. EML formalization. https://github.com/nasqret/emlformalization, 2026.

[34] Jean G. P. Nicod. A reduction in the number of the primitive propositions of logic. Proceedings of the Cambridge Philosophical Society, 19:32–41, 1917.

[35] A. Odrzywołek. All Elementary Functions from a Single Binary Operator. JACM, submitted, 2026.

[36] OpenAI. An OpenAI model has disproved a central conjectureindiscretegeometry. https://openai.com/index/ model-disproves-discrete-geometry-conjecture/, May 2026. 20 May 2026.

[37] OpenAI. Ten advances in mathematics and theoretical computer science. https://openai.com/index/ten-advances-in-mathematics/, August 2026. 1 August 2026.

[38] Orson Peters. Generating integers using EML. Code Golf Stack Exchange, April 2026. Code-golf challenge posted April 13, 2026 under the username or1p; accessed August 28, 2026.

[39] Alessandro Razeto and Nicola Rossi. Can π generate itself? a monte carlo analysis of 314 trillion digits, 2026.

[40] André Rieu. [Thread on the historical development of NOR and NAND as universal gates]. X, June 2026. Posted under the username @superrieu; in German.

[41] Claude E. Shannon. A symbolic analysis of relay and switching circuits. Transactions of the American Institute of Electrical Engineers, 57(12):713 723, 1938.

[42] Henry Maurice Sheffer. A set of five independent postulates for boolean algebras, with application to logical constants. Transactions of the American Mathematical Society, 14(4):481–488, 1913.

[43] Wacław Sierpiński. Sur les fonctions de plusieurs variables. Fundamenta Mathematicae, 33:169–173, 1945. In French. Zbl 0060.13111.

[44] Robert Smith. Not all elementary functions can be expressed with exp-minuslog. https://www.stylewarning.com/posts/not-all-elementary/, 2026.

[45] David A. Sprecher. On the structure of continuous functions of several variables. Transactions of the American Mathematical Society, 115:340–355, 1965.

[46] Klaudiusz Sroka and Andrzej Odrzywołek. Constant recognition. https://constantrecognizer.com/, 2026.

[47] Tomasz Stachowiak. Algebraic structure behind Odrzywołek's EML operator. arXiv e-prints, page arXiv:2604.23893, April 2026.

[48] Edward Stamm. Beitrag zur algebra der logik. Monatshefte für Mathematik und Physik, 22:137–149, 1911.

[49] Richard S. Sutton. The bitter lesson. https://www.incompleteideas.net/IncIdeas/BitterLesson.html, March 2019. Essay published March 13, 2019; accessed August 28, 2026.

[50] Victor Taelin and Higher Order Company. HVM: a massively parallel, optimal functional runtime based on interaction combinators. GitHub repository, 2026. Accessed 2 September 2026.

[51] Adam Taylor. Microzed chronicles: EML in FPGA.

[52] Max Tegmark. The mathematical universe. Foundations of Physics, 38(2):101 150, 2008.

[53] Eric W. Weisstein. EML operator. From MathWorld-A Wolfram Web Resource, https://mathworld.wolfram.com/EMLOperator.html. Accessed August 2026.

[54] Stephen Wolfram. A New Kind of Science. Wolfram Media, Champaign, IL, 2002.

[55] Stephen Wolfram. A class of models with the potential to represent fundamental physics. Complex Systems, 29(2):107–536, 2020.

## Appendix A

Wolfram Mathematica implementation of a Goldstern-type single operator

Let $\beta \colon \mathbb { Z } \to \mathbb { N }$ where $\mathbb { N } = \{ 0 , 1 , 2 , \ldots \}$ , be the standard pairing between integers and naturals

$$
\beta ( m ) = \left\{ \begin{array} { l l } { 2 m } & { m \geq 0 , } \\ { - 2 m - 1 } & { m < 0 , } \end{array} \right. \quad \beta ^ { - 1 } ( p ) = \left\{ \begin{array} { l l } { p / 2 } & { p \mathrm { ~ e v e n } , } \\ { - ( p + 1 ) / 2 } & { p \mathrm { ~ o d d } . } \end{array} \right.\tag{A.1a}
$$

Every positive integer factors uniquely as $2 ^ { l } ( 2 a + 1 )$ . Applying this to $\beta ( \lfloor x \rfloor ) + 1$ assigns to every real $x \mathrm { ~ a ~ }$ "floor numb $\mathrm { e r } ^ { \prime \prime } \ l ( x ) \in \mathbb { N }$ and $\mathrm { a \ ^ { 6 } p a y { - } }$ load" $u ( x ) \in \mathbb { R } \colon$

$$
\beta ( \lfloor x \rfloor ) + 1 = 2 ^ { l ( x ) } ( 2 a + 1 ) , \qquad u ( x ) = \beta ^ { - 1 } ( a ) + x - \lfloor x \rfloor .\tag{A.1b}
$$

The map $x \mapsto { \big ( } u ( x ) , l ( x ) { \big ) }$ is an explicit bijection $\mathbb { R }  \mathbb { R } \times \mathbb { N }$ , with inverse

$$
\rho ( v , l ) = \beta ^ { - 1 } \Big [ 2 ^ { l } \big ( 2 \beta ( \lfloor v \rfloor ) + 1 \big ) - 1 \Big ] + v - \lfloor v \rfloor ,\tag{A.1c}
$$

and $S ( x ) = \rho { \big ( } u ( x ) , l ( x ) + 1 { \big ) }$ moves x one floor up, keeping the payload.

The operator is

$$
\begin{array} { r } { x \diamond y = \left\{ \begin{array} { l l } { S ( x ) } & { \mathrm { f o r ~ } y = x , } \\ { \rho ( x , 0 ) } & { \mathrm { f o r ~ } y = S ( x ) , } \\ { u ( x ) + u ( y ) } & { \mathrm { f o r ~ } l ( x ) = 1 , l ( y ) = 0 , } \\ { u ( x ) - u ( y ) } & { \mathrm { f o r ~ } l ( x ) = 2 , l ( y ) = 0 , } \\ { u ( x ) \times u ( y ) } & { \mathrm { f o r ~ } l ( x ) = 3 , l ( y ) = 0 , } \\ { u ( x ) / u ( y ) } & { \mathrm { f o r ~ } l ( x ) = 4 , l ( y ) = 0 , u ( y ) \neq 0 , } \\ { x } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{A.1d}
$$

Writing $\sigma ( t ) = t \diamond t$ (one floor up) and $\varepsilon ( x ) = x \diamond ( x \diamond x )$ (encoding at floor zero) we obtain

$$
\begin{array} { r l r } & { } & { x + y = \sigma \big ( \varepsilon ( x ) \big ) \diamond \varepsilon ( y ) , } \\ & { } & { x - y = \sigma ^ { 2 } \big ( \varepsilon ( x ) \big ) \diamond \varepsilon ( y ) , } \\ & { } & { x \times y = \sigma ^ { 3 } \big ( \varepsilon ( x ) \big ) \diamond \varepsilon ( y ) , } \\ & { } & { \frac { x } { y } = \sigma ^ { 4 } \big ( \varepsilon ( x ) \big ) \diamond \varepsilon ( y ) , } \end{array}
$$

valid for all real $x , y$ (with $y \ne 0$ in the last line). No constant appears anywhere; fully expanded, e.g.,

$$
x + y = \left\{ \left[ x \diamond \left( x \diamond x \right) \right] \diamond \left[ x \diamond \left( x \diamond x \right) \right] \right\} \diamondsuit \left[ y \diamond \left( y \diamond y \right) \right] .
$$

Constants can also be generated from pure- terms, e.g.

$$
0 = \left\{ { \Big ( } { \big [ } x \diamond ( x \diamond x { \big ) } { \big ] } \diamond { \big [ } x \diamond ( x \diamond x { \big ) } { \big ] } { \Big ) } \diamond { \Big ( } { \big [ } x \diamond ( x \diamond x { \big ) } { \big ] } \diamond { \big [ } x \diamond ( x \diamond x { \big ) } { \big ] } { \Big ) } \right\} \diamond { \big [ } x \diamond ( x \diamond x { \big ) } { \big ] } .
$$

Below you can find a working implementation of the above procedure in Wolfram Mathematica.

```ocaml
(* Pairing integers to naturals *)
beta[m_Integer] := If[m >= 0, 2 m, -2 m - 1];
ibeta[p_Integer] := If[EvenQ[p], p/2, -(p + 1)/2];
(* Floor number and payload: x <-> (val, level) is a bijection
R < - > R x N *)
level[x_] := IntegerExponent[beta[Floor[x]] + 1, 2];
val[x_] := Module[{p = beta[Floor[x]] + 1, 1},
1 = IntegerExponent[p, 2];
ibeta[(p/2^1 - 1)/2] + (x - Floor[x])];
rho[v_, 1_Integer] := ibeta[2^1 (2 beta[Floor[v]] + 1) - 1] + (
v - Floor[v]);
```

24 EML-DIVERSITY PRINTED ON SEPTEMBER 11, 2026   
S[x\_] := rho[val[x], level[x] + 1]; (\* same payload, one   
floor up \*)   
(\* THE single operator \*)   
op[x\_, y\_] := Which[   
y == x, S[x], (\*climb \*)   
y == S[x], rho[x, 0], (\* encode \*)   
1 <= level[x] <= 4 && level[y] == 0,   
Switch[level[x],   
1, val[x] + val[y],   
2, val[x] - val[y],   
3, val[x] val[y],   
4, If[val[y] == 0, x, val[x]/val[y]]],   
True, x];   
(\* Pure-op terms for the four operations \*)   
s[x\_] := op[x, x];   
enc[x\_] := op[x, s[x]];   
plus[x\_, y\_]:= op[s[enc[x]], enc[y]];   
minus[x\_, y\_] := op[s[s[enc[x]]], enc[y]];   
times[x\_, y\_] := op[s[s[s[enc[x]]]], enc[y]];   
divide[x\_, y\_] := op[s[s[s[s[enc[x]]]]], enc[y]];   
(\* Examples \*)   
plus=op[op[op[x, op[x, x]], op[x, op[x, x]]], op[y, op[y, y]]]   
(\* x+y \*)   
zero=op[op[op[op[x, op[x, x]], op[x, op[x, x]]], op[op[x, op[x,   
x]], op[x, op[x, x]]]],   
op[x, op[x, x]]] (\* zero \*)   
(\* Some values are required to resolve conditionals \*)   
plus /. {x->EulerGamma, y->Glaisher}   
zero /. x->Khinchin