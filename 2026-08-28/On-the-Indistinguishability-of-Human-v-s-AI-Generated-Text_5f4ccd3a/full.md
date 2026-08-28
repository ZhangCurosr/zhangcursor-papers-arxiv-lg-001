# On the Indistinguishability of Human v/s AI Generated Text

Jaee Ponde Aritra Das

Mihir More Debayan Gupta

Truth Audit Labs {jaee, aritra, mihir, debayan}@truthauditlabs.ai

## Abstract

The rapid improvement of LLMs has made distinguishing AI-generated text from human writing a pressing problem. This challenge is further amplified by paraphrasing tools designed to make machine-generated text appear more "human". We study how access to human writing samples can be used to strategically paraphrase machine-generated responses toward the human distribution. Under a multi-sample setting with human and machine responses to the same prompts, we show that repeated paraphrasing moves the machine distribution toward the empirical human distribution under simple mixing and stability conditions. Our results derive an explicit convergence rate, extend the analysis to a finite-sample setting, and characterize how the required number of human samples and paraphrasing rounds scale with the desired error.

## 1 Introduction

Large language models can now generate fluent, convincing text across a wide range of academic tasks. Their growing use in academia has raised serious concerns about plagiarism, undisclosed AI assistance, and the integrity of written submissions [1]. Recent studies show that AI-generated academic writing can achieve quality comparable to human-written work, making authorship increasingly difficult to infer from the text alone [2]. In response, universities and other institutions have increasingly turned to AI-text detection tools as part of their academic-integrity processes [3]. However, these detectors are fragile to rewriting. Prior work shows that paraphrasing or repeated rewriting can substantially reduce the detectability of machine-generated text while preserving its content and quality [4, 3]. Understanding how repeated paraphrasing changes machine-generated text, and in particular whether it systematically moves such text toward the human distribution, is therefore a crucial problem for reliable AI-text attribution.

A useful way to understand the limits of AI-text detection is through the distance between the human and machine distributions. Sadasivan et al. [4] show that detecting a single machine-generated response can become unreliable once machine text is sufficiently similar to human writing. In particular, they argue that no detector can remain reliable when the human and machine text distributions become too close, and they further show that repeated paraphrasing can substantially weaken existing detectors. Chakraborty et al. [5] give a complementary result: even if individual responses are difficult to distinguish, access to many independent responses makes detection possible, because small differences between the human and machine distributions accumulate across samples.

An interesting question is the adversarial version of the multi-sample setting. In practice, an adversary may not only have access to machine-generated text, but also to examples of how a particular person writes. These human examples can then be used to guide repeated paraphrasing of the machine text. For instance, given a student’s past essays and machine-generated essays on the same topics, can the machine essays be rewritten to increasingly resemble the student’s writing while preserving their meaning and quality?

We study this dynamic of human-guided paraphrasing and formalize when access to human examples can progressively move machine-generated text toward the human distribution, how quickly this movement occurs, and how much human data is required. Our main contributions are:

• We show that repeated, semantics-preserving perturbations can move the machine-generated distribution toward the empirical human distribution, with an explicit convergence rate (Theorem 1).

• We extend this guarantee to finite samples, bounding the distance between the observed perturbed-machine distribution and the true human distribution after a finite semantic representation (Theorem 2).

• We show that the paraphraser becomes increasingly stable on human writing as more human examples are available, with this stability improving at a square-root rate (Theorem 3).

## 2 Preliminaries

Before discussing the results, we define the main notations used in this paper. Let $\mathcal { X }$ denote the space of prompts and let $\mathcal { V }$ denote the space of textual outputs. For a prompt $x \in \mathcal { X }$ , let $P _ { M } ( y \mid x )$ denote the distribution of machine-generated responses and let $\bar { P _ { H } ( y \mid x ) }$ denote the distribution of human-written responses. In practice, the true human distribution is unknown, and we instead observe

$$
\mathcal { H } _ { m } = \{ h _ { 1 } , \ldots , h _ { m } \} , \qquad h _ { j } \overset { \mathrm { i i d } } { \sim } P _ { H } ( \cdot \mid x ) ,
$$

and

$$
{ \mathcal { M } } _ { p } = \{ y _ { 1 } , \ldots , y _ { p } \} , \qquad y _ { i } \overset { \mathrm { i i d } } { \sim } P _ { M } ( { \cdot } \mid x ) .
$$

Definition 1 (Empirical distributions). These samples define the empirical distributions

$$
\widehat { P } _ { H } ^ { ( m ) } = \frac 1 m \sum _ { j = 1 } ^ { m } \delta _ { h _ { j } } , \qquad \widehat { P } _ { M } ^ { ( p ) } = \frac 1 p \sum _ { i = 1 } ^ { p } \delta _ { y _ { i } } ,
$$

where $\delta _ { z }$ places probability one on z.

Throughout, we use Total Variation distance, $\begin{array} { r } { d _ { \mathrm { T V } } ( P , Q ) = \frac { 1 } { 2 } \sum _ { z \in \mathcal { V } } | P ( z ) - Q ( z ) | } \end{array}$

Definition 2 (Quality control oracle). The quality control oracle is defined as $Q : \mathcal { X } \times \mathcal { Y }  [ 0 , 1 ]$ The value $Q ( x , z )$ measures whether z is a correct, relevant, clear, and useful response to prompt x. Fix a quality threshold $q _ { 0 }$ . An output is accepted only if $Q ( x , z ) \geq q _ { 0 }$

Definition 3 (Semantic preservation oracle). Let $\mathcal { Z }$ be the space of all perturbed outputs. Sem : $\mathcal { X } \times \mathcal { Y } \times \mathcal { Z }  [ 0 , 1 ]$ . The value $\mathrm { S e m } ( x , y , z )$ measures whether z preserves the meaning of the original response y for prompt x. We fix a threshold $s _ { 0 }$ . A rewrite of $y$ is accepted only if Sem $( x , y , z ) \ge s _ { 0 }$

We argue that these tools are available in real life, wherein existing detectors can instantiate $D$ while $Q$ and Sem can be implemented using the language model itself as an evaluator. For repeated perturbations of an original response $y _ { i } ,$ semantic preservation is always checked against $y _ { i } .$ , not only against the most recent rewrite, preventing a sequence of individually small changes from drifting into a different answer. Define the admissible set for $y _ { i }$ by

$$
\begin{array} { r } { \mathcal { A } _ { i } = \left\{ z \in \mathcal { V } : Q ( x , z ) \geq q _ { 0 } \mathrm { ~ a n d ~ } \mathrm { S e m } ( x , y _ { i } , z ) \geq s _ { 0 } \right\} . } \end{array}
$$

Definition 4 (Quality-conditioned perturbation kernel). We begin with a base rewrite rule $R _ { i , m } ( z ^ { \prime } \mid z )$ that generates candidate perturbations of the current response. The accepted kernel is then obtained by restricting this proposal to rewrites that satisfy the quality and semantic-preservation constraints. It may use the human sample $\mathcal { H } _ { m }$ . The accepted perturbation kernel is

$$
T _ { i , m } ( z , z ^ { \prime } ) = \frac { R _ { i , m } ( z ^ { \prime } \mid z ) \mathbf { 1 } \{ z ^ { \prime } \in A _ { i } \} } { \displaystyle \sum _ { u \in \mathcal { V } } R _ { i , m } ( u \mid z ) \mathbf { 1 } \{ u \in \mathcal { A } _ { i } \} } .
$$

We assume that the denominator is positive for every state reached by the process. Hence $T _ { i , m } ( z , \mathcal { A } _ { i } ) = 1$ . For a distribution $P$ on $\mathcal { V } _ { \mathrm { ~ ~ } }$ , write $P T _ { i , m }$ for the distribution after one perturbation step:

$$
( P T _ { i , m } ) ( z ^ { \prime } ) = \sum _ { z \in \mathcal { V } } P ( z ) T _ { i , m } ( z , z ^ { \prime } ) .
$$

We claim that such kernels are plausible in practice consistent with work showing strong semantic preservation and modest quality degradation in paraphrasing tools. [6, 4, 7].

Assumption 1 (Human admissibility). For every machine response $y _ { i }$ and every human response $\boldsymbol { h } _ { j } \in \mathcal { H } _ { m }$

$$
Q ( x , h _ { j } ) \geq q _ { 0 } \qquad \mathrm { a n d } \qquad \mathrm { S e m } ( x , y _ { i } , h _ { j } ) \geq s _ { 0 } .
$$

Equivalently, every human response lies in the admissible set $A _ { i }$ , so that

$$
{ \widehat { P } } _ { H } ^ { ( m ) } ( A _ { i } ) = 1 \qquad \mathrm { f o r e v e r y } i .
$$

This is natural in our setting, since the human and machine responses correspond to the same prompt and are intended to represent comparable answers.

Assumption 2 (Human stability). We define the empirical human stability by

$$
\varepsilon _ { m } = \operatorname* { m a x } _ { 1 \leq i \leq p } d _ { \mathrm { T V } } \left( \widehat { P } _ { H } ^ { ( m ) } T _ { i , m } , \widehat { P } _ { H } ^ { ( m ) } \right) .\tag{2.1}
$$

The human stability parameter $\varepsilon _ { m }$ measures how much the perturbation oracle shifts the empirical human distribution. A small $\varepsilon _ { m }$ means human text is approximately stationary under the paraphraser. Assumption 3 (Block mixing). There exist an integer block length $\ell _ { m } \geq 1$ and a constant $\rho _ { m } \in [ 0 , 1 )$ such that, for every i and all distributions P and $Q$ supported on $A _ { i } ,$

$$
\begin{array} { r } { d _ { \mathrm { T V } } \left( P T _ { i , m } ^ { \ell _ { m } } , Q T _ { i , m } ^ { \ell _ { m } } \right) \leq \rho _ { m } d _ { \mathrm { T V } } ( P , Q ) . } \end{array}\tag{2.2}
$$

Assumption 3 formalizes the idea that repeated perturbations make the output progressively forget its starting point. $\ell _ { m }$ ensures that contraction need not occur at every rewrite. It is sufficient that it emerges over blocks of perturbations. Proposition 1 gives a simple condition under which this behavior is guaranteed.[8]

## 3 Related Work

AI-generated text detection. Early detectors used statistical patterns in a language model’s token probabilities. GLTR, for example, displays token ranks to help a reader identify generated text [9]. Later systems learned classifiers from human and machine examples, or combined signals from several language models [10]. Zero-shot methods avoid training a separate detector. [11, 12]. A separate line of work inserts a statistical watermark during generation and tests for that signal later [13]. These methods can be effective in the setting in which they are tested. Their performance, however, can change under new generators, domains, decoding rules, and edits [3, 14]. We do not propose another detector. We study how the distribution that a detector must distinguish changes under repeated rewriting.

Limits of AI detection Several papers study detection as a statistical testing problem. Varshney et al. [15] analyze limits that arise when machine text closely matches human text. Sadasivan et al. [4] give a sharper distributional view. They relate the performance of the best single-text detector to the Total Variation distance between the human and machine distributions. They also show empirically that recursive paraphrasing can lower the accuracy of many detectors while causing only modest quality loss. Chakraborty et al. [5] study the complementary multi-sample setting. They show that small differences between fixed human and machine distributions can accumulate when a detector receives many independent responses.

Paraphrasing Attacks. A large empirical literature shows that rewriting weakens machine-text detectors. DIPPER is a paragraph-level paraphraser with controls for lexical diversity and content order. It preserves the main meaning of a passage while evading several detector families [16]. Recursive paraphrasing strengthens this attack by applying a paraphraser more than once [4]. Other attacks replace selected words or search for prompts that change writing style [17]. Detector scores can also be used as training rewards. This produces generators that are directly optimized to be hard to detect [18]. More recent work guides token selection with a detector during paraphrase generation [19].

Authorship Obfuscation Our use of human examples is also related to authorship obfuscation and style transfer. Early attacks changed a document until an authorship classifier no longer linked it to its writer [20, 21]. Combinatorial paraphrasing later improved content preservation and supported both untargeted obfuscation and imitation of a chosen writing style [22]. Low-resource style-transfer methods use a small set of target-author examples and optimize a balance between style change and semantic preservation [23].

Repeated rewriting as a dynamical process. Other work studies what repeated rewriting does over many rounds. Tripto et al. [24] show that successive paraphrases preserve much of the content but progressively replace the original author’s style with the paraphraser’s style. Wang et al. [25] view successive paraphrasing as a dynamical system and observe fixed points and short attractor cycles. Geng et al. [26] model iterative rephrasing as a Markov chain and study recurrence, diversity, and the effect of decoding choices.

Quality Preserving Perturbations. A closely related theoretical use of a mixing perturbation process appears in the watermarking literature. Zhang et al. [27] assume a quality oracle and a perturbation oracle that induces a rapidly mixing walk over high-quality outputs. They use this process to show that a strong watermark can be removed without knowing the secret key.

## 4 Our Results

## 4.1 Convergence of the full distribution

We first give a simple sufficient condition for block mixing. It is a Doeblin condition on the $\ell _ { m }$ -step kernel, not on every individual perturbation step.

Proposition 1 (Sufficient block-mixing condition). Suppose there exists a number $\lambda _ { m } \in ( 0 , 1 ]$ and, for every i, a probability distribution $\nu _ { i , m }$ such that,for every $z \in A _ { i }$

$$
\begin{array} { r } { T _ { i , m } ^ { \ell _ { m } } ( z , \cdot ) = \lambda _ { m } \nu _ { i , m } ( \cdot ) + ( 1 - \lambda _ { m } ) \mu _ { i , m , z } ( \cdot ) , } \end{array}\tag{3.1}
$$

where $\mu _ { i , m , z }$ is a probability distribution that may depend on z. Then Assumption 3 holds with

$$
\rho _ { m } \leq 1 - \lambda _ { m } .\tag{3.2}
$$

Proof. For any distributions P and Q supported on $A _ { i } .$ , applying (3.1) gives

$$
\begin{array} { r } { P T _ { i , m } ^ { \ell _ { m } } = \lambda _ { m } \nu _ { i , m } + ( 1 - \lambda _ { m } ) P \mu _ { i , m } , } \end{array}\tag{3.3}
$$

and similarly,

$$
\begin{array} { r } { Q T _ { i , m } ^ { \ell _ { m } } = \lambda _ { m } \nu _ { i , m } + ( 1 - \lambda _ { m } ) Q \mu _ { i , m } . } \end{array}\tag{3.4}
$$

The common component $\lambda _ { m } \nu _ { i , m }$ cancels when taking Total Variation distance, so

$$
d _ { \mathrm { T V } } \left( P T _ { i , m } ^ { \ell _ { m } } , Q T _ { i , m } ^ { \ell _ { m } } \right) = ( 1 - \lambda _ { m } ) d _ { \mathrm { T V } } \left( P \mu _ { i , m } , Q \mu _ { i , m } \right) .\tag{3.5}
$$

Since Markov kernels are nonexpansive in Total Variation,

$$
\begin{array} { r } { d _ { \mathrm { T V } } \left( P T _ { i , m } ^ { \ell _ { m } } , Q T _ { i , m } ^ { \ell _ { m } } \right) \leq ( 1 - \lambda _ { m } ) d _ { \mathrm { T V } } ( P , Q ) . } \end{array}\tag{3.6}
$$

Hence Assumption 3 holds with $\rho _ { m } \leq 1 - \lambda _ { m }$

Condition (3.1) states that, after $\ell _ { m }$ perturbations, at least a $\lambda _ { m }$ fraction of the output distribution is common across all starting responses. Thus only the remaining $1 - \lambda _ { m }$ fraction can retain information about the initial response, which directly gives the block contraction in Assumption 3.

## 4.2 Convergence of the empirical distribution

For each starting machine response $y _ { i }$ , define $P _ { i , 0 } = \delta _ { y _ { i } } , P _ { i , k } = \delta _ { y _ { i } } T _ { i , m } ^ { k }$ . The average perturbed machine law is $\begin{array} { r } { \overline { { P } } _ { M , k } ^ { \left( p , m \right) } = \frac { 1 } { p } \sum _ { i = 1 } ^ { p } P _ { i , k } } \end{array}$

Lemma 1 (Preservation at every round). Under the human admissibility and quality

$$
P _ { i , k } ( \mathcal { A } _ { i } ) = 1\tag{4.1}
$$

for every i and every $k \geq 0 .$

Proof. The claim holds at $k = 0$ because the original response is assumed admissible. If it holds at round $k ,$ the quality and meaning preservation condition places the next output in $\mathbf { \mathcal { A } } _ { i }$ with probability one. Induction completes the proof. □

Define the initial average distance $\begin{array} { r } { \overline { { \Delta } } _ { 0 } = \frac { 1 } { p } \sum _ { i = 1 } ^ { p } d _ { \mathrm { T V } } \left( \delta _ { y _ { i } } , \widehat { P } _ { H } ^ { ( m ) } \right) } \end{array}$

Theorem 1 (Empirical machine-to-human movement under block mixing). We show that repeated perturbations progressively erase dependence on the initial machine response, while human stability keeps the resulting distribution close to the empirical human distribution.

For each $k \geq 0$ , write

$$
k = q _ { k } \ell _ { m } + r _ { k } , \qquad q _ { k } = \left\lfloor \frac { k } { \ell _ { m } } \right\rfloor , \qquad 0 \leq r _ { k } < \ell _ { m } .\tag{4.2}
$$

Under human admissibility, quality and meaning preservation, and conditions $( I ) – ( 2 )$ , for every i and every $k \geq 0$

$$
d _ { \mathrm { T V } } \left( P _ { i , k } , \widehat { P } _ { H } ^ { ( m ) } \right) \leq \rho _ { m } ^ { q _ { k } } d _ { \mathrm { T V } } \left( \delta _ { y _ { i } } , \widehat { P } _ { H } ^ { ( m ) } \right) + \left( \ell _ { m } \frac { 1 - \rho _ { m } ^ { q _ { k } } } { 1 - \rho _ { m } } + r _ { k } \right) \varepsilon _ { m } .\tag{4.3}
$$

Consequently,

$$
\boxed { d _ { \mathrm { T V } } \left( \overline { { P } } _ { M , k } ^ { \left( p , m \right) } , \widehat { P } _ { H } ^ { \left( m \right) } \right) \leq b _ { k , m , p } }\tag{4.4}
$$

where

$$
b _ { k , m , p } = \rho _ { m } ^ { q _ { k } } \overline { { \Delta } } _ { 0 } + \left( \ell _ { m } \frac { 1 - \rho _ { m } ^ { q _ { k } } } { 1 - \rho _ { m } } + r _ { k } \right) \varepsilon _ { m } .\tag{4.5}
$$

Proof. Fix i and write

$$
\Delta _ { i , k } = d _ { \mathrm { T V } } \left( P _ { i , k } , \widehat { P } _ { H } ^ { ( m ) } \right) .\tag{4.6}
$$

First, one-step stability and nonexpansiveness of Markov kernels imply that, for every integer $s \geq 1$

$$
d _ { \mathrm { T V } } \left( \widehat { P } _ { H } ^ { ( m ) } T _ { i , m } ^ { s } , \widehat { P } _ { H } ^ { ( m ) } \right) \leq \sum _ { j = 0 } ^ { s - 1 } d _ { \mathrm { T V } } \left( \widehat { P } _ { H } ^ { ( m ) } T _ { i , m } ^ { j + 1 } , \widehat { P } _ { H } ^ { ( m ) } T _ { i , m } ^ { j } \right)\tag{4.7}
$$

$$
\leq s \varepsilon _ { m } .\tag{4.8}
$$

Under human admissibility, both $P _ { i , k }$ and $\widehat { P } _ { H } ^ { ( m ) }$ are supported on $\mathbf { \mathcal { A } } _ { i }$ . At block boundaries, the triangle inequality, (2.2), and (4.8) give

$$
\begin{array} { r } { \Delta _ { i , ( q + 1 ) \ell _ { m } } \leq d _ { \mathrm { T V } } \left( P _ { i , q \ell _ { m } } T _ { i , m } ^ { \ell _ { m } } , \widehat { P } _ { H } ^ { ( m ) } T _ { i , m } ^ { \ell _ { m } } \right) + d _ { \mathrm { T V } } \left( \widehat { P } _ { H } ^ { ( m ) } T _ { i , m } ^ { \ell _ { m } } , \widehat { P } _ { H } ^ { ( m ) } \right) } \end{array}\tag{4.9}
$$

$$
\leq \rho _ { m } \Delta _ { i , q \ell _ { m } } + \ell _ { m } \varepsilon _ { m } .\tag{4.10}
$$

Iteration yields

$$
\Delta _ { i , q \ell _ { m } } \leq \rho _ { m } ^ { q } \Delta _ { i , 0 } + \ell _ { m } \frac { 1 - \rho _ { m } ^ { q } } { 1 - \rho _ { m } } \varepsilon _ { m } .\tag{4.11}
$$

For a remainder $1 \leq r < \ell _ { m }$ , nonexpansiveness and (4.8) give

$$
\begin{array} { r } { \Delta _ { i , q \ell _ { m } + r } \leq d _ { \mathrm { T V } } \left( P _ { i , q \ell _ { m } } T _ { i , m } ^ { r } , \widehat { P } _ { H } ^ { ( m ) } T _ { i , m } ^ { r } \right) + d _ { \mathrm { T V } } \left( \widehat { P } _ { H } ^ { ( m ) } T _ { i , m } ^ { r } , \widehat { P } _ { H } ^ { ( m ) } \right) } \end{array}\tag{4.12}
$$

$$
\leq \Delta _ { i , q \ell _ { m } } + r \varepsilon _ { m } .\tag{4.13}
$$

Combining this inequality with (4.11) proves (4.3). The proof shows that each block of perturbations contracts the current machine–human discrepancy by a factor $\rho _ { m } .$ , while incurring only the additional drift caused by imperfect human stability. Iterating this one-block relation yields the stated convergence rate. It is easy to see that if $\varepsilon _ { m } = 0$ , then

$$
\begin{array} { r } { d _ { \mathrm { T V } } \left( \overline { { P } } _ { M , k } ^ { \left( p , m \right) } , \widehat { P } _ { H } ^ { \left( m \right) } \right) \leq \rho _ { m } ^ { \left\lfloor k / \ell _ { m } \right\rfloor } \overline { { \Delta } } _ { 0 } \longrightarrow 0 . } \end{array}\tag{4.14}
$$

## 5 Sample Requirements

In Section 3, we showed that repeated perturbations can drive the machine distribution toward the empirical human distribution. A natural next question is how much data is required for this guarantee to be meaningful: how many human samples and machine samples are necessary to ensure that the observed perturbed distribution is close to the human distribution?

For comparisons about the semantics of the text, a finite map $\phi : \mathcal { y }  \{ 1 , \ldots , r \}$ , where the categories are semantic classes. For a distribution $P ,$ let $P ^ { \phi }$ denote the distribution of $\phi ( Y )$ when $Y \sim P$ . Since applying $\phi$ cannot increase Total Variation, $d _ { \mathrm { T V } } ( P ^ { \phi } , Q ^ { \phi } ) \leq d _ { \mathrm { T V } } ( P , Q )$ . All forward bounds therefore remain valid after this representation is applied. For the learning-rate derivation below, we additionally treat the r cells as the state space of the perturbation chain.

Lemma 2 (Finite-state Total Variation concentration). To translate the distributional convergence from Section 3 into a finite-sample guarantee, we first quantify how well an empirical measure over r categories approximates the average distribution generating those observations. Let $Z _ { 1 } , \ldots , Z _ { n }$ be independent random variables taking values in $\bar { \{ 1 , \ldots , r \} }$ , not necessarily identically distributed. Define

$$
\widehat { P } _ { n } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \delta _ { Z _ { j } } , \qquad \overline { { P } } _ { n } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathcal { L } ( Z _ { j } ) .\tag{5.1}
$$

Then, with probability at least $1 - \delta$

$$
d _ { \mathrm { T V } } ( \widehat { P } _ { n } , \overline { { P } } _ { n } ) \leq c _ { r } ( n , \delta ) : = \frac { \sqrt { r } + \sqrt { 2 \log ( 1 / \delta ) } } { 2 \sqrt { n } } .\tag{5.2}
$$

Proof. Write $p _ { j , a } = \mathbb { P } ( Z _ { j } = a )$ . By Cauchy–Schwarz,

$$
\mathbb { E } \left\| \widehat { P } _ { n } - \overline { { P } } _ { n } \right\| _ { 1 } \leq \sum _ { a = 1 } ^ { r } \sqrt { \operatorname { V a r } ( \widehat { P } _ { n } ( a ) ) }\tag{5.3}
$$

$$
\leq \sqrt { r \sum _ { a = 1 } ^ { r } \mathrm { V a r } ( \widehat { P } _ { n } ( a ) ) }\tag{5.4}
$$

$$
= { \sqrt { { \frac { r } { n ^ { 2 } } } \sum _ { j = 1 } ^ { n } \left( 1 - \sum _ { a = 1 } ^ { r } p _ { j , a } ^ { 2 } \right) } } \leq { \sqrt { \frac { r } { n } } } .\tag{5.5}
$$

Hence $\mathbb { E } d _ { \mathrm { T V } } ( \widehat { P } _ { n } , \overline { { P } } _ { n } ) \leq \sqrt { r } / ( 2 \sqrt { n } )$ . Replacing one observation changes the Total Variation distance by at most $1 / n$ . McDiarmid’s inequality therefore adds at most $\sqrt { \log ( 1 / \delta ) / ( 2 n ) }$ , which gives (5.2). □

The empirical approximation error decreases at the $n ^ { - 1 / 2 }$ rate, with dependence on the number of categories (r) and the confidence level (δ). Thus, with more samples, the empirical distribution across the (r) categories increasingly reflects the underlying distribution.

## 5.1 Observed machine samples and the true human target

We now ask whether the perturbed-machine samples themselves approximate the true human distribution, rather than only the empirical human target. This requires accounting for the finite-sample

error introduced by observing only p perturbed outputs and m human responses. Draw one perturbed output from each starting machine response,

$$
Y _ { i } ^ { ( k ) } \sim P _ { i , k } , \qquad i = 1 , \dots , p ,\tag{5.6}
$$

independently, and define

$$
\widehat { P } _ { M , k } ^ { \mathrm { o b s } } = \frac { 1 } { p } \sum _ { i = 1 } ^ { p } \delta _ { Y _ { i } ^ { ( k ) } } .\tag{5.7}
$$

The following theorem bounds the Total Variation distance between the observed perturbed-machine distribution and the true human distribution.

Theorem 2 (Finite-sample Total Variation bound). Fix a finite representation $\phi : \mathcal { V } \to \{ 1 , \dots , r \}$ Under the assumptions ofTheorem 1, with probability at least $1 - \delta _ { \mathrm { { \scriptsize \cdot } } }$

$$
d _ { \mathrm { T V } } \left( \left( \widehat { { \cal P } } _ { M , k } ^ { \mathrm { o b s } } \right) ^ { \phi } , { \cal P } _ { H } ^ { \phi } \right) \leq b _ { k , m , p } + c _ { r } \left( p , \frac { \delta } { 2 } \right) + c _ { r } \left( m , \frac { \delta } { 2 } \right) .\tag{5.8}
$$

Ifthe target ofinterest is only the observed empirical human distribution, then with probability at least $1 - \delta ,$

$$
\begin{array} { r } { d _ { \mathrm { T V } } \left( \left( \widehat { P } _ { M , k } ^ { \mathrm { o b s } } \right) ^ { \phi } , \left( \widehat { P } _ { H } ^ { ( m ) } \right) ^ { \phi } \right) \leq b _ { k , m , p } + c _ { r } ( p , \delta ) . } \end{array}\tag{5.9}
$$

Proof. For the first claim, apply the triangle inequality:

$$
d _ { \mathrm { T V } } \left( \left( \widehat { P } _ { M , k } ^ { \mathrm { o b s } } \right) ^ { \phi } , P _ { H } ^ { \phi } \right) \leq d _ { \mathrm { T V } } \left( \left( \widehat { P } _ { M , k } ^ { \mathrm { o b s } } \right) ^ { \phi } , \left( \overline { { P } } _ { M , k } ^ { ( p , m ) } \right) ^ { \phi } \right)\tag{5.10}
$$

$$
+ d _ { \mathrm { T V } } \left( \left( \overline { { { P } } } _ { M , k } ^ { \left( p , m \right) } \right) ^ { \phi } , \left( \widehat { P } _ { H } ^ { \left( m \right) } \right) ^ { \phi } \right)\tag{5.11}
$$

$$
+ d _ { \mathrm { T V } } \left( \left( \widehat { P } _ { H } ^ { ( m ) } \right) ^ { \phi } , P _ { H } ^ { \phi } \right) .\tag{5.12}
$$

Lemma 2 bounds the first and third terms, and Theorem 1 bounds the middle term. A union bound gives probability $1 - \delta$ . The second claim omits the final human-estimation term. □

Our bound consists of three parts. The first, $b _ { k , m , p } ,$ is the perturbation error from Theorem $1 ;$ it is small once enough rounds have been applied. The second, $c _ { r } ( p , \delta / 2 )$ , is the error from observing only p perturbed outputs. The third, $c _ { r } ( m , \delta / 2 )$ , is the error from estimating the human distribution from m samples. Each vanishes as $k , p ,$ and m grow, so the observed perturbed-machine distribution converges to the true human distribution under the representation ϕ.

Corollary 1 (Explicit m and $p$ for the sampling terms). To make the two sampling terms in (5.8) sum to at most ϵ, it is sufficient that

$$
m , p \geq \frac { \left( \sqrt { r } + \sqrt { 2 \log ( 2 / \delta ) } \right) ^ { 2 } } { \epsilon ^ { 2 } } .\tag{5.13}
$$

Thus the sufficient scaling is

$$
m , p = O \left( \frac { r + \log ( 1 / \delta ) } { \epsilon ^ { 2 } } \right) .\tag{5.14}
$$

Proof. Under (5.13), each term $c _ { r } ( n , \delta / 2 )$ is at most $\epsilon / 2$ by direct substitution into (5.2).

The requirement in (5.14) grows linearly in the number of representation classes r and like $1 / \epsilon ^ { 2 }$ in the target accuracy, but only logarithmically in $1 / \delta$ . Halving ϵ therefore requires four times as many samples, whereas a much stronger confidence guarantee requires only a modest increase.

## 5.2 Deriving a pertubation rate

We now bound $\varepsilon _ { m } .$ , previously assumed, in terms of the number of observed human responses m. Among the kernels satisfying the required contraction property, we select the one that comes closest to leaving the empirical human distribution unchanged.

We work on the finite state space $\{ 1 , \ldots , r \}$ described above, interpret the preceding movement theorem on this state space, and write

$$
H = P _ { H } ^ { \phi } , \qquad { \widehat { H } } _ { m } = \left( { \widehat { P } } _ { H } ^ { ( m ) } \right) ^ { \phi } .\tag{5.15}
$$

Fix a block length $\ell \geq 1$ and a block contraction factor $\rho \in [ 0 , 1 )$ . For each starting response i, let ${ \mathfrak { T } } _ { i }$ be a nonempty finite class of admissible Markov kernels on $\{ 1 , \ldots , r \}$ such that every $T \in { \mathfrak { T } } _ { i }$ satisfies

$$
d _ { \mathrm { T V } } ( P T ^ { \ell } , Q T ^ { \ell } ) \leq \rho d _ { \mathrm { T V } } ( P , Q )\tag{5.16}
$$

for all distributions $P , Q$ . Proposition 1 gives one way to enforce this constraint through an ℓ-step minorization condition.

Select the learned kernel by empirical stationarity minimization:

$$
T _ { i , m } \in \mathop { \mathrm { a r g } } _ { T \in \mathfrak { T } _ { i } } \sin { d _ { \mathrm { T V } } } \left( \widehat { H } _ { m } T , \widehat { H } _ { m } \right) .\tag{5.17}
$$

Theorem 3 (Derived stability rate). If the feasible class contains a kernel that preserves the true human distribution, we quantify how closely the empirically selected kernel reaches this stability. Assume that, for every i, the feasible class contains a population-stationary comparator $T _ { i } ^ { \star } \in \dot { \mathfrak { T } } _ { i }$ satisfying

$$
H T _ { i } ^ { \star } = H .\tag{5.18}
$$

Then the kernels selected by (5.17) satisfy, with probability at least $1 - \delta ,$

$$
\varepsilon _ { m } \leq 2 c _ { r } ( m , \delta ) = \frac { \sqrt { r } + \sqrt { 2 \log ( 1 / \delta ) } } { \sqrt { m } } .\tag{5.19}
$$

The resulting $m ^ { - 1 / 2 }$ rate shows that the human-stability error decreases with the amount ofhuman data: as m increases, the empirically learned perturbation rule becomes increasingly stable with respect to the human distribution.

Proof. By empirical optimality,

$$
\begin{array} { r } { d _ { \mathrm { T V } } \left( \widehat { H } _ { m } T _ { i , m } , \widehat { H } _ { m } \right) \leq d _ { \mathrm { T V } } \left( \widehat { H } _ { m } T _ { i } ^ { \star } , \widehat { H } _ { m } \right) . } \end{array}\tag{5.20}
$$

Stationarity of H and nonexpansiveness of Markov kernels give

$$
d _ { \mathrm { T V } } \left( \widehat { H } _ { m } T _ { i } ^ { \star } , \widehat { H } _ { m } \right) \leq d _ { \mathrm { T V } } \left( \widehat { H } _ { m } T _ { i } ^ { \star } , H T _ { i } ^ { \star } \right) + d _ { \mathrm { T V } } ( H , \widehat { H } _ { m } )\tag{5.21}
$$

$$
\leq 2 d _ { \mathrm { T V } } ( \widehat { H } _ { m } , H ) .\tag{5.22}
$$

Taking the maximum over i yields

$$
\varepsilon _ { m } \leq 2 d _ { \mathrm { T V } } ( \widehat { H } _ { m } , H ) .\tag{5.23}
$$

Lemma 2 applied to the human sample gives (5.19).

The realizability condition (5.18) can be relaxed. If the best feasible comparator has population one-step stationarity defect at most η, the same argument adds η to the right-hand side of (5.19); the stochastic term remains $m ^ { - 1 / 2 }$

Corollary 2 (Explicit block and sample rate). We now substitute the learned stability rate into the convergence bound to make the dependence on the number of perturbation rounds and human samples explicit. Under Theorem 3, set $\ell _ { m } = \ell$ and $\rho _ { m } \leq \rho .$ At a block endpoint $k = q \ell$ , with probability at least $1 - \delta$

$$
b _ { q \ell , m , p } \leq \rho ^ { q } + \frac { \ell \left( \sqrt { r } + \sqrt { 2 \log ( 1 / \delta ) } \right) } { ( 1 - \rho ) \sqrt { m } } .\tag{5.24}
$$

For an arbitrary $k = q \ell + r _ { k }$

$$
b _ { k , m , p } \leq \rho ^ { q } + \left( \frac { \ell } { 1 - \rho } + \ell - 1 \right) \frac { \sqrt { r } + \sqrt { 2 \log ( 1 / \delta ) } } { \sqrt { m } } .\tag{5.25}
$$

Consequently,

$$
b _ { k , m , p } = O \left( \rho ^ { \lfloor k / \ell \rfloor } + \frac { \ell } { 1 - \rho } \sqrt { \frac { r + \log ( 1 / \delta ) } { m } } \right) .\tag{5.26}
$$

Proof. Use $\overline { { \Delta } } _ { 0 } \leq 1$ , Theorem 1, and (5.19). At a block endpoint the remainder term vanishes. For arbitrary $k ,$ , use $r _ { k } \le \ell - 1$ and $( 1 - \rho ^ { q } ) / ( 1 - \rho ) \leq 1 / ( 1 - \bar { \rho } )$ □

In (5.26), the term $\rho ^ { \lfloor k / \ell \rfloor }$ decays geometrically in the number of perturbation rounds, while the term of order $m ^ { - 1 / 2 }$ depends only on the human sample size.

Proposition 2 (Human data and perturbation rounds from the derived rate). We now invert the rate in Corollary 4.5 to obtain explicit sufficient choices ofthe number ofperturbation blocks and human samples needed to achieve a target error level ϵ. Fix $0 < \epsilon < 1$ and $0 < \delta < 1$ . For $0 < \rho < 1$ , it is sufficient to choose

$$
q \geq \left\lceil \frac { \log ( 2 / \epsilon ) } { - \log \rho } \right\rceil , \qquad k = q \ell ,\tag{5.27}
$$

$$
m \ge \frac { 4 \ell ^ { 2 } \left( \sqrt { r } + \sqrt { 2 \log ( 1 / \delta ) } \right) ^ { 2 } } { ( 1 - \rho ) ^ { 2 } \epsilon ^ { 2 } }\tag{5.28}
$$

to guarantee $b _ { k , m , p } \leq \epsilon$ with probability at least $1 - \delta . \ : I f \rho = 0 ,$ , one block, $q = 1$ , is sufficient.

Proof. The choices make the geometric and statistical terms in (5.24) at most $\epsilon / 2$ each.

Proposition 2 makes the two requirements explicit. The number of perturbation rounds in (5.27) grows only logarithmically in $1 \bar { / } \epsilon$ , while the number of human samples in (5.28) grows like $1 / \epsilon ^ { 2 }$ The human sample size therefore governs how small ϵ can be made.

## 6 Discussion

Our results reframe paraphrasing attacks as a question about data. Prior work shows that recursive paraphrasing weakens detectors, but an unguided paraphraser drifts toward its own style rather than any particular person’s. Access to human examples changes this. Theorem 1 shows convergence up to an error determined by how stable the human distribution is under the paraphraser, while Theorem 3 shows that this stability improves as more human examples become available.

Our analysis has two main limitations. First, the finite-sample guarantees are stated under a finite representation $\phi ,$ so matching a coarse representation may still leave differences that a detector can exploit. Our theory requires the paraphrasing process to satisfy the block-mixing condition, but we do not empirically verify this property for practical paraphrasers.

These limitations suggest several directions for future work. A more refined theory could introduce a parameter that captures variation within the human distribution and study how this changes the required sample size. It would also be useful to develop practical training procedures for paraphrasers that use human examples while satisfying conditions assumed here.

## References

[1] Heather Johnston, Rebecca F. Wells, Elizabeth M. Shanks, Timothy Boey, and Bryony N. Parsons. Student perspectives on the use of generative artificial intelligence technologies in higher education. International Journal for Educational Integrity, 20(1):2, 2024.

[2] Will Yeadon, Elise Agra, Oto-Obong Inyang, Paul Mackay, and Arin Mizouri. Evaluating AI and human authorship quality in academic writing through physics essays. European Journal of Physics, 45(5):055703, 2024.

[3] Debora Weber-Wulff, Alla Anohina-Naumeca, Sonja Bjelobaba, Tomáš Foltýnek, Jean Guerrero-Dib, Olumide Popoola, Petr Šigut, and Lorna Waddington. Testing of detection tools for AI-generated text. International Journal for Educational Integrity, 19(1):26, 2023.

[4] Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi. Can AI-generated text be reliably detected? arXiv preprint arXiv:2303.11156, 2023.

[5] Souradip Chakraborty, Amrit Bedi, Sicheng Zhu, Bang An, Dinesh Manocha, and Furong Huang. Position: On the possibilities of AI-generated text detection. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 6093–6115. PMLR, 2024.

[6] Elyas Masrour, Bradley Emi, and Max Spero. DAMAGE: Detecting adversarially modified AI generated text. arXiv preprint arXiv:2501.03437, 2025.

[7] Yixuan Even Xu, Ziqian Zhong, Aditi Raghunathan, Fei Fang, and J. Zico Kolter. Base models look human to AI detectors. arXiv preprint arXiv:2605.19516, 2026.

[8] Jeffrey S. Rosenthal. Minorization conditions and convergence rates for markov chain monte carlo. Journal ofthe American Statistical Association, 90(430):558–566, 1995.

[9] Sebastian Gehrmann, Hendrik Strobelt, and Alexander Rush. GLTR: Statistical detection and visualization of generated text. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics: System Demonstrations, pages 111–116. Association for Computational Linguistics, 2019.

[10] Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. Ghostbuster: Detecting text ghostwritten by large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1702–1717. Association for Computational Linguistics, 2024.

[11] Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. DetectGPT: Zero-shot machine-generated text detection using probability curvature. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 24950–24962. PMLR, 2023.

[12] Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature. In The Twelfth International Conference on Learning Representations, 2024.

[13] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 17061–17084. PMLR, 2023.

[14] Liam Dugan, Alyssa Hwang, Filip Trhlík, Andrew Zhu, Josh Magnus Ludan, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch. RAID: A shared benchmark for robust evaluation of machine-generated text detectors. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics, pages 12463–12492. Association for Computational Linguistics, 2024.

[15] Lav R. Varshney, Nitish Shirish Keskar, and Richard Socher. Limits of detecting text generated by large-scale language models. In 2020 Information Theory and Applications Workshop, pages 1–5. IEEE, 2020.

[16] Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. In Advances in Neural Information Processing Systems, volume 36, 2023.

[17] Zhouxing Shi, Yihan Wang, Fan Yin, Xiangning Chen, Kai-Wei Chang, and Cho-Jui Hsieh. Red teaming language model detectors with language models. Transactions of the Association for Computational Linguistics, 12:174–189, 2024.

[18] Charlotte Nicks, Eric Mitchell, Rafael Rafailov, Archit Sharma, Christopher D. Manning, Chelsea Finn, and Stefano Ermon. Language model detectors are easily optimized against. In The Twelfth International Conference on Learning Representations, 2024.

[19] Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, and Soheil Feizi. Adversarial paraphrasing: A universal attack for humanizing AI-generated text. In Advances in Neural Information Processing Systems, volume 38, 2025.

[20] Michael Robert Brennan and Rachel Greenstadt. Practical attacks against authorship recognition techniques. In Proceedings of the Twenty-First Conference on Innovative Applications of Artificial Intelligence, pages 60–65. AAAI Press, 2009.

[21] Janek Bevendorff, Martin Potthast, Matthias Hagen, and Benno Stein. Heuristic authorship obfuscation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1098–1108. Association for Computational Linguistics, 2019.

[22] Tommi Gröndahl and N. Asokan. Effective writing style transfer via combinatorial paraphrasing. Proceedings on Privacy Enhancing Technologies, 2020(4):175–195, 2020.

[23] Shuai Liu, Shantanu Agarwal, and Jonathan May. Authorship style transfer with policy optimization. arXiv preprint arXiv:2403.08043, 2024.

[24] Nafis Irtiza Tripto, Saranya Venkatraman, Dominik Macko, Robert Moro, Ivan Srba, Adaku Uchendu, Thai Le, and Dongwon Lee. A ship of theseus: Curious cases of paraphrasing in LLM-generated texts. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pages 6608–6625. Association for Computational Linguistics, 2024.

[25] Zhilin Wang, Yafu Li, Jianhao Yan, Yu Cheng, and Yue Zhang. Unveiling attractor cycles in large language models: A dynamical systems view of successive paraphrasing. arXiv preprint arXiv:2502.15208, 2025.

[26] Mingmeng Geng, Amr Mohamed, Guokan Shang, Michalis Vazirgiannis, and Thierry Poibeau. Markovian generation chains in large language models. arXiv preprint arXiv:2603.11228, 2026.

[27] Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, and Boaz Barak. Watermarks in the sand: Impossibility of strong watermarking for language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 58851–58880. PMLR, 2024.