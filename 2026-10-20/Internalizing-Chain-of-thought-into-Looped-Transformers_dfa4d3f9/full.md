# Internalizing Chain-of-thought into Looped Transformers

Kehang Zhou Pinyan Lu<sup>∗</sup>

Institute for Theoretical Computer Science Shanghai University of Finance and Economics kehang.zhou@icloud.com lu.pinyan@mail.shufe.edu.cn

## Abstract

Chain-of-thought (CoT) enables constant-bit-size transformers to perform general-purpose computation through generated tokens [Li and Wang, 2025a], whereas looped transformers carry out the same computation by recurrence in hidden states (latent thought). We construct a family of constant-bit-size looped transformers that interpolates between CoT and latent thought. For a prescribed loop count $L = L ( n )$ and a fixed integer � ≥ 1 satisfying $s ( n ) ^ { 1 / k } = \Omega ( L )$ , our transformer simulates a time-�(�), space-�(�) Turing machine using a context window of �(�(�)/�) and $O ( t ( n ) s ( n ) ^ { 1 / k } / L )$ generated tokens, reducing both resources by a factor of � relative to the corresponding unlooped queue-machine simulation. The construction also holds with softmax attention at constant bit-size, under the finite-precision kernel specified in this paper.

## 1 Introduction

Transformers [Vaswani et al., 2017] can perform general-purpose computation through recurrence. A central notion is Turing completeness: the ability to simulate any Turing machine, the standard mathematical model of algorithmic computation. Informally, a Turing-complete model can carry out any computation that a computer program can perform, given suficient time and memory.

One form of recurrence is autoregressive generation, in which the model repeatedly appends a token to its sequence and uses the resulting sequence to generate the next token. Chain-of-thought (CoT) uses these generated tokens for intermediate computation [Wei et al., 2022]. The context window—the range of token positions accessible to attention—serves as the visible workspace.

Pérez et al. [2021] proved that transformers with autoregressive generation are Turing-complete. The earliest constructions use precision that grows with the input, together with a full-history context that scales with running time [Bhattamishra et al., 2020, Merrill and Sabharwal, 2024]. The full-history regime has a provable ceiling: with a polynomial-time next-token map and polynomial context, standard full-history CoT decides only languages in P, hence not all of PSPACE unless P = PSPACE [Yang et al., 2025]. Sliding-window attention and summarized CoT then bring the visible context down from the running time to the space bound �(�(�)) [Schuurmans et al., 2024, Yang et al., 2025, Brösamle and Eckstein, 2026], while precision is reduced first to �(log(� + �(�))) and then to a constant [Merrill and Sabharwal, 2024, Li et al., 2024]. Li and Wang [2025a] complete this line: a constant-bit-size transformer that attends only at fixed relative ofsets is Turing-complete. Li and Wang [2025b] then bring the cost down to $O ( s ( n ) ^ { 1 / k } )$ CoT tokens per simulated machine step by increasing the head–layer product.

Looped transformers provide the complementary form of recurrence, in depth: a shared block is applied repeatedly to the hidden state, performing multiple iterations before emitting a token and carrying intermediate results in latent representations rather than an explicit CoT trace [Giannou et al., 2023, Yang et al., 2024, Geiping et al., 2025]. This internal reasoning is a form of latent thought [Saunshi et al., 2025, Xu and Sato, 2026], and Saunshi et al. [2025] show that loop iterations can simulate CoT steps one-for-one. How does moving computation from generated tokens into hidden-state recurrence change the context window and the number of generated tokens required for Turing-machine simulation?

Our contribution. We construct a family of constant-bit-size looped transformers that interpolates between explicit CoT and latent thought. One shared decoder block is applied � times per generated token, so part of the computation moves from the generated trace into hidden activations. An intermediate queue machine represents the Turing-machine computation, and each of its transitions can be realized either as an explicit CoT step or as one iteration of that block. Relative to the queue-machine simulations of Li and Wang [2025a,b], grouping � simulated transitions into each generated token reduces both the context window and the CoT length by a factor of �.

The mechanism is a per-iteration KV cache read through a sliding window. Iteration � writes its result into the iteration-� cache, the cached entries together hold the $\Theta ( s ( n ) )$ machine cells, and attention at that iteration reads only $O ( s ( n ) / L )$ positions. A context window of that length therefore completes the simulation of an �(�)-space machine. The learned parameters are fixed, machine-dependent constants: they do not grow with � or with �. The positional encoding, the context length, the loop count, and the initial cache are instantiated from the resource bounds. The same construction holds with softmax attention under the finite-precision kernel of Section 5. In the statement below, �<sup>′</sup> is the number of tapes of the simulated machine and � is the hierarchy parameter of the queue simulation (Section 2).

Theorem (informal). Any $( t ( n ) , s ( n ) )$ time-space bounded Turing machine can be simulated by a constantbit-size looped transformer (with its input pre-loaded into the initial KV cache as specified in Section 4.3) with head–layer product $K = 8 k k ^ { \prime }$ and context window length $O ( s ( n ) / L )$ , that takes $O ( s ( n ) ^ { 1 / k } / L )$ CoT steps for each simulated TM step, leading to a total CoT length of $O ( t ( n ) \cdot s ( n ) ^ { 1 / k } / L )$ , for every loop count $L = L ( n )$ and every fixed integer $k \geq 1$ with $s ( n ) ^ { 1 / k } = \Omega ( L )$ . At � = 1, the bounds recover [Li and Wang, 2025b, Thm 1].

For the constructed simulation, explicit and latent reasoning are not two capabilities but two placements of the same general-purpose computation realized by recurrence. Decreasing � externalizes that computation as generated tokens. Increasing � folds that many consecutive CoT steps into each token as latent iterations. The loop count is the exchange rate between the external trace and the internal computation.

Proof idea. We fold � consecutive machine steps of an existing CoT simulation into � iterations of the shared block, replacing � generated tokens by one. The obstacle is layer alignment: in ordinary CoT every past token is visible to every attention layer, whereas iteration � of a looped block reads only iteration-� cache entries of earlier positions, so the inputs consumed at a simulated step must sit in the cache at the layer that executes it. Li–Wang’s queue-machine simulation admits the required reorganization: each simulated step’s queue fronts are cached at the iteration that consumes them. Section 4.5 shows the alignment on one queue.

## Potential messages for real LLMs

The same construction suggests several questions for practical reasoning models. Each is a conjecture: none is backed by an experiment, and none is a claim about a deployed model.

• Looped transformers have the expressive capacity to internalize a substantial part of a chain of thought into recurrent state. Raising the loop count replaces generated reasoning tokens by computation inside the loop, and general-purpose computation is preserved. The loop count is then a way to separate reasoning computation from the tokens a reader sees.

• The number of generated CoT tokens need not measure the serial computation a model performs. In the simulation, one generated token summarizes � consecutive steps, and the intermediate states stay in the recurrent state. Visible length and underlying computation can therefore diverge.

• At large � this creates a learning dificulty. The explicit trace is strided: only every �-th state is emitted as a token, and the steps in between stay latent. Next-token prediction does not supervise those latent steps, so the computation may be dificult to acquire from pretraining alone.

• The decoupling of explicit reasoning trace and latent computation exposes a potential learning challenge at large �: the explicit reasoning trace becomes increasingly strided, since only every �-th computational state is exposed as a token, while intermediate computation remains latent. Such a trajectory is not directly supervised by the standard next-token prediction objective, which conjecturally could make this form of computation dificult to acquire through pretraining alone.

• The construction also suggests how a long-reasoning model might control latent computation, persistent state, and visible tokens separately: take the loop count as an explicit budget for latent computation, and supervise only selected boundary states, so that not every internal step has to be written out as text.

Section 2 fixes the models of computation, and Section 3 the looped transformer. Section 4 gives the hardmax construction and its main theorem, with the initialization interface stated explicitly (Section 4.3) and a worked example at the end (Section 4.5). Section 5 lifts it to softmax at constant bit-size. Section 6 discusses related work; Appendix A constructs congruent queue lengths, Appendix B records the transformer components, Appendix C the exact softmax kernel contract, and Appendix D the detailed comparison table; Appendix E collects notation.

## 2 Models of computation

We simulate Turing machines through an intermediate model, a synchronous multi-queue machine, following Li and Wang [2025a,b].

## 2.1 Turing machines and queue machines

Multi-tape Turing machine. A �<sup>′</sup>-tape Turing machine (TM; �<sup>′</sup> is the tape count) is a tuple $\langle \Sigma , Q , \delta \rangle$ where

• Σ is a finite tape alphabet, including 0, 1, and a blank symbol #.

• � is a finite set of states, including a start state $q _ { \mathrm { s t a r t } }$ and halting states $q _ { \mathrm { a c c } } , q _ { \mathrm { r e j } }$

$\delta : Q \times \Sigma ^ { k ^ { \prime } }  Q \times \Sigma ^ { k ^ { \prime } - 1 } \times ( \{ \mathrm { L e f t , S t a y , R i g h t } \} ) ^ { k ^ { \prime } }$ is a transition function.

The first tape is a read-only input tape, initially holding a finite {0, 1}-string of input �; the remaining �<sup>′</sup>−1 tapes start empty. A TM is $( t ( n ) , s ( n ) )$ ) time-space bounded if, on every input of length �, it halts within �(�) steps and visits at most �(�) distinct cells across the �<sup>′</sup> tapes. Throughout the paper, �<sup>′</sup> always denotes the tape count, leaving � free for the hierarchy exponent of Fact 2.2 below.

Synchronous multi-queue machine. The intermediate model is a synchronous multi-queue machine: the multi-queue form of the Post machine [Post, 1936] in the variant of Li and Wang [2025b, §2.2]. Each queue is a first-in, first-out tape, read at the front and written at the rear. At every step every queue pops its front symbol and appends one symbol, so the queues move in lockstep. There is no separate stay-or-move choice; every front advances by one cell.

• Σ is a finite alphabet, containing 0, 1, and the blank #.

• � is a finite state set, containing the start state $q _ { \mathrm { s t a r t } }$ and the halting states $q _ { \mathrm { a c c } } , q _ { \mathrm { r e j } } .$

$\delta : Q \times \Sigma ^ { K }  Q \times \Sigma ^ { K }$ reads the � front symbols and returns the next state together with the � symbols appended at the rears.

Initially the input queue holds the input string and the other �−1 queues hold only blanks. The space of a queue machine is the maximum number of symbols stored in its queues during the computation, over inputs of length �. In the rest of the paper, �(�) is only the simulated Turing machine’s space bound; the queue machine uses $O ( s ( n ) )$ space.

We take queue 0 to be the input queue, and it is the longest. Every halting state is absorbing: for $q \in \{ q _ { \mathrm { a c c } } , q _ { \mathrm { r e j } } \}$ and every �-tuple $\sigma \in \Sigma ^ { K }$ , set $\delta ( q , \sigma ) = ( q , \# , . . . , \# )$ . The convention makes every run an infinite log. It adds at most $L { - } 1$ extra transitions in the final pass; the halting verdict is unchanged (Proposition 4.2).

Definition 2.1 (K-queue execution log). Let $M = \left( \Sigma , Q , \delta \right)$ be a synchronous �-queue machine with fixed queue lengths $\ell _ { 0 } , \ldots , \ell _ { K - 1 }$ , on input $x = x _ { 0 } \cdot \cdot \cdot \cdot x _ { n - 1 }$ of length $n \leq \ell _ { 0 } .$ , and let $s _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { r } \ell _ { r }$ . Its execution log is $( \sigma _ { m } , q _ { m } ) _ { m \geq 0 } .$ , where $\sigma _ { m } = \mathsf { \bar { ( } } \sigma _ { m } ^ { ( 0 ) } , \ldots , \sigma _ { m } ^ { ( K - 1 ) } )$ . The entries with $m \le s _ { \mathrm { m a x } } - 1$ are given directly: the input feed, for $0 \leq m \leq n - 1$

$$
( \sigma _ { m } , q _ { m } ) = ( x _ { m } , \mathcal { H } ^ { K - 1 } , q _ { \mathrm { s t a r t } } ) ,
$$

and the padding, for $n \le m \le s _ { \mathrm { m a x } } - 1$

$$
( \sigma _ { m } , q _ { m } ) = ( \# ^ { K } , \ q _ { \mathrm { s t a r t } } ) .
$$

For $m \geq s _ { \operatorname* { m a x } } - 1$

$$
( q _ { m + 1 } , \sigma _ { m + 1 } ) = \delta \bigl ( q _ { m } , \sigma _ { m - \ell _ { 0 } + 1 } ^ { ( 0 ) } , \ldots , \sigma _ { m - \ell _ { K - 1 } + 1 } ^ { ( K - 1 ) } \bigr ) ,
$$

every front index $m - \ell _ { r } + 1 \geq s _ { \mathrm { m a x } } - \ell _ { r } \geq 0$ being given or previously produced; the absorbing convention makes the log infinite. The remaining run length after the cache is filled is

$$
\tau ( n ) = \operatorname* { m a x } \big ( 1 , \ \operatorname* { m i n } \{ m \ \lvert \ q _ { m } \in \{ q _ { \mathrm { a c c } } , q _ { \mathrm { r e j } } \} \} - \big ( s _ { \mathrm { m a x } } - 1 \big ) \big ) .
$$

The step-� front of queue � is the symbol appended at step $m - \ell _ { r } + 1 ; \tau ( n )$ counts the transitions from step $s _ { \operatorname* { m a x } } - 1$ , the first step whose fronts are available, to the first halting state.

The conversion from Turing machines to this model is cited, not re-derived:

Fact 2.2 (Queue Turing machines are Turing-complete [Li and Wang, 2025b, Thm. 2, Step I]). For every fixed integer $k \geq 1 , \mathrm { a } t ( n )$ -time, �(�)-space $k ^ { \prime } { \mathrm { - t a p e } }$ Turing machine converts to a synchronous $K = 6 k k ^ { \prime }$ -queue machine running in $O ( t ( n ) s ( n ) ^ { 1 / k } )$ steps and $O ( k ^ { \prime } s ( n ) )$ total queue space. Each tape becomes two stacks and each stack � levels of queues. Writing $B = \lceil s ( n ) ^ { 1 / k } \rceil$ , level $g \left( 1 \leq g \leq k \right)$ uses two content-half queues of length $B ^ { g }$ and one bufer of length $2 B ^ { g }$ ; consecutive half lengths divide one another, and the cost analysis is in terms of these lengths [Li and Wang, 2025b, §3].

Two remarks on scope. First, the level structure and the cost bounds are what we cite; the congruent base �, the single length per level, the bufer split, and the resource bookkeeping are argued in Appendix A. Second, � is fixed before instantiation (it sets the head count $K = 8 k k ^ { \prime } )$ and � is prescribed afterwards, subject to $s ( n ) ^ { 1 / k } = \Omega ( L )$

## 3 The model: looped transformers with a per-iteration cache

## 3.1 Decoder-only transformers with sliding-window attention

A decoder-only transformer [Vaswani et al., 2017] is a parameterized next-token map $\mathcal { V } ^ { * } \to \mathcal { V }$ . The forward pass for position � sets $h _ { i } ^ { ( 0 ) } = \mathrm { e m b } ( \nu _ { i } )$ and applies, at each layer �, an attention sublayer followed by a feed-forward network, both with residual connections:

$$
h _ { i } ^ { ( l + 0 . 5 ) } = W _ { \cal O } ^ { ( l ) } \cdot \mathrm { c o n c a t } _ { h } \Big ( \sum _ { j } s _ { h , i , j } ^ { ( l ) } \cdot W _ { \nu } ^ { ( l , h ) } h _ { j } ^ { ( l ) } \Big ) , \qquad \tilde { h } _ { i } ^ { ( l ) } = h _ { i } ^ { ( l + 0 . 5 ) } + b ^ { ( l ) } + h _ { i } ^ { ( l ) } ,
$$

$$
h _ { i } ^ { ( l + 1 ) } = \mathrm { F F } ^ { ( l ) } \left( \tilde { h } _ { i } ^ { ( l ) } \right) + \tilde { h } _ { i } ^ { ( l ) } ,
$$

where the inner sums range over the context window and the concatenation runs over the � heads. The learned parameters comprise the embedding and output head, query and key matrices ${ W _ { q } ^ { ( l , h ) } , W _ { k } ^ { ( l , h ) } \in \mathbb { R } ^ { d _ { k } \times d } }$ value matrices $W _ { \nu } ^ { ( l , h ) } \in \mathbb { R } ^ { d _ { \nu } \times d }$ , output matrices $W _ { O } ^ { ( l ) } \in \mathbb { R } ^ { d \times H d _ { \nu } }$ , biases $b ^ { ( l ) }$ , and feed-forward networks $\mathrm { F F } ^ { ( l ) }$ . A relative positional function $\mathrm { p o s } ^ { ( l ) } : \mathbb { N }  \mathbb { R } ^ { d _ { k } }$ is specified separately and enters the key side of the scores: the weights $s _ { h , i , } ^ { ( l ) }$ form the hardmax distribution (all weight on the score’s maximizer) over scores $\langle W _ { q } ^ { ( l , h ) } h _ { i } ^ { ( l ) } , W _ { k } ^ { ( l , h ) } h _ { j } ^ { ( l ) } + \mathrm { p o s } ^ { ( l ) } ( i - j ) \rangle$ . The next token is $\nu _ { i + 1 } = \mathrm { a r g m a x } ( W _ { \mathrm { o u t } } h _ { i } ^ { ( L ) } )$ at the last layer, for a decoder of depth �.

In the sliding-window setting with window �, position � attends to $\{ i - w + 1 , \ldots , i \} \cap \mathbb { N }$ . The KV cache is indexed by (position, layer): $\overline { { W _ { k } ^ { ( l , h ) } } } h _ { j } ^ { ( l ) }$ and $\hat { W } _ { \nu } ^ { ( l , h ) } h _ { j } ^ { ( l ) }$ are written and fixed when position � completes layer �.

Remark. With sliding-window attention and a persistent KV cache, a layer-� transformer can depend on $L ( w - 1 ) + 1$ positions, more than the visible window. This span alone does not provide the alignment needed for simulation: each queue value must still sit in the correct iteration’s cache (Section 4.5).

## 3.2 Looped transformers

Looped transformers re-apply a fixed unit instead of stacking independently parameterized layers [Giannou et al., 2023, Yang et al., 2024, Geiping et al., 2025]. We adopt a weight-tying version [Dehghani et al., 2019, Lan et al., 2020]: the unit acts � times on the current position’s residual stream per generated token, with a separate per-iteration KV cache carrying state across positions.

Definition 3.1 (looped transformer). A looped transformer is a decoder-only transformer consisting of

• a loop unit $u _ { \varphi } \mathrm { : }$ a stack of $p \geq 1$ decoder layers specified by the shared weights $\varphi ;$

• the embedding emb and the output head $W _ { \mathrm { o u t } }$ , applied outside the loop;

together with a loop count $L \in \mathbb { N } .$ . Given �, the forward pass computes

$$
\begin{array} { r l } & { h _ { i } ^ { ( 0 ) } = \mathrm { e m b } ( \nu _ { i } ) , \qquad h _ { i } ^ { ( l + 1 ) } = u _ { \varphi } \big ( h _ { i } ^ { ( l ) } ; C _ { i } ^ { ( l ) } \big ) \quad ( 0 \le l < L ) , } \\ & { \nu _ { i + 1 } = \mathrm { a r g m a x } \big ( W _ { \mathrm { o u t } } h _ { i } ^ { ( L ) } \big ) , } \end{array}
$$

the �-th application of $u _ { \varphi }$ being iteration � (layer and iteration are used interchangeably). Unrolling the loop yields a $p L$ -layer transformer in which each iteration applies the same $p$ layers. Shared parameters are independent of �.

Reading rule (per-iteration cache). The cache $C _ { i } ^ { ( l ) }$ holds the iteration-� key and value entries of the retained positions $j < i .$ . Iteration � of position � attends only to those same-iteration entries; the current position’s key and value are computed from $h _ { i } ^ { ( l ) }$ and are frozen when written. A fixed iteration never reads another iteration’s cache. Section 4.5 shows the discipline on one queue, and the construction has to follow it.

A looped transformer is constant-bit-size if the number of distinct learned parameters, their bit precision, and the bit precision of scalar intermediate values are all $O ( 1 )$ , independently of input length, loop count, and generation length.

## 4 The hardmax construction

We first construct the simulation with hardmax attention; Section 5 realizes the same selection exactly under a finite-precision softmax kernel. Throughout this section, queue � has length $\ell _ { r } \equiv 1$ (mod �) (queue lengths satisfying this congruence are constructed in Appendix A). Set $w = \operatorname* { m a x } _ { r } w _ { r }$ and $s _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { r } \ell _ { r }$ . Each forward pass executes � consecutive machine steps. Head � reads the front of queue � at distance $w _ { r } - 1$ attention retrieves that front by a fixed relative ofset, and the feed-forward network applies the transition.

## 4.1 Tokens and residual layout

At nonterminal position �, the token is $( q _ { L i } , \sigma _ { L i } )$ , where $\sigma _ { L i } = ( \sigma _ { L i } ^ { ( 0 ) } , \dots , \sigma _ { I . i } ^ { ( K - 1 ) } ) \in \Sigma ^ { K }$ is the tuple appended at log index ��. Generated tokens are thus machine-state symbols, not natural-language steps.

The embedding uses the layout specified in Appendix B:

$$
\begin{array} { r } { \boldsymbol { h } _ { i } ^ { ( 0 ) } = \big ( 1 , \mathrm { ~ } e _ { q _ { L i } } , \mathrm { ~ o n e h o t } ( \sigma _ { L i } ^ { ( 0 ) } ) , \dots , \mathrm { o n e h o t } ( \sigma _ { L i } ^ { ( K - 1 ) } ) , 0 ^ { | \Sigma | K } \big ) . } \end{array}
$$

The blocks store respectively a constant, the current state, the appended tuple, and � scratch registers for retrieved fronts, giving $d = 2 | \Sigma | K + | Q | + 1 = \Theta ( K )$ . At iteration �, attention writes the fronts and the feed-forward network applies �, leaving the residual equal to the embedding of $( q _ { L i + l + 1 } , \sigma _ { L i + l + 1 } )$ . After iteration $L - 1$ , the output head emits $( q _ { L ( i + 1 ) } , \sigma _ { L ( i + 1 ) } )$ , or ACCEPT/REJECT if the state is halting, and the generation stops.

## 4.2 Positional encoding and the attention heads

The relative positional encoding is the simplified form of Li and Wang [2025b, Eq. (2)]: writing $e _ { r }$ for the basis vector of $\mathbb { R } ^ { d _ { k } }$ indexed by head number � (Appendix B fixes the indexing conventions), $d _ { k } = K$

$$
\mathrm { p o s } ( i - j ) \ = \ \Delta \sum _ { r : i - j = w _ { r } - 1 } e _ { r } ,
$$

whose �-th coordinate is Δ exactly at distance $i - j = w _ { r } - 1$ . The query of head � is the constant $e _ { r }$ and the key is set to be 0. The positional encoding enters the key side of the scores (Section 3). Thus

$$
\mathrm { s c o r e } _ { r } ( i , j ) = \langle e _ { r } , \mathrm { p o s } ( i - j ) \rangle = \Delta [ i - j = w _ { r } - 1 ] .
$$

For every pass $i \geq w - 1 , j ^ { * } = i - w _ { r } + 1$ is the unique maximizer, so hardmax selects queue $r { } _ { \mathrm { { s } } }$ designated position: the ofset $w _ { r } - 1$ encodes “front of queue � at the current iteration,” and Lemma 4.1 proves that the entry selected is the front consumed at step $L i + l$ . Equal-length queues remain separate via distinct heads and value projections. Appendix B gives the matrices.

Value entries and transition. Let $V _ { r } ( j , l ) = W _ { \nu } ^ { ( r ) } h _ { i } ^ { ( l ) } \in \mathbb { R } ^ { d _ { \nu } } , d _ { \nu } = | \Sigma |$ , be head $r { } _ { \mathrm { { s } } }$ cached value at position $j$ and iteration �, and let $V ( j , l )$ denote the represented �-tuple. Each $\boldsymbol { W } _ { \nu } ^ { ( r ) }$ projects symbol block $r ; W _ { O }$ writes the selected values into the front blocks, and the feed-forward network applies �. All � iterations share the same positional encoding, value projections, and � lookup, so the construction satisfies Definition 3.1.

## 4.3 The loop invariant

Initialization. Generation begins at pass $w - 1$ with a loaded cache: for $0 \leq i \leq w - 2$ and $0 \leq l \leq L - 1$ , set the cached value entry $V ( i , l ) = \sigma _ { L i + l }$ , and set the current token to $( q _ { L ( w - 1 ) } , \sigma _ { L ( w - 1 ) } )$ . Since $L ( w - 1 ) = s _ { \mathrm { m a x } } - 1$ the first simulated transition occurs at step $s _ { \operatorname* { m a x } } - 1$ ; the remaining run length is $\tau ( n )$ (Definition 2.1).

We treat this initialization as an explicit interface assumption. It prescribes which entries a working system must place in the cache before generation; the cost of producing them is not included in any bound of this paper, and we do not prove that a natural-order prompt prefill computes them.

Lemma 4.1. For every position � and iteration �, the value entry $V ( i , l )$ encodes the �-tuple at log index $L i + l { \mathrm { : } }$ block � holds $\dot { \sigma } _ { L i + l } ^ { ( \bar { r } ) } .$

Proof. The initialization gives the claim for $0 \leq i \leq w - 2$ and all �, and the token at position $w - 1$ supplies its iteration-0 entry $V ( w - 1 , 0 ) = \sigma _ { L ( w - 1 ) }$

Consider pass $i \geq w - 1$ and iteration �, and suppose the residual entering iteration � contains $( q _ { L i + l } , \sigma _ { L i + l } )$ Head � attends to position $i - w _ { r } + 1$ , whose iteration-� entry was written in an earlier pass and is therefore the log entry

$$
\begin{array} { r } { V ( i - w _ { r } + 1 , l ) ^ { ( r ) } ~ = ~ \sigma _ { L ( i - w _ { r } + 1 ) + l } ^ { ( r ) } ~ = ~ \sigma _ { L i + l - \ell _ { r } + 1 } ^ { ( r ) } , } \end{array}
$$

the front of queue � for step $L i + l ,$ , because $\ell _ { r } - 1 = L ( w _ { r } - 1 )$ . The residual stream supplies $q _ { L i + l } .$ , and the feed-forward network computes

$$
\left( q _ { L i + l + 1 } , \sigma _ { L i + l + 1 } \right) \ = \ \delta \left( q _ { L i + l } , \sigma _ { L i + l - \ell _ { 0 } + 1 } ^ { ( 0 ) } , \ldots , \sigma _ { L i + l - \ell _ { K - 1 } + 1 } ^ { ( K - 1 ) } \right) .
$$

For $l ~ < ~ L - 1$ , this result is cached as $V ( i , l + 1 )$ , and the residual entering iteration $l + 1$ contains $( q _ { L i + l + 1 } , \sigma _ { L i + l + 1 } )$ ; for $l = L - 1$ , the output head emits $( q _ { L ( i + 1 ) } , \sigma _ { L ( i + 1 ) } )$ , whose embedding supplies $V ( i + 1 , 0 )$ . Every value entry is therefore the corresponding log entry, as claimed. □

Figure 1 illustrates one pass.

## 4.4 The main theorem

Proposition 4.2 (queue-level construction). Let $M = \left( \Sigma , Q , \delta \right)$ be a synchronous �-queue machine over a fixed alphabet Σ, with absorbing halting states and lengths $\ell _ { r } = L ( w _ { r } - 1 ) + 1$ . Under the initialization of Section 4.3, the construction of Sections $4 . I { \ - } { 4 . 3 }$ gives a decoder-only looped transformer with one shared block, � heads, window $w = \operatorname* { m a x } _ { r } w _ { r } ,$ , embedding dimension $2 | \Sigma | K + | Q | + 1$ , and constant bit-size. It emits the execution log at stride � and therefore uses

$$
T _ { \mathrm { c o t } } = \left\lceil \frac { \tau ( n ) } { L } \right\rceil
$$

generated tokens, including at most $L - 1$ absorbing transitions in the final pass.

Proof. The decoder block, � heads, window �, dimension $2 | \Sigma | K + | Q | + 1$ , and constant bit-size are those of Sections 4.1–4.2. By Lemma 4.1, at every pass $i \geq w - 1$ and every iteration � the cached values are the log entries $\sigma _ { L i + l }$ , so head � retrieves the front of queue � at machine step $L i + l$ and the feed-forward network applies �. After � iterations the residual is therefore the embedding of $( q _ { L ( i + 1 ) } , \sigma _ { L ( i + 1 ) } )$ . Appendix B (B.6) emits this pair as the token at position � + 1, or ACCEPT/REJECT if the state is halting. Generation thus starts from the initialization of Section 4.3 at index $L ( w - 1 ) = s _ { \operatorname* { m a x } } - 1$ and follows the log at stride � for the remaining $\tau ( n )$ steps, giving $T _ { \mathrm { c o t } } = \lceil \tau ( n ) / L \rceil$ with at most $L - 1$ absorbing transitions in the final pass. □

![](images/b9a67ce6368359aa4e8850365b468fd3f6935ea918b2071eeb62f41336e852ac.jpg)  
Figure 1: One pass at position �. At iteration �, head � retrieves queue $r { } _ { \mathrm { { s } } }$ front from $V ( i - w _ { r } + 1 , l )$ , and the feed-forward network advances the machine by one step. The final write is emitted as the token at position $i + 1$

Theorem 4.3. Any $( t ( n ) , s ( n ) )$ time-space bounded �<sup>′</sup>-tape Turing machine can be simulated by a constantbit-size looped transformer (with its input encoded in the initial KV cache as specified in Section 4.3) with head–layer product $K = 8 k k ^ { \prime }$ and context window length $O ( s ( n ) / L )$ , that takes $O ( s ( n ) ^ { 1 / k } / L )$ CoT stepsfor each simulated TM step, leading to a total CoT length of $O ( t ( n ) \cdot s ( n ) ^ { 1 / k } / L )$ , for every loop count $L = L ( n )$ and everyfixed integer $k \geq 1$ with $s ( n ) ^ { 1 / k } = \Omega ( L )$ .

Proof. For the prescribed � and the given fixed � satisfying $s ( n ) ^ { 1 / k } \ = \ \Omega ( L )$ , the congruent lengths supplied by Appendix A (Lemma $\mathbf { A . } 1 )$ give a step-equivalent $K = 8 k k ^ { \prime }$ -queue simulation with run length $\tau ( n ) = O ( t ( n ) s ( n ) ^ { 1 / k } )$ and window $w = O \left( s ( n ) / L \right)$ . Proposition 4.2 yields

$$
T _ { \mathrm { c o t } } = O \left( \frac { t ( n ) s ( n ) ^ { 1 / k } } { L } \right) .
$$

Multiplying by � gives the serial-work bound. The cache stores � positions at each of � iterations, so total KV state is $\Theta ( L w ) = \Theta ( s ( n ) )$ ). □

Resource accounting. The factor-� reduction applies to visible resources only: generated tokens and the visible window shrink by �, while the total cached KV state $\Theta ( L w ) = \Theta ( s ( n ) )$ and the serial block work $L \cdot T _ { \mathrm { c o t } } = O ( t ( n ) s ( n ) ^ { 1 / k } )$ are unchanged from $L = 1$ . Table 1 tabulates both regimes. The � heads of the single shared block are re-applied at each iteration, so the head–layer product does not grow with �.

A concrete instance: the balanced point $s ( n ) = m ^ { 2 } , k = 2 , L = \Theta ( m )$ is admissible because $s ( n ) ^ { 1 / k } =$ $m = \Omega ( L )$ ), and it gives $O ( t ( n ) )$ generated tokens, a window of $O ( m )$ positions, $\Theta ( m ^ { 2 } )$ cached entries, and $O ( t ( n ) m )$ loop executions.

<table><tr><td>Quantity</td><td>L = 1 (endpoint)</td><td>admissible  $L = O ( s ( n ) ^ { 1 / k } )$ </td></tr><tr><td>Generated tokens</td><td> $O ( t ( n ) s ( n ) ^ { 1 / k } )$ </td><td> $O \big ( t ( n ) s ( n ) ^ { 1 / k } / L \big )$  , exactly  $\lceil \tau ( n ) / L \rceil$ </td></tr><tr><td>Visible positions (window)</td><td> $O ( s ( n ) )$ </td><td> $O \big ( s ( n ) / L \big )$ </td></tr><tr><td>Loop executions (serial block work)</td><td> $O ( t ( n ) s ( n ) ^ { 1 / k } )$ </td><td> $\dot { O ( t ( n ) s ( n ) ^ { 1 / k } ) }$  , unchanged</td></tr><tr><td>Total cached KV state</td><td> $\Theta ( s ( n ) )$ </td><td> $\Theta ( L w ) = \Theta ( s ( n ) )$  , unchanged</td></tr><tr><td>Learned model size</td><td></td><td>O(1), machine-dependent O(1), machine-dependent</td></tr></table>

Table 1: The construction at the endpoint $L = 1$ (which recovers the bounds of Li and Wang, 2025b) and at a general admissible loop count. The simulated Turing machine is the same, while the queue lengths are re-instantiated for each �; only the first two rows shrink with �.

## 4.5 A worked example

Take one queue of length $\ell = 7$ and loop count � = 3. The window is then $w = 3 \colon$ at pass � and iteration �, the head reads position � − 2, at relative distance $w - 1 = 2$ , in the same iteration. Since $\ell - 1 = 6 = L ( w - 1 )$ , the entry at position $i - 2$ and iteration � is the front consumed at step $3 i + l .$ . The construction keeps a separate copy of this bookkeeping for every queue. One queue is enough to see the alignment.

Before generation, the loaded cache of Section 4.3 must already hold $V ( i , l ) = \sigma _ { L i + l }$ for $0 \leq i \leq w - 2 = 1$ and $l \in \{ 0 , 1 , 2 \} \colon \sigma _ { 0 } , \sigma _ { 1 } , \sigma _ { 2 }$ at position 0 and $\sigma _ { 3 } , \sigma _ { 4 } , \sigma _ { 5 }$ at position 1, together with the token $( q _ { 6 } , \sigma _ { 6 } )$ at position 2. Here $L ( w - 1 ) = 6 = s _ { \operatorname* { m a x } } - 1$ , and $s _ { \operatorname* { m a x } } = 7$ is the length of this queue, so generation begins at step 6, the first step whose front is already in the cache.

Table 2 lists the first passes. At pass $i = 3$ the three iterations execute steps 9, 10, and 11, and iteration � reads $V ( 1 , l ) = \sigma _ { 3 + l }$ . On a queue of length 7, the symbol consumed at step � was appended at step $m - 7 + 1$ so those fronts are the symbols appended at steps 3, 4, and 5. In general, the symbol read at pass � and iteration � has log index

$$
3 ( i - 2 ) + l \ = \ 3 i + l - 7 + 1 .
$$

Each pass advances the machine from log index 3� to $3 i + 3$ and emits the state at the boundary. The visible tokens $( q _ { 6 } , \sigma _ { 6 } ) , ( q _ { 9 } , \sigma _ { 9 } ) , ( q _ { 1 2 } , \sigma _ { 1 2 } ) , . .$ . are the log sampled at stride $L = 3$ . Intermediate steps, such as steps 7 and 8 inside pass 2, are never emitted.
<table><tr><td>Pass i</td><td>Window</td><td>Steps executed</td><td>Reads  $( l = 0 , 1 , 2 )$ </td><td>Residual advances from</td><td>Emits</td></tr><tr><td>2</td><td>{0, 1, 2}</td><td>6,7,8</td><td> $\sigma _ { 0 } , \sigma _ { 1 } , \sigma _ { 2 }$  (at pos. 0)</td><td>6,7,8</td><td> $( q _ { 9 } , \sigma _ { 9 } )$ </td></tr><tr><td>3</td><td>{1,2,3}</td><td>9, 10,11</td><td> $\sigma _ { 3 } , \sigma _ { 4 } , \sigma _ { 5 }$  (at pos. 1)</td><td>9, 10, 11</td><td> $( q _ { 1 2 } , \sigma _ { 1 2 } )$ </td></tr><tr><td>4</td><td>{2,3,4}</td><td>12, 13, 14</td><td>(at pos. 2)  $\sigma _ { 6 } , \sigma _ { 7 } , \sigma _ { 8 }$ </td><td>12, 13, 14</td><td> $( q _ { 1 5 } , \sigma _ { 1 5 } )$ </td></tr></table>

Table 2: One queue with $L = 3 , \ell = 7 ,$ , and $w = 3$ . At pass � and iteration �, the head reads position $i - 2$ in iteration �, which is log index $3 ( i - 2 ) + l = 3 i + l - 7 + 1$ . The pass advances the machine by three steps and emits every third log state.

Three features of the example are what the construction uses. First, iteration 1 of pass 3 reads $V ( 1 , 1 ) = \sigma _ { 4 }$ the same iteration of an earlier position. It does not read an iteration-0 entry, and it does not mix iterations. Section 3 states this per-iteration rule. Second, a value entry is frozen once written. Initialization supplies $V ( 1 , \cdot ) ;$ pass 2 writes $V ( 2 , \cdot )$ , and pass 4 reads that same entry. Third, the window has to be the one the congruence gives. The front required at step 11 is $\sigma _ { 5 } .$ , written at position 1, exactly $w - 1 = 2$ positions back. With a window of 2, position 1 would already lie outside the positions pass 3 can read. We call a representation layer-aligned when, at every simulated step, the inputs that step consumes already sit in the KV cache at the iteration, or layer, that executes the step.

## 5 Softmax simulates hardmax at constant bit-size

Section 4 uses hardmax attention. Theorem 4.3 also holds for a max-first binary32 softmax kernel, with no growth in precision as the window or the loop count grows. The five-clause kernel contract, the underflow arithmetic, and the complete proof are in Appendix C.

Write $a _ { i , j }$ for the attention score of candidate position � in the row of query position �. For a row with unique maximizer $j ^ { * }$ , define the attention score gap

$$
\Delta \ = \ a _ { i , j ^ { * } } \ - \ \operatorname* { m a x } _ { j \neq j ^ { * } } a _ { i , j } ,
$$

the score separation of [Brösamle and Eckstein, 2026, Def. D.1].

The kernel, in words. The kernel evaluates the exponential only on inputs shifted by the row maximum, using a fixed binary32 exponential that returns exactly 1 at 0 and +0 at every input at most $- 2 5 6 ;$ numerator and denominator are accumulated in binary32 and divided once; the quotient is cast once at the output boundary, to a fixed format that contains 0 and 1 exactly. The role of the threshold is arithmetic: since $2 5 6 > 1 5 0 \ln { 2 }$ , we have $e ^ { - 2 5 6 } < 2 ^ { - 1 5 0 }$ , the round-to-nearest midpoint between 0 and the least positive binary32 subnormal, so a correctly rounded binary32 exponential returns +0 on every input at most −256 (Appendix C).

Lemma 5.1 (softmax implements unique hardmax exactly). Let $( a _ { j } ) _ { j }$ be a score row with unique maximizer �<sup>∗</sup> and score gap at least 256. Then the max-first kernel’s computed attention output is exactly $\nu _ { j ^ { * } }$ , independently of the row length, provided $\nu _ { j ^ { * } }$ is exactly representable in the accumulation format and preserved by the output cast—in particularfor the values in {0, 1} used in this construction.

Proof. By the kernel, the shifted inputs are $\hat { a } _ { j ^ { * } } = 0$ and $\hat { a } _ { j } \le - 2 5 6$ for every $j \neq j ^ { * }$ . The exponential row is exactly the one-hot indicator of �<sup>∗</sup>: $E _ { 3 2 } ( \hat { a } _ { j ^ { * } } ) = 1$ and $E _ { 3 2 } ( \hat { a } _ { j } ) = + 0$ for every $j \neq j ^ { * }$ . Numerator and denominator are therefore the exact sums $\nu _ { j ^ { * } }$ and 1, each partial sum containing only zeros and one nonzero term. Division and the single boundary cast preserve the result. □

Corollary 5.2 (the main results hold under softmax). Set $\Delta = 2 5 6$ in Section 4.2. Then every score row has scores in {256, 0}, a unique maximizer at distance $w _ { r } - 1$ , and shifted losers exactly −256, all exactly representable in binary32. The value vectors are one-hot, and the remaining parameters andfeed-forward computations use afixed machine-dependentformat as specified in Appendix B. By Lemma 5.1 every attention output equals the hardmax one-hot ofSection 4.2 bitfor bit, so Theorem 4.3 and Proposition 4.2 hold with softmax attention under the max-first kernel: the parameters are those ofthe hardmax construction rescaled by thefixed ratio between its score gap and the kernel constant 256 (here 1, since $\Delta = 2 5 6 )$ , the resources are unchanged, and the bit-size is still $O ( 1 )$ , independent of� and �.

Remark (scope). The result is conditional on the stated kernel, in three ways. First, it requires a binary32 exponential that returns +0 below a fixed threshold, without a positive floor, logit soft-capping, or an additive � in the normalizer; the exact contract is Appendix C. Second, the arithmetic is fixed-format: values not exactly representable in it (arbitrary real value vectors, for instance) are outside the exactness statement. Third, no claim is made about kernels beyond the contract: subtract-max is standard in production softmax implementations (Appendix C records examples), but that observation is context, not a conformance proof, and in particular online and fused attention kernels with their own epilogues are not thereby covered. Appendix C also treats two variants: a kernel that does not subtract the maximum (then Section 4.2 can set the winner’s score to 0 and every other score to −256), and a kernel with a positive scale, canonically $1 / \sqrt { d _ { k } }$ (then instantiate $\Delta = \lceil 2 5 6 \sqrt { d _ { k } } \rceil$ ; here $d _ { k } = K$ is a machine constant, so $\Delta$ remains $O ( 1 ) )$ . Nothing generalizes to exact-real softmax.

The comparison with exact-real arithmetic is a comparison of assumptions. Brösamle and Eckstein [2026] simulate Turing machines with low-precision softmax and summarized CoT, separating the winner from the other scores with depth, width, and precision that grow logarithmically. The present construction fixes a constant score gap and lets finite-format underflow make the selection exact; the only added cost is that constant gap Δ. These are diferent precision and architecture assumptions. We do not claim that their arithmetic implements exact hard selection.

## 6 Related work

We keep this section argument-driven; Appendix D carries the detailed comparison table of transformers simulating (�, �)-bounded Turing machines, with per-row notes and normalized notation.

Transformer expressivity. RASP models transformers as readable programs [Weiss et al., 2021, Lindner et al., 2023]. Variants such as B-RASP, C-RASP, and RASP-L, together with subsequent formal-language analyses, characterize the languages recognized under diferent architectural restrictions [Angluin et al., 2024, Yang and Chiang, 2024, Zhou et al., 2024, Strobl et al., 2024, 2025, Huang et al., 2025]; see Strobl et al. [2024] for a survey. On the circuit side, Merrill and Sabharwal [2023] show that constant-depth, log-precision transformers without CoT are limited to ${ \mathsf { T C } } ^ { 0 } { \mathsf { - s t y } }$ le shallow circuits. Feng et al. [2023] connect CoT length to circuit depth: polynomially long CoT lifts transformers from shallow-circuit problems to all of P.

Looped transformers and latent thought. Looped transformers extend serial computation by re-applying a shared block [Giannou et al., 2023, Yang et al., 2024, Geiping et al., 2025]; Merrill and Sabharwal [2025] show that logarithmically deep looped transformers recognize regular languages and solve graph connectivity without CoT. Saunshi et al. [2025] study how latent thought can simulate computations expressed through CoT. Xu and Sato [2026] establish simulation results and complexity-class characterizations for CoT and latent thought, together with conditional separations: latent iterations support more eficient parallel computation in their deterministic setting, while stochastic CoT has advantages for approximate counting and sampling under their stated assumptions. Their deterministic comparison allows the embedding dimension to grow with input length. Our result is complementary: for a fixed Turing machine and hierarchy parameter, we construct a family with fixed machine-dependent learned size and precision in which varying � exchanges explicit CoT steps for latent iterations. We account for generated tokens, visible context, serial loop work, and persistent per-iteration KV state under the loaded-cache interface. This is an interpolation for the constructed recurrent simulation, not an equivalence of arbitrary CoT and latent-thought models or a claim that all intermediate tokens can be eliminated.

Autoregressive Turing-machine simulations. Autoregressive transformers are Turing-complete [Pérez et al., 2021, Bhattamishra et al., 2020, Merrill and Sabharwal, 2024, Li et al., 2024, Qiu et al., 2025], with precisions and dimensions ranging from unbounded to constant. Jiang et al. [2026] establish softmax Turing completeness via counter machines with full-history attention and no fixed numerical-precision bound. Two later lines of work bring the context down to the space bound. Context-management schemes erase or summarize intermediate CoT: PENCIL [Yang et al., 2025] achieves maximal context $O ( s ( n ) )$ , and Cui et al. [2026] show that the management policy itself determines the computable class; Brösamle and Eckstein [2026] keep every segment at $O ( s ( n ) )$ with low-precision softmax, at logarithmically growing depth, width, and precision. Sliding-window attention uses locality in place of erasure: Schuurmans et al. [2024] gave the earliest bounded-window universality, with a circular-queue context; Li and Wang [2025a] prove constant-bit-size Turing completeness for an $O ( s ( n ) )$ window via a Post-machine simulation and characterize $\mathsf { W I N D O W } [ s ( n ) ] = \mathsf { S P A C E } [ s ( n ) ]$ ; Li et al. [2026] remove the need for positional encodings; and Li and

Wang [2025b] reach $O ( s ( n ) ^ { 1 / k } )$ CoT tokens per TM step, with window $s ( n ) + 1$ , by increasing the head–layer product. Our construction starts from that last simulation and folds � of its transitions into each token by making the queue representation layer-aligned (Section 4.5). At � = 1 it recovers that simulation’s bounds.

## 7 Discussion

The following limitations bound the result.

• Loaded cache. The input is pre-loaded into the initial KV cache (Section 4.3); the initialization cost is excluded from all bounds, and ordinary prompt prefill is not proved to produce the required entries.

• Visible resources only. The factor-� reduction applies to generated tokens and to the visible window only; total cached KV state and serial block work are unchanged from � = 1.

• Kernel-conditional softmax. The softmax result holds only under the max-first binary32 contract of Appendix C, for values exactly representable in the accumulation and output formats; it does not extend to exact-real softmax, to kernels with positive floors or logit soft-capping, or to production kernels beyond that contract.

• Generated tokens are machine states. The visible trace is a sequence of internal machine-state symbols at stride �, not a natural-language chain of thought; CoT-length statements concern this trace.

• No learning claims. Nothing here shows that such computations are learned by pretraining, reinforcement learning, or any other training signal; the messages of Section 1 are conjectural, and the supervision question for strided traces is open.

• Bounds, not measurements. All quantitative statements are asymptotic bounds of the construction; no experiment, speedup, or deployment claim is made.

## AI use statement

We used generative AI tools to improve the readability of the manuscript and to identify relevant literature. We also used the Paper Assistant Tool (PAT) to review the paper’s technical content. After assessing its feedback, we corrected errors it identified and cited relevant papers it suggested. The authors reviewed the AI-assisted revisions and take responsibility for the final text, citations, and mathematical claims.

## References

Dana Angluin, David Chiang, and Andy Yang. Masked hard-attention transformers recognize exactly the star free languages. In Proceedings ofthe Conference on Language Modeling (COLM), 2024. arXiv:2310.13897; introduces Boolean RASP (B-RASP).

Satwik Bhattamishra, Arkil Patel, and Navin Goyal. On the computational power of transformers and its implications in sequence modeling. In Proceedings of the 24th Conference on Computational Natural Language Learning (CoNLL), pages 455–475, 2020.

Moritz Brösamle and Stephan Eckstein. The expressive power of low precision softmax transformers with (summarized) chain-of-thought, 2026. arXiv:2605.18079.

Guanyu Cui, Zhewei Wei, and Kun He. Position: The Turing-completeness of real-world autoregressive transformers relies heavily on context management, 2026. arXiv:2605.19514, accepted to the ICML 2026 Position Paper Track.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. In Proceedings of the 7th International Conference on Learning Representations (ICLR), 2019.

Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Da-Wei He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), pages 70757–70798, 2023.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pages 46242–46293, 2025.

Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In Proceedings ofthe 40th International Conference on Machine Learning (ICML 2023), volume 202 of Proceedings of Machine Learning Research, pages 11398–11442, 2023.

Xinting Huang, Andy Yang, Satwik Bhattamishra, Yash Sarrof, Andreas Krebs, Hattie Zhou, Preetum Nakkiran, and Michael Hahn. A formal framework for understanding length generalization in transformers. In Proceedings of the 13th International Conference on Learning Representations (ICLR), 2025. arXiv:2410.02140.

Hongjian Jiang, Michael Hahn, Georg Zetzsche, and Anthony Widjaja Lin. Softmax transformers are Turing-complete. In Proceedings of the 14th International Conference on Learning Representations (ICLR), 2026. arXiv:2511.20038.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In Proceedings of the 8th International Conference on Learning Representations (ICLR), 2020.

Qian Li and Yuyi Wang. Constant bit-size transformers are Turing complete. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pages 69254–69273, 2025a. doi: 10.52202/085713-2083.

Qian Li and Yuyi Wang. Eficient Turing machine simulation with transformers, 2025b. arXiv:2512.00003.

Qian Li, Xinyu Mao, and Shang-Hua Teng. Rethinking the role of positional encoding: Sliding-window transformers without PE remain Turing complete, 2026. arXiv:2606.01532.

Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. In Proceedings ofthe 12th International Conference on Learning Representations (ICLR), 2024. arXiv:2402.12875.

David Lindner, János Kramár, Matthew Rahtz, Thomas McGrath, and Vladimir Mikulik. Tracr: Compiled transformers as a laboratory for interpretability. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023. arXiv:2301.05062.

William Merrill and Ashish Sabharwal. The parallelism tradeof: Limitations of log-precision transformers. Transactions ofthe Associationfor Computational Linguistics, 11:531–545, 2023.

William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought. In Proceedings ofthe 12th International Conference on Learning Representations (ICLR), 2024. arXiv:2310.07923.

William Merrill and Ashish Sabharwal. A little depth goes a long way: The expressive power of logdepth transformers. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), 2025. arXiv:2503.03961.

Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is Turing-complete. Journal of Machine Learning Research, 22(75):1–35, 2021.

Emil L. Post. Finite combinatory processes — formulation 1. Journal of Symbolic Logic, 1(3):103–105, 1936.

Ruizhong Qiu, Zhe Xu, Wenxuan Bao, and Hanghang Tong. Ask, and it shall be given: On the Turing completeness of prompting. In Proceedings ofthe 13th International Conference on Learning Representations (ICLR), 2025. arXiv:2411.01992.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. In Proceedings of the 13th International Conference on Learning Representations (ICLR), 2025. arXiv:2502.17416.

Dale Schuurmans, Hanjun Dai, and Franco Zanini. Autoregressive large language models are computationally universal, 2024. arXiv:2410.03170.

Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. In Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 464–468, 2018.

Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What formal languages can transformers express? A survey. Transactions of the Association for Computational Linguistics, 12: 543–561, 2024.

Lena Strobl, Dana Angluin, David Chiang, Jonathan Rawski, and Ashish Sabharwal. Transformers as transducers. Transactions of the Association for Computational Linguistics, 13:200–219, 2025. doi: 10.1162/tacl\_a\_00736. arXiv:2404.02040; extends B-RASP to transductions.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30 (NIPS 2017), 2017.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pages 24824–24837, 2022.

Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers. In Proceedings of the 38th International Conference on Machine Learning (ICML 2021), volume 139 of Proceedings of Machine Learning Research, pages 9979–9991, 2021.

Kevin Xu and Issei Sato. A formal comparison between chain of thought and latent thought. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. arXiv:2509.25239, camera-ready version.

Andy Yang and David Chiang. Counting like transformers: Compiling temporal counting logic into softmax transformers. In Proceedings ofthe Conference on Language Modeling (COLM), 2024. arXiv:2404.04393; introduces C-RASP.

Chenxiao Yang, Nathan Srebro, David McAllester, and Zhiyuan Li. PENCIL: Long thoughts with short memory. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML 2025), volume 267 of Proceedings of Machine Learning Research, pages 71092–71156. PMLR, 2025.

Liu Yang, Kangwook Lee, Robert D. Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. arXiv:2311.12424.

Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, and Preetum Nakkiran. What algorithms can transformers learn? a study in length generalization. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024. arXiv:2310.16028; introduces RASP-L.

## A Congruent queue lengths

Grouping � consecutive steps of a synchronous �-queue machine into a single block requires that the front of queue � at global step $L i + l$ be the symbol written at the same intra-block ofset of an earlier block. That symbol was written at step $L i + l - \ell _ { r } + 1$ , so the alignment

$$
L i + l - \ell _ { r } + 1 \ = \ L ( i - w _ { r } + 1 ) + l
$$

holds for all � and $0 \leq l < L$ if and only if

$$
\ell _ { r } \equiv 1 { \pmod { L } } , \qquad w _ { r } = { \frac { \ell _ { r } - 1 } { L } } + 1 .
$$

In the conversion of Fact 2.2, level $g$ has two content-half queues of length $B ^ { g }$ and one bufer queue of length $2 B ^ { g }$ , where $B = \lceil s ( n ) ^ { 1 / k } \rceil$ . We split the bufer into two queues of length $B ^ { g }$ , so that all four physical queues per stack at the level have the same length.

Lemma A.1. Prescribe $L = L ( n ) \geq 1$ and choose a fixed integer $k \geq 1$ satisfying $s ( n ) ^ { 1 / k } = \Omega ( L )$ . Let $B = \lceil s ( n ) ^ { 1 / k } \rceil$ and

$$
b = \operatorname* { m i n } \{ b ^ { \prime } \geq \operatorname* { m a x } \{ B , L + 1 \} \ : \ b ^ { \prime } \equiv 1 { \pmod { L } } \} , \qquad \ell _ { g } = b ^ { g } \qquad ( 1 \leq g \leq k ) .
$$

Then the conversion can be implemented with $K = 8 k k ^ { \prime }$ queues, all four queues per stack at level � having length $\ell _ { g }$ . Every queue length satisfies $\ell _ { g } \equiv 1$ (mod $L )$ , and the resulting machine runs in $O ( t ( n ) s ( n ) ^ { 1 / k } )$ steps using $O ( k ^ { \prime } s ( n ) )$ queue space. Moreover,

$$
w _ { r } = \frac { \ell _ { r } - 1 } { L } + 1 \geq 2 , \qquad w = O \left( \frac { s ( n ) } { L } \right) .
$$

Proof. Run the conversion of Fact 2.2 with geometric base �. Replacing � by $b \geq B$ preserves the divisibility between consecutive levels and provides at least the original capacity. Split each length- $2 \ell _ { g }$ bufer, as any queue of length $m _ { 1 } + m _ { 2 }$ can be simulated step-for-step by two queues of lengths $m _ { 1 }$ and $m _ { 2 } ;$ each stack now uses four length $\boldsymbol { \cdot } \boldsymbol { \ell } _ { g }$ queues per level, so a $k ^ { \prime } .$ -tape machine uses $2 \cdot 4 k k ^ { \prime } = 8 k k ^ { \prime }$ queues. The split simulation is step-for-step equivalent to the original one.

By minimality,

$$
b \leq \operatorname* { m a x } \{ B , L + 1 \} + L - 1 = O ( s ( n ) ^ { 1 / k } ) ,
$$

because ${ \cal L } \ = \ { \cal O } ( s ( n ) ^ { 1 / k } )$ . Together with $b \ \geq \ B .$ , this gives $b ~ = ~ \Theta ( s ( n ) ^ { 1 / k } )$ and $\ell _ { k } \ = \ b ^ { k } \ = \ \Theta ( s ( n ) )$ The conversion therefore takes ${ \cal O } ( t ( n ) b ) = { \cal O } ( t ( n ) s ( n ) ^ { 1 / k } )$ steps and $\begin{array} { r } { O ( k ^ { \prime } \sum _ { g = 1 } ^ { k } \ell _ { g } ) = O ( k ^ { \prime } s ( n ) ) } \end{array}$ queue space. Finally, $b \equiv 1$ (mod �) implies $\ell _ { g } = b ^ { g } \equiv 1$ (mod $L )$ , while $b \geq \check { L } + 1$ gives $w _ { r } \geq 2 ;$ hence $w = \Theta ( \ell _ { k } / L ) = O ( s ( n ) / L )$ □

The construction introduces no additional transitions, $\operatorname { s o } \tau ( n ) = O ( t ( n ) s ( n ) ^ { 1 / k } )$ , the run length used in Proposition 4.2 and Theorem 4.3.

## B Transformer components

This appendix specifies the transformer of Theorem 4.3 / Proposition 4.2 in full. Throughout, $M = \left( \Sigma , Q , \delta \right)$ is the congruent synchronous �-queue machine of Proposition 4.2, with a fixed alphabet $\Sigma , | Q | = c$ , halting states normalized as absorbing (Section 2), and lengths $\ell _ { r } = L ( w _ { r } - 1 ) + 1$ with integers $w _ { r } \geq 2 ; w = \operatorname* { m a x } _ { r } w _ { r } .$ $s _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { r } \ell _ { r }$ , so � $L = s _ { \operatorname* { m a x } } - 1 + L$ . The model is the looped transformer of Definition 3.1 with one decoder layer $( p = 1 )$ $H = K$ heads (head � serves queue $r )$ , key dimension $d _ { k } = K$ , value dimension $d _ { \nu } = | \Sigma |$ , and loop count �; the same block, embedding, and output head are reused at every iteration and every pass. Fix an indexing map $\iota : \Sigma  \{ 1 , \ldots , | \Sigma | \}$

Indexing conventions. Coordinates of the residual stream are numbered from 0, as in the table of B.1: coordinate 0 holds the constant, the state block occupies coordinates $1 , \ldots , c$ with state $q$ (states counted from 0) at coordinate $1 + q ,$ , and each symbol or front block $r \in \{ 0 , \ldots , K - 1 \}$ occupies |Σ| consecutive coordinates whose within-block ofsets are the 1-based positions $\iota ( \alpha ) \in \{ 1 , \ldots , | \Sigma | \}$ of symbols �. Coordinates of value space $( \mathbb { R } ^ { d _ { \nu } } )$ and the concatenated head outputs $( \mathbb { R } ^ { | \Sigma | K } )$ are numbered from 1, while coordinates of key space $( \mathbb { R } ^ { d _ { k } } )$ are indexed by the head number $r \in \{ 0 , \ldots , K - 1 \}  \colon e _ { r }$ denotes the standard basis vector with index $r ,$ and $e _ { 0 }$ exists. Matrix entries are written �[output coordinate, input coordinate], using the conventions of the respective spaces. The attention projections take the residual stream as input: thus $W _ { q } ^ { ( r ) } [ r , 0 ] = 1$ reads the constant (residual coordinate 0) into key coordinate $r ,$ while $\begin{array} { r } { W _ { \nu } ^ { ( r ) } [ \iota ( \alpha ) , c + | \Sigma | r + \iota ( \alpha ) ] = 1 } \end{array}$ reads symbol block �, ofset $\iota ( \alpha )$ , into value coordinate �(�).

## B.1 Vocabulary and embedding

$\mathcal { V } = ( \boldsymbol { Q } \times \Sigma ^ { K } ) \cup$ {ACCEPT, REJECT}. The generated token at position � is $( q _ { L i } , \sigma _ { L i } )$ — the state and the �-tuple appended at log index $L i - \mathrm { o r } ,$ at halting, one of the two terminal tokens (B.6). $d = 2 | \Sigma | K + c + 1$ , laid out in blocks:

<table><tr><td>coordinates</td><td>block</td><td>content</td></tr><tr><td>0</td><td>constant</td><td>1</td></tr><tr><td> $1 , \ldots , c$ </td><td>state</td><td>one-hot  $e _ { q }$  of the state  $q$ </td></tr><tr><td> $c + | \Sigma | r + 1 , \ldots , c + | \Sigma | ( r { + } 1 )$ </td><td>symbol r</td><td>one-hot of  $\boldsymbol { \sigma } ^ { ( r ) } \in \Sigma$ </td></tr><tr><td> $c + | \Sigma | K + | \Sigma | r + 1 , \ldots , \ c + | \Sigma | K + | \Sigma | ( r + 1 )$ </td><td>front r</td><td>one-hot of the attention-retrieved front; 0 at embedding</td></tr></table>

emb $\left( q , \sigma \right)$ equals the displayed row of the table above,

$$
\mathrm { e m b } ( q , \sigma ) = \bigl ( 1 , \ e _ { q } , \ \mathrm { o n e h o t } ( \sigma ^ { ( 0 ) } ) , \dots , \mathrm { o n e h o t } ( \sigma ^ { ( K - 1 ) } ) , \ 0 , \dots , 0 \bigr ) ;
$$

terminal tokens take emb $( \mathrm { A C C E P T } ) = \mathrm { e m b } ( q _ { \mathrm { a c c } } , \# ^ { K } )$ and emb $( \mathrm { R E J E C T } ) = \mathrm { e m b } ( q _ { \mathrm { r e j } } , \# ^ { K } )$ (generation stops at either).

## B.2 Relative positional encoding

The relative positional encoding (the form of Shaw et al., 2018; Li and Wang, 2025b, Eq. (2)): for distance $i - j \in \{ 0 , \ldots , w - 1 \}$ ,

$$
\mathrm { p o s } ( i - j ) \ = \ \Delta \sum _ { r : i - j = w _ { r } - 1 } e _ { r } \ \in \ \mathbb { R } ^ { d _ { k } } ,
$$

where $\Delta > 0$ is the fixed constant of Section 4.2 and $e _ { r }$ is the basis vector of head coordinate �. Head � retrieves ofset $w _ { r } - 1$ by the query $e _ { r } , \mathrm { s o } \left. e _ { r } , \mathrm { p o s } ( i - j ) \right. = \Delta [ i - j = w _ { r } - 1 ]$ . The encoding is unlearnable and depends on $( \ell _ { 0 } , \dots , \ell _ { K - 1 } , L )$ only — the learnable parameter vector is one and the same for all input lengths, the analogue of [Li and Wang, 2025b, Remark 2].

## B.3 The attention heads

For head $r \in \{ 0 , \ldots , K - 1 \}$ , the matrices $W _ { q } ^ { ( r ) } , W _ { k } ^ { ( r ) } \in \mathbb { R } ^ { d _ { k } \times d }$ and $W _ { \nu } ^ { ( r ) } \in \mathbb { R } ^ { d _ { \nu } \times d } ;$

• Query ${ W } _ { q } ^ { ( r ) }$ : single nonzero entry $W _ { q } ^ { ( r ) } [ r , 0 ] = 1$ , copying the constant coordinate of ℎ onto key coordinate �. Then $W _ { q } ^ { ( r ) } h _ { i } ^ { ( l ) } = e _ { r }$ for every token, every iteration.

• Key $W _ { k } ^ { ( r ) } = 0 :$ the keys carry no content; the positional encoding pos $( i - j )$ of B.2 enters the key side of the scores (Section 3). In the cache only the value entries are informative.

• Score: score $\begin{array} { r } { \cdot ( i , j ) = \langle W _ { q } ^ { ( r ) } h _ { i } ^ { ( l ) } , ~ W _ { k } ^ { ( r ) } h _ { i } ^ { ( l ) } + \mathrm { p o s } ( i - j ) \rangle = \langle e _ { r } , \mathrm { p o s } ( i - j ) \rangle = \Delta [ i - j = w _ { r } - 1 ] } \end{array}$ Over the window $\{ i - w + 1 , \ldots , i \}$ this is $\Delta$ at the single position $j ^ { * } = i - w _ { r } + 1$ (in range at every pass, since $i \geq w - 1$ and $w _ { r } \leq w )$ and 0 elsewhere. Self-attention has score 0 because $w _ { r } \geq 2$ . The unique maximum has gap Δ (Section 4.2), so hardmax returns the one-hot vector $s _ { r , i , \cdot } ^ { ( l ) } = \mathbf { 1 } _ { j ^ { \ast } }$

• Value $W _ { \nu } ^ { ( r ) } \colon W _ { \nu } ^ { ( r ) } [ \iota ( \alpha ) , c + | \Sigma | r + \iota ( \alpha ) ] = 1$ for $\alpha \in \Sigma -$ the projection onto symbol block �. With the write-time invariant (Lemma 4.1), $\bar { W } _ { \nu } ^ { ( r ) } h _ { j } ^ { ( l ) } = \mathrm { o n e h o t } \big ( \sigma _ { L j + l } ^ { ( r ) } \big )$ , hence the head output

$$
a _ { i , r } ^ { ( l ) } ~ = ~ \mathrm { o n e h o t } \Big ( \sigma _ { L ( i - w _ { r } + 1 ) + l } ^ { ( r ) } \Big ) ~ = ~ \mathrm { o n e h o t } \Big ( \sigma _ { L i + l - \ell _ { r } + 1 } ^ { ( r ) } \Big ) ,
$$

the front of queue � at machine step $L i + l -$ the retrieval identity, using $\ell _ { r } - 1 = L ( w _ { r } - 1 )$

## B.4 Head combination

concat<sub>�</sub> $a _ { i , r } ^ { ( l ) } \in \mathbb { R } ^ { | \Sigma | K } ; W _ { O } \in \mathbb { R } ^ { d \times | \Sigma | K }$ with $W _ { \cal O } [ c + | \Sigma | K + | \Sigma | r + \iota ( \alpha ) , | \Sigma | r + \iota ( \alpha ) ] = 1$ for every � and $\alpha \in \Sigma$ , all other entries $0 ; b = 0$ . The residual update $\tilde { h } _ { i } ^ { ( l ) } = h _ { i } ^ { ( l ) } + W _ { O }$ concat writes the � retrieved fronts into the front blocks; the constant, state, and symbol blocks pass through unchanged.

## B.5 Feed-forward network: the transition

FF is a two-layer ReLU network of width $| Q | \cdot | \Sigma | ^ { K } + O ( d )$ (a machine constant) realizing $\mathrm { F F } ( \tilde { h } ) + \tilde { h } =$ $( 1 , e _ { q ^ { \prime } }$ , onehot $( \sigma ^ { \prime } ) , 0 , \ldots , 0 )$ where $( q ^ { \prime } , \sigma ^ { \prime } ) = \delta ( q , f )$ , with $q$ in the state block, $\sigma$ in the symbol blocks, and $f$ in the front blocks of $\tilde { h } .$ . For the transition lookup, define

$$
\begin{array} { r } { u _ { ( \hat { q } , \hat { f } ) } ( \tilde { h } ) \ = \ 2 \mathrm { R e L U } \Big ( \tilde { h } _ { 1 + \hat { q } } + \sum _ { r } \tilde { h } _ { c + | \Sigma | K + | \Sigma | r + \iota ( \hat { f } ^ { ( r ) } ) } - ( K + 1 ) + \frac { 1 } { 2 } \Big ) \ \in \ \{ 0 , 1 \} , } \end{array}
$$

equal to 1 exactly on the table row $( \hat { q } , \hat { f } )$ matching $\tilde { h } ;$ then

$$
\begin{array} { l } { { \mathrm { F F } ( \tilde { h } ) \ = \ \displaystyle \sum _ { ( \hat { q } , \hat { f } ) } u _ { ( \hat { q } , \hat { f } ) } ( \tilde { h } ) \cdot \big ( 0 , \ e _ { \delta _ { 1 } ( \hat { q } , \hat { f } ) } , \ \mathrm { o n e h o t } ( \delta _ { 2 } ( \hat { q } , \hat { f } ) ) , \ 0 ^ { | \Sigma | K } \big ) } } \\ { { \ ~ - \ \big ( 0 , \ e _ { q } , \ \mathrm { o n e h o t } ( \sigma ) , \ \mathrm { o n e h o t } ( f ) \big ) , } } \end{array}
$$

where onehot $( f )$ is the concatenation of the � one-hot front blocks. The linear subtraction is implemented inside the two-layer ReLU network by $O ( d )$ passthrough units: every relevant coordinate of $\tilde { h }$ is nonnegative on the invariant domain, so $\mathrm { R e L U } ( \tilde { h } _ { a } ) = \tilde { h } _ { a }$ . Coordinate 0 is left unchanged. Weights and biases belong to a fixed machine-dependent set, including the threshold bias $- ( K + { \frac { 1 } { 2 } } )$ ; total parameter bit-size is $O ( ( | Q | | \Sigma | ^ { K } + d ) d )$ constant for a fixed machine and independent of � and �. On invariant states, all intermediate sums range over a fixed finite set of half-integers, so a fixed machine-dependent numerical format sufices for exact evaluation. The absorbing convention for halting states (Section 2) is part of $\delta ;$ mid-pass halting needs no extra machinery.

## B.6 Output head

$W _ { \mathrm { o u t } } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ . For each $( q , \sigma ) \in Q \times \Sigma ^ { K } \colon W _ { \mathrm { o u t } } [ ( q , \sigma ) , 1 + q ] = 1 \mathrm { a n d } W _ { \mathrm { o u t } } [ ( q , \sigma ) , c + | \Sigma | r + \iota ( \sigma ^ { ( r ) } ) ] = 1$ for every $r ;$ terminal rows $W _ { \mathrm { o u t } } [ \mathrm { A C C E P T } , 1 + q _ { \mathrm { a c c } } ] = W _ { \mathrm { o u t } } [ \mathrm { R E J E C T } , 1 + q _ { \mathrm { r e j } } ] = K + 2$ . Since $h _ { i } ^ { ( \bar { L } ) }$ has exactly one nonzero per one-hot block, the ordinary row of the emitted token scores $K + 1$ , every other ordinary row less, and the terminal rows score $K + 2$ exactly when the state block is halting (and 0 otherwise): the argmax is $( q _ { L ( i + 1 ) } , \sigma _ { L ( i + 1 ) } )$ at non-halting passes and ACCEPT/REJECT at halting ones. Entries are $O ( K )$ , constant for a fixed machine.

## B.7 The loop and the input interface

One layer per unit (Definition 3.1 with $p = 1 ) \colon h _ { i } ^ { ( l + 1 ) } = \mathrm { F F } ( \tilde { h } _ { i } ^ { ( l ) } ) + \tilde { h } _ { i } ^ { ( l ) }$ for $l = 0 , \ldots , L - 1$ , the matrices of B.1–B.6 identical at every iteration. The per-iteration KV cache (Section 3) holds, for each position $j$ and iteration $l ,$ the value entries $\begin{array} { r } { W _ { \nu } ^ { ( r ) } h _ { j } ^ { ( l ) } = \operatorname { o n e h o t } ( \sigma _ { L j + l } ^ { ( r ) } ) , r = 0 , \dots , K - 1 - } \end{array}$ the keys are zero (B.3), the positional encoding entering only the scores (B.2). The input interface is the initialization of Section 4.3: the value entries of positions $0 , \ldots , w - 2$ at all iterations contain the �-tuples $\sigma _ { 0 } , \ldots , \sigma _ { s _ { \mathrm { m a x } } - 2 }$ of the log, and the current token at position $w - 1$ is $( q _ { s _ { \mathrm { m a x } } - 1 } , \sigma _ { s _ { \mathrm { m a x } } - 1 } )$ . Generation starts at pass $w - 1$

## B.8 Faithfulness

Proposition B.1. Under the initialization of Section 4.3, the transformer of B.1–B.7 emits, at each pass $i \geq w - 1$ , the token $( q _ { L ( i + 1 ) } , \sigma _ { L ( i + 1 ) } )$ , and after $\lceil \tau ( n ) / L \rceil$ passes the terminal token matching �’s verdict on �; there is no other behavior.

Proof. The proof of Lemma 4.1 (Section 4.3) rests on two structural facts that hold verbatim for the components above: (a) within a pass, state moves only through the residual stream — the attention reads same-iteration value entries only, and each iteration’s feed-forward network consumes the fronts its attention wrote; (b) value entries are frozen at write time. With the base case supplied by the initialization of Section 4.3 and the retrieval identity of B.3, the two-stage induction of Section 4.3 goes through unchanged; the output head of B.6 turns the final residual into the token or the verdict, and the absorbing convention makes mid-pass halting consistent. □

## C The exact softmax kernel contract

This appendix records the kernel contract of Section 5 exactly, the arithmetic behind the −256 threshold, and the two kernel variants. Every attention score is a finite binary32 value (IEEE single precision; $\mathrm { R N } _ { 3 2 }$ denotes round-to-nearest in binary32). Given a score row $( a _ { j } ) _ { j }$ and value vectors $( \nu _ { j } ) _ { j }$ , the kernel computes the attention output as follows:

• (i) compute the row maximum $\mu = \operatorname* { m a x } _ { j } a _ { j } ;$

• (ii) form the shifted inputs $\hat { a } _ { j } = \mathrm { R N } _ { 3 2 } ( a _ { j } - \mu ) ;$

• (iii) apply a fixed binary32 exponential primitive $E _ { 3 2 }$ with $E _ { 3 2 } ( 0 ) = 1$ and $E _ { 3 2 } ( x ) = + 0$ for every $x \leq - 2 5 6 ;$

• (iv) accumulate the numerator $\begin{array} { r } { \sum _ { j } E _ { 3 2 } ( \hat { a } _ { j } ) \nu _ { j } } \end{array}$ and the normalizer $\textstyle \sum _ { j } E _ { 3 2 } ( { \hat { a } } _ { j } )$ in binary32, dividing only after both sums;

• (v) cast the quotient once, at the output boundary, to a fixed format containing 0 and 1 exactly.

The exactness of Lemma 5.1 additionally requires the winner’s value $\nu _ { j ^ { * } }$ to be exactly representable in the accumulation format and preserved by the boundary cast; the fixed format of (v) contains 0 and 1 exactly, which covers the one-hot values used in this construction (Corollary 5.2), and not arbitrary real value vectors.

The threshold. Since $2 5 6 > 1 5 0 \ln { 2 } .$ we have $e ^ { - 2 5 6 } < 2 ^ { - 1 5 0 }$ , the round-to-nearest midpoint between 0 and the least positive binary32 subnormal $2 ^ { - 1 4 9 }$ , so a correctly rounded binary32 exponential returns +0 on every input at most −256. Flush-to-zero only strengthens the property. Any cutof larger than 150 ln 2 in magnitude has the same property; $2 5 6 = 2 ^ { 8 }$ is chosen so that the rescaling factor of Corollary 5.2 is a power of two.

Remark. (1) The lemma requires a binary32 exponential that returns +0 below a fixed threshold, without positive floor or logit soft-capping. (2) Subtracting the maximum converts overflow into underflow and thereby helps to simulate hard selection. We include it in our softmax kernel because it is standard in production implementations.<sup>1</sup> For a kernel that does not subtract the maximum, the positional encoding of Section 4.2 can set the winner’s score to 0 and every other score to −256. (3) If the kernel includes a positive scale, canonically $1 / \sqrt { d _ { k } }$ , instantiate $\Delta = \lceil 2 5 6 \sqrt { d _ { k } } \rceil$ . Here $d _ { k } = K$ is a machine constant, so Δ remains $O ( 1 )$

## D Detailed comparison of Turing-machine simulations

Table 3 compares transformers simulating (�, �)-bounded Turing machines, extending the pointer given in Section 6.

Notation normalization. The rows “Li & Wang 2025a” and “Li & Wang 2025b” abbreviate Li and Wang [2025a] and Li and Wang [2025b], respectively; the row “Jiang et al. $2 0 2 6 ^ { \circ }$ carries the venue year of Jiang et al. [2026]. Depth is printed as � for this work, where the source of each competitor row uses its own layer-count symbol; the main text likewise uses � for the generic decoder depth, which coincides with the loop count for the one-layer loop unit. This work’s window and CoT bounds presuppose the admissibility condition $s ( n ) ^ { 1 / k } = \Omega ( L )$ and the loaded-cache interface of Section 4.3; the initialization cost is excluded from this work’s bounds (the comparison rows report the sources’ own generation accounting).

<table><tr><td>Source</td><td>Precision</td><td>Dim.</td><td>Depth</td><td>Window</td><td>CoT per TM-step</td></tr><tr><td>Pérez et al. 2021</td><td>O(log t)</td><td>0(1)</td><td> $O ( 1 )$ </td><td> $n + t$ </td><td>1</td></tr><tr><td>Bhattamishra et al. 2020</td><td>unbounded</td><td>0(1)</td><td>1</td><td> $n + t$ </td><td>1</td></tr><tr><td>Merrill &amp; Sabharwal O(log t) 2024</td><td></td><td>0(1)</td><td>0(1)</td><td> $n + t$ </td><td>1</td></tr><tr><td>Li et al. 2024</td><td>0(1)</td><td> ${ \cal O } ( \log t ) ^ { \ddag }$ </td><td>2</td><td>O(t log t)</td><td>O(log t) amort.</td></tr><tr><td>Qiu et al. 2025</td><td>O(log t)</td><td>0(1)</td><td>0(1)</td><td> $O ( t \log t )$ </td><td>O(log t) amort.</td></tr><tr><td>Li &amp; Wang 2025a</td><td>0(1)</td><td>0(1)</td><td>1</td><td>S</td><td>S</td></tr><tr><td>Li &amp; Wang 2025b</td><td>0(1)</td><td>0(1)</td><td>_†</td><td>s+1</td><td> $O ( s ^ { 1 / k } )$ </td></tr><tr><td>Jiang et al. 2026°</td><td>unbounded</td><td>0(1)</td><td>0(1)</td><td> $n + 2 ^ { O ( t ) }$ </td><td>1 per counter step</td></tr><tr><td>Brösamle &amp; Eckstein O(log log s)</td><td></td><td>O(log s)</td><td>O(log s)</td><td> $O ( s ) \ ( \mathrm { S C o T } )$ </td><td>O(1) amort.</td></tr><tr><td>2026 This work</td><td>0(1)</td><td>0(1)</td><td>L</td><td> $O ( s / L )$ </td><td> ${ \cal O } ( s ^ { 1 / k } / L )$ </td></tr></table>

marks constructions using softmax attention (this work via the finite-precision kernel of Section 5).  
<sup>†</sup> Li & Wang 2025b report only the head–layer product $K = H \cdot L ;$ their depth is left unspecified.  
<sup>‡</sup> Li et al. [2024] state � (log �) embedding dimension under the assumption that �(�) is polynomial, so $\lceil \log _ { 2 } ( t ( n ) + n ) \rceil = O ( \log n )$

Table 3: Comparing transformers simulating (�, �)-bounded Turing machines. Depth is the number of sequential decoder layers as reported by each source; for this work it is the loop count �—one shared decoder block executed � times per generated token. This work prescribes � and chooses � subject to $s ( n ) ^ { 1 / k } = \Omega ( L )$ Li & Wang 2025b is the endpoint �=1.

## E Notation

<table><tr><td>Symbol</td><td>Formally defined Meaning</td><td></td></tr><tr><td> $t ( n ) , s ( n )$ </td><td>2</td><td>time and space bounds of the simulated Turing machine</td></tr><tr><td> $k , k ^ { \prime }$ </td><td>2</td><td>fixed hierarchy exponent  $( s ( n ) ^ { 1 / k } = \Omega ( L ) )$  ; tape count</td></tr><tr><td>K</td><td>2.2, A</td><td>queue  ${ \tt c o u n t } = 8 k k ^ { \prime } ;$  number of attention heads</td></tr><tr><td>L</td><td>3.1</td><td>loop count, and the decoder depth: one shared block executed L times per token</td></tr><tr><td> $m$ </td><td>2.1</td><td>execution-log step index</td></tr><tr><td> $g$ </td><td>2.2</td><td>hierarchy level index  $( 1 \leq g \leq k )$ </td></tr><tr><td> $( \sigma _ { m } , q _ { m } )$ </td><td>2.1</td><td>execution log: K-tuple appended and state at step m</td></tr><tr><td> $\tau ( n )$ </td><td>2.1</td><td>run length after the cache is filled</td></tr><tr><td> $T _ { \mathrm { { c o t } } }$ </td><td>4.2</td><td>generated tokens; exactly  $\lceil \tau ( n ) / L \rceil$  , asymptotically  $O ( t ( n ) s ( n ) ^ { 1 / k } / L )$ </td></tr><tr><td> $i , l$ </td><td>3.1,4</td><td>pass (position) and iteration indices</td></tr><tr><td> $\ell _ { r } , w _ { r } , w$ </td><td>2.1, A</td><td>length of queue  $r ; w _ { r } = ( \ell _ { r } - 1 ) / L + 1 ; w = \mathrm { m a x } _ { r } w _ { r }$ </td></tr><tr><td> $s _ { \mathrm { m a x } }$ </td><td>2.1</td><td> $\operatorname* { m a x } _ { r } \ell _ { r }$ </td></tr><tr><td> $b , B$ </td><td>2.2, A</td><td>congruent queue base;  $B = \lceil s ( n ) ^ { 1 / k } \rceil$  (cited conversion)</td></tr><tr><td> $d , d _ { k } , d _ { \nu } , c$ </td><td>4.1, B.1</td><td>residual dimension  $2 | \Sigma | K + | Q | + 1 ;$  key dim.  $K ;$  value dim. |Σ|; |Q|</td></tr><tr><td> $a _ { i , j }$ </td><td>5</td><td>attention score (row i, candidate j)</td></tr><tr><td>Δ</td><td>4.2</td><td>score gap; set to 256 in Corollary 5.2</td></tr><tr><td> $\mu , \hat { a } _ { j } , E _ { 3 2 }$ </td><td>C</td><td>kernel row maximum; shifted input; binary32 exponential</td></tr><tr><td>p</td><td>3.1</td><td>layers per loop unit  $( p = 1$  in the construction)</td></tr></table>