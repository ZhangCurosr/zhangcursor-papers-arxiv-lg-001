# COMPUTATIONS OF THE SLICE GENUS AND THE UNKNOTTING NUMBER OF LINKS VIA MACHINE LEARNING

YUTONG DAI, OLIVER HAYMAN, ANDRÁS JUHÁSZ, AND LUDOVICO MORELLATO

Abstract. Links are disjoint unions of circles smoothly embedded in S<sup>3</sup>. We use reinforcement learning and Bayesian optimisation to obtain new upper bounds on several link invariants that are not known to be algorithmically computable: the slice genus and the unknotting number for links, and the strong slice genus for algebraically split links. We also compute lower bounds using known invariants. Combining the upper and lower bounds, we obtain new exact values in many cases. Our unknotting agents can reproduce the non-additivity of the unknotting number for several counterexamples due to Brittenham and Hermiller, in some cases finding new unknotting trajectories.

## 1 Introduction

The unknotting number and the smooth slice genus are topological invariants of links that are not known to be algorithmically computable, as they are defined as minima over infinite sets. The unknotting number of a link is the minimal number of crossing changes needed to turn a diagram of the link into a diagram of the unlink. The dificulty is that every link admits infinitely many diagrams. For example, out of the 249 prime knots with up to ten crossings, we do not know the unknotting number of ten.

The (weak) slice genus of a link L is the minimal genus of a compact, connected, oriented surface in the 4-ball that bounds L. If the components of L are $K _ { 1 } , \ldots , K _ { \ell } .$ , the strong slice genus of L is the minimum of $\textstyle \sum _ { i = 1 } ^ { \ell } g ( F _ { i } )$ , where $F _ { 1 } , \ldots , F _ { \ell }$ are pairwise disjoint compact, connected, oriented surfaces in the 4-ball with $\partial F _ { i } = K _ { i }$ . Such a collection of surfaces exists if and only if the link is algebraically split; i.e., lk $( K _ { i } , K _ { j } )$ = 0 for i = j. Surfaces bounded by L in the 4-ball can be represented by movies starting with L and ending with the empty link that consist of isotopies, births and deaths of unlinked unknot components, and band attachments. The set of possible movies is infinite. If we disallow births of unlinked unknot components, then we obtain what is known as a ribbon surface, and minimising the genus over such surfaces gives rise to the notions of the weak and strong ribbon genera. The ribbon genus is at least the slice genus.

The application of machine learning to knot-theoretic problems has developed rapidly over the last decade. On the supervised side, Hughes [Hug20] trained neural networks to predict knot invariants directly from a diagram, and Jejjala, Kar, and Parrikar [JKP19] to recover the hyperbolic volume of a knot from its Jones polynomial. More broadly, Davies et al. [Dav+21] used machine learning feature attribution techniques to uncover previously unknown relationships between invariants, leading to new conjectures in knot theory. On the reinforcement learning side, several works have framed the simplification of a knot as a sequential decision problem. Gukov, Halverson, Ruehle, and Sułkowski [Guk+21] trained an agent to recognise and untie the unknot via Markov moves and braid relations, while Applebaum et al. [App+25] used a reinforcement learning agent to bound the unknotting number, producing along the way a large dataset of hard unknot diagrams.

Closer to our work, Gukov, Halverson, Manolescu, and Ruehle [Guk+25] looked for band moves exhibiting knots as ribbon. They framed the search for a ribbon disc as a game aimed at finding a band path very close to the one we describe in Section 3. The present work follows this line, unifying the computation of the unknotting number and the weak and strong slice genera (including the ribbon versions) into a single band path framework. To obtain upper bounds on the unknotting number and the various slice genera, we implemented three agents: a random walker tuned via Bayesian optimisation, a neural network trained with reinforcement learning, and a mixed agent that combines the two. While in the work of Gukov et al. the random walker outperformed the RL agent, the opposite holds in our case. The novelty of our work is that we study links instead of just knots and all the aforementioned variants of the slice genus, not only ribbonness (i.e., whether a knot has ribbon genus zero). Furthermore, our unified framework allows us to bound the unknotting number.

While finding an upper bound on the unknotting number or slice genus involves a search over an infinite space of moves, verifying that a proposed sequence achieves a claimed upper bound is fast. We phrase the computation of both invariants as a single combinatorial game on a diagram, and we let an agent play it. A won episode returns a certificate: an explicit sequence of moves that provides a verifiable upper bound for the invariant. The environment never predicts an invariant; it produces a sequence of moves that one can verify by hand. We also compute existing lower bounds obtained via invariants such as various signatures, the invariants τ and ν due to Ozsváth and Szabó, and the Rasmussen s-invariant. When the upper bound obtained via an agent matches the resulting lower bound, the invariant is determined exactly. The lower bounds we match against are computed independently of the agents, using the results collected in Appendix B.

The numerical results we obtained with the trained reinforcement learning agents and the tuned random walkers can be found in Section 5. Prior to our work, no unknotting number data were available for the 1,007 unoriented 11-crossing, at least two-component prime links. We determine the exact value for 690 and obtain lower and upper bounds for the remaining ones.

When studying the slice genus, allowing the birth of unlinked unknot components did not improve our upper bounds, which is consistent with the higher-genus slice–ribbon conjecture stating that the slice and ribbon genera agree. As the slice genus is at most the ribbon genus and all the invariants that we compute are lower bounds on the slice genus, our results apply to both the slice and ribbon genera. Hence, below, we report only on the slice genus. Of all oriented, at least two-component prime links with up to 11 crossings, the weak slice genus was previously undetermined for 407 links, and the strong slice genus for 732 algebraically split links. We determine the weak slice genus for 144 of them. The remaining 263 turn out to all have weak slice genus zero or one. We determine the strong slice genus for 676 algebraically split links and strictly improve the bounds for the remaining 56. Some of our results have already been incorporated into the [LinkInfo] dataset.

We also studied the slice genus and unknotting number of 14-crossing prime knots, of which there are 46,972. Dunfield and Gong [DG25] showed 1,194 have slice genus zero, and 408 have slice genus at least one. We determine a further 44,838 values exactly and strictly narrow the interval for 940 more, bringing the census to 46,032 exact values. For the unknotting number, the upper bounds of all 14-crossing prime knots come from [Brittenham], and the new contribution is purely on the lower bound. The obstructions give a non-trivial lower bound for 1,288 knots, of which 1,097 become exactly determined, with the new bound meeting Brittenham’s upper bound.

When starting this project in 2023, our main motivation for developing the unknotting agents was to find counterexamples to the additivity of the unknotting number. The knots K and $K ^ { \prime }$ are called symbionts if $u ( K \sharp K ^ { \prime } ) < u ( K ) + u ( K ^ { \prime } )$ . The first symbionts were produced by Brittenham and Hermiller [BH25; BH26]. We tested the unknotting agents on some of their connected sums; see Section 5.5. The agents were able to independently reproduce that $u ( K \sharp \mathfrak { s } _ { 1 0 } ) \le 4$ for ten symbionts K of $9 _ { 1 0 } .$ , and that $u ( 1 0 _ { 4 1 } \sharp \mathrm { m } ( 9 _ { 1 0 } ) ) \le 4 ;$ in each case, the two summands have unknotting numbers 2 and 3, adding up to 5. None of these connected sums appear in the datasets we used to train, tune, or test the agents. Four of the sequences found by the agents reproduce almost exactly the unknotting sequence in the proof of [BH26, Theorem 1.1], and a fifth reproduces its mirror image; the others are original. Two of them pass through $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$ and then through K15n4005, which does not appear in the sequence of Brittenham and Hermiller. This gives a second unknotting sequence of three crossing changes for $\mathsf { 4 } _ { 1 } \sharp \mathsf { 9 } _ { 1 0 }$ . The new unknotting sequences are depicted in Appendix C.

## Data availability

The code, the datasets, the trained agents, and the evaluations are available at

https://github.com/andras-juhasz/slice-genus-unknotting-rl/releases/tag/v1.0-preprint

## AI declaration

The first and second authors wrote large parts of the codebase and did not use AI to assist with coding. The third author used ChatGPT 5.6 and 6 Pro to proofread the manuscript, audit the code, and fix minor bugs. The fourth author used Claude Opus 5 to proofread the manuscript and assist with code development.

## 2 Basics: links, diagrams, the band game and RL

We work in the 3-sphere $S ^ { 3 }$ . A smooth ℓ-component link in $S ^ { 3 }$ is a smooth embedding

$$
L \colon \bigcup _ { i = 1 } ^ { \ell } S ^ { 1 } \times \{ i \} \hookrightarrow S ^ { 3 } ,
$$

and a knot is a 1-component link. We usually identify L with its image, which is a smooth, oriented one-dimensional submanifold of $S ^ { 3 }$ , and we denote its connected components by $K _ { 1 } , \ldots , K _ { \ell } .$ Links are regarded up to ambient isotopy, that is, up to a smooth family of difeomorphisms of $S ^ { 3 }$ starting at the identity and carrying one link onto the other.

Although links live in $S ^ { 3 }$ , they can be faithfully encoded by 2-dimensional pictures, called link diagrams. A diagram D of L is a projection of L onto a plane, which is an immersion with only transverse double points, called crossings, at each of which we record which of the two strands lies above the other. The over/under decoration is exactly what is needed to recover L from D up to ambient isotopy, so a diagram encodes the link up to equivalence. By Reidemeister’s theorem [Cro04, Theorem 3.8.1], two diagrams represent equivalent links if and only if one can be transformed into the other via a sequence of planar isotopies and Reidemeister moves, of which there are three.

Diagrams are also convenient from a computational perspective, since they admit a compact encoding known as a PD code. Let D be an oriented link diagram with n crossings. Label each crossing with an integer from 1 to n. Then start from a crossing and walk along the link following the orientation, labelling each segment between two crossings with integers starting from 1 in increasing order, and repeating for every component, always increasing the segment labels. The resulting Planar Diagram code (PD code) is the ordered list of n 4-tuples, where the i-th tuple corresponds to the i-th crossing and encodes the labels of the four segments around the crossing, starting from the incoming under-strand and going around counter-clockwise. The isotopy type of a connected link diagram in $S ^ { 2 }$ is uniquely determined by its PD code. In $\mathbb { R } ^ { 2 } .$ the only ambiguity when reconstructing a connected diagram from a PD code is the choice of the unbounded region. A given diagram admits several PD codes, depending on the choice of labels. For multi-component links with few crossings on some components, it might happen that the orientation is not uniquely determined, and this problem is avoided by adding the data of crossing signs ( 1) to the PD code. A PD code for the trefoil can be seen in Figure 1.

![](images/8e75f3e8c90726e7dc5f4b6dc814b74651dee0f24f9e7df4091599ec25b31b57.jpg)  
Figure 1. A PD code of the trefoil is [(6, 3, 1, 4), (4, 1, 5, 2), (2, 5, 3, 6)], where the crossings are labelled as Ci for $i \in \{ 1 , 2 , 3 \}$

## 2.1 The unknotting number and the band path

The unknotting number, denoted by $u ,$ is a numerical invariant measuring how knotted a link is. To define it, we need the notion of crossing change, an elementary move that lets the link pass through itself. It is a regular homotopy of the link which is an embedding except for a single time when there is exactly one transverse double point, where two strands of the link pass through each other. $\mathrm { B y }$ iterating this move finitely many times, any link can be turned into the unknot; see [Ada04, Section 3.1]. We are interested in the minimum number of times we need to perform this move.

Definition 2.1 (Unknotting number). Let L be a link in $S ^ { 3 }$ with $\ell \geq 1$ components. The unknotting number $u ( L )$ is the minimum number of crossing changes required to turn L into the ℓ-component unlink.

For $\ell \geq 2 ,$ the unknotting number is also called “unlinking number”. We will call it the unknotting number for both knots and links. Computing this value is very dificult, and no general algorithm is currently known. Even though the problem is 3-dimensional in nature, we can recast it in the completely two-dimensional framework of link diagrams. Given a crossing c of a diagram $D ,$ by a crossing change at $c ,$ we mean the local modification that swaps which of the two strands at $c$ lies above the other, leaving the rest of $D$ unchanged. Every diagram D can be turned into a diagram of an unlink by changing at most $c ( D )$ crossings, where $c ( D )$ is the number of crossings of $D .$ Write $u ( D )$ for the minimal number of crossing changes needed to obtain the unlink from $D .$ This is only an upper bound on $u ( L )$ , as a diferent diagram may admit a shorter sequence. Indeed, for any link $L ,$ one has

$$
u ( L ) = \operatorname* { m i n } _ { D \underset { \mathrm { f o r } L } { \mathrm { d i a g r a m } } } u ( D ) .
$$

While $u ( D )$ can be computed by changing all subsets of crossings of size at most $c ( D ) / 2$ (which is still computationally expensive), every link has infinitely many diagrams, and there are links for which the minimum is not achieved by a minimal crossing number diagram.

To remove this dependence on a specific diagram while still working with combinatorial moves in dimension 2, we package the data of “deforming first, then performing a crossing change” into a single elementary object called a band path.

Definition 2.2 (Band path). A band path on a diagram of L consists of

an edge of the diagram between two neighbouring crossings of $L ;$

a sequence of crossing-increasing Reidemeister II moves we want to perform on that edge, possibly interleaved with Reidemeister I moves. This is the “path” along which we slide the chosen edge to reach a target edge of the diagram.

At the end of the band path, we are going to perform the crossing change at one of the two new crossings created by the last Reidemeister II move. An example can be seen in Figure 2.

![](images/8f1741776c746d2dbb1f1cd91c521d08250e8e38ca98a101fc69104010f8d222.jpg)  
Figure 2. The trefoil has unknotting number at most 1: movie with a band.

A crossing change on a diagram can be represented by a band path. Indeed, we can perform a single Reidemeister II move (and possibly one Reidemeister I move), where the initial and target edges meet at the crossing we want to change, followed by changing one of the two new crossings and then locally simplifying using Reidemeister moves. Again, see Figure 2. Therefore, from now on, when we say “we perform a crossing change”, we actually mean that we choose a band path and change one of the two crossings created by the final Reidemeister II move. The unknotting number $u ( L )$ of a link L is the minimal number of band paths needed to turn a diagram of L into a diagram of the unlink. This allows us to work with a fixed diagram instead of having to explore all diagrams of L. However, there are infinitely many band paths. Our implementation only allows paths that do not intersect themselves and visit each edge, including the initial and target edges, at most once (visiting the same component of the complement of the diagram multiple times is allowed).

Remark 2.3 (Orientation independence of $u ( L ) )$ ). The unknotting number is defined purely in terms of crossing changes, and a crossing change simply swaps the overstrand and the understrand at a crossing, independently of any orientation. Therefore, $u ( L )$ is well defined for unoriented links. This is in sharp contrast with many of the invariants we will use to bound $u ( L )$ in Appendix B, which do require a choice of orientation.

## 2.2 The slice genus

A central theme in low-dimensional topology is the study of surfaces bounded by a given knot. Seifert [Sei35] gave an algorithm producing, for any knot $K$ , a compact, connected, oriented surface $S \subset S ^ { 3 }$ with $\partial S = K$ . The minimal genus of such a surface is an invariant of $K ,$ called the Seifert genus of K and denoted by $g _ { 3 } ( K )$ . Both the algorithm and the definition apply verbatim to a link L.

Let $B ^ { 4 } = \{ x \in \mathbb { R } ^ { 4 } \mid \| x \| \leq 1 \}$ , so that $S ^ { 3 } = \partial B ^ { 4 }$ , and fix a knot K in $S ^ { 3 }$

Definition 2.4 (Slice surface and slice genus). A slice surface for a knot K is a compact, connected, oriented surface $F$ smoothly embedded in $B ^ { 4 }$ such that $\partial { \cal F } = { \cal K }$ . The slice genus $g _ { 4 } ( K )$ of K is the minimal genus of a slice surface for K.

Note that $g _ { 4 } ( K ) \leq g _ { 3 } ( K )$ , as a Seifert surface can be embedded in $B ^ { 4 }$ to become a slice surface. A useful way to visualise a slice surface is as a movie of links. Let $r \colon B ^ { 4 } \to$ R be the radial function. We regard $t : = 1 - r$ as time and intersect the surface with successive 3-dimensional level sets. At a regular time, we obtain a link in a copy of $S ^ { 3 }$ , which we represent by a diagram in the plane. As t increases, these diagrams change by planar isotopies, Reidemeister moves, and a non-trivial local change called a band move, which records a saddle point of the surface with respect to t.

The movie starts at $t = 0$ with a diagram of our knot K and ends with a diagram of the unlink. Then we glue discs to the components of the final unlink, corresponding to local maxima of t on the surface. The whole trace swept out by the moving diagram, together with these final caps, is a compact, oriented surface $F \subset B ^ { 4 }$ with $\partial F = K ;$ i.e., a slice surface for K. Its genus is determined entirely by the moves we performed during the movie, so every such movie gives an explicit upper bound on $g _ { 4 } ( K )$ .

The slice genus and the unknotting number are related by the inequality $g _ { 4 } ( K ) \leq u ( K )$ . This can be seen by smoothing the double points of the immersed surface in $B ^ { 4 }$ that the regular homotopy in ${ \check { S } } ^ { 3 }$ from K to the unknot traces out, and noting that each smoothing increases the genus by one. It should now be clear that the band framework that we developed for $u ( K )$ in Section 2.1 can be reused almost verbatim to produce diagrammatic upper bounds on $g _ { 4 } ( K )$ as well. The key is to perform a band move instead of crossing changes at the end of the band: instead of swapping the over/under information at one of the new crossings produced by the final Reidemeister II move of a band, we identify a target edge, cut the target and the band, and reconnect them in a way that agrees with the orientation of the two edges; see Figure 3.

![](images/06325d32f0b181fb919fc171bb06bdeb533690ffd8c7c65380fffb9f1b03df78.jpg)

![](images/bd7df74e8c52a4aa979e15ac15c2cd1569f8547d9e59f6d8e6d1d1ce3d10ff58.jpg)  
Figure 3. Band move. The top strand (in black) is the target edge and the bottom one (in green) is the final part of the band before the band move takes place. Orientation is crucial. On the left, no twist is needed. On the right, we have two possible cases, which we can think of as performing a final Reidemeister I move before the band move takes place. This Reidemeister I move is right-handed in the top picture and left-handed in the bottom picture.

We can therefore play exactly the same game as before on a fixed diagram of K (choose a strand to deform, slide it next to a target strand via Reidemeister II moves, possibly interleaved with Reidemeister I moves) but now we conclude each band with a band move rather than a crossing change. A sequence of such band paths taking a diagram of K to the unknot produces a slice surface for K whose genus is an explicit upper bound on $g _ { 4 } ( K )$

This unifies the two computations into a single combinatorial game on a fixed diagram of K. We choose a sequence of band paths, each ending either in a crossing change or in a band move, that simplifies the diagram to the unknot or an unlink, and try to minimise the upper bound we obtain. This is exactly the setup for our agents in Section 3.

Example 2.5 (Slice surface of genus one for the trefoil). As a concrete example, we study the case of the left-handed trefoil $K$ . Figures 4 and 5 show a slice movie building a slice surface of genus one for the trefoil. At time $t \in [ 0 , 1 ]$ , we draw the portion $F ^ { t }$ of the surface F that we have constructed so far, and above it the link ${ \cal L } _ { t } = \partial { \cal F } ^ { t } \backslash$ K (i.e., the “bottom boundary” with respect to the radial function on $B ^ { 4 } )$ . While we represent $L _ { t }$ with its actual link diagram, we represent the “horizontal sections” of $F ^ { t }$ as circles.

![](images/55ce9529912d086531d31c5aab25a24d95aacad4ef42610288992707b3a9135c.jpg)

![](images/bbebd266e27d9b852e70616c836e2a4ead6552f9d24a224f2e234b9f4d607411.jpg)

![](images/38a1ab3653b5180f27927ffe2b29f470b4d0b677699fdf5ecc58e1035f3fb4c5.jpg)

![](images/6f4bc13643621e0a997950c10782a10ffc2caef3752750b22a6a3a670b9f8195.jpg)  
(a) Time $t _ { a } = 0$  
(b) Time $t _ { b } > t _ { a }$

![](images/2fe9d30e42025fd910e573977cc45763208eab98f6e4d0af108e67e8eb907682.jpg)  
(c) Time $t _ { c } > t _ { b }$  
(d) Time $t _ { d } > t _ { c }$  
(e) Time $t _ { e } > t _ { d }$  
Figure 4. Building a slice surface for the trefoil: first part.

Let us start with the trefoil $K = L _ { 0 }$ depicted in Figure 4a. Recall that K lives in $S ^ { 3 }$ ; i.e., the boundary of $B ^ { 4 }$ . We push the trefoil from $S ^ { 3 }$ inside the ball $B ^ { 4 }$ , obtaining at time $t _ { b }$ a cylinder with the trefoil as its base; see Figure 4b. We select a portion of $L _ { t _ { b } }$ highlighted in green in the figure. We continue pushing down the link while deforming it with a Reidemeister II move until we reach the state depicted in Figure 4c. At this point, we can perform the band move shown in Figure 4d, which corresponds to adding a saddle to $F ^ { t _ { d } }$ and results in the link $L _ { t _ { e } }$ shown in Figure 4e.

The link $L _ { t _ { e } }$ has two components that we colour in red and blue, as shown in Figure 5f. As time increases, we simplify $L _ { t _ { e } }$ using a Reidemeister II move and a Reidemeister I move to obtain $L _ { t _ { g } }$ , which is the standard diagram of the Hopf link in Figure 5g. We now perform another band move on the link, which results in another saddle on the surface; see Figure 5h. Note that this saddle fuses the two components into a knot $L _ { t _ { h } }$ . We can simplify $L _ { t _ { h } }$ via two Reidemeister I moves, obtaining $L _ { t _ { i } }$ , the unknot shown in Figure 5i. At this point, we can cap of the final boundary component by gluing a disc to ${ \mathrm { i t } } ,$ obtaining a surface $F$ embedded in $B ^ { 4 }$ for which $\partial { \cal F } = { \cal K }$ ; this can be seen in Figure 5j. This shows that the slice genus of the trefoil is at most one, concluding our example.

![](images/35e68965b3c1c3d5fa414d94ddb742b12e4aeb7f250db5e9e5545719bb6235a8.jpg)

![](images/efd648386955f3623cd5f8e8b10c64c21a5ce7cb3098f7b2684af802d0f446b7.jpg)

![](images/80f506793b7469fe6a68527092816329038e24a553976b4293f7f3bc35cdf74b.jpg)

![](images/b2793bbcb5c18fedcad1bec8d3be41c44c8de324c0c7bec053d3d1dc0375a15e.jpg)

![](images/46f3658e33bde1fd9b50d588c48068e78d1ca0961105b4e72940c03665503a32.jpg)

![](images/f9d63f439df72cfb6f277b9d2a920192622905b0e9f87b2bbe2320240c231283.jpg)

(f) Time $t _ { f } > t _ { e }$  
![](images/b0a31a2713e5f3417f7c6c3c344f302d381adbb1f445a6c50a40838352358a26.jpg)  
(g) Time $t _ { g } > t _ { f }$

(h) Time $t _ { h } > t _ { g }$  
![](images/e44e67fc65aa2d728df6e22825a7fda991e714f398b67c1b0e9f15efcad7c1c3.jpg)  
(i) Time $t _ { i } > t _ { h }$

![](images/641232e53c5031e8ce235d2873373063e5189dc743427534e240393437ebcefb.jpg)  
(j) Time $t _ { j } > t _ { i }$  
Figure 5. Building a slice surface for the trefoil: second part.

## 2.2.1 Weak and strong slice genera for links

There are two extensions of the notion of slice genus to links.

Definition 2.6 (Weak and strong slice genera). Let $L = \cup _ { i = 1 } ^ { \ell } K _ { i }$ be a link. A weak slice surface for L is a compact, connected, oriented surface $F \subset B ^ { 4 }$ such that $\partial { \cal F } = L$ . The weak slice genus $g _ { 4 } ( L )$ is the minimal genus of a weak slice surface for L.

A strong slice surface for L is a compact, oriented surface $F \subset B ^ { 4 }$ with ℓ disjoint components $F _ { 1 } , \ldots , F _ { \ell }$ such that $\partial F _ { i } = K _ { i }$ for $i \in \{ 1 , \ldots , \ell \}$ . The strong slice genus $g _ { 4 } ^ { * } ( L )$ is the minimal total genus of a strong slice surface for L; i.e.,

$$
g _ { 4 } ^ { * } ( L ) = \operatorname* { m i n } \left\{ \sum _ { i = 1 } ^ { \ell } g ( F _ { i } ) : F = \bigcup _ { i = 1 } ^ { \ell } F _ { i } \subset B ^ { 4 } { \mathrm { ~ i s ~ a ~ s t r o n g ~ s l i c e ~ s u r f a c e ~ f o r ~ } } L \right\}
$$

We could allow disconnected surfaces in the definition of the weak slice genus without changing $g _ { 4 } ( L )$ , since one can take the connected sum of the components of a disconnected slice surface without changing the total genus. In the literature, the term “slice genus” is used for either the weak or the strong slice genus. In this paper, we refer to the weak slice genus as the “slice genus”.

Note that $g _ { 4 } ( L ) \ \leq \ g _ { 4 } ^ { * } ( L )$ for every link L. Indeed, if we take a strong slice surface for L and remove the discs we attached to the unlink “on the bottom”, we can attach ℓ 1 bands between unknots from diferent connected components and make it connected. This process does not change the boundary or the total genus. Not every link bounds a strong slice surface. Indeed, if $\textstyle \bigcup _ { i = 1 } ^ { \ell } F _ { i }$ is a strong slice surface for L, then $\operatorname { l k } ( K _ { i } , K _ { j } ) = F _ { i } \cdot F _ { j } = 0$ for any i, $j \in \{ 1 , \ldots , \ell \}$ with $i \neq j ;$ see Cavallo [Cav15, Lemma 5.1]. We call a link L algebraically split if $\operatorname { l k } ( K _ { i } , K _ { j } ) = 0$ for all $i \neq j ,$ , and we set $g _ { 4 } ^ { * } ( L ) = + \infty$ for a link L that is not algebraically split (using $\operatorname* { m i n } ( \varnothing ) = + \infty )$ . With this convention, $g _ { 4 } ^ { * } ( L )$ is defined for every link, and it is finite exactly for the algebraically split ones. If $L ^ { \prime }$ is obtained from L by reversing the orientation of a component, then $g _ { 4 } ^ { * } ( L ) = g _ { 4 } ^ { * } ( L ^ { \prime } )$ ; see [Cav15, Proposition 5.2].

## 2.2.2 Ribbon surfaces and the slice–ribbon conjecture

As we watch a movie as in Section 2.2, only two kinds of moves change the topology of the surface we are building: a band move, which adds a saddle, and the birth of a new unknotted component, which adds a local maximum of the radial function on $B ^ { 4 }$ . Forbidding the second kind singles out a smaller class of surfaces and therefore yields a possibly larger invariant.

Definition 2.7 (Ribbon surface and ribbon genus). A ribbon surface for K is a slice surface $F \subset B ^ { 4 }$ for K with no local maxima with respect to the radial function of $B ^ { 4 }$ . The ribbon genus $g _ { r } ( K )$ of K is the minimal genus of a ribbon surface for K. A knot with $g _ { r } ( K ) = 0$ is called a ribbon knot; equivalently, it bounds a ribbon disc in $B ^ { 4 }$

Every ribbon surface is, in particular, a slice surface, so the minimum defining $g _ { 4 } ( K )$ is taken over a larger set, and

$$
g _ { 4 } ( K ) \leq g _ { r } ( K ) .
$$

Whether this inequality can be strict when $g _ { 4 } ( K ) = 0$ is the content of a long-standing open problem, raised by Fox in 1961.

Conjecture 2.8 (Slice–ribbon conjecture, [Fox61, Problem 25]). Every slice knot is ribbon; that $i s , g _ { 4 } ( K ) = 0$ implies $g _ { r } ( K ) = 0$

The conjecture is still open. Families of potential counterexamples have been proposed, but none of them have been confirmed. See [GST10, Section 8] for details. The higher-genus version of the conjecture is also open, which states that $g _ { 4 } ( K ) = g _ { r } ( K )$ for every knot $K$

Definition 2.7 of ribbon surfaces for knots extends to links exactly as Definition 2.6 does, by requiring the ribbon surface to be connected in the weak case and to have one component per link component in the strong case. We write $g _ { r } ( L )$ and $g _ { r } ^ { * } ( L )$ for the two resulting invariants. We say that the link L is ribbon if $g _ { r } ^ { * } ( L ) = 0 ; \mathrm { i . e . }$ , when its components bound disjoint ribbon discs.

Conjecture 2.8 has a weak and a strong version for links, and both are open. The strong genus-zero statement, that every slice link is ribbon, is the one usually meant in the literature $( \mathrm { e . g . }$ , in [GST10, Section $8 ] )$ . It has been verified by Aceto, Kim, Park, and Ray [Ace+21] for 4-stranded 2-component pretzel links.

Upper bounds on both $g _ { r }$ and $g _ { r } ^ { * }$ are computed by the agents of Section 3, where the ribbon frameworks are obtained from the slice ones by removing the birth of an unknotted component from the set of allowed moves.

## 2.2.3 Computation of the slice genus from a movie

We explain how to compute the genus of a slice surface from a movie. For a compact, connected, orientable surface F with ∂F boundary components,

$$
g ( F ) = \frac { 2 - \chi ( F ) - | \partial F | } { 2 } ;\tag{2.1}
$$

see [FM11, Section 1.1.1]. Start with an ℓ-component link L for which we want to build the slice surface, and let $F ^ { t }$ be the surface we have built until time t. Before any births or band attachments, $\chi ( F ^ { t } ) = 0$ , as $F ^ { t }$ is a disjoint union of annuli. Assume we have reached the end of the slice movie, obtaining a compact, connected, orientable surface $F ,$ which is a cobordism between L and the m-component unlink. Therefore, the surface has $\ell + m$ boundary components. Each birth increases the Euler characteristic by one, while each band attachment decreases it by one. Denoting by b the number of births and by $\beta$ the number of band attachments, the Euler characteristic of the surface is $b - \beta ,$ , and equation (2.1) gives

$$
g ( F ) = { \frac { 2 - ( b - \beta ) - ( \ell + m ) } { 2 } } .\tag{2.2}
$$

By capping the m components of the unlink, we increase the Euler characteristic by m and remove m boundary components; hence, the genus remains the same.

We can actually use the above formula during an episode to compute the “partial genus” of the surface we are building. This is useful for understanding “how much genus has been created so $\mathrm { f a r } ^ { \prime \prime } ;$ , since this quantity can only increase as the episode continues. When the intermediate surface $F ^ { t }$ is disconnected, with components $F _ { 1 } ^ { t } , \ldots , F _ { c _ { t } } ^ { t }$ , we define its total genus as $g ( F ^ { t } ) : =$ $\textstyle \sum _ { i = 1 } ^ { c _ { t } } g ( F _ { i } ^ { t } )$ . By equation (2.2), if $L _ { t }$ has $m _ { t }$ components and we have encountered $b _ { t }$ births and $\beta _ { t }$ band attachments by time t, then

$$
g ( F ^ { t } ) = { \frac { 2 c _ { t } + \beta _ { t } - b _ { t } - \ell - m _ { t } } { 2 } } .\tag{2.3}
$$

Let $K _ { 1 } , \ldots , K _ { \ell }$ be the components of the link L. A strong slice surface $F$ for L is disconnected by definition. Write $F = \sqcup _ { j = 1 } ^ { \ell } F _ { j }$ , where the $F _ { j }$ are the connected components of the surface and $\partial F _ { j } = K _ { j }$ . We compute $g ( F _ { j } )$ for each $j \in \{ 1 , \ldots , \ell \}$ using equation (2.2), and the total genus of $F$ as

$$
g ( F ) : = \sum _ { j = 1 } ^ { \ell } g ( F _ { j } ) = \frac { 2 \ell + \beta - b - \ell - m } { 2 } .\tag{2.4}
$$

Note that a movie of $F$ fuses every unknot component that is born into a surface component that intersects $L .$ . In our implementation, we track every unknot component born separately and discard it if it does not fuse into a component that intersects L by the end of the episode.

## 2.3 RL basics

Reinforcement learning (RL) is a machine learning framework for sequential decision problems. The first ideas behind RL, the principle of learning from rewards and punishments, go back to the 1950s, but the most striking demonstrations are much more recent. In the 1990s, TD-Gammon [Tes91] reached near world-class play in backgammon, the first sign that this kind of training could lead to genuine expertise. In 2016, AlphaGo [Sil+16] defeated world champion Lee Sedol at Go, a game long considered too complex for computers to play at the highest level. In 2017, AlphaGo Zero [Sil+17] learned Go from scratch, without ever looking at a human game, suggesting that the method can discover its own playing style entirely from experience.

The basic picture is that of an agent that interacts with an environment by repeatedly choosing actions. Good moves are rewarded with a positive score, while bad moves receive a negative one, and the agent’s goal is to maximise its total score over time. Through trial and error, an initially random agent gradually learns strategies that lead to higher cumulative rewards.

The interaction is organised as a loop, sketched in Figure 6. From a given state, the agent observes the environment and selects one of the available actions according to its current strategy. The environment then carries out the action and returns both a reward, quantifying how good that move was, and a new state, which plays the role of “present state” in the following iteration. The process repeats until the episode ends. The rewards collected along the way are what the agent uses to revise its strategy between episodes, rather than a rule for choosing the next move.

To make this picture more precise, the agent’s strategy is encoded by a policy $\pi ( a \mid s )$ , which is a typically probabilistic prescription assigning to each state s a distribution over the available actions a. The agent does not simply pick the action with the highest immediate reward. Rather, its goal is to maximise the scalar discounted return

$$
\mathbb { E } _ { \boldsymbol { \pi } } \left[ \sum _ { { t } \geq 0 } \boldsymbol { \gamma } ^ { t } \boldsymbol { r } _ { t } \right]
$$

collected over an entire episode, where $r _ { t }$ is the reward received at time t, and $\gamma \in \mathsf { \Gamma } ( 0 , 1 ]$ is a discount factor. When $\gamma \ < \ 1$ , it weights immediate rewards more than future ones. To reason about these long-term consequences, the agent typically estimates a value function $V ^ { \pi } ( s )$ (or, alternatively, an action-value function $Q ^ { \pi } ( s , a ) )$ , defined as the expected cumulative reward obtained by starting from s (and taking action a) and following the policy π from then on. We instead set $\gamma = 1$ . Since the setting is episodic with a maximum number of actions, an episode is guaranteed to terminate, and this value for γ is allowed. Defining $\gamma = 1$ permits creating a reward function that always gives a higher return for a smaller invariant value found. More details can be found in Section 3.5. During evaluation, the smallest invariant found is retained separately.

![](images/882707d489124d33d5aec3c9bed29df36f2794b7179c31bc739d258adfdc9139.jpg)  
Figure 6. Schematic of the RL loop.

A core challenge in reinforcement learning is the exploration versus exploitation trade-of: the agent must exploit the actions that currently look best while still exploring less familiar ones to avoid getting stuck in suboptimal habits. The policy is not given in advance, but is learned over many episodes. In the deep reinforcement learning setting used in this work, π and V<sup>π</sup> are represented by neural networks and updated via gradient-based optimisation. This is the approach that powered AlphaGo and underlies the work described in the rest of this article.

Using RL to compute the unknotting number and the slice genus. We can phrase the computation of the unknotting number and the slice genus as a game. The RL framework is set up as follows.

Episode: starting from a link and trying to reach the unlink.

State: the link diagram, the band path being built, and additional data about the episode in progress.

Action: extending a band path, performing a crossing change (for the unknotting number), or a band move (for the slice genus).

Reward: the smaller the resulting unknotting number or slice genus, the higher the score.

## 3 The environment

In this section, we present the environment we have built. The environment encodes the game the agent will play via states, actions, rewards, and terminal conditions. We therefore deliberately describe it without any reference to the agent, which is nothing but a choice of a policy π inside the environment. Indeed, the same environment is shared by the three agents of Section 4: the RL policy, the random walker (RW), and the mixed RL+RW agent.

Before any episode takes place, we need to fix two fundamental parameters:

The framework selects the set of moves, the terminal condition, and the invariant that the episode computes (Section 3.1).

The shape $( \boldsymbol { c } _ { \mathrm { m a x } } , \boldsymbol { \ell } _ { \mathrm { m a x } } )$ is a pair of upper bounds on the number of crossings and link components a diagram is allowed to have during an episode. It is what makes all observations live in a common, fixed-dimensional space. The default we use is $( c _ { \mathrm { m a x } } , \ell _ { \mathrm { m a x } } ) = ( 1 0 0 , 1 6 ) .$

Throughout this section, we denote by c the number of crossings of the current diagram and by ℓ the number of link components. An outline of an episode is given in Appendix A alongside a worked example.

Implementation. The environment is implemented as a hybrid Nim/Python system. The diagram modifications, the construction of the band, the surface bookkeeping, the invariant features, and the random walker are written in Nim and compiled into shared libraries, while the RL layer (the Gymnasium environments, the datasets, and the experiment drivers) is written in Python. Link manipulation and simplification rely on spherogram-nim [Spherogram-nim], the Nim port of spherogram (the SnapPy module [SnapPy] that manages links).

## 3.1 Frameworks

With this environment, we can play five diferent games by slightly changing the rules and the move set. These are: unknotting, ribbon, slice, strong ribbon, and strong slice.

Unknotting. This is the framework in which we can apply crossing changes to a link diagram. A crossing change is applied via a band, which starts at an edge of the diagram and may travel over or under other edges. The crossing change itself takes place when the band “passes through” a final edge. A won episode returns an upper bound on the unknotting number of the input link.

Ribbon and slice. These are the frameworks for finding upper bounds on the ribbon genus and the slice genus of a link. They operate almost identically to the unknotting framework, but we apply a band move instead of a crossing change when we end the band. We equip every link component with a connected component label (CCL for short). Two current link components have the same CCL precisely when they belong to the same connected component of the evolving surface F<sup>t</sup>. This is used to keep track of the component count in the genus computation of equation (2.3). In the slice frameworks, it is also possible, before starting a band, to add a crossingless unknotted component to the diagram; i.e., a local maximum of the radial function on the surface. The ribbon frameworks forbid this. Internally, we apply a reverse Reidemeister II move to the crossingless unknot born and a neighbouring edge in order to provide crossing strands along which band sums can later be performed. The unknot also receives a new provisional CCL marked as containing no original link component. When a fusion joins diferent CCL classes, these classes and their associated surface bookkeeping records (see Section 3.2) are merged. The implementation retains the larger CCL for the merged class. If a crossingless unknot born reaches the terminal state of an episode, it is discarded.

The strong ribbon and strong slice frameworks. The strong ribbon and strong slice frameworks find upper bounds on the strong ribbon genus and the strong slice genus, respectively.

In the strong ribbon framework, a band may only join strands with the same CCL. In the strong slice framework, it may also join diferent CCL classes provided at least one contains no original link component. Thus, no connected surface component can contain two diferent original link components. In the strong slice case, where unknot\_birth is allowed, the crossingless unknotted components introduced are managed as in the slice framework. These frameworks apply only to algebraically split links, and links outside this family are rejected when the episode is created.

## 3.2 States and actions

State. A state of the game can be thought of as a tuple (D, B, FM), where:

D is the working diagram; i.e., its PD code, together with the vector of crossing signs, the number of split unknotted components, and the CCL data for each link component (relevant only in the genus frameworks). If the diagram is split, then it is made into a connected diagram via inverse Reidemeister II moves.

B is the partially built band, represented by a list of quadruples

## [crossing, strand, over/under, twists].

The pair [crossing, strand] encodes an edge of the diagram and the side into which the band extends. The entry over/under specifies whether the band goes over or under this edge. The fourth entry records the signed number of half-twists inserted in the band segment following that crossing strand. Thus, the internal state records both the total number of twists and their placement along the band.

FM is the framework metadata. For all frameworks, it stores the number of crossings of the initial diagram of an episode. In the unknotting framework, it stores the number of crossing changes performed. In the genus frameworks, it is a table indexed by CCL used for surface bookkeeping. For each surface component, it records the number of bands attached to it, the number of initial link components (including unknots introduced by birth) and the number of current link components belonging to it, and the number of original link components it contains (newborn unknots excluded).

Actions. An action is encoded as a string that can be of three diferent forms:

ccw\_twist/cw\_twist: insert a counter-clockwise (resp. clockwise) half-twist into the band;

unknot\_birth: add a new unlinked trivial component (slice frameworks only);

C{c}S{i}\_{type}: here (c, i) identifies a crossing strand and {type} is one of the following:

– start: begin a band at the crossing strand (c, i);

– over/under: route the band into the next face, passing over (resp. under) the strand (c, i);

end1/end2: in the unknotting framework, these describe whether the band goes over or under the final strand; in the genus frameworks, they specify whether we attach the band at (c, i) with a positive or negative twist inserted just before closing it;

end0: attach the band at (c, i) with no extra twist (genus frameworks only).

Note that only the end actions and unknot\_birth modify the link. An end builds the new PD code and the new crossing signs, and then calls a simplification routine. All the other actions only extend the band path.

Size of the action space. For each of the $c _ { \mathrm { m a x } }$ crossings and each of their 4 strands, the environment emits start, over, under, end1, end2, plus the two global twist actions: $2 0 c _ { \operatorname* { m a x } } + 2$ actions. Genus frameworks add one end0 per crossing strand, giving $2 4 c _ { \mathrm { m a x } } + 2 .$ , and the slice frameworks append unknot\_birth as the last index. At $c _ { \operatorname* { m a x } } = 1 0 0$ , this is 2002, 2402, and 2403 discrete actions, respectively.

Legality and masking. Most of these actions are illegal in any given state, and the illegal ones are masked out before the agent selects an action so that a policy never assigns a positive probability to an illegal move. Illegal actions consist of:

starting a band while one is already in progress, or on a crossing strand that does not exist;

routing the band over or under a strand that does not lie on the band’s current face;

re-crossing an already visited edge;

reaching a position admitting no legal continuation;

mixing twists of opposite signs within the same band or exceeding the maximum allowed number of twists max\_twists (we use max\_twists = 5 for the RL agent);

ending a band before it has started;

ending a band on a strand not reachable from the current face, or, in the strong frameworks, joining diferent CCL classes that both contain an original link component;

performing a band move that is incompatible with the orientations;

giving birth to an unknot outside the slice frameworks or while a band is in progress.

In the strong slice framework, a surface component containing no original link component may merge with another CCL class, subject to the other legality conditions. This remains possible after earlier fusions with other birth components.

## 3.3 The observation

The observation is a dictionary of named NumPy arrays of type float64, whose shapes depend only on $( \boldsymbol { c } _ { \mathrm { m a x } } , \boldsymbol { \ell } _ { \mathrm { m a x } } )$ . Observation values are not clipped. Each array is represented by a Gymnasium Box with unbounded numerical limits and a matching dtype. The value 1000 is used both as raw padding and as a sentinel for quantities that cannot be computed on a given diagram. The environment replaces raw padding by 0, except in band\_matrix, where it uses $- 3$ . Unavailable invariant values are also replaced by 0. Each invariant that may be unavailable has a flag equal to 0 when its value is available and 1 otherwise. The observation common to all the frameworks consists of the thirteen entries of Table 1, which at $( c _ { \mathrm { m a x } } , \ell _ { \mathrm { m a x } } ) = ( 1 0 0 , 1 6 )$ amount to 1923 floats. The base observation for the unknotting framework additionally carries the number of crossing changes performed, for a total of 1924 floats. In the genus frameworks, the base observation includes the last three entries of Table 2, for a total of 2020 floats. Note that some of these tensors are intentionally made redundant in order to make the base observation as explicit as possible for the agent.

To this base observation, we may concatenate a number of topological invariants that the agent might not be able to extract directly from the diagram. We encoded some of them (the linking matrix, the determinant, the signature, the Alexander and Jones polynomials, and a few others) in order to determine which ones help the agent improve its accuracy. Which of them actually help is an experimental question, and is the subject of Section 4.1.

Remark 3.1. The shape is a hard upper bound on diagram size. A diagram exceeding $c _ { \mathrm { m a x } }$ crossings or $\ell _ { \mathrm { m a x } }$ components during an episode cannot be represented (illegal state), and the episode is lost rather than truncated. Since band moves may increase the number of crossings, some bands are inadmissible.

<table><tr><td rowspan=1 colspan=1>Tensor Name</td><td rowspan=1 colspan=1>Tensor Shape</td><td rowspan=1 colspan=1>Tensor Content</td></tr><tr><td rowspan=1 colspan=1>PD_code</td><td rowspan=1 colspan=1> $c _ { \operatorname* { m a x } } \times 4$ </td><td rowspan=1 colspan=1>PD code of the current diagram</td></tr><tr><td rowspan=1 colspan=1>crossing_signs</td><td rowspan=1 colspan=1>Cmax</td><td rowspan=1 colspan=1>sign of each crossing</td></tr><tr><td rowspan=1 colspan=1>writhe</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>writhe of the diagram</td></tr><tr><td rowspan=1 colspan=1>gauss_code</td><td rowspan=1 colspan=1> $2 c _ { \mathrm { m a x } } + \ell _ { \mathrm { m a x } }$ </td><td rowspan=1 colspan=1>Gauss code</td></tr><tr><td rowspan=1 colspan=1>num_link_components</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>number of link components that have crossings</td></tr><tr><td rowspan=1 colspan=1>unlinked_unknot_components</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>number of crossingless unknot components</td></tr><tr><td rowspan=1 colspan=1>num_connected_components</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>number of connected components of the diagramcontaining crossings, counted before the reverseReidemeister II moves that connect them</td></tr><tr><td rowspan=1 colspan=1>band_path</td><td rowspan=1 colspan=1> $2 c _ { \mathrm { m a x } } \times 3$ </td><td rowspan=1 colspan=1>the band built so far: each row is a triple (crossing,strand, over/under), where the first two entriesdetermine a crossing strand and the last one is thesign of the action (0 for start, +1 if the band goesover the strand, -1 if it goes under).</td></tr><tr><td rowspan=1 colspan=1>total_twists</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>signed number of twists in the current band</td></tr><tr><td rowspan=1 colspan=1>band_matrix</td><td rowspan=1 colspan=1> $c _ { \operatorname* { m a x } } \times 4$ </td><td rowspan=1 colspan=1>the band path re-indexed per crossing strand: eachposition corresponds to a crossing strand and eachentry is $0 / 1 / { - } 1$ if that strand lies on the band,according to its start/over/under sign, -2 if thecrossing exists but the strand was never visited, and-3 if the crossing strand does not exist (padding)</td></tr><tr><td rowspan=1 colspan=1>band_twists</td><td rowspan=1 colspan=1> $2 c _ { \mathrm { m a x } }$ </td><td rowspan=1 colspan=1>signed number of twists in each segment of the band,row-by-row with band_path</td></tr><tr><td rowspan=1 colspan=1>simplified_num_crossings</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>number of crossings of the current diagram aftersimplification</td></tr><tr><td rowspan=1 colspan=1>init_num_crossings</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>number of crossings of the initial diagram of theepisode</td></tr></table>

Table 1. The elements of the base observation common to all frameworks (they describe the diagram and the band being built).

## 3.4 Terminal states: won, lost, and truncated episodes

An episode finishes when it reaches a terminal state. This happens on three diferent occasions: either the episode is won (i.e., the diagram represents an unlink), or it is lost (i.e., an illegal state has been reached), or it is truncated (i.e., the step budget max\_actions, which we set to 80 in training, is exhausted).

A state can be illegal for diferent reasons:

the agent picks an illegal action (the action mask makes this unreachable, but the check is kept as a guard);

the diagram no longer fits the shape $( \boldsymbol { c } _ { \mathrm { m a x } } , \boldsymbol { \ell } _ { \mathrm { m a x } } ) ;$

a band in progress terminates prematurely with no legal continuation (which can only happen in the strong frameworks because of the CCL constraint).

If an episode is won, the agent has produced a certificate for an upper bound. In the unknotting case, the agent’s answer coincides with the number of crossing changes performed during the episode; i.e., the number of end actions. In the other frameworks, we need to compute the genus of the surface built as indicated in Section 2.2.3. After the episode terminates in any way, we move to the next link in the dataset and start a new one.

<table><tr><td>Tensor Name</td><td>Tensor Shape</td><td>Tensor Content</td></tr><tr><td>crossing_changes</td><td>1</td><td>number of crossing changes performed so far (unknotting only)</td></tr><tr><td>component_ccl</td><td> $\ell _ { \mathrm { m a x } }$ </td><td>CCL of each link component (genus frameworks only)</td></tr><tr><td>genera_info</td><td> $\ell _ { \mathrm { m a x } } \times 5$ </td><td>one row per CCL, recording the CCL, the number of bands attached to that surface component, the number of initial link components (including unknots introduced by birth) and the number of current link components belonging to it, and the number of original link components it contains (genus</td></tr><tr><td>current_genus</td><td>1</td><td>frameworks only) genus of the surface built so far (genus frameworks only)</td></tr></table>

Table 2. The elements of the base observation specific to each framework: crossing\_changes is used in the unknotting framework and the last three in the genus frameworks.

## 3.5 Rewards

All frameworks use potential-based reward shaping. This means that the reward R splits into an immediate reward term and a potential reward Φ. The potential reward is a function of the state, while the immediate term also depends on the action. We denote by s the present state, a the action applied, and $s ^ { \prime }$ the next state.

The immediate term. The immediate term rewards winning, punishes errors, and charges a price for every other action. This splits again into two terms: the action reward $r ( a , s ^ { \prime } )$ , a function of the action and the next state, and the answer charge $C ( s , s ^ { \prime } )$ , a function of the present and next states. In Table 3, we report all the possible action reward values.

<table><tr><td>Event</td><td> $\overline { { { r ( a , s ^ { \prime } ) } } }$ </td></tr><tr><td>the diagram becomes terminal (episode won) an error state is entered (episode lost) a non-terminal end action (a band move or crossing change) an unknot_birth any other action, with pre-action band length  $L \geq 9$ </td><td> $+ R _ { \mathrm { w i n } }$   $- R _ { \mathrm { w i n } }$  -5 -1  $- \operatorname* { m i n } ( L - 7 , 5 )$ </td></tr></table>

Table 3. The action rewards.

Conditions are tested in the following order: error, win, birth, end, and other. The band length test $L \geq 9$ uses the pre-action band path length $L ,$ and twist actions do not increase that length. All non-winning actions are charged. An end action that reaches a terminal state receives a base reward $+ R _ { \mathrm { w i n } }$ without an additional 5 charge.

The answer charge is designed to give negative rewards for any increase in the returned invariant value, so that lower answers are preferred. We first need to define an invariant cap $i _ { \mathrm { c a p } }$ . We guarantee that the reward of a won episode is higher than that of any episode that is not won and that lower answers are better than higher ones, both up to $i _ { \mathrm { c a p } }$ . In Remark 3.2, we discuss the reward behaviour past this cap. We set $i _ { \mathrm { c a p } } = 8$ as it is the highest invariant value in the datasets we are interested in. Denote by $i ( s )$ the current invariant value at the state s. In the unknotting framework, this is the number of crossing changes made so far. In the genus frameworks, it is the current genus of the surface being built; see Section 2.2.3. Note that for the initial state $s _ { 0 } .$ we have $i ( s _ { 0 } ) = 0$ . The answer charge depends on the framework. In the case of ribbon and slice, the cost of increasing the genus is charged immediately by defining $C$ as

$$
C \left( s , s ^ { \prime } \right) = W \big ( i ( s ^ { \prime } ) - i ( s ) \big ) ,
$$

where $W \in \mathbb { N }$ is an integer factor called “answer weight”. For unknotting, strong ribbon, and strong slice, the cost of increasing the invariant value is charged only at the end of an episode. $C$ is defined as

$$
C \left( s , s ^ { \prime } \right) = \left\{ \begin{array} { l l } { W \operatorname* { m i n } \bigl ( i ( s ^ { \prime } ) , i _ { \mathrm { c a p } } \bigr ) , } & { \mathrm { i f ~ } s ^ { \prime } \mathrm { ~ i s ~ w o n , } } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

This diference is justified in Remark 3.2.

The potential term. The potential Φ brings the number of crossings of the diagram into the reward. The potential contributes to the final reward as $\gamma \Phi \left( s ^ { \prime } \right) - \Phi \left( s \right)$ , generally discounting the potential of the new state by the factor $\gamma \in ( 0 , 1 ]$ . As already stated in Section $2 . 3 ,$ we use $\gamma = 1$ . This shaping term telescopes along an episode and it guides the agent towards promising states with fewer crossings [NHR99]. More specifically,

$$
\Phi ( s ) = \left\{ { \begin{array} { l l } { 0 } & { \mathrm { i f ~ } s \mathrm { ~ i s ~ t e r m i n a l ~ o r ~ a n ~ e r r o r ~ s t a t e } , } \\ { - \kappa c ( s ) \quad \quad \quad } & { \mathrm { o t h e r w i s e } , } \end{array} } \right.
$$

where $c ( s )$ is the number of crossings of the simplified current diagram, and $\kappa = 6 5 / ( 1 + \sqrt { c _ { 0 } } )$ scales the crossing term by the number of crossings $c _ { 0 }$ of the initial diagram. In particular, an error transition receives no potential shaping. The numerator of $\kappa$ is an empirical constant left unchanged in all our experiments. The total shaping accumulated over a successful episode is

$$
\sum _ { t = 0 } ^ { T - 1 } \left( \Phi ( s _ { t + 1 } ) - \Phi ( s _ { t } ) \right) = - \Phi ( s _ { 0 } ) + \Phi ( s _ { T } ) .
$$

For a successful episode, this is $\kappa c _ { 0 }$ . Note that

$$
\kappa c _ { 0 } = \frac { 6 5 c _ { 0 } } { 1 + \sqrt { c _ { 0 } } } = 6 5 \left( \sqrt { c _ { 0 } } - 1 + \frac { 1 } { 1 + \sqrt { c _ { 0 } } } \right) ,
$$

which is asymptotic to $6 5 \sqrt { c _ { 0 } }$ . Links whose crossing numbers difer by an order of magnitude still receive shaping of comparable size.

The complete reward. The complete reward is defined as

$$
R \left( s , a , s ^ { \prime } \right) = \left\{ \begin{array} { l l } { - \sigma R _ { \mathrm { w i n } } , } & { s ^ { \prime } \mathrm { ~ i s ~ a n ~ e r r o r ~ s t a t e } , } \\ { \sigma \big ( r \left( a , s ^ { \prime } \right) - C \left( s , s ^ { \prime } \right) + \Phi \left( s ^ { \prime } \right) - \Phi \left( s \right) \big ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

where $\sigma \in ( 0 , 1 )$ is a rescaling factor.

It remains to choose the values of $R _ { \mathrm { w i n } } , ~ W$ , and $\sigma .$ The first two depend on three factors: the maximum number of actions allowed in an episode, the invariant cap defined above, and the rewards of non-terminal actions of Table 3, which are between 1 and 5. We choose $W = 5 \cdot 8 0 = 4 0 0$ to guarantee that an episode returning a lower invariant is better rewarded than one returning a higher value. Indeed, a won episode has at most 79 non-terminal actions (plus one winning end action), which cost at most $5 \cdot 7 9 = 3 9 5 < W$ in total, so a single unit of invariant outweighs any diference in the length of the path. Then, to guarantee that a won episode is rewarded more than a truncated one, we choose $R _ { \mathrm { w i n } } = 5 \cdot 8 0 ( i _ { \mathrm { c a p } } + 1 ) = 3 6 0 0$ . Lastly, the choice of $\sigma = 0 . 0 1$ rescales the rewards to the order of 10 so that they do not saturate the value network’s tanh units.

Remark 3.2 (On the answer weight W and the invariant cap $i _ { \mathrm { c a p } } )$ . Recall that $i _ { \mathrm { c a p } } = 8$ is the highest invariant value in our datasets. We chose W so that one unit of answer costs more than all the actions of a won episode together, and $R _ { \mathrm { w i n } } = W ( i _ { \mathrm { c a p } } + 1 )$ so that it exceeds the largest charge, $W i _ { \mathrm { c a p } } ,$ by W as well.

In the unknotting, strong ribbon, and strong slice frameworks, the answer is charged once, when the episode is won. In the strong frameworks, a band can reach a state with no legal move, and the episode then ends on an error. When we charged the genus band by band in these frameworks, ending an episode early on an error was cheaper than playing on and paying for the genus built, and the agents learned to end their episodes on an error. In the unknotting framework, charging every crossing change taught the agent to avoid crossing changes, although they are its only end actions. Without the cap, the first wins found in training used many crossing changes and were worth less than a truncated episode, and the episodes of the agent went back to the time limit. Since the charge is at most $W i _ { \mathrm { c a p } } ,$ every won episode is better rewarded than every episode that is not won, whatever its answer. The better reward of a lower answer is guaranteed below the cap. Answers from $i _ { \mathrm { c a p } }$ and above receive the same charge and are compared only through the length of the path. If the initial diagram has at least two crossings, a truncated episode loses less than $R _ { \mathrm { w i n } }$ through its actions and the crossings it adds, so an error is worse rewarded than any other outcome.

In the ribbon and slice frameworks, the genus is charged as the surface is built, with no cap. Here a lower genus is always better rewarded. However, the reward of a won episode is guaranteed to be higher than that of every episode that is not won only up to genus $i _ { \mathrm { c a p } }$ . A truncated episode has already paid for the genus it built, so an error can be cheaper than a long truncated episode. Since ribbon and slice always have a legal move, they reach an error only when the diagram outgrows the shape.

## 4 The agents

Three agents play the game of Section 3: a trained RL policy, a Bayesian-optimised random walker, and a mixture of the two. This section describes each of them and how we selected the best one for each framework.

Accuracy. To compare the performance of diferent agents, we measure their accuracy. This is defined as the percentage of knots and links for which the agent finds a sequence of moves that matches or improves the best known upper bound.

Working datasets. For each framework, we created the following four disjoint datasets that have been used at various stages:

The training dataset is used to train the RL policy.

The validation dataset is used to select the best RL checkpoint and the best parameters for the random walker and the mixed agent. A fixed subset of 500 of its links, the coarse validation set, is used for the cheaper passes.

The development dataset is used to choose the input features of the RL policy.

The test dataset is used only to evaluate the final agents after every selection and tuning step has been completed and is never used to select or tune anything.

The datasets were constructed to provide the models with a diverse collection of samples. In them, we can find prime knots and prime multi-component links, connected sums of knots and links, and, in the strong frameworks, Whitehead doubles of links, with varying invariant values and link sizes. We split the samples by underlying primes. A prime knot or link, together with all its orientations and mirror images, lies in exactly one dataset, and the connected sums and Whitehead doubles of a dataset are built from the primes of that dataset only. Hence, no prime link appears in two datasets in any role. The assignment is the same for every framework, so a link is in the test dataset of every framework or of none. Roughly 70% of the links go to training and 10% to each of the other three datasets. Moreover, we split the samples so that the proportions of types, sizes, and invariant values are similar across the four datasets. Given the similarity between the ribbon and slice frameworks, and the strong ribbon and strong slice frameworks, these pairs of frameworks share the same datasets. The resulting dataset sizes are reported in Table 4.

<table><tr><td>Framework</td><td>Training</td><td>Validation</td><td>Development</td><td>Test</td><td>Total</td></tr><tr><td>unknotting</td><td>6310</td><td>838</td><td>838</td><td>841</td><td>8827</td></tr><tr><td>ribbon, slice</td><td>5701</td><td>800</td><td>801</td><td>807</td><td>8109</td></tr><tr><td>strong ribbon, strong slice</td><td>5153</td><td>731</td><td>868</td><td>690</td><td>7442</td></tr></table>

Table 4. Dataset sizes. The coarse validation set consists of 500 of the validation links.

Allocated agent budget. We quote evaluation budgets as B/N budget units per link, where B is the nominal total allocation and N is the size of the dataset. An attempted episode containing a actions is charged a + 1 units: one unit per action and one unit accounting for reset per attempted episode, including unsuccessful and zero-action attempts. The evaluator cycles through the links, starting another episode only while its remaining budget is positive. Once started, an episode runs until an ordinary stopping condition, including pruning or the episode action limit. It is not interrupted by exhaustion of the evaluation budget. Consequently, the final episode can exceed the nominal allocation. The recorded answer for each link is the best found across its attempts, with only strict improvements replacing the stored answer.

For parallel evaluation, the dataset is partitioned into shards $S _ { j }$ . Writing $\begin{array} { r } { d _ { j } = \sum _ { i \in S _ { i } } c _ { i } ^ { 3 / 2 } } \end{array}$ where $c _ { i }$ is the number of crossings of the diagram of link i, shard j receives a budget of

$$
B _ { j } = \left\lceil B \frac { d _ { j } } { \sum _ { k } d _ { k } } \right\rceil .
$$

Each shard applies the episode-accounting rule independently. Upward rounding and the final episode in each shard can therefore make the consumed budget exceed B. The quoted per-link figures are nominal average allocations, not exact environment-action counts or hard per-link limits.

## 4.1 The RL policy and the choice of input features

For our first agent, we trained an RL policy to interact with the environment described in Section 3. Indeed, its observation contains every part of the state on which the legal actions, the next state, or the reward depend; these are the values reported in Tables 1 and 2. The observation is fed into a pair of neural networks: a policy network that outputs a distribution over actions, and a value network that estimates the expected cumulative reward of the current state. Both networks are built on top of Stable-Baselines3’s MultiInputPolicy [Raf+19]. For training, we use the SB3-Contrib implementation of MaskablePPO, an implementation of Proximal Policy Optimization (PPO) with invalid action masking. PPO is an on-policy actor-critic policy gradient algorithm introduced by Schulman, Wolski, Dhariwal, Radford and Klimov [Sch+17], and is currently among the most widely used RL algorithms in practice.

The fixed-size tensors forming the dictionary observation are flattened and concatenated, and every entry is brought to order one by a fixed map with no trainable parameters: a division by a fixed scale or, for larger entries, a signed logarithm. Without this scaling, large entries saturate the first tanh layer, and the policy stops depending on the state. The resulting vector is passed to separate policy and value networks, each consisting of two fully connected layers with 256 units each with tanh activations. We use a discount factor $\gamma = 1$ (since the reward of Section 3.5 is designed for undiscounted returns), a learning rate $3 \times 1 0 ^ { - 4 }$ , a batch size 64, ten optimisation epochs per update, a GAE parameter $\lambda = 0 . 9 5$ , and a PPO clipping range 0.2. The remaining PPO hyperparameters are left at their library defaults.

At inference time, the policy defines a categorical distribution over the fixed action space. Before each prediction, actions that are illegal in the current state are masked out, and the distribution is renormalised over the remaining actions. We sample from this masked distribution and repeat the procedure until the episode terminates or reaches its step limit.

## 4.1.1 The feature-selection protocol

The first question we needed to address was which features would provide the RL agent with the most informative representation of the current state. The base observation for each framework is shown in Tables 1 and 2. Several additional candidate features can be added to this observation, but it is not clear a priori which of them would provide the agent with the most useful information. We tested the following features:

Baseline (base observation only);

MT: the Murasugi–Tristram bound;

Seif: the upper bound on the Seifert genus obtained using Seifert’s algorithm;

LM: the Linking Matrix;

CCLM: the CCL Linking Matrix, where entry (i, j) is the total linking number of the link components with Connected Component Labels i and j;

CompM: the Components Matrix, where entry (i, j) records on which link component (labelled by integers) the j-th strand of the i-th crossing lies;

DetSig: the link determinant and signature;

Alex: the Alexander polynomial;

Jones: the Jones polynomial (only for links up to 16 crossings, since the cost of computing it grows exponentially with the number of crossings);

Theta: evaluations of the Θ-invariant of Bar-Natan and van der Veen [BV26], at a single evaluation point (the cost grows linearly with the number of points, and Theta is already the slowest feature to train with).

All these quantities can be recomputed from the current diagram every time it changes. We did not include invariants such as the Rasmussen invariant s, whose computation is far more expensive. In addition, we trained using pairs of features for each framework, pairing the two features whose models solved the most complementary sets of development links: LM+MT (unknotting), DetSig+LM (ribbon), DetSig+Seif (slice), MT+Jones (strong ribbon), and CompM+DetSig (strong slice). This provided no improvement overall. Each feature pair scored below the best of the two features on the development dataset, between 3.6% and 12.0% depending on the framework. The overall score of the feature pair was also below the score of the best model of each framework, so we do not consider them further.

To select the best feature set for each framework, we trained one model per candidate feature and compared them under the following protocol.

Checkpoint selection (validation). For each configuration, we trained for roughly 1,000,000 timesteps (between about 160 and 195 timesteps per training link), saving the policy parameters every 15,000 timesteps. We call each saved snapshot a (training) checkpoint, obtaining 66 checkpoints in total. The budget is deliberately generous, so that training can continue beyond the peak in validation performance. This matters because the best-performing checkpoint often occurs early in training.

Let us denote by $C _ { 1 } , \ldots , C _ { 6 6 }$ the checkpoints obtained during training. A naive way to select one would be to validate all of them on the validation set and keep the one with the highest accuracy score. However, this could result in favouring whichever checkpoint was most inflated by noise [SW06]. Moreover, the validation curve itself is noisy, and the raw maximum might not be the best stopping point [Pre12]. Therefore, the checkpoint validation consists of two phases.

(1) First, a coarse pass. We score the first checkpoint $C _ { 1 }$ and every third checkpoint $\left( { { C _ { 3 n } } } \right)$ on the coarse validation set, with a nominal evaluation budget of 200 steps per link. Writing $a ( C _ { 3 n } )$ for the corresponding accuracy score, we compute a moving average over adjacent scored checkpoints rather than take the noisy arg-max:

$$
\begin{array} { r } { s ( C _ { 3 n } ) = \frac { 1 } { 3 } \left( a ( C _ { 3 n - 3 } ) + a ( C _ { 3 n } ) + a ( C _ { 3 n + 3 } ) \right) , } \end{array}
$$

where $C _ { 1 }$ plays the role of $C _ { 0 }$ , and the average is taken over the available neighbours at the two ends of the curve. This whole procedure is computationally cheap, and it allows us to locate the plateau of the validation curve by looking at the five checkpoints with the highest smoothed accuracies.

(2) Second, the refinement. We then evaluate these checkpoints and their immediate neighbours, $C _ { 3 n - 1 }$ and $C _ { 3 n + 1 }$ , on the full validation set. The best of these is selected as the best model for that configuration.

Feature choice (development) and budget. The selected checkpoint of each configuration is evaluated exactly once on the development dataset, and the configuration with the highest accuracy there is selected for its framework. Any configuration within one standard error of the best counts as a tie, and a tie is resolved in favour of the feature that is cheaper to compute. The test dataset plays no role in this choice.

Training multiple seeds of the same feature showed that the accuracy of the selected checkpoint varies with the seeding by a few percentage points. Therefore, the above selection process has to be considered a mix of model selection and feature selection.

Every evaluation presented above (coarse, full-validation, and development) gives the RL agent a nominal evaluation budget of about 200 units per link. This allocation is a compromise between computational feasibility and model accuracy. On one hand, each link needs several search attempts because the policy is stochastic, and a valid sequence of moves may not be found on the first try. On the other hand, it would take too long to evaluate the models with a substantially larger budget. The development evaluation was performed at both 200 and 800 units per link, the latter being the budget given for the test evaluation of Section 4.4. The winning configuration is the same in both.

Longer training. The validation curves of several configurations were still rising at the end of training, so we trained ten configurations, chosen among the best of their frameworks, for 2,000,000 timesteps: DetSig and LM on ribbon; Seif, Jones, and Alex on slice; MT on strong ribbon; and CompM, LM, Seif, and DetSig on strong slice. No unknotting configuration was included, since the best ones reach their peak within the first 75,000 timesteps. Only one of these runs improved on the model selected for its framework: the MT configuration of the strong ribbon framework, whose development accuracy went from 12.7% to 18.4%, with the best checkpoint at 1,755,000 timesteps. We therefore trained it for 1,000,000 timesteps more, with the new best checkpoint being the one at 2,775,000 timesteps, where the new development accuracy is 23.3%. We use this model for the strong ribbon framework.

Best feature set per framework. The selected feature configuration for each framework is given in Table 5.
<table><tr><td>Framework</td><td>Selected feature</td><td>Accuracy</td><td>Baseline</td></tr><tr><td>unknotting</td><td>Baseline</td><td>66.8%</td><td>66.8%</td></tr><tr><td>ribbon</td><td>DetSig (determinant and signature)</td><td>63.3%</td><td>56.4%</td></tr><tr><td>slice</td><td>Seif (Seifert genus upper bound)</td><td>60.2%</td><td>57.9%</td></tr><tr><td>strong ribbon</td><td>MT (Murasugi-Tristram bound)</td><td>23.3%</td><td>7.7%</td></tr><tr><td>strong slice</td><td>CompM (components matrix)</td><td>14.1%</td><td>6.2%</td></tr></table>

Table 5. Selected input feature per framework, with its accuracy on the development set at 200 units per link and, for comparison, the accuracy of the bare state observation. Since each selected accuracy is the best of ten, it is biased upwards. The unbiased figures are those measured on the test set in Section 4.4.

## 4.2 The random walker and its Bayesian optimisation

Alongside the RL agent, we also built a random walker (RW) agent that is able to interact with the environment. The random walker is governed by two relative action weights and three additional parameters. The weights p\_twist and p\_end control the relative frequencies of twist and end actions, respectively. The parameter p\_birth is the conditional probability of adding an unknot when no band is in progress and is used only in the slice frameworks. Finally, max\_twists and max\_actions bound the number of twists per band and the number of actions per episode, respectively.

At each step, the random walker first removes action types for which no legal action exists. The remaining types start, over, under, twist, and end receive relative weights 1, 1, 1, p\_twist, and p\_end, respectively. An action type is sampled proportionally to these weights, after which a concrete legal action of that type is selected uniformly. In particular, the twist direction is chosen uniformly among the legal directions, while an end action is obtained by sampling a legal final strand and then a legal end sign uniformly. In the slice frameworks, when no band is in progress, an unknot\_birth action is first taken with probability p\_birth; otherwise, the ordinary weighted sampling rule is used.

Bayesian optimisation. For each framework, these parameters were tuned via Bayesian optimisation (BO) using the accuracy on the same coarse validation sets as the metric employed for the coarse validation of the RL models. Bayesian optimisation is well suited to this problem, as the objective is a black box with no accessible gradient, evaluations are expensive, and the space of weights is low-dimensional. The optimiser builds a probabilistic model of the accuracy over that space and uses it to choose which weights to try next, spending its budget where the model is promising or still uncertain rather than on a blind grid. We used the Bayesian optimisation implementation developed by Nogueira [Nog14], with its default configuration.<sup>1</sup>

The BO was performed with a budget of 300 iterations (50 initial exploratory iterations plus 250 tuning iterations), each using a nominal evaluation budget of 500,000 units (1,000 per link). We first searched the space $\mathtt { p \_ t w i s t } \in [ 0 . 1 , 5 . 0 ]$ , p\_end $\in [ 0 . 5 , 1 0 . 0 ]$ , max\_twists [1, 5], max\_actions $\in \ [ 5 , 5 0 ]$ , and $\mathtt { p \_ b i r t h } \in \ [ 0 . 0 , 1 . 0 ]$ for the two slice frameworks. On four frameworks, the optimum of p\_end lay at the upper edge of this box, and that of p\_twist close to its lower edge. We therefore repeated the search with p\_twist $\in \ [ 0 . 0 1 , 5 . 0 ]$ and p\_end [0.5, 100], both sampled on a logarithmic scale, running two independent optimisations with diferent random seeds for each framework. Instead of adopting the best configuration found by the optimiser, we re-evaluated the ten best configurations of each run five times on the full validation set, with the nominal budget of 800 units per link used in the final benchmark, and adopted the one with the highest mean accuracy. The tuned defaults are given in Table 6, where max\_twists and max\_actions are converted to integer values. Since p\_twist and p\_end are relative weights, the table also reports the probabilities they induce at a step inside a band where every action type is legal:

$$
P ( \mathrm { e n d } ) = \frac { \mathrm { \Delta } \mathrm { p } _ { - } \mathrm { e n d } } { 2 + \mathrm { p } _ { - } \mathrm { t w i s t } + \mathrm { p } _ { - } \mathrm { e n d } } , \qquad P ( \mathrm { t w i s t } ) = \frac { \mathrm { \Delta } \mathrm { p } _ { - } \mathrm { t w i s t } } { 2 + \mathrm { p } _ { - } \mathrm { t w i s t } + \mathrm { p } _ { - } \mathrm { e n d } } .
$$

<table><tr><td>Framework</td><td>p_twist</td><td>p_end</td><td> $P ( { \mathrm { e n d } } )$ </td><td> $P ( \mathrm { t w i s t } )$ </td><td>max_ twists</td><td>max_ actions</td><td>p_birth</td><td>Accuracy</td></tr><tr><td>unknotting</td><td>0.061</td><td>90.44</td><td>97.8%</td><td>0.07%</td><td>1</td><td>10</td><td></td><td> $5 8 . 2 \pm 0 . 3 \%$ </td></tr><tr><td>ribbon</td><td>0.351</td><td>98.31</td><td>97.7%</td><td>0.35%</td><td>1</td><td>20</td><td></td><td> $5 6 . 9 \pm 0 . 6 \%$ </td></tr><tr><td>slice</td><td>0.012</td><td>86.63</td><td>97.7%</td><td>0.01%</td><td>1</td><td>21</td><td>0.005</td><td> $5 6 . 4 \pm 0 . 2 \%$ </td></tr><tr><td>strong ribbon</td><td>0.023</td><td>90.02</td><td>97.8%</td><td>0.03%</td><td>1</td><td>17</td><td></td><td> $1 0 . 6 \pm 0 . 2 \%$ </td></tr><tr><td>strong slice</td><td>0.022</td><td>89.80</td><td>97.8%</td><td>0.02%</td><td>4</td><td>17</td><td>0.001</td><td> $1 0 . 7 \pm 0 . 3 \%$ </td></tr></table>

Table 6. Tuned random walk parameters per framework, with the probabilities of ending a band and of inserting a twist that they induce. The last column is the validation accuracy, as mean and standard error over five evaluations at 800 units per link.

In all ten optimisation runs, the best configuration ends a band with probability between 95.3% and 97.9%, and the adopted walkers almost never twist. So, the walker’s best strategy is to close a band as soon as it legally can. The optimum of p\_end is again close to the upper edge of the box, but this is saturation rather than truncation: setting $\mathsf { p } _ { - } \mathsf { e n d } = 1 0 ^ { 4 }$ so that P(end) 99.98% changes the validation accuracy of the tuned walkers by between 0.1% and $+ 1 . 6 \%$ , never by more than 2.2 standard errors (five paired evaluations at 800 units per link). The parameter that separates the frameworks is max\_actions: 10 for unknotting, 17 for the strong frameworks, 20 for ribbon, and 21 for slice. On unknotting, a cap of 10 allows at most five crossing changes, so the walker gives up the links whose best known upper bound is at least six, about 7% of each dataset, in exchange for more attempts on the others. The two optimisations chose 10 and 11 out of a range reaching 50.

We now remark on the last column of Table 6. Each evaluation of an objective is a single stochastic run over the coarse validation set, so the accuracy it returns is noisy, and what the optimiser reports is the largest of a few hundred such draws. As with the raw validation maximum over checkpoints in Section 4.1, this is biased upwards by an amount that grows with the budget [SW06]. This is why we re-evaluated the candidates, and on three frameworks, the optimiser’s own best configurations scored up to 3.5% below the adopted ones. The figures in the table are still tuning figures, since the coarse validation set is part of the validation set, and are not themselves performance claims. Every accuracy we compare across agents is measured on the held-out test sets.

## 4.3 The mixed RL+RW agent

As a final agent, we built a mixed agent that follows the trained RL policy most of the time and hands control to the random walker (RW) for the remaining steps. At the start of each episode, we draw the initial probability of selecting the random walker and then decay this probability as the episode proceeds. The mixed agent is governed by the following parameters:

p\_twist: the relative weight assigned to twist actions when the random walker is in control;

$\mathbb { P } \cdot$ \_end: the relative weight assigned to end actions when the random walker is in control;

p\_birth: the conditional probability of adding an unknot when the random walker is in control and no band is in progress (this parameter is meaningful only for the slice frameworks);

p\_rw\_range: an interval $[ a , b ] \subseteq [ 0 , 1 ]$ , with $a \leq b ,$ from which the initial probability of following the random walker is drawn at the start of each episode;

min\_p\_rw: a floor on that probability, leaving the random walker a residual chance of intervening late in the episode, once the decay has taken efect;

step\_decay and end\_decay: the multiplicative factors at which the random walker’s influence fades within an episode; the former is applied after an ordinary action, and the latter after a band is ended.

Compared with the pure random walker, max\_twists and max\_actions are absent, and those limits are inherited from the RL environment rather than tuned for each agent.

How the mixed agent works. The two strategies are combined as follows. At the start of each episode, an initial probability $p _ { \mathrm { r w } } ^ { ( 0 ) }$ is sampled uniformly from the interval p\_rw\_range. This probability is multiplied by end\_decay after every end action and by step\_decay after every other action. We write $p _ { \mathrm { r w } } ^ { \dot { ( t ) } }$ for its value at step t. For instance, after t non-end actions, we will get $p _ { \mathrm { r w } } ^ { ( t ) } = p _ { \mathrm { r w } } ^ { ( 0 ) } \cdot \left( \mathsf { s t e p \_ d e c a y } \right) ^ { t }$ . At each step, the action is taken by the random walker with probability $\mathrm { m a x } ( \mathrm { m i n _ { - } p _ { - } r w } , p _ { \mathrm { r w } } ^ { ( t ) } )$ , and by the trained RL policy with the complementary probability.

When the random walker is selected, it performs an unknot birth with probability $\mathtt { p \_ b i r t h . }$ provided the framework permits it and no band has started. Otherwise, it samples an action type among those currently legal, with relative weights 1 for start, over, and under, p\_twist for a twist, and $\mathsf { p } _ { - }$ \_end for an end action, and then chooses the crossing strand (or the twist direction) uniformly among the legal ones. The random walker, therefore, injects stochastic exploration early in the episode and gradually hands control over to the RL policy as the episode develops.

## 4.3.1 Bayesian optimisation of the mixed agent

We tuned these parameters via Bayesian optimisation, using each framework’s best RL model as the policy component. We used the same coarse validation sets employed for the coarse validation of the RL models, with a budget of 200 objective evaluations per optimisation (40 initial points plus 160 optimisation iterations) and a nominal evaluation budget of 200 units per link for each objective evaluation. As for the random walker, we searched $\mathtt { p _ { - } }$ twist and $\mathtt { p _ { - } }$ end on the wider logarithmic box of Section 4.2, ran two independent optimisations per framework, and re-evaluated the five best configurations of each run five times on the full validation set at 800 units per link, adopting the one with the highest mean accuracy. We re-evaluated five configurations per run rather than the ten of the random walker because a step of the mixed agent costs several times a step of the walker.

As we show below, the parameter that matters most is the range p\_rw\_range from which the mixing rate is drawn. From a practical standpoint, we encoded p\_rw\_range by two endpoints $\left[ \mathrm { p } _ { - } \mathrm { r w } _ { - } \mathrm { r a n g e } _ { \mathrm { l o w } } , \mathrm { p } _ { - } \mathrm { r w } _ { - } \mathrm { r a n g e } _ { \mathrm { h i g h } } \right]$ , sampled independently from [0, 1] and then sorted. We parametrised min\_p\_rw as a fraction $\varepsilon \in [ 0 , 1 ]$ of the lower endpoint, so that

$$
\begin{array} { r } { \tt { \operatorname* { m i n } \_ p _ { - } r w } = \tt { \varepsilon } \cdot \tt { p _ { - } r w \_ r a n g e } _ { \log w } . } \end{array}
$$

This parametrisation guarantees that every configuration proposed by the optimiser is admissible, so that no evaluation is wasted on an invalid one. Sampling min\_p\_rw independently could place it above the sampled range and make it override both the range and the decay.

Which parameters actually matter. To identify which parameters carry a detectable signal, we use a simple descriptive signal-to-noise comparison. We first estimate a noise floor from independent repeated evaluations of identical parameter configurations. For each pair of repeats, we compute the absolute diference between their accuracies and define the noise floor as the median of these diferences. This captures the run-to-run variation due to the stochasticity of the agent and the evaluation procedure.

We then estimate the marginal efect of each parameter. We split its sampled values into five equally populated groups, compute the mean accuracy within each group, and define its marginal accuracy range as the diference between the largest and the smallest of these five means. For p\_rw\_range, we use the midpoint of the range and all the evaluations. For the other parameters, we only use the evaluations in the three lowest groups of p\_rw\_range, whose means difer by less than the noise floor, so that the efect of the mixing rate does not mask theirs.

Table 7 compares these accuracy ranges with the noise floor. We call a parameter significant when its accuracy range is substantially larger than the noise floor. Across all five frameworks, p\_rw\_range meets this criterion, and at this budget, its efect is negative. The accuracy is flat over the lower mixing rates and drops for the higher ones. The ranges of all the other parameters are below the noise floor, with one exception, p\_birth on the slice framework, which we discuss below. We therefore conclude that p\_rw\_range is the only parameter that exhibits a consistent and substantial marginal efect on accuracy.

<table><tr><td rowspan="2">Framework</td><td rowspan="2">Noise floor</td><td colspan="5">Accuracy range of</td></tr><tr><td>p_rw_range</td><td>p_twist</td><td>p_end</td><td>min_p_rw</td><td> $\mathtt { p \_ b i r t h }$ </td></tr><tr><td>unknotting</td><td>1.10</td><td>18.3</td><td>0.32</td><td>0.69</td><td>0.61</td><td></td></tr><tr><td>ribbon</td><td>1.10</td><td>9.4</td><td>0.91</td><td>0.80</td><td>0.89</td><td></td></tr><tr><td>slice</td><td>1.00</td><td>10.4</td><td>0.92</td><td>0.80</td><td>0.56</td><td>2.01</td></tr><tr><td>strong ribbon</td><td>1.20</td><td>12.6</td><td>0.70</td><td>0.30</td><td>0.83</td><td></td></tr><tr><td>strong slice</td><td>0.80</td><td>6.7</td><td>0.34</td><td>0.20</td><td>0.39</td><td>0.37</td></tr></table>

Table 7. Accuracy ranges associated with each parameter, computed on the evaluations of the Bayesian optimisation, compared with the noise floor (coarse validation set, 200 units per link). All values are expressed in percentage points.

A plausible explanation for this behaviour is the following. The parameters p\_twist, p\_end, and p\_birth determine the behaviour of the random walker conditional on that component being selected, and under the tuned schedules of Table 8, the random walker controls only a handful of the agent’s steps per episode. The RW-specific parameters, therefore, have few opportunities to afect the final accuracy. The exception is p\_birth in the slice framework, where more unknot births steadily lower the accuracy, from 59.7% in the lowest group to 57.7% in the highest. This agrees with the tuned random walker, whose p\_birth is pinned near zero. In the strong slice framework, p\_birth has no measurable efect. We nevertheless keep a small p\_birth in the slice framework (0.05 in the adopted configuration) so that the moves of the random walker also include the unknot birth, the move that distinguishes the slice framework from the ribbon one. The efect of min\_p\_rw is also below the noise floor, so we kept the value of the adopted configuration, which is at most 0.023.

Optimal mixing rate and decay. The mixing rate decides how much of the search goes into random exploration. This pays of only when the budget leaves room for many attempts per link, so the best mixing rate depends on the budget, while the other parameters only shape the few moves that the random walker makes. Indeed, among the ten configurations re-evaluated above (the five best of each optimisation run), the accuracy at 800 units per link grows with the mixing rate, with correlations between +0.42 and +0.90, whereas at the 200 units per link of the optimisation, high mixing rates lower it.

The parameters p\_twist, p\_end, p\_birth, and min\_p\_rw at 800 units per link made no diference. We therefore chose the mixing rate by a grid search at the budget of the final benchmark, together with step\_decay, which sets how quickly the random walker hands control over to the policy and may interact with it. The optimisation had held both decays fixed, at step\_decay = 0.9 and end\_decay = 0.75. Every episode starts with the same mixing rate

$$
k \in \{ 0 . 0 5 , 0 . 1 5 , 0 . 3 0 , 0 . 6 0 , 0 . 8 0 , 1 . 0 0 \} ,
$$

crossed with

$$
\mathsf { s t e p \mathrm { _ d e c a y } } \in \{ 0 . 1 5 , 0 . 3 0 , 0 . 6 0 , 0 . 9 0 \} ,
$$

except 0.90 for $k \geq 0 . 8 0$ , since at $k = 0 . 6 0$ it was already the worst decay rate on every framework, and every other parameter is held at the value selected above. Each of the resulting 22 grid points was evaluated at least 4 times on the coarse validation set at 800 units per link.

The two rates interact: a faster decay helps more at a high mixing rate than at a low one. On unknotting, ribbon, and on the two strong frameworks, the gain from the fastest decay is between 2.2% and 9.4% larger at $k = 0 . 6 0$ than at $k = 0 . 0 5$ , by at least 3 standard errors. On slice, it goes in the same direction, by about 1.5%, but within 2 standard errors.

We then re-evaluated the two best grid points (with the range p\_rw\_range = [k, k] of zero width) of each framework, together with the configuration selected by the optimiser, five times on the full validation set with fresh random seeds, and adopted a grid point only if it beat that configuration by more than one standard error. This happened on all five frameworks, with gains between +2.2% and +6.1%.

As a last step to find the best mixed parameters, we tested ranges [a, b] with $a < b$ , letting step\_decay vary as well. The accuracies of the ranges are close to the mean of the grid accuracies over [a, b] at the same decay rate; there is however a small gain growing with the width of the range. We evaluated twelve ranges per framework four times on the coarse validation set at 800 units per link, and re-evaluated the three best ones, against the grid point selected above with $\mathtt { p \_ r w \_ r a n g e } = [ k , k ]$ , five times on the full validation set with fresh random seeds. On four frameworks, the best of these configurations was more accurate than that grid point, by between 0.2% and 0.9%, and on strong ribbon the two tied ( 0.03%), always within two standard errors. The two choices are therefore equivalent, and we adopt the ranges. In the final configuration, step\_decay is 0.15, the lowest value of the grid, on unknotting and ribbon, and 0.30 on slice, strong ribbon, and strong slice. Testing step\_decay separately at 0 and 0.05 on the first two showed no efect, and we keep 0.15 as the tuned value. The end\_decay parameter was tested separately at 0.15, 0.45, 0.75, and 1.0, and showed no significant efect on the score of the mixed agent. It therefore stayed at 0.75 throughout.

Tuned defaults. Table 8 collects the tuned parameters of the mixed agent for each framework.
<table><tr><td>Framework</td><td>p_rw_range</td><td>step_decay</td><td>min_p_rw</td><td>p_birth</td><td>Accuracy</td></tr><tr><td>unknotting</td><td>(0.70, 1.00)</td><td>0.15</td><td>0.022</td><td></td><td> $8 6 . 8 \pm 0 . 3 \%$ </td></tr><tr><td>ribbon</td><td>(0.90, 1.00)</td><td>0.15</td><td>0.006</td><td></td><td> $8 1 . 9 \pm 0 . 3 \%$ </td></tr><tr><td>slice</td><td>(0.80, 0.90)</td><td>0.30</td><td>0.0003</td><td>0.050</td><td> $8 1 . 0 \pm 0 . 4 \%$ </td></tr><tr><td>strong ribbon</td><td>(0.75, 1.00)</td><td>0.30</td><td>0.0025</td><td></td><td> $4 9 . 0 \pm 0 . 6 \%$ </td></tr><tr><td>strong slice</td><td>(0.70, 1.00)</td><td>0.30</td><td>0.023</td><td>0.007</td><td> $3 4 . 0 \pm 0 . 7 \%$ </td></tr></table>

Table 8. Tuned mixed agent parameters per framework. The accuracy is measured on the validation set at 800 units per link, as mean and standard error over five evaluations with fresh random seeds.

Three remarks on these values are in order. First, p\_twist and p\_end are omitted because they showed no detectable efect under this analysis (Table 7). Any value in the search range performs equally well, and the optimiser’s choice carries no information. As remarked above, end\_decay stays at 0.75 throughout.

Second, under the tuned schedules, the random walker acts almost only at the start of an episode. The random walker takes the first action of an episode with probability k, the second with probability k step\_decay, and so on, until the floor min\_p\_rw is reached, so the expected number of random actions per episode, $k / ( 1 - { \mathsf { s t e p } } _ { - } { \mathsf { d e c a y } } )$ , ranges from 1.0 for unknotting to 1.25 for strong ribbon. The tuned mixed agent therefore randomises the first one or two actions of an episode and then follows the RL policy. Since the first action of an episode can only start a band or, in the slice frameworks, add an unknot, this also explains why p\_twist and p\_end carry no signal.

Lastly, the accuracies in the last column are not comparable across frameworks, since they are measured on diferent validation sets and refer to diferent invariants. The strong frameworks score much lower than the others here, as they do in the pure RL and pure RW cases.

A possible explanation. The above discussion suggests using the mixed RL+RW agent on every framework. Indeed, this was also confirmed by the analysis we report in Section 4.4. We can ofer a possible explanation for this. It is instructive to compare our findings with the results reported by Gukov, Halverson, Manolescu, and Ruehle [Guk+25]. There, the best-performing agent at finding ribbon discs was a Bayesian-optimised random walker, with the trained RL policy slightly behind it. In our setting, the opposite holds when the agents are given the same number of steps. As we discuss in Section 4.4, our RL agents then outperform the optimised random walkers on every framework. Given the same computing time instead, the walker overtakes the pure policy on the ribbon and slice frameworks and draws level with it on the others, and only the mixed agent stays ahead of the random walker on unknotting and on the strong frameworks (Section 4.4.1). Note that the two benchmarks measure diferent quantities (theirs is an accuracy computed on knots known to be ribbon), so the comparison should be read qualitatively only. We believe the diference is due to the wider action set available to our agents, which lets the RL policy choose the right sequence of moves more reliably.

The benefit of mixing the two agents can be explained in an analogous way by claiming that the random walker helps the RL agent where it struggles the most. It is most visible where the policy is weakest. On the links with the lowest invariant value of every framework except strong ribbon, the pure RL policy is less accurate than the tuned random walker (for instance, 44.0% against 56.6% on the ribbon test links of genus zero), and the mixed agent beats both (64.6%). These are the links on which many cheap random attempts might pay of. On strong ribbon, the pure RL policy is already more accurate than the random walker on the links of genus one (35.2% against 30.2%), and mixing helps more on the links of higher genus. After training, the policy tends to retry the same first band on a large share of the episodes of a link, and a single random action at the start is enough to diversify its attempts (Table 9). Under the tuned schedule, indeed, the random walker takes only one or two actions per episode before handing control back to the RL policy. Its benefit therefore appears to come from an early stochastic intervention, which can move the episode into a state from which the RL policy is more likely to find a successful trajectory.

<table><tr><td>Framework</td><td>Most common first band rl rl_rw</td><td>Distinct first bands per episode rl rl_rw</td></tr><tr><td>unknotting</td><td>43% 9%</td><td>0.11 0.35</td></tr><tr><td>ribbon</td><td>63% 6%</td><td>0.12 0.64</td></tr><tr><td>slice</td><td>62% 9%</td><td>0.13 0.68</td></tr><tr><td>strong ribbon</td><td>79% 9%</td><td>0.15 0.88</td></tr><tr><td>strong slice</td><td>73% 9%</td><td>0.18 0.89</td></tr></table>

Table 9. Diversity of the attempts of the pure RL policy (rl) and of the mixed agent (rl\_rw) on the development set, at 800 units per link. Most common first band: share of the episodes of a link that open with its most common first band, median over the links. Distinct first bands per episode: number of distinct first bands of a link divided by its number of episodes, averaged over the links.

## 4.4 Best agent among RW, RL, and RL+RW

To determine the best agent for each framework, we evaluated all agents on the held-out test datasets, using a nominal evaluation budget of 800 units per link. The same analysis was performed at budgets of 200, 400, and 1,600 units per link, without changing the ranking of the agents. We therefore report only the results of the analysis at 800 units per link.

Benchmark setup. Alongside the three agents discussed above (which we call bo\_rw, rl, and rl\_rw), we also tested a fourth agent, naive\_rw: a non-optimised random walker with equal weights.<sup>2</sup> This agent is useful as a baseline. The metric is the accuracy on the test datasets, which were untouched by model selection and parameter tuning. Each agent was run five times with diferent random seeds, and we report the mean accuracy over those runs, together with its standard deviation.

Because all four agents were evaluated on the same links, we computed their diferences linkby-link. This paired comparison controls for variation in the intrinsic dificulty of individual links. Links of the same dataset may share a prime summand, however, so their diferences are not independent. We therefore estimate the standard error of the mean diference allowing for correlation between any two links that share a prime. Writing $e _ { i }$ for the deviation of the diference on link i from its mean over the N links of the dataset, we set

$$
\mathrm { S E } ^ { 2 } = \frac { 1 } { N ^ { 2 } } \sum _ { i } \sum _ { j \sim i } e _ { i } e _ { j } ,
$$

where $j \sim i$ means that $j = i$ or that the links i and j share a prime. This reduces to the usual standard error when no two links share a prime. Clustering the links by the connected components of the relation is not possible here, as in the strong frameworks, connected sums join all the test links into a single component. Accounting for the randomness of the agents, the finite size of the test sets, and the shared primes, the standard error of the diferences discussed below lies between 0.8% and 4.3%, and every one of these diferences is at least 3.7 standard errors away from zero.

Benchmark result. Table 10 summarises the benchmark results. The ordering is the same in every framework: naive\_rw bo\_rw rl < rl\_rw. Table 11 reports the three successive accuracy diferences.
<table><tr><td rowspan="2">Framework</td><td colspan="4">Accuracy (%) of</td><td rowspan="2">Best agent</td></tr><tr><td> $\mathtt { n a i v e \_ r w }$ </td><td> $\mathtt { b o \_ r w }$ </td><td>rl</td><td> $\mathbf { r } \mathbf { 1 \_ r } \mathbf { \tilde { w } }$ </td></tr><tr><td>unknotting</td><td> $\overline { { 8 . 4 \pm 0 . 8 } }$ </td><td> $\overline { { 5 8 . 6 \pm 0 . 9 } }$ </td><td> $8 2 . 4 \pm 0 . 8$ </td><td> $\mathbf { 8 7 . 4 \pm 0 . 7 }$ </td><td> $\mathbf { r } \mathbf { 1 } _ { - } \mathbf { r } \mathbf { w }$ </td></tr><tr><td>ribbon</td><td> $1 0 . 2 \pm 0 . 9$ </td><td> $5 4 . 6 \pm 1 . 1$ </td><td> $7 3 . 3 \pm 0 . 4$ </td><td> ${ \bf 8 2 . 1 \pm 0 . 5 }$ </td><td> $\mathbf { r } \mathbf { 1 } _ { - } \mathbf { r } \mathbf { w }$ </td></tr><tr><td>slice</td><td> $9 . 8 \pm 0 . 5$ </td><td> $5 5 . 2 \pm 2 . 2$ </td><td> $6 8 . 9 \pm 0 . 5$ </td><td> ${ \bf 8 0 . 6 \pm 1 . 1 }$ </td><td> $\mathbf { r } \mathbf { 1 } _ { - } \mathbf { r } \mathbf { w }$ </td></tr><tr><td>strong ribbon</td><td> $1 . 2 \pm 0 . 3$ </td><td> $1 2 . 7 \pm 1 . 2$ </td><td> $3 7 . 9 \pm 1 . 7$ </td><td> ${ \bf 5 1 . 9 \pm 0 . 8 }$ </td><td> $\mathbf { r } \mathbf { 1 } _ { - } \mathbf { r } \mathbf { w }$ </td></tr><tr><td>strong slice</td><td> $1 . 4 \pm 0 . 5$ </td><td> $1 2 . 1 \pm 0 . 7$ </td><td> $2 7 . 6 \pm 1 . 3$ </td><td> ${ \bf 3 6 . 7 \pm 1 . 2 }$ </td><td> $\mathbf { r } \mathbf { 1 } _ { - } \mathbf { r } \mathbf { w }$ </td></tr></table>

Table 10. Accuracy of the four agents on the test set of each framework, expressed as percentages, with mean and standard deviation over five runs at 800 units per link. The highest mean accuracy is in bold.

<table><tr><td>Framework</td><td>Tuning the walker  $\left( \mathtt { b o \_ r w - n a i v e \_ r w } \right)$ </td><td> $\overline { { \mathrm { R W } \to \mathrm { R L } } }$   $( \mathtt { r } 1 - \mathtt { b } 0 \_ { \mathtt { r } \mathtt { w } } )$ </td><td> $\overline { { \mathrm { R L } \to \mathrm { R L } \mathrm { + R W } } }$   $( \pmb { \mathrm { r } } \pmb { \mathrm { 1 } } _ { - } \pmb { \mathrm { r w } } - \pmb { \mathrm { r } } \pmb { \mathrm { 1 } } )$ </td></tr><tr><td>unknotting</td><td>+50.2</td><td>+23.8</td><td>+5.1</td></tr><tr><td>ribbon</td><td>+44.4</td><td>+18.7</td><td>+8.8</td></tr><tr><td>slice</td><td>+45.4</td><td>+13.8</td><td>+11.7</td></tr><tr><td>strong ribbon</td><td>+11.6</td><td>+25.2</td><td>+14.0</td></tr><tr><td>strong slice</td><td>+10.6</td><td>+15.5</td><td>+9.2</td></tr></table>

Table 11. Accuracy diferences between the various models, in percentage points. Recall that the margin of error on these diferences lies between 0.8 and 4.3 points.

Tuning the random walker improves accuracy by between +10.6% and +50.2% over the naive walker, while replacing the tuned walker with the RL policy adds a further +13.8% to +25.2%. Both steps are significant in every framework. On unknotting, ribbon, and slice, where the tuned random walker already solves more than half of the links, tuning it is the largest improvement. In the two strong frameworks, the learned policy contributes more than the tuning.

Mixing the walker into the policy improves accuracy further on every framework, by between +5.1% on unknotting and +14.0% on strong ribbon. As expected from the discussion in Section 4.3.1, on every framework except strong ribbon the gain is large on the links with the lowest invariant value, where the pure policy is less accurate than the tuned random walker, and the mixed agent is more accurate than both. The mixed agent also gains on the other classes of links and loses significantly on none.

A higher accuracy, however, does not mean that the mixed agent solves every link that the RL policy solves. The two agents solve measurably diferent sets of links: in a single run on ribbon, 100 test links are solved only by the mixed agent and 29 only by the pure RL policy.

## 4.4.1 Equal computing time

The benchmark above gives every agent the same number of steps, not the same computing time. A step of the RL policy is more expensive than a step of the random walker, since it requires the computation of the input features and a forward pass of the policy network. On the machines used for the benchmark, a policy step costs between 4.0 and 6.8 times a random walker step, depending on the framework. The mixed agent costs no more than the pure RL policy.

To check whether the ranking survives when the computing time is matched, we repeated the comparison on the test datasets, giving the tuned random walker the number of steps per link that matches, on the same machine, the running time of the RL policy at 800 steps per link. The random walker thus received between about 3,500 and 5,700 steps per link on average, depending on the framework, while the RL policy and the mixed agent kept 800. The results are in Table 12.

<table><tr><td rowspan="2">Framework</td><td rowspan="2">bo_rw steps per link</td><td colspan="3">Accuracy (%) of</td><td rowspan="2"> $\mathtt { b o \_ r w } - \mathtt { r } \mathtt { 1 }$ </td><td rowspan="2"> $\mathtt { b o \_ r w } - \mathtt { r } \mathtt { 1 \_ r w }$ </td></tr><tr><td> $\mathtt { b o \_ r w }$ </td><td>rl</td><td> $\pmb { \mathrm { r } } \pmb { \mathrm { l } } _ { - } \pmb { \mathrm { r } } \pmb { \mathrm { w } }$ </td></tr><tr><td>unknotting</td><td>3452</td><td> $\overline { { 8 1 . 6 \pm 1 . 0 } }$ </td><td> $\overline { { 8 2 . 3 \pm 0 . 8 } }$ </td><td> $\mathbf { 8 7 . 4 \pm 0 . 7 }$ </td><td> $- 0 . 7 \pm 1 . 6$ </td><td> $- 5 . 9 \pm 1 . 4$ </td></tr><tr><td>ribbon</td><td>5196</td><td> ${ \bf 8 4 . 3 \pm 0 . 9 }$ </td><td> $7 3 . 3 \pm 0 . 4$ </td><td> $8 2 . 1 \pm 0 . 5$ </td><td> $+ 1 1 . 0 \pm 2 . 1$ </td><td> $+ 2 . 2 \pm 1 . 8$ </td></tr><tr><td>slice</td><td>5725</td><td> ${ \bf 8 4 . 4 \pm 0 . 5 }$ </td><td> $6 8 . 9 \pm 0 . 5$ </td><td> $8 0 . 6 \pm 1 . 1$ </td><td> $+ 1 5 . 5 \pm 1 . 9$ </td><td> $+ 3 . 8 \pm 1 . 5$ </td></tr><tr><td>strong ribbon</td><td>5088</td><td> $2 9 . 4 \pm 1 . 0$ </td><td> $3 7 . 9 \pm 1 . 7$ </td><td> ${ \bf 5 1 . 9 \pm 0 . 8 }$ </td><td> $- 8 . 5 \pm 4 . 8$ </td><td> $- 2 2 . 5 \pm 4 . 7$ </td></tr><tr><td>strong slice</td><td>3999</td><td> $2 5 . 7 \pm 1 . 5$ </td><td> $2 7 . 6 \pm 1 . 3$ </td><td> ${ \bf 3 6 . 7 \pm 1 . 2 }$ </td><td> $1 . 9 \pm 4 . 6$ </td><td> $1 1 . 1 \pm 4 . 1$ </td></tr></table>

Table 12. The agents on the test set when the tuned random walker is given the computing time of the RL policy. Accuracies are mean and standard deviation over five runs. Diferences are in percentage points, with the standard error of the paired comparison link-by-link, allowing for links that share a prime.

On the two strong frameworks, the mixed agent remains the best, 22.5% and 11.1% ahead of the walker, and on unknotting, it leads by 5.9%. On ribbon and slice, instead, the random walker with matched time is at least as accurate as the mixed agent, by 2.2% and 3.8%, and much more accurate than the pure RL policy, by 11.0% and 15.5%, respectively. On ribbon, the random walker’s lead over the mixed agent is only 1.2 standard errors, so we consider it a tie. The pure RL policy is not significantly more accurate than the tuned random walker on any framework. With the matched computing time on unknotting and strong slice, the two agents difer by less than one standard error. On strong ribbon, the pure policy is ahead of the tuned walker by 8.5%, which is only 1.8 standard errors.

By looking at the link classes, we can explain this diference. The random walker wins on links of genus zero and one, and on those of unknotting number at most three. The policy wins where the answer requires a long and specific sequence of moves. On the strong frameworks, the walker solves none of the 20 test links of genus five, even with more than four times as many steps, while the policy solves between 69% and 89% of them, depending on the framework and on the mixing. On ribbon and slice, 44% of the test links have genus zero or one.

In general, we run both the random walker and the mixed agent on the same dataset to maximise the chance of finding new upper bounds.

## 5 New bounds

We report, per database tier, how much the known values were improved, splitting the contribution into theoretical (the bounds of Appendix B) and agentic (an explicit band path found by the RW or RL+RW agent). We denote a value to be exact when its lower and upper bounds coincide, bounded when it is confined to an interval, and unknown when nothing is on record. A value is agentic only when an agent’s witness strictly improved the previously known upper bound.

Each tier below is therefore split into two parts, one per stage. First, the theoretical bounds part: beginning from the starting knowledge provided by the external databases, we apply a bounds-only pipeline to it. This applies all the results reported in Appendix B and verifies if better lower and upper bounds exist. Second, the agent bounds part: this starts where the theoretical pipeline ends and reports what the agent adds on top.

## 5.1 Building the reference bounds dataset

Several public sources report values for slice genera and unknotting numbers, and they do not always agree. For some links, one source’s bounds are sharper than another’s. We therefore assembled a single, up-to-date dataset with the best known bounds on the invariants we target by pulling together every public knot/link database we are aware of and merging them with explicit provenance. This dataset plays two roles: it is the reference against which any improvement reported below is measured, and it is what the agents are trained and evaluated on.

Sources. We ingest the databases below, splitting them into three sections: knots up to 13 crossings, 14-crossing knots, and at least two-component links up to 11 crossings.

[KnotInfo] and [LinkInfo], our main references: data on the unknotting number for prime knots up to 13 crossings and at least two-component prime links up to 10 crossings, and on the slice genus for prime knots up to 13 crossings and at least two-component prime links up to 11 crossings;

[KnotAtlas]: data on the unknotting number for prime knots up to 11 crossings and the slice genus for prime knots up to 10 crossings;

[Brittenham]: data on the unknotting number for prime knots up to 14 crossings;

[DG25]: data on the slice genus for prime knots up to 19 crossings, where they determine whether a knot is slice, not slice $( g _ { 4 } \ge 1 )$ , or undetermined $\left( g _ { 4 } \ge 0 \right)$ ;

[Jab26]: data on the unknotting number and the slice genus for prime knots up to 13 crossings;

[Lee26]: data on the unknotting number for prime knots up to 13 crossings.

KnotInfo for prime knots up to 13 crossings and LinkInfo for at least two-component prime links up to 11 crossings are taken as the main references. In particular, we use names, PD codes, and orientation conventions from there. For 14-crossing prime knots, the SnapPy census supplies the names and PD codes for all 46,972 knots. Crossing signs are part of the data: a (PD code, crossing signs) pair uniquely determines the oriented diagram, and signs are preserved across all computations.

Merging the datasets. After collecting all the data, we merge the various sources together. This means that for an invariant whose exact value is not known and for which we have only an upper and a lower bound, we compare the diferent sources and keep the best bounds, recording for each invariant a \_lower\_bound\_source and \_upper\_bound\_source provenance column naming which database set the tightest bound on each side.

Discrepancies in the datasets. During the merge, two discrepancies emerged among the databases. First, the unknotting number of $1 0 _ { 6 }$ is recorded as having unknotting number 3 in various databases, while from [BH26] we know its value is either 2 or 3.

Second, the unknotting numbers of three further knots disagree between [KnotInfo] and [Brittenham]: these are $1 0 _ { 1 6 2 } , 1 0 _ { 1 6 4 }$ , and $1 0 _ { 1 6 5 }$ . We think this is due to the presence of the Perko pair in Brittenham’s database, and we adopt the KnotInfo data in the merged release.

Narrowing by theoretical bounds. After merging the various references into a single dataset, we ran a bounds-only pipeline that narrows the recorded intervals where possible using the classical theoretical bounds discussed in Appendix B. Note that since $g _ { 4 } \leq u$ and $g _ { 4 } \leq g _ { 4 } ^ { * }$ , all the lower bounds for the slice genus are also used for the strong slice genus and the unknotting number. For a positive diagram, Theorem B.29 determines the slice genus exactly, which is a lower bound on the unknotting number. Whenever it matches the recorded unknotting upper bound, we can determine the unknotting number exactly.

Narrowing via agents. The narrowed dataset is the input that the agents start from, so any value the agent later pins down is genuinely beyond what these classical bounds already provide. We present in the next sections the results we obtained from the theoretical bounds and from the agents’ upper bounds, splitting the improvements by contribution.

## 5.2 At least two-component prime links with up to 11 crossings

In the dataset, there are 4,188 oriented prime links with at least two components (1,424, disregarding orientation). Note that we did not have any value for the strong slice genus of any link, nor the unknotting number for all the 11-crossings links (2,920 oriented links).

## 5.2.1 Theoretical bounds

The bounds-only pipeline acts unevenly across the three invariants. For the unknotting number, it works almost entirely on the 2,924 links carrying no recorded value at all. It determines 1,132 of them outright and leaves the other 1,792 with a bounded interval, so that after this stage, no unknotting number is completely unknown. For the slice genus, which starts out with only 407 non-exact cases, it determines 19 of them. For the strong slice genus, which no database tabulates, it supplies the entire starting point. 3,456 of the links have two components with non-zero pairwise linking number, hence $g _ { 4 } ^ { * } = \infty$ (counted as an exact value). The remaining 732 have vanishing pairwise linking numbers and therefore finite $g _ { 4 } ^ { * }$ . These remain all non-exact after the bounds are computed.

## 5.2.2 Agent bounds

Starting from the narrowed dataset above, the best-performing RW and RL+RW agents for each framework try to find explicit movies that realise a lower invariant value than the known upper bound. Table 14 shows the improvements made by the agents. When studying the slice genera,

<table><tr><td rowspan="2">Invariant</td><td colspan="3">Start</td><td colspan="2">Theoretical contribution</td><td colspan="2">After theory</td></tr><tr><td>exact</td><td>bounded</td><td>unknown</td><td>made exact</td><td>better bounds</td><td>exact</td><td>bounded</td></tr><tr><td>unknotting u</td><td>1,264</td><td>0</td><td>2,924</td><td>1,132</td><td>1,792</td><td>2,396</td><td>1,792</td></tr><tr><td>slice genus g4</td><td>3,781</td><td>407</td><td>0</td><td>19</td><td>0</td><td>3,800</td><td>388</td></tr><tr><td>strong slice genus 9*</td><td>0</td><td>0</td><td>4,188</td><td>3,456</td><td>732</td><td>3,456</td><td>732</td></tr></table>

Table 13. At least two-component oriented prime links with up to 11 crossings: contribution of the theoretical bounds.

allowing the birth of unlinked unknot components did not improve our upper bounds. Hence, we report only on the slice genus.
<table><tr><td></td><td colspan="2">After theory</td><td colspan="2">Agent contribution</td><td colspan="2">Final</td></tr><tr><td>Invariant</td><td>exact</td><td>bounded</td><td>made exact</td><td>better bounds</td><td>exact</td><td>bounded</td></tr><tr><td>unknotting u</td><td>2,396</td><td>1,792</td><td>846</td><td>934</td><td>3,242</td><td>946</td></tr><tr><td>slice genus 94</td><td>3,800</td><td>388</td><td>125</td><td>0</td><td>3,925</td><td>263</td></tr><tr><td>strong slice genus 9*</td><td>3,456</td><td>732</td><td>676</td><td>56</td><td>4,132</td><td>56</td></tr></table>

Table 14. At least two-component oriented prime links with up to 11 crossings: contribution of the agent, on top of Table 13.

## 5.2.3 Comments

We now briefly comment and provide some statistics on the results found for each invariant.

Unknotting number (11-crossing links). No table of unknotting numbers for 11-crossing at least two-component prime links was previously available (2,920 oriented links). Disregarding orientation, of the 1,007 unique 11-crossing at least two-component prime links, we determine the exact value for 690 and confine each of the remaining 317 to an interval. The 2,920 oriented 11-crossing links split into 1,978 exact values and 942 intervals, of which 810 have width one and 132 width two.

Weak slice genus. The slice genus was undetermined for 407 links. The theoretical bounds settle 19 of them, and the agent a further 125, all of which turn out to be slice. Of the latter, 4 links have 9 crossings, 23 have 10, and 98 have 11. The 263 links still undetermined are all confined to $g _ { 4 } \in \{ 0 , 1 \}$ ; i.e., the only open question left is whether they are slice.

Strong slice genus. Of the 732 links left with finite strong slice genus, the agent determines the exact value for 676 and strictly narrows the interval of the other 56. Out of those 56, 54 links are confined to $g _ { 4 } ^ { \ast } \in \{ 1 , 2 \}$ and 2 links to $g _ { 4 } ^ { \ast } \in \{ 0 , 1 \}$ . Table 15 collects the strong slice genus of the 676 links for which we determined an exact value.

<table><tr><td>g4</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td># links</td><td>4</td><td>388</td><td>238</td><td>44</td><td>2</td></tr></table>

Table 15. The 676 links of finite strong slice genus whose value was determined exactly.

## 5.3 Prime knots with up to 13 crossings

There are 12,965 prime knots with up to 13 crossings. Neither stage improved on the known data. The theoretical bounds reproduce the tabulated values without sharpening any of them, and no agent beat a known upper bound. The agents did reproduce the tabulated upper bound on 12,865 (99.2% of the total) of the knots for the unknotting number and on 12,964 (99.9% of the total) for the slice genus. On the remaining 100 knots (unknotting number) and on 12a<sub>631</sub> (ribbon genus), the best answer stayed one above the tabulated value.

## 5.4 Prime knots with 14 crossings

There are 46,972 prime knots with 14 crossings. Here, the two stages divide the work by invariant. The improvements on the slice genus side come almost entirely from the agent, while on the unknotting number side they come entirely from theory.

## 5.4.1 Theoretical bounds

While no slice genus dataset for 14-crossing knots was available, Dunfield and Gong [DG25] do report sliceness for 1,194 of them, and slice obstruction for 408, which we record as the bound $g _ { 4 } \geq 1$ . That is the whole of the starting knowledge, and the only exact value it supplies is $g _ { 4 } = 0 .$ The theoretical bounds then determine a further 1,671 values exactly and strictly narrow the interval of 44,107 more, bringing the census to 2,865 exact values.

For the unknotting number, the upper bounds of all 46,972 knots come from [Brittenham], and the new contribution is purely on the lower bound side. The obstructions give a non-trivial lower bound for 1,288 knots, of which 1,097 become exactly determined, the new bound meeting Brittenham’s upper bound. Table 16 shows the contribution of the theoretical pipeline. Note that no value remains completely unbounded after the bounds are computed.

<table><tr><td></td><td colspan="3">Start</td><td colspan="2">Theoretical contribution</td><td colspan="2">After theory</td></tr><tr><td>Invariant</td><td>exact</td><td>bounded</td><td>unknown</td><td>made exact</td><td>better bounds</td><td>exact</td><td>bounded</td></tr><tr><td>slice genus g4</td><td>1,194</td><td>408</td><td>45,370</td><td>1,671</td><td>44,107</td><td>2,865</td><td>44,107</td></tr><tr><td>unknotting u</td><td>19,781</td><td> $^ \mathrm { 2 7 , 1 9 1 }$ </td><td>0</td><td>1,097</td><td>191</td><td>20,878</td><td>26,094</td></tr></table>

Table 16. Prime knots with 14 crossings: contribution from the theoretical bounds.

## 5.4.2 Agent bounds

The agent contributes only to the slice genus. It makes exact 43,167 of the 44,107 intervals left open above and strictly narrows 914 more, bringing the census to an exact slice genus for 46,032 knots, with each of the remaining 940 narrowed to an interval of at most two values.

Again, allowing the birth of unlinked unknot components did not improve our upper bounds. For the unknotting number, it finds no unknotting sequence that improves on Brittenham’s upper bounds, so that row remains unchanged.

<table><tr><td rowspan="2">Invariant</td><td colspan="2">After theory</td><td colspan="2">Agent contribution</td><td colspan="2">Final</td></tr><tr><td>exact</td><td>bounded</td><td>made exact</td><td>better bounds</td><td>exact</td><td>bounded</td></tr><tr><td>slice genus  $g _ { 4 }$ </td><td>2,865</td><td>44,107</td><td>43,167</td><td>914</td><td>46,032</td><td>940</td></tr><tr><td>unknotting u</td><td>20,878</td><td>26,094</td><td></td><td></td><td>20,878</td><td>26,094</td></tr></table>

Table 17. Prime knots with 14 crossings: what the agent contributes on top of Table 16.

## 5.4.3 Comments

We now briefly comment on and provide some statistics regarding the results found for each invariant.

Unknotting number lower bounds source. The sources of the new theoretical bounds that improve the previously known values are summarised in Table 18.

<table><tr><td>New lower bound source</td><td># knots</td></tr><tr><td>τ and  $\nu _ { s }$ </td><td>957</td></tr><tr><td>positive and g4 = upper bound</td><td>327</td></tr><tr><td>Murasugi-Tristram</td><td>4</td></tr><tr><td>total</td><td>1,288</td></tr></table>

Table 18. New lower bounds on the unknotting number for 14-crossing prime knots, with source.

Slice genus distribution. In the following tables, we can observe the distribution of the values found for the slice genus. Table 19 summarises the 46,032 exact values, where the 1,194 knots with $g _ { 4 } = 0$ are exactly the slice knots of Dunfield and Gong [DG25]; every value above 0 is new.

<table><tr><td> $g _ { 4 }$ </td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td># knots</td><td>1,194</td><td>29,531</td><td>10,178</td><td>3,822</td><td>1,088</td><td>202</td><td>17</td></tr></table>

Table 19. The 46,032 knots with 14 crossings whose slice genus was determined exactly.

Table 20 summarises the intervals of values left undetermined for the 940 remaining knots by counting how many knots have each interval.

<table><tr><td>interval</td><td>[0, 1]</td><td>[1, 2]</td><td>[2, 3]</td></tr><tr><td># knots</td><td>10</td><td>919</td><td>11</td></tr></table>

Table 20. The 940 knots with 14 crossings whose slice genus was narrowed to a short interval, but not determined exactly.

## 5.5 Non-additivity of the unknotting number

Our original motivation for developing the unknotting framework was to find counterexamples to the additivity of the unknotting number. Following the results of Brittenham and Hermiller [BH26], we ran the unknotting agents on some connected sums of symbiont knots. In their paper, they call two knots $K$ and $K ^ { \prime }$ symbionts if $u ( K \sharp K ^ { \prime } ) < u ( K ) + u ( K ^ { \prime } )$ . They showed that $4 _ { 1 }$ and $9 _ { 1 0 }$ are symbionts; see [BH26, Theorem 1.1]. Furthermore, $4 _ { 1 }$ is Gordian adjacent to $8 _ { 4 } , ~ 9 _ { 8 } , ~ 9 _ { 1 7 } , ~ 9 _ { 4 0 } , ~ 1 0 _ { 4 1 } , ~ 1 0 _ { 8 9 } , ~ 1 0 _ { 1 1 5 } , ~ 1 0 _ { 1 2 1 } , ~ 1 0 _ { 1 4 0 }$ , and $1 0 _ { 1 4 4 }$ , as $4 _ { 1 }$ is contained in a minimal unknotting sequence for each of them; see the set S<sub>2</sub> in [BH26]. It now follows from [BH26, Lemma 4.1] that $8 _ { 4 } , 9 _ { 8 } , 9 _ { 1 7 } , 9 _ { 4 0 } , 1 0 _ { 4 1 } , 1 0 _ { 8 9 } , 1 0 _ { 1 1 5 } , 1 0 _ { 1 2 1 } , 1 0 _ { 1 4 0 }$ , and $1 0 _ { 1 4 4 }$ are symbionts of $9 _ { 1 0 }$ Since $4 _ { 1 }$ is amphichiral, $\mathrm { m } ( 9 _ { 1 0 } )$ is a symbiont of $4 _ { 1 }$ as well, and hence of the same knots.

Our unknotting agent found that

$$
8 _ { 4 } \sharp 9 _ { 1 0 } , 9 _ { 8 } \sharp 9 _ { 1 0 } , 9 _ { 1 7 } \sharp 9 _ { 1 0 } , 9 _ { 4 0 } \sharp 9 _ { 1 0 } , 1 0 _ { 4 1 } \sharp 9 _ { 1 0 } , 1 0 _ { 8 9 } \sharp 9 _ { 1 0 } ,
$$

$$
1 0 _ { 1 1 5 } \sharp 9 _ { 1 0 } , 1 0 _ { 1 2 1 } \sharp 9 _ { 1 0 } , 1 0 _ { 1 4 0 } \sharp 9 _ { 1 0 } , 1 0 _ { 1 4 4 } \sharp 9 _ { 1 0 } , 1 0 _ { 4 1 } \sharp \mathrm { m } \big ( 9 _ { 1 0 } \big )
$$

all have unknotting number at most four. $\mathrm { A s ~ } 8 _ { 4 } , 9 _ { 8 } , 9 _ { 1 7 } , 9 _ { 4 0 } , 1 0 _ { 4 1 } , 1 0 _ { 8 9 } , 1 0 _ { 1 1 5 } , 1 0 _ { 1 2 1 } , 1 0 _ { 1 4 0 }$ , and $1 0 _ { 1 4 4 }$ have unknotting number two, and $u ( \mathfrak { g } _ { 1 0 } ) = u ( \mathrm { m } ( \mathfrak { g } _ { 1 0 } ) ) = 3$ , we recover the non-additivity of the unknotting number for the above connected sums. Some of the prime summands were present in the datasets on which we trained or tested the models, while the exact connected sums were not. The exact presence of the knots in the splits is reported in Table 21.

<table><tr><td>Split</td><td>Knot</td></tr><tr><td>training</td><td> $\overline { { 8 _ { 4 } \sharp \mathrm { m } ( 9 _ { 2 5 } ) } }$   $9 _ { 8 } \sharp \mathsf { 9 } _ { 3 8 }$   $1 0 _ { 1 1 5 } \sharp 1 0 _ { 1 2 4 }$ </td></tr><tr><td>validation</td><td> $1 0 _ { 4 1 }$   $1 0 _ { 4 1 } \sharp 1 0 _ { 7 4 }$ </td></tr><tr><td>coarse validation</td><td></td></tr><tr><td>development testing</td><td></td></tr><tr><td></td><td> $1 0 _ { 1 4 0 } \sharp 1 0 _ { 1 5 2 }$   $\mathsf { 9 } _ { 1 0 } \sharp 1 0 _ { 8 4 }$   $9 _ { 1 0 } \sharp \mathrm { m } ( 1 0 _ { 1 4 } )$   $9 _ { 1 0 } \sharp \mathrm { m } \big ( 1 0 _ { 1 5 2 } \big )$ </td></tr></table>

Table 21. Rows of the unknotting splits containing $8 _ { 4 } , 9 _ { 8 } , 9 _ { 1 0 } , 9 _ { 1 7 } , 9 _ { 4 0 } , 1 0 _ { 4 1 }$ 2 $1 0 _ { 8 9 } , \ 1 0 _ { 1 1 5 } , \ 1 0 _ { 1 2 1 } , \ 1 0 _ { 1 4 0 } , \ \mathrm { o r } \ 1 0 _ { 1 4 4 }$ (either chirality) as the knot itself or as a connected summand.

Unknotting sequences. We comment briefly on the unknotting sequences we found. Each band but one is of length one; i.e., the agent performs a crossing change by passing the band through an edge contained in the boundary of the same face where it started the band. The exception is the last band of the sequence for $1 0 _ { 4 1 } \sharp \mathrm { m } ( 9 _ { 1 0 } )$ , which first passes under an edge and ends in the adjacent face. After each band, the diagram is then simplified before starting a new one. In half of the bands, the band corresponds to exactly one diagrammatic crossing change.

Recall the unknotting sequence proposed in the proof of [BH26, Theorem 1.1], which is

$$
4 _ { 1 } \sharp 9 _ { 1 0 } \to \mathrm { m } \big ( \mathtt { K } 1 5 \mathrm { n } 4 8 6 6 \big ) \to 8 _ { 1 4 } \to \mathrm { u n k n o t } .\tag{5.1}
$$

All the unknotting sequences we found reach $8 _ { 1 4 }$ , or its mirror for $1 0 _ { 4 1 } \sharp \mathrm { m } ( 9 _ { 1 0 } )$ , after three crossing changes and then the unknot. The sequences for $8 _ { 4 } \sharp 9 _ { 1 0 } , \ : 9 _ { 1 7 } \sharp 9 _ { 1 0 } , \ : 1 0 _ { 8 9 } \sharp 9 _ { 1 0 }$ , and $1 0 _ { 1 4 4 } 7 9 _ { 1 0 }$ find exactly $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$ with the first crossing change and then follow the same sequence $\left( 5 . 1 \right)$ . The sequence for $1 0 _ { 4 1 } \sharp \mathrm { m } ( 9 _ { 1 0 } )$ is the mirror image of the one for $1 0 _ { 8 9 } \sharp \boldsymbol { 9 } _ { 1 0 } \colon$ it passes through $4 _ { 1 } \sharp \mathrm { m } ( 9 _ { 1 0 } )$ K15n4866, and $\mathbf { m } ( 8 _ { 1 4 } )$ The sequences for $9 _ { 8 } \sharp \mathsf { 9 } _ { 1 0 }$ and $1 0 _ { 1 1 5 } { \sharp } 9 _ { 1 0 }$ also find $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$ with the first crossing change, but then pass through a diferent knot, K15n4005. This gives a second sequence of three crossing changes from $4 _ { 1 } \sharp \mathfrak { s } _ { 1 0 }$ to the unknot and shows that $u ( \mathrm { K } 1 5 \mathrm { n } 4 0 0 5 ) = 2 ,$ since $| \tau ( \mathrm { K } 1 5 \mathrm { n } 4 0 0 5 ) | = 2 .$ . The sequences for $9 _ { 4 0 } \sharp \mathsf { S } _ { 1 0 }$ and $1 0 _ { 4 1 } \sharp \mathsf { \mathsf { S } } _ { 1 0 }$ reach K15n4005 after two crossing changes, passing through a knot that is not in the SnapPy tables of knots up to 15 crossings. On $1 0 _ { 1 4 0 } { \sharp 9 _ { 1 0 } } \qquad $ , the mixed agent reaches m(K15n4866) after two crossing changes without passing through $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$ , while the random walker paths on $1 0 _ { 1 4 0 } \sharp \mathsf { 9 } _ { 1 0 }$ and $1 0 _ { 1 2 1 } \sharp \mathfrak { s } _ { 1 0 }$ pass through two knots not in the tables and have only the last step $\left( 8 _ { 1 4 } \right.$ to the unknot) in common with sequence (5.1). All the unknotting sequences are collected in Table 22. The sequences that difer from (5.1) are drawn step by step in Appendix C.

## Appendix A One episode in the band environment

The pipeline of a single episode in the environment of Section 3 is as follows.

1 A new link is extracted from the dataset : the episode starts   
2 The observation for the link is created   
3 We then loop until one of the following holds :   
4 - the diagram is terminal --> episode won   
5 - an illegal state is reached --> episode lost   
6 - the step budget is reached --> episode truncated   
7 Loop :   
8 - all legal actions are enumerated ( the illegal ones are masked out )   
9 - the agent samples a legal action from the policy π   
10 -- if the action extends the band ( start / over / under / twist ):   
11 ∗ only the band path data inside the observation space is updated ,   
12 while the diagram is untouched   
13 -- if the action ends the band ( end0 / end1 / end2 ):   
14 ∗ the diagram is rebuilt with the attached band   
15 ∗ the new diagram is simplified via spherogram - nim   
16 ∗ (for genus computations ) the surface bookkeeping is updated   
17 -- if the action is unknot\_birth :   
18 ∗ a new unlinked trivial component is added to the diagram   
19 - the reward is computed and the new observation is created   
20   
21 A won episode returns a certificate : an explicit sequence of moves that   
22 provides a verifiable upper bound on the invariant  
Listing 1. The pipeline of an episode.

## A.1 A worked example

We close this appendix by displaying one full episode, with all the observations printed. The episode we have chosen is exactly the one presented in Example 2.5. It builds a genus-1 surface for the trefoil (Figures 4 and 5). In the following discussion, recall that we denote a strand of a link by C{x}S{y}, where x is the label of the crossing and y identifies which of the four strands at that crossing it is: the zeroth is the incoming under-strand, and the others are numbered in increasing order, going counter-clockwise around the crossing.

<table><tr><td colspan="2">Agent Unknotting sequence</td><td></td><td></td><td>→ unknot</td></tr><tr><td>RW</td><td> $\overline { { 8 _ { 4 } \sharp 9 _ { 1 0 } } }$  →  $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$   $\left[ 1 , 4 \right] \left( 2 + 3 \right)$   $\left[ 2 , 3 \right] \left( 1 + 3 \right)$ </td><td>→ m(K15n4866) → 2</td><td> $8 _ { 1 4 }$  1</td><td>0</td></tr><tr><td>RW  $9 _ { 8 } \sharp \mathsf { 9 } _ { 1 0 }$   $\left[ 3 , 4 \right] \left( 2 + 3 \right)$ </td><td>→  $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$   $\left[ 2 , 3 \right] \left( 1 + 3 \right)$ </td><td>→  $\begin{array} { r l r } { \mathsf { K 1 5 n 4 0 0 5 } } & { { } \to } & { 8 _ { 1 4 } } \end{array}$  2</td><td>1</td><td>→ unknot 0</td></tr><tr><td>RW  $\left[ 3 , 4 \right] \left( 2 + 3 \right)$ </td><td> $9 _ { 1 7 } \sharp \mathfrak { S } _ { 1 0 } \qquad $   $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$   $\left[ 2 , 3 \right] \left( 1 + 3 \right)$ </td><td>→ m(K15n4866) → 2</td><td> $8 _ { 1 4 }$  1</td><td>→ unknot 0</td></tr><tr><td>RW  $9 _ { 4 0 } \sharp \mathfrak { S } _ { 1 0 } \qquad \to$   $\left[ 3 , 4 \right] \left( 2 + 3 \right)$ </td><td>K20 [2, 3]</td><td>→ K15n4005 → 2</td><td> $8 _ { 1 4 }$  1</td><td>→ unknot 0</td></tr><tr><td>RW  $1 0 _ { 4 1 } \sharp \mathfrak { S } _ { 1 0 } \qquad $   $\left[ 3 , 4 \right] \left( 2 + 3 \right)$ </td><td>K21 → [2,3]</td><td>K15n4005 → 2</td><td> $8 _ { 1 4 }$  1</td><td>→ unknot 0</td></tr><tr><td>RW  $1 0 _ { 8 9 } \sharp 9 _ { 1 0 } \quad \mathrm { ~ \to ~ }$   $\left[ 1 , 4 \right] \left( 2 + 3 \right)$ </td><td> $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$   $\left[ 2 , 3 \right] \left( 1 + 3 \right)$ </td><td>→ m(K15n4866) → 2</td><td> $8 _ { 1 4 }$  1</td><td>→ unknot 0</td></tr><tr><td>RW  $1 0 _ { 1 1 5 } \sharp 9 _ { 1 0 } \quad \to$   $\left[ 2 , 4 \right] \left( 2 + 3 \right)$ </td><td> $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$  →  $\left[ 2 , 3 \right] \left( 1 + 3 \right)$ </td><td>K15n4005 → 2</td><td> $8 _ { 1 4 }$  1</td><td>→ unknot 0</td></tr><tr><td>RW  $1 0 _ { 1 2 1 } \sharp \mathfrak { s } _ { 1 0 }$   $\left[ 3 , 4 \right] \left( 2 + 3 \right)$ </td><td>→ K21 → 3</td><td>K23 → 2</td><td> $8 _ { 1 4 }$  1</td><td>→ unknot 0</td></tr><tr><td>RW  $1 0 _ { 1 4 0 } \sharp \mathfrak { S } _ { 1 0 } \quad \to$   $\left[ 2 , 4 \right] \left( 2 + 3 \right)$ </td><td>K21 → [2,3]</td><td>K23 → 2</td><td>814 1</td><td>→ unknot 0</td></tr><tr><td>RL+RW  $1 0 _ { 1 4 0 } \sharp \mathfrak { S } _ { 1 0 } \quad \to$   $\left[ 2 , 4 \right] \left( 2 + 3 \right)$ </td><td>K21 [2,3]</td><td>→ m(K15n4866) → 2</td><td>814 1</td><td>→ unknot 0</td></tr><tr><td>RW  $1 0 _ { 1 4 4 } \sharp \mathfrak { S } _ { 1 0 } \quad \to$   $\left[ 3 , 4 \right] \left( 2 + 3 \right)$ </td><td> $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$   $\left[ 2 , 3 \right] \left( 1 + 3 \right)$ </td><td>→ m(K15n4866) → 2</td><td> $8 _ { 1 4 }$  1</td><td>→ unknot 0</td></tr><tr><td>RW  $\overline { { { 1 0 _ { 4 1 } } _ { \sharp } \mathrm { m } \big ( 9 _ { 1 0 } \big ) \  } }$   $\left[ 1 , 4 \right] \left( 2 + 3 \right)$ </td><td> $4 _ { 1 } \sharp \mathrm { m } ( 9 _ { 1 0 } ) \to$   $\left[ 2 , 3 \right] \left( 1 + 3 \right)$ </td><td>K15n4866 2</td><td>→ m(814) → unknot 1</td><td>0</td></tr></table>

Table 22. The unknotting sequences found by the tuned random walker (RW) and by the mixed agent (RL+RW); each arrow is one crossing change. The knot Kn is not in SnapPy’s census up to 15 crossings, found as a diagram with n crossings. Knots with the same label Kn in diferent rows are diferent as they are hyperbolic and their volumes difer. The knots K15n4005 and K15n4866 have the chirality of the diagrams SnapPy stores under these names. Below each knot is its unknotting number, or the interval known for it, whose the upper end is the number of crossing changes in the sequence, and whose lower end is the best lower bound of Appendix B. Next to a connected sum, in grey, are the unknotting numbers of its two summands, whose sum is the value that additivity would give.

The episode consists of two bands. The first one is of length 2. It starts, slides under a strand, and ends. The second one is of length 1; i.e., it starts and ends without passing over or under anything. Such bands are the shortest the environment allows. As in Example 2.5, attaching the first band to the trefoil yields the Hopf link, and attaching the second to the Hopf link yields the final unknot.

The framework is slice, and the environment is one of the slice models selected in Table 5 (the model called Slice\_Seif\_840k in the header of the transcript). The observation is the one described in Section 4.1, with the Seifert genus bound (Seif) appended, and the rewards are those of Section 3.5. Note that, in the following transcript, we omit the padding. A line such as [+97 pad] records the 97 suppressed rows of 1,000s, and a dash marks a tensor with no live entries. The value 500 inside gauss\_code closes the Gauss code of a link component, and the value 10 inside band\_matrix marks a crossing strand that exists but has not been visited by the band. The observation is printed as the environment builds it. Before it reaches the policy network, these sentinels are re-encoded and every entry is scaled, as described in Sections 3.3 and 4.1. The last seven entries of each block are the Seifert genus bound (seifert\_genus\_bound) and the entries that complete the description of the state: the twist of each band segment (band\_twists), the number of crossings of the current diagram after simplification and of the initial diagram, the CCL of each link component (component\_ccl), one row of genera\_info per CCL, recording the CCL, the number of bands attached to that surface component, the initial number of link components (including the unknots born), the current number of link components, and the number of original link components it contains, and the genus of the surface built so far (current\_genus). Finally, an end action is displayed as two blocks. The STEP block shows the diagram just after the band has been attached and the PD code rebuilt, and the SIMPLIFY block that follows shows the same diagram once spherogram-nim has simplified it; this is the diagram on which the next step acts. Since the rebuilt diagram before the simplification is never a state of the environment, its STEP block shows only the entries relative to the diagram and the band path. In Listing 2, we present the episode.

1   
2 framework = slice link = 3\_1 shape = (100 , 16)   
3 policy = Slice\_Seif\_840k   
4 max\_twists = 5 max\_actions = 80 observation = base + SeifertGenusBound   
5 ================================================================================   
6   
7 RESET link 0 = 3\_1: 3 crossings , 1 component genus so far = 0   
8 PD\_code : (6 ,3 ,1 ,4) (4 ,1 ,5 ,2) (2 ,5 ,3 ,6) [+97 pad ]   
9 crossing\_signs : -1 -1 -1   
10 writhe : -3   
11 gauss\_code : -1 2 -3 1 -2 3 -500   
12 num\_link\_components : 1   
13 unlinked\_unknot\_components : 0   
14 num\_connected\_components : 1   
15 band\_path : - (no band in progress )   
16 total\_twists : +0   
17 band\_matrix : ( -10 , -10 , -10 , -10) ( -10 , -10 , -10 , -10) ( -10 , -10 , -10 , -10)   
18 seifert\_genus\_bound : 1   
19 band\_twists -   
20 simplified\_num\_crossings : 3   
21 init\_num\_crossings : 3   
22 component\_ccl : 0   
23 genera\_info : (0 ,0 ,1 ,1 ,1)   
24 current\_genus : 0   
25   
26   
27 STEP 1 action = C2S3\_start (13 of 2403 legal ) reward = -0.0100   
28 -> 3 crossings , 1 component genus so far = 0   
29 PD\_code : (6 ,3 ,1 ,4) (4 ,1 ,5 ,2) (2 ,5 ,3 ,6) [+97 pad ]   
30 crossing\_signs : -1 -1 -1   
31 writhe : -3   
32 gauss\_code : -1 2 -3 1 -2 3 -500   
33 num\_link\_components : 1   
34 unlinked unknot components : 0   
35 num\_connected\_components : 1   
36 band\_path : (2 ,3 ,0)   
37 total\_twists : +0   
38 band\_matrix : ( -10 , -10 , -10 , -10) ( -10 , -10 , -10 ,0) ( -10 , -10 , -10 , -10)   
39 seifert\_genus\_bound : 1   
40 band\_twists : 0

<table><tr><td></td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=5>simplified_num_crossings</td></tr><tr><td></td><td rowspan=1 colspan=1>42</td><td rowspan=3 colspan=5>init_num_crossingscomponent_cclgenera_info</td></tr><tr><td></td><td rowspan=1 colspan=1>43</td></tr><tr><td></td><td rowspan=1 colspan=1>44</td></tr><tr><td></td><td rowspan=1 colspan=1>454647</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=5>STEP 2  action = C2S2_under</td></tr><tr><td></td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=5>-&gt; 3 crossings, 1 component</td></tr><tr><td></td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=5>PD_code</td></tr><tr><td></td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=5>crossing_signs</td></tr><tr><td></td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=5>writhe</td></tr><tr><td></td><td rowspan=1 colspan=1>53</td><td rowspan=3 colspan=5>gauss_codenum_link_componentsunlinked_unknot_components : 0</td></tr><tr><td></td><td rowspan=1 colspan=1>54</td></tr><tr><td></td><td rowspan=1 colspan=1>55</td></tr><tr><td></td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=5>num_connected_components</td></tr><tr><td></td><td rowspan=1 colspan=1>57</td><td rowspan=3 colspan=5>band_pathtotal_twistsband_matrix</td></tr><tr><td></td><td rowspan=1 colspan=1>58</td></tr><tr><td></td><td rowspan=1 colspan=1>59</td></tr><tr><td></td><td rowspan=1 colspan=1>60</td><td rowspan=2 colspan=5>seifert_genus_boundband_twists</td></tr><tr><td></td><td rowspan=1 colspan=1>61</td></tr><tr><td></td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=5>simplified_num_crossings</td></tr><tr><td></td><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=5>init_num_crossings</td></tr><tr><td></td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=5>component_ccl</td></tr><tr><td></td><td rowspan=1 colspan=1>65</td><td rowspan=2 colspan=5>genera_infocurrent_genus</td></tr><tr><td></td><td rowspan=1 colspan=1>66</td></tr><tr><td></td><td rowspan=1 colspan=1>6768</td><td rowspan=1 colspan=5></td></tr><tr><td></td><td rowspan=1 colspan=1>69</td><td rowspan=1 colspan=5>STEP 3  action = C2S1_endO</td></tr><tr><td></td><td rowspan=1 colspan=1>70</td><td rowspan=2 colspan=5>-&gt; band attached, diagram rebuiltPD_code</td></tr><tr><td></td><td rowspan=1 colspan=1>71</td></tr><tr><td></td><td rowspan=1 colspan=1>72</td><td rowspan=2 colspan=5>crossing_signswrithe</td></tr><tr><td></td><td rowspan=1 colspan=1>73</td></tr><tr><td></td><td rowspan=1 colspan=1>74</td><td rowspan=2 colspan=5>gauss_codenum_link_components</td></tr><tr><td></td><td rowspan=1 colspan=1>75</td></tr><tr><td></td><td rowspan=1 colspan=1>76</td><td rowspan=1 colspan=5>unlinked_unknot_components : 0</td></tr><tr><td></td><td rowspan=1 colspan=1>77</td><td rowspan=3 colspan=5>num_connected_componentsband_pathtotal_twists</td></tr><tr><td></td><td rowspan=1 colspan=1>78</td></tr><tr><td></td><td rowspan=1 colspan=1>79</td></tr><tr><td></td><td rowspan=1 colspan=1>80</td><td rowspan=2 colspan=5>band_matrix</td></tr><tr><td></td><td rowspan=1 colspan=1>8182</td></tr><tr><td></td><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=5>SIMPLIFY  the diagram rebuilt by step 3</td></tr><tr><td></td><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=5>-&gt; 2 crossings, 2 components</td></tr><tr><td></td><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=5>PD_code</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=1>86</td><td rowspan=2 colspan=5>writhe</td><td rowspan=1 colspan=1>crossing_signs</td></tr><tr><td></td><td rowspan=1 colspan=1>87</td></tr><tr><td></td><td rowspan=1 colspan=1>88</td><td rowspan=3 colspan=5>gauss_codenum_link_componentsunlinked_unknot_components : 0</td></tr><tr><td></td><td rowspan=1 colspan=1>89</td></tr><tr><td></td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=5>unlinked_unknot_com</td></tr><tr><td></td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=5>num_connected_com</td><td rowspan=1 colspan=1>ponents</td></tr><tr><td></td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=5>band_path</td></tr><tr><td></td><td rowspan=1 colspan=1>93</td><td rowspan=3 colspan=5>total_twistsband_matrixseifert_genus_bound</td></tr><tr><td></td><td rowspan=1 colspan=1>94</td></tr><tr><td></td><td rowspan=1 colspan=1>95</td></tr><tr><td></td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=5>band_twists</td></tr><tr><td></td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=5>simplified_num_crossings   :</td></tr><tr><td rowspan=1 colspan=2>98</td><td rowspan=2 colspan=5>init_num_crossingscomponent_ccl</td></tr><tr><td rowspan=1 colspan=2>99</td></tr><tr><td rowspan=1 colspan=2>100</td><td rowspan=3 colspan=5>genera_infocurrent_genus</td></tr><tr><td rowspan=1 colspan=2>101</td></tr><tr><td rowspan=1 colspan=2>102103</td></tr><tr><td rowspan=1 colspan=2>104</td><td rowspan=1 colspan=5>STEP 4 action = C2SO_start</td></tr><tr><td rowspan=1 colspan=2>105</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td rowspan=3 colspan=5></td></tr><tr><td></td><td></td><td rowspan=4 colspan=5>-&gt; 2 crossings, 2 componentswrithe</td></tr><tr><td rowspan=1 colspan=2>106</td></tr><tr><td rowspan=1 colspan=2>107</td><td rowspan=1 colspan=5>crossing_signs</td></tr><tr><td rowspan=1 colspan=2>108</td></tr><tr><td rowspan=1 colspan=2>109110</td><td rowspan=1 colspan=5>gauss_code</td></tr><tr><td rowspan=2 colspan=2>112113</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>uulin</td><td rowspan=1 colspan=1>_unknot_c</td></tr><tr><td rowspan=1 colspan=5>num_connected_componentsband_path</td><td rowspan=1 colspan=1>ted_</td><td rowspan=1 colspan=1>mponents</td></tr></table>

```csv
114 total_twists : +0
115 band_matrix ( -10 , -10 , -10 , -10) (0 , -10 , -10 , -10)
116 seifert_genus_bound 0
117 band_twists 0
118 simplified_num_crossings 2
119 init_num_crossings 3
120 component_ccl 0 0
121 genera_info (0 ,1 ,1 ,2 ,1)
122 current_genus 0
123
124
125 STEP 5 action = C2S3_end0 (5 of 2403 legal ) reward = +32.4758
126 -> band attached , diagram rebuilt ( not yet simplified )
127 PD_code : (1 ,4 ,2 ,1) (3 ,2 ,4 ,3) [+98 pad ]
128 crossing_signs : -1 -1
129 writhe : -2
130 gauss_code : 1 -1 2 -2 -500
131 num_link_components : 1
132 unlinked_unknot_components : 0
133 num_connected_components : 1
134 band_path : (2 ,0 ,0) (2 ,3 ,0)
135 total_twists : +0
136 band_matrix : ( -10 , -10 , -10 , -10) (0 , -10 , -10 ,0)
137
138
139 SIMPLIFY the diagram rebuilt by step 5 ( spherogram -nim , global mode )
140 -> 0 crossings , 1 unlinked unknot genus so far = 1
141 PD_code : ( no crossings ) [+100 pad ]
142 crossing_signs
143 writhe +0
144 gauss_code
145 num_link_components : 0
146 unlinked_unknot_components : 1
147 num_connected_components : 0
148 band_path ( no band in progress )
149 total_twists +0
150 band_matrix
151 seifert_genus_bound 0
152 band_twists
153 simplified_num_crossings : 0
154 init_num_crossings 3
155 component_ccl
156 genera_info (0 ,2 ,1 ,1 ,1)
157 current_genus : 1
158
159 ================================================================================
160 EPISODE WON reached the unlink with 2 band moves ,
161 resulting in a slice surface of genus 1 with 1 unlinked unknot component
162 return = sum of the rewards = +32.6337 ( gamma = 1)
163
```  
Listing 2. A complete episode on the trefoil, in the slice framework.

Reading the transcript. The first band starts at the crossing strand C2S3, slides under the strand C2S2 (a Reidemeister II move, recorded as the entry (2, 2, 1) of band\_path and as the 1 in band\_matrix), and is attached at C2S1 with no final twist. Attaching the band rebuilds the diagram with 5 crossings (step 3). Simplification then cuts this down to a 2-crossing, 2- component diagram of the Hopf link $L _ { t _ { g } }$ of Figure 5g. The second band is the shortest one possible. It begins at C2S0 and ends at C2S3, two strands of the same crossing that share a face. The band rebuilds the 2-crossing diagram (1,4,2,1) (3,2,4,3) of step 5, in which each crossing repeats one of its labels and is therefore a Reidemeister I loop. Undoing both loops is precisely what the last simplification does, and what is left is the unknot, which the environment reports as a diagram with 0 crossings and unlinked\_unknot\_components equal to 1.

The line genus so far shows the genus of the surface built up to that point. It is still 0 after the first band (the surface is a pair of pants, with χ = 1 and 3 boundary components)

and becomes 1 after the second one since $2 g = 2 + \# \{ { \mathrm { b a n d s } } \} - \#$ boundary components = $2 + 2 - ( 1 + 1 )$ . The genera\_info row records the same computation for the single surface component: $( 0 , 1 , 1 , 2 , 1 )$ after the first band, which is attached to it and raises its boundary components from 1 to 2, and (0, 2, 1, 1, 1) at the end. The episode therefore returns the certificate $g _ { 4 } ( 3 _ { 1 } ) \leq 1$ , which we know to be sharp.

Note that no unknot\_birth was played, even though the slice framework ofers it (which is why the action space has size $2 4 0 3 = 2 4 \cdot 1 0 0 + 2 + 1$ rather than 2402). This is expected since, for the trefoil, the ribbon genus already equals the slice genus, and adding a maximum to the surface would only make the certificate longer.

In Step 5, the reward is computed as

$$
\begin{array} { r l r } & { } & { R ( s , a , s ^ { \prime } ) = \sigma \big ( R _ { \mathrm { w i n } } - W ( i ( s ^ { \prime } ) - i ( s ) ) + \Phi ( s ^ { \prime } ) - \Phi ( s ) \big ) } \\ & { } & { = 0 . 0 1 \big ( 3 6 0 0 - 4 0 0 \cdot ( 1 - 0 ) + 2 \kappa \big ) = 3 2 . 4 7 5 8 , } \end{array}
$$

using $\begin{array} { r } { \kappa = \frac { 6 5 } { 1 + \sqrt { 3 } } : } \end{array}$ the reward for winning, minus the charge for the unit of genus that the band adds, plus the shaping term for the two crossings that it removes, all scaled by 0.01; see Section 3.5. Since $\gamma = 1$ , the return of the episode is the plain sum of its rewards, 32.6337.

A further feature of the environment is visible in the transcript: the efect of the action masking of Section 3.2. Given the small size of the trefoil, only 13, 6, 8, 9, and 5 of the 2403 actions are legal at the five steps, respectively, so the agent chooses from a set smaller by more than two orders of magnitude.

## Appendix B Bounds

The environment we developed, presented in Section 3, produces upper bounds on the invariants we are interested in, in the form of an explicit sequence of moves. What turns an upper bound into an exact value is a matching lower bound, computed independently. This appendix collects the bounds we evaluate before computing explicit band paths using some agent. We assume the invariants themselves $( \sigma , \tau , s , g _ { 3 } )$ are known to the reader, and we provide references for each bound rather than restating the definitions.

## B.1 Relations between the three invariants

The following proposition is well known and connects the unknotting number and the slice genus;   
see e.g. [App+25, Section 1.1].

Proposition B.1. For any link $L ,$ we have $g _ { 4 } ( L ) \leq u ( L )$

Therefore, every lower bound on the smooth slice genus automatically gives a lower bound on the unknotting number, and we will not recompute those bounds for both invariants. Note that, on the other hand, the strong slice genus does not satisfy $g _ { 4 } ^ { * } ( L ) \leq u ( L )$

Remark B.2 (Orientation dependence). The weak slice genus depends on the choice of orientation of $L ,$ since the bounding surface F must be oriented compatibly with $L ,$ while the strong slice genus is independent of the orientation. Since $u ( L )$ is also orientation-independent (Remark 2.3), the strongest lower bounds from the slice genus are

$$
g _ { 4 } ^ { * } ( L ) \geq \operatorname* { m a x } _ { \vec { o } } g _ { 4 } ( L _ { \vec { o } } ) , \qquad u ( L ) \geq \operatorname* { m a x } _ { \vec { o } } g _ { 4 } ( L _ { \vec { o } } ) ,
$$

where $\vec { o } \in \{ + 1 , - 1 \} ^ { \ell }$ is a choice of orientation.

Let us present a lower bound for the unknotting number of links that does not factor through the slice genus. The following inequality is not stated explicitly by Collari [Col21], but it appears

in the proof of [Col21, Lemma 2.8]. The statement of [Col21, Lemma 2.8] comes with the additional hypothesis that the strong splitting number equals the absolute linking number; however, that is needed to produce an equality.

Proposition B.3 (From the proof of [Col21, Lemma 2.8]). Let $L = K _ { 1 } \cup \cdot \cdot \cdot \cup K _ { \ell }$ be a link. Then, we have

$$
| \mathrm { I k } | ( L ) + \sum _ { i = 1 } ^ { \ell } u ( K _ { i } ) \leq u ( L ) ,
$$

where $| \mathrm { l k } | ( L )$ denotes the absolute linking number of $L ,$ which is the sum of the absolute values of the linking numbers between the components of L.

Remark B.4. The inequality of Proposition B.3 is an equality whenever L is simply-linked. See Theorem B.30 for details.

Lastly, let us state the following trivial lower bound on the strong slice genus.

Lemma B.5 (Lower bound on the strong slice genus). Given a link $L = K _ { 1 } \cup \cdot \cdot \cdot \cup K _ { \ell }$ , we have

$$
\sum _ { i = 1 } ^ { \ell } g _ { 4 } ( K _ { i } ) \leq g _ { 4 } ^ { * } ( L ) .
$$

Indeed, given a strong slice surface $F$ of minimal genus for $L ,$ we get that $\textstyle F = \sqcup _ { i = 1 } ^ { \ell } F _ { i }$ with $\partial F _ { i } = K _ { i }$ and $g _ { 4 } ( K _ { i } ) \leq g ( F _ { i } )$ . Using existing databases and possibly new computations of the slice genera of $K _ { 1 } , \ldots , K _ { \ell }$ , we might derive a non-trivial lower bound from this inequality.

## B.2 Upper bounds

We have the following elementary bounds on the smooth slice genus.

Proposition B.6. For any link $L ,$

$$
g _ { 4 } ^ { t o p } ( L ) \leq g _ { 4 } ( L ) \leq g _ { 3 } ( L ) .
$$

Indeed, any smoothly embedded surface is also locally flat; hence, any smooth slice surface for $L$ is also a topological slice surface for $L .$ . Furthermore, we can push the interior of any Seifert surface for L in $S ^ { \bar { 3 } }$ into $B ^ { 4 }$ , resulting in a properly embedded surface bounding L in the 4-ball.

We can compute $g _ { 3 } ( K )$ via knot Floer homology, which, for knots, was implemented by Ozsváth and Szabó [SnapPy]. This gives us a practical slice genus upper bound for knots. While we do not have an implementation of knot Floer homology for links, we can compute an upper bound on the Seifert genus of links, which results in an upper bound on the weak slice genus.

Theorem B.7 (Upper bound from Seifert surfaces, [Cav15, Proposition 4.1]). Given a link L with ℓ components and a diagram D, Seifert’s algorithm produces a surface Σ with $\chi ( \Sigma ) =$ $O ( D ) - c ( D )$ , where $O ( D )$ is the number of Seifert circles and $c ( D )$ is the number of crossings. If the Seifert surface has $r ( D )$ components, then

$$
g _ { 4 } ( L ) \leq g _ { 3 } ( L ) \leq g ( \Sigma ) = \frac { 2 r ( D ) - \ell - O ( D ) + c ( D ) } { 2 } .\tag{B.1}
$$

We have the following simple upper bound on the unknotting number given by the crossing number.

Proposition B.8 (Trivial upper bound). For any link $L ,$

$$
u ( L ) \leq \left\lfloor { \frac { \operatorname { c } ( L ) } { 2 } } \right\rfloor .
$$

Proof from [AT20, Discussion after Theorem 1.1]. It is a known fact that any diagram D of L with c crossings can be turned into a diagram of the unlink by changing some of its crossings. Let $\mathcal { C } ( D )$ be the set of crossings of D and $A \subset { \mathcal { C } } ( D )$ be the subset of crossings that yields a diagram U of the unlink if changed. Let now S be the diagram we obtain from D if we change all the crossings in $B = \mathcal { C } ( D ) \setminus A$ . This is the mirror of U, which is again the unlink. Therefore, we get the unlink with at most min $\begin{array} { r } { \left\{ | A | , | B | \right\} \le \frac { c ( D ) } { 2 } } \end{array}$ crossing changes. □

No upper bound on the strong slice genus is available to us. A Seifert surface is not a strong slice surface, so every finite upper bound on $g _ { 4 } ^ { * } ( L )$ comes from an explicitly exhibited surface, which is precisely what the agents of Section 4 produce.

## B.3 Lower bounds, by technique

## B.3.1 Signature-based

The signature is a classical and computationally cheap link invariant. Given any Seifert surface S for a link $L ,$ obtained for instance by Seifert’s algorithm applied to a diagram of $L ,$ it is the signature of $A + A ^ { T }$ , where $A$ is a Seifert matrix associated with S. It was introduced by Trotter [Tro62] for knots and extended to links by Murasugi [Mur65, Theorem 3.1], who also showed that it bounds the (weak) slice genus of links.

Theorem B.9 (Signature bound [Mur65, Theorem 9.1]). For any link L with ℓ components,

$$
{ \frac { | \sigma ( L ) | - \ell + 1 } { 2 } } \leq g _ { 4 } ( L ) ,
$$

where $\sigma ( L )$ is the signature of L. For a knot K, this reads ${ \frac { 1 } { 2 } } \left| \sigma ( K ) \right| \leq g _ { 4 } ( K )$

Sharper bounds come from “signature-like” quantities, the main one being the Levine–Tristram signature $\sigma _ { L } ( \omega )$ and nullity $\eta _ { L } ( \omega )$ at $\omega \in S ^ { 1 }$ , introduced independently by Levine [Lev69] and Tristram [Tri69]. We refer to [Con21, Definition 1] for the definition and for a complete survey. That they give slice genus bounds was originally proved by Tristram for $\omega ~ = ~ - 1$ or $\omega = \exp \bigl ( 2 q \pi i \bigl / 2 q + 1 \bigr )$ with $2 q + 1$ an odd prime, and extended by Kaufman [Kau78] to prime power order roots of unity at the cost of requiring the surface to be connected. Note that Tristram and Kaufman’s nullity $\eta _ { \omega } ( L )$ equals our $\eta _ { L } ( \omega ) + 1$ , as defined in [Tri69, Definition 2 14] and [Kau78, above Theorem 4.1]. The case $\omega = - 1$ recovers the signature.

Theorem B.10 (Murasugi–Tristram inequality [Tri69, Theorem 2 27][Kau78, Theorem 4.1]). Let L be an ℓ-component link and $\omega \in S ^ { 1 } \subset \mathbb { C }$ be a primitive a-th root of unity, where $a = p ^ { s }$ with p prime. Then, we have

$$
\frac { | \sigma _ { L } ( \omega ) | + \eta _ { L } ( \omega ) - ( \ell - 1 ) } { 2 } \leq g _ { 4 } ( L ) . ^ { 3 }
$$

Remark B.11 (Disconnected surfaces and the strong slice genus). Tristram’s Theorem 2 27 holds for $\omega = - 1$ and for $\omega = \exp ( 2 q \pi i / 2 q + 1 )$ with $2 q + 1$ an odd prime [Tri69, Definitions 2 3, 2 14] and is stated for an arbitrary, possibly disconnected, spanning surface $F \subset B ^ { 4 }$ . If F has m connected components, it reads

$$
| \sigma _ { L } ( \omega ) | + | ( \eta _ { L } ( \omega ) + 1 ) - m | \le 2 g ( F ) + \ell - m .
$$

Theorem B.10 is the case $m = 1$ , which Kaufman extended to all roots of unity of prime power order. In the strong slice case, the surface $F = F _ { 1 } \sqcup \cdots \sqcup F _ { \ell }$ is a disjoint union of ℓ surfaces, one per component of $L ,$ so $m = \ell$ and the inequality becomes a bound on the strong slice genus:

$$
\frac { | \sigma _ { L } ( \omega ) | + | ( \eta _ { L } ( \omega ) + 1 ) - \ell | } { 2 } \leq g _ { 4 } ^ { * } ( L ) .
$$

In particular, if L is strongly slice, then $\sigma _ { L } ( \omega ) = 0$ and $\eta _ { L } ( \omega ) = \ell - 1$ . See [Con21, Theorem 3] for a version valid for locally flat surfaces and for all ω outside the set of Knotennullstellen, and [FG03, Theorem 2.1] for the connected case.

Lower bound specific to $u ( L )$ . The Levine–Tristram signature for links provides a lower bound directly on $u ( L )$ without factoring through $g _ { 4 } ( L )$ . It rests on a lemma of Nagel and Owens [NO15, Lemma 2.1], which describes how the signature and nullity behave when a single crossing is changed, extending the results of Cochran and Lickorish [CL86, Proposition 2.1] and Stoimenow [Sto04, Theorem 5.1] for knots to the multi-component case. Conway [Con21, Proof of Proposition 3] noted that this lemma and its proof hold not only for $\omega = - 1$ , as Nagel and Owens stated, but for any $\omega \in S _ { * } ^ { 1 } = S ^ { 1 } \setminus \{ 1 \}$ , and deduced the following bound.

Theorem B.12 ([Con21, Theorem 1]). Let L be an oriented ℓ-component link and let $\omega \in S _ { * } ^ { 1 }$ Then

$$
\frac { | \sigma _ { L } ( \omega ) | + | \eta _ { L } ( \omega ) - \ell + 1 | } { 2 } \leq u ( L ) . ^ { 4 }
$$

## B.3.2 Knot Floer homology

Knot Floer homology was first introduced independently by Ozsváth and Szabó [OS04] and Rasmussen [Ras03]. They defined a concordance invariant $\tau ;$ see [OS03b].

Theorem B.13 ([OS03b, Corollary 1.3]). The τ-invariant satisfies

$$
| \tau ( K ) | \leq g _ { 4 } ( K ) .
$$

Ozsváth and Szabó [OS11, Definition 9.1] introduced a refinement ν of $\tau _ { : }$ which satisfies $\nu ( K ) \in \{ \tau ( K ) , \tau ( K ) + 1 \}$ and also bounds the slice genus from below.

Proposition B.14 ([HW16, Proposition 2.3 and Proposition 2.4]). For a knot K,

$$
\nu ( K ) \leq g _ { 4 } ( K ) .
$$

Both τ and ν can be computed for knots via the knot\_floer\_homology module of [SnapPy].

An extension of τ to links. Cavallo [Cav18] introduced a version of τ for links via link grid homology, and is a strong concordance invariant that yields a lower bound on the weak slice genus of a link. However, as of now, there is no publicly available implementation that computes $\tau$ for links.

Theorem B.15 ([Cav18, Proposition 1.4]). For an ℓ-component link $L _ { ☉ }$

$$
| \tau ( L ) | + 1 - \ell \leq g _ { 4 } ( L ) .
$$

## B.3.3 Khovanov homology

Khovanov [Kho00] categorified the Jones polynomial and introduced Khovanov homology. Building on the work of Lee [Lee05], Rasmussen [Ras10, Definition 3.4] introduced a new knot invariant $s ,$ later called the Rasmussen s-invariant, that provides a lower bound on the slice genus. This invariant was subsequently extended to multi-component links by Beliakova and Wehrli [BW08, Section 7.1].

Theorem B.16 (Rasmussen’s s-invariant lower bound [Ras10, Theorem 1][Cav15, Equation (4.1)]). For an ℓ-component link $L ,$ we have

$$
\frac { | s ( L ) | + 1 - \ell } { 2 } \leq g _ { 4 } ( L ) .
$$

One of the properties proved for the link version is of particular interest to us.<sup>5</sup>

Lemma B.17 ([BW08, equation (7.1)]). Let F be a smooth oriented cobordism from $L _ { 1 }$ to $L _ { 2 }$ such that every connected component of F intersects both $L _ { 1 }$ and $L _ { 2 }$ . Then

$$
| s ( L _ { 1 } ) - s ( L _ { 2 } ) | \leq - \chi ( F ) .
$$

Applying this cobordism estimate to the case of a strong cobordism, we can produce a lower bound for the strong slice genus of a link.

Corollary B.18 (Strong slice genus bound). Let L be an ℓ-component, algebraically split link. Then

$$
\frac { | s ( L ) + \ell - 1 | } { 2 } \leq g _ { 4 } ^ { * } ( L ) .
$$

Proof. Let $\textstyle F = \bigcup _ { i = 1 } ^ { \ell } F _ { i }$ be a minimal genus strong cobordism between L and $U ,$ , the ℓ-component unlink. Then $\begin{array} { r } { \sum _ { i } \dot { g } ( \bar { F } _ { i } ) = g _ { 4 } ^ { * } ( L ) } \end{array}$ . Assume that $\partial F _ { i } = - K _ { i } \sqcup U _ { i }$ , where $U _ { i }$ is the i-th component of U. As F has ℓ connected components and 2ℓ boundary circles, by equation (2.1), we get that its Euler characteristic is

$$
\chi ( F ) = \sum _ { i = 1 } ^ { \ell } \chi ( F _ { i } ) = \sum _ { i = 1 } ^ { \ell } ( 2 - 2 g ( F _ { i } ) - | \partial F _ { i } | ) = 2 \ell - 2 g _ { 4 } ^ { * } ( L ) - 2 \ell = - 2 g _ { 4 } ^ { * } ( L ) .
$$

Since every component of $F$ intersects $L$ and $U ,$ Lemma B.17 applies. As $s ( U ) = 1 - \ell ,$ we get

$$
| s ( L ) + \ell - 1 | \leq - \chi ( F ) = 2 g _ { 4 } ^ { \ast } ( L ) ,
$$

which concludes the proof.

On the computability of the Rasmussen s-invariant. Several pieces of software were created to compute the s-invariant for knots. In our codebase, we integrated the implementation by Schütz [Sch25b], which, to our knowledge, is the only one able to compute the s-invariant for links. Details on the algorithm used for knots can be found in Schütz [Sch21; Sch22], and for links in [Sch25a].

## B.3.4 Slice-torus link invariants

Livingston [Liv04] defined a class of knot invariants that Lewark [Lew14] later called slice-torus invariants, and he observed that they could be generalised from taking values in even integers to reals. They were extended to links by Cavallo and Collari [CC20], and they proved several results concerning weak and strong sliceness. What follows is a short survey of these results and the bounds they yield on the weak and strong slice genera of a link.

Example B.19 (Known slice-torus link invariants [CC20, Examples 2.3–2.4]). The following are slice-torus link invariants for an ℓ-component link L:

(1) the Ozsváth–Szabó τ -invariant for links (Section B.3.2): $\nu _ { \tau } ( L ) : = \tau ( L )$ ;

(2) the Rasmussen s-invariant for links (Section B.3.3): $\begin{array} { r } { \nu _ { s } ( L ) : = \frac { s ( L ) + \ell - 1 } { 2 } } \end{array}$

The following result generalises what we already stated for s and τ in this framework.

Theorem B.20 (Slice-torus bound on the weak slice genus [CC20, Proposition 1.2]). $I f \nu$ is a slice-torus link invariant and L is an ℓ-component link, then:

$$
\begin{array} { r } { - g _ { 4 } ( L ) \le \nu ( L ) \le g _ { 4 } ( L ) + \ell - 1 . } \end{array}
$$

Furthermore, $i f L$ is strongly slice, then $\nu ( L ) = 0$

We state the bound on the slice genus that we obtain from the above theorem. Since the slice genus is invariant under mirroring and orientation reversal, we can also consider the values we obtain from the slice-torus invariant applied to $- \mathbf { m } ( L )$

Corollary B.21 (Lower bound on the weak slice genus). For any slice-torus link invariant $\nu$ and ℓ-component link $L ,$

$$
g _ { 4 } ( L ) \geq \operatorname* { m a x } \left\{ - \nu ( L ) , \nu ( L ) - \ell + 1 , \ - \nu ( - \mathrm { m } ( L ) ) , \nu ( - \mathrm { m } ( L ) ) - \ell + 1 \right\} .
$$

Theorem B.20 follows from the following cobordism bound.

Theorem B.22 (Cobordism bound [CC20, Proposition 2.11]). Let ν be a slice-torus link invariant. $I f \Sigma \subset S ^ { 3 } \times [ 0 , 1 ]$ is a cobordism from $L _ { 0 }$ to $L _ { 1 }$ with n connected components, each intersecting both $L _ { 0 }$ and $L _ { 1 }$ , then

$$
\nu ( L _ { 1 } ) - g ( \Sigma ) - \ell _ { 1 } + n \leq \nu ( L _ { 0 } ) \leq \nu ( L _ { 1 } ) + g ( \Sigma ) + \ell _ { 0 } - n ,
$$

where $\ell _ { i }$ denotes the number of components of $L _ { i }$

Indeed, by taking a slice surface for a link L (i.e., a cobordism between L and the unknot), one obtains the inequalities of Theorem B.20. Moreover, we can get a better bound on the strong slice genus by taking a minimal genus strong slice surface for L; i.e., a cobordism between $L$ and the ℓ-component unlink $U _ { \ell }$ , and recalling that $g ( F ) = g _ { 4 } ^ { * } ( L )$ . Various terms cancel, and we obtain the following corollary.

Corollary B.23. Let L be an ℓ-component algebraically split link. Then, for any slice-torus link invariant $\nu ,$

$$
| \nu ( L ) | \leq g _ { 4 } ^ { * } ( L ) .
$$

Since the strong slice genus is invariant under mirroring and orientation reversal, we obtain

$$
\operatorname* { m a x } \{ | \nu ( L ) | , | \nu ( - \operatorname { m } ( L ) ) | \} \leq g _ { 4 } ^ { * } ( L ) .
$$

By Lemma B.5, if we denote by $K _ { 1 } , \ldots , K _ { \ell }$ the components of the link $L ,$ , then

$$
\operatorname* { m a x } \left\{ \left| \nu ( L ) \right| , \left| \nu ( - \operatorname { m } ( L ) ) \right| , \sum _ { i = 1 } ^ { \ell } \left| \nu ( K _ { i } ) \right| \right\} \leq g _ { 4 } ^ { * } ( L ) .\tag{B.2}
$$

Indeed, $g _ { 4 } ( K ) = g _ { 4 } ^ { * } ( K )$ for a knot K. By Corollary B.23, we have $| \nu ( K ) | \le g _ { 4 } ( K )$ . Together with $\begin{array} { r } { \sum _ { i = 1 } ^ { \ell } g _ { 4 } ( K _ { i } ) \le g _ { 4 } ^ { * } ( L ) } \end{array}$ , we obtain inequality (B.2).

Bounds on the unknotting number. By studying the behaviour of slice-torus invariants under crossing changes, we can also obtain bounds on the unknotting number that do not factor through the slice genus.

The first result gives a lower bound on the unknotting number of a link by studying a splitting sequence. Recall that the splitting number of a link, denoted spe by Cavallo and $\mathrm { C o l l a r i ^ { 6 } }$ , is the minimum number of crossing changes needed to change an ℓ-component link into a disjoint union of ℓ knots. Clearly, $\tilde { s p } ( L ) \leq u ( L )$ since an unknotting sequence is also a splitting sequence.

Theorem B.24 ([CC20, Theorem 1.5]). Suppose that ν is a slice-torus link invariant and L is a link with components $K _ { 1 } , \ldots , K _ { \ell }$ . Then

$$
\left| \nu ( L ) - \sum _ { i = 1 } ^ { \ell } \nu ( K _ { i } ) \right| \leq \widetilde s p ( L ) \leq u ( L ) .
$$

The above theorem is based on the following result, which describes the efect of a crossing change on a slice-torus link invariant.

Proposition B.25 (Crossing changes and slice-torus link invariants [CC20, Proposition 2.10]). Let $D _ { - }$ be obtained from $D _ { + }$ by replacing a positive crossing with a negative one, and let $L _ { \pm }$ be the corresponding links. Then, for every slice-torus link invariant $\nu ,$

$$
0 \leq \nu \left( L _ { + } \right) - \nu \left( L _ { - } \right) \leq 1 .
$$

Using Proposition B.25, we can obtain another bound on the unknotting number of a link.

Corollary B.26 (Bound on the unknotting number). Let ν be a slice-torus link invariant, and let L be a link. $I f L$ admits an unknotting sequence consisting of $u _ { + }$ changes $o f$ a positive crossing into a negative one and $u _ { - }$ changes of a negative crossing into a positive one, then

$$
\nu ( L ) \leq u _ { + } \qquad a n d \qquad - \nu ( L ) \leq u _ { - } .
$$

Combining the two inequalities, we get $| \nu ( L ) | \leq u ( L )$

Proof. Consider an unknotting sequence from L with $u _ { + }$ positive crossing changes, $u _ { - }$ negative ones, and $u _ { + } + u _ { - } = N = u ( L )$ . We can describe the sequence from $L$ to the unlink $U _ { \ell }$ as

$$
L = L ^ { ( 0 ) } \xrightarrow { u _ { 1 } } L ^ { ( 1 ) } \xrightarrow { u _ { 2 } } \cdots \xrightarrow { u _ { N } } L ^ { ( N ) } = U _ { \ell }
$$

Let $\delta _ { j } : = \nu ( L ^ { ( j ) } ) - \nu ( L ^ { ( j - 1 ) } )$

If $u _ { j }$ turns a positive crossing into a negative one $( { \mathrm { i . e . } }$ , is one of the $u _ { + } )$ , then in the notation of Proposition B.25, we have $L ^ { ( j - 1 ) } = L _ { + }$ and $L ^ { ( j ) } = L _ { - } , \mathrm { s o ~ } 0 \le \nu ( L ^ { ( j - 1 ) } ) -$ $\nu ( L ^ { ( j ) } ) \leq 1$ , which gives $- 1 \leq \delta _ { j } \leq 0$

$$
u _ { j }
$$

$$
( \mathrm { i . e . }
$$

$$
u _ { - } )
$$

If turns a negative crossing into a positive one , is one of the , then with $L ^ { ( j - 1 ) } = L _ { - }$ and $L ^ { ( j ) } = L _ { + }$ , we obtain $0 \leq \nu ( L ^ { ( j ) } ) - \nu ( L ^ { ( j - 1 ) } ) \leq 1 ;$ that is, $0 \leq \delta _ { j } \leq 1$ In particular, by summing all N inequalities,

$$
- u _ { + } \leq \sum _ { j = 1 } ^ { N } \delta _ { j } \leq u _ { - } .
$$

Recall that $\nu ( U _ { \ell } ) = 0 ,$ , so

$$
\begin{array} { l } { - \nu ( L ) = \nu ( U _ { \ell } ) - \nu ( L ) = \nu ( L ^ { ( N ) } ) - \nu ( L ^ { ( 0 ) } ) } \\ { \displaystyle \qquad = \nu ( L ^ { ( N ) } ) - \nu ( L ^ { ( N - 1 ) } ) + \nu ( L ^ { ( N - 1 ) } ) - \dots - \nu ( L ^ { ( 1 ) } ) + \nu ( L ^ { ( 1 ) } ) - \nu ( L ^ { ( 0 ) } ) } \\ { \displaystyle \qquad = \sum _ { j = 1 } ^ { N } \delta _ { j } . } \end{array}
$$

Hence,

$$
- u _ { + } \leq - \nu ( L ) \leq u _ { - } ,
$$

which yields the two stated inequalities. Finally, since $u _ { + } \geq$ max $\{ \nu ( L ) , 0 \}$ and $u _ { - } \geq \operatorname* { m a x } \{ - \nu ( L ) , 0 \}$ , we get $u ( L ) = u _ { + } + u _ { - } \geq | \nu ( L ) |$ . □

## B.3.5 The Fox–Milnor condition

The Alexander polynomial $\Delta _ { K } ( t )$ of a knot K is a classical invariant, introduced by Alexander [Ale28]. It is a Laurent polynomial that is well-defined up to a factor $\pm t ^ { m }$ for some integer m. Fox and Milnor proved [FM66] that this polynomial yields an obstruction to sliceness.

Theorem B.27 ([FM66, Theorem 2]). If K is a slice knot, then its Alexander polynomial $\Delta _ { K } ( t )$ is of the form

$$
\Delta _ { K } ( t ) = t ^ { m } \cdot p \left( t \right) p \left( 1 / t \right) ,
$$

where $p ( t )$ is a polynomial with integral coeficients and m is an integer.

The determinant of a knot agrees with $| \Delta _ { K } ( - 1 ) |$ . Hence, we obtain the following corollary, which provides a simpler obstruction to sliceness.

Corollary B.28. If a knot K is slice, then its determinant is a perfect square.

## B.4 Special families

For certain special families of links, we can give stronger bounds. What follows is a short survey on some of these, namely: positive and negative links, torus links, alternating links, and pseudothin links.

## B.4.1 Positive and negative links

In this section, we study positive and negative links, whose four-genera can be computed exactly and directly from a diagram. Recall that an oriented link is positive if it admits a diagram where all crossings are positive, and negative if it admits a diagram where all crossings are negative.

The positive case was settled in stages. Nakamura [Nak00, Theorem 1.1] proved that for a non-split positive link, the slice genus, the Seifert genus, and the genus of the surface produced by Seifert’s algorithm all agree, so $g _ { 4 } ( L )$ can be read of from any positive diagram. For knots, the same result was later reproved by Rasmussen [Ras10, Theorem 4], using a purely combinatorial argument in Khovanov homology, which ties both genera to the s-invariant. The resulting diagrammatic formula for s was then extended to positive links by Cavallo [Cav15, Proposition 3.3]. All of these are instances of a single statement about slice-torus link invariants, due to Cavallo and Collari, which is the form we use.

Theorem B.29 (Positive links and slice-torus invariants [CC20, Theorem 1.3]). Let L be an ℓ- component positive link with $\ell _ { s }$ split components and positive diagram D that has $c ( D )$ crossings and $O ( D )$ Seifert circles. Then, for any slice-torus link invariant $\nu ,$

$$
\nu ( L ) = g _ { 3 } ( L ) + \ell - \ell _ { s } = \frac { c ( D ) - O ( D ) + \ell } { 2 } .
$$

If L is also non-split, then

$$
\nu ( L ) = g _ { 4 } ( L ) + \ell - 1 ,\tag{B.3}
$$

and consequently $g _ { 4 } ( L ) = g _ { 3 } ( L )$

The negative case follows by mirroring. If D is a negative diagram of $L ,$ then the mirror image $\mathbf { m } ( D )$ is a positive diagram of $\mathrm { m } ( L )$ with the same numbers of crossings and Seifert circles. Since the four-genus is invariant under mirroring, one obtains for a non-split negative diagram D of an ℓ-component link L that

$$
g _ { 4 } ( L ) = g _ { 3 } ( L ) = \frac { c ( D ) - O ( D ) + 2 - \ell } { 2 } .
$$

Unknotting number of positive links. While no formula is known for arbitrary positive links, the following results help us to compute some unknotting numbers. We say that a link is simply-linked if it admits a diagram where the crossings between any two fixed components have the same sign. Clearly, positive links are simply-linked.

Theorem B.30 (Unknotting number of simply-linked links [Col21, Proposition 4.1 and Theorem 1.9]). $I f L = K _ { 1 } \cup \cdots \cup K _ { \ell }$ is a simply-linked link, then

$$
| \mathrm { I k } | ( L ) = u ( L ) - \sum _ { i = 1 } ^ { \ell } u ( K _ { i } ) ,
$$

where $| \mathrm { l k } | ( L )$ is the absolute linking number of $\therefore L ; i . e .$ , the sum of the absolute values of $\operatorname { T k } ( K _ { i } , K _ { j } )$ for all $i < j$ . For positive links, we have $| \mathrm { l k } | ( L ) = \mathrm { l k } ( L )$

Therefore, we can quickly compute the unknotting numbers of several positive links with up to 11 crossings by knowing the unknotting numbers of positive knots with up to 9 crossings and identifying the link components using topological invariants.

## B.4.2 Torus links

Recall that the torus link $\mathrm { T } _ { a , b }$ with $\operatorname* { g c d } ( a , b ) = \ell$ is an ℓ-component link lying on a standard torus in $S ^ { 3 }$ , wrapping a times around the meridian and b times around the longitude. When $\operatorname* { g c d } ( a , b ) = 1 , \operatorname { T } _ { a , b }$ is a knot.

Theorem B.31 (Milnor conjecture). For $a , b \in \mathbb { Z } _ { + }$ and $\operatorname* { g c d } ( a , b ) = 1$

$$
g _ { 4 } ( \mathrm { T } _ { a , b } ) = g _ { 3 } ( \mathrm { T } _ { a , b } ) = u ( \mathrm { T } _ { a , b } ) = \frac { ( a - 1 ) ( b - 1 ) } { 2 } .
$$

This was originally proved using gauge theory by Kronheimer and Mrowka [KM93; KM95]. Subsequently, Ozsváth and Szabó [OS03b, Corollary 1.7] reproved it using Heegaard Floer homology by showing that $\tau ( \mathrm { T } _ { a , b } ) = ( a - 1 ) ( b - 1 ) / 2$ . Rasmussen [Ras10, Corollary 1] gave a purely combinatorial proof using the s-invariant in Khovanov homology. The unknotting number, the four-ball genus, and the Seifert genus of torus links are also known.

Theorem B.32 (Unknotting number of positive torus links [Kaw02, Example 4.1]). Let $\mathrm { T } _ { a , b }$ be a torus link with $a , b \in \mathbb { Z } _ { + }$ and all components oriented coherently. Let $\ell = \operatorname* { g c d } ( a , b )$ . Then

$$
u ( \mathrm { T } _ { a , b } ) = { \frac { ( a - 1 ) ( b - 1 ) + \ell - 1 } { 2 } } .
$$

## B.4.3 Alternating links

For alternating links, the Rasmussen s-invariant agrees with the signature up to an overall sign (depending on the sign convention for the signature), which is much faster to compute. On an alternating diagram, it can be read of from the chequerboard surface; see Ozsváth and Szabó [OS03a, Theorem 3.1].

Proposition B.33 ([Ras10, Theorem 3] for knots, [Cav15, Corollary 3.2] for links). If L is a non-split alternating link, then $| s ( L ) | = { \dot { | \sigma } } ( L ) |$

Since $\tau ( K ) = - \sigma ( K ) / 2$ for alternating knots [OS03b, Theorem 1.4], the signature, s, and τ all give the same lower bound. Combining Proposition B.33 with Theorem B.16, we recover the classical lower bound

$$
g _ { 4 } ( L ) \geq { \frac { | \sigma ( L ) | + 1 - \ell } { 2 } }\tag{B.4}
$$

for an alternating link L. This also holds in the broader class of quasi-alternating links [OS05, Definition 3.1], by using

$$
\tau ( L ) = \frac { \ell - 1 - \sigma ( L ) } { 2 } , \quad \tau ( \mathrm { m } ( L ) ) = \ell - 1 - \tau ( L )
$$

due to Cavallo [Cav18, Corollary 3.7].

On the unknotting side, McCoy [McC17, Theorem 1] proved Kohn’s conjecture, which states that an alternating knot satisfies $u ( K ) = 1$ if and only if it has an unknotting crossing in every alternating diagram. More recently, the signature was shown to determine $u ( L )$ for special alternating links, which admit an alternating diagram where one of the planar spanning surfaces is a Seifert surface for the link.

Theorem B.34 ([MP26, Theorem 1 and Corollary 2]). Let L be an oriented non-split special alternating link with ℓ components. Then $\begin{array} { r } { u ( L ) \ = \ \frac { | \sigma ( L ) | + \ell - 1 } { 2 } } \end{array}$ if and only if L can be unlinked by that many crossing changes in any alternating or minimal diagram. In particular, $i f K$ is a special alternating knot, then $u ( K ) = \left| \sigma ( K ) \right| / 2$ if and only if K can be unknotted by $| \sigma ( K ) | / 2$ crossing changes in any alternating or minimal diagram.

## B.4.4 Pseudo-thin links

Pseudo-thin links form a class for which the Rasmussen s-invariant and related bounds can be computed more explicitly. This class includes all alternating links (see Ozsváth and Szabó [OS05, Lemma 3.2]), quasi-alternating links (see Cavallo [Cav15, Proposition 3.7]), and positive and negative links (see Cavallo [Cav15, after Proposition 3.7]), allowing us to apply the following result to all these classes of links.

Theorem B.35 (Strong slice genus of pseudo-thin links [Cav15, Theorem 5.4]). Let $L \ =$ $K _ { 1 } \cup \cdots \cup K _ { \ell }$ be an ℓ-component non-split pseudo-thin link which is also algebraically split $( i . e .$ lk $( K _ { i } , K _ { j } ) = 0$ for all $i \neq j )$ . Then

$$
\frac { | s ( L ) | + \ell - 1 } { 2 } \leq g _ { 4 } ^ { * } ( L ) .
$$

Moreover, $| s ( L ) | = | \sigma ( L ) |$ for an alternating link L by Proposition B.33. Hence, we can compute the signature instead of the Rasmussen s-invariant for them, which is computationally cheaper.

## B.5 Computational considerations

Some of the results presented above (Proposition B.3, Lemma B.5, and Theorem B.30) require knowing the unknotting numbers or the slice genera of the individual link components. This makes them somewhat impractical in a general setting. Our implementation addresses this issue using two distinct strategies: one general and one specific to the dataset we investigated.

For the general case, we estimate the required invariants of the components using the other bounds presented in this section. Given a link $L = K _ { 1 } \cup \cdot \cdot \cdot \cup K _ { \ell }$ , we first compute the bounds for each component $K _ { i }$ and then use these estimates in the inequalities for links obtained from the results mentioned above.

For our specific dataset, which considers links with up to 11 crossings, we can often identify the exact link components by querying a comprehensive table of knots with up to ten crossings. The table collects all prime knots with up to ten crossings, their mirror images, and all the possible connected sums that stay within this crossing limit. Completeness follows from the classification of knot diagrams through ten crossings by Cantarella, Chapman, and Mastin [CCM16]. In this range, every composite knot admitting an n-crossing diagram has prime summands whose crossing numbers total at most n. Hence, the table includes every possible knot type. For each entry of the table, we store its PD code, its crossing signs, and a vector of pre-computed invariants: Seifert genus, τ, total rank of the hat version of knot Floer homology, determinant, Alexander polynomial, $\sigma ,$ and Jones polynomial. Given a diagram for a component $K _ { i }$ of our link $L ,$ the identification proceeds as follows:

We simplify the diagram of $K _ { i }$ . If it simplifies to ten crossings or fewer, we proceed. Otherwise, we attempt identification through visible connected summands, each within the table’s crossing limit.

We compute the invariant vector for $K _ { i }$ and check how many table entries match it.

If the invariant vector matches a single entry, the knot is uniquely identified. If the remaining candidates difer only by mirroring, it is identified up to mirroring. Otherwise, we retain all matching candidates. For each invariant, we use the minimum of their lower bounds and the maximum of their upper bounds, provided the corresponding bounds are available for every candidate.

For prime candidates, we retrieve the available lower and upper bounds on the unknotting number and slice genus from our database. For composite candidates, we obtain upper bounds by summing the corresponding upper bounds of their prime summands, using subadditivity. After combining candidates as above, we combine these bounds with the independently computed bounds for $K _ { i }$ and apply the relevant link inequalities.

## Appendix C The unknotting sequences

In this appendix, we collect the drawings of the unknotting sequences of Table 22 that difer from the sequence $\left( 5 . 1 \right)$ of Brittenham and Hermiller. In each subsection, we present one sequence $D _ { 0 }  D _ { 1 }  D _ { 2 }  D _ { 3 }  D _ { 4 }$ , where $D _ { 4 }$ is a diagram of the unknot. We draw one figure per crossing change: the diagram $D _ { k }$ with the band of the crossing change in dashed red, before the simplification that gives $D _ { k + 1 }$ . The circled numbers are the crossings of the PD code of $D _ { k }$ and the grey numbers its edge labels; the arrows give the orientation. For each drawing we report

its PD code before the crossing change;

• <sup>its</sup> <sup>crossing</sup> <sup>signs;</sup>

the band path, which is drawn in dashed red, as a list of actions (Section 3.2), as printed by the replay scripts of the supporting data.

An action C{x}S{y}\_{type} refers to the crossing strand $( x , y ) { \mathrm { : } }$ the edge meeting crossing x at position y of its PD tuple, the positions being numbered counter-clockwise from the incoming under-strand. A band with start action C{x}S{y}\_start starts on that edge, in the face whose boundary contains the crossing strands $( x , y )$ and (x, y 1 mod 4); that is, in the corner of crossing x between them. An action $\mathrm { C } \{ \bf { x } \} \mathrm { { s } } \{ \mathrm { y } \}$ \_over (resp. $\mathtt { C } \{ \mathtt { x } \} \mathtt { S } \{ \mathtt { y } \}$ \_under) takes the band over (resp. under) the edge of $( x , y )$ into the next face. The last action, C{x}S{y}\_end1 or $\mathbb { C } \{ \mathrm { x } \} \mathrm { s } \{ \mathrm { y } \} .$ \_end2, takes the band across the edge of $( x , y )$ , which lies on the boundary of the face the band is in, so that the band clasps that edge. The two endings tell us how the crossing change is performed. The clasp gives two crossings between the band and the final edge. Following the knot from crossing x (the x in C{x}S{y}\_start) along the band, we meet one of them before the tip of the band, the closer crossing, and the other one after it, the farther crossing. With end1, the band passes over the final edge at the closer crossing and under it at the farther one. With end2, the opposite. Diagrams that are the same up to relabelling are drawn in the same way.

## C.1 Unknotting sequence for $9 _ { 8 } \sharp \mathsf { 9 } _ { 1 0 }$

The sequence found by the tuned random walker is

$$
9 _ { 8 } \sharp 9 _ { 1 0 } \to 4 _ { 1 } \sharp 9 _ { 1 0 } \to { \mathrm { K } } 1 5 { \mathrm { n } } 4 0 0 5 \to 8 _ { 1 4 } \to { \mathrm { u n k n o t . } }\tag{C.1}
$$

In Figure 7 we see the diagram $D _ { 0 }$ of $9 _ { 8 } \sharp \mathsf { 9 } _ { 1 0 }$ to which we apply the first band. Before the unknotting band, we have

PD code:

```prolog
[(11, 26, 12, 27), (13, 24, 14, 25),
(15, 21, 16, 20), (17, 29, 18, 28),
(19, 23, 20, 22), (21, 17, 22, 16),
(23, 14, 24, 15), (25, 12, 26, 13),
(27, 19, 28, 18), (29, 5, 30, 4),
(31, 3, 32, 2), (33, 9, 34, 8),
(35, 7, 0, 6), (1, 11, 2, 10),
(3, 31, 4, 30), (5, 33, 6, 32),
(7, 35, 8, 34), (9, 1, 10, 0)].
```

![](images/1bb45b23a0f43b800ebe6694c4ff6dbe8937c6a42d64cc90db1796952c06fc56.jpg)

Crossing signs:

The band applied is

Figure 7. Sequence (C.1), band from diagram $D _ { 0 }$ of $9 _ { 8 } \sharp \mathsf { 9 } _ { 1 0 }$ to diagram $D _ { 1 }$ of $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$

![](images/cb0decd6aed6bd7c536922ceb5b8570ef526a3049b505b490cbf91817d2914ed.jpg)  
Figure 8. Sequence (C.1), band from diagram $D _ { 1 }$ of $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$ to diagram $D _ { 2 }$ of K15n4005.

In Figure 9 we see the diagram $D _ { 2 }$ of K15n4005 to which we apply the third band. Before the unknotting band, we have PD code:

[(10, 17, 11, 18), (1, 21, 2, 20), (28, 10, 29, 9), (14, 22, 15, 21), (4, 24, 5, 23), (8, 28, 9, 27), (6, 26, 7, 25), (16, 11, 17, 12), (18, 16, 19, 15), (19, 3, 20, 2), (22, 0, 23, 29), (24, 8, 25, 7), (26, 6, 27, 5), (3, 12, 4, 13), (13, 0, 14, 1)].

Crossing signs:

[-1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1].

The band applied is ['C8S0\_start', 'C10S3\_end2'].

In Figure 8 we see the diagram $D _ { 1 }$ of $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$ to which we apply the second band. Before the unknotting band, we have PD code:

[(5, 10, 6, 11), (23, 15, 24, 14), (21, 5, 22, 4), (7, 13, 8, 12), (25, 17, 0, 16), (3, 21, 4, 20), (1, 19, 2, 18), (9, 6, 10, 7), (11, 9, 12, 8), (13, 25, 14, 24), (15, 23, 16, 22), (17, 3, 18, 2), (19, 1, 20, 0)].

Crossing signs:

[-1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]. The band applied is ['C7S3\_start', 'C4S0\_end1'].

![](images/602cedcc7c6ccf9fb675e7b03572cc8b975afcdec7fdc1a3ff12e0e338d4a5ab.jpg)  
Figure 9. Sequence (C.1), band from diagram $D _ { 2 }$ of K15n4005 to diagram $D _ { 3 }$ of $8 _ { 1 4 }$

![](images/006f1a262efd6b784099ab126876187ef228c45a03d5d88f5928c598354cc935.jpg)

Figure 10. Sequence (C.1), band from diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to diagram $D _ { 4 }$ of the unknot.

In Figure 10 we see the diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to which we apply the fourth band. Before the unknotting band, we have PD code:

[(5, 13, 6, 12), (9, 2, 10, 3), (3, 8, 4, 9), (1, 5, 2, 4), (13, 7, 14, 6), (11, 15, 12, 14), (15, 11, 0, 10), (7, 1, 8, 0)].   
Crossing signs:   
[1, -1, -1, 1, 1, 1, 1, 1].   
The band applied is   
['C7S2\_start', 'C7S1\_end2'].

After the last crossing change, we obtain the unknot.

## C.2 Unknotting sequence for $9 _ { 4 0 } \sharp \mathsf { S } _ { 1 0 }$

The sequence found by the tuned random walker is

$$
9 _ { 4 0 } \sharp 9 _ { 1 0 } \to \mathtt { K } 2 0 \to \mathtt { K } 1 5 \mathtt { n } 4 0 0 5 \to 8 _ { 1 4 } \to \mathrm { u n k n o t } .\tag{C.2}
$$

In Figure 11 we see the diagram $D _ { 0 }$ of $9 _ { 4 0 } \sharp \mathsf { \mathsf { S } } _ { 1 0 }$ to which we apply the first band. Before the unknotting band, we have PD code:

[(11, 25, 12, 24), (13, 22, 14, 23), (15, 21, 16, 20), (17, 13, 18, 12), (19, 28, 20, 29), (21, 27, 22, 26), (23, 19, 24, 18), (25, 16, 26, 17), (27, 15, 28, 14), (29, 5, 30, 4), (31, 3, 32, 2), (33, 9, 34, 8), (35, 7, 0, 6), (1, 11, 2, 10), (3, 31, 4, 30), (5, 33, 6, 32), (7, 35, 8, 34), (9, 1, 10, 0)].

The band applied is ['C2S3\_start', 'C15S0\_end2'].

![](images/fc905b920f782f59baa05c763a04fbd736b17d8974df4e97f6d8bdaf05ab5774.jpg)  
Figure 11. Sequence (C.2), band from diagram $D _ { 0 }$ of $9 _ { 4 0 } \sharp \mathsf { S } _ { 1 0 }$ to diagram $D _ { 1 }$ of K20.

![](images/d9b8aeb35bd31b4a3ef6edcbab6fc674b8c244e1b5b9e5a420e5f1a7340eae4d.jpg)  
Figure 12. Sequence (C.2), band from diagram $D _ { 1 }$ of K20 to diagram $D _ { 2 }$ of K15n4005.

In Figure 13 we see the diagram $D _ { 2 }$ of K15n4005 to which we apply the third band. Before the unknotting band, we have PD code:

[(27, 20, 28, 21), (21, 26, 22, 27), (18, 8, 19, 7), (16, 6, 17, 5), (14, 4, 15, 3), (24, 0, 25, 29), (9, 22, 10, 23), (25, 12, 26, 13), (2, 24, 3, 23), (28, 14, 29, 13), (1, 11, 2, 10), (4, 18, 5, 17), (6, 16, 7, 15), (8, 20, 9, 19), (11, 1, 12, 0)].

Crossing signs:

[-1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1].

The band applied is ['C1S3\_start', 'C6S0\_end1'].

In Figure 12 we see the diagram $D _ { 1 }$ of K20 to which we apply the second band. Before the unknotting band, we have PD code:

[(31, 7, 32, 6), (33, 4, 34, 5), (35, 3, 36, 2), (37, 33, 38, 32), (39, 10, 0, 11), (3, 9, 4, 8), (5, 39, 6, 38), (7, 36, 8, 37), (9, 35, 10, 34), (11, 23, 12, 22), (13, 21, 14, 20), (15, 29, 16, 28), (17, 27, 18, 26), (19, 31, 20, 30), (21, 13, 22, 12), (25, 15, 26, 14), (27, 17, 28, 16), (29, 19, 30, 18), (1, 24, 2, 25), (23, 0, 24, 1)].

Crossing signs:   
[1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1].   
The band applied is   
['C13S1\_start', 'C7S0\_end1'].

![](images/b391cc64bfd3c7ef38319b9b21fd75d9c388104e6896a0dc3dbe7665f11f6a53.jpg)  
Figure 13. Sequence (C.2), band from diagram $D _ { 2 }$ K15n4005 to diagram $D _ { 3 }$ of $8 _ { 1 4 }$

```prolog
[(11, 24, 12, 25), (13, 27, 14, 26),
(15, 20, 16, 21), (17, 29, 18, 28),
(19, 16, 20, 17), (21, 31, 22, 30),
(23, 12, 24, 13), (25, 23, 26, 22),
(27, 19, 28, 18), (29, 15, 30, 14),
(31, 5, 32, 4), (33, 3, 34, 2),
(35, 9, 36, 8), (37, 7, 0, 6),
(1, 11, 2, 10), (3, 33, 4, 32),
(5, 35, 6, 34), (7, 37, 8, 36),
(9, 1, 10, 0)].
Crossing signs:
[-1, 1, -1, 1, -1, 1, -1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1,
1].
The band applied is
['C10S1_start', 'C2S2_end2'].
```

![](images/3189e9de5f3de631383cdd115a37f99eceb6c40fb429449726fbf72b48955c06.jpg)

Figure 14. Sequence (C.2), band from diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to diagram $D _ { 4 }$ of the unknot.

In Figure 14 we see the diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to which we apply the fourth band. Before the unknotting band, we have PD code:

[(9, 2, 10, 3), (3, 8, 4, 9), (5, 13, 6, 12), (15, 11, 0, 10), (13, 7, 14, 6), (1, 5, 2, 4), (11, 15, 12, 14), (7, 1, 8, 0)].   
Crossing signs:   
[-1, -1, 1, 1, 1, 1, 1, 1].   
The band applied is   
['C1S1\_start', 'C3S2\_end1'].

After the last crossing change, we obtain the unknot.

## C.3 Unknotting sequence for $1 0 _ { 4 1 } \sharp \mathsf { \mathsf { S } } _ { 1 0 }$

The sequence found by the tuned random walker is

$$
1 0 _ { 4 1 } \sharp 9 _ { 1 0 } \to \mathtt { K } 2 1 \to \mathtt { K } 1 5 \mathtt { n } 4 0 0 5 \to 8 _ { 1 4 } \to \mathtt { u n k n o t } .\tag{C.3}
$$

In Figure 15 we see the diagram $D _ { 0 }$ of $1 0 _ { 4 1 } \sharp \mathsf { \mathsf { S } } _ { 1 0 }$ to which we apply the first band. Before the unknotting band, we have

![](images/027b28996f384db604e8a69167e196398661b33a4511bc8fe2d1ab0d60483cc5.jpg)

Figure 15. Sequence (C.3), band from diagram $D _ { 0 }$ of $1 0 _ { 4 1 } \sharp \mathsf { \mathsf { S } } _ { 1 0 }$ to diagram $D _ { 1 }$ of K21.

![](images/ea681a7c9c797c74b9387191e3d2ef06c0d1bc176a93d5dcf17bca47a03b2a74.jpg)  
Figure 16. Sequence (C.3), band from diagram $D _ { 1 }$ of K21 to diagram $D _ { 2 }$ of K15n4005.

In Figure 17 we see the diagram $D _ { 2 }$ of K15n4005 to which we apply the third band. Before the unknotting band, we have PD code:

[(3, 20, 4, 21), (18, 1, 19, 2), (29, 4, 0, 5), (2, 8, 3, 7), (22, 12, 23, 11), (24, 14, 25, 13), (5, 28, 6, 29), (26, 16, 27, 15), (19, 9, 20, 8), (16, 28, 17, 27), (14, 24, 15, 23), (10, 18, 11, 17), (12, 26, 13, 25), (6, 22, 7, 21), (9, 1, 10, 0)].

Crossing signs:

[-1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1].

The band applied is ['C6S2\_start', 'C9S0\_end1'].

In Figure 16 we see the diagram $D _ { 1 }$ of K21 to which we apply the second band. Before the unknotting band, we have PD code:

[(7, 22, 8, 23), (9, 25, 10, 24), (11, 18, 12, 19), (15, 27, 16, 26), (17, 14, 18, 15), (19, 29, 20, 28), (21, 8, 22, 9), (23, 21, 24, 20), (25, 17, 26, 16), (27, 11, 28, 10), (29, 41, 30, 40), (31, 39, 32, 38), (33, 5, 34, 4), (35, 3, 36, 2), (37, 7, 38, 6), (39, 31, 40, 30), (1, 33, 2, 32), (3, 35, 4, 34), (5, 37, 6, 36), (41, 12, 0, 13), (13, 0, 14, 1)].

Crossing signs:   
[-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   
1, -1, -1].   
The band applied is   
['C7S2\_start', 'C7S1\_end2'].

![](images/42e04c3b00e9518b2ad08e11738faa2c61448728ea373ad710a33c1f8d31dfd1.jpg)

Figure 17. Sequence (C.3), band from diagram $D _ { 2 }$ of K15n4005 to diagram $D _ { 3 }$ of $8 _ { 1 4 }$

![](images/12b9c9e0ef51dc74695019fd1f1c6055f6aeaabd4cc9e75df6736adcda0a0657.jpg)

In Figure 18 we see the diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to which we apply the fourth band. Before the unknotting band, we have PD code:

Figure 18. Sequence (C.3), band from $D _ { 3 }$ of $8 _ { 1 4 }$ to $D _ { 4 }$ of the unknot.

```matlab
[(11, 15, 12, 14), (9, 2, 10, 3),
(3, 8, 4, 9), (1, 5, 2, 4),
(13, 7, 14, 6), (15, 11, 0, 10),
(5, 13, 6, 12), (7, 1, 8, 0)].
Crossing signs:
[1, -1, -1, 1, 1, 1, 1, 1].
The band applied is
['C7S0_start', 'C7S3_end2'].
```

After the last crossing change, we obtain the unknot.

## C.4 Unknotting sequence for $1 0 _ { 1 1 5 } { \sharp } 9 _ { 1 0 }$

The sequence found by the tuned random walker is

$$
1 0 _ { 1 1 5 } \sharp \vartheta _ { 1 0 } \to 4 _ { 1 } \sharp \vartheta _ { 1 0 } \to { \mathrm { K 1 5 n } } 4 0 0 5 \to 8 _ { 1 4 } \to { \mathrm { u n k n o t . } }\tag{C.4}
$$

The diagrams $D _ { 2 }$ and $D _ { 3 }$ and their bands are those of the sequence of $9 _ { 4 0 } \sharp \mathsf { 9 } _ { 1 0 }$ , up to relabelling.

In Figure 19 we see the diagram $D _ { 0 }$ of $1 0 _ { 1 1 5 } { \sharp } 9 _ { 1 0 }$ to which we apply the first band. Before the unknotting band, we have PD code:

```prolog
[(11, 27, 12, 26), (19, 31, 20, 30),
(25, 20, 26, 21), (21, 12, 22, 13),
(13, 24, 14, 25), (23, 16, 24, 17),
(15, 29, 16, 28), (29, 15, 30, 14),
(17, 22, 18, 23), (27, 19, 28, 18),
(31, 5, 32, 4), (33, 3, 34, 2),
(35, 9, 36, 8), (37, 7, 0, 6),
(1, 11, 2, 10), (3, 33, 4, 32),
(5, 35, 6, 34), (7, 37, 8, 36),
(9, 1, 10, 0)].
```

![](images/29d97a17cbb5f381773e3864ed54060432842e62344853c7c559f1ed82eea6a7.jpg)

[1, 1, -1, -1, -1, -1, 1, 1, -1,   
1, 1, 1, 1, 1, 1, 1, 1, 1,   
1].

The band applied is   
['C2S1\_start', 'C2S0\_end1'].  
Figure 19. Sequence (C.4), band from diagram $D _ { 0 }$ of $1 0 _ { 1 1 5 } { \sharp } 9 _ { 1 0 }$ to diagram $D _ { 1 }$ of $4 _ { 1 }$ ♯9<sub>10</sub>.

Crossing signs:

![](images/bb76e91d2594a56f8a0fed32b2cddd86dcd9158c6b06acf0ab3ed2588aa6edb0.jpg)  
Figure 20. Sequence (C.4), band from diagram $D _ { 1 }$ of $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$ to diagram $D _ { 2 }$ of K15n4005.

In Figure 21 we see the diagram $D _ { 2 }$ of K15n4005 to which we apply the third band. Before the unknotting band, we have PD code:

[(6, 26, 7, 25), (12, 18, 13, 17), (2, 22, 3, 21), (29, 19, 0, 18), (4, 24, 5, 23), (15, 8, 16, 9), (26, 8, 27, 7), (24, 4, 25, 3), (9, 14, 10, 15), (20, 12, 21, 11), (16, 2, 17, 1), (19, 29, 20, 28), (22, 6, 23, 5), (27, 10, 28, 11), (13, 0, 14, 1)].

Crossing signs:

[1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1]. The band applied is ['C8S3\_start', 'C13S0\_end1'].

In Figure 20 we see the diagram $D _ { 1 }$ of $\mathtt { 4 _ { 1 } } \sharp \mathfrak { s } _ { 1 0 }$ to which we apply the second band. Before the unknotting band, we have PD code:

[(1, 19, 2, 18), (6, 10, 7, 9), (23, 15, 24, 14), (21, 13, 22, 12), (25, 17, 0, 16), (8, 3, 9, 4), (19, 3, 20, 2), (17, 25, 18, 24), (4, 7, 5, 8), (10, 6, 11, 5), (11, 23, 12, 22), (13, 21, 14, 20), (15, 1, 16, 0)].

The band applied is ['C10S1\_start', 'C9S1\_end2'].

![](images/4851b4f233df0ebc95b47d668916cae7e989f957a1cbac89dd6f8891266d0f16.jpg)  
Figure 21. Sequence (C.4), band from diagram $D _ { 2 }$ of K15n4005 to diagram $D _ { 3 }$ of $8 _ { 1 4 }$

![](images/5b215a51534da9aec0ad9736d1885cf6b8fd985fb6424dd3d5b84ec69d377678.jpg)

In Figure 22 we see the diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to which we apply the fourth band. Before the unknotting band, we have PD code:

[(9, 2, 10, 3), (1, 5, 2, 4), (13, 7, 14, 6), (3, 8, 4, 9), (15, 11, 0, 10), (5, 13, 6, 12), (11, 15, 12, 14), (7, 1, 8, 0)].

Figure 22. Sequence (C.4), band from diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to diagram $D _ { 4 }$ of the unknot.

After the last crossing change, we obtain the unknot.

## C.5 Unknotting sequence for $1 0 _ { 1 2 1 } \sharp \mathfrak { s } _ { 1 0 }$

The sequence found by the tuned random walker is

$$
1 0 _ { 1 2 1 } \sharp 9 _ { 1 0 } \to \mathtt { K } 2 1 \to \mathtt { K } 2 3 \to 8 _ { 1 4 } \to \mathrm { u n k n o t } .\tag{C.5}
$$

In Figure 23 we see the diagram $D _ { 0 }$ of $1 0 _ { 1 2 1 } \sharp \mathfrak { s } _ { 1 0 }$ to which we apply the first band. Before the unknotting band, we have

```prolog
[(11, 20, 12, 21), (13, 19, 14, 18),
(15, 31, 16, 30), (17, 25, 18, 24),
(19, 26, 20, 27), (21, 17, 22, 16),
(23, 29, 24, 28), (25, 12, 26, 13),
(27, 15, 28, 14), (29, 23, 30, 22),
(31, 5, 32, 4), (33, 3, 34, 2),
(35, 9, 36, 8), (37, 7, 0, 6),
(1, 11, 2, 10), (3, 33, 4, 32),
(5, 35, 6, 34), (7, 37, 8, 36),
(9, 1, 10, 0)].
```

![](images/1b0180aac5f2e82f23d5a806602f626dde24aa86935ffbd58c40a0675031b987.jpg)  
Figure 23. Sequence (C.5), band from diagram $D _ { 0 }$ of $1 0 _ { 1 2 1 } \sharp \mathfrak { s } _ { 1 0 }$ to diagram $D _ { 1 }$ of K21.

![](images/406d172bb787406a1e2612782dd13e69176d02b36337c4c972c8c3b41f39bbb8.jpg)  
Figure 24. Sequence (C.5), band from diagram $D _ { 1 }$ of K21 to diagram $D _ { 2 }$ of K23.

In Figure 25 we see the diagram $D _ { 2 }$ of K23 to which we apply the third band. Before the unknotting band, we have PD code:

[(10, 23, 11, 24), (14, 22, 15, 21), (18, 38, 19, 37), (20, 30, 21, 29), (22, 31, 23, 32), (26, 20, 27, 19), (28, 34, 29, 33), (30, 13, 31, 14), (32, 16, 33, 15), (34, 28, 35, 27), (35, 5, 36, 4), (38, 2, 39, 1), (40, 10, 41, 9), (42, 8, 43, 7), (45, 13, 0, 12), (3, 37, 4, 36), (6, 40, 7, 39), (8, 42, 9, 41), (11, 45, 12, 44), (5, 16, 6, 17), (17, 2, 18, 3), (43, 24, 44, 25), (25, 0, 26, 1)].

Crossing signs:

[-1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]. The band applied is ['C7S3\_start', 'C4S0\_end1'].

In Figure 24 we see the diagram $D _ { 1 }$ of K21 to which we apply the second band. Before the unknotting band, we have PD code:

[(8, 19, 9, 20), (10, 18, 11, 17), (14, 30, 15, 29), (16, 24, 17, 23), (18, 25, 19, 26), (20, 16, 21, 15), (22, 28, 23, 27), (24, 9, 25, 10), (26, 12, 27, 11), (28, 22, 29, 21), (30, 0, 31, 41), (32, 40, 33, 39), (34, 6, 35, 5), (36, 4, 37, 3), (38, 8, 39, 7), (40, 32, 41, 31), (2, 34, 3, 33), (4, 36, 5, 35), (6, 38, 7, 37), (1, 12, 2, 13), (13, 0, 14, 1)].

![](images/57c98e5d5784403e9fbe0309316b09b87f1e0b3f19bf4536dbf15e41909cd51f.jpg)

Figure 25. Sequence (C.5), band from diagram $D _ { 2 }$ of K23 to diagram $D _ { 3 }$ of $8 _ { 1 4 }$

![](images/a422be39c7398894c03d26b21b6734f4a0d73bdf60c8f5a4aa579ca63d2e8af5.jpg)

In Figure 26 we see the diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to which we apply the fourth band. Before the unknotting band, we have PD code:

Figure 26. Sequence (C.5), band from diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to diagram $D _ { 4 }$ of the unknot.

[(3, 8, 4, 9), (13, 7, 14, 6), (1, 5, 2, 4), (15, 11, 0, 10), (9, 2, 10, 3), (11, 15, 12, 14), (5, 13, 6, 12), (7, 1, 8, 0)].

After the last crossing change, we obtain the unknot.

## C.6 Unknotting sequence for $1 0 _ { 1 4 0 } \sharp \mathsf { 9 } _ { 1 0 }$ (random walker)

The sequence found by the tuned random walker is

$$
1 0 _ { 1 4 0 } \sharp 9 _ { 1 0 } \to \mathtt { K } 2 1 \to \mathtt { K } 2 3 \to 8 _ { 1 4 } \to \mathrm { u n k n o t } .\tag{C.6}
$$

The diagram $D _ { 3 }$ and its band are those of the sequence of $1 0 _ { 4 1 } \sharp \mathsf { \mathsf { S } } _ { 1 0 }$ , up to relabelling.

In Figure 27 we see the diagram $D _ { 0 }$ of $1 0 _ { 1 4 0 } \sharp \mathsf { 9 } _ { 1 0 }$ to which we apply the first band. Before the unknotting band, we have PD code:

```prolog
[(11, 25, 12, 24), (13, 20, 14, 21),
(15, 22, 16, 23), (16, 28, 17, 27),
(18, 26, 19, 25), (21, 14, 22, 15),
(23, 31, 24, 30), (26, 18, 27, 17),
(28, 20, 29, 19), (29, 13, 30, 12),
(31, 5, 32, 4), (33, 3, 34, 2),
(35, 9, 36, 8), (37, 7, 0, 6),
(1, 11, 2, 10), (3, 33, 4, 32),
(5, 35, 6, 34), (7, 37, 8, 36),
(9, 1, 10, 0)].
```

![](images/96436fb0ed068681ac4a06eb22fe9d4d41c039af4f8d5ec910e77e6b943f5932.jpg)

[1, -1, -1, 1, 1, -1, 1, 1, 1,   
1, 1, 1, 1, 1, 1, 1, 1, 1,   
1].

The band applied is ['C3S0\_start', 'C16S0\_end2'].

Figure 27. Sequence (C.6), band from diagram $D _ { 0 }$ of $1 0 _ { 1 4 0 } \sharp \mathsf { \mathsf { S } } _ { 1 0 }$ to diagram $D _ { 1 }$ of K21.

![](images/d1374f731130ef14bd482164bd7e611252b974fd10a9c79dffc81c188adaacaf.jpg)

Figure 28. Sequence (C.6), band from diagram $D _ { 1 }$ of K21 to diagram $D _ { 2 }$ of K23.

In Figure 29 we see the diagram $D _ { 2 }$ of K23 to which we apply the third band. Before the unknotting band, we have PD code:

[(39, 9, 40, 8), (36, 1, 37, 2), (40, 5, 41, 6), (44, 12, 45, 11), (4, 10, 5, 9), (3, 38, 4, 39), (7, 15, 8, 14), (10, 0, 11, 45), (13, 7, 14, 6), (12, 42, 13, 41), (15, 29, 16, 28), (17, 27, 18, 26), (19, 35, 20, 34), (21, 33, 22, 32), (24, 38, 25, 37), (27, 17, 28, 16), (31, 19, 32, 18), (33, 21, 34, 20), (35, 23, 36, 22), (43, 30, 44, 31), (29, 42, 30, 43), (2, 25, 3, 26), (23, 0, 24, 1)].

Crossing signs:

[1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]. The band applied is ['C6S1\_start', 'C6S0\_end1'].

In Figure 28 we see the diagram $D _ { 1 }$ of K21 to which we apply the second band. Before the unknotting band, we have PD code:

[(38, 12, 39, 11), (37, 4, 38, 5), (41, 8, 0, 9), (2, 14, 3, 13), (5, 11, 6, 10), (7, 40, 8, 41), (9, 17, 10, 16), (12, 4, 13, 3), (15, 7, 16, 6), (14, 40, 15, 39), (17, 29, 18, 28), (19, 27, 20, 26), (21, 35, 22, 34), (23, 33, 24, 32), (25, 37, 26, 36), (27, 19, 28, 18), (31, 21, 32, 20), (33, 23, 34, 22), (35, 25, 36, 24), (1, 30, 2, 31), (29, 0, 30, 1)].

Crossing signs:

[1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1]. The band applied is ['C4S0\_start', 'C11S3\_end2'].

![](images/2fe02a52e78c70ecebf3f99d3f9814041edb8134e77d6e46148b95f2b6a9e9fb.jpg)

Figure 29. Sequence (C.6), band from diagram $D _ { 2 }$ of K23 to diagram $D _ { 3 }$ of $8 _ { 1 4 }$

![](images/edd30af884c1b596d171be34b335504a5260925940eff48aa5cf5adf8d0bf1dd.jpg)

In Figure 30 we see the diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to which we apply the fourth band. Before the unknotting band, we have PD code:

Figure 30. Sequence (C.6), band from diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to diagram $D _ { 4 }$ of the unknot.

[(13, 7, 14, 6), (3, 8, 4, 9),   
(15, 11, 0, 10), (1, 5, 2, 4),   
(11, 15, 12, 14), (5, 13, 6, 12),   
(9, 2, 10, 3), (7, 1, 8, 0)].   
Crossing signs:   
[1, -1, 1, 1, 1, 1, -1, 1].   
The band applied is   
['C7S0\_start', 'C7S3\_end2'].

After the last crossing change, we obtain the unknot.

## C.7 Unknotting sequence for $1 0 _ { 1 4 0 } \sharp \mathsf { 9 } _ { 1 0 }$ (mixed agent)

The sequence found by the mixed agent is

$$
1 0 _ { 1 4 0 } \sharp 9 _ { 1 0 } \to \mathtt { K 2 1 } \to \mathrm { m ( \mathtt { K 1 5 n 4 8 6 6 } ) \to 8 _ { 1 4 } \to u n k n o t . }\tag{C.7}
$$

The diagram $D _ { 0 }$ is that of the random walker’s sequence above (the same PD code); the band difers.

In Figure 31 we see the diagram $D _ { 0 }$ of $1 0 _ { 1 4 0 } \sharp \mathsf { 9 } _ { 1 0 }$ to which we apply the first band. Before the unknotting band, we have PD code:

```prolog
[(11, 25, 12, 24), (13, 20, 14, 21),
(15, 22, 16, 23), (16, 28, 17, 27),
(18, 26, 19, 25), (21, 14, 22, 15),
(23, 31, 24, 30), (26, 18, 27, 17),
(28, 20, 29, 19), (29, 13, 30, 12),
(31, 5, 32, 4), (33, 3, 34, 2),
(35, 9, 36, 8), (37, 7, 0, 6),
(1, 11, 2, 10), (3, 33, 4, 32),
(5, 35, 6, 34), (7, 37, 8, 36),
(9, 1, 10, 0)].
```

[1, -1, -1, 1, 1, -1, 1, 1, 1,   
1, 1, 1, 1, 1, 1, 1, 1, 1,   
1].

The band applied is ['C7S2\_start', 'C14S0\_end1'].

![](images/4b4bbbc9c5c55823d1f5d24912debc4cee818911bbc75dfcce0ac0ceef51b0cd.jpg)

Figure 31. Sequence (C.7), band from diagram $D _ { 0 }$ of $1 0 _ { 1 4 0 } \sharp \mathsf { 9 } _ { 1 0 }$ to diagram $D _ { 1 }$ of K21.

The band applied is ['C6S3\_start', 'C6S2\_end1'].

![](images/92d8ecf0ab941371d8181846aaf2034265f0d2a36ad439c4b9046c96f63c5f3f.jpg)  
Figure 32. Sequence (C.7), band from diagram D<sub>1</sub> of K21 to diagram D<sub>2</sub> of m(K15n4866).

In Figure 33 we see the diagram D<sub>2</sub> of m(K15n4866) to which we apply the third band. Before the unknotting band, we have PD code:

[(7, 20, 8, 21), (2, 5, 3, 6), (21, 6, 22, 7), (4, 10, 5, 9), (29, 19, 0, 18), (8, 1, 9, 2), (27, 17, 28, 16), (23, 13, 24, 12), (25, 15, 26, 14), (10, 4, 11, 3), (11, 25, 12, 24), (13, 23, 14, 22), (15, 29, 16, 28), (17, 27, 18, 26), (19, 1, 20, 0)].

[-1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1].

The band applied is ['C9S3\_start', 'C10S3\_end2'].

In Figure 32 we see the diagram D<sub>1</sub> of K21 to which we apply the second band. Before the unknotting band, we have PD code:

[(12, 26, 13, 25), (14, 21, 15, 22), (16, 23, 17, 24), (17, 31, 18, 30), (19, 27, 20, 26), (22, 15, 23, 16), (24, 34, 25, 33), (27, 19, 28, 18), (31, 21, 32, 20), (32, 14, 33, 13), (34, 6, 35, 5), (36, 4, 37, 3), (38, 10, 39, 9), (40, 8, 41, 7), (2, 12, 3, 11), (4, 36, 5, 35), (6, 38, 7, 37), (8, 40, 9, 39), (10, 0, 11, 41), (1, 28, 2, 29), (29, 0, 30, 1)].

![](images/5ec183cee931f56bc430bdf65c548e90b2bbce963b6bcf4ad24d39bbadc700f2.jpg)  
Figure 33. Sequence (C.7), band from diagram D of m(K15n4866) to diagram D<sub>3</sub> of 8<sub>14</sub>.

![](images/5dbd01efa3d8d3a9f5c80cd38b3733d487f779b0389c1ebebb3e0c78b342c8f1.jpg)

Figure 34. Sequence (C.7), band from diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to diagram $D _ { 4 }$ of the unknot.

In Figure 34 we see the diagram $D _ { 3 }$ of $8 _ { 1 4 }$ to which we apply the fourth band. Before the unknotting band, we have PD code:

After the last crossing change, we obtain the unknot.

[(13, 7, 14, 6), (3, 8, 4, 9), (9, 2, 10, 3), (1, 5, 2, 4), (11, 15, 12, 14), (5, 13, 6, 12), (15, 11, 0, 10), (7, 1, 8, 0)].

Crossing signs:

The band applied is

['C7S1\_start', 'C7S0\_end1'].

## Bibliography

[Brittenham] Mark Brittenham. Unknotting number search. Accessed: 2026. url: https://www. math.unl.edu/\~mbrittenham2/unknottingsearch/ (visited on 05/29/2026).

[KnotAtlas] Dror Bar-Natan and Kim Morrison. KnotAtlas: The Knot Atlas. url: https:// katlas.org/wiki/Main\_Page (visited on 05/29/2026).

[KnotInfo] Charles Livingston and Allison H. Moore. KnotInfo: Table of Knot Invariants. url: https://knotinfo.org/ (visited on 05/28/2026).

[LinkInfo] Charles Livingston and Allison H. Moore. LinkInfo: Table of Link Invariants. url: https://knotinfo.org/linkinfo/ (visited on 03/16/2025).

[SnapPy] Marc Culler et al. SnapPy, a computer program for studying the geometry and topology of 3-manifolds. url: http://snappy.computop.org.

[Spherogram-nim] Yutong Dai. Spherogram-nim. Nim port of https://github.com/3- manifolds/ Spherogram. url: https://github.com/ascchrvalstr/Spherogram-nim.

[Ace+21] Paolo Aceto et al. “Pretzel links, mutation, and the slice-ribbon conjecture”. In: Mathematical Research Letters 28.4 (2021), pp. 945–966. doi: 10.4310/MRL.2021. v28.n4.a1.

[Ada04] Colin C. Adams. The Knot Book: An Elementary Introduction to the Mathematical Theory of Knots. Providence, RI: American Mathematical Society, 2004. isbn: 978- 0-8218-3678-1.

[Ale28] J. W. Alexander. “Topological invariants of knots and links”. In: Trans. Amer. Math. Soc. 30 (1928), pp. 275–306. doi: 10.1090/S0002-9947-1928-1501429-1.

[App+25] Taylor Applebaum et al. “The unknotting number, hard unknot diagrams, and reinforcement learning”. In: Experimental Mathematics (2025). doi: 10.1080/10586458. 2025.2542174. arXiv: 2409.09032 [math.GT].

[AT20] Yuta Akimoto and Kouki Taniyama. “Unknotting numbers and crossing numbers of spatial embeddings of a planar graph”. In: Journal of Knot Theory and Its Ramifications 29.14 (2020), p. 2050095. doi: 10.1142/S0218216520500959.

[BH25] Mark Brittenham and Susan Hermiller. Unknotting number is not additive under connected sum. 2025. arXiv: 2506.24088 [math.GT].

[BH26] Mark Brittenham and Susan Hermiller. Unknotting number and connected sums: The knots 4 and 5 . 2026. arXiv: 2601.18757. url: https://arxiv.org/abs/2601. 18757.

[BV26] Dror Bar-Natan and Roland van der Veen. “A Fast, Strong, Topologically Meaningfu and Fun Knot Invariant”. In: (2026). Version 4, 6 May 2026. arXiv: 2509.18456 [math.GT]. url: https://arxiv.org/abs/2509.18456.

[BW08] Anna Beliakova and Stephan Wehrli. “Categorification of the Colored Jones Polynomial and Rasmussen Invariant of Links”. In: Canadian Journal of Mathematics 60.6 (2008), pp. 1240–1266. doi: 10.4153/CJM-2008-053-1.

[Cav15] Alberto Cavallo. “On the slice genus and some concordance invariants of links”. In: Journal of Knot Theory and Its Ramifications 24.4 (2015), p. 1550021. doi: 10.1142/S0218216515500212.

[Cav18] Alberto Cavallo. “The concordance invariant tau in link grid homology”. In: Algebraic & Geometric Topology 18.4 (2018), pp. 1917–1951. doi: 10.2140/agt.2018.18.1917.

[CC20] Alberto Cavallo and Carlo Collari. “Slice-torus Concordance Invariants and Whitehead Doubles of Links”. In: Canadian Journal of Mathematics 72.6 (2020), pp. 1423– 1462. doi: 10.4153/S0008414X19000294.

[CCM16] Jason Cantarella, Harrison Chapman, and Matt Mastin. “Knot probabilities in random diagrams”. In: Journal of Physics A: Mathematical and Theoretical 49.40 (Sept. 2016), p. 405001. issn: 1751-8121. doi: 10.1088/1751-8113/49/40/405001. url: http://dx.doi.org/10.1088/1751-8113/49/40/405001.

[CL86] Tim D. Cochran and W. B. Raymond Lickorish. “Unknotting information from 4- manifolds”. In: Transactions of the American Mathematical Society 297.1 (1986), pp. 125–142. doi: 10.1090/S0002-9947-1986-0849471-4.

[Col21] Carlo Collari. “Slice-torus link invariants, combinatorial invariants and positivity conditions”. In: Bulletin of the London Mathematical Society 53.4 (2021), pp. 1072– 1092. doi: https : / / doi . org / 10 . 1112 / blms . 12485. eprint: https : / / londmat hsoc.onlinelibrary.wiley.com/doi/pdf/10.1112/blms.12485. url: https: //londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/blms.12485.

[Con21] Anthony Conway. “The Levine–Tristram Signature: A Survey”. In: 2019-20 MATRIX Annals. MATRIX Book Series. Springer International Publishing, 2021, pp. 31–56. DOI:10.1007/978-3-030-62497-2 2.

[Cro04] Peter R. Cromwell. Knots and Links. Cambridge: Cambridge University Press, 2004. [Dav+21] Alex Davies et al. “Advancing mathematics by guiding human intuition with AI”. In: Nature 600.7887 (2021), pp. 70–74. doi: 10.1038/s41586-021-04086-x.

[DG25] Nathan M. Dunfield and Sherry Gong. “Ribbon concordances and slice obstructions: experiments and examples”. In: (2025). arXiv: 2512.21825. url: https://arxiv. org/abs/2512.21825.

[FG03] Vincent Florens and Patrick M. Gilmer. “On the slice genus of links”. In: Algebraic & Geometric Topology 3.2 (2003), pp. 905–920. doi: 10.2140/agt.2003.3.905.

[FM11] Benson Farb and Dan Margalit. A Primer on Mapping Class Groups (PMS-49). Princeton University Press, 2011. doi: 10.23943/princeton/9780691147949.001. 0001.

[FM66] Ralph H. Fox and John W. Milnor. “Singularities of 2-spheres in 4-space and cobordism of knots”. In: Osaka Journal of Mathematics 3.2 (1966), pp. 257–267. url: https://projecteuclid.org/euclid.ojm/1200691730.

[Fox61] R. H. Fox. “Some problems in knot theory”. In: Topology of 3-manifolds and related topics (Proc. The Univ. of Georgia Institute, 1961). Prentice-Hall, Inc., Englewood Clifs, NJ, 1961, pp. 168–176.

[GST10] Robert E. Gompf, Martin Scharlemann, and Abigail Thompson. “Fibered knots and potential counterexamples to the Property 2R and Slice-Ribbon Conjectures”. In: Geometry & Topology 14.4 (2010), pp. 2305–2347. doi: 10.2140/gt.2010.14.2305. arXiv: 1103.1601 [math.GT].

[Guk+21] Sergei Gukov et al. “Learning to unknot”. In: Machine Learning: Science and Technology 2.2 (2021), p. 025035. doi: 10.1088/2632-2153/abe91f. arXiv: 2010.16263 [math.GT].

[Guk+25] Sergei Gukov et al. “Searching for ribbons with machine learning”. In: Machine Learning: Science and Technology 6.2 (2025), p. 025065. doi: 10.1088/2632-2153/ 362. arXiv: 2304 09304 [ G ].

[Hug20] Mark C. Hughes. “A neural network approach to predicting and computing knot invariants”. In: Journal ofKnot Theory and Its Ramifications 29.3 (2020), p. 2050005. doi: 10.1142/S0218216520500054. arXiv: 1610.05744 [math.GT].

[HW16] Jennifer Hom and Zhongtao Wu. “Four-ball genus bounds and a refinement of the Ozsváth-Szabó tau invariant”. In: Journal of Symplectic Geometry 14.1 (2016), pp. 305–323. doi: 10.4310/JSG.2016.v14.n1.a12.

[Jab26] Michal Jabłonowski. “Integer Knot Invariants: Inequalities, Computations, and Open Problems”. In: (2026). arXiv: 2605.22652. url: https://arxiv.org/abs/2605. 22652.

[JKP19] Vishnu Jejjala, Arjun Kar, and Onkar Parrikar. “Deep learning the hyperbolic volume of a knot”. In: Physics Letters B 799 (2019), p. 135033. doi: 10 . 1016 / j . physletb.2019.135033. arXiv: 1902.05547 [hep-th].

[Kau78] Louis H. Kaufman. “Signature of branched fibrations”. In: Knot Theory. Ed. by Jean-Claude Hausmann. Springer Berlin Heidelberg, 1978, pp. 203–217. doi: 10 . 1007/BFb0062972.

[Kaw02] Tomomi Kawamura. “On unknotting numbers and four-dimensional clasp numbers of links”. In: Proceedings of the American Mathematical Society 130.1 (2002), pp. 243– 252. doi: 10.1090/S0002-9939-01-06000-2.

[Kho00] Mikhail Khovanov. “A categorification of the Jones polynomial”. In: Duke Mathematical Journal 101.3 (2000), pp. 359–426. doi: 10.1215/S0012-7094-00-10131-7. url: https://doi.org/10.1215/S0012-7094-00-10131-7.

[KM93] Peter B. Kronheimer and Tomasz S. Mrowka. “Gauge theory for embedded surfaces, I”. In: Topology 32.4 (1993), pp. 773–826. doi: 10.1016/0040-9383(93)90051-V.

[KM95] Peter B. Kronheimer and Tomasz S. Mrowka. “Gauge theory for embedded surfaces, II”. In: Topology 34.1 (1995), pp. 37–97. doi: 10.1016/0040-9383(94)E0003-3.

[Lee05] Eun Soo Lee. “An endomorphism of the Khovanov invariant”. In: Advances in Mathematics 197.2 (2005), pp. 554–586. doi: 10.1016/j.aim.2004.10.015.

[Lee26] Seong-Jin Lee. “Computation of unknotting numbers: which knot breaks the Bernhard-Jablan Conjecture”. In: (2026). arXiv: 2609.09861 [math.GT]. url: https://arxiv. org/abs/2609.09861.

[Lev69] Jerome Levine. “Knot cobordism groups in codimension two”. In: Commentarii Mathematici Helvetici 44 (1969), pp. 229–244. doi: 10.1007/BF02564525.

[Lew14] Lukas Lewark. “Rasmussen’s spectral sequences and the slN-concordance invariants”. In: Advances in Mathematics 260 (2014), pp. 59–83. doi: 10.1016/j.aim.2014.04. 003.

[Liv04] Charles Livingston. “Computations of the Ozsváth-Szabó knot concordance invariant”. In: Geometry & Topology 8 (2004), pp. 735–742. doi: 10.2140/gt.2004.8.735.

[McC17] Duncan McCoy. “Alternating knots with unknotting number one”. In: Advances in Mathematics 305 (2017), pp. 757–802. issn: 0001-8708. doi: https://doi.org/ 10.1016/j.aim.2016.09.033. url: https://www.sciencedirect.com/science/ article/pii/S0001870816312944.

[MP26] Duncan McCoy and JungHwan Park. “Special alternating links of minimal unlinking number”. In: (2026). arXiv: 2603.10732 [math.GT]. url: https://arxiv.org/abs/ 2603.10732.

[Mur65] Kunio Murasugi. “On a certain numerical invariant of link types”. In: Transactions of the American Mathematical Society 117 (1965), pp. 387–422. doi: 10.2307/1994215.

[Nak00] Takuji Nakamura. “Four-genus and unknotting number of positive knots and links”. In: Osaka Journal of Mathematics 37.2 (2000), pp. 441–451. doi: 10.18910/8270.

[NHR99] Andrew Y. Ng, Daishi Harada, and Stuart J. Russell. “Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping”. In: Proceedings of the Sixteenth International Conference on Machine Learning. Morgan Kaufmann Publishers Inc., 1999, pp. 278–287.

[NO15] Matthias Nagel and Brendan Owens. “Unlinking information from 4-manifolds”. In: Bulletin of the London Mathematical Society 47.6 (Dec. 2015), pp. 964–979. issn: 0024-6093. doi: 10.1112/blms/bdv072. url: https://doi.org/10.1112/blms/ bdv072.

[Nog14] Fernando Nogueira. Bayesian Optimization: Open source constrained global optimization tool for Python. Version 2.0.0. 2014. url: https://github.com/bayesianoptimization/BayesianOptimization.

[OS03a] Peter Ozsváth and Zoltán Szabó. “Heegaard Floer homology and alternating knots”. In: Geometry & Topology 7 (2003), pp. 225–254. doi: 10.2140/gt.2003.7.225.

[OS03b] Peter Ozsváth and Zoltán Szabó. “Knot Floer homology and the four-ball genus”. In: Geometry & Topology 7.2 (2003), pp. 615–639. doi: 10.2140/gt.2003.7.615.

[OS05] Peter Ozsváth and Zoltán Szabó. “On the Heegaard Floer homology of branched double-covers”. In: Advances in Mathematics 194.1 (2005), pp. 1–33. doi: 10.1016/ j.aim.2004.05.008.

[OS11] Peter Ozsváth and Zoltán Szabó. “Knot Floer homology and rational surgeries”. In: Algebraic & Geometric Topology 11.1 (2011), pp. 1–68. doi: 10.2140/agt.2011.11. 1.

[Pre12] Lutz Prechelt. “Early Stopping — But When?” In: Neural Networks: Tricks of the Trade: Second Edition. 2nd. Vol. 7700. Lecture Notes in Computer Science. Springer Berlin Heidelberg, 2012, pp. 53–67. isbn: 978-3-642-35289-8. doi: 10.1007/978-3- 642-35289-8\_5. url: https://doi.org/10.1007/978-3-642-35289-8\_5.

[Raf+19] Antonin Rafin et al. Stable Baselines3. https://github.com/DLR- RM/stablebaselines3. 2019.

[Ras03] Jacob Rasmussen. Floer homology and knot complements. 2003. doi: 10 . 48550 / arXiv.math/0306378.

[Ras10] Jacob Rasmussen. “Khovanov homology and the slice genus”. In: Inventiones mathematicae 182.2 (2010), pp. 419–447. doi: 10.1007/s00222-010-0275-6.

[Sch+17] John Schulman et al. Proximal Policy Optimization Algorithms. 2017. arXiv: 1707. 06347 [cs.LG].

[Sch21] Dirk Schütz. “A fast algorithm for calculating S-invariants”. In: Glasgow Mathematical Journal 63.2 (2021), pp. 378–399. doi: 10.1017/S0017089520000257.

[Sch22] Dirk Schütz. “Corrigendum to: A fast algorithm for calculating S-invariants”. In: Glasgow Mathematical Journal 64.2 (2022), pp. 526–526. doi: 10.1017/S001708952 100032X.

[Sch25a] Dirk Schütz. Eficient calculations of S-invariants for links. 2025. arXiv: 2508.11373 [math.GT]. url: https://arxiv.org/abs/2508.11373.

[Sch25b] Dirk Schütz. KnotJob. Version website. 2025. url: https://www.maths.dur.ac.uk/ users/dirk.schuetz/knotjob.html

[Sei35] Herbert Seifert. “Über das Geschlecht von Knoten”. In: Mathematische Annalen 110 (1935), pp. 571–592. doi: 10.1007/BF01448044.

[Sil+16] David Silver et al. “Mastering the game of Go with deep neural networks and tree search”. In: Nature 529.7587 (2016), pp. 484–489. doi: 10.1038/nature16961.

[Sil+17] David Silver et al. “Mastering the game of Go without human knowledge”. In: Nature 550.7676 (2017), pp. 354–359. doi: 10.1038/nature24270.

[Sto04] A. Stoimenow. “Polynomial values, the linking form and unknotting numbers”. In: Mathematical Research Letters 11.5-6 (2004), pp. 755–769. doi: 10.4310/MRL.2004. v11.n6.a4.

[SW06] James E. Smith and Robert L. Winkler. “The Optimizer’s Curse: Skepticism and Postdecision Surprise in Decision Analysis”. In: Management Science 52.3 (2006), pp. 311–322. doi: 10.1287/mnsc.1050.0451.

[Tes91] Gerald Tesauro. “Practical Issues in Temporal Diference Learning”. In: Advances in Neural Information Processing Systems. Ed. by J. Moody, S. Hanson, and R.P. Lippmann. Vol. 4. Morgan-Kaufmann, 1991. url: https://proceedings.neurips. cc/paper\_files/paper/1991/file/68ce199ec2c5517597ce0a4d89620f55-Paper. pdf.

[Tri69] A. G. Tristram. “Some cobordism invariants for links”. In: Mathematical Proceedings of the Cambridge Philosophical Society 66 (1969), pp. 251–264. doi: 10 . 1017 / S 0305004100044947.

[Tro62] H. F. Trotter. “Homology of Group Systems With Applications to Knot Theory”. In: Annals of Mathematics 76.3 (1962), pp. 464–498. doi: 10.2307/1970369.

Department of Mathematics<sub>,</sub> Humboldt-Universität zu Berlin<sub>,</sub> Unter den Linden 6<sub>,</sub> 10099 Berlin<sub>,</sub> Germany

DRW Holdings<sub>,</sub> 540 West Madison Street<sub>,</sub> Suite 2500<sub>,</sub> Chicago<sub>,</sub> IL 60661<sub>,</sub> US Email address: ohayman@drwholdings.com

Mathematical Institute<sub>,</sub> University of Oxford<sub>,</sub> Andrew Wiles Building<sub>,</sub> Radcliffe Observatory Quarter, Woodstock Road, Oxford, OX2 6GG, UK Email address: juhasza@maths.ox.ac.uk

Department of Mathematics<sub>,</sub> Università di Padova<sub>,</sub> via Trieste<sub>,</sub> 63<sub>,</sub> 35121 Padova<sub>,</sub> Italy Email address: ludovico.morellato@phd.unipd.it