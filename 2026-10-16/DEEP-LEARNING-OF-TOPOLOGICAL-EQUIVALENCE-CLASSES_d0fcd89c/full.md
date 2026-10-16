# DEEP LEARNING OF TOPOLOGICAL EQUIVALENCE CLASSES

LINDEN DISNEY-HOGG<sup>∗</sup> <sup>ID</sup> AND MAX ZUEV <sup>ID</sup>

Abstract. We demonstrate an application of deep learning to the problem of identifying whether two generating vectors correspond to topologically equivalent finite group actions on Riemann surfaces, showing how architectures developed for semantic understanding have use for pure mathematical problems. In doing so we discuss how desired algebraic structure influences model design, and explore which structural properties of topological equivalence classes the model has successfully learnt.

## 1. Introduction

A fundamental problem in the study of Riemann surfaces is an understanding of their moduli, and one key approach to this is via describing the possible holomorphic actions of a finite group G on a smooth compact connected Riemann surface $S ,$ defined as monomorphisms $\epsilon : G \to \operatorname { A u t } ( S )$ . A foundational result in this direction is the theorem of Hurwitz showing that when S has genus $g \ \geq \ 2 , \ \vert \operatorname { A u t } ( S ) \vert \ \leq$ $8 4 ( g - 1 ) \ [ \mathrm { H u r 9 2 } ]$

Associated to the image of a group action $\mathcal { G } = \epsilon ( G ) \leq \mathrm { A u t } ( S )$ one has the quotient map $\tau : S  S / \mathcal { G }$

Definition 1.1 ([LR22]). Given an action of G on S, define the quotient genus to be $g _ { 0 } = g ( S / \mathcal { G } ) \ge 0$ . Moreover, letting $Q _ { i } , i = 1 , \ldots , r _ { \mathrm { { \scriptsize 3 } } }$ , denote the branch points $o f \pi , f o r$ any $P _ { i } \in \pi ^ { - 1 } ( Q _ { i } )$ denote with $c _ { i } = | \mathcal { G } _ { P _ { i } } | > 1$ the order of the stabiliser of $P _ { i }$ in G. We call the integer tuple $( g _ { 0 } ; c _ { 1 } , \ldots , c _ { r } )$ the signature of the action.

Moreover, given any finite group G and signature $( g _ { 0 } ; c _ { 1 } , \ldots , c _ { r } )$ , define a generating vector associated to the signature to be $\{ \alpha _ { 1 } , \beta _ { 1 } , \ldots , \alpha _ { g _ { 0 } } , \beta _ { g _ { 0 } } , \gamma _ { 1 } , \ldots , \gamma _ { r } \} \subset G$ such that

$\mathrm { ~ , ~ } \left\{ \alpha _ { 1 } , \beta _ { 1 } , \ldots , \alpha _ { g _ { 0 } } , \beta _ { g _ { 0 } } , \gamma _ { 1 } , \ldots , \gamma _ { r } \right\}$ generates G, • γ<sub>i</sub> has order c<sub>i</sub>, and

$$
\bullet \ \Pi _ { i = 1 } ^ { g _ { 0 } } ( \alpha _ { i } \beta _ { i } \alpha _ { i } ^ { - 1 } \beta _ { i } ^ { - 1 } ) \prod _ { j = 1 } ^ { r } \gamma _ { j } = 1 .
$$

The Riemann existence theorem states that, given a finite group $G ,$ a signature $( g _ { 0 } ; c _ { 1 } , \ldots , c _ { r } )$ , and a generating vector in $G$ associated to this signature, then there exists an action of G on a surface $S$ of genus g if and only if the Riemann-Hurwitz equation

$$
2 g - 2 = \left| G \right| \left[ \left( 2 g _ { 0 } - 2 \right) + \sum _ { i = 1 } ^ { r } \left( 1 - \frac { 1 } { c _ { i } } \right) \right]
$$

holds [Bro90]. In this work we shall restrict ourselves to investigating group actions where $g _ { 0 } = 0$ (and hence $r \geq 3 )$ , but this is not a theoretical barrier to the application of our methods.

Remark 1.2. Given the data of the character table of G one may use character theory to estimate the dificulty of finding all generating vectors of G for a particular signature, and the expected number. Denote with $C _ { i }$ a conjugacy class in $G ,$ let $x _ { i } \in C _ { i }$ , and denote with $\{ \chi _ { i } \}$ the irreducible characters of G. It is then known the size of the conjugacy classes may be determined by the orthogonality relation

$$
\frac { 1 } { | G | } \sum _ { i } \chi _ { i } ( x _ { r } ) \chi _ { i } ( x _ { s } ) = \frac { \delta _ { r s } } { | C _ { r } | } .
$$

Fixing conjugacy classes $C _ { 1 } , \ldots , C _ { r }$ one may imagine an algorithm which searches for generating vectors with $\gamma _ { i } \in C _ { i }$ by looping over the cartesian product $C _ { 1 } \times \cdots \times$ $C _ { r - 1 }$ and checking whether the inverse of the product is in $C _ { r }$ . This is a search over a space of size $\left| C _ { 1 } \right| \times \cdots \times \left| C _ { r - 1 } \right|$ which may be determined, and the number of solutions is given by Frobenius’ formula [Bre00, Theorem 14.1]

$$
| \{ ( \gamma _ { 1 } , \dots , \gamma _ { r } ) | \gamma _ { i } \in C _ { i } , \prod _ { i = 1 } ^ { r } \gamma _ { i } = 1 \} | = \frac { \prod _ { i = 1 } ^ { r } | C _ { i } | } { | G | } \sum _ { i } \frac { \prod _ { j = 1 } ^ { r } \chi _ { i } ( x _ { j } ) } { \chi _ { i } ( 1 ) ^ { r - 2 } } .
$$

As is natural in mathematics one wants to describe all such group actions up to a concept of equivalence; many such concepts of equivalence have been considered [Bro22], but in this paper we shall focus on perhaps the most well studied, namely topological equivalence.

Definition 1.3 ([Kur87], [Bro22]). Actions $\epsilon _ { i }$ of G on Riemann surfaces $S _ { i } \ ( i =$ 1, 2) of the same genus are said to be topologically equivalent if there exists $h : S _ { 1 }  S _ { 2 }$ an orientation preserving homeomorphism and $\sigma \in \operatorname { A u t } ( G )$ such that $\forall g \in G , \epsilon _ { 2 } ( g ) = h \circ \epsilon _ { 1 } ( \sigma ( g ) ) \circ h ^ { - 1 }$

This definition of equivalence is particularly important because topological equivalence classes biject with conjugacy classes of finite subgroups of the mapping class group, a fundamental object in many areas of mathematics [Bro90]. It is clear that the signature of a group action is an invariant of its topological equivalence class up to permutation of the $c _ { i }$ . Through the theory of surface kernel epimorphisms one can show that topological equivalence classes correspond to orbits of generating vectors under the action [Bro90, BRTC23] of

• Aut(G), with $\sigma \in \operatorname { A u t } ( G )$ acting on $( \gamma _ { 1 } , \ldots , \gamma _ { r } )$ as

$$
\sigma \cdot ( \gamma _ { 1 } , \dots , \gamma _ { r } ) = ( \sigma ( \gamma _ { 1 } ) , \dots , \sigma ( \gamma _ { r } ) ) ,
$$

and

• the subgroup of the braid group $\mathbb { B } _ { r }$ generated by $\Phi _ { i } , i = 1 , \dots , r - 1$ , acting as

$$
\Phi _ { i } \cdot ( \gamma _ { 1 } , \dots , \gamma _ { r } ) = ( \gamma _ { 1 } , \dots , \gamma _ { i - 1 } , \gamma _ { i } \gamma _ { i + 1 } \gamma _ { i } ^ { - 1 } , \gamma _ { i } , \dots , \gamma _ { r } ) .
$$

We shall further fix the ordering of the signature $\big ( c _ { 1 } , \ldots , c _ { r } \big )$ , and consider only the subgroup of B<sub>r</sub> which does not permute the signature.

Example 1.4. Consider the example of $G = \langle \gamma | \gamma ^ { 4 } = 1 \rangle \cong C _ { 4 }$ acting with signature $( 0 ; 4 , 4 , 4 , 4 )$ in genus $g = 3$ . Any generating vector must be of the form

$$
v = ( \gamma ^ { a _ { 1 } } , \gamma ^ { a _ { 2 } } , \gamma ^ { a _ { 3 } } , \gamma ^ { a _ { 4 } } ) ,
$$

where $a _ { i } = 1 , 3$ and $\textstyle \sum _ { i } a _ { i } \equiv 0$ mod 4. $\operatorname { A u t } ( G ) = \langle \gamma \mapsto \gamma ^ { 3 } \rangle$ and any permutation preserves the signature, so we have two equivalence classes of generating vector

• two of the $a _ { i }$ equal 1, and two of the $a _ { i }$ equal 3,

• all of the $a _ { i }$ are equal.

Example 1.5. Consider the group $G = A _ { 4 }$ the alternating group on $\it 4$ elements acting with signature $( 0 ; 3 , 3 , 3 , 3 )$ in genus $g \ : = \ : 5$ . Any generating vector must consist of four 3-cycles, two examples are

$$
v _ { 1 } = ( ( 1 , 2 , 4 ) , ( 1 , 3 , 4 ) , ( 2 , 3 , 4 ) , ( 1 , 2 , 3 ) ) ,
$$

$$
v _ { 2 } = ( ( 2 , 3 , 4 ) , ( 1 , 3 , 4 ) , ( 1 , 2 , 4 ) , ( 1 , 2 , 3 ) ) .
$$

One may check that these are in distinct equivalence classes despite difering by a permutation.

Given a generating vector it is easy to compute its associated signature but this is not a complete invariant of the equivalence class, that is there exist topologically inequivalent generating vectors with the same signature (as demonstrated in Examples 1.4 and 1.5). There are more refined invariants, such as the character of the induced representation of $G$ on the space $\mathcal { H } ^ { q } ( S )$ of holomorphic q-diferentials on $S ,$ which are similarly quick to compute but are still not complete. The exception to this is in the case where $G$ is cyclic and the ‘total rotation data system’ provides a complete invariant [Bro22].

In contrast, one may compute the entire orbit of a generating vector using modern computer algebra [BRTC23] and use this to determine if two generating vectors are topologically equivalent, but this process can become slow and memory intensive dependent on the size of $\operatorname { A u t } ( G ) \times \mathbb { B } _ { r }$ . Indeed, in general the orbit problem is at least as hard as the graph isomorphism problem [CEJS98].

Example 1.6. For a finite abelian group G it is known that $| \mathrm { A u t } ( G ) | \geq \phi ( | G | )$ ， where ϕ is Euler’s totient function. Moreover when G is abelian the action $o f$ the braid generator $\Phi _ { i }$ on a generating vector $( \gamma _ { 1 } , \dots , \gamma _ { r } )$ is simply to swap $\gamma _ { i }$ and $\gamma _ { i + 1 }$ , and so the braid action is just that of the symmetric group $S _ { r }$ . As such when considering the topological equivalence classes for generating vectors of length r for an abelian group $G ,$ orbits are generated by the action of a group of at least order $\phi ( | G | ) r !$

Faced with these dificulties we sought to investigate what alternative computational tools could provide insight into the problem of determining whether two generating vectors correspond to topologically equivalent actions. Motivated by recent successes in the application of machine learning to understanding group actions on Riemann surfaces [BDH25] as well as in using deep learning to understand structure of the symmetric group [PGdG26] we decided to use a deep learning approach. In particular, we trained a twin model originally designed to decide if two sequences of text share the same ‘sentiment’ $\mathrm { [ Y Z G ^ { + } 1 9 ] }$ : in our context the generating vectors gave rise to sequences of tokens for which sharing sentiment corresponded to being in the same equivalence class. In doing so we add to a growing corpus of literature exploring the applicability of machine/deep learning techniques to pure mathematics, see [DL26] for a review. The field of Riemann surfaces and particularly those with automorphisms is well placed to benefit from the development of this technology thanks to the wealth of precomputed data available from the LMFDB [LMF26, Pau23].

The final best-performing model successfully compared generating vectors in the test dataset with an accuracy of 85.8%. This paper provides further evidence of how modern computational developments can be leveraged to gain insight into mathematical problems. We also hope it will serve as an impetus to further explore the interplay between such fields. §2 provides specific details of the model architecture employed, the training data used, how results were analysed, and other key technical information, while §3 provides the results of investigations. Finally, §4 discusses the interpretation of these results, as well as directions of future inquiry.

## 2. Methods

We approach the problem as one of supervised binary classification: given a pair of generating vectors sharing a signature, predict whether they lie in the same topological equivalence class. This approach, in contrast to one which assigns a generating vector to an entry in a list of known equivalence classes, was used because a fixed-class classifier cannot by construction generalise to equivalence classes absent from its training data.

Remark 2.1. An early proof-of-concept design as described above, one which assigned a generating vector to a fixed list of equivalence classes, performed well within its class list but was still ultimately abandoned for generalisability reasons.

All models were implemented in PyTorch [PGM<sup>+</sup>19], adapting the existing opensource implementation of the RE2 text-matching architecture $[ \mathrm { Y } \mathrm { Z } \mathrm { G } ^ { + } 1 9 ] ;$ ; our final model architecture is visualised in Figure 1. Our modifications are confined to the input side of the network (tokenisation and embedding), so nothing hand-engineered about the group theory enters the model except through the tokens themselves; this afords us the benefit of being able to compare the model accuracy on our task of learning equivalence classes against its accuracy on the task of sentiment analysis for which it was designed. While we shall not repeat the description of the architecture provided in $\mathrm { [ Y Z G ^ { + } 1 9 ] }$ , we will provide comments where appropriate.

Training was undertaken on the Aire HPC system at the University of Leeds, each run occupying a single NVIDIA L40S 48GB (PCIe) GPU. Models were trained with the Adam optimiser under a cross-entropy loss, with learning rate $2 \times 1 0 ^ { - 3 }$ decaying exponentially (×0.94 per 25,600 training samples, floored at $1 0 ^ { - 4 } )$ after a 20,480-sample warm-up, batch size 8 with 4-step gradient accumulation (an efective batch of 32), and gradient clipping at 3. Model performance was assessed using a standard train-validation-test split; throughout training, the model was evaluated periodically on the validation data and the highest-scoring checkpoint was retained, with the subsequent accuracy on the test data reported. A run terminated when its evaluation performance failed to improve for 256,000 consecutive samples, or at a 24-hour wall limit.

![](images/0df63fd1ee789382818dcab53881a7a5b9ada76dd8d1472d9eda4cdeda5f3248.jpg)  
Fig. 1 The twin network architecture. The model has two identical halves, each half processing one generating vector. A tokenised element of a generating vector is passed through three identical processing blocks (with distinct weights) in series: the output of a convolutional encoder is concatenated with the block input, an alignment layer relates the encodings to those of the corresponding generating vector in the opposite half, and the outputs are fused and pooled over blocks. Each element of the generating vector is processed in parallel through this multiblock structure (aligning with its corresponding element in the opposite generating vector). These vectors are augmented by adding residuals constructed by passing them through a 1d convolution and GeLU activation, and the maximum over elements gives a feature vector for each generating vector as input for a prediction layer, answering the question whether two generating vectors lie in the same topological equivalence class.

As training data we computed all generating vectors for all group actions on Riemann surfaces of genus $2 \leq g \leq 4 8$ with quotient genus $g _ { 0 } = 0$ , and found their topological equivalence classes. Data of all possible pairs of groups and signatures which act on curves of genus $2 \leq g \leq 4 8$ were provided by Jen Paulhus [Pau15].<sup>1</sup> From this data, code adapted from [BRTC23] was run in SageMath [Sag26] to find all generating vectors and partition them into their equivalence classes. This resulted in 2,996 distinct groups, 27,122 signatures for these groups, 176,728 topological equivalence classes, and 960,201,074 generating vectors. From this we generated non-overlapping train, validation and test datasets of pairs of generating vectors with the same signature via random sampling (uniformly over signatures, with probability 0.5 that a pair were in the the same equivalence class), the train dataset consisting of $N = 3 0 0 , 0 0 0$ pairs, the validation dataset consisting of 10, 000 pairs. The test dataset is a class-uniform census: for each equivalence class we sample four same-class pairs and up to four diferent-class pairs sharing its signature, giving 86, 760 pairs. In these datasets signatures of length $r \leq 9 8$ were observed (the maximum corresponding to $C _ { 2 }$ acting as the hyperelliptic involution on a curve of genus $g = 4 8 )$ , and we also performed an evaluation limiting to signatures of length $r \leq 4$ and $r \leq 1 6$ to see how performance varied. Results are given in Table 1.

Remark 2.2. In optimising the code for computing all the generating vectors and their topological equivalence classes, the main improvement came from precomputing many properties for each finite group G. In particular, we generated an ordered list of elements, and thus turned group multiplication, inverse, the order of a group element, and the action of generators of Aut(G) into integer arrays. This means that a single call of the computer algebra system Sage (and implicitly GAP) was required for each group, and then computing every generating vector and its orbit for each signature associate to that group could be done entirely using Python arrays.

Each element of the generating vector is then tokenised following the approach of [PGdG26]: an element of the symmetric group $S _ { n }$ is written as a product of at most n − 1 transpositions by decomposing l-cycles as

$$
( a _ { 1 } , \ldots , a _ { l } ) = ( a _ { 1 } , a _ { 2 } ) ( a _ { 2 } , a _ { 3 } ) \ldots ( a _ { l - 1 } , a _ { l } ) ,
$$

transpositions in $S _ { n }$ are given by integers via the map

$$
s _ { i j } : = ( i , j ) \mapsto ( i - 1 ) + n ( j - 1 ) \in \{ 0 , \ldots , n ^ { 2 } - 1 \} ,
$$

and so the tokenisation of the element $s _ { i _ { 1 } j _ { 1 } } \cdot \cdot \cdot s _ { i _ { l } j _ { l } } \in S _ { n }$ is a sequence of tokens $t _ { 1 } , \ldots , t _ { l }$ , with each $t _ { k } \in \{ 0 , \ldots , n ^ { 2 } - 1 \}$ . In generating our dataset every group was given by a faithful permutation representation of degree at most $n = 5 0 0$ , so tokenising via the symmetric group is well defined. In practice we also add a token for the order of the element, and then append pad tokens to fix the sequence length for each group element to be the same.

Remark 2.3. Abstract groups were given a faithful permutation representation via the GAP commands IsomorphismPermGroup and SmallerDegreePermutation

Representation. The upper bound of $n = 5 0 0$ was not strict, a lower value could be found and so one could use fewer distinct tokens to tokenise the element of generating vectors.

Remark 2.4. Transpositions were chosen in [PGdG26] because they are known to generate the symmetric group, and the maximum word length for an element of the symmetric group as written in the alphabet of transpositions is known in advance. Not all the possible tokens in $\{ 0 , \ldots , n ^ { 2 } - 1 \}$ will appear, for example not i − 1 for $i = 1 , \ldots , n$ which corresponds to $s _ { i 1 }$

One can in principle tokenise elements of the symmetric group in an analogous way provided any generating set for which one knows the maximum possible word length (and can solve the word problem eficiently). As a step in experimentation we attempted using a single transposition and 3-cycles as the generating set, but this led to too many tokens and so was abandoned as a possible tokenising step. One may ask the question as to whether diferent generating sets can be chosen leading to a smaller number of tokens.

The tokenisation process produces sequences of varying token length for each element of a generating vector. Given generating vectors $( \gamma _ { 1 } ^ { ( j ) } , \dots , \gamma _ { r } ^ { ( j ) } ) \ ( j = 1 , 2 )$ being compared the model generates feature vectors for $\gamma _ { i } ^ { ( 1 ) }$ and $\gamma _ { i } ^ { ( 2 ) }$ using a copy of the embedding-encoder-fusion-pooling process implemented in the text-matching architecture of $\mathrm { [ Y Z G ^ { + } 1 9 ] }$

Remark 2.5. The model in [YZG<sup>+</sup>19] consists of a series of repeated blocks which process the input. As in the original paper we experimented with varying the number of blocks to be 1, 2, 3 and $^ { 4 , }$ seeing only marginal impact upon performance. As such, we kept the default value of 3 blocks.

From these feature vectors $v _ { i }$ for each element $\gamma _ { i } ,$ , a feature vector for the whole generating vector is found as follows: $( v _ { 1 } , \ldots , v _ { r } )$ is passed through a 1d convolution layer followed by GeLU activation to give a residual $c _ { i } .$ and then an elementwise max is applied to $v _ { i } + c _ { i }$ across each element of the generating vector. The prediction layer from $\mathrm { [ Y Z G ^ { + } 1 9 ] }$ in symmetric configuration is used to get an answer from the two feature vectors corresponding to the input pair.

Remark 2.6. The use of the 1d convolution before applying the maximum over element in a generating vector ensures that the model in not invariant under permuting the elements of the generating vector, as in this latter scenario model errors are guaranteed from configurations such as in Example 1.5. In development we experimented with positional embedding features but found that this performed worse.

Conversely, using the symmetric prediction layer means that the model prediction for whether two generating vectors are in the same equivalence class does not depend on the order they are provided to the model as one would desire.

We compared two methods of padding generating vectors so that a variable length r was accepted,

(1) padding the generating vector with copies of the identity element subsequently tokenised, efectively fixing the generating vector length to be some maximum $r ,$ and

(2) padding the generating vector with PAD tokens subsequently ignored using an attention mask in the softmax and pooling layer.

The results of this padding comparison are given in Table 1.

Tokens enter the network through a trainable embedding table, initialised in two diferent ways,

(1) analytically, from the $2 4 \times 2 4$ matrix representation induced from the permutation representation of $S _ { 2 4 } ~ \leq ~ S _ { 5 0 0 }$ : the token of a transposition is initialised to its flattened permutation matrix, giving embedding vectors of dimension 576, and

(2) randomly initialised with dimension 576, chosen to allow for comparison with the matrix representation, where any diference in performance between the two initialisations may be attributed structure rather than size.

The results of this comparison are given in Table 1.

Fixing the use of the padding mask as the method for handling variable r and initialising with the random embedding, we wished to explore how the model would perform generalising to generating vectors considerably diferent to the training data. To do this, we trained the model only on groups in the training data of order $| G | \le 6 4$ , validating and testing on groups of order $| G | > 6 4$ . The value of 64 was chosen to strike a balance between seeing suficiently many smaller groups while leaving a big enough variety of larger groups to test on; the number of distinct finite groups of order n peaks considerably at $n = 6 4$ , and approximately 80% of signatures in our dataset occur for $| G | \le 6 4$ , thus informing this choice. With this configuration, again the validation and test datasets contained 10, 000 and 86, 760 pairs respectively, but we varied the size of the training dataset N to investigate how this limited performance. The results of this are given in Table 2.

To better understand any errors in classification that occurred in the latter test stage (including only pairs generating groups of order $| G | > 6 4 )$ we labelled the pairs with data on the groups they generate from the LMFDB [LMF26], and then compared model performance (when trained on $N = 1 0 ^ { 6 }$ pairs generating groups of order $| G | \leq 6 4 )$ on subclasses of pairs by plotting Receiver Operating Characteristic (ROC) curves (Figures 2 and 3) with the corresponding Area-Under-Curve (AUC) values. In particular we focused on the following subclasses of generating vector pairs:

• cross-group, where the generating vectors generate diferent groups,

• same-group non-abelian, where the generating vectors generate the same non-abelian group,

• same-group cyclic, where the generating vectors generate the same cyclic group,

• same-group other abelian, where the generating vectors generate the same non-cyclic abelian group,

• same-group p-group, where the generating vectors generate the same pgroup.

Moreover, to investigate patterns in the feature vectors found by the model we performed dimensional reduction using t-distributed Stochastic Neighbour Embedding (t-SNE) plots of each pair’s feature vectors, in particular using the absolute elementwise diference and product (Figures 4 and 5), as these are the symmetric combinations of the features for each generating vector which enter the prediction layer.

<table><tr><td>Signature Length</td><td colspan="2">Accuracy</td></tr><tr><td></td><td>Random Embedding</td><td>Matrix Embedding</td></tr><tr><td> $r \leq 4 ^ { * }$ </td><td>0.702</td><td>0.681</td></tr><tr><td> $r \leq 4$ </td><td>0.698</td><td>0.693</td></tr><tr><td> $r \leq 1 6$ </td><td>0.850</td><td>0.828</td></tr><tr><td> $r \leq 9 8$ </td><td>0.858</td><td>0.825</td></tr></table>

Table 1. Model accuracy on 86, 760 test pairs when trained on 300, 000 pairs with |G| unrestricted and varying range for $r .$ ∗ indicates that identity padding was used.

<table><tr><td>Training Pairs</td><td>Accuracy</td></tr><tr><td> $\overline { { 1 \times 1 0 ^ { 5 } } }$ </td><td>0.743</td></tr><tr><td> $2 \times 1 0 ^ { 5 }$ </td><td>0.767</td></tr><tr><td> $3 \times 1 0 ^ { 5 }$ </td><td>0.789</td></tr><tr><td> $5 \times 1 0 ^ { 5 }$ </td><td>0.781</td></tr><tr><td> $\mathrm { 8 \times 1 0 ^ { 5 } }$ </td><td>0.792</td></tr><tr><td> $1 \times 1 0 ^ { 6 }$ </td><td>0.795</td></tr><tr><td> $1 . 3 \times 1 0 ^ { 6 }$ </td><td>0.795</td></tr><tr><td> $2 \times 1 0 ^ { 6 }$ </td><td>0.792</td></tr></table>

Table 2. Accuracy against training set size (train $| G | \leq 6 4$ , test $| G | > 6 4 )$

## 3. Results

3.1. Classification accuracy. Table 1 reports the test accuracy of each training configuration when not limiting |G| under the aforementioned train-validation-test split, organised by padding method, maximum r and embedding choice. The best performance reported was an accuracy of 85.8%. We see that the choice of embedding initialisation only made a meaningful diference (∼ 2%) when using identity padding or for larger r where the padding mask was used, but that uniformly the random embedding gave better performance. We were unable to test the accuracy of the model with identity padding for larger maximum values of r due to memory constraints. Comparing $r \leq 1 6$ and $r \leq 9 8 $ with the padding mask we saw that there was no significant diference in accuracy.

Table 2 shows the performance of the model restricted to training on a varying number of generating vectors from groups of order $| G | \le 6 4$ when evaluated on a test dataset consisting of pairs of generating vectors from groups with $| G | > 6 4$ For these experiments the embedding initialisation was fixed to be random, and the padding method used an attention mask. Model performance is seen to plateau after reaching approximately 0.795.

3.2. Error analysis. We analyse the errors of the random-embedding model on the test set. Of the 86, 760 pairs, the model misclassified 17, 786. The error rates on specific subclasses are given in Table 3. Of the subclasses considered the model performs best on the same-group p-group subclass, closely followed by the crossgroup subclass. The model performs similarly well on the same-group non-abelian class, and conversely performs poorly for all same-group abelian pairs both cyclic and otherwise.

<table><tr><td>Subclass</td><td>Accuracy</td><td>Number of Pairs</td></tr><tr><td>all pairs</td><td>0.795</td><td>86,760</td></tr><tr><td>cross-group</td><td>0.823</td><td>33,654</td></tr><tr><td>same-group non-abelian</td><td>0.813</td><td>44,861</td></tr><tr><td>same-group cyclic</td><td>0.594</td><td>7,184</td></tr><tr><td>same-group other abelian</td><td>0.557</td><td>1,061</td></tr><tr><td>same-group p-group</td><td>0.829</td><td>5,146</td></tr></table>

Table 3. Model accuracy on subclasses of generating vector pairs

Figures 2 varies the confidence threshold in the output of the model for classification of pairs as in the same equivalence class from 1 to 0, and compares the true positive rate against the false negative rate. Overall the model performs reasonably well as a classifier on all pairs with an AUC of 0.882; this is noticeably higher than the performance on most same-group subclasses considered, indicating that the false-positive rate is largely suppressed by cross-group pairs which are accurately identified as in diferent equivalence classes. The exception is the same-group pgroup subclass, where the high-confidence positive likelihood ratio exceeds that on all pairs.

Because there is no ‘true positive’ for pairs in the cross-group subclass when classifying whether pairs are in the same equivalence class, Figure 3 is a modification of the ROC curve varying the confidence threshold for classification of pairs from 1 to 0.5 (the default value) and comparing how many correct and incorrect decisions are retained; for this curve we may consider the cross-group subclass. One must be careful interpreting the AUC for these curves: even a value near 1 could correspond to a subclass of pairs on which the model performed very poorly if there were many incorrect decisions with confidence near 0.5. We nevertheless see that the model confidently distinguishes a large proportion of cross-group and same-group p-group pairs.

Figures 4 and 5 partially explain the model’s respective high and low performance on the cross-group and same-group other abelian subclasses: the former are well separated by the model in feature space, whereas the latter are not tightly clustered. This perspective further provides a plausible explanation for the marginally higher performance on the same-group p-group subclass than on the same-group nonabelian subclass which contains it, but is unable to explain why the performance on same-group cyclic pairs was poor, suggesting that the prediction layer is indeed identifying local relational properties between the two features.

## 4. Discussion

This paper illustrates how simple deep learning approaches developed in separate contexts may be applied to achieve high accuracy predictions of sophisticated group structure. In particular, a deep learning model developed for sentiment analysis of text has been adapted to learn topological equivalence of generating vectors with an accuracy of 85.8%. Notably this accuracy is comparable with that of the model when evaluated on a text-matching task given similarly size train data. The original RE2 architecture has accuracy 89.2 ± 0.2 on the task of identifying whether, given

![](images/fea2afb680cb1bfb71f9cfb42f05c3b7443bc51010d3cdd5c1c83fdc06696e37.jpg)  
Fig. 2 ROC curves for the model classifying pairs as in the same equivalence class. Subclasses are labelled by the AUC.

400, 000 training pairs of questions from Quora, one is a paraphrase of the other $\mathrm { [ Y Z G ^ { + } 1 9 ] } .$ This is the closest analogue to this paper’s task in the various tests of $[ \mathrm { Y } \mathrm { Z } \mathrm { G } ^ { + } 1 9 ] ;$ : a binary decision on whether two token sequences are equivalent. Our accuracies when training on 300, 000 pairs with $r \leq 9 8$ and |G| unrestricted (see Table 1) are directly comparable to the original architecture’s performance. The comparison is not exact: there are diferences in class balance, embedding dimension, vocabulary size and sequence structure. Nonetheless, there is a striking observation that the architecture learns topological equivalence of generating vectors comparably well to how it learns the semantic equivalence of English sentences, the task it was designed for. This suggests that the model has been able to identify some level of semantic structure in topological equivalence classes.

Unlike the application in $\mathrm { [ Y Z G ^ { + } 1 9 ] }$ which used the 300-dimensional GloVe word vectors, we predominantly did not use a tailored pre-trained embedding as this did not improve model performance. We hypothesise that this is because the matrix embedding we experimented with, implicitly providing the data of a 24-dimensional representation of $S _ { 5 0 0 }$ , is insuficiently helpful for the task of identifying topological equivalence classes. Indeed such a representation cannot be faithful as the minimal dimension for a faithful representation of $S _ { n }$ is $n - 1 \colon$ this would correspond in our case to embedding vectors of dimension $4 9 9 ^ { 2 } \ : = \ : 2 4 9 0 0 1$ which would have been impracticably large. Our results do not rule out the possibility that another tailored embedding would provide improvement, and future work should explore which tokenisations and embeddings of group elements are best tailored to learning finite group theory.

![](images/cc0f86a282bec84c60e2a0b41a82e57f9ab8ffc0d592770d3fdd3a43eee7ae4a.jpg)  
Fig. 3 ROC curves for the model classifying pairs as the confidence threshold is varied. Subclasses are labelled by the AUC.

We saw that, though the model performance did drop when trained only on groups of order $| G | < 6 4$ , it was still able to generalise reasonably well. This bolsters the conclusion that some underlying structure of topological equivalence was learnt, though one must caveat this with the fact that, though a given group is not present in both the train and test data, an element of that group may be present in a generating vector in both. Though accuracy improved with the size of training data available as is natural it eventually plateaued, indicating that there are still aspects of topological equivalence which were missed. It is possible, if apparently unlikely, that with a much larger training dataset performance would begin to increase again and cross a 99% accuracy threshold, so-called ‘grokking’ [PBE<sup>+</sup>22]. As observed in $[ \mathrm { P B E ^ { + } 2 2 } ]$ , this behaviour requires the amount of training data to be approximately at least a third of all possible data: a rough estimate suggests that from our dataset one could generate $O ( 1 0 ^ { 1 3 } )$ pairs of generating vectors with the same signature, many orders of magnitude more than were in the largest train dataset.

![](images/e43144ee0cae0cf9cf7c5f68c714ae3ebb7a66db2da6558172c3f2bd7a18b828.jpg)  
Fig. 4 Dimensional reduction of the absolute elementwise diference in the generating vector features of a pair, stratified by subclass

Data mining tools have demonstrated that a key factor underlying the models strength even when trained only on pairs generating groups of order |G| ≤ 64 is that the model is confidently able to distinguish those pairs which generate diferent groups, even though this information was not a priori provided to the model. Dimensional reduction indicates that the feature vectors of such pairs are well separated from other same-group pairs. What error analysis has highlighted is that the model performs well on the same-group p-group subclass, and that this is not immediately attributable to a separation of feature vectors. We still lack an explanation for why this behaviour occurs. The model’s relative underperformance on abelian groups (both cyclic and non-cyclic) would suggest that the model has not developed the ability to identify when the ordering of elements becomes immaterial, even though the feature vectors for the same-group cyclic subclass are well separated.

It is worthwhile commenting further on the decision to use a deep learning approach in contrast to a machine learning approach, here the distinction being that no feature engineering was undertaken to extract significant data from the input generating vectors beforehand on which to train, instead resorting to tokenisation. In this aspect, our paper is more closely aligned with the approach of [PGdG26] than with that of [BDH25]. In doing so we increase the flexibility of the model at the cost of losing interpretability, and we have been unable to extract any new theorems from this study as of yet. That being said, our error analysis indicates that the model best learnt the topological equivalence classes of p-groups in the dataset, which may indicate that this class of groups is a natural candidate for future mathematical study. Moreover, deep learning is typically more computationally intensive than machine learning, and runtime has prevented our ability to evaluate the performance of the model when trained on more than a tiny fraction of the total possible data.

![](images/9ffd0d0f0dcf8273c2e05085344cfce6fccae9347f648a9a3b6308646579d8ee.jpg)  
Fig. 5 Dimensional reduction of the elementwise product of the generating vector features of a pair, stratified by subclass

By design we did not attempt to force invariance of the model under the action of Aut(G) × B<sub>r</sub> on either generating vector input (appropriately understood), hoping that our model would learn this structure. Employing this, future work could investigate the application of a multiple-testing framework to determine if two generating vectors are in the same equivalence class, comparing multiple pairs of generating vectors from either class at a time.

## References

[BDH25] H. W. Braden and Linden Disney-Hogg, Orbits of theta characteristics, Experimental Mathematics (2025), 1–50.

[Bre00] T. Breuer, Characters and automorphism groups of compact Riemann surfaces, London Mathematical Society Lecture Note Series, vol. 280, Cambridge University Press, 2000.

[Bro90] S. Allen Broughton, The equisymmetric stratification of the moduli space and the Krull dimension of mapping class groups, Topology and its Applications 37 (1990), no. 2, 101–113.

[Bro22] S. Allen Broughton, Equivalence of finite group actions on Riemann surfaces and algebraic curves, Automorphisms of Riemann Surfaces, Subgroups of Mapping Class Groups and Related Topics (Aaron Wootton, S. Allen Broughton, and Jennifer Paulhus, eds.), Contemporary Mathematics, vol. 776, American Mathematical Society, 2022.

[BRTC23] Antonio Behn, Anita M. Rojas, and Miguel Tello-Carrera, A Sage package for n-gonal equisymmetric stratification of M<sub>g</sub>, Experimental Mathematics 32 (2023), no. 1, 54–69.

[CEJS98] E. M. Clarke, E. A. Emerson, S. Jha, and A. P. Sistla, Symmetry reductions in model checking, Computer Aided Verification (Alan J. Hu and Moshe Y. Vardi, eds.), Springer Berlin Heidelberg, 1998, pp. 147–158.

[DL26] Michael R. Douglas and Kyu-Hwan Lee, Mathematical data science, Advances in Theoretical and Mathematical Physics 29 (2026), no. 8, 2305–2325.

[Hur92] A. Hurwitz, Ueber algebraische gebilde mit eindeutigen transformationen in sich, Mathematische Annalen 41 (1892), no. 3, 403–442.

[Kur87] Izumi Kuribayashi, On automorphism groups of a curve as linear groups, J. Math. Soc. Japan 39 (1987), no. 1, 51–77.

[LMF26] The LMFDB Collaboration, The L-functions and modular forms database, https:// www.lmfdb.org, 2026, [Online].

[LR22] Herbert Lange and Rub´ı E. Rodr´ıguez, Decomposition of Jacobians by Prym varieties, Lecture Notes in Mathematics, vol. 2310, Cham: Springer, 2022.

[Pau15] Jennifer Paulhus, Branching data for curves up to genus 48, arXiv:1512.07657, 2015.

[Pau23] Jennifer Paulhus, A database of group actions on Riemann surfaces, Birational Geometry, K¨ahler-Einstein Metrics and Degenerations (Ivan Cheltsov, Xiuxiong Chen, Ludmil Katzarkov, and Jihun Park, eds.), Springer International Publishing, 2023, pp. 693–708.

[PBE<sup>+</sup>22] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra, Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022.

[PGdG26] Max Petschack, Alexandr Garbali, and Jan de Gier, Learning the symmetric group: large from small, Advances in Theoretical and Mathematical Physics 30 (2026), no. 1, 65–81.

[PGM<sup>+</sup>19] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala, PyTorch: an imperative style, high-performance deep learning library, pp. 8026–8037, Curran Associates Inc., 2019.

[Sag26] The Sage Developers, Sagemath, the Sage Mathematics Software System (Version 10.8), https://www.sagemath.org, 2026.

[YZG<sup>+</sup>19] Runqi Yang, Jianhai Zhang, Xing Gao, Feng Ji, and Haiqing Chen, Simple and effective text matching with richer alignment features, Proceedings of the 57th annual meeting of the association for computational linguistics, Association for Computational Linguistics, 2019, Implementation available from https://github.com/alibaba-edu/ simple-effective-text-matching-pytorch, pp. 4699–4709.

Department of Mathematics<sub>,</sub> Imperial College London<sub>,</sub> London SW7 2AZ<sub>,</sub> U.K. Email address: a.disneyhogg@imperial.ac.uk