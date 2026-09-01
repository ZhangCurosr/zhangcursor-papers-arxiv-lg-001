# Rotational Equivariance in Machine Learning: A Comprehensive Tutorial

Peter Lippmann<sup>∗</sup> and Fred A. Hamprecht

Interdisciplinary Center for Scientific Computing (IWR),

Heidelberg University, 69120 Heidelberg, Germany

(Dated: September 1, 2026)

Rotational symmetry is one of the most important structural principles in machine learning on 3D data. In applications ranging from physics and materials science to 3D computer vision, predictions should not depend on an arbitrary choice of coordinate frame. Rotational equivariance captures this requirement mathematically by enforcing that a rotation of the input induces a corresponding transformation of the model output. This tutorial provides a comprehensive introduction to rotational equivariance, starting from the physical and geometric intuition behind coordinate independence and building up the necessary machinery from geometric deep learning, group theory, and representation theory. We introduce message passing on Euclidean graphs, group actions and representations, spherical harmonics, Wigner matrices, tensor products, and Clebsch-Gordan decomposition, and explain how these ingredients give rise to modern equivariant architectures. We then survey the principal strategies for incorporating rotational equivariance in deep learning, including group convolutions, internal tensorial representations, and canonicalization-based methods, and discuss their practical strengths and limitations. The tutorial aims to lower the barrier to the subject by connecting the underlying mathematics to practical model design, by unifying idea that are often expressed in diferent formal languages, and by helping practitioners choose among competing approaches through a clear discussion of their trade-ofs.

## CONTENTS

I. Introduction 1   
II. Symmetries and Coordinate Independence 2   
III. Geometric Deep Learning and Message   
Passing 3   
IV. Groups and Spatial Symmetries 4   
V. Group Representations and Equivariance 4   
VI. Spherical Harmonics and the Irreducible   
Representations of SO(3) 6   
A. Computing Wigner-D matrices 8   
VII. Decomposition of Cartesian Tensors and the   
Tensor Product 9   
VIII. Rotational Equivariance in Deep Learning 11   
A. Equivariance via Group Convolutions 11   
B. Equivariance via Internal Tensorial   
Representations 13   
C. Canonicalization-Based Approaches 13   
D. Conceptual Links between Diferent   
Approaches 15   
IX. Problems with Rotational Equivariance in   
Practice 17   
X. Conclusion 17

## I. INTRODUCTION

Symmetries play an essential role in the physical sciences. For instance, they restrict the form of admissible laws and dynamics: a symmetry rules out any equation of motion that would change under the corresponding transformation. Symmetries serve as guiding principles for formulating theories, e.g. through symmetry-invariant Lagrangians in particle physics. More broadly, the laws of physics are formulated covariantly, meaning that predictions do not depend on arbitrary choices of coordinates or reference frames.

Modeling this coordinate independence in machine learning (ML) has become a central theme in geometric deep learning. Consider a molecule represented by atom positions in R<sup>3</sup>. The global coordinate system is arbitrary, so scalar observables such as the total energy must be invariant under global rotations and reflections, while vector- and tensor-valued quantities (e.g. forces, dipole moments, stresses) must transform consistently between coordinate descriptions. However, a naive encoding of geometry by raw coordinates makes rotated or reflected configurations appear entirely diferent to a standard neural network. As a result, the model must effectively relearn the same physical patterns across many orientations.

A fundamental idea in geometric deep learning is therefore to incorporate symmetry as prior knowledge directly into the architecture by enforcing invariance and equivariance [1–4]. These symmetry-based inductive biases improve generalization and data eficiency: the model need not spend model capacity on discovering symmetries from finite data and guarantees exact sym metries even out of distribution, that is, for regions of the input space not explored during training. Accordingly, rotationally (and often reflection-) equivariant models form the backbone of many state-of-the-art methods across disciplines [5–9].

The motivation for equivariant ML models is not only conceptual but practical as well. For many complex systems we lack closed-form equations, or solving the known underlying equations accurately is prohibitively expensive (e.g. for quantum mechanics, weather, cosmology or high-energy physics). In such settings, data-driven surrogate models are essential for prediction and for comparing measurements to simulations [10– 12]. When data-acquisition is costly, enforcing known symmetries can be decisive: equivariant networks often reach higher accuracy with fewer samples [13, 14] and provide physically consistent predictions by construction, particularly important for the stability of long simulations [15, 16]. More broadly, scientific machine learning aims to combine the strengths of learning-based models with established physical principles, rather than treating neural networks as purely data-driven black boxes. In this sense, equivariant neural networks can be viewed as practical, data-driven surrogates that preserve core physical structure.

This tutorial is organized as follows. We begin by motivating symmetry as the mathematical expression of coordinate independence and introduce geometric deep learning on Euclidean graphs through the language of message passing. We then develop the group-theoretic framework needed to formalize equivariance, including group representations, which are key to describing how diferent physical quantities transform under symmetry operations. Building on this foundation, we introduce Cartesian tensors, spherical harmonics, Wigner-D matrices, tensor products, and the Clebsch-Gordan decomposition. We then turn to architectural design and discuss the principal strategies for achieving rotational equivariance in deep learning, including group convolutions, internal tensorial representations, and canonicalization-based methods. Finally, we examine the practical limitations of these approaches and discuss the trade-ofs between exact built-in equivariance and learned approximate equivariance achieved through data augmentation.

This tutorial is complementary to several excellent existing resources. The guide and overview by Duval et al. [4] ofers a broad and pedagogical overview of geometric deep learning for atomistic systems, while the article by Kondor [17] presents a more abstract and representation-theoretic perspective on equivariant networks. The survey by Han et al. [18] provides a wider overview of equivariant GNNs across geometric domains. The value of the present tutorial lies in its tight focus: rather than surveying geometric deep learning broadly or treating equivariance in full generality, it concentrates specifically on rotational equivariance in 3D and develops the subject from the ground up. Its aim is to give readers a clear path from coordinate independence and symmetry principles to the concrete mathematical tools, such as SO(3)-representations, spherical harmonics, Wigner matrices, and tensor products, that underlie modern equivariant architectures, while also helping practitioners navigate the diferent formalisms and design choices used in the literature.

## II. SYMMETRIES AND COORDINATE INDEPENDENCE

Intuitively, a symmetry of a physical system is a transformation that leaves the relevant properties of the system unchanged. In machine learning, the same idea ap pears in various forms. As illustrated above, the central reason why spatial symmetries matter in scientific machine learning is the arbitrariness of coordinate systems. We need coordinates to represent geometric objects numerically, yet the choice of reference frame is not physical. Let us illustrate this point with a simple example: for two directions $\mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \in \mathbb { R } ^ { 3 }$ , the angle can be computed from coordinates via

$$
\cos \angle ( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } ) = { \frac { \mathbf { v } _ { 1 } ^ { \mathrm { T } } \mathbf { v } _ { 2 } } { \| \mathbf { v } _ { 1 } \| _ { 2 } \| \mathbf { v } _ { 2 } \| _ { 2 } } } , \qquad \| \mathbf { v } _ { i } \| _ { 2 } = { \sqrt { \mathbf { v } _ { i } ^ { \mathrm { T } } \mathbf { v } _ { i } } } ,\tag{1}
$$

but the resulting value is independent of how the vectors are expressed in coordinates.

In some cases, an appropriate coordinate choice can substantially simplify a computation. For example, the volume of an axis-aligned box with side lengths $a , b ,$ c can be written as

$$
V = \int _ { \mathbb { R } ^ { 3 } } \mathbb { I } _ { \mathrm { B o x } } ( x , y , z ) \mathrm { d } x \mathrm { d } y \mathrm { d } z = a b c ,\tag{2}
$$

where $\mathbb { I } _ { \mathrm { B o x } }$ is the indicator function of the box. In a coordinate system aligned with the box edges, the integral factorizes immediately; in a rotated frame, the same integral becomes more cumbersome even though the result must agree. Many fundamental physical theories, such as Maxwell’s formulation of electromagnetism, exploits this redundancy in the coordinate description through suitable gauge choices that simplify equations without changing physical predictions.

The requirement that results obtained in diferent coordinate systems must be consistent is naturally expressed as equivariance with respect to Euclidean symmetries. Intuitively, a function or algorithm is equivariant if transforming its input induces a predictable, corresponding transformation of its output. In that way, equivariant models produce predictions that remain consistent under coordinate changes (or, equivalently, under global transformations of the input geometry).

Crucially, physical laws, formulated as coordinateindependent (covariant) equations, can be seen as equivariant algorithms.<sup>1</sup> For instance, Newton’s second law maps forces to acceleration,

$$
\mathbf { a } ( \mathbf { F } _ { 1 } , \ldots , \mathbf { F } _ { n } , m ) = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { n } \mathbf { F } _ { i } ,\tag{3}
$$

and the predicted acceleration transforms as expected when the input is rotated by any rotation matrix R:

$$
\begin{array} { r l r } & { } & { \mathbf { a } ( R \mathbf { F } _ { 1 } , \ldots , R \mathbf { F } _ { n } , m ) = \frac { 1 } { m } \sum _ { i } R \mathbf { F } _ { i } } \\ & { } & \\ & { } & { \qquad = R \left( \frac { 1 } { m } \sum _ { i } \mathbf { F } _ { i } \right) = R \mathbf { a } ( \mathbf { F } _ { 1 } , \ldots , \mathbf { F } _ { n } , m ) . } \end{array}
$$

In contrast, standard neural networks operating on raw coordinate arrays are generally not equivariant. As a toy example, suppose we concatenate two forces and a mass into an input vector $x = ( \mathbf { F } _ { 1 } \Vert \mathbf { F } _ { 2 } \Vert m ) \in \mathbb { R } ^ { 7 }$ with $\mathbf { F } _ { i } \in \mathbb { R } ^ { 3 }$ and scalar $m \in \mathbb { R }$ , and predict the acceleration $\hat { \mathbf { a } } \in \mathbb { R } ^ { 3 }$ using a single dense layer with weight $W \in \mathbb { R } ^ { 3 \times 7 }$ and bias b $\in \mathbb { R } ^ { 3 }$ , followed by a pointwise nonlinearity such as ReLU,

$$
\hat { \mathbf { a } } = \operatorname { R e L U } ( \underbrace { W x + \mathbf { b } } _ { = \mathbf { z } } )
$$

with $\mathrm { R e L U } ( \mathbf { z } ) ] _ { i } = \operatorname* { m a x } ( 0 , z _ { i } )$ for $i = { 1 , 2 , 3 }$ . An unconstrained matrix W mixes scalar and vector components arbitrarily, as in $w _ { 1 } F _ { 1 , 1 } + w _ { 2 } F _ { 1 , 2 } + w _ { 3 } F _ { 1 , 3 } + w _ { 4 } F _ { 2 , 1 }$ + · · · + w<sub>7</sub>m, which does not transform like a vector under rotations. Therefore, our predicted acceleration would not be consistent across diferent choices of reference frames. Furthermore, component-wise nonlinearities typically break equivariance. A simple example can be given in $\mathbb { R } ^ { \overset { \cdot } { 2 } }$ . Let $P = - I _ { 2 }$ denote the $1 \bar { 8 } 0 ^ { \circ }$ rotation in 2D. Then,

$$
\begin{array} { r } { \mathrm { R e L U } \big ( P ( 1 , 1 ) ^ { \mathrm { T } } \big ) = \mathrm { R e L U } ( ( - 1 , - 1 ) ^ { \mathrm { T } } ) = ( 0 , 0 ) ^ { \mathrm { T } } } \\ { \neq ( - 1 , - 1 ) ^ { \mathrm { T } } = P \mathrm { R e L U } ( ( 1 , 1 ) ^ { \mathrm { T } } ) , } \end{array}
$$

showing that ReLU does not map vectors to vectors in an equivariant way. These basic examples illustrate that equivariance is usually not obtained “for free” but often requires careful design of equivariant architectures.

Key point. Coordinate frames are needed for numerical computation, but results obtained in different coordinate systems must be consistent. This consistency is formalized via the concept of equivariance; physical laws can be viewed as equivariant algorithms.

## III. GEOMETRIC DEEP LEARNING AND MESSAGE PASSING

Data from various domains, such as scans of 3D scenes, molecular structures, detections in particle collisions, or earth science measurements, can be viewed as a set of nodes positioned in Euclidean space and equipped with geometric node features. Concretely, a point cloud consists of nodes $\{ i \}$ with features $\{ f _ { i } \}$ , each located at a position $x _ { i } \in \mathbb { R } ^ { d }$ . A Euclidean graph is a point cloud paired with edges $( i , j )$ , for instance obtained from a point cloud by constructing a k-NN graph or a radius graph.

A widely used deep learning approach for Euclidean graph is the class of message passing neural networks $\mathrm { \bar { ( M P N N s ) } }$ , which iteratively combine and abstract information by sending messages along edges. In a typical layer, below indexed by $k ,$ every node i receives messages $m _ { i j }$ from each node $j$ in its neighborhood $\mathcal { N } ( i )$ and aggregates them in a permutation-invariant way to update its node feature:

$$
f _ { i } ^ { ( k + 1 ) } = \bigoplus _ { j \in { \cal N } ( i ) } m _ { i j } = \bigoplus _ { j \in { \cal N } ( i ) } h \Bigl ( f _ { i } ^ { ( k ) } , f _ { j } ^ { ( k ) } , e _ { i j } , x _ { i } - x _ { j } \Bigr ) .\tag{4}
$$

Here, the message is constructed using a learnable message function (often an MLP) h. Besides the node feature vector of the sending node $f _ { j } ^ { ( k ) }$ , the message function may receive the feature of the receiving node $f _ { i } ^ { ( k ) }$ as input, as well as the relative position vector $x _ { i } - x _ { j }$ and optional edge features $e _ { i j }$ . The L is a permutation-invariant aggregation operator such as sum or mean (or sometimes the component-wise max [19]). Using relative positions $x _ { i } - x _ { j }$ ensures translation invariance, since a global shift of all coordinates leaves all relative displacements unchanged. Since the node update given by Eq. (4) is permutation invariant with respect to the ordering of neighbors, the overall layer applied to each node i is permutation equivariant, i.e. permuting input nodes permutes the output node features accordingly.

Stacking multiple message passing layers in a neural network allows information to propagate over longer distances in the graph and allows to combine node features into higher-level, more abstract representations. In the context of molecular data, for example, nodes usually correspond to atoms (located in 3D space) and input features can encode atom types; message passing layers then combines local geometric information (which atoms are nearby and how are they positioned) into increasingly ab stract chemical representations used to predict the target property. For global molecular properties, one typically applies a permutation-invariant readout (e.g. sum, mean or max pooling) over node features at the end of the network to produce a single prediction per molecule.

MPNNs form a very general family of models. As a special case of the above, MPNNs can also be applied to fully connected graphs, where each node is connected to all others, i.e. $\Breve { \mathcal { N } } ( i ) = \left\{ 1 , \ldots , N \right\}$ (including self-loops).

From this perspective, the self-attention mechanism at the core of the Transformer architecture [20] can also be viewed as an instance of message passing with a learned, “content-dependent” aggregation. Given node features $f _ { i } \in \mathbb { R } ^ { d }$ , we obtain queries, keys, and values from a linear embedding

$$
\begin{array} { r } { q _ { i } = W ^ { Q } f _ { i } , \qquad k _ { j } = W ^ { K } f _ { j } , \qquad v _ { j } = W ^ { V } f _ { j } , } \end{array}\tag{5}
$$

and attention weights

$$
\alpha _ { i j } = \mathrm { s o f t m a x } _ { j } \big ( q _ { i } ^ { \mathrm { T } } k _ { j } / \sqrt { d _ { k } } \big ) : = \frac { \exp \big ( q _ { i } ^ { \mathrm { T } } k _ { j } / \sqrt { d _ { k } } \big ) } { \sum _ { \ell = 1 } ^ { N } \exp \big ( q _ { i } ^ { \mathrm { T } } k _ { \ell } / \sqrt { d _ { k } } \big ) } ,\tag{6}
$$

where $W ^ { Q } , W ^ { K } , W ^ { V }$ are learnable weight matrices and $d _ { k }$ is the dimension of the key and query vectors. The updated representation is then obtained by a normalized, weighted sum aggregation of the value vectors $v _ { j }$

$$
f _ { i } ^ { ( k + 1 ) } = \sum _ { j = 1 } ^ { N } \alpha _ { i j } v _ { j } ,\tag{7}
$$

In a Transformer block, this self-attention operation is typically followed by an MLP, a residual connection, and a normalization layer. Thus, Transformers can be viewed as MPNNs on fully connected graphs with attentionbased message weights, illustrating the versatility of the message passing framework.

## IV. GROUPS AND SPATIAL SYMMETRIES

A symmetry of a physical system is a transformation that leaves certain properties of the system unchanged. For example, in the absence of external fields, globally rotating a molecule changes its orientation but not its energy. In mathematical terms, the set of symmetry transformations is typically described as a group, i.e. a collection of transformations that can be composed and inverted.

Formally, a group is defined as a set G equipped with a binary operation $\cdot : G \times G \to G$ that describes the composition of transformations and satisfies the following four axioms:

• Closure: For all $a , b \in G$ , the composition $a \cdot b$ is again an element of G.

• Associativity: For all $a , b , c \in G , ( a \cdot b ) \cdot c = a \cdot ( b \cdot c )$

• Identity: There exists an element $e \in G$ such that $e \cdot a = a \cdot e = a$ for all $a \in G$

• Inverse: For every $a \in G$ there exists $a ^ { - 1 } \in G$ such that $a \cdot a ^ { - 1 } = a ^ { - 1 } \cdot a = e$

In this tutorial we focus mostly on the Euclidean symmetries: translations, rotations, and reflections. The orthogonal group $\mathrm { O } ( d )$ combines rotations and reflections, while the special orthogonal subgroup $\mathrm { S O } ( d ) \subset \mathrm { O } ( d )$ includes only (orientation-preserving) rotations:

$$
{ \mathrm { O } } ( d ) : = \{ R \in \mathbb { R } ^ { d \times d } \mid R ^ { \mathrm { T } } R = I \} ,\tag{8}
$$

$$
\operatorname { S O } ( d ) : = \{ R \in O ( d ) ~ | \operatorname * { d e t } ( R ) = 1 \} .\tag{9}
$$

The condition $R ^ { \mathrm { { T } } } R = I$ is equivalent to preserving inner products, i.e. $( R u ) ^ { \mathrm { T } } ( R v ) = u ^ { \mathrm { T } } v$ for all $u , v \in \mathbb { R } ^ { d }$ , and $\operatorname* { d e t } ( R ) = - 1$ corresponds to orientation-reversing transformations (reflections). Translations $t \in \mathbb { R } ^ { d }$ (paired with addition) form a non-compact group, acting on points by $x \mapsto x + t$ Combining rotations with translations yields the special Euclidean group of rigid motions $\operatorname { S E } ( d )$ Concretely, we may represent a rigid motion by a pair $( R , t )$ with $R \in \mathrm { S O } ( d )$ and $t \in \mathbb { R } ^ { d }$ , acting on points in d-dimensional space via

$$
x \mapsto R x + t .\tag{10}
$$

The set of all such pairs becomes a group under composition, with group product

$$
( R _ { 1 } , t _ { 1 } ) \cdot ( R _ { 2 } , t _ { 2 } ) \ : = \ ( R _ { 1 } R _ { 2 } , \ t _ { 1 } + R _ { 1 } t _ { 2 } ) ,\tag{11}
$$

identity $( I , 0 )$ , and inverse $( R , t ) ^ { - 1 } = ( R ^ { \mathrm { T } } , - R ^ { \mathrm { T } } t )$ . The Euclidean group $\operatorname { E } ( d )$ also includes reflections, i.e. $R \in$ $\mathrm { O } ( d )$

## V. GROUP REPRESENTATIONS AND EQUIVARIANCE

As we have seen before, physical equations are formulated in a covariant (coordinate-independent) way, that is, they retain the same form under a change of coordinates. Interpreting a physical law as a map or algorithm from inputs to outputs, this covariance is precisely the requirement of equivariance. Namely, the left- and right-hand side of an equation must transform consistently under the same coordinate transformation. Group representations provide the mathematical framework to formalize how diferent geometric objects behave under (symmetry) transformations.

As a basic example let us consider a velocity vector $\mathbf { v } \in \mathbb { R } ^ { 3 }$ . It is a directed geometric quantity which under a rotation $R \in \mathrm { S O ( 3 ) }$ transforms as a vector,

$$
\mathbf { v } \ \mapsto \ \mathbf { v ^ { \prime } } = R \mathbf { v } , \quad { \mathrm { i . e . } } \quad v _ { i } ^ { \prime } = R _ { i j } v _ { j } .\tag{12}
$$

Throughout this tutorial, we consistently use Einstein summation convention, i.e. the summation over indices that are repeated twice is implied. While the components of the velocity vector change, its length remains invariant:

$$
\| \mathbf { v } ^ { \prime } \| _ { 2 } ^ { 2 } = ( R \mathbf { v } ) ^ { \mathrm { T } } ( R \mathbf { v } ) = \mathbf { v } ^ { \mathrm { T } } R ^ { \mathrm { T } } R \mathbf { v } = \mathbf { v } ^ { \mathrm { T } } \mathbf { v } = \| \mathbf { v } \| _ { 2 } ^ { 2 } ,\tag{13}
$$

using $R ^ { \mathrm { { T } } } R = I$ for orthogonal matrices. Beyond vectors, some geometric quantities transform as higher-order tensors. For instance, the inertia tensor $\mathcal { T } \in \breve { \mathbb { R } } ^ { 3 \times 3 }$ of a finite

set of masses $\{ m _ { k } \}$ located at positions $\{ \mathbf { x } _ { k } \in \mathbb { R } ^ { 3 } \}$ is defined as

$$
\mathcal { T } _ { i j } : = \sum _ { k } m _ { k } \big ( \| \mathbf { x } _ { k } \| _ { 2 } ^ { 2 } \delta _ { i j } - x _ { k , i } x _ { k , j } \big ) ,\tag{14}
$$

and transforms under $R \in \mathrm { S O ( 3 ) }$ as

$$
\begin{array} { r } { \mathcal { T } \mapsto \mathcal { T } ^ { \prime } = R \mathcal { T } R ^ { \mathrm { T } } , \quad \quad \mathrm { i . e . } \quad \quad \mathcal { T } _ { i j } ^ { \prime } = R _ { i k } \mathcal { T } _ { k l } R _ { j l } , } \end{array}\tag{15}
$$

which follows directly from the transformation of positions $\mathbf { x } _ { k } \mapsto R \mathbf { x } _ { k }$

Key point. These examples illustrate that diferent geometric objects live in diferent vector spaces and transform diferently under the same symmetry transformation: scalars are invariant, vectors transform linearly, and higher-order Cartesian tensors transform by contracting each index with the rotation.

a. Group representation. Given a group $G$ and a vector space $V ,$ , a group representation $\rho$ is a homomorphism from $G$ to the invertible matrices $\operatorname { G L } ( V )$ , i.e. a mapping that fulfills

$$
\rho ( g _ { 1 } g _ { 2 } ) = \rho ( g _ { 1 } ) \rho ( g _ { 2 } ) , \quad \forall g _ { 1 } , g _ { 2 } \in G ,\tag{16}
$$

where $g _ { 1 } g _ { 2 }$ is the group product and $\rho ( g _ { 1 } ) \rho ( g _ { 2 } )$ the standard matrix product. The representation specifies how elements of the group act on vectors $v \in V$ , i.e. in components $( \rho ( g ) v ) _ { i } = \rho ( g ) _ { i j } v _ { j }$ . The dimension of the representation $\rho$ is defined to be the dimension of $V$ . Condition (16) implies that $\rho ( g ^ { - 1 } ) = ( \rho ( g ) ) ^ { - 1 }$ , see e.g. [21] for details.

b. Cartesian tensor representations. Starting from the defining action of $R \in \operatorname { O } ( d ) \operatorname { o n } v \in \mathbb { R } ^ { d } , ( R v ) _ { i } = R _ { i j } v _ { j } { \mathrm { . } }$ one can obtain higher-order Cartesian tensor representations by acting on each index. For an order-n tensor $T _ { i _ { 1 } \dots i _ { n } }$ , we have

$$
T _ { i _ { 1 } \ldots i _ { n } } ^ { \prime } ~ = ~ R _ { i _ { 1 } j _ { 1 } } \cdot \cdot \cdot R _ { i _ { n } j _ { n } } T _ { j _ { 1 } \ldots j _ { n } } .\tag{17}
$$

For instance, the outer product of two vectors $T _ { i j } \ =$ $( v w ^ { \mathrm { T } } ) _ { i j } ~ = ~ v _ { i } w _ { j }$ transforms as $T _ { i j } ^ { \prime } \ = \ ( R v ) _ { i } ( R w ) _ { j } ^ { } \ =$ $R _ { i k } R _ { j l } T _ { k l }$ . One can easily verify that Eq. (17) indeed satisfies the representation property Eq. (16). To avoid cumbersome indexing, let us illustrate the proof just for $T _ { i j }$ . Given $R _ { 1 } , R _ { 2 } \in \mathrm { O } ( d )$ , in components we have

$$
\begin{array} { r l } & { [ \rho ( R _ { 1 } R _ { 2 } ) T ] _ { i j } = ( R _ { 1 } R _ { 2 } ) _ { i k } ( R _ { 1 } R _ { 2 } ) _ { j l } T _ { k l } } \\ & { \qquad = R _ { 1 , i m } R _ { 1 , j n } \underbrace { R _ { 2 , m k } R _ { 2 , n l } T _ { k l } } _ { = [ \rho ( R _ { 2 } ) T ] _ { m n } } = [ \rho ( R _ { 1 } ) \rho ( R _ { 2 } ) T ] _ { i j } . } \end{array}
$$

where $\rho ( R )$ denotes the Cartesian tensor representation of R acting on $T \in \mathbb { R } ^ { d \times d }$ . Consequently, order-n (or $ { \mathrm { \hat { \rho } r a n k } }  { - } n  { \mathrm { \ ' } } )$ Cartesian tensors form a $d ^ { n }$ -dimensional representation space of $\mathrm { O } ( d )$

![](images/11b243875d6f01724a4b45336a683b397d290d9c24380d14fc27d65cd4e77a69.jpg)  
Figure 1. Equivariant prediction of a molecular dipole moment. Applying a global transformation to the molecular geometry before predicting the dipole moment yields the same result as first predicting the dipole moment and then applying the same transformation to the predicted vector.

c. Equivariance. Using the notion of group representations, we can now formalize the concept of equivariance. Let $G$ be a group and $V , W$ two vector spaces equipped with group representations $\rho _ { \mathrm { i n } }$ and $\rho _ { \mathrm { o u t } }$ respectively. A function ${ \overset { \cdot } { \varphi } } : { \overset { \cdot } { V } } \to W$ is said to be equivariant under the group G if the following holds:

$$
\rho _ { \mathrm { o u t } } ( g ) \varphi ( x ) = \varphi ( \rho _ { \mathrm { i n } } ( g ) x ) , \quad \forall g \in G , \forall x \in V
$$

where the input to $\varphi$ transforms under the representation $\rho _ { \mathrm { i n } } : V \to { \bar { \mathrm { G L } } } ( V )$ and its output under the representation $\rho _ { \mathrm { o u t } } : W \to { \mathrm { G L } } ( W )$ . If $\rho _ { \mathrm { o u t } } ( g ) = I$ (identity) for all $g \in G .$ , the function $\varphi$ is said to be invariant. Intuitively, equivariance means that $\varphi$ commutes with the group action, i.e. transforming the input and then applying $\varphi$ is equivalent to first applying $\varphi$ and then transforming the output, as illustrated in Fig. 1.

As a direct consequence of the definition, an equivariant function $\varphi$ is symmetry preserving, i.e. if the input x has a certain symmetry, e.g. invariance under a subgroup $H \subset G$ , then the output $\varphi ( x )$ will have the same symmetry, since for all $h \in { \bar { H } }$ we have

$$
\varphi ( x ) = \varphi ( \rho _ { \mathrm { i n } } ( h ) x ) = \rho _ { \mathrm { o u t } } ( h ) \varphi ( x ) .\tag{18}
$$

Therefore, $\varphi ( x )$ is also invariant under $H .$

d. Example: the cross product is O(3)-equivariant. A well-known yet non-trivial example of an $\mathrm { O } ( 3 ) \mathrm { - }$ equivariant map is the cross product (or vector product) $\mathbf { u } \times \mathbf { v }$ . It is an insightful exercise to show that indeed,

$$
( R \mathbf { u } ) \times ( R \mathbf { v } ) \ = \ R ( \mathbf { u } \times \mathbf { v } )\tag{19}
$$

for any $R \in \mathrm { S O ( 3 ) }$ and $ { \mathbf { u } } ,  { \mathbf { v } } \in \mathbb { R } ^ { 3 }$ . We start by writing the cross product in components using the totally antisymmetric Levi-Civita symbol $\epsilon _ { i j k }$

$$
( { \bf u } \times { \bf v } ) _ { i } \ = \ \epsilon _ { i j k } u _ { j } v _ { k } ,\tag{20}
$$

with $\epsilon _ { i j k } = 1 \mathrm { i f } ( i , j , k )$ is an even permutation of $( 1 , 2 , 3 )$ $\epsilon _ { i j k } ~ = ~ - 1$ if it is an odd permutation, and $\epsilon _ { i j k } \ = \ 0$ otherwise. Now, inserting $u _ { j }  R _ { j l } u _ { l }$ and $v _ { k } \to R _ { k m } v _ { m }$ we obtain

$$
\big ( ( R \mathbf { u } ) \times ( R \mathbf { v } ) \big ) _ { i } = \epsilon _ { i j k } ( R \mathbf { u } ) _ { j } ( R \mathbf { v } ) _ { k } = \epsilon _ { i j k } R _ { j l } R _ { k m } u _ { l } v _ { m } .\tag{21}
$$

To eliminate one of the matrices R on the right-hand side, we use the following general identity that connects the Levi-Civita symbol to the matrix determinant [22]:

$$
\epsilon _ { i j k } R _ { j l } R _ { k m } R _ { i n } = \operatorname * { d e t } ( R ) \epsilon _ { l m n } .\tag{22}
$$

Contracting both sides with $R _ { n o } ^ { \mathrm { T } }$ and using $R ^ { \mathrm { T } } R = I$ (i.e. $R _ { i n } { R _ { n o } ^ { \mathrm { T } } } = \delta _ { i o } )$ yields

$$
\epsilon _ { o j k } R _ { j l } R _ { k m } = \operatorname* { d e t } ( R ) \epsilon _ { l m n } R _ { o n } .\tag{23}
$$

Renaming the free index $o  i ,$ inserting Eq. (23) into Eq. (21) and using that $\operatorname* { d e t } ( R ) = 1$ for rotation matrices, we can confirm the equivariance by

$$
\begin{array} { r } { \left( ( R \mathbf { u } ) \times ( R \mathbf { v } ) \right) _ { i } = \operatorname* { d e t } ( R ) \epsilon _ { l m n } R _ { i n } u _ { l } v _ { m } = R _ { i n } \epsilon _ { n l m } u _ { l } v _ { m } } \\ { = \left( R \left( \mathbf { u } \times \mathbf { v } \right) \right) _ { i } \qquad ( 2 4 ) } \end{array}
$$

For reflections (orientation-reversing transformations) with $\operatorname* { d e t } ( R ) = - 1$ , the cross product picks up a minus sign in the last step of the derivation. For example, for parity $P = - I \in \mathrm { O } ( 3 ) \setminus \mathrm { S O } ( 3 )$

$$
( P \mathbf { u } ) \times ( P \mathbf { v } ) = \operatorname* { d e t } ( P ) P ( \mathbf { u } \times \mathbf { v } ) = - P ( \mathbf { u } \times \mathbf { v } ) .\tag{25}
$$

Thus $\mathbf { u } \times \mathbf { v }$ is an axial vector (or pseudovector). Under a general orthogonal transformation $R \in \mathrm { O } ( 3 )$ it transforms as

$$
v _ { i } ^ { \prime } = \operatorname* { d e t } ( R ) R _ { i j } v _ { j } .\tag{26}
$$

e. Cartesian pseudotensor representation. As we have seen from the example of pseudovectors, the fact that the orthogonal matrices also include reflections can be used to distinguish the transformation behavior of geometric objects with respect to orientation-reversing transformations (with determinant −1). The following transformation behavior defines another representation, namely the one of pseudotensors:

$$
P _ { i _ { 1 } . . . i _ { n } } ^ { \prime } = \operatorname* { d e t } ( R ) R _ { i _ { 1 } , j _ { 1 } } \ \dots \ R _ { i _ { n } , j _ { n } } P _ { j _ { 1 } \dots j _ { n } } .\tag{27}
$$

The proof that the pseudotensor representation forms a group representation (i.e. satisfies Eq. (16)) works exactly as the one for the tensor representation above, additionally using the fact that the determinant is multiplicative: det $( A B ) = \operatorname* { d e t } ( A ) \operatorname* { d e t } ( B )$

Key point. Group representations specify how symmetry transformations act on diferent geometric objects living in separate vector spaces. They form the basis for transforming features between diferent (local) reference frames in our tensorial message passing approach.

## VI. SPHERICAL HARMONICS AND THE IRREDUCIBLE REPRESENTATIONS OF SO(3)

In this section, we demonstrate how the Cartesian tensor representations of Eq. (17) can be decomposed into irreducible subrepresentations of SO(3), realized by the so-called Wigner-D matrices. We further illustrate that spherical harmonics, which form a complete orthonormal basis for (square-integrable) functions on the sphere $S ^ { 2 }$ , transform precisely under these irreducible representation of SO(3). Based on that, we introduce the notion of spherical tensors.

a. Irreducible subrepresentations. Formally, a subspace $W ~ \subset ~ V$ is called G-invariant, with respect to a representation $\rho : G \to \mathrm { G L } ( V )$ , if $\rho ( g ) w \in W$ for all $g \in G$ and $w \in W$ . In this case, the restriction $\rho | _ { W } : G \to { \mathrm { G L } } ( W ) , g \mapsto \rho ( g )$ defines a subrepresentation. A representation is called irreducible if it has no non-trivial G-invariant subspaces, i.e. if the only invariant subspaces are {0} and V itself [21].

b. Example: rank-2 Cartesian tensors are reducible. Consider the rank-2 Cartesian tensor representation of SO(3) acting on $T \in \mathbb { R } ^ { 3 \times 3 }$ by

$$
T \ \mapsto \ T ^ { \prime } = R T R ^ { \mathrm { T } } , \qquad \mathrm { i . e . } \qquad T _ { i j } ^ { \prime } = R _ { i k } R _ { j l } T _ { k l } .\tag{28}
$$

This representation is reducible because it contains the one-dimensional invariant subspace spanned by the identity matrix span{I}. Indeed, for $T = a I \in \operatorname { s p a n } \{ I \}$ we have $T ^ { \prime } = a R I R ^ { \mathrm { \tilde { T } } } = a I \in \mathrm { s p a n } \{ I \}$ . This component is often called the trace part because every rank-2 tensor admits the canonical decomposition

$$
T = \underbrace { \left( T - { \frac { \operatorname { t r } ( T ) } { 3 } } I \right) } _ { \mathrm { t r a c e l e s s ~ p a r t } } + \underbrace { \frac { \operatorname { t r } ( T ) } { 3 } I } _ { \mathrm { ( i s o t r o p i c ) ~ t r a c e ~ p a r t } } ,\tag{29}
$$

and the coeficient of the isotropic basis tensor I is fully determined by tr(T). We will see how to further decompose a rank-2 Cartesian tensor in Sec. VII below.

c. Spherical Harmonics. Spherical harmonics are scalar functions on the sphere $\begin{array} { r } { \dot { S ^ { 2 } } = \{ \hat { \mathbf { r } } \in \mathbb { R } ^ { 3 } : \| \hat { \mathbf { r } } \| _ { 2 } = 1 \} } \end{array}$ ， indexed by an integer degree $l \in \{ 0 , 1 , \overset { \cdot } { 2 } , \dots \}$ and an order $m \in \{ - l , \ldots , l \}$ . One may work either with the complex spherical harmonics $Y _ { l m } ^ { \mathbb { C } } : S ^ { 2 } \to \mathbb { C }$ or with an equivalent real spherical harmonics basis $Y _ { l m } ^ { \mathbb { R } } : S ^ { 2 } \to \mathbb { R }$ . In either case, the set of functions $\{ Y _ { l m } \} _ { l , m }$ forms a complete orthonormal basis of $L ^ { 2 } ( S ^ { 2 } )$ , that is, any square-integrable real-valued function ${ \dot { f } } : { \dot { S } } ^ { 2 } \to \mathbb { R }$ can be expanded into spherical harmonics as

$$
f ( \hat { \mathbf { r } } ) = \sum _ { l = 0 } ^ { \infty } \sum _ { m = - l } ^ { l } c _ { l m } Y _ { l m } ^ { \mathbb { C } } ( \hat { \mathbf { r } } ) ,\tag{30}
$$

with coeficients $c _ { l m } \in \mathbb { C }$ . Importantly, for f to be realvalued, the coeficients in the complex basis are not in-

dependent but satisfy the following constraint<sup>2</sup>

$$
c _ { l , - m } = ( - 1 ) ^ { m } \overline { { { c _ { l m } } } } \qquad ( m = 1 , \ldots , l ) ,\tag{31}
$$

where $\overline { { c _ { l m } } }$ denotes the complex conjugate.

For given l, we may change between the complex spherical harmonics $Y ^ { \mathbb { C } }$ and the real ones $Y ^ { \mathbb { R } }$ using the following conversion [24]:

$$
Y _ { l m } ^ { \mathbb { R } } ( \hat { \mathbf { r } } ) : = \sqrt { 2 } ( - 1 ) ^ { | m | } \operatorname { I m } \big ( Y _ { l | m | } ^ { \mathbb { C } } ( \hat { \mathbf { r } } ) \big ) , \qquad m < 0 ,\tag{32}
$$

$$
Y _ { l 0 } ^ { \mathbb { R } } ( \hat { \mathbf { r } } ) : = Y _ { l 0 } ^ { \mathbb { C } } ( \hat { \mathbf { r } } ) , \qquad m = 0 ,\tag{33}
$$

$$
Y _ { l m } ^ { \mathbb { R } } ( \hat { \mathbf { r } } ) : = \sqrt { 2 } ( - 1 ) ^ { m } \operatorname { R e } \bigl ( Y _ { l m } ^ { \mathbb { C } } ( \hat { \mathbf { r } } ) \bigr ) , \qquad m > 0 .\tag{34}
$$

The general form of the complex spherical harmonics of degree l is given by

$$
Y _ { l m } ^ { \mathbb { C } } ( \theta , \phi ) \propto P _ { l } ^ { m } ( \cos \theta ) \mathrm { e } ^ { \mathrm { i } m \phi } ,\tag{35}
$$

where $P _ { l } ^ { m }$ are the associated Legendre polynomials. In spherical coordinates $( \theta , \phi ) , \phi \in [ 0 , 2 \pi )$ denotes the azimuthal angle in the xy-plane from the positive x-axis and $\theta \in [ 0 , \pi ]$ is the polar angle from the positive z-axis. Since, in the complex basis, the azimuthal dependence of $Y _ { l m } ^ { \mathbb { C } }$ is given by $\bar { \mathrm { e } } ^ { \mathrm { i } m \phi }$ , in the real basis, $Y _ { l m } ^ { \mathbb { R } }$ correspond to cos(mϕ)- and sin(mϕ)-type angular dependences, respectively.

Up to normalization, the lowest-degree complex harmonics are given by

$$
\begin{array} { r } { Y _ { 0 0 } ^ { \mathbb { C } } ( \theta , \phi ) \propto 1 , \qquad Y _ { 1 0 } ^ { \mathbb { C } } ( \theta , \phi ) \propto \cos \theta , } \\ { Y _ { 1 , \pm 1 } ^ { \mathbb { C } } ( \theta , \phi ) \propto \sin \theta \mathrm { e } ^ { \pm \mathrm { i } \phi } , \qquad } \end{array}
$$

and the corresponding real harmonics (for l = 1) are

$$
\begin{array} { r } { Y _ { 1 , - 1 } ^ { \mathbb { R } } ( \theta , \phi ) \propto \sin \theta \sin \phi , \qquad Y _ { 1 0 } ^ { \mathbb { R } } ( \theta , \phi ) \propto \cos \theta , } \end{array}\tag{36}
$$

$$
Y _ { 1 1 } ^ { \mathbb { R } } ( \theta , \phi ) \propto \sin \theta \cos \phi .\tag{37}
$$

In machine learning one typically uses the real basis so that all features and activations remain real-valued. Figure 2 shows a visualization of the lowest-degree real spherical harmonics as functions on the sphere.

For us, the key property is that, under any rotation $R \in \mathrm { S O ( 3 ) }$ , spherical harmonics of a given degree l mix only among themselves. Concretely, stacking all orders into a vector-valued map

$$
Y ^ { ( l ) } : S ^ { 2 } \to \mathbb { R } ^ { 2 l + 1 } , \qquad Y ^ { ( l ) } ( \hat { \mathbf { r } } ) : = \left( Y _ { l , - l } ^ { \mathbb { R } } ( \hat { \mathbf { r } } ) , \dots , Y _ { l , l } ^ { \mathbb { R } } ( \hat { \mathbf { r } } ) \right) ^ { \mathrm { T } }\tag{38}
$$

(here in the real basis) one obtains a linear transformation law<sup>3</sup>

$$
Y ^ { ( l ) } ( R \hat { \mathbf { r } } ) = D ^ { ( l ) } ( R ) Y ^ { ( l ) } ( \hat { \mathbf { r } } ) , \qquad R \in S O ( 3 ) ,\tag{39}
$$

where $D ^ { ( l ) } ( R ) \in \mathbb { R } ^ { ( 2 l + 1 ) \times ( 2 l + 1 ) }$ is the so-called Wigner-D matrix in that basis. We will discuss in Sec. VI A how to compute $D ^ { ( l ) } ( R )$ eficiently.

Let us explicitly study the transformation behavior of the spherical harmonics of the lowest degrees. For $l = 0$ we have $Y _ { 0 0 } \propto 1$ , hence ${ \cal Y } ^ { ( 0 ) } ( R \hat { \mathbf { r } } ) = { \cal Y } ^ { ( 0 ) } ( \hat { \mathbf { r } } )$ corresponding to the trivial (invariant) representation. For $l = 1$ we recognize that the real harmonics are (up to permutations) equal standard spherical coordinates with radius $r = 1$ . For that compare, Eq. (36) with

$$
x = r \sin \theta \cos \phi , \qquad y = r \sin \theta \sin \phi , \qquad z = r \cos \theta .\tag{40}
$$

Consequently, one can choose a suitable real basis such that

$$
Y ^ { ( 1 ) } ( { \hat { \mathbf { r } } } ) \propto { \hat { \mathbf { r } } } .\tag{41}
$$

In this basis the Wigner-D matrices for $l = 1$ coincide with the standard rotation matrices,

$$
D ^ { ( 1 ) } ( R ) = R .\tag{42}
$$

This convention (used e.g. in the equivariant ML library e3nn [13]) emphasizes that degree-1 harmonics transform like ordinary vectors.

d. Irreducible representations and spherical tensors. A standard classification result in representation theory tells us that the finite-dimensional irreducible representations (irreps) of SO(3) are unique, can be indexed by $l \in \{ 0 , 1 , 2 , \ldots \}$ and have dimension $2 l + 1 \ \lceil 2 5 \rceil$ Since for each l the functions $\{ Y _ { l m } \} _ { m = - l } ^ { l }$ span a $( 2 l + 1 )$ dimensional subspace that is invariant under rotations, the induced actions of $R \in \mathrm { S O } ( 3 )$ on these subspaces are (up to a change of basis) exactly the unique irreducible representation of of $\dot { R } \in \mathrm { S O } ( 3 )$ This motivates the definition of a spherical tensor of degree l as a vector $x \in \mathbb { C } ^ { 2 l + 1 }$ (or in a real basis $x \in \mathbb { R } ^ { 2 l + 1 } )$ that transforms according to

$$
( D ^ { ( l ) } ( R ) x ^ { ( l ) } ) _ { m } = \sum _ { m ^ { \prime } = - l } ^ { l } D _ { m m ^ { \prime } } ^ { ( l ) } ( R ) x _ { m ^ { \prime } } ^ { ( l ) } .\tag{43}
$$

Further, we can distinguish between even and odd parity spherical tensors, which transform under the same SO(3) Wigner-D matrices but difer by a sign under reflections: We use that any orientation-reversing transformation $R \in \mathrm { O } ( 3 ) \backslash \mathrm { S O } ( 3 )$ can be written as $R = P R ^ { \prime }$ with $P = - I$ and $R ^ { \prime } : = P R \in \mathrm { S O } ( 3 )$ to diferentiate between an even-parity spherical tensor $x _ { + } ^ { ( l ) }$ and an odd-parity spherical tensor $x _ { - } ^ { ( l ) }$ , which transform as

$$
x _ { + } ^ { ( l ) } \mapsto D ^ { ( l ) } ( R ^ { \prime } ) x _ { + } ^ { ( l ) } , \qquad x _ { - } ^ { ( l ) } \mapsto - D ^ { ( l ) } ( R ^ { \prime } ) x _ { - } ^ { ( l ) } .\tag{44}
$$

for $R = P R ^ { \prime } \in \mathrm { O } ( 3 ) \setminus \mathrm { S O } ( 3 )$

![](images/2e6d708b014b9c19ca04d6c2d732c24119de57c268ae3720278e84e9c44fee9b.jpg)  
Figure 2. Visualization of the lowest-degree real spherical harmonics $Y _ { l m } ^ { \mathbb { R } }$ as functions on the sphere. For $m = 0$ the harmonics are axially symmetric around the z-axis, while for m $\neq 0$ they have a characteristic cos(mϕ)- or sin(mϕ)-type angular dependence in the xy-plane. The degree l determines the number of nodal lines (circles where the function changes sign).

Key point. Cartesian tensors, Eq. (17), transform under a reducible representation, and can therefore be decomposed into spherical tensors, Eq. (43), which transform via Wigner-D matrices, i.e. under the irreducible representations of SO(3). In Sec. VII, we see explicitly how to perform this decomposition.

## A. Computing Wigner-D matrices

A fast and practical method for computing $D ^ { ( l ) } ( R )$ for a given $R \in \mathrm { S O ( 3 ) }$ is described by Pinchon and Hoggan [26]. Their construction works entirely in the real basis and evaluates $D ^ { ( l ) } ( R )$ in $\mathcal { O } ( l ^ { 3 } )$ time using precomputed, degree-dependent building blocks. Since the Wigner-D matrices are tied to the transformation behavior of the spherical harmonics by Eq. (39), we construct the representation matrices by studying how the spherical harmonics transform under rotations. Following [26], we order the real spherical harmonics for fixed l as

$$
\bigl ( Y _ { l 0 } ^ { \mathbb { R } } , Y _ { l 1 } ^ { \mathbb { R } } , Y _ { l , - 1 } ^ { \mathbb { R } } , Y _ { l 2 } ^ { \mathbb { R } } , Y _ { l , - 2 } ^ { \mathbb { R } } , \ldots , Y _ { l l } ^ { \mathbb { R } } , Y _ { l , - l } ^ { \mathbb { R } } \bigr ) ^ { \mathrm { T } } .\tag{45}
$$

As we have seen before, in this convention, $Y _ { l m } ^ { \mathbb { R } }$ and $Y _ { l , - m } ^ { \mathbb { R } }$ (for $m > 0 )$ carry the azimuthal dependence cos(mϕ) and sin(mϕ), respectively. A rotation about the z-axis by angle α acts as $\phi \mapsto \phi + \alpha$ and therefore mixes each pair $( \cos ( m \phi ) , \sin ( m \phi ) )$ by a planar rotation of angle mα. Consequently, the representation matrix for $R _ { z } ( \alpha )$

is block-diagonal:

$$
\begin{array} { r l r } & { } & { X ^ { ( l ) } ( \alpha ) = \mathrm { d i a g } \Big ( 1 , R ( \alpha ) , R ( 2 \alpha ) , \ldots , R ( l \alpha ) \Big ) , } \\ & { } & { \mathrm { w h e r e \quad } R ( \vartheta ) = \Big ( \displaystyle \cos \vartheta - \sin \vartheta \Big ) , } \\ & { } & { \mathrm { w h e r e \quad } R ( \vartheta ) = \Big ( \displaystyle \sin \vartheta \quad \cos \vartheta \Big ) , } \end{array}\tag{46}
$$

which corresponds to Eq. (12) in [26]. To leverage that rotations around the z-axis take this particularly simple form, we decompose any rotation $R \in \mathrm { S O ( 3 ) }$ via a z-y-z Euler decomposition

$$
R = R _ { z } ( \alpha ) R _ { y } ( \beta ) R _ { z } ( \gamma ) ,\tag{47}
$$

where $R _ { z } ( \cdot )$ and $R _ { y } ( \cdot )$ are rotations around the z- and yaxes, respectively, and $\alpha , \beta ,$ γ are the corresponding Euler angles. The Wigner-D matrix for R can then be computed from the homomorphism property of representations (cf. Eq. (16)):

$$
D ^ { ( l ) } ( R ) = X ^ { ( l ) } ( \alpha ) D ^ { ( l ) } ( R _ { y } ( \beta ) ) X ^ { ( l ) } ( \gamma ) .\tag{48}
$$

The key idea by Pinchon and Hoggan [26] is to reduce the remaining, more complicated y-rotation to a z-rotation by conjugation with the (orthogonal) coordinate swap

$$
J : ~ ( x , y , z ) \mapsto ( x , z , y ) , \qquad J \in \mathrm { O } ( 3 ) \backslash \mathrm { S O } ( 3 ) , \quad J ^ { 2 } = I .\tag{49}
$$

Based on the transformation rules of Eq. (44) for orientation-reversing orthogonal transformations, J in duces a well-defined linear operator $J ^ { ( l ) } \in \mathbb { R } ^ { \left( 2 l + 1 \right) ^ { \prime } \times \left( 2 l + 1 \right) }$

on space of degree-l spherical tensors. The induced operator similarly satisfies

$$
( J ^ { ( l ) } ) ^ { 2 } = I \qquad \Rightarrow \qquad ( J ^ { ( l ) } ) ^ { - 1 } = J ^ { ( l ) } .\tag{50}
$$

Using once more the homomorphism property of the Wigner-D matrices, we can write the more complicated y-rotation as

$$
D ^ { ( l ) } ( R _ { y } ( \beta ) ) = J ^ { ( l ) } X ^ { ( l ) } ( \beta ) J ^ { ( l ) } .\tag{51}
$$

The matrices $J ^ { ( l ) }$ can be precomputed once per degree l. In [26] they are constructed recursively by propagating the known action of J at l = 1 (i.e. on $( x , y , z ) )$ to higher degrees. Combining Eq. (47) and (51) yields the eficient Pinchon-Hoggan factorization

$$
D ^ { ( l ) } ( R ) = X ^ { ( l ) } ( \alpha ) J ^ { ( l ) } X ^ { ( l ) } ( \beta ) J ^ { ( l ) } X ^ { ( l ) } ( \gamma ) ,\tag{52}
$$

where each $X ^ { ( l ) } ( \cdot )$ is cheap to evaluate (independent $2 \times 2$ blocks $R ( m \cdot ) )$ , and $J ^ { ( l ) }$ is fixed for a given l and can be reused across all rotations. This yields an accurate and practical $\mathcal { O } ( l ^ { 3 } )$ scheme for computing Wigner-D matrices in the real basis.

## VII. DECOMPOSITION OF CARTESIAN TENSORS AND THE TENSOR PRODUCT

We have established that Cartesian tensor representations (cf. Eq. (17)) are in general reducible, while the irreducible SO(3) representations are realized by the Wigner-D matrices $\overset { \prime } { D } ^ { ( l ) } \dot { ( } R ) \in \mathbb { R } ^ { ( 2 l + 1 ) \times ( 2 l + 1 ) }$ . In this section, we connect both representation types and introduce the tensor product and its irreducible (Clebsch-Gordan) decomposition. The latter is one of the key building blocks in state-of-the-art SO(3)-equivariant neural networks [5, 27, 28].

a. Tensor products of representations. Given two tensors $A _ { i _ { 1 } \dots i _ { n } }$ and $B _ { j _ { 1 } \ldots j _ { m } } ,$ their tensor (outer) product is defined component-wise by

$$
( A \otimes B ) _ { i _ { 1 } . . . i _ { n } j _ { 1 } . . . j _ { m } } : = A _ { i _ { 1 } . . . i _ { n } } B _ { j _ { 1 } . . . j _ { m } } .\tag{53}
$$

If A and B are Cartesian tensors and transform under rotations as in Eq. (17), then $A \otimes B$ transforms as

$$
\begin{array} { r l r } {  { \bigl ( A \otimes B \bigr ) _ { i _ { 1 } \ldots i _ { n } j _ { 1 } \ldots j _ { m } } ^ { \prime } } } \\ & { = R _ { i _ { 1 } k _ { 1 } } \cdot \cdot \cdot R _ { i _ { n } k _ { n } } R _ { j _ { 1 } \ell _ { 1 } } \cdot \cdot \cdot R _ { j _ { m } \ell _ { m } } \bigl ( A \otimes B \bigr ) _ { k _ { 1 } \ldots k _ { n } \ell _ { 1 } \ldots \ell _ { m } } , } \end{array}\tag{54}
$$

so the tensor product again forms a tensor representation. The same statement holds if $x ^ { ( l _ { 1 } ) }$ and $\dot { y } ^ { ( l _ { 2 } ) }$ are spherical tensors, then $x ^ { ( l _ { 1 } ) } \otimes y ^ { ( l _ { 2 } ) }$ transforms under the product representation $D ^ { ( l _ { 1 } ) } \otimes D ^ { ( l _ { 2 } ) }$ While the tensor (outer) product yields an equivariant output tensor by construction, stacking outer products in a neural network would quickly lead to a combinatorial blow-up in the feature dimensionality. This motivates the decomposing into irreps, that is, a set of smaller, more manageable building blocks with well-defined transformation behavior.

b. Decomposition of Cartesian tensors. Cartesian tensors of rank 0 (scalars) and rank 1 (vectors) are already irreducible under SO(3). The first non-trivial case is a rank-2 Cartesian tensor. To keep track of the dimensions in a tensor decomposition, we introduce the shorthand d to denote an irrep of dimension $d ,$ that is, $2 l + 1$ refers to the unique (2l + 1)-dimensional irrep labeled by l (cf. Sec. VI). Since a rank-2 Cartesian tensor has the same transformation behavior, $\begin{array} { r } { T _ { i j } ^ { \prime } = R _ { i k } R _ { j \ell } T _ { k \ell } , } \end{array}$ as the tensor (outer) product of two vectors, we associate $T _ { i j }$ with 3 ⊗ 3. First, we define the symmetric and antisymmetric parts of $T _ { i j }$ by

$$
\begin{array} { r } { S _ { i j } = \frac 1 2 ( T _ { i j } + T _ { j i } ) , \qquad A _ { i j } = \frac 1 2 ( T _ { i j } - T _ { j i } ) . } \end{array}\tag{55}
$$

Both span invariant subspaces (subrepresentations), since for example

$$
\begin{array} { r } { S _ { i j } ^ { \prime } = \frac 1 2 ( T _ { i j } ^ { \prime } + T _ { j i } ^ { \prime } ) = R _ { i k } R _ { j \ell } \frac 1 2 ( T _ { k \ell } + T _ { \ell k } ) = R _ { i k } R _ { j \ell } S _ { k \ell } , } \end{array}\tag{56}
$$

and similarly for $A _ { i j } .$ . The antisymmetric subspace is 3- dimensional and is isomorphic to a pseudovector by use of the Levi-Civita symbol:

$$
a _ { k } : = { \textstyle \frac 1 2 } \epsilon _ { k i j } A _ { i j } , \qquad A _ { i j } = \epsilon _ { i j k } a _ { k } .\tag{57}
$$

In fact, it is easy to confirm<sup>4</sup> that in this form a transforms as $\mathbf { a } ^ { \prime } = R \mathbf { : }$ a for $R \in \mathrm { S O ( 3 ) }$ acting on $A _ { i j }$ and that a<sup>′</sup> picks up an additional sign under reflections, cf. Eq. (26). Thus, the antisymmetric part realizes the same 3 irrep as an ordinary vector, but with diferent parity under $\mathrm { O ( 3 ) }$

The symmetric part further decomposes into trace and traceless parts. As we have seen before (cf. Eq. (29)), the trace part forms a closed subspace equipped with the trivial (invariant) representation (l = 0). The remaining symmetric traceless part

$$
\begin{array} { r } { \tilde { S } _ { i j } : = S _ { i j } - \delta _ { i j } \frac { \mathrm { T r } ( S ) } { 3 } } \end{array}\tag{58}
$$

is 5-dimensional and can be shown to correspond to the l = 2 irrep. Altogether, in dimension notation,

$$
\underline { { { 3 } } } \otimes \underline { { { 3 } } } ~ = ~ \underline { { { 1 } } } \oplus \underline { { { 3 } } } \oplus \underline { { { 5 } } } .\tag{59}
$$

c. Clebsch-Gordan decomposition and the coupled tensor product. More generally, the tensor product of two SO(3) irreps decomposes as

$$
\underline { { 2 l _ { 1 } + 1 } } \otimes \underline { { 2 l _ { 2 } + 1 } } = \bigoplus _ { L = | l _ { 1 } - l _ { 2 } | } ^ { l _ { 1 } + l _ { 2 } } \underline { { 2 L + 1 } } .\tag{60}
$$

Using Gauss’s summation formula, one may easily verify that the dimensions match by showing that

$$
\sum _ { L = | l _ { 1 } - l _ { 2 } | } ^ { l _ { 1 } + l _ { 2 } } ( 2 L + 1 ) = ( 2 l _ { 1 } + 1 ) ( 2 l _ { 2 } + 1 ) .\tag{61}
$$

As an example, we can use Eq. (60) to examine how a rank-3 Cartesian tensor decomposes into irreps. A rank-3 Cartesian tensor corresponds to $\underline { { { 3 } } } \otimes \underline { { { 3 } } } \otimes \underline { { { 3 } } } .$ . Using Eq. (59) iteratively yields

$$
{ \begin{array} { r l } { { \underline { { 3 } } } \otimes { \underline { { 3 } } } \otimes { \underline { { 3 } } } } & { = \ ( { \underline { { 1 } } } \oplus { \underline { { 3 } } } \oplus { \underline { { 5 } } } ) \otimes { \underline { { 3 } } } } \\ & { = \ { \underline { { 1 } } } \otimes { \underline { { 3 } } } \oplus { \underline { { 3 } } } \otimes { \underline { { 3 } } } \oplus { \underline { { 5 } } } \otimes { \underline { { 3 } } } } \\ & { = \ { \underline { { 3 } } } \oplus ( 1 \oplus { \underline { { 3 } } } \oplus { \underline { { 5 } } } ) \oplus ( { \underline { { 3 } } } \oplus { \underline { { 5 } } } \oplus { \underline { { 7 } } } ) } \\ & { = \ { \underline { { 1 } } } \oplus { \underline { { 3 } } } \oplus { \underline { { 3 } } } \oplus { \underline { { 5 } } } \oplus { \underline { { 5 } } } \oplus { \underline { { 7 } } } } \end{array} }\tag{62}
$$

i.e. one scalar, three vector-type components, two $l = 2$ components, and one $l = 3$ component.

While Eq. (60) provides a decomposition in terms of subspace dimensions, the numerical irreps decomposition of spherical tensors is implemented by so-called Clebsch-Gordan (CG) coeficients, which define a change of basis from the product basis (irrep<sub>1</sub> ⊗ irrep<sub>2</sub>) to a coupled basis (direct sum of irreps): Let $x ^ { ( l _ { 1 } ) }$ and $y ^ { ( l _ { 2 } ) }$ be spherical tensors transforming as

$$
\begin{array} { l } { { ( D ^ { ( l _ { 1 } ) } ( R ) x ^ { ( l _ { 1 } ) } ) _ { m _ { 1 } } = \displaystyle \sum _ { m _ { 1 } ^ { \prime } = - l _ { 1 } } ^ { l _ { 1 } } D _ { m _ { 1 } m _ { 1 } ^ { \prime } } ^ { ( l _ { 1 } ) } ( R ) x _ { m _ { 1 } ^ { \prime } } ^ { ( l _ { 1 } ) } , } } \\ { { ( D ^ { ( l _ { 2 } ) } ( R ) y ^ { ( l _ { 2 } ) } ) _ { m _ { 2 } } = \displaystyle \sum _ { m _ { 2 } ^ { \prime } = - l _ { 2 } } ^ { l _ { 2 } } D _ { m _ { 2 } m _ { 2 } ^ { \prime } } ^ { ( l _ { 2 } ) } ( R ) y _ { m _ { 2 } ^ { \prime } } ^ { ( l _ { 2 } ) } . } } \end{array}
$$

Their tensor product transforms under the product representation:

$$
\begin{array} { r l r } {  { \big ( ( D ^ { ( l _ { 1 } ) } ( R ) \otimes D ^ { ( l _ { 2 } ) } ( R ) ) ( x ^ { ( l _ { 1 } ) } \otimes y ^ { ( l _ { 2 } ) } ) \big ) _ { m _ { 1 } m _ { 2 } } } } \\ & { } & { = \sum _ { m _ { 1 } ^ { \prime } , m _ { 2 } ^ { \prime } } D _ { m _ { 1 } m _ { 1 } ^ { \prime } } ^ { ( l _ { 1 } ) } ( R ) D _ { m _ { 2 } m _ { 2 } ^ { \prime } } ^ { ( l _ { 2 } ) } ( R ) x _ { m _ { 1 } ^ { \prime } } ^ { ( l _ { 1 } ) } y _ { m _ { 2 } ^ { \prime } } ^ { ( l _ { 2 } ) } . } \end{array}\tag{63}
$$

To extract the L-irrep component of the product representation, we define the Clebsch-Gordan (coupled) product

$$
\begin{array} { l } { { \displaystyle z ^ { ( L ) } = x ^ { ( l _ { 1 } ) } \otimes _ { \mathrm { C G } } y ^ { ( l _ { 2 } ) } } , } \\ { { \displaystyle z _ { M } ^ { ( L ) } : = \sum _ { m _ { 1 } , m _ { 2 } } C _ { l _ { 1 } m _ { 1 } , l _ { 2 } m _ { 2 } } ^ { L M } x _ { m _ { 1 } } ^ { ( l _ { 1 } ) } y _ { m _ { 2 } } ^ { ( l _ { 2 } ) } } , } \end{array}\tag{64}
$$

with $| l _ { 1 } - l _ { 2 } | \le L \le l _ { 1 } + l _ { 2 }$ and $- L \leq M \leq L$ . The famous Clebsch-Gordan coeficients are denoted as $C _ { l _ { 1 } m _ { 1 } , l _ { 2 } m _ { 2 } } ^ { L M } .$ In the literature, the Clebsch-Gordan product is often referred to simply as “tensor product”, which can be confusing since the CG product combines the usual tensor (outer) product with an additional change of basis afterwards. We use the notation $\otimes _ { \mathrm { C G } }$ vs. $\otimes$ to make this distinction explicit.

The defining property of the CG coeficients is that the resulting coupled tensor $z ^ { ( L ) } = x ^ { ( l _ { 1 } ) } \otimes _ { \mathrm { C G } } y ^ { ( l _ { 2 } ) }$ transforms as an L-type spherical tensor under $R \in \mathrm { S O } ( 3 )$ . This condition is equivalent to imposing equivariance of the CG tensor product. Concretely, equivariance of the CG decomposition requires that, for all $R \in \mathrm { S O ( 3 ) }$

$$
\begin{array} { r l r } & { } & { z ^ { ( L ) } \big ( D ^ { ( l _ { 1 } ) } ( R ) x ^ { ( l _ { 1 } ) } , D ^ { ( l _ { 2 } ) } ( R ) y ^ { ( l _ { 2 } ) } \big ) } \\ & { } & { \quad = \ D ^ { ( L ) } ( R ) z ^ { ( L ) } \big ( x ^ { ( l _ { 1 } ) } , y ^ { ( l _ { 2 } ) } \big ) , } \end{array}\tag{65}
$$

i.e. we equate “transform then couple” with “couple then transform”. Expanding both sides and using that this equation must hold for any tensors of $x ^ { ( l _ { 1 } ) } , y ^ { ( l _ { 2 } ) }$ , we obtain the three-D equation. In components, we require that for any $R \in \mathrm { S O ( 3 ) }$

$$
\begin{array} { c } { { \displaystyle \sum _ { m _ { 1 } = - l _ { 1 } } ^ { l _ { 1 } } \sum _ { m _ { 2 } = - l _ { 2 } } ^ { l _ { 2 } } C _ { l _ { 1 } m _ { 1 } , l _ { 2 } m _ { 2 } } ^ { L M } D _ { m _ { 1 } m _ { 1 } ^ { \prime } } ^ { ( l _ { 1 } ) } ( R ) D _ { m _ { 2 } m _ { 2 } ^ { \prime } } ^ { ( l _ { 2 } ) } ( R ) } } \\ { { = \displaystyle \sum _ { M ^ { \prime } = - L } ^ { L } D _ { M M ^ { \prime } } ^ { ( L ) } ( R ) C _ { l _ { 1 } m _ { 1 } ^ { \prime } , l _ { 2 } m _ { 2 } ^ { \prime } } ^ { L M ^ { \prime } } . } } \end{array}\tag{66}
$$

Together with a normalization and a phase convention, this constraint determines the coeficients uniquely. As discussed in Sec. VI A, in the complex sphericalharmonics convention, a z-rotation acts diagonally on spherical tensors, i.e. $D _ { m m ^ { \prime } } ^ { ( l ) } ( R _ { z } ( \alpha ) ) = e ^ { - i m \alpha } \delta _ { m m ^ { \prime } }$ . Plugging $R = R _ { z } ( \alpha )$ into Eq. (66) yields, for all $\alpha ,$

$$
\begin{array} { r l r } & { } & { e ^ { - i ( m _ { 1 } ^ { \prime } + m _ { 2 } ^ { \prime } ) \alpha } C _ { l _ { 1 } m _ { 1 } ^ { \prime } , l _ { 2 } m _ { 2 } ^ { \prime } } ^ { L M } = e ^ { - i M \alpha } C _ { l _ { 1 } m _ { 1 } ^ { \prime } , l _ { 2 } m _ { 2 } ^ { \prime } } ^ { L M } } \\ & { } & { \Rightarrow C _ { l _ { 1 } m _ { 1 } ^ { \prime } , l _ { 2 } m _ { 2 } ^ { \prime } } ^ { L M } = 0 \mathrm { u n l e s s } M = m _ { 1 } ^ { \prime } + m _ { 2 } ^ { \prime } , } \end{array}\tag{67}
$$

which is one of the standard Clebsch-Gordan selection rules. The same sparsity structure holds for the CG coeficients in the real basis. In practice, CG coeficients are rarely computed by solving Eq. (66) directly. Instead, one evaluates so-called Wigner 3j symbols using fast recursion schemes [29, 30].

d. Excursion: angular-momentum coupling as intuition for CG rules. In fact, there is a deep mathematical connection between the coupling of spherical tensors and angular momentum coupling in quantum mechanical two-spin systems. In quantum mechanics, spatial rotations act on state vectors via unitary operators; mathematically, this action is formulated via the special unitary group SU(2). Intuitively, this means that the complex vectors, on which these unitary matrices act, return to themselves only after a $7 2 0 ^ { \circ }$ rotation. However, there is a one-to-one correspondence between integer-l representations of SU(2) and the irreps of SO(3). In this correspondence, l and m can be interpreted as the total angular momentum and its z-component, respectively. In the coupling of two spins, the z-components simply add up, which explains the selection rule $M = m _ { 1 } + m _ { 2 } .$ while the allowed total angular momenta L ranges from $| l _ { 1 } - l _ { 2 } |$ (maximal anti-alignment of the individual angular momenta) to $l _ { 1 } + l _ { 2 }$ (maximal alignment). [24]

e. Computational scaling of the CG product. The coupled product Eq. (64) is a structured contraction with a rank-3 CG tensor. Treating this contraction naively, the cost of one CG product for a single $( l _ { 1 } , l _ { 2 } , L )$ path is often reported as $\bar { \mathcal { O } } ( \ell ^ { 3 } )$ when $l _ { 1 } , l _ { 2 } , L \sim \ell . ^ { 5 }$ In many equivariant architectures one carries all irreps up to a maximal degree L [28, 31, 32]. Then there are $\bar { \mathcal { O } } ( L ^ { 2 } )$ input pairs $( l _ { 1 } , l _ { 2 } )$ , each producing $\mathcal { O } ( L )$ many output spherical tensors. Hence, there are $\mathcal { O } ( L ^ { 3 } )$ input to output paths in total. Combining this with $ { \mathcal { O } } ( L ^ { 3 } )$ per path yields a naive overall scaling of $\mathcal { O } ( L ^ { 6 } )$ for evaluating all tensor product couplings up to degree L.

This cost can become a practical bottleneck: in many equivariant neural networks CG tensor products are executed at essentially every layer. Recent work by Passaro and Zitnick [31] reduce the cost by converting $\mathrm { S O ( 3 ) }$ -convolutions to mathematically equivalent ${ \mathrm { S O } } ( 2 )$ -convolutions, bringing down the total complexity from $\stackrel { \cdot } { O } ( L ^ { 6 } )$ to $\mathcal { O } ( L ^ { 3 } )$ .

Key point. The outer product of tensors is equivariant, but the output representation is reducible. To keep the representations of feature vectors in neural networks compact, we therefore additionally apply an equivariant Clebsch-Gordan decomposition into irreducible representations. In practice, the decomposition is realized by a change of basis using the Clebsch-Gordan coeficients.

## VIII. ROTATIONAL EQUIVARIANCE IN DEEP LEARNING

With the representation-theoretic background in place, we can now illustrate how equivariance can be achieved in neural networks. In this chapter we focus on rotational equivariance, as this is the most involved part of Euclidean symmetry handling; including reflections as well is mostly straightforward. Permutation equivariance is typically ensured by construction in message passing networks, as the aggregation over neighbors is a set operation, and translation invariance is typically achieved by using only relative coordinates $\mathbf { x } _ { i } - \mathbf { x } _ { j }$ and distances $\| { \bf x } _ { i } - { \bf x } _ { j } \| _ { 2 }$ (cf. Sec. III). Rotational equivariance, in contrast, requires a well-defined transformation behavior of all geometric features under $\mathrm { S O ( 3 ) }$ (at least for some approaches). Below we give a brief overview of the most common techniques; for a broader discussion of equivariant message passing we refer the reader to [4].

a. Data augmentation (learned, approximate equivariance). In general, the simplest way to achieve (approximate) equivariance with respect to a set of transformations is to augment the training data, i.e. present the model with randomly transformed samples during training. This strategy is completely independent of the model and works with any architecture and any group. However, the equivariance must be learned and is therefore not guaranteed (in particular, it may fail out-of-distribution), and the learned equivariance is not exact. Still, data augmentation can be very efective in practice and is, for instance, successfully used in large-scale systems such as AlphaFold 3 [33].

b. Equivariance using tensorial internal representations. Historically the first approach to exact rotational invariance (as special case of equivariance) was to feed only invariant geometric inputs, such as distances, angles, or dihedral angles, into a standard (non-equivariant) network [34–38]. While this allows the use of typical deep learning building blocks (linear layers, activations, norm layers, etc.), these approaches are not able to commu nicate non-scalar geometric information (such as directions) during message passing. To overcome this limitation, several approaches for built-in rotational equivariance have been developed that leverage tensorial internal features that transform under specified group representations. Tracking the representation (i.e. transformation behavior) of internal features enables these models to predict equivariant outputs. Tensorial internal features have also shown to improve performance even when the final target is invariant [39, 40]. In particular, higherorder tensor representations are helpful in tasks where angular information matters, e.g. for predicting forces in molecules [15, 41]. Exact equivariance in these architectures is ensured by requiring that every layer is equivariant. Two widely used families of such architectures are (i) group-convolutional networks based on the regular representation (see Sec. VIII A below), and (ii) tensor field networks based on irreducible representations (see Sec. VIII B).

## A. Equivariance via Group Convolutions

For discrete groups, a conceptually clean approach to equivariance is to represent features in the network as functions on the group. In practice, this is achieved by a lifting step: an input signal $f : X \to \mathbb { R } ^ { C }$ (e.g. an image on $X = \mathbb { Z } ^ { 2 } )$ with C feature channels is mapped to a group-indexed feature map $F : G \to \mathbb { R } ^ { C }$ by evaluating the signal in all possible transformed frames (indexed by elements $g \in G )$ . Subsequent layers then process F via convolutions over G. We will see that, in this way, equivariance is realized as an indexing symmetry of the lifted feature map [1, 42, 43].

To explicitly follow the construction, we first introduce yet another group representation: the (left) regular representation. Concretely, let G be a finite group that acts on an input domain X (e.g. planar rotations by $9 0 °$ acting on $\bar { X } = \mathbb { Z } ^ { 2 } )$ . Then, we can define the left action on signals $f : X \to \mathbb { R } ^ { C }$ 2

$$
[ L _ { u } f ] ( x ) \ : = \ f ( u ^ { - 1 } \cdot x ) , \qquad u \in G , \ x \in X ,\tag{68}
$$

which acts on a function $f$ by evaluating it in transformed coordinates. We again take the viewpoint that the function itself remains unchanged while the argument at which it is evaluated is transformed. On a groupindexed feature map $F : G \to \mathbb { R } ^ { C }$ , the analogous left regular action is given by

$$
[ L _ { u } F ] ( g ) : = F ( u ^ { - 1 } g ) , \qquad u , g \in G .\tag{69}
$$

This action defines a group representation, since the homomorphism condition $L _ { u _ { 1 } } L _ { u _ { 2 } } ~ = ~ L _ { u _ { 1 } u _ { 2 } }$ (compare to Eq. (16)) follows directly from the associativity of the group G.

a. Lifting as a G-correlation. To move from a signal $f : \dot { X } \xrightarrow { } \mathbb { R } ^ { C }$ to a group-indexed feature map, one applies a lifting layer. Following [1], the lift is conveniently written as a cross-correlation with a learnable filter $\psi : X \to \mathbb { R } ^ { C \times C ^ { \prime } }$

$$
F ( g ) = [ f \star \psi ] ( g ) : = \sum _ { x \in X } f ( x ) \psi ( g ^ { - 1 } \cdot x ) , \qquad g \in G .\tag{70}
$$

That is, the filter $\psi ( g ^ { - 1 } { \cdot } x )$ is applied on the input signal f “transformed” by g, exactly as in the usual $\mathbb { Z } ^ { d }$ correlation $\begin{array} { r } { [ f \star \psi ] ( t ) = \sum _ { x } \breve { f } ( x ) \psi ( x - t ) ) . } \end{array}$ The output of the lift is now a function on $G ,$ , and transforms under the left regular action. Indeed, the lifting defined by Eq. (70) is G-equivariant in the sense that transforming the input signal and then lifting is equivalent to lifting and then applying the left regular action on the group index:

$$
[ ( L _ { u } f ) \star \psi ] ( g ) \ = \ [ L _ { u } ( f \star \psi ) ] ( g ) , \qquad \forall u , g \in G ,\tag{71}
$$

which can be shown as follows:

$$
\begin{array} { l } { { \displaystyle [ ( L _ { u } f ) \star \psi ] ( g ) } } \\ { { \mathrm { = \displaystyle \sum _ { x \in X } ( L _ { u } f ) ( } x ) \psi ( g ^ { - 1 } \cdot x ) = \sum _ { x \in X } f ( u ^ { - 1 } \cdot x ) \psi ( g ^ { - 1 } \cdot x ) } } \end{array}
$$

$$
\begin{array} { r l r } {  { \stackrel { x = u \cdot y } { = } } \sum _ { y \in X } f ( y ) \psi ( g ^ { - 1 } \cdot ( u \cdot y ) ) = \sum _ { y \in X } f ( y ) \psi ( ( u ^ { - 1 } g ) ^ { - 1 } \cdot y ) }  \\ & { } & \\ & { = } & { [ f \star \psi ] ( u ^ { - 1 } g ) = [ L _ { u } ( f \star \psi ) ] ( g ) , \quad \quad \quad ( 7 2 ) } \end{array}
$$

where we used that $x \mapsto u \cdot x$ is a bijection on X (e.g. a permutation of pixels for discrete rotations). Thus, the lift turns an input transformation $( [ L _ { u } f ] ( x ) \overset { \prime } { = } f ( u ^ { - 1 } \cdot x ) )$ into a re-indexing of the group coordinate $\big ( [ L _ { u } F ] ( g ) =$ $F ( u ^ { - 1 } g ) )$ .

b. Group convolution on $G .$ Once features live on $G ,$ the natural analogue of translation correlation is correlation over the group. Given $F : G \to \mathbb { R } ^ { C }$ and a filter $\kappa : G  \mathbb { R } ^ { C \times C ^ { \prime } }$ , we define a group convolution as

$$
[ F \star \kappa ] ( g ) : = \sum _ { h \in G } F ( h ) \kappa ( g ^ { - 1 } h ) , \qquad g \in G .\tag{73}
$$

This is the direct generalization of $\mathbb { Z } ^ { d }$ convolution: the kernel depends only on the relative group element $g ^ { - 1 } h .$ i.e. on the displacement from $g$ to h measured in group coordinates. The group convolution (73) is equivariant with respect to the left regular action:

$$
[ ( L _ { u } F ) \star \kappa ] ( g ) = [ L _ { u } ( F \star \kappa ) ] ( g ) , \qquad \forall u , g \in G .\tag{74}
$$

The proof uses again a change of variables, completely analogous to the one above.

In practice, e.g. for images, one typically retains the spatial index in the lifting and correlates jointly over space and group. A convolution over the group combined with a kernel that is local in X (as in ordinary CNNs) then takes the following form:

$$
( F \star \kappa ) ( x , g ) : = \sum _ { y \in X } \sum _ { h \in G } F ( y , h ) \kappa ( x - y , g ^ { - 1 } h ) ,\tag{75}
$$

so the kernel depends on the relative spatial displacement $x - y$ and the relative group displacement $g ^ { - 1 } h$ . This setting is used, for example, in roto-translation equivariant SE(2) CNNs [42].

c. Nonlinearities and readout. A practical benefit of this construction is that the left regular action (69) acts by re-indexing the group coordinate and does not mix the channel values in $\mathbb { R } ^ { C }$ . Therefore, channel-wise nonlinearities commute with the action: for any scalar nonlinearity σ, we have $\sigma ( L _ { u } F ) = L _ { u } \sigma ( F )$ . This allows standard deep learning layers of alternating group-correlations and channel-wise nonlinearities. In the end, the group index is typically removed by pooling (e.g. averaging) over $^ { g , }$ which is invariant under re-indexing to obtain an invariant or equivariant output (e.g. a segmentation that is equivariant under global rotations by 90<sup>◦</sup> [44]).

Since the feature maps carry an explicit group index, the cost scales with the size |G| of the finite group. In 2D, discretizing SO(2) is straightforward by sampling angles equidistant on [0, 2π], forming the dihedral rotation group. In 3D, however, using a subgroup with exact clo sure under composition becomes restrictive: beyond the dihedral rotation group (which is only planar), the only finite closed rotation groups correspond to the rotational symmetries of the Platonic solids (tetrahedral, octahedral, icosahedral) [45]. While finer discretizations improve the approximation of the full group, they increase runtime and memory, making the choice of resolution a practical accuracy-eficiency trade-of [46].

## B. Equivariance via Internal Tensorial Representations

Probably the most widely used approach for exact SO(3) equivariance is the method of tensor field networks<sup>7</sup>, also referred to as e3nn-style networks, which are often based on the Python package e3nn [13]. In this approach, internal features transform as direct sums of irreducible representations of $\mathrm { S O ( 3 ) }$ and equivariance is enforced by composing equivariant building blocks [5, 6, 13, 27, 39, 47]. In contrast to group convolution networks, these architectures can no longer treat every internal activation as an individual number: the coordinates of vectors (and higher-order tensors) must be processed jointly to maintain equivariance. As a result, tensor field networks rely on representation-aware building blocks, such as equivariant linear maps, equivariant nonlinearities, and norm layers, as well as specialized tensor operations (the Clebsch-Gordan product introduced in Sec. VII) to mix the information of spherical tensors of diferent l.

Concretely, node features consist of spherical tensors $f _ { i } ^ { ( l ) } ~ \in ~ \mathbb { R } ^ { 2 l + 1 }$ transforming under Wigner-D matrices. Messages are constructed from tensorial node features and tensorial convolution filters coupled through the Clebsch-Gordan product. The convolution filters in a typical SO(3)-convolution consist of a learnable (scalar) radial function, and an angular part given by spherical harmonics [13, 39, 47]:

$$
\begin{array} { r l } {  { \Big [ f _ { j } ^ { ( l _ { 1 } ) } \otimes _ { \mathrm { C G } } ( \mathcal { R } ( r _ { i j } ) Y ^ { ( l _ { 2 } ) } ( \hat { \mathbf { r } } _ { i j } ) ) \Big ] _ { m _ { 3 } } ^ { ( l _ { 3 } ) } } \qquad } & { } \\ & { = \sum _ { m _ { 1 } = - l _ { 1 } } ^ { l _ { 1 } } \sum _ { m _ { 2 } = - l _ { 2 } } ^ { l _ { 2 } } C _ { l _ { 1 } m _ { 1 } , l _ { 2 } m _ { 2 } } ^ { l _ { 3 } m _ { 3 } } f _ { j , m _ { 1 } } ^ { ( l _ { 1 } ) } \mathcal { R } ( r _ { i j } ) Y _ { m _ { 2 } } ^ { ( l _ { 2 } ) } ( \hat { \mathbf { r } } _ { i j } ) , } \end{array}\tag{76}
$$

where $r _ { i j } = \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 }$ and $\hat { \mathbf { r } } _ { i j } = ( \mathbf { x } _ { i } - \mathbf { x } _ { j } ) / \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \|$ and ⊗<sub>CG</sub> denotes the tensor product followed by a decomposition into spherical tensors (cf. Sec. VII).

a. Equivariant linear layers, normalization, and nonlinearities. Equivariant linear layers respect the representation structure: for each degree l, one may freely mix channels within the l-type, but not between diferent degrees. Concretely, if a node carries $C _ { l }$ channels of degree-l features, we collect them as $f ^ { ( l ) } \in \dot { \mathbb { R } } ^ { C _ { l } \times ( 2 l + 1 ) }$ with components $f _ { c m } ^ { ( l ) }$ , where $c \in \{ 1 , \ldots , C _ { l } \}$ indexes the channel and $m \in \{ - l , \ldots , l \}$ the (2l+1) irrep components. Then, an equivariant linear map may mix channels but must act identically on all irrep components, i.e. in components

$$
f _ { c ^ { \prime } m } ^ { ( l ) \prime } = \sum _ { c = 1 } ^ { C _ { l } } W _ { c ^ { \prime } c } ^ { ( l ) } f _ { c m } ^ { ( l ) } , \qquad W ^ { ( l ) } \in \mathbb { R } ^ { C _ { l } ^ { \prime } \times C _ { l } } .\tag{77}
$$

That is, the same weight matrix $W ^ { ( l ) }$ is applied for every m. This preserves equivariance because the SO(3) action, in contrast, acts on the m-index.

For nonlinearities, scalar $( l = 0 )$ features can use standard pointwise activations. For $l > 0 .$ , a popular choice are gated nonlinearities [48]: in each feature vector, we reserve one scalar feature $s _ { c }$ per tensorial feature $f _ { c } ^ { ( l ) }$ Then, we apply a nonlinearity σ to $s _ { c }$ to obtain a scalar prefactor, and scale the tensor, i.e. all its components, by

$$
f _ { c m } ^ { ( l ) } \mapsto \sigma \left( s _ { c } \right) f _ { c m } ^ { ( l ) } \quad ( \mathrm { ~ n o ~ s u m ~ o v e r ~ } c \mathrm { ~ } ) .\tag{78}
$$

Since $\sigma ( s _ { c } )$ is a scalar, the scaling commutes with rotations and hence preserves equivariance. In a similar way, normalization layers are typically defined through invariant statistics so that they do not introduce direction dependent biases [6].

A very related line of work are steerable CNNs developed specifically for data on regular Cartesian grids (images and 3D volumes) [48, 49]. Steerable CNNs and SO(3)-equivariant message passing on point clouds share the same principle: feature channels carry representations and convolution kernels are constrained so that each layer is equivariant. The main diference, besides the data domain, is the operational form of the kernels. Steerable CNNs implement equivariance via constrained filter banks and standard discrete convolutions, whereas in tensor field message passing kernels are continuous functions. In both cases, equivariance is obtained by composing equivariant operations.

Related to the idea of bundling node features in spherical tensors transforming under $\mathrm { S O ( 3 ) }$ irreps, specialized architectures exist which use elements of the projective geometric (or Cliford) algebra to achieve Euclidean rotation and translation equivariance [50, 51]. Notably, there is also an alternative line of work that uses Cartesian tensor representations instead of irreps and that avoids the use of costly CG tensor products [52–55].

## C. Canonicalization-Based Approaches

Nonetheless, many popular message passing architectures operating on Euclidean graphs are not equivariant per se [19, 56, 57]. Canonicalization ofers an alternative route to exact equivariance that can be built around non-equivariant backbone architectures and avoids specialized architectural building blocks.

a. Group averaging. In principle, a simple construction to guarantee invariance (or equivariance) is group averaging. For that, one applies a set of transformations to the input (similar to the lifting described in Sec. VIII A), runs the same backbone network for each transformed input, and average the outputs. While this provides a principled guarantee, it can be computationally expensive when the group is large or continuous (and one must approximate the average by a finite set). Practical variants include averaging over a small set predicted from the input, as in frame-averaging methods for rotation equivariance [58, 59].

![](images/7b5ebc9ef0071b4317432dde9931b7cdb4fbc3a6a0c3875a9f65ef51cd417b09.jpg)

![](images/c5ea04dbacff5030fd7c1dec1f531be718fe86169021b8304b7b47a6d342c41c.jpg)  
(a) Local Canonicalization  
(b) Global Canonicalization  
Figure 3. Local vs. global canonicalization of a molecular geometry. Left: Local canonicalization with distinct local frames constructed from nearest-neighbor directions. The left and right molecular substructures, which difer only by a 180 rotation, are equivalent under local canonicalization. Right: Under global canonicalization, all “local” frames are identical. Consequently, identical local substructures appearing in diferent orientations are not treated equivalently. While both approaches achieve global equivariance, only local canonicalization also realizes local equivariance.

b. Global canonicalization. Instead of averaging over multiple group elements, one may also pick one canonical pose in an equivariant way, transform the input into that canonical frame, and then apply a standard backbone. Concretely, a small network (or heuristic) predicts a global orientation, and one canonicalizes the geometric input before processing it with the main model [60]. This factors out the global orientation of the input so that the model output will be invariant. Thus, canonicalization approaches avoid the need for specialized equivariant layers and can provide exact equivariance when the canonical pose is well-defined. The canonical orientation can be estimated in various ways depending on the data modality by using subnetworks [61], PCA-like heuristics [62], asymmetric units [63], anchor points [7], or as a combination of local orientations [64, 65]. Importantly, if the downstream task requires an equivariant (non-scalar) output, one can “undo” the canonicalization at the end of the pipeline. That is, one predicts a geometric quantity in the canonical frame and rotates the prediction back to the original frame using the inverse of the estimated pose.

ordinate frame for each node, into which the geometric input features are transformed. As for global canonicalization, this ensures that the local coordinates of the node features are independent of the global orientation of the input and are thus invariant. One advantage of using local reference frames rather than a single global frame is that geometrically similar substructures are described by similar local features. This may be viewed as a form of local equivariance, as illustrated in Fig. 3. More precisely, under local canonicalization, the same local substructure appearing in diferent orientations is represented by the same local features, because the local reference frame is constructed from the local geometry and therefore rotates together with the substructure. In other words, when expressed in its local frame, the geometry “looks” the same irrespective of its global orientation. By contrast, under global canonicalization all local fea tures are expressed in a single global frame, so identical local substructures appearing in diferent orientations are not represented equivalently. Local canonicalization can thus capture local symmetries that global canonicalization cannot, which may improve learning and generalization.

However, a practical hurdle for local canonicalization is the following communication barrier: the same geometric object expressed in diferent local frames has diferent coordinates. Therefore, naive message passing between nodes with diferent frames does not communicate di rect geometric information faithfully. Many existing approaches for local canonicalization do not address this issue, which substantially limits the communication be tween nodes that have diferent local frames [14]. The authors of [67] and [8] try to learn an approximate frameto-frame transition, and in a recent line of work, exact frame-to-frame transitions are used to enable faithful communication of directed, tensor-valued information [14, 70, 71].

One conceptual, and potentially practical, caveat of (local) canonicalization approaches is the possible discontinuity of the predicted local reference frames used to canonicalize geometric features. From a theoretical perspective, there are impossibility results showing that globally continuous canonicalization maps cannot exist in full generality [72]. Intuitively, discontinuities arise whenever the input geometry does not uniquely determine an orientation. For perfectly symmetric objects (e.g. a sphere), the choice of frame is irrelevant: any reference frame “sees” the same input. The more delicate regime is approximate symmetry. Consider an almost spherical shape with a tiny dent. Even if this asymme try is infinitesimal, it singles out a preferred direction, e.g. the x-axis of our predicted frame. However, removing the infinitesimal dent and making a dent in another place, which overall amounts to an infinitesimal deformation, can therefore induce a finite change in the predicted x-direction. In other words, a near-symmetric input can lead to a locally ill-conditioned orientation estimate.

In practice, one can show that local frames are typically robust to random perturbations of the input geometry, in particular when trained with noisy data [14]. Another promising mitigation strategy is frame averaging, e.g. in the form proposed by Pozdnyakov and Ceriotti [59]. The idea is to predict multiple local frames per node and downweigh contributions from those that are close to being discontinuous with respect to the input geometry.

Table I. Qualitative comparison of common approaches to rotational equivariance in geometric deep learning. ✓ indicates the property typically holds, ✗ indicates it typically does not hold. Continuous G means the method supports continuous (infinite) groups without discretizing the group itself. Architectural flexibility indicates whether equivariance is achieved without enforcing architecural constraints, such as representation-aware nonlinearities. Non-scalar messages refers to whether the method allows to communicate directed geometric information during message passing. The limitations of each approach are highlighted in red circles.
<table><tr><td>Approach</td><td>Exact equiv.</td><td>Contin. G</td><td>Non-scalar messages</td><td>Local equiv.</td><td>Architectural flexibility</td><td>Continuous predictions</td></tr><tr><td>Data augmentation</td><td>X</td><td></td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Group CNNs</td><td>√</td><td>X</td><td>√</td><td>V</td><td>√</td><td>√</td></tr><tr><td>Tensor field networks</td><td>√</td><td></td><td>√</td><td>√</td><td>X</td><td>√</td></tr><tr><td>Global canonicalization</td><td>√</td><td>√</td><td>√</td><td>X</td><td>√</td><td>X</td></tr><tr><td>Local canonicalization</td><td>√</td><td>√</td><td>X</td><td>V</td><td>√</td><td>X</td></tr><tr><td>Local can. + tensor msg.</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>X</td></tr></table>

Key point. Data augmentation ofers a straightforward approach to learn approximate equivariances but exact equivariance is typically achieved by stacking specialized layers that are equivariant by design, such as group convolutions or SO(3)- convolutions based on the Clebsch-Gordon tensor product. Canonicalization approaches avoid architectural constraints but rely on the prediction of a well-defined reference pose to canonicalize geometric inputs.

In summary, we have discussed five main approaches to achieve rotational equivariance in geometric deep learning: data augmentation, group convolutions (group CNNs), tensor field networks, global canonicalization, and local canonicalization. Each of these approaches has its own advantages and limitations. Data augmentation is simple but only provides approximate equivariance. Group convolution and internal tensorial representations achieve exact equivariance but require specialized architectural designs and can be computationally expensive. Global canonicalization avoids architectural constraints but may not capture local symmetries efectively. Local canonicalization can additionally capture local symmetries. A qualitative comparison of the diferent approaches to rotational equivariance is given in Table I.

## D. Conceptual Links between Diferent Approaches

a. Data augmentation and canonicalization. Clearly, a global change of reference frame is equivalent to a global rotation of a geometric structure. Using global canonicalization, where the reference frame is chosen canonically and in an equivariant way, the resulting representation becomes independent of the original global orientation (as is shown further below). If, however, the frame is chosen randomly rather than by a principled construction, global “canonicalization” efectively reduces to data augmentation by random global rotations.

A similar perspective applies to local canonicalization. First we note that global canonicalization can in fact be viewed as a special case of local canonicalization: if all local frames are chosen identically, this is equivalent to selecting a single global reference frame (see Fig. 3). However, with local frames constructed equivariantly from the local geometry, the model can achieve local equivariance, as described above. This provides a useful in ductive bias, since local motifs that appear in diferent orientations can then be recognized as rotated versions of the same underlying pattern. This raises the question of what happens when local frames are chosen at random. In that case, the model should not rely on information encoded in the specific frame choice; its predictions should be independent of the chosen gauge, that is, of the particular set of local frames. This viewpoint connects local canonicalization to gauge-equivariant neural networks [73, 74], which likewise make use of local frames but construct their operations so that predictions are in dependent of the specific frame choice.

An ablation study in [14] compares geometrically constructed local frames with randomly chosen ones. The results suggest that informative, geometry-adapted local frames are beneficial when the model is allowed to exploit them explicitly. This highlights an important trade-of between robustness to frame choice and the additional geometric signal that can be extracted from informative canonical frames.

b. A scalarization and tensorization perspective on canonicalization. At the core of (local) canonicalization stands the idea that by transforming geometric features into equivariantly predicted (local) reference frames, the coordinates of the geometric features become invariant. Concretely, each local frame is represented by a rotation matrix $R \in \mathrm { S O ( 3 ) }$ whose rows $\mathbf { \mathbf { \mathbf { \mathbf { i } } } } _ { 1 } ^ { \mathrm { \Delta } } , \mathbf { \mathbf { \mathbf { \mathbf { n } } } } _ { 2 } ^ { \mathrm { T } } , \mathbf { \mathbf { \mathbf { \mathbf { n } } } } _ { 3 } ^ { \mathrm { T } }$ are geometric basis vectors that transform equivariantly under global rotations. Concretely,

$$
R _ { i } = ( \mathbf { n } _ { i , 1 } , \mathbf { n } _ { i , 2 } , \mathbf { n } _ { i , 3 } ) ^ { \mathrm { T } } = \left(  \begin{array} { l } { { \begin{array} { l } { { \begin{array} { l } { { \begin{array} { l } { { \begin{array} { l } { \end{array} } } } \\ { { { \begin{array} { l } { { \begin{array} { l } { \end{array} } } } \\ { { { \begin{ - } { \mathbf { n } } _ { i , 1 } } } \end{array} } } \end{array} } } } } \\ { { \begin{array} { l } { { \begin{array} { r l } { { \begin{array} { r l } { { \begin{array} { r l } { - } } \end{array} } } \\ { { { \begin{array} { l } { \end{array} } } \end{array} } } \end{array} } } } \end{array} } } } } \\ { { \begin{array} { l } { { \begin{array} { r l } { { \begin{array} { r l } { { \begin{array} { r l } { { \begin{array} { r l } \end{array} } } \end{array} } } \end{array} } } } \end{array} } } \end{array} } } \end{array} } \right) . \end{array} \end{array}\tag{79}
$$

In the literature, this change of coordinates, which amounts to a projection onto the basis vectors $\mathbf { n } _ { i } ,$ is sometimes referred to as scalarization [8]. After processing the resulting invariant quantities with standard (nonequivariant) building blocks, the output is transformed back to the global frame, a step that can be viewed as tensorization.

Let us briefly illustrate the perspective of scalarization and tensorization in formulae. Let $\mathbf { x } \in \mathbb { R } ^ { 3 }$ be a vectorvalued feature in the global frame and let $R \in \mathrm { S O ( 3 ) }$ denote the local frame with $R _ { i j } = ( { \bf n } _ { i } ) _ { j }$ , i.e. $R _ { i j }$ is given by the j-th component of the i-th basis vector. When transforming x into the local frame R, we obtain a scalarized representation of x in the form of its local coordinates,

$$
\begin{array} { r } { \mathbf { s } = R \mathbf { x } \in \mathbb { R } ^ { 3 } , \qquad \mathrm { w i t h } \qquad s _ { i } \ = \ \mathbf { n } _ { i } ^ { \mathrm { T } } \mathbf { x } , \quad i = 1 , 2 , 3 . } \end{array}\tag{80}
$$

That is, in the transformation into the local frame the vector x is projected onto each basis vector n to obtain scalar coordinates $s _ { i }$ . If the input is rotated by a global rotation $Q \in \mathrm { S O ( 3 ) }$ , the vector transforms as ${ \bf x } \mapsto { \bf x } ^ { \prime } =$ $Q \mathbf { x } .$ . Furthermore, equivariant frame prediction implies that $R \mapsto R ^ { \prime } = { \dot { R } } Q ^ { \hat { \mathrm { T } } }$ , since $\mathbf n _ { i } \mapsto \mathbf n _ { i } ^ { \prime } = Q \mathbf n _ { i }$ for each i (cf. Eq. 79). Consequently,

$$
{ \bf s } ^ { \prime } = R ^ { \prime } { \bf x } ^ { \prime } = ( R Q ^ { \mathrm { T } } ) ( Q { \bf x } ) = R { \bf x } = { \bf s } ,\tag{81}
$$

so the coordinates $s _ { i }$ are indeed invariant under global rotations. After processing $s _ { i }$ with an arbitrary (nonequivariant) function, we obtain new scalar coordinates. Let us group three of these into $s _ { i } , \ i = 1 , 2 , 3$ . Then, to produce an equivariant vector output x˜, we tensorize $\tilde { s } _ { i }$ by rotating back into the global frame via

$$
\tilde { { \bf x } } = R ^ { \mathrm { T } } \tilde { { \bf s } } \mathrm { o r } \mathrm { e q u i v a l e n t l y } \tilde { { \bf x } } = \sum _ { i = 1 } ^ { 3 } \tilde { s } _ { i } { \bf n } _ { i } .\tag{82}
$$

That is, the transformation back into the global frame of reference amounts to a linear combination of the basis vectors $\mathbf { n } _ { i }$ with the scalar coeficients ${ \tilde { s } } _ { i } .$ . Indeed, since the basis vectors are predicted equivariantly, it follows directly that x˜ transforms equivariantly under a global rotation $Q , \mathrm { i . e . } \tilde { \mathbf { x } } ^ { \prime } = Q \tilde { \mathbf { x } }$

The same viewpoint extends to rank-2 Cartesian tensors, which are transformed into a local frame, i.e. scalarized, via

$$
{ \boldsymbol { S } } = R T R ^ { \mathrm { { T } } } \in \mathbb { R } ^ { 3 \times 3 } , \qquad { \mathrm { w i t h } } \qquad S _ { i j } = \mathbf { n } _ { i } ^ { \mathrm { { T } } } T \mathbf { n } _ { j } .\tag{83}
$$

Equivalently, nine invariant local features grouped into $\tilde { S } \in \mathbb { R } ^ { 3 \times 3 }$ are tensorized by transforming them back into the global frame of reference via

$$
\tilde { T } = R ^ { \mathrm { T } } \tilde { S } R , \mathrm { o r e q u i v a l e n t l y } \tilde { T } = \sum _ { i , j } \tilde { S } _ { i j } { \bf n } _ { i } { \bf n } _ { j } ^ { \mathrm { T } } .\tag{84}
$$

Clearly, as a sum of outer products of the basis vectors $\mathbf { n } _ { i } ,$ the output $\tilde { T }$ transforms equivariantly under global rotations. Higher-order Cartesian tensors follow analogously: one scalarizes by contracting each index with a basis vector of the local frame, and tensorizes by expanding back with the same basis.

c. Connection to tensor field networks and irreducible representations. This perspective of scalarization and tensorization can be used to connect (local) canonical ization to e3nn-style architectures and equivariant message passing based on SO(3)-convolutions as discussed in Sec. VIII B. The coupled Clebsch-Gordan product between spherical tensor features $f ^ { ( l _ { 1 } ) }$ and the spherical harmonics $Y ^ { ( l _ { 2 } ) } ( \hat { \mathbf { r } } _ { i j } )$ , cf. Eq. (76), contains a scalar output path exactly if $l _ { 1 } = l _ { 2 } = l$ due to the composition rule Eq. (60). In that case, we may interpret the scalar output channel

$$
f ^ { ( 0 ) } : = \left( f ^ { ( l ) } \otimes _ { \mathrm { C G } } Y ^ { ( l ) } ( \hat { \mathbf { r } } _ { i j } ) \right) ^ { ( 0 ) } = \sum _ { m = - l } ^ { l } f _ { m } ^ { ( l ) } Y _ { l m } ( \hat { \mathbf { r } } _ { i j } )\tag{85}
$$

as a scalarization, in which we project the feature $f ^ { ( l ) }$ onto the basis given by the spherical harmonics $Y _ { l m } ( \hat { \mathbf { r } } _ { i j } )$ The resulting scalar feature $f ^ { ( 0 ) }$ is invariant under global rotations and is typically processed with unconstrained building blocks in e3nn-style architectures (as in canonicalization-based approaches). Afterwards, a CG-tensor product with spherical harmonics $Y ^ { ( l ) } ( \hat { \mathbf { r } } _ { i j } )$ can be used to tensorize the scalar features $\tilde { f } ^ { ( 0 ) }$ through the only available path:

$$
\left( \tilde { f } ^ { ( 0 ) } \otimes _ { \mathrm { C G } } Y ^ { ( l ) } ( \hat { \mathbf { r } } _ { i j } ) \right) ^ { ( l ) } = \tilde { f } ^ { ( 0 ) } Y ^ { ( l ) } ( \hat { \mathbf { r } } _ { i j } ) = : \tilde { f } ^ { ( l ) } ,\tag{86}
$$

i.e. the scalar multiplication of the vector of spherical harmonics $Y ^ { ( l ) } ( \hat { \mathbf { r } } _ { i j } )$ , resulting in a spherical tensor $\tilde { f } ^ { ( l ) }$

Focusing only on the scalar paths of the Clebsch-Gordan tensor product in the beginning of the pipeline (scalarization) and on the scalar multiplication of spherical features provided by the spherical harmonics in the end (tensorization) reveals an interesting connection between canonicalization and e3nn-style architectures. One key diference (besides the other tensorial channels in e3nn-style architectures) is that in (local) canonicalization, we project and expand tensorial features with a complete basis, i.e. three orthonormal vectors, instead of the projection only defined by the relative distance vector $\hat { \mathbf { r } } _ { i j }$ in the case of e3nn-style architectures.

## IX. PROBLEMS WITH ROTATIONAL EQUIVARIANCE IN PRACTICE

As we have seen, equivariant architectures are conceptually well motivated and have become a standard tool in geometric deep learning for good reasons. In practice, however, the specialized building blocks are usually computationally more demanding or do not align as well with highly optimized hardware primitives, such as dense matrix multiplication, leading to substantial runtime overheads [31]. Further, the constraints imposed by exact equivariance can complicate optimization: initialization and training recipes are less standardized, and several works report that such models require careful tun ing and can be dificult to train successfully at scale [75– 78]. For example, Pertigkiozoglou et al. [76] emphasize that equivariant models “can be dificult to optimize and require careful hyperparameter tuning to train successfully”. These issues matter in diferent ways across contexts: while large-scale industrial settings may aford substantial compute, the dominant bottleneck is often human time for model development, and debugging; in academic settings, both compute budgets and the added engineering complexity can become limiting factors.

As a consequence, while exact equivariance remains desirable in principle, relaxing or even dropping built-in symmetries has become a (re-)emerging pattern in practice, particularly in molecular machine learning [75, 79– 83]. A prominent example is AlphaFold 3, where the authors state that “we find that no invariance or equivariance with respect to global rotations and translation of the molecule are required in the architecture and so we omit them to simplify the machine learning architecture” [33]. This trend highlights a central tradeof: we would like the physical guarantees of exact equivariance, but without sacrificing the architectural freedom and efficiency of standard deep learning components.

In the context of rotational equivariance, some recent works even suggest that enforcing exact symmetries may hinder optimal scaling behavior as more data becomes available [75, 78]. This hypothesis resonates with the tough message of Richard Sutton’s “bitter lesson” [84]: while incorporating prior knowledge into architectures can yield short-term gains and is intellectually satisfying, Richard Sutton postulates that it inhibits the longrun progress of deep learning that comes from scaling general-purpose architectures. One may argue that we have indeed observed Sutton’s bitter lesson in practice in two prominent examples. First, deep learning models in general apply end-to-end representation learning, where the network discovers its own features from data, instead of using hand-crafted input features. Second, transformer-based Large Language Models (LLMs) are immensely successful at scale and rely on the generalpurpose self-attention mechanism (see Sec. III), which can be interpreted as a content-dependent dynamic routing of information; in this sense, the model learns content-dependent patterns (or biases) from data rather than enforcing these. Relatedly, modern LLMs utilize content-dependent routing mechanisms such as mixture of-experts [85, 86], another example of data-driven specialization. At the same time, even transformers incorporate inductive biases. As discussed in Sec. III, they can be seen as an instance of message passing networks and as such enforce exact permutation equivariance. At least for such a large symmetry group, enforcing equivariance is indispensable, right?

Regarding rotational equivariance, for instance, Batzner et al. [15] reported improved data eficiency for models with built-in symmetry, arguing that such models need not allocate network capacity to learning the symmetry from data. The least we can say today is that the current picture is less clear [87] and that there is conflicting empirical evidence. This connects to the recent finding that built-in equivariance via local canonicalization can be more data eficient – in the sense that model performance improves faster as more data becomes available – while data augmentation can yield better accuracies in the low-data regime [14, 70, 71]. Put diferently, equivariance often corresponds to a steeper scaling curve, but a steeper slope does not necessarily imply better absolute accuracy for small datasets. For these reasons, across-the-board prescriptions to either always enforce equivariance, or never enforce equivariance, seem unhelpful. In practice, it is rarely clear a priori how important exact symmetry is for a given problem, and how this interacts with dataset size, target quantity, and the optimization landscape. We believe that conducting more systematic side-by-side comparisons of approximate equivariance via data augmentation and built-in equivariance is an interesting avenue for future research.

## X. CONCLUSION

Rotational equivariance provides a principled way to encode coordinate independence in machine learning models for 3D data, ensuring that models respond to rotations in a structured and physically meaningful way. This tutorial has developed the subject from first principles to practical constructions, beginning with symmetry and coordinate independence, and building through geometric deep learning, group representations, spherical harmonics, Wigner matrices, tensor products, and Clebsch-Gordan decomposition to the main architectural strategies used in modern equivariant learning. By relating these ingredients to one another and making explicit the conceptual links between diferent formalisms, the tutorial aims to give readers a coherent understanding of a field whose central ideas are often introduced in diferent mathematical and architectural languages.

Understanding rotational equivariance is useful not only because it clarifies the theoretical foundations of modern 3D machine learning, but also because it informs practical model design. In applications, one must choose between diferent ways of incorporating geometric structure, each with its own trade-ofs in expressivity, computational eficiency, implementation complexity, and numerical stability. It is therefore important to understand when exact symmetry should be built into a model, when approximate methods may be suficient, and how different architectural choices reflect diferent assumptions about the task and the data. This is particularly relevant in scientific and engineering settings, where high-quality 3D data are often limited and simulation data can be costly to obtain, so that appropriate symmetry-aware inductive biases can play an important role in improving data eficiency and generalization.

## ACKNOWLEDGMENTS

This work has received funding from the Klaus Tschira Stiftung gGmbH (Simplaix project) and the Wildcard program from the Carl Zeiss Stiftung. Further, it is supported by Deutsche Forschungsgemeinschaft (DFG) under Germany’s Excellence Strategy EXC-2181/1 – 390900948 (the Heidelberg STRUCTURES Excellence Cluster) as well as under project number 240245660 – SFB 1129. The authors acknowledge support by the state of Baden-Württemberg through bwHPC and the German Research Foundation (DFG) through grant INST 35/1597-1 FUGG.

## REFERENCES

[1] T. Cohen and M. Welling, Group equivariant convolutional networks, in International conference on machine learning (PMLR, 2016) pp. 2990–2999.

[2] T. S. Cohen, M. Geiger, and M. Weiler, A general theory of equivariant cnns on homogeneous spaces, Advances in neural information processing systems 32 (2019).

[3] M. M. Bronstein, J. Bruna, T. Cohen, and P. Veličković, Geometric deep learning: Grids, groups, graphs, geodesics, and gauges, arXiv preprint arXiv:2104.13478 (2021).

[4] A. Duval, S. V. Mathis, C. K. Joshi, V. Schmidt, S. Miret, F. D. Malliaros, T. Cohen, P. Liò, Y. Bengio, and M. Bronstein, A hitchhiker’s guide to geometric gnns for 3d atomic systems, arXiv:2312.07511 [cs.LG] (2024), arXiv:2312.07511 [cs.LG].

[5] I. Batatia, D. P. Kovacs, G. Simm, C. Ortner, and G. Csanyi, MACE: Higher order equivariant message passing neural networks for fast and accurate force fields, in Advances in Neural Information Processing Systems, Vol. 35 (2022) pp. 11423–11436.

[6] Y.-L. Liao, B. M. Wood, A. Das, and T. Smidt, EquiformerV2: Improved equivariant transformer for scaling to higher-degree representations, in The Twelfth International Conference on Learning Representations (2024).

[7] Y. Lou, Z. Ye, Y. You, N. Jiang, J. Lu, W. Wang, L. Ma, and C. Lu, Crin: rotation-invariant point cloud analysis and rotation estimation via centrifugal reference frame, in Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37 (2023) pp. 1817–1825.

[8] Y. Du, L. Wang, D. Feng, G. Wang, S. Ji, C. P. Gomes, Z.-M. Ma, et al., A new perspective on building eficient and expressive 3D equivariant graph neural networks, Advances in Neural Information Processing Systems 36 (2024).

[9] S. Aykent and T. Xia, Gotennet: Rethinking eficient 3d equivariant graph neural networks, in The Thirteenth International Conference on Learning Representations (2025).

[10] R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M. Fortunato, F. Alet, S. Ravuri, T. Ewalds, Z. Eaton-Rosen, W. Hu, et al., Learning skillful mediumrange global weather forecasting, Science 382, 1416 (2023).

[11] J. Spinner, V. Bresó, P. De Haan, T. Plehn, J. Thaler, and J. Brehmer, Lorentz-equivariant geometric algebra transformers for high-energy physics, in Advances in Neural Information Processing Systems, Vol. 37 (2024) 2405.14806.

[12] R. Remme, T. Kaczun, T. Ebert, C. A. Gehrig, D. Geng, G. Gerhartz, M. K. Ickler, M. V. Klockow, P. Lippmann, J. S. Schmidt, S. Wagner, A. Dreuw, and F. A. Hamprecht, Stable and accurate orbital-free density functional theory powered by machine learning, Journal of the American Chemical Society 147, 28851 (2025).

[13] M. Geiger and T. Smidt, e3nn: Euclidean neural networks, arXiv preprint arXiv:2207.09453 (2022).

[14] P. Lippmann, G. Gerhartz, R. Remme, and F. A. Hamprecht, Beyond canonicalization: How tensorial messages improve equivariant message passing, in International Conference on Representation Learning, Vol. 2025, edited by Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (2025) pp. 88067–88087.

[15] S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky, E(3)-equivariant graph neural networks for data-eficient and accurate interatomic potentials, Nature communications 13, 2453 (2022).

[16] J. T. Frank, O. T. Unke, K.-R. Müller, and S. Chmiela, A euclidean transformer for fast and stable machine learned force fields, Nature Communications 15, 6539 (2024).

[17] R. Kondor, The principles behind equivariant neural networks for physics and chemistry, Proceedings of the National Academy of Sciences 122, e2415656122 (2025).

[18] J. Han, Y. Rong, T. Xu, and W. Huang, Geometrically equivariant graph neural networks: A survey, arXiv preprint arXiv:2202.07230 (2022).

[19] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, Pointnet++: Deep hierarchical feature learning on point sets in a metric space, Advances in neural information processing systems 30 (2017).

[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, Attention is all you need, Advances in Neural Information

Processing Systems 30 (2017).

[21] N. Jeevanjee, An introduction to tensors and group theory for physicists (Springer, 2011).

[22] S. Lipschutz and M. L. Lipson, Linear algebra (MacGraw-Hill, 2009).

[23] G. Arfken and J. Romain, Mathematical methods for physicists (1967).

[24] A. R. Edmonds, Angular momentum in quantum mechanics, Vol. 4 (Princeton university press, 1996).

[25] W. Fulton and J. Harris, Representation theory: a first course (Springer Science & Business Media, 2013).

[26] D. Pinchon and P. E. Hoggan, Rotation matrices for real spherical harmonics: general rotations of atomic orbitals in space-fixed axes, Journal of Physics A: Mathematical and Theoretical 40, 1597 (2007).

[27] A. Musaelian, S. Batzner, A. Johansson, L. Sun, C. J. Owen, M. Kornbluth, and B. Kozinsky, Learning local equivariant representations for large-scale atomistic dynamics, Nature Communications 14, 579 (2023).

[28] Y.-L. Liao, B. M. Wood, A. Das, and T. Smidt, Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations, in The Twelfth International Conference on Learning Representations (2024).

[29] K. Schulten and R. G. Gordon, Exact recursive evaluation of 3 j-and 6 j-coeficients for quantum-mechanical coupling of angular momenta, Journal of Mathematical Physics 16, 1961 (1975).

[30] H. T. Johansson and C. Forssén, Fast and accurate evaluation of wigner 3 j, 6 j, and 9 j symbols using prime factorization and multiword integer arithmetic, SIAM Journal on Scientific Computing 38, A376 (2016).

[31] S. Passaro and C. L. Zitnick, Reducing SO(3) convolutions to SO(2) for eficient equivariant gnns, in International Conference on Machine Learning (PMLR, 2023) pp. 27420–27438.

[32] X. Fu, B. M. Wood, L. Barroso-Luque, D. S. Levine, M. Gao, M. Dzamba, and C. L. Zitnick, Learning smooth and expressive interatomic potentials for physical property prediction, arXiv preprint arXiv:2502.12147 (2025).

[33] J. Abramson, J. Adler, J. Dunger, R. Evans, T. Green, A. Pritzel, O. Ronneberger, L. Willmore, A. J. Ballard, J. Bambrick, et al., Accurate structure prediction of biomolecular interactions with alphafold 3, Nature 630, 493 (2024).

[34] K. T. Schütt, H. E. Sauceda, P.-J. Kindermans, A. Tkatchenko, and K.-R. Müller, Schnet–a deep learning architecture for molecules and materials, The Journal of Chemical Physics 148 (2018).

[35] J. Gasteiger, J. Groß, and S. Günnemann, Directional message passing for molecular graphs, in International Conference on Learning Representations (2020).

[36] X. Li, R. Li, G. Chen, C.-W. Fu, D. Cohen-Or, and P.- A. Heng, A rotation-invariant framework for deep point cloud analysis, IEEE transactions on visualization and computer graphics 28, 4503 (2021).

[37] J. Gasteiger, F. Becker, and S. Günnemann, Gemnet: Universal directional graph neural networks for molecules, Advances in neural information processing systems 34, 6790 (2021).

[38] Y. Liu, L. Wang, M. Liu, Y. Lin, X. Zhang, B. Oztekin, and S. Ji, Spherical message passing for 3d molecular graphs, in International conference on learning representations (2022).

[39] F. Fuchs, D. Worrall, V. Fischer, and M. Welling, SE(3)- transformers: 3D roto-translation equivariant attention networks, Advances in neural information processing systems 33, 1970 (2020).

[40] J. Brandstetter, R. Hesselink, E. van der Pol, E. J. Bekkers, and M. Welling, Geometric and physical quantities improve E(3) equivariant message passing, in International Conference on Learning Representations (2022).

[41] L. Zitnick, A. Das, A. Kolluru, J. Lan, M. Shuaibi, A. Sriram, Z. Ulissi, and B. Wood, Spherical channels for modeling atomic interactions, Advances in Neural Information Processing Systems 35, 8054 (2022).

[42] E. J. Bekkers, M. W. Lafarge, M. Veta, K. A. Eppenhof, J. P. Pluim, and R. Duits, Roto-translation covariant convolutional networks for medical image analysis, in International conference on medical image computing and computer-assisted intervention (Springer, 2018) pp. 440–448.

[43] E. J. Bekkers, S. Vadgama, R. D. Hesselink, P. A. Van der Linden, and D. W. Romero, Fast, expressive se (n) equivariant networks through weightsharing in position-orientation space, arXiv preprint arXiv:2310.02970 (2023).

[44] D. Worrall and G. Brostow, Cubenet: Equivariance to 3d rotation and translation, in Proceedings of the European Conference on Computer Vision (ECCV) (2018) pp. 567– 584.

[45] N. Beisert, Symmetries in physics, ETH Zurich (2015).

[46] M. M. Islam, R. Anand, D. R. Wessels, F. de Kruif, T. P. Kuipers, R. Ying, C. I. Sánchez, S. Vadgama, G. Bökman, and E. J. Bekkers, Platonic transformers: A solid choice for equivariance, arXiv preprint arXiv:2510.03511 (2025).

[47] N. Thomas, T. Smidt, S. Kearnes, L. Yang, L. Li, K. Kohlhof, and P. Riley, Tensor field networks: Rotation- and translation-equivariant neural networks for 3d point clouds, arXiv:1802.08219 [cs.LG] (2018), arXiv:1802.08219 [cs.LG].

[48] M. Weiler, M. Geiger, M. Welling, W. Boomsma, and T. S. Cohen, 3D steerable cnns: Learning rotationally equivariant features in volumetric data, Advances in Neural Information Processing Systems 31 (2018).

[49] M. Weiler and G. Cesa, General E(2)-Equivariant Steerable CNNs, in Conference on Neural Information Processing Systems (NeurIPS) (2019).

[50] J. Brehmer, P. de Haan, S. Behrends, and T. S. Cohen, Geometric algebra transformer, in Advances in Neural Information Processing Systems, Vol. 36 (2023) pp. 35472– 35496.

[51] D. Ruhe, J. Brandstetter, and P. Forré, Cliford group equivariant neural networks, in Advances in Neural Information Processing Systems, Vol. 36 (2023) pp. 62922– 62990.

[52] C. Deng, O. Litany, Y. Duan, A. Poulenard, A. Tagliasacchi, and L. J. Guibas, Vector neurons: A general framework for SO(3)-equivariant networks, in Proceedings of the IEEE/CVF International Conference on Computer Vision (2021) pp. 12200–12209.

[53] V. G. Satorras, E. Hoogeboom, and M. Welling, E (n) equivariant graph neural networks, in International conference on machine learning (PMLR, 2021) pp. 9323– 9332.

[54] B. Cheng, Cartesian atomic cluster expansion for machine learning interatomic potentials, npj Computational

Materials 10, 157 (2024).

[55] G. Simeon and G. De Fabritiis, TensorNet: Cartesian tensor representations for eficient learning of molecular potentials, in Advances in Neural Information Processing Systems, Vol. 36 (2023) pp. 37334–37353.

[56] Y. Liu, B. Fan, S. Xiang, and C. Pan, Relation-shape convolutional neural network for point cloud analysis, in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019) pp. 8895–8904.

[57] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, Dynamic graph cnn for learning on point clouds, ACM Transactions on Graphics (tog) 38, 1 (2019).

[58] O. Puny, M. Atzmon, E. J. Smith, I. Misra, A. Grover, H. Ben-Hamu, and Y. Lipman, Frame averaging for invariant and equivariant network design, in International Conference on Learning Representations (2022).

[59] S. Pozdnyakov and M. Ceriotti, Smooth, exact rotational symmetrization for deep learning on point clouds, in Advances in Neural Information Processing Systems, Vol. 36 (2023) pp. 79469–79501.

[60] S.-O. Kaba, A. K. Mondal, Y. Zhang, Y. Bengio, and S. Ravanbakhsh, Equivariance with learned canonicalization functions, in International Conference on Machine Learning (PMLR, 2023) pp. 15546–15566.

[61] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, Pointnet: Deep learning on point sets for 3d classification and segmentation, in Proceedings of the IEEE conference on computer vision and pattern recognition (2017) pp. 652–660.

[62] F. Li, K. Fujiwara, F. Okura, and Y. Matsushita, A closer look at rotation-invariant deep point cloud analysis, in Proceedings of the IEEE/CVF International Conference on Computer Vision (2021) pp. 16218–16227.

[63] J. Baker, S.-H. Wang, T. de Fernex, and B. Wang, An explicit frame construction for normalizing 3D point clouds, in Forty-first International Conference on Machine Learning (2024).

[64] Y. Zhao, T. Birdal, J. E. Lenssen, E. Menegatti, L. Guibas, and F. Tombari, Quaternion equivariant capsule networks for 3D point clouds, in European conference on computer vision (Springer, 2020) pp. 1–19.

[65] C. Zhao, J. Yang, X. Xiong, A. Zhu, Z. Cao, and X. Li, Rotation invariant point cloud analysis: Where local geometry meets global topology, Pattern Recognition 127, 108626 (2022).

[66] Z. Zhang, B.-S. Hua, W. Chen, Y. Tian, and S.-K. Yeung, Global context aware convolutions for 3D point cloud understanding, in 2020 International Conference on 3D Vision (IEEE, 2020) pp. 210–219.

[67] X. Wang and M. Zhang, Graph neural network with local frame for molecular potential energy surface, in Proceedings of the First Learning on Graphs Conference (PMLR, 2022) pp. 19:1–19:30.

[68] S. Luo, J. Li, J. Guan, Y. Su, C. Cheng, J. Peng, and J. Ma, Equivariant point cloud analysis via learning orientations for message passing, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022) pp. 18932–18941.

[69] W. Du, H. Zhang, Y. Du, Q. Meng, W. Chen, N. Zheng, B. Shao, and T.-Y. Liu, Se (3) equivariant graph neural networks with complete local frames, in International Conference on Machine Learning (PMLR, 2022) pp. 5583–5608.

[70] G. Gerhartz, P. Lippmann, and F. A. Hamprecht, Equiv-

ariance by local canonicalization: A matter of representation, in NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations (2025).

[71] J. Spinner, L. Favaro, P. Lippmann, S. Pitz, G. Gerhartz, T. Plehn, and F. A. Hamprecht, Lorentz local canonicalization: How to make any network lorentz-equivariant, in The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025).

[72] N. Dym, H. Lawrence, and J. W. Siegel, Equivariant frames and the impossibility of continuous canonicalization, in International Conference on Machine Learning (PMLR, 2024) pp. 12228–12267.

[73] T. Cohen, M. Weiler, B. Kicanaoglu, and M. Welling, Gauge equivariant convolutional networks and the icosahedral cnn, in International conference on Machine learning (PMLR, 2019) pp. 1321–1330.

[74] P. De Haan, M. Weiler, T. Cohen, and M. Welling, Gauge equivariant mesh cnns: Anisotropic convolutions on geometric graphs, arXiv preprint arXiv:2003.05425 (2020).

[75] Y. Wang, A. A. Elhag, N. Jaitly, J. M. Susskind, and M. A. Bautista, Swallowing the bitter pill: Simplified scalable conformer generation, arXiv preprint arXiv:2311.17932 (2023).

[76] S. Pertigkiozoglou, E. Chatzipantazis, S. Trivedi, and K. Daniilidis, Improving equivariant model training via constraint relaxation, Advances in Neural Information Processing Systems 37, 83497 (2024).

[77] A. A. Elhag, T. K. Rusch, F. Di Giovanni, and M. Bronstein, Relaxed equivariance via multitask learning, arXiv preprint arXiv:2410.17878 (2024).

[78] E. Qu and A. Krishnapriyan, The importance of being scalable: Improving the speed and accuracy of neural network interatomic potentials across chemical domains, Advances in Neural Information Processing Systems 37, 139030 (2024).

[79] M. F. Langer, S. N. Pozdnyakov, and M. Ceriotti, Probing the efects of broken symmetries in machine learning, arXiv preprint arXiv:2406.17747 (2024).

[80] M. Eissler, T. Korjakow, S. Ganscha, O. T. Unke, K.-R. MÃžller, and S. Gugler, How simple can you go? an ofthe-shelf transformer approach to molecular dynamics, arXiv preprint arXiv:2503.01431 (2025).

[81] A. Manolache, L. F. O. Chamon, and M. Niepert, Learning (approximately) equivariant networks via constrained optimization, in The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025).

[82] T. Berndt and J. Stühmer, Approximate equivariance via projection-based regularisation, arXiv preprint arXiv:2601.05028 (2026).

[83] F. Bigi, P. Pegolo, A. Mazitov, and M. Ceriotti, Pushing the limits of unconstrained machine-learned interatomic potentials, arXiv preprint arXiv:2601.16195 (2026).

[84] R. Sutton, The bitter lesson, Incomplete Ideas (blog) 13, 38 (2019).

[85] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, Adaptive mixtures of local experts, Neural computation 3, 79 (1991).

[86] W. Fedus, B. Zoph, and N. Shazeer, Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity, Journal of Machine Learning Research 23, 1 (2022).

[87] J. Brehmer, S. Behrends, P. de Haan, and T. Cohen, Does equivariance matter at scale?, arXiv preprint arXiv:2410.23179 (2024).