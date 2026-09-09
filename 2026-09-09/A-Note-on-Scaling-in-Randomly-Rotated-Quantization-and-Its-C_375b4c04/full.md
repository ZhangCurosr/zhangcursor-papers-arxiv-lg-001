# A Note on Scaling in Randomly Rotated Quantization and Its Connection to the CDEF +1 Pythagorean Relation

Uri Erez   
School of Electrical Engineering   
Tel Aviv University   
Tel Aviv, Israel   
uri@eng.tau.ac.il

## Abstract

Quantization schemes based on randomized rotations have recently received renewed attention, including the roles of MMSE and unbiased reconstruction scalings. In this note, we point out the connection to classical results in statistical signal processing and communication theory. Specifically, the two re construction scales used in the EDEN line of work admit a natural interpretation as finite-dimensional, realization-dependent counterparts of the Wiener and unbiased coeficients in the classical CDEF formulation. At finite blocklength, the CDEF +1 relation holds pointwise for each rotation realization as an exact geometric (Pythagorean) identity, but does not hold after averaging the distortions over the rotation. The classical SNR relation SNR = SNR +1 is recovered as d → ∞: once the overall scale is handled separately, the empirical coordinate statistics of a randomly rotated vector approach their i.i.d. Gaussian counterparts, and the rotation-dependent quantities concentrate. Importantly, EDEN goes beyond this classical correspondence: for every finite d, its Haar-rotation formulation guarantees exact conditional unbiasedness, a stronger property than the second-order notion of unbiasedness in CDEF. We further comment on two distinct roles random rotations play in quantization: one is approximate Gaussianization of the coordinates; the other is decorrelation of reconstruction errors across quantization branches.

## 1 Introduction

Quantization schemes based on randomized rotations have recently received renewed attention, primarily due to their central role in LLM quantization; representative examples include [1, 2, 3]. Some of the underlying ideas, and in particular the random-rotate–quantize–rotate-back (RQRB) pipeline, can be traced back to classical areas in information theory and statistical signal processing; see, e.g., the discussions in [3, 4]. The purpose of this note is to shed some further light on these connections. Specifically, we:

1. Trace two distinct roles of the RQRB pipeline in the quantization literature: approximate Gaussianization on the one hand, and decorrelation of quantization errors on the other, with the latter corresponding to its interpretation as a dithering mechanism [9, 10].

2. Relate the two reconstruction scalings employed in the EDEN line of work [5, 6] to the classical theory of Ciofi, Dudevoir, Eyuboglu, and Forney (CDEF) [7]. Namely, the EDEN scalings are precisely finitedimensional, realization-dependent counterparts of the classical Wiener and unbiased CDEF scalings. This correspondence is exact for every rotation realization; as the dimension grows, concentration connects these realization-dependent quantities to the classical scalar-Gaussian setting and its familia +1 SNR relation.

## 2 The Classical Biased/Unbiased Relation

We begin with the classical scalar setting. Let X and Y be correlated zero-mean random variables, and denote $P _ { X } = \mathbb { E } [ X ^ { 2 } ] , P _ { Y } = \mathbb { E } [ Y ^ { 2 } ]$ , and $r _ { X Y } = \mathbb { E } [ X Y ]$

As we recall next, Ciofi, Dudevoir, Eyuboglu, and Forney (CDEF) distinguish two natural linear scalings of $Y$ , corresponding to biased and unbiased linear MMSE estimation [7]; see also Forney’s later Hilbert-space exposition [8].

## 2.1 Wiener scaling

The Wiener, or linear-MMSE, coeficient minimizes $\mathbb { E } [ ( X - \alpha Y ) ^ { 2 } ]$ and is given by

$$
\alpha _ { \mathrm { W } } = \alpha _ { \mathrm { W } } ( r _ { X Y } , P _ { Y } ) \triangleq \frac { r _ { X Y } } { P _ { Y } } .\tag{1}
$$

By the orthogonality principle, the optimal estimation error satisfies

$$
\mathbb { E } [ E Y ] = 0 ,\tag{2}
$$

where $\hat { X } = \alpha _ { \mathrm { { W } } } Y$ is the linear MMSE estimator and $E = X - { \hat { X } } . { } ^ { 1 }$

Thus, Wiener scaling makes the estimation error orthogonal to the observation and yields the backwardchannel relation

$$
X = \alpha _ { \mathrm { W } } Y + E ,\tag{3}
$$

with $E \perp Y$

## 2.2 Unbiased (unit signal gain) scaling

As developed in CDEF [7], the unbiased coeficient $\mathrm { i s ^ { 2 } }$

$$
\alpha _ { \mathrm { U } } = \alpha _ { \mathrm { U } } ( r _ { X Y } , P _ { X } ) \triangleq \frac { P _ { X } } { r _ { X Y } } .\tag{4}
$$

The unbiased estimator is then

$$
Z \triangleq \alpha _ { \mathrm { U } } Y = X + N ,\tag{5}
$$

where

$$
{ \mathbb E } [ X N ] = \alpha _ { \mathrm { U } } { \mathbb E } [ X Y ] - { \mathbb E } [ X ^ { 2 } ] = 0 .\tag{6}
$$

Thus $\alpha _ { \mathrm { U } }$ makes the efective noise orthogonal to the source $( N \perp X )$ . The relation (5) is called the forwardchannel realization.

Remark 1. Here “unbiased” is used in the the weaker, second-order CDEF sense of unit signal gain. The condition X ⊥ N does not by itself imply the (conditional) unbiasedness relation $\mathbb { E } [ Z | X ] = X$ , equivalently $\mathbb { E } [ N | X ] = 0$ . See discussion of EDEN in the sequel.

Writing $P _ { N } = \mathbb { E } [ N ^ { 2 } ]$ , the SNR of the unbiased, forward-channel representation is naturally defined as

$$
\mathsf { S N R } _ { \mathrm { M M S E , U } } = \frac { P _ { X } } { P _ { N } } .\tag{7}
$$

If one starts with the forward channel relation $Z = X + N$ , the Wiener reconstruction is obtained by the additional scaling

$$
\hat { X } = \alpha _ { \mathrm { { W | U } } } Z ,
$$

with reconstruction error

$$
E _ { \alpha } = X - \alpha Z = ( 1 - \alpha ) X - \alpha N .\tag{8}
$$

Since $X \perp N$

$$
\mathbb { E } [ E _ { \alpha } ^ { 2 } ] = ( \alpha - 1 ) ^ { 2 } P _ { X } + \alpha ^ { 2 } P _ { N } .\tag{9}
$$

The minimizing coeficient relative to the unit-gain representation is

$$
\alpha _ { \mathrm { W | U } } = \frac { P _ { X } } { P _ { X } + P _ { N } } .\tag{10}
$$

Thus, we have $\begin{array} { r } { \alpha _ { \mathrm { W | U } } = \frac { \alpha _ { \mathrm { W } } } { \alpha _ { \mathrm { U } } } } \end{array}$ , or equivalently, $\alpha _ { \mathrm { W } } = \alpha _ { \mathrm { W | U } } \alpha _ { \mathrm { U } }$ . The resulting MMSE is

$$
{ \mathsf { M M S E } } = { \frac { P _ { X } P _ { N } } { P _ { X } + P _ { N } } } .\tag{11}
$$

Defining<sup>3</sup>

$$
\mathsf { S N R } _ { \mathrm { M M S E } } = \frac { \mathsf { P } _ { \mathsf { X } } } { \mathsf { M M S E } } ,\tag{12}
$$

gives the biased/unbiased SNR relation of Lemma 2 of CDEF,

$$
\mathsf { S N R } _ { \mathrm { M M S E } } = \mathsf { S N R } _ { \mathrm { M M S E , U } } + 1 .\tag{13}
$$

## 2.3 The Pythagorean interpretation

The identity also has the geometric interpretation emphasized by CDEF [7]. The forward representation

$$
Z = X + N ,
$$

with $X \perp N$ , forms one right triangle. The Wiener estimate

$$
\hat { X } = \alpha _ { \mathrm { W | U } } Z
$$

is the orthogonal projection of X onto the one-dimensional subspace spanned by $Z ,$ so that

$$
X = { \hat { X } } + E ,
$$

with $E \perp Z$ (and also $E \perp { \hat { X } } )$ . This is the second right triangle in CDEF’s Fig. 9; Forney develops the same projection geometry in his Hilbert-space treatment of MMSE estimation [8]. The +1 relation is therefore Pythagoras applied to the forward- and backward-channel decompositions. We identify in the sequel these rules within EDEN [5]. Namely, replacing random variables by fixed vectors and the mean-square inner product by the Euclidean inner product gives the same two scaling rules for every realization.

## 2.4 Why the forward-channel SNR is natural for averaging

The forward-channel realization (5) is the natural viewpoint when several quantized reconstructions are to be linearly combined as elaborated on in [9, 10, 11].<sup>4</sup> To that end, let us now relate the framework to the problem of averaging independently quantized elements of a vector $( X _ { 1 } , \ldots , X _ { m } )$ , where the elements are correlated. To further simplify the exposition assume the extreme case where all entries are identical; see Section II.B in [10]. Namely, we now identify

$$
Y _ { i } = Q _ { i } ( X _ { i } ) = Q _ { i } ( X )\tag{14}
$$

as the output of the i-th quantizer $Q _ { i } ( \cdot )$ . Ideally, we would like to conclude that

$$
Z _ { i } = X + N _ { i } , \qquad i = 1 , \dots , m ,
$$

where $Z _ { i } = \alpha _ { \mathrm { { U } } , i } Y _ { i }$ . Assume that the $N _ { i }$ are zero mean, have equal power $\mathbb { E } [ N _ { i } ^ { 2 } ] = P _ { N }$ , and satisfy $\mathbb { E } [ X N _ { i } ] = 0$ and $\mathbb { E } [ N _ { i } N _ { j } ] = 0$ for $i \neq j$ . Under these assumptions, their average is again a forward channel with unit signal gain,

$$
{ \overline { { Z } } } = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } Z _ { i } = X + { \overline { { N } } } ,\tag{15}
$$

where

$$
\overline { { { N } } } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } N _ { j } ,
$$

and

$$
\mathbb { E } \left[ \overline { { N } } ^ { 2 } \right] = \frac { 1 } { m ^ { 2 } } \sum _ { j = 1 } ^ { m } \mathbb { E } [ N _ { i } ^ { 2 } ] = P _ { N } / m .
$$

We therefore obtain

$$
\mathsf { S N R } _ { \mathrm { M M S E , U , a v e } } = m \mathsf { S N R } _ { \mathrm { M M S E , U } } .\tag{16}
$$

Thus, the forward-channel representation is natural for averaging, since the estimation errors behave as additive, zero-mean, mutually uncorrelated noises. We next discuss several mechanisms for realizing this model.

## 2.5 Comments on the Roles of the Random Rotate-Quantize-Rotate Back Pipeline

Classical subtractive dithering realizes this forward-channel model for a uniform quantizer without overload, using uniform dither over one quantization interval: the noise is independent of the input, and independent dithers make the noises in diferent descriptions independent [12].

In [9, 10], the random rotate–quantize–rotate-back (RQRB) pipeline was proposed as an alternative form of dither for scalar quantization, with the goal of asymptotically eliminating second-order error correlations in distributed quantization of correlated sources. Both Haar rotations and randomized Hadamard transforms are studied, with the latter providing a computationally eficient implementation. The sources in that work are already modeled as temporally i.i.d. Gaussian (e.g., following transform coding), so RQRB is not used for Gaussianization. Rather, diferent random rotations are applied across quantization branches, making the resulting quantization errors asymptotically uncorrelated (via random rotations) and uncorrelated from the source (via CDEF scaling); hence enabling noncoherent combining.

Shortly afterward, Suresh et al. [13] proposed using the RQRB pipeline in the closely related context of distributed mean estimation, with the random rotation playing a diferent role. A common randomized Hadamard transform is used across clients (branches in the terminology of [10]) to transform a deterministically modeled source vector so as to control the dynamic range of the elements to be fed into stochastic scalar quantizers. With independent stochastic-rounding randomness across clients, the errors are zero mean and pairwise uncorrelated, ensuring noncoherent combining.

In turn, EDEN [5] employs RQRB with an independent Haar rotation for each client and deterministic scalar quantization. The rotation makes the relevant empirical coordinate statistics of an arbitrary input vector asymptotically Gaussian, permitting scalar quantization optimized for a Gaussian source, while a realization-dependent scaling ensures unbiasedness in the strong sense (see Remark 1). Independence across client rotations is not needed for Gaussianization or single-client unbiasedness. It is used at the averaging stage: together with conditional unbiasedness, it makes the reconstruction errors conditionally independent and eliminates their cross correlations. Thus, in EDEN, the rotations serve both as a Gaussianizing transform and, in the second-order sense of [10], as dither enabling noncoherent combining.

## 3 EDEN Scaling as a Deterministic Realization-Dependent Counterpart of CDEF

Let $\mathbf { X } \in \mathbb { R } ^ { d }$ be a given non-zero vector and consider a realization of a Haar rotation U. The EDEN scheme forms the rotate–quantize–rotate-back observation Y before reconstruction scaling as

$$
\mathbf { Y } = \eta _ { \mathbf { X } } ^ { - 1 } U ^ { \mathsf { T } } Q ( \mathbf { V } ) ,
$$

where

$$
\eta \mathbf { x } = \frac { \sqrt { d } } { \| \mathbf { X } \| } ,
$$

and

$$
\mathbf { V } = \eta \mathbf { x } U \mathbf { X } .
$$

For this deterministic pair, define the correlation

$$
r _ { \mathbf { X Y } } ^ { ( d ) } = \frac { 1 } { d } \langle \mathbf { X } , \mathbf { Y } \rangle ,
$$

and normalized energies $\begin{array} { r } { P _ { \mathbf { X } } ^ { ( d ) } = \frac { 1 } { d } \| \mathbf { X } \| ^ { 2 } , P _ { \mathbf { Y } } ^ { ( d ) } = \frac { 1 } { d } \| \mathbf { Y } \| ^ { 2 } } \end{array}$

The two CDEF scalings of Section 2 then become

$$
\alpha _ { \mathrm { W } } = \alpha _ { \mathrm { W } } \Big ( r _ { \mathbf { X Y } } ^ { ( d ) } , P _ { \mathbf { Y } } ^ { ( d ) } \Big ) = \frac { \langle \mathbf { X } , \mathbf { Y } \rangle } { \| \mathbf { Y } \| ^ { 2 } } ,\tag{17}
$$

and

$$
\alpha _ { \mathrm { U } } = \alpha _ { \mathrm { U } } \Big ( r _ { \mathbf { X Y } } ^ { ( d ) } , P _ { \mathbf { X } } ^ { ( d ) } \Big ) = \frac { \| \mathbf { X } \| ^ { 2 } } { \langle \mathbf { X } , \mathbf { Y } \rangle } .\tag{18}
$$

Indeed,

$$
r _ { \mathbf { X } \mathbf { Y } } ^ { ( d ) } = P _ { \mathbf { X } } ^ { ( d ) } \left( { \frac { 1 } { d } } \sum _ { i = 1 } ^ { d } V _ { i } Q ( V _ { i } ) \right) ,
$$

and hence, when EDEN is expressed as a scaling of Y,

$$
\alpha _ { \mathrm { U } } = \frac { d } { \sum _ { i = 1 } ^ { d } V _ { i } Q ( V _ { i } ) } .\tag{19}
$$

Equivalently, the coeficient multiplying $U ^ { \mathsf { T } } Q ( \mathbf { V } )$ is $\eta _ { \mathbf { X } } ^ { - 1 } \alpha _ { \mathrm { U } } = \| \mathbf { X } \| ^ { 2 } / \langle U \mathbf { X } , Q ( \mathbf { V } ) \rangle$ , which is the scale used in the original EDEN formulation [5].

EDEN assumes that the reconstruction scale is represented without quantization error. It immediately follows that

$$
\begin{array} { r } { { \bf Z } = \alpha _ { \mathrm { U } } { \bf Y } = { \bf X } + { \bf N } , } \end{array}\tag{20}
$$

with $\mathbf { X } \perp \mathbf { N } .$ for every rotation realization, while $\widehat { \mathbf { X } } = \alpha \mathrm { w } \mathbf { Y }$ is the corresponding Euclidean projection.   
These are precisely the deterministic counterparts of the two CDEF scalings.

Treating U as random, Theorem 2.1 in [5] additionally proves the stronger finite-dimensional (unbiasedness) statement:

$$
\mathbb { E } _ { U } [ \mathbf { Z } ] = \mathbf { X } .\tag{21}
$$

Equivalently, if X is viewed as random, and independent of U, then $\mathbb { E } [ \mathbf { Z } \mid \mathbf { X } ] = \mathbf { X }$ . EDEN’s realizationdependent scaling enforces exact source–error orthogonality for every realization; rotational symmetry addi tionally yields exact conditional unbiasedness.

## 3.1 SNR Interpretation and Relation to the Stochastic Scalar Setting Define

$$
A ( U ) = \frac { \| \mathbf { X } \| ^ { 2 } \| \mathbf { Y } \| ^ { 2 } } { \langle \mathbf { X } , \mathbf { Y } \rangle ^ { 2 } } .
$$

The normalized errors of the unit-gain and Wiener reconstructions are given, respectively, by

$$
D _ { \mathrm { U } } ( U ) = \frac { \| \alpha _ { \mathrm { U } } { \bf Y } - { \bf X } \| ^ { 2 } } { \| { \bf X } \| ^ { 2 } } = A ( U ) - 1 ,\tag{22}
$$

and

$$
D _ { \mathrm { W } } ( U ) = \frac { \| \alpha _ { \mathrm { W } } { \bf Y } - { \bf X } \| ^ { 2 } } { \| { \bf X } \| ^ { 2 } } = 1 - \frac { 1 } { A ( U ) } .\tag{23}
$$

Note that, for every realization, they indeed satisfy the CDEF Pythagorean relation:

$$
\frac { 1 } { D _ { \mathrm { W } } ( U ) } = \frac { 1 } { D _ { \mathrm { U } } ( U ) } + 1 .
$$

Define the averaged distortions over U by $D _ { \mathrm { U } } ^ { ( d ) } = \mathbb { E } _ { U } [ D _ { \mathrm { U } } ( U ) ]$ and ${ D _ { \mathrm { W } } ^ { ( d ) } = \mathbb { E } _ { U } [ D _ { \mathrm { W } } ( U ) ] }$ . Define further the corresponding two SNRs by

$$
\mathsf { S N R } _ { \mathrm { M M S E , U } } ^ { ( d ) } = \frac { 1 } { D _ { \mathrm { U } } ^ { ( d ) } }
$$

and

$$
\mathsf { S N R } _ { \mathrm { M M S E } } ^ { ( d ) } = \frac { 1 } { D _ { \mathrm { W } } ^ { ( d ) } }
$$

Concavity of $t / ( 1 + t )$ implies that by Jensen:

$$
\mathsf { S N R } _ { \mathrm { M M S E } } ^ { ( d ) } \geq \mathsf { S N R } _ { \mathrm { M M S E , U } } ^ { ( d ) } + 1 .\tag{24}
$$

The large-d interpretation is as follows. For a Haar rotation,

$$
\mathbf { V } \triangleq \frac { \sqrt { d } \mathbf { G } } { \Vert \mathbf { G } \Vert } ,
$$

where

$$
\mathbf { G } \sim { \mathcal { N } } ( \mathbf { 0 } , I _ { d } ) .
$$

As d grows, $\| \mathbf G \| / \sqrt { d }$ concentrates around one, and the empirical quantities entering $A ( U )$ approach their scalar Gaussian counterparts. Indeed,

$$
A ( U ) = \frac { \frac { 1 } { d } \sum _ { i = 1 } ^ { d } Q ( V _ { i } ) ^ { 2 } } { \left( \frac { 1 } { d } \sum _ { i = 1 } ^ { d } V _ { i } Q ( V _ { i } ) \right) ^ { 2 } } .
$$

Hence, if $a = \mathbb { E } [ G Q ( G ) ] , b = \mathbb { E } [ Q ( G ) ^ { 2 } ]$ , and $G \sim \mathcal { N } ( 0 , 1 )$ , then $A ( U )$ approaches $b / a ^ { 2 }$

Consequently, the realization-dependent EDEN scalings approach the corresponding CDEF scalings for the scalar Gaussian model, and the Jensen gap vanishes. We therefore recover the CDEF relation asymptotically:

$$
\mathsf { S N R } _ { \mathrm { M M S E } } ^ { ( d ) } - \mathsf { S N R } _ { \mathrm { M M S E , U } } ^ { ( d ) } \longrightarrow 1 ,
$$

as $d \to \infty .$

## Acknowledgment

The author thanks Or Ordentlich for drawing attention to the EDEN line of work and, in particular, to its reconstruction scaling.

## References

[1] J. Chee, Y. Cai, V. Kuleshov, and C. M. De Sa, “QuIP: 2-bit quantization of large language models with guarantees,” Advances in neural information processing systems, vol. 36, pp. 4396–4429, 2023.

[2] S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, P. Cameron, M. Jaggi, D. Alistarh, T. Hoefler, and J. Hensman, “QuaRot: Outlier-free 4-bit inference in rotated LLMs,” Advances in Neural Information Processing Systems, vol. 37, pp. 100 213–100 240, 2024.

[3] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, “TurboQuant: Online vector quantization with near-optimal distortion rate,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 56 418–56 439.

[4] O. Ordentlich and Y. Polyanskiy, “High-rate quantized matrix multiplication I,” IEEE BITS the Information Theory Magazine, 2026.

[5] S. Vargaftik, R. B. Basat, A. Portnoy, G. Mendelson, Y. B. Itzhak, and M. Mitzenmacher, “EDEN: Communication-eficient and robust distributed mean estimation for federated learning,” in International Conference on Machine Learning. PMLR, 2022, pp. 21 984–22 014.

[6] R. Ben-Basat, Y. Ben-Itzhak, G. Mendelson, M. Mitzenmacher, A. Portnoy, and S. Vargaftik, “A note on TurboQuant and the earlier DRIVE/EDEN line of work,” arXiv preprint arXiv:2604.18555, 2026.

[7] J. M. Ciofi, G. P. Dudevoir, M. V. Eyuboglu, and G. D. Forney, “MMSE decision-feedback equalizers and coding. I. equalization results,” IEEE transactions on Communications, vol. 43, no. 10, pp. 2582– 2594, 1995.

[8] G. D. Forney Jr, “Shannon meets Wiener II: On MMSE estimation in successive decoding schemes,” arXiv preprint cs/0409011, 2004.

[9] R. Hadad, “Dithered quantization via orthogonal transformations and a Cauchy–Schwarz-like inequality,” Master’s thesis, Department of Electrical Engineering, Tel Aviv University, Tel Aviv, Israel, Jan. 2016. [Online]. Available: http://www.eng.tau.ac.il/∼uri/theses/hadad msc.pdf

[10] R. Hadad and U. Erez, “Dithered quantization via orthogonal transformations,” IEEE Transactions on Signal Processing, vol. 64, no. 22, pp. 5887–5900, 2016.

[11] J. Østergaard, U. Erez, and R. Zamir, “Incremental refinements and multiple descriptions with feedback,” IEEE Transactions on Information Theory, vol. 68, no. 10, pp. 6915–6940, 2022.

[12] R. M. Gray and D. L. Neuhof, “Quantization,” IEEE transactions on information theory, vol. 44, no. 6, pp. 2325–2383, 1998.

[13] A. T. Suresh, X. Y. Felix, S. Kumar, and H. B. McMahan, “Distributed mean estimation with limited communication,” in International conference on machine learning. PMLR, 2017, pp. 3329–3337.