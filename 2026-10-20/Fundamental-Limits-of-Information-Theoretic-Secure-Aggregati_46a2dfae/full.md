# Fundamental Limits of Information-Theoretic Secure Aggregation with User Dropouts in Federated Learning

Jinbao Zhu, Kai Wan, and Giuseppe Caire

## Abstract

Secure aggregation enables a server to compute the sum of users’ local computation results, such as gradients or model updates in federated learning, without revealing their individual values. When users may drop out during protocol execution, secure aggregation generally relies on correlated random keys to mask the local results while allowing the server to recover the sum of the surviving users’ local results. In this federated learning setting, a secure aggregation protocol consists of three phases: key distribution, masked-gradient upload, and mask elimination. In the key-distribution phase, users establish correlated random keys to protect their local gradients. Users then use these keys to mask their gradients before uploading them to the server. Because user dropouts may prevent the masks from canceling completely, the users provide additional information in the mask-elimination phase, enabling the server to remove the residual masks and recover the desired gradient sum. Existing information-theoretic studies typically design the mask-elimination communication for a prescribed dropout threshold and evaluate its cost under the worstcase combination of dropouts across the masked-gradient upload and mask-elimination phases. Such a characterization does not reveal how the required communication varies with the actual dropout scenario, and their schemes cannot exploit such variation to reduce communication. Moreover, the fundamental randomness and communication costs of establishing the correlated keys remain unexplored. Motivated by these limitations, we formulate an adaptive secure aggregation framework under the T-colluding security constraint that supports all feasible dropout scenarios and adapts its communication strategy to the actual dropout scenario observed during protocol execution. We develop an adaptive secure aggregation scheme and derive converse bounds on the randomness needed to generate correlated keys, the communication needed to distribute them, the communication for uploading masked gradients, and the additional communication required to eliminate residual masks. In particular, the latter two bounds are characterized separately for each dropout set realized during the masked-gradient upload phase. By comparing the achievable rates with the converse bounds, we show that the proposed scheme achieves the information-theoretically optimal masked-gradient communication rate for every feasible dropout scenario. It also achieves the optimal mask-elimination communication rate over almost all feasible dropout regimes. In addition, the achieved key rate and key-distribution communication rate are each either information-theoretically optimal or within a multiplicative factor of two of their respective optima. These results establish the first fundamental characterization of the communication and randomness resources required for secure aggregation with user dropouts.

## Index Terms

Federated learning, secure aggregation, information-theoretic security, user dropouts, fundamental limits.

## I. INTRODUCTION

Federated learning (FL) has emerged as a prominent distributed machine learning paradigm that enables multiple users to collaboratively train a global model under the coordination of a central server without directly sharing their raw data [1]–[4]. In a typical FL system, each user performs local training on its private data and uploads only the resulting gradients or model updates to the server, which aggregates them, typically through a linear operation such as summation or averaging, to update the global model for the next training iteration. By keeping raw training data on local devices, FL significantly reduces the risk of direct privacy leakage. However, recent studies on gradient inversion attacks have demonstrated that sensitive training data can still be reconstructed from the shared gradient [5]–[8]. Consequently, protecting individual gradients from disclosure is essential for preserving user privacy in FL

Secure aggregation [9]–[11] was introduced to provide such privacy protection by enabling the server to recover the aggregate of users’ gradients without revealing any individual gradient or compromising model accuracy, making it a key privacy-preserving primitive in federated learning systems. In practical deployments, however, secure aggregation protocols must also tolerate user dropouts, which occur when participating users become unavailable during protocol execution. Such dropouts may result from network failures, limited computational resources, battery depletion, or temporary disconnections [12]–[14]. Since both the number and identities of dropout users are unknown in advance and may vary across protocol executions, such uncertainty substantially complicates the design of efficient secure aggregation protocols that can handle all feasible dropout scenarios while preserving privacy guarantees.

To address this difficulty, secure aggregation schemes supporting user dropouts [9], [10], [15], [16] generally rely on correlated random keys held by the users. These keys serve two complementary purposes: each user’s key provides the randomness needed to mask its individual gradient, while the correlation among users’ keys provides the structure needed to eliminate the resulting masks during aggregation. To establish these correlated keys, users first communicate with one another. Each user then uses the established keys to mask its local gradient and uploads the resulting masked gradient to the server. When some users drop out, however, the resulting masks may not fully cancel, leaving a residual aggregate mask that prevents the server from directly recovering the desired aggregate. Consequently, additional information must be communicated to the server to eliminate the residual aggregate mask and complete the aggregation. These operations naturally divide the communication process into three phases: a key-distribution phase, a masked-gradient upload phase, and a mask-elimination phase. The key-distribution phase is performed as an offline setup before the aggregation begins and is therefore assumed to be completed while all users are available. In contrast, both the masked-gradient upload and mask-elimination phases occur during online aggregation and are affected by user dropouts. Dropouts during the masked-gradient upload phase determine which masked gradients are available for aggregation, whereas dropouts during the mask-elimination phase determine which users remain available to provide the information required to eliminate the residual aggregate mask. Characterizing the fundamental communication costs incurred in these phases, together with the randomness for establishing the correlated keys, has become a central problem in the information-theoretic study of secure aggregation with dropouts.

The information-theoretic secure aggregation problem with user dropouts was first investigated by Zhao and Sun [17] under the T-colluding security constraint, where the server aims to recover the sum of the local gradients of the surviving users while learning no information about the individual gradients of the non-colluding users beyond what is revealed by the desired sum and the gradients of the colluding users. Their dropout model assume that only users that successfully upload their masked gradients can participate in the subsequent mask-elimination phase; equivalently, the dropout set in the masked-gradient upload phase is required to be a subset of that in the mask-elimination phase. Under this dropout model, Zhao and Sun [17] characterized the optimal communication overheads of both online phases. Specifically, the communication overhead of the masked-gradient upload phase was characterized directly, whereas that of the mask-elimination phase was characterized by taking the worst case over the admissible dropout scenarios across the two online phases. They also constructed an achievable scheme, albeit under a key-distribution model that assumes the existence of a trusted third party. Subsequently, So et al. [18] proposed a lightweight secure aggregation scheme based on Shamir’s secret sharing [19] for correlated random key distribution, thereby removing the need for a trusted third party and reducing the required random key size. Further work by Wan et al. [20]–[22] studied the secure aggregation problem under the symmetric groupwise keydistribution model, in which every subset of users of the same size shares one uncoded common random key and all such shared keys have the same entropy size. Under this prescribed key-distribution structure, they characterized the optimal communication overheads of both online phases, with the mask-elimination communication evaluated under the worst-case dropout scenarios across the two phases. Beyond the secure aggregation problem with user dropouts, numerous variants of information-theoretic secure aggregation have been investigated, including secure aggregation without user dropouts [23], [24], hierarchical secure aggregation [25], [26], vector secure aggregation [27], [28], and sparse secure aggregation [29], [30]. Collectively, these works have substantially advanced the understanding of information-theoretic secure aggregation under various system models and privacy constraints.

Despite these advances, existing information-theoretic secure aggregation works with user dropouts [17], [18], [20]–[22] still suffer from three fundamental limitations.

• Existing secure aggregation schemes design the mask-elimination procedure for a predetermined dropout threshold. To guarantee successful aggregation under any dropout scenario up to this threshold, their maskelimination communication is determined by the dropout scenario associated with the prescribed threshold, rather than by the dropout scenario actually encountered during protocol execution. This design leads to a fundamental inefficiency. When the actual number of dropouts is smaller than the prescribed threshold, the communication capability of surviving users is underutilized, resulting in unnecessary communication overhead. On the other hand, if the actual number of dropouts exceeds the predetermined threshold, the aggregation process fails. Therefore, existing schemes cannot simultaneously guarantee robustness against all feasible dropout scenarios and achieve communication efficiency tailored to the actual number of dropout users during protocol execution. The benefit of adapting communication to the actual number of dropout users has been demonstrated in gradient coding [31]. In particular, Cao et al. [32] proposed an adaptive gradient coding scheme that achieves the optimal communication cost for each realized number of user dropouts under a given computation-load constraint. Although both gradient coding and secure aggregation aim to compute the sum of gradients in the presence of user dropouts, they address different challenges. Gradient coding relies on data allocation and redundant computation across users to recover the gradient sum despite user dropouts. In contrast, secure aggregation allows users to compute their gradients locally without sharing their private training data and relies on correlated random keys to protect individual gradients while enabling the recovery of the desired gradient sum. Consequently, adapting the mask-elimination communication to the actual dropout scenario while preserving information-theoretic security requires a different approach.

• Existing works of secure aggregation with user dropouts have primarily focused on characterizing the communication overheads of the masked-gradient upload and mask-elimination phases, while largely overlooking the resources required to establish correlated random keys. Although the fundamental limits of key randomness and key-distribution communication have been fully characterized for secure aggregation without user dropouts [23], [24], these results do not directly extend to the setting with user dropouts. This is because the correlated keys must be established before the identities of the surviving users are known, while ensuring both correctness and information-theoretic security under all feasible dropout scenarios. These additional requirements introduce new challenges in determining the minimum randomness and communication resources needed for key establishment. Consequently, the fundamental limits of key randomness and key-distribution communication for secure aggregation with user dropouts remain largely unexplored.

• Existing works characterize the communication overhead of the mask-elimination phase by taking the worst case over the dropout scenarios in both the masked-gradient upload and mask-elimination phases. Such a joint worst-case characterization overlooks the fact that the dropout set in the masked-gradient upload phase is already known when mask elimination phase begins and can therefore be used to adapt the subsequent communication. Characterizing the optimal mask-elimination communication overhead conditioned on each realized dropout set in the masked-gradient upload phase remains an open problem.

Motivated by these limitations, we revisit the T-colluding information-theoretic secure aggregation problem with user dropouts in a federated learning system consisting of N users and a central server. We consider a general dropout model in which U users may drop out during the masked-gradient upload phase and S users may drop out during the subsequent mask-elimination phase. Unlike prior dropout formulations [17], [18], [20]–[22], we do not impose a set-inclusion relation between the dropout sets in these two phases, allowing user availability to vary during protocol execution. Such variations in user availability may arise from intermittent connectivity, network delays, heterogeneous computation times, or temporary communication failures. For example, a user that does not participate in the masked-gradient upload phase may become available again during the subsequent mask-elimination phase. Under this model, we aim to characterize the fundamental randomness and communication costs of secure aggregation with user dropouts. In particular, we investigate the communication overheads of the masked-gradient upload and mask-elimination phases for each realized dropout set in the masked-gradient upload phase, rather than considering only the joint worst-case dropout scenarios across both phases. The main contributions of this paper are summarized as follows.

• We formulate an adaptive secure aggregation framework with user dropouts, in which a single scheme accommodates all feasible dropout scenarios and its communication strategy dynamically adapts to the actual dropout scenario encountered during protocol execution. We introduce four normalized performance measures: the key rate, key-distribution communication rate, masked-gradient communication rate, and mask-elimination communication rate. For each realized dropout set in the masked-gradient upload phase, we derive converse bounds on the masked-gradient communication rate and the mask-elimination communication rate. Moreover, we establish information-theoretic lower bounds on the key rate and key-distribution communication rate.

• We propose an adaptive secure aggregation scheme that simultaneously achieves a key rate of $N ( T + 1 )$ , a key-distribution communication rate of $N ( N - 1 )$ , a masked-gradient communication rate of $N - U$ , and a mask-elimination communication rate of $\frac { N - S } { N - S - T }$ for all feasible dropout scenarios satisfying $0 \leq U \leq N - 1$ and $0 \leq S \leq N - T - 1$

• By comparing the achievable rates with the converse bounds, we show that, for every dropout set U realized during the masked-gradient upload phase, the proposed scheme achieves the information-theoretically optimal masked-gradient communication rate. For each such dropout set U, it also achieves the optimal mask-elimination communication rate whenever $1 \leq S \leq N - T - 1$ or $( S = 0$ and $T < N - U )$ ). In these regimes, the optimal mask-elimination communication rate is independent of the dropout set realized during the masked-gradient upload phase. The optimal mask-elimination communication rate remains unknown in the remaining regime $S = 0$ and $T \geq N - U$ . Finally, the achieved key rate and key-distribution communication rate are each shown to be either optimal or within a multiplicative factor of two of their respective information-theoretic optima.

The remainder of this paper is organized as follows. Section II formulates the adaptive secure aggregation problem with user dropouts. Section III presents the main results, including the achievable rate region, the converse bounds, and the corresponding gap characterization. Section IV develops an adaptive secure aggregation scheme. Section V establishes the converse bounds. Section VI provides a detailed comparison with existing information-theoretic secure aggregation works. Finally, this paper is concluded in Section VII.

Notation: Let boldface capital letters denote matrices and calligraphic letters denote sets. The cardinality of a set U is denoted by |U|. For any non-negative integers m and n satisfying $m \leq n ,$ , let [n] and $[ m : n ]$ denote the sets $\{ 1 , 2 , \ldots , n \}$ and $\{ m , m + 1 , \ldots , n \}$ , respectively.

## II. PROBLEM FORMULATION

Consider a federated learning system consisting of a central server and N users, where each user $i \in [ N ]$ holds a private local gradient $W _ { i }$ , represented as a vector of L symbols drawn independently and uniformly from a finite field $\mathbb { F } _ { q }$ of size q, i.e.,

$$
H ( W _ { 1 } , \dots , W _ { N } ) = \sum _ { i \in [ N ] } H ( W _ { i } ) ,\tag{1}
$$

$$
H ( W _ { 1 } ) = \ldots = H ( W _ { N } ) = L .\tag{2}
$$

In the federated learning system, we assume the existence of error-free private channels among users and between each user and the server to facilitate information-theoretic privacy guarantees. The objective is to recover the sum of the users’ local gradients at the server in the presence of user dropouts, while guaranteeing information-theoretic privacy of each individual gradient against any collusion between the server and up to $T$ users. We consider a passive (honest-but-curious) adversary model, under which the server and colluding users follow the prescribed protocol but may combine all information available to them in an attempt to infer the non-colluding users’ gradients.

To this end, a secure aggregation scheme is divided into three phases: the key-distribution phase, the maskedgradient upload phase, and the mask-elimination phase. The details are presented below.

Key-Distribution Phase: Each user $i \in [ N ]$ locally generates a private random key $\mathcal { Z } _ { i }$ over $\mathbb { F } _ { q }$ and then shares an encoded version $\mathcal { Z } _ { i , j }$ of this key with every other user $j \in [ N ] \backslash \{ i \}$ , i.e.,

$$
H ( \mathcal Z _ { i , j } | \mathcal Z _ { i } ) = 0 , \quad \forall i \in [ N ] , j \in [ N ] \backslash \{ i \} .\tag{3}
$$

Since the users’ local random keys are generated independently of their gradient vectors, the resulting independence relationships can be stated as follows:

$$
H ( W _ { 1 } , \dots , W _ { N } , \mathcal { Z } _ { 1 } , \dots , \mathcal { Z } _ { N } ) = \sum _ { i \in [ N ] } H ( W _ { i } ) + \sum _ { i \in [ N ] } H ( \mathcal { Z } _ { i } ) .\tag{4}
$$

Masked-Gradient Upload Phase: Upon receiving the encoded keys $\{ \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \backslash \{ i \} }$ from all other users, user $i \in [ N ]$ generates a masked version $X _ { i }$ of its local gradient $W _ { i }$ using its local private key $\mathcal { Z } _ { i }$ together with the received encoded keys, i.e.,

$$
H ( X _ { i } | W _ { i } , { \mathcal { Z } } _ { i } , \{ { \mathcal { Z } } _ { j , i } \} _ { j \in [ N ] \backslash \{ i \} } ) = 0 , \quad \forall i \in [ N ] .\tag{5}
$$

User i then uploads the masked gradient $X _ { i }$ to the server. However, due to user dropouts, the server may only wait to receive masked gradients from a subset of the fastest users. Let $\mathcal { U } \subseteq [ N ]$ with size $0 \le U \le N - 1$ denote the set of dropout users in the masked-gradient upload phase, i.e., the users whose masked gradients are not received by the server by the prescribed upload deadline. Although both the identities of the dropouts U and their total number U are initially unknown, the server learns them upon completion of the masked-gradient upload phase. The server then receives the masked gradients $\{ X _ { i } \} _ { i \in [ N ] \backslash \mathcal { U } }$ and aims to recover the sum of the gradients $\begin{array} { r } { W _ { s u m } ^ { \mathcal { U } } \triangleq \sum _ { i \in [ N ] \backslash \mathcal { U } } W _ { i } } \end{array}$ of all surviving users $[ N ] \backslash \mathcal { U }$

Mask-Elimination Phase: To recover the desired gradient sum $W _ { s u m } ^ { \boldsymbol { U } }$ , the server broadcasts the identities of the dropout users U to all users. Upon receiving U, each user $i \in [ N ]$ stops transmitting the masked gradient $X _ { i }$ and instead generates an ordered sequence of M messages ${ \mathcal { V } } _ { i , 1 } ^ { \mathcal { U } } , { \mathcal { V } } _ { i , 2 } ^ { \mathcal { U } } , \dots , { \mathcal { V } } _ { i , M } ^ { \mathcal { U } }$ . These messages are transmitted to the server sequentially in their prescribed order until sufficient messages have been received to recover $W _ { s u m } ^ { \boldsymbol { U } } .$ , at which point the server broadcasts a stop signal to terminate any further transmission. These messages are deterministic functions of the information currently available to user i, i.e., for any $i \in [ N ]$

$$
H ( \mathcal { Y } _ { i , 1 } ^ { \mathcal { U } } , \mathcal { Y } _ { i , 2 } ^ { \mathcal { U } } , \dotsc , \mathcal { Y } _ { i , M } ^ { \mathcal { U } } | W _ { i } , \mathcal { Z } _ { i } , \{ \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \backslash \{ i \} } , \mathcal { U } ) = 0 .\tag{6}
$$

Due to user dropouts in the mask-elimination phase, the server may receive messages from only a subset of fastest users. More precisely, let $S \subseteq [ N ]$ with size $0 \leq S \leq S _ { m a x }$ denote the set of dropout users in the mask-elimination phase, $\mathrm { i . e . , }$ the users that fail to return all mask-elimination messages requested by the server before the phase terminates, where $S _ { m a x }$ denotes the maximum number of tolerated dropouts. We impose no inclusion relation between U and S. In particular, a user may drop out in one phase but participate in the other. For a realized dropou set S, the server collects the first $M _ { S }$ messages from each surviving user in $[ N ] \backslash S .$ . Here, $M _ { S }$ is a predetermined parameter specifying the number of messages required from each surviving user for successful decoding under S dropouts in the mask-elimination phase. Since S is unknown in advance, the server does not know beforehand which $M _ { S }$ will apply and therefore collects the ordered messages sequentially until sufficient messages have been received for decoding. To ensure communication efficiency, it is natural to require $M _ { 0 } \le M _ { 1 } \le \dots \le M _ { S _ { \mathrm { m a x } } }$ reflecting that fewer surviving users may require each of them to provide more messages for successful decoding. The secure aggregation scheme must satisfy the following two information-theoretic constraints.

• Correctness: It must be guaranteed that the server can decode the desired sum $W _ { s u m } ^ { \boldsymbol { U } }$ for all feasible dropout scenarios satisfying $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ and $S \subseteq [ N ]$ with size $0 \leq S \leq S _ { m a x }$ , i.e.,

$$
H ( W _ { s u m } ^ { \mathcal { U } } | \{ X _ { i } \} _ { i \in [ N ] \backslash \mathcal { U } } , \{ \mathcal { Y } _ { i , m } ^ { \mathcal { U } } \} _ { i \in [ N ] \backslash \mathcal { S } , m \in [ M _ { S } ] } ) = 0 .\tag{7}
$$

This constraint allows the server to recover the sum of the gradients of all surviving users $[ N ] \backslash \mathcal { U }$ in the masked-gradient upload phase, while adapting the mask-elimination communication to the realized dropout scenario in the subsequent phase. In particular, the number of mask-elimination messages required from each surviving user is adjusted according to the realized number of dropouts S. Together, these features ensure that the proposed scheme can dynamically adapt to the actual dropout scenario encountered during protocol execution.

• Security: The collusion between the server and any subset of up to $T$ users must not learn any additional information about the gradients of all users, beyond what is inherently revealed by the desired sum and the local gradients of the T colluding users. Formally, for any subset of colluding users $\mathcal { T } \subseteq [ N ]$ with $| \mathcal { T } | \leq T$ the following security constraint must hold for all feasible dropout scenarios satisfying $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ and $S \subseteq [ N ]$ with size $0 \leq S \leq S _ { m a x }$ :

$$
I ( \{ W _ { i } \} _ { i \in [ N ] } ; \mathsf { V i e w } _ { \mathscr { T } } ^ { \mathscr { U } } | W _ { s u m } ^ { \mathscr { U } } , \{ W _ { i } \} _ { i \in \mathscr { T } } ) = 0 ,\tag{8}
$$

where $\mathsf { V i e w } _ { T } ^ { \mathcal { U } } = \left\{ \{ \mathcal { Z } _ { i } \} _ { i \in T } , \{ \mathcal { Z } _ { i , j } , \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \setminus \{ i \} , i \in T } , \{ X _ { i } \} _ { i \in [ N ] } , \{ \mathcal { Y } _ { i , m } ^ { \mathcal { U } } \} _ { i \in [ N ] , m \in [ M ] } \right\}$ represents the joint view of the server and the colluding users in $\tau _ { \ast }$ , consisting of all information that may be available to them in the worst case during protocol execution.<sup>2</sup>

The performance of the secure aggregation scheme is measured by the following four normalized metrics:<sup>3</sup>

1. The key rate $R _ { Z }$ , defined as the number of key symbols required per desired aggregation symbol to ensure information-theoretic security, is given by

$$
R _ { Z } \triangleq \frac { \sum _ { i \in [ N ] } H ( \mathcal { Z } _ { i } ) } { H ( W _ { s u m } ^ { \mathcal { U } } ) } .\tag{9}
$$

2. The key-distribution communication rate $R _ { K }$ , defined as the number of transmitted key symbols per desired aggregation symbol during the key-distribution phase, is given by

$$
R _ { K } \triangleq \frac { \sum _ { i \in [ N ] } \sum _ { j \in [ N ] \backslash \{ i \} } H ( \mathcal { Z } _ { i , j } ) } { H ( W _ { s u m } ^ { \mathcal { U } } ) } .\tag{10}
$$

3. The masked-gradient communication rate $R _ { \mathcal { U } }$ , defined as the number of transmitted symbols per desired aggregation symbol under the dropout set U during the masked-gradient upload phase, is given by

$$
R _ { \mathcal { U } } = \frac { \displaystyle \sum _ { i \in [ N ] \backslash \mathcal { U } } H ( X _ { i } ) } { \displaystyle H ( W _ { s u m } ^ { \mathcal { U } } ) } , \quad \mathcal { U } \subseteq [ N ] , 0 \leq | \mathcal { U } | \leq N - 1 .\tag{11}
$$

4. The mask-elimination communication rate $R _ { S } ^ { \mathcal { U } }$ for any given dropout set $\mathcal { U } \subseteq [ N ]$ with $0 \leq | \mathcal { U } | \leq N - 1$ in the masked-gradient upload phase, defined as the worst-case number of transmitted symbols per desired aggregation symbol during the mask-elimination phase with S dropout users, is given by

$$
R _ { S } ^ { \mathcal { U } } = \frac { \underset { S \subseteq [ N ] , | S | = S } { \operatorname* { m a x } } \ \underset { i \in [ N ] \backslash S } { \sum } \ \underset { m \in [ M _ { S } ] } { \sum } H ( \mathcal { Y } _ { i , m } ^ { \mathcal { U } } ) } { H ( W _ { s u m } ^ { \mathcal { U } } ) } , \quad 0 \leq S \leq S _ { m a x } .\tag{12}
$$

In Section $\mathrm { V } { \cdot } \mathrm { A } ,$ , we show that under the correctness and security constraints in (7) and (8), the maximum number of dropouts that can be tolerated during the mask-elimination phase is $N - T - 1$ for every feasible security parameter T, regardless of the dropout set U during the masked-gradient upload phase. Therefore, throughout this paper, we focus on the nontrivial regime $0 \le S \le N - T - 1$ and set $S _ { m a x } = N - T - 1$ , so that the secure aggregation scheme is robust against all feasible dropout scenarios.

Definition 1 (Achievable Schemes and Rate Tuples for Secure Aggregation with User Dropouts). A secure aggregation scheme is said to be achievable if it satisfies the correctness constraint (7) and the security constraint (8) for all feasible dropout scenarios specified by $\mathcal { U } \subseteq [ N ]$ with $0 \leq U \leq N - 1$ and $S \subseteq [ N ]$ with $0 \leq S \leq N - T - 1$ Furthermore, for any given dropout set $\textit { u } \subseteq [ N ]$ of size $0 \le U \le N - 1$ and any given dropout parameter $0 \leq S \leq N - T - 1$ , the rate tuple $( R _ { Z } , R _ { K } , R _ { { \mathcal U } } , R _ { S } ^ { \mathcal { U } } )$ is said to be achievable if there exists an adaptive secure aggregation scheme whose key rate and communication rates do not exceed the prescribed values $R _ { Z } , R _ { K } , R _ { { \mathcal { U } } } ,$ and $R _ { S } ^ { \mathcal { U } } ;$ , respectively, under the dropout scenario in which the dropout set during the masked-gradient upload phase is U and the number of dropout users during the mask-elimination phase is S.

Let $R _ { Z } ^ { * } , R _ { K } ^ { * } , R _ { U } ^ { * }$ , and $R _ { S } ^ { { \cal U } * }$ denote the optimal key rate, key-distribution communication rate, masked-gradient communication rate, and mask-elimination communication rate, respectively, each defined as the infimum of the corresponding rate over all achievable rate tuples. The objective of this paper is to characterize the fundamental limits of these optimal rates and to develop adaptive secure aggregation schemes that achieve these limits as closely as possible across all feasible dropout scenarios.

## III. MAIN RESULTS

The main results of this paper are summarized in the following three theorems. Theorem 1 establishes an achievable rate region for the secure aggregation problem with user dropouts. Theorem 2 provides information theoretic lower bounds on the optimal rates. Finally, Theorem 3 characterizes the multiplicative gap between the achievable rates and the information-theoretic optima. Throughout this paper, we focus on the nontrivial parameter regime $N > 1$ and $0 \le T < N - 1 . ^ { 4 }$

Theorem 1. For the secure aggregation problem with user dropouts and any system parameters $N > 1$ and $0 \leq T < N - 1$ , there exists a secure aggregation scheme that achieves the rate tuple

$$
\left( R _ { Z } , R _ { K } , R _ { \mathcal { U } } , R _ { S } ^ { \mathcal { U } } \right) = \left( N ( T + 1 ) , N ( N - 1 ) , N - U , \frac { N - S } { N - T - S } \right)
$$

for every feasible dropout scenario encountered during protocol execution, i.e., for every dropout set $\mathcal { U } \subseteq [ N ]$ of size $0 \leq U \leq N - 1$ during the masked-gradient upload phase and every feasible number $0 \leq S \leq N - T - 1$ of dropouts during the mask-elimination phase. Furthermore, the following rate region is achievable:

$$
\mathcal { R } _ { M , S } = \left\{ ( R _ { Z } , R _ { K } , R _ { M } , R _ { S } ^ { U } ) \in \mathbb { R } ^ { 4 } : R _ { Z } \geq N ( T + 1 ) , R _ { K } \geq N ( N - 1 ) , R _ { \mathcal { U } } \geq N - U , R _ { S } ^ { U } \geq \frac { N - S } { N - T - S } \right\} .
$$

The achievability of Theorem 1 is established in Section IV through the construction of an adaptive secure aggregation scheme that achieves the rate tuple $\begin{array} { r } { \left( N ( T + 1 ) , N ( N - 1 ) , N - U , \frac { N - S } { N - T - S } \right) } \end{array}$ for every feasible dropout scenario satisfying $0 \le U \le N - 1$ and $0 \le S \le N - T - 1$ . We next establish converse bounds on the four rates and then compare them with the achievable rates to characterize the optimality and multiplicative gaps of the secure aggregation scheme established in Theorem 1.

Theorem 2. For the secure aggregation problem with user dropouts and any system parameters $N > 1$ and $0 \leq T < N - 1$ , the optimal rate tuple $( R _ { Z } ^ { * } , R _ { K } ^ { * } , R _ { { Z } } ^ { * } , R _ { { S } } ^ { { \mathcal { U } } * } )$ satisfies the following lower bounds for every dropout set $\textit { u } \subseteq [ N ]$ of size $0 \le U \le N - 1$ during the masked-gradient upload phase and every feasible number $0 \leq S \leq N - T - 1$ of dropouts during the mask-elimination phase:

$$
\left\{ \begin{array} { l l } { R _ { Z } ^ { * } \geq \left\{ \begin{array} { l l } { \frac { N ( N - 1 ) ( N - T ) } { 2 ( N - T - 1 ) } , } & { i f \ T + 1 < N < 2 T + 1 } \\ { N ( T + 1 ) , } & { i f \ N \geq 2 T + 1 } \end{array} \right. } \\ { R _ { K } ^ { * } \geq \frac { N ( N - 1 ) ( N - T ) } { 2 ( N - T - 1 ) } , } & { f o r \ a l l f e a s i b l e \ p a r a m e t e r s } \\ { ( N - 1 ) R _ { Z } ^ { * } + ( N - 2 T - 1 ) R _ { K } ^ { * } \geq N ( N - 1 ) ( N - T ) , } & { i f \ N \geq 2 T + 1 } \\ { R _ { U } ^ { * } \geq N - U , } & { f o r \ a l l f e a s i b l e \ p a r a m e t e r s } \\ { R _ { S } ^ { l * } \geq \frac { N - S } { N - S - T } , } & { i f \ 1 \leq S \leq N - T - 1 \ o r \ \big ( S = 0 \ a n d \ T < N - U \big ) } \end{array} \right.
$$

The proof of Theorem 2 is provided in Section V. Comparing these lower bounds with the achievable rate presented in Theorem 1 immediately yields the following characterization of the multiplicative gaps.

Theorem 3. For the secure aggregation scheme, which achieves the rate tuple $\left( N ( T + 1 ) , N ( N - 1 ) , N - U , \frac { N - S } { N - T - S } \right)$ for every dropout set $\mathcal { U } \subseteq [ N ]$ of size $0 \leq U \leq N - 1$ during the masked-gradient upload phase and every feasible number $0 \leq S \leq N - T - 1$ of dropouts during the mask-elimination phase, the multiplicative gaps between the four achievable rates and their respective information-theoretic optima are characterized as follows:

$$
\left\{ \begin{array} { l l } { \frac { R _ { z } } { R _ { z } } < 2 , } & { i f T + 1 < N < 2 T + 1 } \\ { \frac { R _ { z } } { R _ { z } } = 1 , } & { i f \ N \geq 2 T + 1 } \\ { \frac { R _ { x } } { R _ { x } ^ { * } } < 2 , } & { i f \ 0 \leq T < N - 2 } \\ { \frac { R _ { x } } { R _ { x } ^ { * } } = 1 , } & { i f \ T = N - 2 } \\ { \frac { R _ { u } } { R _ { u } ^ { * } } = 1 , } & { f o r \ a l l \ f e a s i b l e \ p a r a m e t e r s } \\ { \frac { R _ { x } ^ { * } } { R _ { x } ^ { * } } = 1 , } & { i f \ 1 \leq S \leq N - T - 1 \ o r \ \left( S = 0 \ a n d \ T < N - U \right) } \end{array} \right.
$$

In summary, for each given dropout set realized during the masked-gradient upload phase, the secure aggregation scheme established in Theorem 1 achieves the optimal masked-gradient communication rate for all feasible parameters. It also achieves the optimal mask-elimination communication rate whenever $1 \le S \le N - T - 1$ or $( S = 0 \mathrm { \ a n d \ } T < N - U )$ . In these regimes, the optimal mask-elimination communication rate is $\frac { N - S } { N - T - S }$ , which is independent of both the identity and the number of dropout users during the masked-gradient upload phase. When $S = 0$ and $T \geq N - U ,$ , the optimality of the mask-elimination communication rate remains open. Moreover, the key rate is optimal whenever $N \geq 2 T + 1$ and is within a factor of two of the information-theoretic optimum when $T + 1 < N < 2 T + 1$ . The key-distribution communication rate is within a factor of two of the optimum for all $0 \leq T < N - 2$ and is optimal when $T = N - 2$ . More specifically, Fig. 1 provides a graphical illustration of the achievable and converse regions under two representative parameter settings, namely $( N , T ) = ( 6 , 3 )$ and $( N , T ) = ( 1 0 , 3 )$ , corresponding to the regimes $T + 1 < N < 2 T + 1$ and $N \geq 2 T + 1$ , respectively. Since the achievable and converse bounds coincide for the masked-gradient communication rate and the mask-elimination communication rate in almost all feasible parameters, the figure focuses exclusively on the tradeoff between the key rate $R _ { Z }$ and the key-distribution communication rate $R _ { K }$ . The blue region represents the achievable rate region established in Theorem 1, while the green region corresponds to rate tuples ruled out by the converse bounds in

Theorem 2. The remaining red region represents the unresolved gap between the achievable and converse regions.   
As characterized in Theorem 3, this gap is at most a factor of two.

![](images/4b3d2e317dd8ac1d2acc6234e78ef0de363bd33a012c5b0c6e03fc390fe1b426.jpg)  
(a) $( N , T ) = ( 6 , 3 )$

![](images/a378f1ff9e1a4723094d4aabfdd84fece9a41df47581d81a18c69c28085590c2.jpg)  
(b) $( N , T ) = ( 1 0 , 3 )$  
Fig. 1. Achievable and converse regions for the key rate $R _ { Z }$ and the key-distribution communication rate $R _ { K }$ under two representative parameter settings: $( N , T ) = ( 6 , 3 )$ and $( N , T ) = ( 1 0 , 3 )$ . The blue region is achievable, whereas the green region is ruled out by the converse bounds. The remaining red region represents a gap of at most a factor of two.

Remark 1. For the T-colluding secure aggregation problem without user dropouts $( \mathrm { i . e . , } U = 0 )$ , references [23], [24] showed that the desired gradient sum can be recovered by having each user upload only one masked gradient, after which the server sums all received masked gradients to obtain it, without requiring any additional mask-elimination communication. This naturally raises the following question: For the T-colluding secure aggregation problem with user dropouts, can one design an adaptive secure aggregation scheme that requires no mask-elimination messages when $U = 0$ and invokes the mask-elimination phase only when $U > 0 ?$ Our converse result answers this question in the negative. Specifically, the information-theoretic lower bounds in Theorem 2 show that $R _ { \mathcal { U } } ^ { * } \ge N - U$ and $\begin{array} { r } { R _ { S } ^ { \mathcal { U } * } \geq \frac { N - S } { N - T - S } } \end{array}$ for each given dropout set $\mathcal { U } \subseteq [ N ]$ of size $0 \leq U \leq N - 1$ during the masked-gradient upload phase. Since $R _ { S } ^ { { \cal U } * }$ remains strictly positive even when $U = 0 ,$ , the mask-elimination phase is information-theoretically necessary, regardless of whether any dropout actually occurs in the masked-gradient upload phase.

## IV. PROOF OF THEOREM 1: ACHIEVABILITY

In this section, we develop an adaptive secure aggregation scheme that achieves the rate tuple $( N ( T + 1 ) , N ( N -$ 1), $\begin{array} { r } { N - U , \frac { N - S } { N - T - S } ) } \end{array}$ for all feasible dropout scenarios. The key idea is to establish correlated random keys through a layered polynomial encoding structure that enables the mask-elimination communication to adapt to the actual number of dropouts. Specifically, each user generates a random mask of length L to protect its local gradient, together with T L additional random symbols to securely distribute the random mask among users while protecting it against up to $T$ colluding users, resulting in a key rate of $N ( T + 1 )$ . The random mask is then organized into $N - T$ matrices, each representing an encoding layer. In particular, each subsequent layer is constructed by reusing selected symbols from the preceding layers, such that the first m layers collectively enable the recovery of the aggregate mask in the presence of $m - 1$ dropouts during the mask-elimination phase, for every $m \in [ N - T ]$ . The layers are securely encoded into polynomials, whose evaluations are distributed among the users during the keydistribution phase. Each user distributes a total of L encoded symbols to every other user, yielding a key-distribution communication rate of $N ( N - 1 )$ ). During the masked-gradient upload phase, each user uploads a masked gradient of length L. When U users drop out, the server receives $N - U$ masked gradients, resulting in a communication rate of $N - U$ . After the server identifies the dropout set $u ,$ the users generate an ordered sequence of $N - T$ mask-elimination messages, each constructed from the encoded keys corresponding to one of the $N - T$ encoding layers. Rather than transmitting all these messages according to a prescribed worst-case dropout threshold, each user transmits them sequentially until the server has received sufficient information to recover the aggregate mask. When $S$ users drop out during the mask-elimination phase, the server only needs to collect the first $S + 1$ messages from each of the $N - S$ surviving users. The server first recovers the information encoded in the last received layer and then uses the recovered information to successively decode the preceding layers through polynomial interpolation, thereby reconstructing the aggregate mask and recovering the desired gradient sum. By construction, the first $S + 1$ messages contain a total of $\frac { L } { N - T - S }$ symbols per surviving user, yielding a mask-elimination communication rate of $\frac { N - S } { N - T - S }$ . Consequently, the established correlated keys can support all feasible dropout scenarios, while the mask-elimination communication dynamically adapts to the actual number of dropouts.

To illustrate the underlying construction, we first present a detailed example.

## A. Illustrative Example

In this example, we consider the secure aggregation problem with system parameters $N = 5$ and $T = 2$ , where the server aims to sum the gradients of all surviving users, while guaranteeing the privacy of each individual gradient against any collusion between the server and up to $T = 2$ users. To ensure that the proposed scheme can dynamically tolerate all feasible dropouts, we require that $S _ { m a x } = N - T - 1 = 2 , M = S _ { m a x } + 1 = 3 .$ , and $L = 6$

Key-Distribution Phase: Each user $i \in [ 5 ]$ locally generates a random key vector of length $L = 6$ over the finite field $\mathbb { F } _ { q } .$ given by

$$
C _ { i } = \left( C _ { i , 1 } , C _ { i , 2 } , C _ { i , 3 } , C _ { i , 4 } , C _ { i , 5 } , C _ { i , 6 } \right) , \quad \forall i \in [ 5 ] .\tag{13}
$$

The key vector $C _ { i }$ can be carefully rearranged into $M = 3$ matrices $\mathbf { P } ^ { ( i , 1 ) } , \mathbf { P } ^ { ( i , 2 ) }$ , and $\mathbf { P } ^ { ( i , 3 ) }$ , given by

$$
\mathbf { P } ^ { ( i , 1 ) } = \left[ \begin{array} { l l } { C _ { i , 1 } } & { C _ { i , 4 } } \\ { C _ { i , 2 } } & { C _ { i , 5 } } \\ { C _ { i , 3 } } & { C _ { i , 6 } } \end{array} \right] , \quad \mathbf { P } ^ { ( i , 2 ) } = \left[ \begin{array} { l } { C _ { i , 3 } } \\ { C _ { i , 6 } } \end{array} \right] , \quad \mathbf { P } ^ { ( i , 3 ) } = \left[ \begin{array} { l l l } { C _ { i , 2 } } & { C _ { i , 5 } } & { C _ { i , 6 } } \end{array} \right] .
$$

Then, for each $m = 1 , 2 , 3$ , the matrix $\mathbf { P } ^ { ( i , m ) }$ is secretly shared with all other users to ensure robustness against the presence of $S = m - 1$ dropouts during the subsequent mask-elimination phase. More precisely, based on these

three matrices, user $i \in [ 5 ]$ constructs the following secure encoding polynomials:

$$
f _ { 1 } ^ { ( i , 1 ) } ( x ) = D _ { 1 , 1 } ^ { ( i , 1 ) } + D _ { 2 , 1 } ^ { ( i , 1 ) } x + C _ { i , 1 } x ^ { 2 } + C _ { i , 2 } x ^ { 3 } + C _ { i , 3 } x ^ { 4 } ,\tag{14}
$$

$$
f _ { 2 } ^ { ( i , 1 ) } ( x ) = D _ { 1 , 2 } ^ { ( i , 1 ) } + D _ { 2 , 2 } ^ { ( i , 1 ) } x + C _ { i , 4 } x ^ { 2 } + C _ { i , 5 } x ^ { 3 } + C _ { i , 6 } x ^ { 4 } ,\tag{15}
$$

$$
f _ { 1 } ^ { ( i , 2 ) } ( x ) = D _ { 1 , 1 } ^ { ( i , 2 ) } + D _ { 2 , 1 } ^ { ( i , 2 ) } x + C _ { i , 3 } x ^ { 2 } + C _ { i , 6 } x ^ { 3 } ,\tag{16}
$$

$$
f _ { 1 } ^ { ( i , 3 ) } ( x ) = D _ { 1 , 1 } ^ { ( i , 3 ) } + D _ { 2 , 1 } ^ { ( i , 3 ) } x + C _ { i , 2 } x ^ { 2 } ,\tag{17}
$$

$$
f _ { 2 } ^ { ( i , 3 ) } ( x ) = D _ { 1 , 2 } ^ { ( i , 3 ) } + D _ { 2 , 2 } ^ { ( i , 3 ) } x + C _ { i , 5 } x ^ { 2 } ,\tag{18}
$$

$$
f _ { 3 } ^ { ( i , 3 ) } ( x ) = D _ { 1 , 3 } ^ { ( i , 3 ) } + D _ { 2 , 3 } ^ { ( i , 3 ) } x + C _ { i , 6 } x ^ { 2 } ,\tag{19}
$$

where $D _ { 1 , 1 } ^ { ( i , 1 ) } , \ldots , D _ { 2 , 3 } ^ { ( i , 3 ) }$ are independent random noise symbols locally generated by user i, which are used to guarantee the privacy of the key vector $C _ { i }$ against any $T = 2$ colluding users. Accordingly, the private random key $\mathcal { Z } _ { i }$ required by user i is given by

$$
\mathcal { Z } _ { i } = \left\{ C _ { i } , \{ D _ { k , \ell } ^ { ( i , 1 ) } \} _ { k \in [ 2 ] , \ell \in [ 2 ] } , \{ D _ { k , 1 } ^ { ( i , 2 ) } \} _ { k \in [ 2 ] } , \{ D _ { k , \ell } ^ { ( i , 3 ) } \} _ { k \in [ 2 ] , \ell \in [ 3 ] } \right\} , \quad \forall i \in [ 5 ] .
$$

Let $\alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } , \alpha _ { 4 } , \alpha _ { 5 }$ be distinct elements in $\mathbb { F } _ { q }$ . For any $i , j \in [ 5 ]$ , the encoded key $\mathcal { Z } _ { i , j }$ is obtained by evaluating these secure encoding polynomials at $x = \alpha _ { j }$ , i.e.,

$$
\mathcal { Z } _ { i , j } = \left\{ f _ { 1 } ^ { ( i , 1 ) } ( \alpha _ { j } ) , f _ { 2 } ^ { ( i , 1 ) } ( \alpha _ { j } ) , f _ { 1 } ^ { ( i , 2 ) } ( \alpha _ { j } ) , f _ { 1 } ^ { ( i , 3 ) } ( \alpha _ { j } ) , f _ { 2 } ^ { ( i , 3 ) } ( \alpha _ { j } ) , f _ { 3 } ^ { ( i , 3 ) } ( \alpha _ { j } ) \right\} .
$$

Then, user $i \in [ 5 ]$ shares the encoded key $\mathcal { Z } _ { i , j }$ with each other user $j \in [ 5 ] \backslash \{ i \}$

Masked-Gradient Upload Phase: Without loss of generality, assume that the local gradient $W _ { i }$ of each user $i \in [ 5 ]$ is represented as a row vector. User i locally computes a masked gradient $X _ { i } = W _ { i } + C _ { i }$ using the random key $C _ { i }$ and then uploads it to the server. Suppose that users 3 and 4 drop out during this phase, i.e., the dropout set is $\mathcal { U } = \{ 3 , 4 \}$ }. Consequently, the server receives only the masked gradients $\{ X _ { 1 } , X _ { 2 } , X _ { 5 } \}$ from the surviving users and seeks to recover the desired gradient sum $W _ { 1 } + W _ { 2 } + W _ { 5 }$ . To this end, the server computes the sum of the received masked gradients as

$$
X _ { 1 } + X _ { 2 } + X _ { 5 } = ( W _ { 1 } + W _ { 2 } + W _ { 5 } ) + ( C _ { 1 } + C _ { 2 } + C _ { 5 } ) .
$$

Therefore, recovering the desired sum $W _ { 1 } + W _ { 2 } + W _ { 5 }$ reduces to eliminating the aggregate mask $C _ { 1 } + C _ { 2 } + C _ { 5 }$ Mask-Elimination Phase: To eliminate the aggregate mask, the server broadcasts the dropout set $\mathcal { U } = \{ 3 , 4 \}$ to all users. Upon receiving $u ,$ each user $i \in [ 5 ]$ generates and sends the $M = 3$ messages $\mathcal { V } _ { i , 1 } ^ { \{ 3 , 4 \} } , \mathcal { V } _ { i , 2 } ^ { \{ 3 , 4 \} } , \mathcal { V } _ { i , 3 } ^ { \{ 3 , 4 \} }$ in the given order, based on the locally available encoding keys $\{ \mathcal { Z } _ { j , i } \} _ { j \in \{ 1 , 2 , 5 \} }$ , given by

$$
\begin{array} { r } { \mathcal { V } _ { i , 1 } ^ { \{ 3 , 4 \} } = \left\{ { \begin{array} { l } { f _ { 1 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) } \\ { f _ { 2 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) } \end{array} } \right\} , } \end{array}
$$

$$
\mathcal { V } _ { i , 2 } ^ { \{ 3 , 4 \} } = \left\{ f _ { 1 } ^ { ( 1 , 2 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 2 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 2 ) } ( \alpha _ { i } ) \right\} ,
$$

$$
\begin{array} { r } { \mathcal { V } _ { i , 3 } ^ { \{ 3 , 4 \} } = \left\{ \begin{array} { l l } { f _ { 1 } ^ { ( 1 , 3 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 3 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 3 ) } ( \alpha _ { i } ) } \\ { f _ { 2 } ^ { ( 1 , 3 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 2 , 3 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 5 , 3 ) } ( \alpha _ { i } ) } \\ { f _ { 3 } ^ { ( 1 , 3 ) } ( \alpha _ { i } ) + f _ { 3 } ^ { ( 2 , 3 ) } ( \alpha _ { i } ) + f _ { 3 } ^ { ( 5 , 3 ) } ( \alpha _ { i } ) } \end{array} \right\} . } \end{array}
$$

The proposed secure aggregation scheme can dynamically tolerate the presence of $S = 0 , 1 , 2$ dropouts during the mask-elimination phase. Specifically, when exactly S users drop out for all $S = 0 , 1 , 2$ , the server can recover the desired gradient sum by collecting the first $M _ { S } = S + 1$ messages from each user in any set of $5 - S$ surviving users. The corresponding decoding procedures are described below.

In the presence of $S = 0$ dropouts, the server waits for the first message from each of users 1, 2, 3, 4, and 5. It suffices to show that the aggregate mask $C _ { 1 } + C _ { 2 } + C _ { 5 }$ can be decoded from the following received messages:

$$
\left\{ \mathcal { V } _ { i , 1 } ^ { \{ 3 , 4 \} } \right\} _ { i \in [ 5 ] } = \left\{ \begin{array} { l l } { f _ { 1 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) } \\  f _ { 2 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) \rule { 0 ex } { 4 ex } \right\} _ { i \in [ 5 ] } . \end{array}
$$

Since the polynomial $f _ { 1 } ^ { ( 1 , 1 ) } ( x ) + f _ { 1 } ^ { ( 2 , 1 ) } ( x ) + f _ { 1 } ^ { ( 5 , 1 ) } ( x )$ has degree 4 by (14), the server can interpolate this polynomial from the received data $\{ f _ { 1 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) \} _ { i \in [ 5 ] }$ . The coefficients of the interpolated polynomial then yield the sums $\{ C _ { 1 , 1 } + C _ { 2 , 1 } + C _ { 5 , 1 } , C _ { 1 , 2 } + C _ { 2 , 2 } + C _ { 5 , 2 } , C _ { 1 , 3 } + C _ { 2 , 3 } + C _ { 5 , 3 } \}$ . Similarly, by (15), the server can recover the sums $\{ C _ { 1 , 4 } + C _ { 2 , 4 } + C _ { 5 , 4 } , C _ { 1 , 5 } + C _ { 2 , 5 } + C _ { 5 , 5 } , C _ { 1 , 6 } + C _ { 2 , 6 } + C _ { 5 , 6 } \}$ from the received data $\{ f _ { 2 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) \} _ { i \in [ 5 ] }$ . Therefore, the aggregate mask $C _ { 1 } + C _ { 2 } + C _ { 5 }$ can be recovered according to (13).

In the presence of $S = 1$ dropouts, without loss of generality, assume that user 5 drops out. The server waits for the first two messages from each of users 1, 2, 3, and 4. It remains to show that the aggregate mask $C _ { 1 } + C _ { 2 } + C _ { 5 }$ can be recovered from the following received messages:

$$
\left\{ \mathcal { V } _ { i , 1 } ^ { \{ 3 , 4 \} } \right\} _ { i \in [ 4 ] } = \left\{ \begin{array} { l l } { f _ { 1 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) } \\ { f _ { 2 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) } \\ { f _ { 1 } ^ { ( 1 , 2 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 2 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 2 ) } ( \alpha _ { i } ) } \end{array} \right\} _ { i \in [ 4 ] } .
$$

From (16), the polynomial $f _ { 1 } ^ { ( 1 , 2 ) } ( x ) + f _ { 1 } ^ { ( 2 , 2 ) } ( x ) + f _ { 1 } ^ { ( 5 , 2 ) } ( x )$ of degree 3 can be interpolated from the received data $\{ f _ { 1 } ^ { ( 1 , 2 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 2 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 2 ) } ( \alpha _ { i } ) \} _ { i \in [ 4 ] }$ , thereby allowing the server to recover the sums $C _ { 1 , 3 } + C _ { 2 , 3 } + C _ { 5 , 3 }$ and $C _ { 1 , 6 } + C _ { 2 , 6 } + C _ { 5 , 6 }$ . Furthermore, the polynomial $f _ { 1 } ^ { ( 1 , 1 ) } ( x ) + f _ { 1 } ^ { ( 2 , 1 ) } ( x ) + f _ { 1 } ^ { ( 5 , 1 ) } ( x )$ (resp. $f _ { 2 } ^ { ( 1 , 1 ) } ( x ) + f _ { 2 } ^ { ( 2 , 1 ) } ( x ) +$ $f _ { 2 } ^ { ( 5 , 1 ) } ( x ) )$ of degree 4 can be interpolated from the received data $\{ f _ { 1 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) \} _ { i \in [ 4 ] }$ (resp. $\{ f _ { 2 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) \} _ { i \in [ 4 ] } )$ along with the previously recovered sums $C _ { 1 , 3 } + C _ { 2 , 3 } + C _ { 5 , 3 }$ (resp. $C _ { 1 , 6 } + C _ { 2 , 6 } + C _ { 5 , 6 } )$ . Therefore, the aggregate mask $C _ { 1 } + C _ { 2 } + C _ { 5 }$ can be recovered by (14)-(15).

In the presence of $S = 2$ dropouts, without loss of generality, assume that users 4 and 5 drop out. The server wait for the first three message from each of users 1, 2, and 3. It remains to show that the aggregate mask $C _ { 1 } + C _ { 2 } + C _ { 5 }$

can be decoded from the following received messages:

$$
\left\{ \begin{array} { l l } { y _ { i , 1 } ^ { ( 3 , 4 ) } } \\ { y _ { i , 1 } ^ { ( 3 , 4 ) } } \\ { y _ { i , 2 } ^ { ( 3 , 4 ) } } \\ { y _ { i , 3 } ^ { ( 3 , 4 ) } } \end{array} \right\} = \left\{ \begin{array} { l l } { f _ { 1 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) } \\ { f _ { 2 } ^ { ( 1 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 2 , 1 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 5 , 1 ) } ( \alpha _ { i } ) } \\ { f _ { 1 } ^ { ( 1 , 2 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 2 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 2 ) } ( \alpha _ { i } ) } \\ { f _ { 1 } ^ { ( 1 , 3 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 2 , 3 ) } ( \alpha _ { i } ) + f _ { 1 } ^ { ( 5 , 3 ) } ( \alpha _ { i } ) } \\ { f _ { 2 } ^ { ( 1 , 3 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 2 , 3 ) } ( \alpha _ { i } ) + f _ { 2 } ^ { ( 5 , 3 ) } ( \alpha _ { i } ) } \\ { f _ { 3 } ^ { ( 1 , 3 ) } ( \alpha _ { i } ) + f _ { 3 } ^ { ( 2 , 3 ) } ( \alpha _ { i } ) + f _ { 3 } ^ { ( 5 , 3 ) } ( \alpha _ { i } ) } \\ { f _ { 4 } ^ { ( 1 , 3 ) } ( \alpha _ { i } ) + f _ { 4 } ^ { ( 2 , 3 ) } ( \alpha _ { i } ) + f _ { 4 } ^ { ( 5 , 3 ) } ( \alpha _ { i } ) } \end{array} \right\} .
$$

The decoding procedure is similar to the above cases and is based on Lagrange polynomial interpolation. It proceeds in three steps: 1) By (17)–(19), the sums $C _ { 1 , 2 } + C _ { 2 , 2 } + C _ { 5 , 2 } , C _ { 1 , 5 } + C _ { 2 , 5 } + C _ { 5 , 5 }$ , and $C _ { 1 , 6 } + C _ { 2 , 6 } + C _ { 5 , 6 }$ are recovered from the received messages. 2) By (16), the sum $C _ { 1 , 3 } + C _ { 2 , 3 } + C _ { 5 , 3 }$ is recovered using the received messages along with the previously recovered sum $C _ { 1 , 6 } + C _ { 2 , 6 } + C _ { 5 , 6 } . \ 3 )$ By (14)–(15), the polynomials $f _ { 1 } ^ { ( 1 , 1 ) } ( x ) + f _ { 1 } ^ { ( 2 , 1 ) } ( x ) +$ $f _ { 1 } ^ { ( 5 , 1 ) } ( x )$ and $f _ { 2 } ^ { ( 1 , 1 ) } ( x ) + f _ { 2 } ^ { ( 2 , 1 ) } ( x ) + f _ { 2 } ^ { ( 5 , 1 ) } ( x )$ are interpolated from the received messages along with the currently available sums $\{ C _ { 1 , r } + C _ { 2 , r } + C _ { 5 , r } \} _ { r \in \{ 2 , 3 , 5 , 6 \} }$ , thereby recovering the aggregate mask $C _ { 1 } + C _ { 2 } + C _ { 5 }$

In general, during the mask-elimination phase, each user successively computes and transmits up to $M = 3$ messages to the server. For any realized dropout set $S \subseteq [ 5 ]$ with size $S \in \{ 0 , 1 , 2 \}$ , the server can recover the desired gradient sum once every surviving user in $[ 5 ] \backslash S$ has successfully uploaded its first $M _ { S } = S + 1$ messages. Therefore, in the mask-elimination phase, the proposed scheme dynamically adapts the number of messages transmitted by each surviving user to the realized dropout scenario.

It is straightforward to verify that the proposed secure aggregation scheme achieves a key rate of 15, a keydistribution communication rate of 20, a masked-gradient communication rate of 3 in the presence of $U = 2$ dropouts, and a mask-elimination communication rate of $\frac { 5 - S } { 3 - S }$ in the present of S dropouts for all $S = 0 , 1 , 2 .$

## B. General Construction

In this subsection, we present the general construction of the proposed secure aggregation scheme, which can recover the desired gradient sum $W _ { s u m } ^ { \boldsymbol { U } }$ for all feasible dropout scenarios satisfying $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq$ $N - 1$ and $S \subseteq [ N ]$ with size $0 \le S \le N - T - 1$ . The proposed construction builds upon the principles of Shamir’s secret sharing [19], [33] and Staircase secret sharing [34], [35].

To ensure that the server can adaptively tolerate the presence of S dropouts for all feasible $0 \leq S \leq N - T - 1$ during the mask-elimination phase, we set $M = S _ { m a x } + 1 = N - T$ and choose the length L of the gradient vector as

$$
{ \cal L } = \mathrm { l c m } \{ N - T , N - T - 1 , \ldots , 1 \} ,
$$

where $\operatorname { l c m } ( \cdot )$ denotes the least common multiple. Let ${ \alpha } _ { 1 } , { \alpha } _ { 2 } , \ldots , { \alpha } _ { N }$ be N distinct elements in $\mathbb { F } _ { q }$ that are publicly available to all users and the server.

Key-Distribution Phase: In this phase, each user $i \in [ N ]$ first locally generates a random key vector $C _ { i }$ of length L over $\mathbb { F } _ { q } .$ , and then rearranges $C _ { i }$ into $M = N - T$ matrices, denoted by $\mathbf { P } ^ { ( i , 1 ) } , \mathbf { P } ^ { ( i , 2 ) } , \dots , \mathbf { P } ^ { ( i , M ) }$ . For any

given $m \in [ M ]$ , the matrix $\mathbf { P } ^ { ( i , m ) }$ has dimensions $\left( N - T - m + 1 \right) \times \lambda _ { m }$ and is introduced to provide robustness against the presence of $S = m - 1$ dropouts during the mask-elimination phase, where

$$
\lambda _ { m } = \left\{ \begin{array} { l l } { \displaystyle \frac { L } { N - T } } & { \mathrm { i f ~ } m = 1 } \\ { \displaystyle \frac { L } { ( N - T - m + 1 ) ( N - T - m + 2 ) } } & { \mathrm { i f ~ } 2 \leq m \leq M } \end{array} \right. .\tag{20}
$$

More specifically, these matrices are designed in an iterative manner as follows:

• In the initial case, the matrix $\mathbf { P } ^ { ( i , 1 ) }$ with dimensions $\begin{array} { r } { ( N - T ) \times \frac { L } { N - T } } \end{array}$ is constructed by distributing the L elements in $C _ { i }$ across all entries of $\mathbf { P } ^ { ( i , 1 ) }$ , given by

$$
P _ { k , \ell } ^ { ( i , 1 ) } = C _ { i , ( \ell - 1 ) ( N - T ) + k } , \quad \forall k \in [ N - T ] , \ell \in \bigg [ \frac { L } { N - T } \bigg ] .\tag{21}
$$

Here, $P _ { k , \ell } ^ { ( i , m ) }$ denotes the entry in the k-th row and ℓ-th column of the matrix $\mathbf { P } ^ { ( i , m ) }$ for any $i \in [ N ] , m \in$ $[ M ] , k \in [ N - T - m + 1 ]$ , and $\ell \in [ \lambda _ { m } ]$ , and $C _ { i , \ell }$ denotes the ℓ-th element of the vector $C _ { i }$ for any $\ell \in [ L ]$

• In the iterative phase, for any given $m \in [ 2 : M ]$ , the matrix $\mathbf { P } ^ { ( i , m ) }$ with dimensions $\left( N - T - m + 1 \right) \times \lambda _ { m }$ is constructed by distributing the elements from the $\left( N - T - m + 2 \right)$ -th row of the previously constructed matrices $\mathbf { P } ^ { ( i , 1 ) } , \mathbf { P } ^ { ( i , 2 ) } , \dots , \mathbf { P } ^ { ( i , m - 1 ) }$ across all entries of $\mathbf { P } ^ { ( i , m ) }$ . This is feasible because the total number of elements in the $\left( N - T - m + 2 \right)$ -th row of the matrices $\mathbf { P } ^ { ( i , 1 ) } , \mathbf { P } ^ { ( i , 2 ) } , \ldots , \mathbf { P } ^ { ( i , m - 1 ) }$ exactly matches the number of entries in $\mathbf { P } ^ { ( i , m ) }$ , i.e., $\begin{array} { r } { \sum _ { r \in [ m - 1 ] } \lambda _ { r } = \frac { L } { N - T - m + 2 } = ( N - T - m + 1 ) \lambda _ { m } } \end{array}$ by (20). Therefore, we can set the entry $P _ { k , \ell } ^ { ( i , m ) }$ of the matrix $\mathbf { P } ^ { ( i , m ) }$ as

$$
P _ { k , \ell } ^ { ( i , m ) } = P _ { ( N - T - m + 2 ) , \ell ^ { \prime } } ^ { ( i , m ^ { \prime } ) } , \forall k \in [ N - T - m + 1 ] , \ell \in [ \lambda _ { m } ] ,\tag{22}
$$

where for any $k \in [ N { - } T { - } m { + } 1 ]$ and $\ell \in [ \lambda _ { m } ] , ( m ^ { \prime } , \ell ^ { \prime } )$ is a unique pair such that $( \ell - 1 ) ( N - T - m + 1 ) + k =$ $\begin{array} { r } { \sum _ { r \in [ m ^ { \prime } - 1 ] } \lambda _ { r } + \ell ^ { \prime } } \end{array}$ for some $m ^ { \prime } \in [ m - 1 ]$ and $\ell ^ { \prime } \in [ \lambda _ { m ^ { \prime } } ]$

We complete the construction of $\mathbf { P } ^ { ( i , 1 ) } , \mathbf { P } ^ { ( i , 2 ) } , \dots , \mathbf { P } ^ { ( i , M ) }$ by executing the iterative procedure for $m = 2 , 3 , \dots , M .$ Then, for each $m \in [ M ]$ and $\ell \in [ \lambda _ { m } ]$ , we construct a secure encoding polynomial $f _ { \ell } ^ { ( i , m ) } ( x )$ of degree at most $N - m$ to encode the $N - T - m + 1$ entries in the ℓ-th column of matrix $\mathbf { P } ^ { ( i , m ) }$ , given by

$$
f _ { \ell } ^ { ( i , m ) } ( x ) = \sum _ { k \in [ T ] } D _ { k , \ell } ^ { ( i , m ) } x ^ { k - 1 } + \sum _ { k \in [ N - T - m + 1 ] } P _ { k , \ell } ^ { ( i , m ) } x ^ { T + k - 1 } ,\tag{23}
$$

where $D _ { 1 , \ell } ^ { ( i , m ) } , \dots , D _ { T , \ell } ^ { ( i , m ) }$ are T independent random symbols generated locally by user i over $\mathbb { F } _ { q }$ , which serve as privacy-preserving noise against any set of up to $T$ colluding users. Overall, the collection of all private random symbols generated by user i is given by

$$
\mathcal { Z } _ { i } = \left\{ C _ { i } , \{ D _ { k , \ell } ^ { ( i , m ) } \} _ { m \in [ M ] , k \in [ T ] , \ell \in [ \lambda _ { m } ] } \right\} , \quad \forall i \in [ N ] .\tag{24}
$$

For any given $i , j \in [ N ]$ , the encoded key $\mathcal { Z } _ { i , j }$ is constructed by evaluating the secure encoding polynomial $f _ { \ell } ^ { ( i , m ) } ( x )$ at $x = \alpha _ { j }$ for all $m \in [ M ]$ and $\ell \in [ \lambda _ { m } ]$ , i.e.,

$$
\begin{array} { r } { \mathcal { Z } _ { i , j } = \left\{ { f } _ { \ell } ^ { ( i , m ) } ( \alpha _ { j } ) : m \in [ M ] , \ell \in [ \lambda _ { m } ] \right\} . } \end{array}\tag{25}
$$

User $i \in [ N ]$ then shares the encoded key $\mathcal { Z } _ { i , j }$ with user $j \in [ N ] \backslash \{ i \}$ . This completes the key-distribution phase.

Masked-Gradient Upload Phase: Each user $i \in [ N ]$ computes the masked gradient $X _ { i }$ by masking its local gradient $W _ { i }$ with the random key $C _ { i } .$ , i.e.,

$$
X _ { i } = W _ { i } + C _ { i } , \quad \forall i \in [ N ] .\tag{26}
$$

User i then uploads $X _ { i }$ to the server. After collecting the masked gradients, the server identifies the set of dropout users U and computes

$$
\sum _ { i \in [ N ] \setminus \mathcal { U } } X _ { i } = \sum _ { i \in [ N ] \setminus \mathcal { U } } W _ { i } + \sum _ { i \in [ N ] \setminus \mathcal { U } } C _ { i } ,\tag{27}
$$

where the term $\sum _ { i \in [ N ] \backslash M } C _ { i }$ corresponds to the aggregate mask. Therefore, the server can recover the desired gradient sum $\sum _ { i \in [ N ] \backslash M } W _ { i }$ by removing the aggregate mask $\sum _ { i \in [ N ] \backslash M } C _ { i }$

Mask-Elimination Phase: In order to eliminate the aggregate mask $\textstyle \sum _ { i \in [ N ] \backslash { \mathcal { U } } } C _ { i }$ , the server broadcasts the dropout set U to all users. Then, for any $i \in [ N ]$ and $m \in [ M ]$ , user i can generate the message $\mathcal { V } _ { i , m } ^ { \mathcal { U } }$ using the locally available encoded keys $\{ \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \backslash \mathcal { U } }$ in (25), given by

$$
\begin{array} { r } { \mathcal { Y } _ { i , m } ^ { \mathcal { U } } = \left\{ \boldsymbol { f } _ { \ell } ^ { ( \mathcal { U } , m ) } ( \alpha _ { i } ) : \ell \in [ \lambda _ { m } ] \right\} , } \end{array}\tag{28}
$$

where for any $\ell \in [ \lambda _ { m } ]$

$$
f _ { \ell } ^ { ( \mathcal { U } , m ) } ( \alpha _ { i } ) = \sum _ { j \in [ N ] \backslash \mathcal { U } } f _ { \ell } ^ { ( j , m ) } ( \alpha _ { i } ) ,
$$

which corresponds to the evaluation of the following polynomial $f _ { \ell } ^ { ( \mathcal { U } , m ) } ( x )$ of degree at most $N - m$ at $x = \alpha _ { i }$

$$
f _ { \ell } ^ { ( \mathcal { U } , m ) } ( x ) = \sum _ { k \in [ T ] } \bigg ( \sum _ { \substack { j \in [ N ] \backslash \mathcal { U } } } D _ { k , \ell } ^ { ( j , m ) } \bigg ) x ^ { k - 1 } + \sum _ { \substack { k \in [ N - T - m + 1 ] } } \bigg ( \sum _ { \substack { j \in [ N ] \backslash \mathcal { U } } } P _ { k , \ell } ^ { ( j , m ) } \bigg ) x ^ { T + k - 1 } .\tag{29}
$$

Notably, the messages ${ \mathcal { V } } _ { i , 1 } ^ { \mathcal { U } } , { \mathcal { V } } _ { i , 2 } ^ { \mathcal { U } } , \dots , { \mathcal { V } } _ { i , M } ^ { \mathcal { U } }$ form an ordered transmission sequence. User i transmits these messages to the server sequentially in the prescribed order until it receives a stop signal from the server.

Let $M _ { S } = S + 1$ , i.e., in the presence of S dropouts for any given $0 \leq S \leq N - T - 1$ , the server waits to receive the first $S + 1$ messages from each of the surviving users in $[ N ] \backslash S$ for any dropout set $S \subseteq [ N ]$ with size S. Since S is unknown in advance, the stopping point depends on the realized dropout scenario. For any $0 \leq S \leq N - T - 1$ , once the required $M _ { S } = S + 1$ messages have been received from all surviving users under the corresponding dropout scenario, the server broadcasts a stop signal to terminate further message transmission. In total, for any dropout set $S \subseteq [ N ]$ with size $0 \leq S \leq N - T - 1$ , the server collects the following messages:

$$
\begin{array} { r } { \big \{ \mathcal { V } _ { i , m } ^ { \mathcal { U } } : i \in [ N ] \backslash \mathcal { S } , m \in [ S + 1 ] \big \} , \quad \forall \mathcal { S } \subseteq [ N ] , | \mathcal { S } | = S . } \end{array}\tag{30}
$$

By (28), this collection can be equivalently expressed as

$$
\left\{ f _ { \ell } ^ { ( \mathcal { U } , m ) } ( \alpha _ { i } ) : m \in [ S + 1 ] , \ell \in [ \lambda _ { m } ] , i \in [ N ] \backslash \mathcal { S } \right\} .\tag{31}
$$

Next, we prove that for any given $0 \leq S \leq N - T - 1$ and any dropout set $S \subseteq [ N ]$ with size $S ,$ the server can decode the aggregate mask $\sum _ { j \in [ N ] \backslash d } C _ { j }$ from the received messages in (31). Similar to the key-distribution phase, the decoding is carried out in an iterative manner below:

• In the initial case, we demonstrate that the server can decode the sum of the matrices $\{ \mathbf { P } ^ { ( j , S + 1 ) } \} _ { j \in [ N ] \backslash \mathcal { U } } ,$ i.e., $\begin{array} { r } { \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , S + 1 ) } } \end{array}$ , from the received data $\{ f _ { \ell } ^ { ( \mathcal { U } , S + 1 ) } ( \alpha _ { i } ) \} _ { \ell \in [ \lambda _ { S + 1 } ] , i \in [ N ] \setminus S }$ . By (29), the polynomial $f _ { \ell } ^ { ( \mathcal { U } , S + 1 ) } ( x )$ has degree at most $N - S - 1$ for every $\ell \ \in \ \left[ \lambda _ { S + 1 } \right]$ . Since the server knows the $N - S$ evaluations $\{ f _ { \ell } ^ { ( \mathcal { U } , S + 1 ) } ( \alpha _ { i } ) \} _ { i \in [ N ] \backslash \mathcal { S } }$ , it can uniquely interpolate $f _ { \ell } ^ { ( \mathcal { U } , S + 1 ) } ( x )$ and then recover the coefficients of the interpolated polynomial, which contain the entries $\begin{array} { r } { \{ \sum _ { j \in [ N ] \backslash \mathcal { U } } P _ { k , \ell } ^ { ( j , S + 1 ) } \} _ { k \in [ N - T - S ] } } \end{array}$ in the ℓ-th column of $\begin{array} { r } { \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , S + 1 ) } } \end{array}$ . Repeating this procedure for all $\ell \in [ \lambda _ { S + 1 } ]$ recovers the entire matrix $\begin{array} { r } { \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , S + 1 ) } } \end{array}$ • We next consider the iterative step. For any $m \in [ S ]$ , suppose that the matrices $\begin{array} { r } { \{ \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , m + r ) } \} _ { r \in [ S - m + 1 ] } } \end{array}$ have already been recovered. We show that the server can then recover $\sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , m ) }$ from the received messages $\{ f _ { \ell } ^ { ( \mathcal { U } , m ) } ( \alpha _ { i } ) \} _ { \ell \in [ \lambda _ { m } ] , i \in [ N ] \backslash \mathcal { S } }$ . By the construction of the matrices $\{ \mathbf { P } ^ { ( j , m ) } \} _ { j \in [ N ] , m \in [ M ] }$ in (22), for every $r \in [ S - m + 1 ]$ , the matrix $\begin{array} { r } { \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , m + r ) } } \end{array}$ contains the entries in the $\left( N - T - m - r + 2 \right)$ -th row of the previous matrices $\begin{array} { r } { \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , 1 ) } , \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , 2 ) } , \dots , \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , m + r - 1 ) } } \end{array}$ . Hence, for any $\ell \in [ \lambda _ { m } ]$ , the entries $\begin{array} { r } { \{ \sum _ { j \in [ N ] \setminus \mathcal { U } } P _ { N - T - m - r + 2 , \ell } ^ { ( j , m ) } \} _ { r \in [ S - m + 1 ] } } \end{array}$ can be obtained from the already recovered sum matrices $\begin{array} { r } { \{ \sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , m + r ) } \} _ { r \in [ S - m + 1 ] } } \end{array}$ . Moreover, by (29), the polynomial $f _ { \ell } ^ { ( \mathcal { U } , m ) } ( x )$ has degree at most $N - m$ . Since the server knows the $N - S$ evaluations $\{ f _ { \ell } ^ { ( \mathcal { U } , m ) } ( \alpha _ { i } ) \} _ { i \in [ N ] \backslash \mathcal { S } }$ together with the $S - m + 1$ coefficients $\begin{array} { r } { \{ \sum _ { j \in [ N ] \setminus \mathcal { U } } P _ { N - T - m - r + 2 , \ell } ^ { ( j , m ) } \} _ { r \in [ S - m + 1 ] } } \end{array}$ , the polynomial $f _ { \ell } ^ { ( \mathcal { U } , m ) } ( x )$ can be uniquely interpolated. Finally, the coefficients of the polynomial $f _ { \ell } ^ { ( \mathcal { U } , m ) } ( x )$ yield the ℓ-th column of $\sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , m ) }$ and repeating this procedure for all $\ell \in [ \lambda _ { m } ]$ recovers the entire matrix $\sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , m ) }$ By iteratively applying the above argument for $m = S , S - 1 , \ldots , 1$ , the server recovers the matrix $\sum _ { j \in [ N ] \backslash \mathcal { U } } \mathbf { P } ^ { ( j , 1 ) }$ from the received messages in (31). Since this matrix is precisely the aggregate mask $\sum _ { j \in [ N ] \backslash d } C _ { j }$ by (21), the server can eliminate the aggregate mask and recover the desired gradient sum $\sum _ { j \in [ N ] \backslash \mathcal { U } } W _ { j }$ by (27).

We now characterize the rates achieved by the proposed scheme. By (24) and (20), the total amount of random keys generated by all users is $\begin{array} { r } { N ( L + T \sum _ { m \in [ M ] } \lambda _ { m } ) = N ( T + 1 ) L } \end{array}$ . Therefore, the achieved key rate is $R _ { Z } = N ( T + 1 )$ For any distinct $i , j \in [ N ]$ , the encoded key $\mathcal { Z } _ { i , j }$ contains $\begin{array} { r } { \sum _ { m \in [ M ] } \lambda _ { m } = L } \end{array}$ symbols by (25) and (20). Since each user sends an encoded key to every other user, the achieved key-distribution communication rate is $R _ { K } = N ( N - 1 )$ In the masked-gradient upload phase, for any given dropout set $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ , the server receives $N - U$ masked gradients, each consisting of $L$ symbols. Hence, the achieved masked-gradient communication rate is $R _ { \mathcal { U } } = N - U$ . In the mask-elimination phase, for each such U and any given dropout set $S \subseteq [ N ]$ with size $0 \le S \le N - T - 1$ , the server collects the first $S + 1$ mask-elimination messages from each of the $N - S$ surviving users by (30). Since the m-th mask-elimination message contains $\lambda _ { m }$ symbols for $m \in [ S + 1 ]$ by (28), the achieved mask-elimination communication rate is $\begin{array} { r } { \frac { ( N - S ) \sum _ { m \in [ S + 1 ] } \lambda _ { m } } { L } = \frac { N - S } { N - T - S } } \end{array}$ . Consequently, for every feasible pair of dropout sets $( \mathcal { U } , \mathcal { S } )$ satisfying $0 \le U \le N - 1$ and $0 \le S \le N - T - 1$ , the proposed scheme achieves the rate tuple:

$$
\left( N ( T + 1 ) , N ( N - 1 ) , N - U , \frac { N - S } { N - T - S } \right) .
$$

The privacy of the proposed scheme is formally established in the Appendix. Here, we provide intuition by explaining how the three communication phases jointly ensure the privacy constraint. In the key-distribution phase, each user $i \in [ N ]$ distributes each symbol of its private random key $C _ { i }$ using T additional independent random noise symbols, ensuring that $C _ { i }$ remains information-theoretically secure against any set of up to $T$ colluding users. In the masked-gradient upload phase, the local gradient $W _ { i }$ is protected by the one-time-pad style mask $C _ { i } .$ so that the uploaded masked gradient $X _ { i } = W _ { i } + C _ { i }$ reveals no information about $W _ { i }$ . In the mask-elimination phase, the transmitted messages are deterministic functions of the desired gradient sum $W _ { s u m } ^ { \boldsymbol { U } }$ and the information available to the server and any $T$ colluding users from previous phases. Therefore, conditioned on the desired sum $W _ { s u m } ^ { \boldsymbol { U } }$ , these mask-elimination messages provide no additional information beyond what is already available from previous phases. Together, these observations imply that the server colluding with any up to $T$ users cannot infer any information about individual gradients of the remaining users beyond the desired gradient sum.

## V. PROOF OF THEOREM 2: CONVERSE

In this section, we derive converse bounds for the key rate, the key-distribution communication rate, the masked gradient communication rate, the mask-elimination communication rate, and the maximum tolerable number of dropouts in the mask-elimination phase. Note that converse bounds on the masked-gradient communication rate, the mask-elimination communication rate, and the maximum tolerable number of dropouts were previously established in [17] under a joint worst case over the dropout scenarios in both the masked-gradient upload and mask-elimination phases. Since these bounds do not directly apply to our dropout-dependent performance characterization, new converse bounds must be derived for each given dropout set realized during the masked-gradient upload phase.

Before presenting the formal proof, we first establish four useful lemmas, which provide the key technical tools needed for the subsequent converse argument. To simplify notation, let $\mathcal { D } _ { i }$ denote the collection of encoded keys that user $i \in [ N ]$ sends to other users and receives from other users during the key-distribution phase, i.e.,

$$
\mathcal { D } _ { i } = \{ \mathcal { Z } _ { i , j } , \mathcal { Z } _ { j , i } : j \in [ N ] \backslash \{ i \} \} , \quad \forall i \in [ N ] .\tag{32}
$$

Lemma 1. For any given security parameter $0 \leq T < N - 1$ and any set of colluding users $\mathcal { T } \subseteq [ N ]$ with $| \mathcal { T } | \leq T$ let $\mathcal { U } \subseteq [ N ]$ be an arbitrary subset such that $0 \leq | \mathcal { U } | \leq N - 1$ and $[ N ] \backslash \mathcal { T } \not \subseteq \mathcal { U } ,$ i.e., the dropout set U does not include all the remaining non-colluding users in $[ N ] \backslash \tau$ . Then, we have

$$
I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] } , \{ W _ { i } \} _ { i \in \mathcal { T } } , \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } ) = 0 .
$$

Proof. For any colluding users $\mathcal { T } \subseteq [ N ]$ with $| \mathcal { T } | \leq T$ and $0 \leq T < N - 1$ , let $\mathcal { U } \subseteq [ N ]$ be an arbitrary subset satisfying $[ N ] \backslash \mathcal { T } \not \subseteq \mathcal { U } .$ The lemma is proved by considering separately the two cases $0 \leq \left| \mathcal { U } \right| < N { - } 1$ and $| \mathcal { U } | = N - 1$ We first consider the case $0 \leq | \mathcal { U } | < N - 1$ . Since the dropout set $\mathcal { U } \subseteq [ N ]$ satisfies $[ N ] \backslash \mathcal { T } \not \subseteq \mathcal { U }$ , there exists at least one surviving non-colluding user $k \in [ N ] \backslash ( \mathcal { U } \cup \mathcal { T } )$ . Let $\mathcal { U } ^ { \prime } = \mathcal { U } \cup \{ k \}$ , which satisfies $1 \leq | \mathcal { U } ^ { \prime } | \leq N - 1$ The proof for this case then proceeds as follows.

$$
\begin{array} { r l } & { 0 \overset { ( a ) } { = } I ( \{ W _ { i } \} _ { i \in [ N ] } ; \mathrm { V i e w } _ { \mathcal { T } } ^ { \mathcal { U } ^ { \prime } } | W _ { s u m } ^ { \mathcal { U } ^ { \prime } } , \{ W _ { i } \} _ { i \in \mathcal { T } } ) } \\ & { \quad \ge I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] } , \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ { D } _ { i } \} _ { i \in \mathcal { T } } | W _ { s u m } ^ { \mathcal { U } ^ { \prime } } , \{ W _ { i } \} _ { i \in \mathcal { T } } ) } \\ & { \quad = I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] } , \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ { D } _ { i } \} _ { i \in \mathcal { T } } , W _ { s u m } ^ { \mathcal { U } ^ { \prime } } , \{ W _ { i } \} _ { i \in \mathcal { T } } ) - I ( W _ { s u m } ^ { \mathcal { U } } ; W _ { s u m } ^ { \mathcal { U } ^ { \prime } } , \{ W _ { i } \} _ { i \in \mathcal { T } } ) } \\ & { \overset { ( b ) } { = } I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] } , \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ { D } _ { i } \} _ { i \in \mathcal { T } } , W _ { s u m } ^ { \mathcal { U } ^ { \prime } } , \{ W _ { i } \} _ { i \in \mathcal { T } } ) } \\ &  \quad \ge I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] } , \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ { D } _ { i } \} _ { i \in \mathcal { T } } , \{ W _ { i } \} _ { i \in \mathcal { T } }  \end{array}
$$

where (a) follows from the security constraint in (8), and (b) follows from the facts that the gradients $W _ { 1 } , W _ { 2 } , \ldots , W _ { N }$ are mutually independent by (1), and that the index k satisfies $k \in [ N ] \backslash \mathcal { U } , k \notin [ N ] \backslash \mathcal { U } ^ { \prime }$ , and $k \notin \mathcal T$ , which together imply that $I ( W _ { s u m } ^ { \mathcal { U } } ; W _ { s u m } ^ { \mathcal { U } ^ { \prime } } , \{ W _ { i } \} _ { i \in \mathcal { T } } ) = 0$

Next, we consider the case $| \mathcal { U } | = N - 1 . \operatorname { L e t } \mathcal { U } ^ { \prime } = [ N ] \backslash \mathcal { U } ,$ which satisfies $| \mathcal { U } ^ { \prime } | = 1$ . Denote the unique element of $\mathcal { U } ^ { \prime }$ by k, i.e., $\mathcal { U } ^ { \prime } = \{ k \} = [ N ] \backslash \mathcal { U }$ . Furthermore, since $[ N ] \backslash \mathcal { T } \not \subseteq \mathcal { U } ,$ we have $k \notin \mathcal T$ . Then, by the security constrain in (8), we obtain

$$
\begin{array} { r l } &  \begin{array} { r l } & { 0 = - l _ { 4 } ( | A _ { 1 } ^ { \prime } | ) _ { 4 \times ( B _ { 2 } ^ { \prime } ) , \mathcal { R } ( u , \mathcal { R } ) } \left( \mathcal { R } _ { u , \mathcal { R } } ^ { 2 } - \{ B _ { 1 } ^ { \prime } \} \right) } \\ & { \geq \frac { l _ { 4 } } { \alpha } \Big ( | B _ { 2 } ^ { \prime } | \zeta _ { 2 } | u | \mathcal { R } _ { 1 } \Big ) \left( \mathcal { R } _ { 2 } ^ { 2 } | \zeta _ { 2 } | u _ { 2 \times \mathcal { T } } ^ { 2 } \right) } \\ & { \leq \frac { l _ { 4 } } { \alpha } \Big ( | B _ { 2 } ^ { \prime } | \zeta _ { 3 } | u | \mathcal { R } _ { 1 } \Big ) \left( \mathcal { R } _ { 2 } ^ { 2 } | \zeta _ { 2 } | u _ { 2 \times \mathcal { T } } ^ { 2 } \right) \frac { l _ { 4 } ^ { 2 } } { \alpha ^ { 2 } } \Big | u _ { 4 \times \mathcal { R } } \Big | \mathcal { R } _ { 1 } \Big | \mathcal { R } _ { 1 } \Big | \mathcal { R } _ { 2 } \Big | \mathcal { R } _ { 2 } \Big | \mathcal { R } _ { 2 } \Big | } \\ & { \overset { \mathrm { O } } { = } \frac { l _ { 4 } ^ { 2 } ( | A _ { 1 } ^ { \prime } | \zeta _ { 4 } | \mathcal { R } _ { 2 } | ) \mathcal { O } } { \alpha ^ { 2 } } \Big [ \mathcal { R } _ { 2 } | A _ { 2 } ^ { \prime } | \zeta _ { 3 } | u _ { 2 \times \mathcal { T } } ^ { 2 } \Big ] } \\ &  \overset { \mathrm { O } } { = } \Big ( | B _ { 2 } ^ { \prime } | \zeta _ { 3 } | u | \mathcal { R } _ { 1 } \Big ) \{ \mathcal { R } _ { 2 } ^ { 2 } | \zeta _ { 4 } | u _ { 2 \times \mathcal { T } } ^ { 2 } \} \Big [ \mathcal { R } _ { 2 } | A _ { 1 } ^ { \prime } | \zeta \end{array} \end{array}
$$

where (a) and (d) follow from the fact that $X _ { i }$ is a deterministic function of $\{ W _ { i } , \mathcal { Z } _ { i } , \mathcal { D } _ { i } \}$ by (5) and (32) for any $i \in \mathcal { T } ; ( b )$ and (c) follow from the facts that $k \notin \mathcal { T }$ and that $\{ W _ { i } \} _ { i \in \mathcal { T } }$ is independent of $\{ W _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } }$ and $\{ \mathcal { Z } _ { i } \} _ { i \in [ N ] }$ by (4); (e) is because $\{ k \} = [ N ] \backslash \mathcal { U }$ . Consequently, we complete the proof of the lemma. □

Lemma 2. For any given security parameter $0 { \le } T { < } N { - } 1$ and any subset of colluding users $\mathcal { T } \subseteq [ N ]$ with $| \mathcal { T } | = T$

$$
H ( \{ W _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } | \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \{ X _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } ) = ( N - T ) L .\tag{33}
$$

Proof. The proof of the lemma is divided into two cases according to the value of the security parameter $T \colon T = 0$ and $1 \leq T < N - 1$ . We first focus on the former case. In the case of $T = 0$ , for any given dropout set $\mathcal { U } \subseteq [ N ]$ satisfying $0 \leq \left| \mathcal { U } \right| \leq N - 1$ , we have

$$
\begin{array} { r l } & { 0 \overset { ( a ) } { = } I ( \{ W _ { i } \} _ { i \in [ N ] } ; \{ X _ { i } \} _ { i \in [ N ] } | W _ { s u m } ^ { M } ) } \\ & { \ : = I ( \{ W _ { i } \} _ { i \in [ N ] } , W _ { s u m } ^ { M } ; \{ X _ { i } \} _ { i \in [ N ] } ) - I ( W _ { s u m } ^ { M } ; \{ X _ { i } \} _ { i \in [ N ] } ) } \\ & { \overset { ( b ) } { = } I ( \{ W _ { i } \} _ { i \in [ N ] } ; \{ X _ { i } \} _ { i \in [ N ] } ) - I ( W _ { s u m } ^ { M } ; \{ X _ { i } \} _ { i \in [ N ] } ) } \\ & { \overset { ( c ) } { = } I ( \{ W _ { i } \} _ { i \in [ N ] } ; \{ X _ { i } \} _ { i \in [ N ] } ) } \\ & { \ : = H ( \{ W _ { i } \} _ { i \in [ N ] } ) - H ( \{ W _ { i } \} _ { i \in [ N ] } | \{ X _ { i } \} _ { i \in [ N ] } ) } \\ & { \overset { ( d ) } { = } N L - H ( \{ W _ { i } \} _ { i \in [ N ] } | \{ X _ { i } \} _ { i \in [ N ] } ) , } \end{array}
$$

where (a) follows from the security constraint in (8); (b) follows from the facts that $W _ { s u m } ^ { \boldsymbol { U } }$ is a deterministic function of $\{ \{ W _ { i } \} _ { i \in [ N ] } , { \mathcal { U } } \}$ , and that the dropout set U is publicly known and is treated as a fixed subset of users; (c) follows by Lemma 1 such that $I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] } ) = 0 ;$ (d) is due to (1) and (2). Therefore, the proof of the lemma is completed for the case $T = 0$

Next, we turn to the case $1 \leq T < N - 1$ . For any subset $\mathcal { T } \subseteq [ N ]$ with size T, let $\mathcal { U } \subseteq [ N ]$ be a dropout set satisfying $0 \leq | \mathcal { U } | \leq N - 1$ and $[ N ] \backslash u \subseteq \tau$ . Then, according to the security constraint in (8),

$$
\begin{array} { r l } & { 0 = I ( \{ W _ { i } \} _ { i \in [ N ] } ; \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \{ X _ { i } \} _ { i \in [ N ] } | W _ { s u m } ^ { \mathcal { U } } , \{ W _ { i } \} _ { i \in \mathcal { T } } ) } \\ & { \overset { ( a ) } { = } I ( \{ W _ { i } \} _ { i \in [ N ] \setminus \mathcal { T } } ; \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \{ X _ { i } \} _ { i \in [ N ] } | \{ W _ { i } \} _ { i \in \mathcal { T } } ) } \\ & { \ge I ( \{ W _ { i } \} _ { i \in [ N ] \setminus \mathcal { T } } ; \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \{ X _ { i } \} _ { i \in [ N ] \setminus \mathcal { T } } | \{ W _ { i } \} _ { i \in \mathcal { T } } ) } \\ & { \overset { ( b ) } { = } I ( \{ W _ { i } \} _ { i \in [ N ] \setminus \mathcal { T } } ; \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \{ X _ { i } \} _ { i \in [ N ] \setminus \mathcal { T } } ) , } \end{array}\tag{34}
$$

where (a) is due to $[ N ] \backslash \mathcal { U } \ \subseteq \ \mathcal { T }$ such that $W _ { s u m } ^ { \boldsymbol { U } }$ can be determined by $\{ W _ { i } \} _ { i \in { \mathscr { T } } }$ , and (b) follows from the independence of $\{ W _ { i } \} _ { i \in { \mathscr { T } } }$ from $\{ W _ { i } \} _ { i \in [ N ] \backslash T }$ and $\{ \mathcal { Z } _ { i } \} _ { i \in [ N ] }$ by (4), together with the fact that $\{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } }$ and $\{ X _ { i } \} _ { i \in [ N ] \backslash T }$ are the functions of $\{ W _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } }$ and $\{ \mathcal { Z } _ { i } \} _ { i \in [ N ] }$ by (3)–5 and (32). Furthermore, by (34),

$$
\begin{array} { r l } & { ~ H ( \{ W _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } | \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \{ X _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } ) } \\ & { = H ( \{ W _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } ) - I ( \{ W _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } ; \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \{ X _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } ) } \\ & { = H ( \{ W _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } ) } \\ & { = ( N - T ) L , } \end{array}
$$

which completes the proof of the lemma for the case $1 \leq T < N - 1$

Lemma 3. For any given security parameter $0 \leq T < N - 1$ , any subset of colluding users $\mathcal { T } \subseteq [ N ]$ with size T, and any index $k \in [ N ] \backslash \tau$

$$
H ( \{ W _ { i } \} _ { i \in [ N ] \setminus \mathcal { T } } | \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \mathcal { D } _ { k } , \{ X _ { i } \} _ { i \in [ N ] \setminus \mathcal { T } } ) = 0 .\tag{35}
$$

Proof. For any subset of colluding users $\mathcal { T } \subseteq [ N ]$ with size $0 \leq T < N - 1$ and any index $k \in [ N ] \backslash \tau$ , the lemma is proved by showing that $H ( W _ { i } | \mathcal { D } _ { i } , X _ { i } ) = 0$ for $i = k$ and $H ( W _ { i } | \{ \mathcal { D } _ { j } \} _ { j \in \mathcal { T } } , \mathcal { D } _ { k } , X _ { i } ) = 0$ for $i \in [ N ] \backslash ( { \mathcal { T } } \cup \{ k \} )$ . We first focus on the former case. For any $i \in [ N ]$ , we have

$$
\begin{array} { r l } & { 0 \leq I ( W _ { i } ; \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \setminus \{ i \} } | \mathcal { D } _ { i } , X _ { i } ) } \\ & { \leq I ( W _ { i } , X _ { i } , \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \setminus \{ i \} } | \mathcal { D } _ { i } ) } \\ & { \overset { ( a ) } { = } I ( W _ { i } , \mathcal { Z } _ { i } ; \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \setminus \{ i \} } | \mathcal { D } _ { i } ) } \\ & { \leq I ( W _ { i } , \mathcal { Z } _ { i } , \{ \mathcal { Z } _ { i , j } \} _ { j \in [ N ] \setminus \{ i \} } ; \{ \mathcal { Z } _ { j } , \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \setminus \{ i \} } ) } \\ & { \overset { ( b ) } { = } I ( W _ { i } , \mathcal { Z } _ { i } ; \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \setminus \{ i \} } ) } \\ & { \overset { ( c ) } { = } 0 , } \end{array}\tag{36}
$$

where (a) follows from the fact that $X _ { i }$ is a deterministic function of $\{ W _ { i } , \mathcal { Z } _ { i } , \mathcal { D } _ { i } \}$ by (5) and (32); (b) follows by (3); (c) is due to (4). Furthermore, for any $i \in [ N ]$ , let $\mathcal { U } = [ N ] \backslash \{ i \}$ and $S \subseteq [ N ]$ satisfying $i \in S$ and $1 \leq | S | \leq N - T - 1$ . Then, by (36),

$$
H ( W _ { i } | { \mathcal D } _ { i } , X _ { i } ) = H ( W _ { i } | { \mathcal D } _ { i } , X _ { i } ) - I ( W _ { i } ; \{ { \mathcal Z } _ { j } \} _ { j \in [ N ] \setminus \{ i \} } | { \mathcal D } _ { i } , X _ { i } )
$$

$$
= H ( W _ { i } | \mathcal { D } _ { i } , X _ { i } , \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \backslash \{ i \} } )
$$

$$
\stackrel { \left( a \right) } { = } H ( W _ { i } | \mathcal { D } _ { i } , X _ { i } , \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \backslash \{ i \} } , \{ W _ { j } \} _ { j \in [ N ] \backslash \{ i \} } , \mathcal { U } )\tag{37}
$$

$$
\overset { ( b ) } { = } H ( W _ { i } | \mathcal { D } _ { i } , X _ { i } , \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \backslash \{ i \} } , \{ W _ { j } \} _ { j \in [ N ] \backslash \{ i \} } , \{ \mathcal { V } _ { j , m } ^ { \mathcal { U } } \} _ { j \in [ N ] \backslash \mathcal { S } , m \in [ M ] } , \mathcal { U } )\tag{38}
$$

(39)

where (a) follows from the facts that $\{ W _ { j } \} _ { j \in [ N ] \backslash \{ i \} }$ is independent of $\left\{ W _ { i } , { \mathcal { D } } _ { i } , X _ { i } , \left\{ { \mathcal { Z } } _ { j } \right\} _ { j \in [ N ] \setminus \left\{ i \right\} } \right\}$ by (3), (5), (4), and (32), and that the dropout set U is publicly known and is treated as a fixed subset of users; (b) follows by (6); (c) follows by applying $\mathcal { U } = [ N ] \backslash \{ i \}$ to the correctness constraint in (7).

We next focus on the latter case. For any $k \in [ N ] \backslash \tau$ and $i \in [ N ] \backslash ( { \mathcal { T } } \cup \{ k \} )$ ),

$$
\begin{array} { r l } & { 0 \leq I ( W _ { i } , X _ { i } ; \{ \mathcal { Z } _ { j } \} _ { j \in { \mathcal { T } } } , \mathcal { Z } _ { k } | \{ \mathcal { D } _ { j } \} _ { j \in { \mathcal { T } } } , \mathcal { D } _ { k } ) } \\ & { \overset { ( a ) } { \leq } I ( W _ { i } , \mathcal { Z } _ { i } , \{ \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \setminus ( \mathcal { T } \cup \{ i , k \} ) } ; \{ \mathcal { Z } _ { j } \} _ { j \in { \mathcal { T } } } , \mathcal { Z } _ { k } | \{ \mathcal { D } _ { j } \} _ { j \in { \mathcal { T } } } , \mathcal { D } _ { k } ) } \\ & { \overset { ( b ) } { \leq } I ( W _ { i } , \mathcal { Z } _ { i } , \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \setminus ( \mathcal { T } \cup \{ i , k \} ) } ; \{ \mathcal { Z } _ { j } \} _ { j \in { \mathcal { T } } } , \mathcal { Z } _ { k } | \{ \mathcal { D } _ { j } \} _ { j \in { \mathcal { T } } } , \mathcal { D } _ { k } ) } \\ & { \overset { ( c ) } { \leq } I ( W _ { i } , \mathcal { Z } _ { i } , \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] \setminus ( \mathcal { T } \cup \{ i , k \} ) } ; \{ \mathcal { Z } _ { j } \} _ { j \in { \mathcal { T } } } , \mathcal { Z } _ { k } ) } \\ & { \overset { ( d ) } { = } 0 , } \end{array}\tag{40}
$$

where (a) follows from the fact that $X _ { i }$ is a function of $\{ W _ { i } , \mathcal { Z } _ { i } , \{ \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ i , k \} ) } , \{ \mathcal { D } _ { j } \} _ { j \in \mathcal { T } } , \mathcal { D } _ { k } \}$ by (5) and (32); (b) and (c) follow by (3); (d) is due to (4). Furthermore, for any $k \in [ N ] \backslash \tau$ and $i \in [ N ] \backslash ( { \mathcal { T } } \cup \{ k \} )$ , let $\mathcal { U } = [ N ] \backslash \{ i \}$ and let $\mathcal { S } = [ N ] \backslash ( \mathcal { T } \cup \{ k \} )$ ) with size $| S | = N - T - 1$ . Then, we have

$$
\begin{array} { r l } & { H ( W _ { i } | \{ \mathcal D _ { j } \} _ { j \in \mathcal { T } } , \mathcal { D } _ { k } , X _ { i } ) \overset { ( a ) } { = } H ( W _ { i } | \{ \mathcal D _ { j } \} _ { j \in \mathcal { T } } , \mathcal { D } _ { k } , X _ { i } , \{ \mathcal Z _ { j } \} _ { j \in \mathcal { T } } , \mathcal { Z } _ { k } ) } \\ & { \overset { ( b ) } { = } H ( W _ { i } | \{ \mathcal D _ { j } \} _ { j \in \mathcal { T } } , \mathcal { D } _ { k } , X _ { i } , \{ \mathcal Z _ { j } \} _ { j \in \mathcal { T } } , \mathcal { Z } _ { k } , \{ W _ { j } \} _ { j \in \mathcal { T } } , W _ { k } , \mathcal { U } ) } \\ & { \overset { ( c ) } { = } H ( W _ { i } | \{ \mathcal D _ { j } \} _ { j \in \mathcal { T } } , \mathcal { D } _ { k } , X _ { i } , \{ \mathcal Z _ { j } \} _ { j \in \mathcal { T } } , \mathcal { Z } _ { k } , \{ W _ { j } \} _ { j \in \mathcal { T } } , W _ { k } , \{ \mathcal { Y } _ { j , m } ^ { u } \} _ { j \in \mathcal { T } \cup \{ k \} , m \in [ M ] } ) } \\ & { \overset { ( d ) } { = } 0 , } \end{array}
$$

where (a) follows from the fact that $\{ \{ \mathcal { Z } _ { j } \} _ { j \in \mathcal { T } } , \mathcal { Z } _ { k } \}$ is independent of $\{ W _ { i } , X _ { i } \}$ conditioned on $\{ \mathcal { D } _ { j } \} _ { j \in \mathcal { T } } , \mathcal { D } _ { k } \}$ by (40); and $( b ) , ( c )$ , and (d) follow from arguments similar to those in (37), (38), and (39), respectively.

Consequently, for any subset of colluding users $\mathcal { T } \subseteq [ N ]$ with size $0 \leq T < N - 1$ and any index $k \in [ N ] \backslash \tau$

$$
\begin{array} { r l } & { \quad H \big ( \{ W _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } \big | \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \mathcal { D } _ { k } , \{ X _ { i } \} _ { i \in [ N ] \backslash \mathcal { T } } \big ) } \\ { \ } & { = H \big ( W _ { k } , \{ W _ { i } \} _ { i \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } \big | \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \mathcal { D } _ { k } , X _ { k } , \{ X _ { i } \} _ { i \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } \big ) } \\ { \ } & { = 0 , } \end{array}
$$

which completes the proof of the lemma.

Lemma 4. For any given security parameter $0 \leq T < N - 1$ , any subset of colluding users $\mathcal { T } \subseteq [ N ]$ with size $T ,$ and any index $k \in [ N ] \backslash \tau$

$$
H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \setminus ( T \cup \{ k \} ) } | \{ \mathcal { Z } _ { k , j } \} _ { j \in T } ) + \sum _ { i \in [ N ] \setminus ( T \cup \{ k \} ) } H ( \mathcal { Z } _ { i , k } | \{ \mathcal { Z } _ { i , j } \} _ { j \in T } ) \geq ( N - T ) L .\tag{41}
$$

Proof. From (33) and (35),

$$
\begin{array} { r l } & { H ( \mathcal { D } _ { k } | \{ \mathcal { D } _ { i } \} _ { i \in T } ) \geq H ( \mathcal { D } _ { k } | \{ \mathcal { D } _ { i } \} _ { i \in T } , \{ X _ { i } \} _ { i \in [ N ] \setminus T } ) } \\ & { \qquad \geq I ( \{ W _ { i } \} _ { i \in [ N ] \setminus T } ; \mathcal { D } _ { k } | \{ \mathcal { D } _ { i } \} _ { i \in T } , \{ X _ { i } \} _ { i \in [ N ] \setminus T } ) } \\ & { \qquad = H ( \{ W _ { i } \} _ { i \in [ N ] \setminus T } | \{ \mathcal { D } _ { i } \} _ { i \in T } , \{ X _ { i } \} _ { i \in [ N ] \setminus T } ) - H ( \{ W _ { i } \} _ { i \in [ N ] \setminus T } | \{ \mathcal { D } _ { i } \} _ { i \in T } , \mathcal { D } _ { k } , \{ X _ { i } \} _ { i \in [ N ] \setminus T } ) } \\ & { \qquad = ( N - T ) L . } \end{array}
$$

Then, according to the facts that $\{ \mathcal { Z } _ { i , j } \} _ { j \in [ N ] \backslash \{ i \} }$ is a deterministic function of $\mathcal { Z } _ { i }$ for any $i \in [ N ]$ by (3), and that $\mathcal { Z } _ { 1 } , \mathcal { Z } _ { 2 } , \ldots , \mathcal { Z } _ { N }$ are mutually independent by (4), we obtain

$$
\begin{array} { r l } { ( N - T ) L \leq H ( \mathcal { D } _ { k } | \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } ) } \\ { } & { = H ( \{ \mathcal { Z } _ { k , j } , \mathcal { Z } _ { j , k } \} _ { j \in [ N ] \backslash \{ k \} } | \{ \mathcal { Z } _ { i , j } , \mathcal { Z } _ { j , i } \} _ { i \in \mathcal { T } , j \in [ N ] \backslash \{ i \} } ) } \\ { } & { = H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } , \{ \mathcal { Z } _ { j , k } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } | \{ \mathcal { Z } _ { i , j } , \mathcal { Z } _ { j , i } \} _ { i \in \mathcal { T } , j \in [ N ] \backslash \{ i \} } ) } \\ { } & { = H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } , \{ \mathcal { Z } _ { j , k } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } | \{ \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \backslash \{ T , i \in \mathcal { T } \} } ) } \\ { } & { = H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } | \{ \mathcal { Z } _ { k , i } \} _ { i \in \mathcal { T } } ) + H ( \{ \mathcal { Z } _ { j , k } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } | \{ \mathcal { Z } _ { j , i } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) , i \in T } ) } \\ { } &  = H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash ( \mathcal { T } \cup \{ k \} ) } | \{ \mathcal { Z } _ { k , i } \} _ { i \in \mathcal { T } } ) + \frac { 1 }  j \in [ N ] \backslash ( \mathcal { Z } _ { j , k }  \end{array}
$$

This completes the proof of the lemma.

## A. Maximum Dropout Tolerance in the Mask-Elimination Phase

In this subsection, we prove that for any achievable secure aggregation scheme, the maximum feasible value of $S _ { m a x }$ is $N - T - 1$ for every security parameter $0 \leq T < N - 1$ and every dropout set $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ during the masked-gradient upload phase. From Section IV, we know that there exists an achievable secure aggregation scheme that can tolerate the presence of S dropout users in the mask-elimination phase for all $0 \leq S \leq N - T - 1$ . Therefore, it remains to prove that no achievable secure aggregation scheme can support $S _ { m a x } \geq N - T$ . Since Lemma 1 holds for every achievable secure aggregation scheme and every security parameter $0 \leq T < N - 1$ , the lemma can be invoked to establish the converse result. We proceed by contradiction.

Assume that $S _ { m a x } \geq N - T$ . Then, for any security parameter $0 \leq T < N - 1$ and any given dropout set $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ , there must exist a colluding user set $\mathcal { T } \subseteq [ N ]$ satisfying $| \mathcal { T } | = T$ and $[ N ] \backslash T \not \subseteq \mathcal { U } ,$ as well as a dropout set $S \subseteq [ N ]$ satisfying $[ N ] \backslash S \subseteq \mathcal { T }$ and $| S | \leq S _ { m a x } .$ . Accordingly, we obtain

$$
\begin{array} { r l } & { L = H ( W _ { s u m } ^ { \mathcal { U } } ) } \\ & { \quad \stackrel { ( a ) } { = } H ( W _ { s u m } ^ { \mathcal { U } } ) - H ( W _ { s u m } ^ { \mathcal { U } } | \{ X _ { i } \} _ { i \in [ N ] \backslash \mathcal { U } } , \{ \mathcal { V } _ { i , m } ^ { \mathcal { U } } \} _ { i \in [ N ] \backslash \mathcal { S } , m \in [ M _ { S } ] } ) } \end{array}
$$

$$
\begin{array} { r l } & { = I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] \setminus \mathcal { U } } , \{ \mathcal { V } _ { i , m } ^ { \mathcal { U } } \} _ { i \in [ N ] \setminus \mathcal { S } , m \in [ M _ { S } ] } ) } \\ & { \overset { ( b ) } { \leq } I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] \setminus \mathcal { U } } , \{ W _ { i } \} _ { i \in \mathcal { T } } , \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } , \mathcal { U } ) } \\ & { \overset { ( c ) } { = } I ( W _ { s u m } ^ { \mathcal { U } } ; \{ X _ { i } \} _ { i \in [ N ] \setminus \mathcal { U } } , \{ W _ { i } \} _ { i \in \mathcal { T } } , \{ \mathcal { Z } _ { i } \} _ { i \in \mathcal { T } } , \{ \mathcal { D } _ { i } \} _ { i \in \mathcal { T } } ) } \\ & { \overset { ( d ) } { = } 0 , } \end{array}
$$

where (a) follows from the correctness constraint in $( 7 ) ; ( b )$ follows by $[ N ] \backslash S \subseteq \mathcal { T }$ and the fact that $\{ \mathcal { V } _ { i , m } ^ { \mathcal { U } } \} _ { m \in [ M _ { S } ] }$ is the deterministic function of $\{ W _ { i } , \mathcal { Z } _ { i } , D _ { i } , \mathcal { U } \}$ for any $i \in [ N ] \backslash \mathcal { S }$ by (6) and (32); (c) follows from the fact that the dropout set U is publicly known and is treated as a fixed subset of users; (d) is due to Lemma 1.

This leads to a contradiction because $L \ge 1$ . Therefore, for any achievable secure aggregation scheme, the maximum number of dropouts that can be tolerated during the mask-elimination phase must satisfy $S _ { m a x } < N - T$ regardless of the dropout set U in the masked-gradient upload phase. Since $S _ { m a x }$ is an integer, its maximum feasible value is $N - T - 1$

## B. Converse for the Key-Distribution Communication Rate

In this subsection, we establish a lower bound on the optimal key-distribution communication rate $R _ { K } ^ { * }$ . For any given security parameter $0 \leq T < N - 1$ , we complete the proof by summing the inequality in (41) over all subsets $\mathcal { T } \subseteq [ N ]$ with size $| \mathcal { T } | = T$ and all $k \in [ N ] \backslash \tau$ , as follows:

$$
\begin{array} { r l } { \binom { N } { T } ( N - T ) ^ { 2 } L \leq } & { \displaystyle \sum _ { t \geq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor } H ( \mathcal { Z } _ { k , s } ) \gets } \\ & { \ : \ : \ : \ : \ : \frac { \eta _ { C } \| N \| \leq \lfloor N \rfloor } { \| T \| = T \| } \chi \langle \mathbf { \phi } | \sum _ { t \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor } H ( \mathcal { Z } _ { s , s } ) \rangle _ { \mathcal { C } ^ { \top } } } \\ & { \ : \ : \ : + \ : \displaystyle \sum _ { t \geq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor } H ( \mathcal { Z } _ { i , s } ) } \\ & { \leq \ : \displaystyle \sum _ { t \geq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor } H ( \mathcal { Z } _ { i , s } ) + \displaystyle \sum _ { t \geq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor } H ( \mathcal { Z } _ { i , s } ) } \\ & { = 2 \cdot \displaystyle \sum _ { t \geq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor } H ( \mathcal { Z } _ { k , s } ) } \\ &  = 2 \binom { N - 2 }  \displaystyle \sum _ { t \geq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor } \sum _ { t \geq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor \leq \lfloor N \rfloor } H ( \mathcal { Z } _ \end{array}
$$

Then, by (10), we obtain the following lower bound on the optimal key-distribution communication rate $R _ { K } ^ { * }$ :

$$
R _ { K } ^ { * } = \frac { \sum _ { i \in [ N ] } \sum _ { j \in [ N ] \backslash \{ i \} } H ( \mathcal { Z } _ { i , j } ) } { H ( W _ { s u m } ^ { \mathcal { U } } ) } \geq \frac { \binom { N } { T } ( N - T ) ^ { 2 } } { 2 \binom { N - 2 } { T } } = \frac { N ( N - 1 ) ( N - T ) } { 2 ( N - T - 1 ) } .
$$

## C. Converse for the Key Rate

In this subsection, we derive a lower bound on the optimal key rate $R _ { Z } ^ { * }$ and a lower bound on a linear combination of $R _ { Z } ^ { * }$ and the optimal key-distribution communication rate $R _ { K } ^ { * }$

For any given security parameter $0 \leq T < N - 1$ , summing the inequality in (41) over all subsets $\mathcal { T } \subseteq [ N ]$ with size $| \mathcal { T } | = T$ and all $k \in [ N ] \backslash \tau$ yields

$$
\begin{array} { r l } { \hat { \mathcal { L } } _ { \beta } ^ { \mathrm { G P } } ( \mathbf { x } , \mathbf { x } , \mathbf { x } , \mathbf { x } ) } & { = \sum _ { j \in \mathcal { N } _ { j } } \sum _ { k = 1 } ^ { N } \omega _ { j } \exp \left( - \frac { \omega _ { j } } { 2 } \exp \left( \omega _ { j } \mathbf { x } \right) \right) \exp \left( - \omega _ { j } \mathbf { x } \right) , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad }  \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad }  \\ & & { = \displaystyle \frac { 1 } { N } \sum _ { j \in \mathcal { N } _ { j } } \sum _ { k = 1 } ^ { N } \omega _ { j } \exp \left( - \frac { \omega _ { j } } { 2 } \exp \left( - \frac { \omega _ { j } } { 2 } \right) \right) \exp \left( - \frac { \omega _ { j } } { 2 } \exp \left( - \frac { \omega _ { j } } { 2 } \right) \right) , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  = \displaystyle \frac { 1 } { N } \sum _ { j \in \mathcal { N } _ { j } } \sum _ { k = 1 } \end{array}
$$

where (a) is due to the well-known Han’s inequality [36, Theorem 17.6.1] such that for any $i \in [ N ]$

$$
\sum _ { \stackrel { \scriptstyle \mathcal { T } \subseteq [ N ] \setminus \{ i \} } { | \mathcal { T } | = T } } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in \mathcal { T } } ) \geq \frac { { \binom { N - 1 } { T } } } { { \binom { N - 1 } { T + 1 } } } \frac { T } { T + 1 } \sum _ { \stackrel { \scriptstyle \mathcal { T } \subseteq [ N ] \setminus \{ i \} } { | \mathcal { T } | = T + 1 } } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in \mathcal { T } } ) .
$$

In the case of $N \geq 2 T + 1$ , the inequality in (42) can be further derived as follows:

$$
\begin{array} { l } { \displaystyle { \binom { N } { T } ( N - T ) ^ { 2 } L \le \binom { N - 1 } { T } \sum _ { k \in [ N ] } H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash \{ k \} } ) + \frac { N - 2 T - 1 } { N - T - 1 } \sum _ { i \in [ N ] } \sum _ { T \subset [ N ] \backslash \{ i \} } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in T } ) } } \\ { \displaystyle { \qquad \le \binom { N - 1 } { T } \sum _ { k \in [ N ] } H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash \{ k \} } ) + \frac { N - 2 T - 1 } { N - T - 1 } \binom { N - 1 } { T + 1 } \sum _ { i \in [ N ] } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in [ N ] \backslash \{ i \} } ) } } \\ { \displaystyle { \qquad = \frac { N - T } { T + 1 } \binom { N - 1 } { T } \sum _ { i \in [ N ] } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in [ N ] \backslash \{ i \} } ) . } } \end{array}
$$

Then, by (3) and (9), we can obtain the following lower bound on the optimal key rate $R _ { Z } ^ { * }$ :

$$
R _ { Z } ^ { * } = \frac { \sum _ { i \in [ N ] } H ( \mathcal { Z } _ { i } ) } { H ( W _ { s u m } ^ { \mathcal { U } } ) }
$$

$$
\begin{array} { r l } & { \geq \frac { \sum _ { i \in [ N ] } H \left( \{ \mathcal { Z } _ { i , j } \} _ { j \in [ N ] \backslash \{ i \} } \right) } { L } } \\ & { \geq \frac { \binom { N } { T } \left( N - T \right) ^ { 2 } } { \frac { N - T } { T + 1 } \binom { N - 1 } { T } } } \\ & { = N ( T + 1 ) . } \end{array}
$$

In the case of $T + 1 < N < 2 T + 1$ , the lower bound on $R _ { Z } ^ { * }$ can be obtained by transforming the inequality in (42) into the following form:

$$
\begin{array} { l } { \displaystyle { \binom { N } { T } ( N - T ) ^ { 2 } L \le \binom { N - 1 } { T } \sum _ { k \in [ N ] } H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash \{ k \} } ) - \frac { 2 T + 1 - N } { N - T - 1 } \sum _ { i \in [ N ] \backslash T \subset [ N ] \backslash \{ i \} } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in T } ) } } \\ { \displaystyle { \overset { ( a ) } { \le } \binom { N } { T } \sum _ { k \in [ N ] } \sum _ { k \in [ N ] \backslash \{ k \} } 2 \frac { 2 T + 1 - N } { N - T - 1 } \frac { \binom { N - 1 } { T + 1 } \binom { T + 1 } { T + 1 } } { N - 1 } \sum _ { i \in [ N ] } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in [ N ] \backslash \{ i \} } ) } } \\ { \displaystyle { = \frac { 2 ( N - T - 1 ) } { N - 1 } \binom { N - 1 } { T } \sum _ { i \in [ N ] } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in [ N ] \backslash \{ i \} } ) } } \\ { \displaystyle { \overset { ( b ) } { \le } \frac { 2 ( N - T - 1 ) } { N - 1 } \binom { N - 1 } { T } L \cdot R _ { z } ^ { * } , } } \end{array}
$$

where (a) follows from the Han’s inequality [36, Theorem 17.6.1] such that for any $i \in [ N ]$

$$
\frac { 1 } { \binom { N - 1 } { T + 1 } } \sum _ { \substack { \mathcal T \subseteq [ N ] \backslash \{ i \} } } \frac { H ( \{ \mathcal Z _ { i , j } \} _ { j \in \mathcal T } ) } { T + 1 } \geq \frac { H ( \{ \mathcal Z _ { i , j } \} _ { j \in [ N ] \backslash \{ i \} } ) } { N - 1 } ,
$$

and (b) follows by (3) and (9). Therefore, in the case of $T + 1 < N < 2 T + 1$ , we have

$$
R _ { Z } ^ { * } \geq \frac { \binom { N } { T } ( N - T ) ^ { 2 } } { \frac { 2 ( N - T - 1 ) } { N - 1 } \binom { N - 1 } { T } } = \frac { N ( N - 1 ) ( N - T ) } { 2 ( N - T - 1 ) } .
$$

Consequently, the lower bound on the optimal key rate $R _ { Z } ^ { * }$ is given by

$$
\begin{array} { r } { R _ { Z } ^ { * } \geq \left\{ \begin{array} { l l } { \frac { N ( N - 1 ) ( N - T ) } { 2 ( N - T - 1 ) } , } & { \mathrm { i f } T + 1 < N < 2 T + 1 } \\ { N ( T + 1 ) , } & { \mathrm { i f } N \geq 2 T + 1 } \end{array} \right. } \end{array}
$$

Moreover, in the case of $N \geq 2 T + 1$ , the inequality in (42) can also be developed as the following form:

$$
\begin{array} { r l r } {  { \binom { N } { T } ( N - T ) ^ { 2 } L \leq \binom { N - 1 } { T } \sum _ { k \in [ N ] } H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash \{ k \} } ) + \frac { N - 2 T - 1 } { N - T - 1 } \sum _ { i \in [ N ] \backslash T \subseteq [ N ] \backslash \{ i \} } H ( \{ \mathcal { Z } _ { i , j } \} _ { j \in T } ) } } \\ & { } & { \leq \binom { N - 1 } { T } \sum _ { k \in [ N ] } H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash \{ k \} } ) + \frac { N - 2 T - 1 } { N - T - 1 } \sum _ { i \in [ N ] \backslash T \subseteq [ N ] \backslash \{ i \} } \sum _ { j \in T } H ( \mathcal { Z } _ { i , j } ) } \\ & { } & { = \binom { N - 1 } { T } \sum _ { k \in [ N ] } H ( \{ \mathcal { Z } _ { k , j } \} _ { j \in [ N ] \backslash \{ k \} } ) + \frac { N - 2 T - 1 } { N - T - 1 } \binom { N - 2 } { T } \sum _ { i \in [ N ] \backslash j \in [ N ] \backslash \{ i \} } H ( \mathcal { Z } _ { i , j } ) . } \end{array}
$$

From (9) and (10), we obtain the following lower bound on a linear combination of $R _ { Z } ^ { * }$ and $R _ { K } ^ { * }$ :

$$
N ( N - 1 ) ( N - T ) \le ( N - 1 ) R _ { Z } ^ { * } + ( N - 2 T - 1 ) R _ { K } ^ { * } .
$$

## D. Converse for the Masked-Gradient Communication Rate

In this subsection, we establish a lower bound on the optimal masked-gradient communication rate $R _ { \mathcal { U } } ^ { * }$ for each given dropout set U with size $0 \leq U \leq N - 1$ during the masked-gradient upload phase.

For any set of colluding users T with size $0 \leq | \mathcal T | < N - 1$ and any given user $k \in [ N ]$ , let $\mathcal { U } \subseteq [ N ]$ and $S \subseteq [ N ]$ denote the dropout sets satisfying $0 \leq | \mathcal { U } | < N - 1 , k \notin \mathcal { U } , k \in \mathcal { S }$ , and $1 \leq | S | \leq N - T - 1$ . Moreover, let $\mathcal { U } ^ { \prime } = \mathcal { U } \cup \{ k \}$ . Then, the masked gradient $X _ { k }$ sent by user $k \in [ N ]$ must satisfy

$$
\begin{array} { r l } { H ( \lambda _ { 3 } ) > H ( \lambda _ { 3 } ) > H ( \lambda _ { 3 } ) > \lambda _ { 3 } < \lambda _ { 3 } > \lambda _ { 4 } > \left| \mathcal { K } _ { 1 } ( \lambda _ { 3 } ) - \left( \mathcal { K } _ { 1 } ( \lambda _ { 4 } ) - \lambda _ { 3 } \right) \right| \mathcal { K } _ { 1 } ( \lambda _ { 3 } ) < \lambda _ { 3 } < \lambda _ { 4 } > \left| \mathcal { K } _ { 2 } ( \lambda _ { 3 } ) - \lambda _ { 3 } \right| ^ { 2 } } \\ & { \leq \frac { \lambda _ { 1 } ( \lambda _ { 3 } ) } { \lambda _ { 3 } } \exp \lambda _ { 3 } \log \lambda _ { 1 } ( \lambda _ { 3 } ) < \lambda _ { 3 } < \lambda _ { 3 } \exp \left( 0 , \mathcal { K } _ { 1 } ( \lambda _ { 3 } ) - \lambda _ { 3 } \right) \exp \lambda _ { 1 } ( \lambda _ { 3 } ) < \lambda _ { 3 } < \lambda _ { 4 } > \left| \mathcal { K } _ { 2 } ( \lambda _ { 3 } ) - \lambda _ { 3 } \right| ^ { 2 } } \\ & { = \frac { \lambda _ { 1 } ( \lambda _ { 3 } ) } { \lambda _ { 3 } } \exp \left( \lambda _ { 3 } \right) \exp \left( 0 , \lambda _ { 3 } \right) \exp \lambda _ { 1 } ( \lambda _ { 3 } ) \exp \lambda _ { 1 } ( \lambda _ { 3 } ) < \lambda _ { 3 } \exp \lambda _ { 1 } ( \lambda _ { 3 } ) } \\ & { - \frac { \lambda _ { 1 } ( \lambda _ { 3 } ) } { \lambda _ { 3 } } \exp \left( 0 , \lambda _ { 3 } \right) \exp \lambda _ { 1 } ( \lambda _ { 3 } ) < \lambda _ { 3 } \exp \lambda _ { 1 } ( \lambda _ { 3 } ) < \lambda _ { 3 } \exp \lambda _ { 2 } \rangle } \\ &  \overset { \mathrm { a b s i n g ~ } } { = } \frac { \lambda _ { 1 } ( \lambda _ { 3 } ) } { \lambda _ { 3 } } \exp \left( \lambda _ { 3 } \right) \exp \lambda _ { 1 } ( \lambda _ { 3 } ) <  \end{array}\tag{43}
$$

where (a) follows from the facts that $X _ { i }$ is the function of $W _ { i }$ and $\{ \mathcal { Z } _ { j } \} _ { j \in [ N ] }$ for any $i \in [ N ] \backslash \mathcal { U } ^ { \prime }$ by $( 3 ) – ( 5 )$ and that $\{ \mathcal { V } _ { i , m } ^ { \mathcal { U } } \} _ { m \in [ M _ { S } ] }$ is the function of $\{ W _ { i } , \{ \mathcal { Z } _ { j } \} _ { j \in [ N ] } , \mathcal { U } \}$ for any $i \in [ N ] \backslash { \cal S }$ by (3) and (6); (b) follows by the fact that the dropout set U is publicly known and is treated as a fixed subset of users; (c) follows from the fact that the random variables $\{ \{ W _ { i } \} _ { i \in [ N ] } , \{ \mathcal { Z } _ { i } \} _ { i \in [ N ] } \}$ are generated independently of each other by (4) and therefore $W _ { k }$ is independent of $\{ W _ { i } \} _ { i \in [ N ] \setminus \mathcal { U } ^ { \prime } } , \{ \mathcal { Z } _ { i } \} _ { i \in [ N ] } , \{ W _ { i } \} _ { i \in [ N ] \setminus \mathcal { S } }$ , and $W _ { s u m } ^ { \mathcal { U } ^ { \prime } } ; ( d )$ is due to the correctness constraint in $( 7 ) ;$ (e) is because $W _ { s u m } ^ { \mathcal { U } } - W _ { s u m } ^ { \mathcal { U } ^ { \prime } } = W _ { k }$ . This implies that, in order to ensure that the server can correctly recover the desired gradient sum $W _ { s u m } ^ { \boldsymbol { U } }$ for every feasible dropout set $\mathcal { U } \subseteq [ N ]$ with $0 \leq | \mathcal { U } | \leq N - 1$ , the masked gradient $X _ { k }$ transmitted by each user $k \in [ N ]$ must have entropy at least L.

Therefore, for any set of dropout users $\mathcal { U } \subseteq [ N ]$ with $| \mathcal { U } | = U$ and $0 \leq U \leq N - 1$ , by (11) and (43), we obtain the following lower bound on the optimal masked-gradient communication rate $R _ { \mathcal { U } } ^ { * }$

$$
R _ { \mathcal { U } } ^ { * } = \frac { \displaystyle \sum _ { k \in [ N ] \backslash \mathcal { U } } H ( X _ { k } ) } { \displaystyle L } \geq N - U .
$$

## E. Converse for the Mask-Elimination Communication Rate

In this subsection, for any given dropout set $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ in the masked-gradient upload phase, we establish a lower bound on the optimal masked-elimination communication rate $R _ { S } ^ { { \cal U } * }$ for $1 \leq S \leq$ $N - T - 1$ and for $S = 0$ with $T < N - U$ . The case $S = 0$ with $T \geq N - U$ remains open.

In the case $1 \le S \le N - T - 1$ , for each such dropout set $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ , let $S \subseteq [ N ]$ be the dropout set in the mask-elimination phase such that $S \cap ( [ N ] \backslash \mathcal { U } ) \neq \emptyset$ . Then, for any set of colluding users ${ \mathcal { T } } \subseteq [ N ] \backslash S$ with $| \mathcal { T } | = T .$ , we must have $[ N ] \backslash \mathcal { T } \not \subseteq \mathcal { U }$ . Therefore, we obtain

$$
\begin{array} { r l } & { L ^ { ( \frac { d } { d } ) } = \int ( \mathbb { W } _ { x \leq m } ^ { \varepsilon } ; \{ X _ { 1 } ^ { \varepsilon } , \varepsilon | x | \} | y | ) , \ \{ \mathcal { S } _ { y , m } ^ { \varepsilon } \} \in [ \mathbb { W } | \backslash S , \operatorname { a r e g u r f a } \} ) } \\ &  = \int I ( \mathbb { W } _ { x \leq m } ^ { \varepsilon } ; \{ X _ { 1 } ^ { \varepsilon } , \varepsilon | x | \} | y | ) , \ \{ \mathcal { S } _ { y , m } ^ { \varepsilon } \} \in \mathbb { F } , \operatorname { a r e g u r f a } \} ; \ | \mathcal { S } _ { y , m } ^ { \varepsilon } \} \\ & { = \int I ( \mathbb { W } _ { x \leq m } ^ { \varepsilon } ; \{ X _ { 1 } ^ { \varepsilon } , \varepsilon | x | \} | y | ) , \ \big \{ \mathcal { S } _ { y , m } ^ { \varepsilon } \} \in \mathcal { W } _ { \varepsilon } \operatorname { W o l d } ; } \\ & { \quad + \int ( \mathbb { W } _ { y \leq m } ^ { \varepsilon } ; \{ X _ { 1 } ^ { \varepsilon } , \varepsilon | x | \} | y | ) , \ \big \{ \mathcal { S } _ { y , m } ^ { \varepsilon } \} \in \mathcal { W } _ { \varepsilon } \operatorname { W o l d } ; } \\ & { \quad + I ( \mathbb { W } _ { y \leq m } ^ { \varepsilon } ; \{ \mathcal { S } _ { y , m } ^ { \varepsilon } \} | y | ) , \ \big \{ \mathcal { S } _ { y , m } ^ { \varepsilon } \} \in \mathcal { W } _ { \varepsilon } \big [ | \mathcal { S } _ { x } | \big ] \in \mathcal { W } \} ; \ | \mathcal { S } _ { y , m } ^ { \varepsilon } \big [ \mathcal { S } _ { y , m } ^ { \varepsilon } \big ] \in \mathcal { W } _ { \varepsilon } \operatorname { W o l d } ; } \\ &  \overset { ( b ) } { \leq } \int I ( \mathbb { W } _ { x \leq m } ^ { \varepsilon } ; \{ X _ { 1 } ^ { \varepsilon } , \varepsilon | x | \} | y | , \mathcal { S } _ { y , m } ^ { \varepsilon } \} | \mathcal { S } _  x , j  \end{array}\tag{44}
$$

where (a) follows from the correctness constraint in (7); (b) follows from the fact that $\{ \mathcal { V } _ { i , m } ^ { \mathcal { U } } \} _ { m \in [ M _ { S } ] }$ is the deterministic function of $\{ W _ { i } , \mathcal { Z } _ { i } , D _ { i } , \mathcal { U } \}$ for any $i \in \mathcal { T }$ by (6) and (32); (c) follows by Lemma 1 and the fact that the dropout set U is publicly known and is treated as a fixed subset of users. Furthermore, by summing the inequality in (44) over all subsets ${ \mathcal { T } } \subseteq [ N ] \backslash S$ with size $| \mathcal { T } | = T$

$$
\begin{array} { r l } {  { \binom { N - S } { T } L \leq \sum _ { \substack { T \subseteq [ N ] \backslash S , | T | = T } } H ( \{ \mathcal { Y } _ { i , m } ^ { \mathcal { U } } \} _ { i \in [ N ] \backslash ( S \cup T ) , m \in [ M _ { S } ] } ) } } \\ & { \leq \sum _ { \substack { T \subseteq [ N ] \backslash S , | T | = T } } \sum _ { \substack { i \in [ N ] \backslash ( S \cup T ) } } H ( \{ \mathcal { Y } _ { i , m } ^ { \mathcal { U } } \} _ { m \in [ M _ { S } ] } ) } \\ & { = \binom { N - S - 1 } { T } \sum _ { i \in [ N ] \backslash S } H ( \{ \mathcal { Y } _ { i , m } ^ { \mathcal { U } } \} _ { m \in [ M _ { S } ] } ) . } \end{array}\tag{45}
$$

Therefore, for any given dropout set $S \subseteq [ N ]$ satisfying $S \cap ( [ N ] \backslash \mathcal { U } ) \neq \emptyset$ and $1 \le S \le N - T - 1$ , by (12) and (45), we obtain the following lower bound on the optimal masked-elimination communication rate $R _ { S } ^ { { \cal U } * }$ :

$$
\begin{array} { r l } { R _ { S } ^ { 2 * } = } & { \frac { S ^ { \prime } \underline { { \Sigma } } _ { 1 } ^ { \mathrm { { m a x } } } ( y ) S ^ { \prime } = \underline { { \Sigma } } _ { 1 } ^ { \prime } H ( | \mathcal { D } _ { 1 , m } ^ { \mathcal { I } } ) } { H ( W _ { \mathrm { s u m } } ^ { \mathcal { I } } ) } } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \geq \frac { i \underline { { \Sigma } } _ { 1 } ^ { \mathrm { C N } } \sum _ { m \in [ \overline { { \Sigma } } ] } \sum _ { i = 1 } ^ { m } \overline { { \mu ( y _ { i , m } ^ { \mathcal { I } } ) } } } { \underline { { \Sigma } } _ { 1 } ^ { \mathrm { { m a x } } } ( | \mathcal { D } _ { 1 , m } ^ { \mathcal { I } } ) _ { m \in [ \overline { { \Sigma } } ] } } } \\ & { \geq \frac { i \underline { { \Sigma } } _ { 1 } ^ { \mathrm { C N } } S ^ { \prime } } { \underline { { \Sigma } } _ { 1 } ^ { \mathrm { { m a x } } } ( | \mathcal { D } _ { 1 , m } ^ { \mathcal { I } } ) _ { m \in [ \overline { { \Sigma } } ] } } } \\ & { \geq \frac { ( N _ { T } - S ) } { ( N _ { S } - S - 1 ) } } \\ & { = \frac { N _ { T } - S } { N _ { S } - S - T } . } \end{array}
$$

In the case $S = 0$ with $T < N - U ,$ , we have $s = \emptyset$ . Then, for each given dropout set $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ and any set of colluding users ${ \mathcal { T } } \subseteq [ N ] \backslash S$ with $| \mathcal { T } | = T$ , we must have $[ N ] \backslash \mathcal { T } \not \subseteq \mathcal { U }$ . Therefore, the above proof remains fully valid in this case, and accordingly, we obtain

$$
R _ { S } ^ { \mathcal { U } * } \geq \frac { N - S } { N - S - T } .
$$

## VI. COMPARISON WITH RELATED WORK

The works most closely related to this paper are [17], [18], [22], which consider the T-colluding information theoretic secure aggregation problem with user dropouts. Although their formulations impose a set-inclusion relation between the dropout sets in the masked-gradient upload and mask-elimination phases, their proposed schemes can also directly support the dropout scenarios considered in this paper. This is because mask-elimination messages in these schemes are generated from the established correlated keys, allowing a user to participate in the maskelimination phase regardless of whether it successfully uploaded its masked gradient in the preceding phase. Table I compares the performance of these schemes with that of the proposed scheme. For a fair comparison, we consider the setting in which all schemes are required to support all feasible dropout scenarios satisfying $0 \leq U \leq N - 1$ and $0 \leq S \leq N - T - 1$ . The key-distribution communication rate is not reported for the schemes in [17], [22]. The scheme in [17] assumes that the correlated random keys are established by a trusted third party, whereas the scheme in [22] assumes a prescribed symmetric groupwise key-distribution structure. Consequently, the communication cost required to establish the correlated randomness is not explicitly modeled in either scheme.

Compared with the schemes in [17], [18], [22], the main advantage of the proposed scheme is its ability to reduce the mask-elimination communication rate by adapting to the realized number of dropouts. To tolerate every feasible number of dropouts $0 \leq S \leq N - T - 1$ during the mask-elimination phase, the previous schemes are designed fo the worst-case value $S = N - T - 1$ and therefore incur a fixed mask-elimination communication rate of $T + 1$ regardless of the realized value of S. By contrast, the proposed scheme dynamically adjusts its communication cost to the realized number of dropouts and achieves the mask-elimination communication rate $\frac { N - S } { N - T - S }$ for every feasible S. This rate is no larger than $T + 1$ , with equality in the worst-case scenario $S = N { - } T { - } 1$ . For $T > 0 ,$ , it is strictly smaller whenever $S < N { - } T { - } 1$ . It is worth emphasizing that this communication gain arises from adapting the mask-elimination communication to the realized dropout scenario, rather than from relaxing the set-inclusion constraint between the dropout sets in the masked-gradient upload and mask-elimination phases.

Another important distinction between this paper and the existing works in [17], [18], [22] lies in the fundamental limit characterization. These works do not investigate the fundamental limits of either the key rate or the keydistribution communication rate. In contrast, we establish, for the first time, information-theoretic lower bounds on these two rates and prove that the rates achieved by the proposed scheme are either optimal or within a multiplicative factor of two of their respective information-theoretic optima. Moreover, existing works characterize the mask elimination communication rate by taking the worst case over the dropout scenarios in both the masked-gradient upload and mask-elimination phases. Such a joint worst-case characterization only establishes the optimality of a single worst-case mask-elimination communication rate. It does not reveal how the required mask-elimination communication varies with the particular dropout set realized during the masked-gradient upload phase. In contrast, for each given dropout set realized during the masked-gradient upload phase, we characterize the fundamental limits of the masked-gradient communication rate and the mask-elimination communication rate. The resulting converse bounds establish the optimality of the proposed scheme in terms of the masked-gradient communication rate for all feasible parameters and in terms of the mask-elimination communication rate whenever $1 \leq S \leq N - T - 1$ or $( S = 0 \mathrm { \ a n d \ } T < N - U )$ . Together with the converse bounds on the key rate and key-distribution communication rate, these results provide a unified fundamental-limit characterization of all four rates considered in this paper.

TABLE I  
PERFORMANCE COMPARISON BETWEEN OUR PROPOSED SCHEME AND THE SECURE AGGREGATION SCHEMES IN [17], [18], [22].
<table><tr><td rowspan=1 colspan=1>Secure Aggregation SchemesPerformance Metrics</td><td rowspan=1 colspan=1>Zhao-Sun Scheme [17] [So e</td><td rowspan=1 colspan=1>t al.&#x27;s Scheme [18] Zh</td><td rowspan=1 colspan=1>ang et al.&#x27;s Scheme [22]</td><td rowspan=1 colspan=1>Our Scheme</td></tr><tr><td rowspan=1 colspan=1>Key Rate $R _ { Z }$ </td><td rowspan=1 colspan=1> $\begin{array} { r } { \overline { { N + T \sum _ { g = T + 1 } ^ { N } \binom { N } { g } } } } \end{array}$ </td><td rowspan=1 colspan=1> $N ( T + 1 )$ </td><td rowspan=1 colspan=1> $\binom { N } { N - T } ( N$ -T)</td><td rowspan=1 colspan=1> $N ( T + 1 )$ </td></tr><tr><td rowspan=1 colspan=1>Key-Distribution Communication Rate $R _ { K }$ </td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1> $N ( N - 1 )$ </td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1> $N ( N - 1 )$ </td></tr><tr><td rowspan=1 colspan=1>Masked-Gradient Communication Rate $R u$ </td><td rowspan=1 colspan=1> $N - U$ </td><td rowspan=1 colspan=1>N-U</td><td rowspan=1 colspan=1>N-U</td><td rowspan=1 colspan=1> $N - U$ </td></tr><tr><td rowspan=1 colspan=1>Mask-Elimination Communication Rate $R _ { S } ^ { \mathcal { U } }$ </td><td rowspan=1 colspan=1>T + 1</td><td rowspan=1 colspan=1> $T + 1$ </td><td rowspan=1 colspan=1>T + 1</td><td rowspan=1 colspan=1> $\frac { N - S } { N - T - S }$ </td></tr><tr><td rowspan=1 colspan=1>Required Finite Field $\mathbb { F } _ { q }$ </td><td rowspan=1 colspan=1> $q \geq N + T + 1$ </td><td rowspan=1 colspan=1> $q \geq N$ </td><td rowspan=1 colspan=1>q → ∞</td><td rowspan=1 colspan=1> $q \geq N$ </td></tr><tr><td rowspan=1 colspan=1> $\operatorname { K e y }$ Establishment Mechanism</td><td rowspan=1 colspan=1>Trusted Third Party</td><td rowspan=1 colspan=1>User Collaboration</td><td rowspan=1 colspan=1>Groupwise $\operatorname { K e y }$ </td><td rowspan=1 colspan=1>User Collaboration</td></tr><tr><td rowspan=1 colspan=1>Construction Type</td><td rowspan=1 colspan=1>Explicit</td><td rowspan=1 colspan=1>Explicit</td><td rowspan=1 colspan=1>Probabilistic</td><td rowspan=1 colspan=1>Explicit</td></tr></table>

## VII. CONCLUSION

In this paper, we investigated secure aggregation with user dropouts under the T-colluding security constraint. We formulated an adaptive secure aggregation framework that supports all feasible dropout scenarios, thereby enabling the communication strategy to dynamically adapt to the actual dropout scenario observed during protocol execution. Within this framework, we established converse bounds on four fundamental resources: the randomness required to generate correlated keys, the communication required to distribute them, the communication for uploading masked gradients, and the additional communication required to eliminate residual masks. In particular, the converse bounds on the latter two resources are characterized separately for each dropout set realized during the masked-gradient upload phase. For a system with N users, we developed an adaptive secure aggregation scheme that achieves the rate tuple $\begin{array} { r } { \left( N ( T + 1 ) , N ( N - 1 ) , N - U , \frac { N - S } { N - T - S } \right) } \end{array}$ for all feasible dropout scenarios, where U and S denote the actual numbers of dropout users during protocol execution in the masked-gradient upload and mask-elimination phases, respectively. By comparing these achievable rates with the converse bounds, we showed that the proposed scheme achieves the optimal masked-gradient communication rate for every feasible dropout set. It also achieves the optimal mask-elimination communication rate whenever $1 \leq S \leq N - T - 1 \ \mathrm { o r } \ ( S = 0 \ \mathrm { a n d } \ T < N - U )$ . Moreover, the achieved key rate and key-distribution communication rate are each either information-theoretically optimal or within a multiplicative factor of two of their respective optima. Overall, this paper provides the first fundamental characterization of the communication and randomness resources required for secure aggregation with user dropouts. Future work includes determining the optimal mask-elimination communication rate in the remaining regime where S = 0 and $T \geq N - U$ , closing the remaining gaps in the key rate and the key-distribution communication rate, and extending the proposed framework to incorporate more practical system constraints.

## APPENDIX

## SECURITY PROOF OF THE PROPOSED SCHEME

In this appendix, we formally prove that the proposed secure aggregation scheme satisfies the privacy constraint in (8). For any colluding user set $\mathcal { T } \subseteq [ N ]$ with $| \tau | \leq T ,$ let $\widetilde { \tau } \subseteq [ N ]$ be an arbitrary set satisfying $| \widetilde { \mathcal { T } } | = T$ and $\mathcal { T } \subseteq \widetilde { \mathcal { T } }$ . Then, for any given dropout scenario satisfying $\mathcal { U } \subseteq [ N ]$ with size $0 \leq U \leq N - 1$ and $S \subseteq [ N ]$ with size $0 \leq S \leq N - T - 1$ , the proposed secure aggregation scheme satisfies

$$
\begin{array} { r l } & { \begin{array} { r l } & { \sum _ { j = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { j } < \gamma _ { j } < \gamma _ { j } , } \\ & { \sum _ { k = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { j } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { j } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { j } , } \end{array} } \\ & { \begin{array} { r l } & { \sum _ { j = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { j } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { j } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { j } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { l } , } \end{array} } \\ & { = \begin{array} { r l } { \sum _ { j = 1 } ^ { \infty } \lambda _ { j } - \gamma _ { j } < \gamma _ { j } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { j } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } < \gamma _ { l } , } \end{array} } \\ & { \begin{ r l } { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } - \gamma _ { l } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } - \gamma _ { l } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } + \gamma _ { l } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { l } + \gamma _ { l } , } \end{array} } \\ &  \begin{array} { r l } & { \sum _ { j = 1 } ^ { \infty } \lambda _ { j } - \gamma _ { j } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } - \gamma _ { l } , } \\ & { \sum _ { l = 1 } ^ { \infty } \lambda _ { j } + \gamma _ { l } , } \\ &  \sum  \end{array}
$$

$$
\begin{array} { l } { \stackrel { ( j ) } { = } I \Big ( \{ W _ { i } \} _ { i \in [ N ] } ; \{ W _ { i } + C _ { i } \} _ { i \in [ N ] } \Big ) } \\ { \stackrel { ( k ) } { = } 0 , } \end{array}
$$

where (a) follows from the fact that $\{ \mathcal { Z } _ { i , j } \} _ { j \in [ N ] \backslash \{ i \} }$ is a deterministic function of $\mathcal { Z } _ { i }$ for any $i \in \mathcal { T }$ by (23)–(25); (b) is due to (24)–(26) and (28); (c) follows from the fact that $\sum _ { j \in [ N ] \backslash d } C _ { j }$ can be recovered from $\{ W _ { i } + C _ { i } \} _ { i \in [ N ] \backslash \mathcal { U } }$ and $W _ { s u m } ^ { \mathcal { U } } ; ( d )$ follows from the facts that $\begin{array} { r } { \{ \sum _ { j \in [ N ] \setminus \mathcal { U } } P _ { k , \ell } ^ { ( j , m ) } \} _ { m \in [ M ] , k \in [ N - T - m + 1 ] , \ell \in [ \lambda _ { m } ] } } \end{array}$ is a function of $\sum _ { j \in [ N ] \backslash d } C _ { j }$ by (21)-(22), and that $\begin{array} { r } { \{ \sum _ { j \in [ N ] \backslash \mathcal { U } } D _ { k , \ell } ^ { ( j , m ) } \} _ { k \in [ T ] } } \end{array}$ is a deterministic function of $\begin{array} { r } { \{ \sum _ { j \in [ N ] \setminus \mathcal { U } } P _ { k , \ell } ^ { ( j , m ) } \} _ { k \in [ N - T - m + 1 ] } } \end{array}$ and $\begin{array} { r } { \{ \sum _ { j \in [ N ] \backslash \mathcal { U } } f _ { \ell } ^ { ( j , m ) } ( \alpha _ { i } ) \} _ { i \in \widetilde { \mathcal { T } } } } \end{array}$ for any given $m \in [ M ]$ and $\ell \in [ \lambda _ { m } ]$ by (23); (e) follows from (29); (f) follows from arguments similar to those used in (c) and $( d ) ; ( g )$ follows from the fact that $\{ f _ { \ell } ^ { ( j , m ) } ( \alpha _ { i } ) \} _ { i \in \tilde { \mathcal { T } } }$ is a deterministic function of $C _ { j }$ and $\{ D _ { k , \ell } ^ { ( j , m ) } \} _ { k \in [ T ] }$ for any given $j \in \mathcal { T } , m \in [ M ]$ , and $\ell \in [ \lambda _ { m } ]$ by (21)–(23); (h) follows from the fact that $\{ C _ { i } \} _ { i \in \mathcal { T } }$ and $\{ D _ { k , \ell } ^ { ( i , m ) } \} _ { i \in { \mathcal { T } } , m \in [ M ] , k \in [ T ] , \ell \in [ \lambda _ { m } ] }$ are generated independently of all other random variables appearing in (46); (i) follows from the fact that $\{ f _ { \ell } ^ { ( j , m ) } ( \alpha _ { i } ) \} _ { j \in [ N ] \backslash \mathcal { T } , m \in [ M ] , \ell \in [ \lambda _ { m } ] , i \in \widetilde { \mathcal { T } } }$ is generated independently of the gradients $\{ W _ { i } \} _ { i \in [ N ] }$ by (23); (j) follows from the fact that the evaluations $\{ f _ { \ell } ^ { ( j , m ) } ( \alpha _ { i } ) \} _ { i \in \tilde { \mathcal { T } } }$ are generated using the T random noise symbols $\{ D _ { k , \ell } ^ { ( j , m ) } \} _ { k \in [ T ] }$ for any $j \in [ N ] \backslash \mathcal { T } , m \in [ M ]$ , and $\ell \in [ \lambda _ { m } ]$ by (23), while $\{ D _ { k , \ell } ^ { ( j , m ) } \} _ { j \in [ N ] \backslash \mathcal { T } , m \in [ M ] , k \in [ T ] , \ell \in [ \lambda _ { m } ] }$ is independent of $\{ W _ { i } , C _ { i } \} _ { i \in [ N ] } ; ( k )$ follows from the fact that $C _ { 1 } , C _ { 2 } , \dots , C _ { N }$ are mutually independent and uniformly distributed over the finite field $\mathbb { F } _ { q }$

## REFERENCES

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics, pp. 1273–1282, PMLR, 2017.

[2] J. Konecnˇ y, H. B. McMahan, F. X. Yu, P. Richtárik, A. T. Suresh, and D. Bacon, “Federated learning: Strategies for improving\` communication efficiency,” arXiv preprint arXiv:1610.05492, 2016.

[3] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning: Challenges, methods, and future directions,” IEEE signal processing magazine, vol. 37, no. 3, pp. 50–60, 2020.

[4] Q. Yang, Y. Liu, T. Chen, and Y. Tong, “Federated machine learning: Concept and applications,” ACM Transactions on Intelligent Systems and Technology (TIST), vol. 10, no. 2, pp. 1–19, 2019.

[5] L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,” Advances in neural information processing systems, vol. 32, 2019.

[6] B. Zhao, K. R. Mopuri, and H. Bilen, “idlg: Improved deep leakage from gradients,” arXiv preprint arXiv:2001.02610, 2020.

[7] J. Geiping, H. Bauermeister, H. Dröge, and M. Moeller, “Inverting gradients-how easy is it to break privacy in federated learning?,” Advances in neural information processing systems, vol. 33, pp. 16937–16947, 2020.

[8] H. Yin, A. Mallya, A. Vahdat, J. M. Alvarez, J. Kautz, and P. Molchanov, “See through gradients: Image batch recovery via gradinversion,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16337–16346, 2021.

[9] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, and K. Seth, “Practical secure aggregation for privacy-preserving machine learning,” in proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 1175–1191, 2017.

[10] J. H. Bell, K. A. Bonawitz, A. Gascón, T. Lepoint, and M. Raykova, “Secure single-server aggregation with (poly) logarithmic overhead,” in Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pp. 1253–1269, 2020.

[11] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al., “Advances and open problems in federated learning,” Foundations and trends® in machine learning, vol. 14, no. 1–2, pp. 1–210, 2021.

[12] K. Bonawitz, H. Eichner, W. Grieskamp, D. Huba, A. Ingerman, V. Ivanov, C. Kiddon, J. Konecnˇ y, S. Mazzocchi, B. McMahan, \` et al., “Towards federated learning at scale: System design,” Proceedings of machine learning and systems, vol. 1, pp. 374–388, 2019.

[13] J. Dean and L. A. Barroso, “The tail at scale,” Communications of the ACM, vol. 56, no. 2, pp. 74–80, 2013.

[14] G. Ananthanarayanan, A. Ghodsi, S. Shenker, and I. Stoica, “Effective straggler mitigation: Attack of the clones,” in 10th USENIX Symposium on Networked Systems Design and Implementation (NSDI 13), pp. 185–198, 2013.

[15] B. Choi, J.-y. Sohn, D.-J. Han, and J. Moon, “Communication-computation efficient secure aggregation for federated learning,” arXiv preprint arXiv:2012.05433, 2020.

[16] Z. Liu, J. Guo, K.-Y. Lam, and J. Zhao, “Efficient dropout-resilient aggregation for privacy-preserving machine learning,” IEEE Transactions on Information Forensics and Security, vol. 18, pp. 1839–1854, 2022.

[17] Y. Zhao and H. Sun, “Information theoretic secure aggregation with user dropouts,” IEEE Transactions on Information Theory, vol. 68, no. 11, pp. 7471–7484, 2022.

[18] J. So, C. He, C.-S. Yang, S. Li, Q. Yu, R. E Ali, B. Guler, and S. Avestimehr, “Lightsecagg: a lightweight and versatile design for secure aggregation in federated learning,” Proceedings of Machine Learning and Systems, vol. 4, pp. 694–720, 2022.

[19] A. Shamir, “How to share a secret,” Communications of the ACM, vol. 22, no. 11, pp. 612–613, 1979.

[20] K. Wan, X. Yao, H. Sun, M. Ji, and G. Caire, “On the information theoretic secure aggregation with uncoded groupwise keys,” IEEE Transactions on Information Theory, vol. 70, no. 9, pp. 6596–6619, 2024.

[21] K. Wan, H. Sun, M. Ji, T. Mi, and G. Caire, “The capacity region of information theoretic secure aggregation with uncoded groupwise keys,” IEEE Transactions on Information Theory, vol. 70, no. 10, pp. 6932–6949, 2024.

[22] Z. Zhang, J. Liu, K. Wan, H. Sun, M. Ji, and G. Caire, “On secure aggregation with uncoded groupwise keys against user dropouts and user collusion,” IEEE Transactions on Information Theory, vol. 71, no. 11, pp. 8391–8413, 2025.

[23] L. Yi, J. Zhu, K. Wan, and X. Tang, “The capacity of information-theoretic secure aggregation in federated learning,” arXiv preprint arXiv:2606.07277, 2026.

[24] Y. Zhao and H. Sun, “Secure summation: Capacity region, groupwise key, and feasibility,” IEEE Transactions on Information Theory, vol. 70, no. 2, pp. 1376–1387, 2023.

[25] X. Zhang, K. Wan, H. Sun, S. Wang, M. Ji, and G. Caire, “Optimal communication and key rate region for hierarchical secure aggregation with user collusion,” IEEE Transactions on Information Theory, vol. 72, no. 2, pp. 1030–1050, 2025.

[26] Z. Li, Y. Zhao, X. Zhang, and G. Caire, “On the fundamental limits of hierarchical secure aggregation with dropout and collusion resilience,” arXiv preprint arXiv:2603.19705, 2026.

[27] X. Yuan and H. Sun, “Vector linear secure aggregation,” in 2025 IEEE International Symposium on Information Theory (ISIT), pp. 1–6, IEEE, 2025.

[28] L. Hu and S. Ulukus, “On the capacity region of individual key rates in vector linear secure aggregation,” arXiv preprint arXiv:2601.03241, 2026.

[29] H. Tang, J. Zhu, and X. Tang, “Secure aggregation with top-k sparsification in decentralized federated learning,” in 2026 IEEE International Symposium on Information Theory (ISIT), pp. 1–6, IEEE, 2026.

[30] H. U. Sami and B. Güler, “Secure gradient aggregation with sparsification for resource-limited federated learning,” IEEE Transactions on Communications, vol. 72, no. 11, pp. 6883–6899, 2024.

[31] R. Tandon, Q. Lei, A. G. Dimakis, and N. Karampatziakis, “Gradient coding: Avoiding stragglers in distributed learning,” in International Conference on Machine Learning, pp. 3368–3376, PMLR, 2017.

[32] H. Cao, Q. Yan, X. Tang, and G. Han, “Adaptive gradient coding,” IEEE/ACM Transactions on Networking, vol. 30, no. 2, pp. 717–734, 2021.

[33] R. J. McEliece and D. V. Sarwate, “On sharing secrets and reed-solomon codes,” Communications of the ACM, vol. 24, no. 9, pp. 583–584, 1981.

[34] R. Bitar and S. El Rouayheb, “Staircase codes for secret sharing with optimal communication and read overheads,” IEEE Transactions on Information Theory, vol. 64, no. 2, pp. 933–943, 2017.

[35] R. Bitar and S. El Rouayheb, “Secret sharing with optimal communication and read overheads,” in 2016 IEEE International Symposium on Information Theory (ISIT), pp. 1396–1400, IEEE, 2016.

[36] T. M. Cover, Elements of information theory. John Wiley & Sons, 1999.