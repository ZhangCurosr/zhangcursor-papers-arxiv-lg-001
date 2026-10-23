# Graph-Recurrent Learning of Emergent Collective Dynamics

Ali Rayat,<sup>1</sup> Acacia Lee,<sup>1</sup> and Gia-Wei Chern<sup>1</sup>

<sup>1</sup>Department of Physics, University of Virginia, Charlottesville, Virginia 22904, USA (Dated: October 23, 2026)

Collective ordering in correlated electron systems emerges from microscopic dynamics, yet the evolution of observable textures need not admit a closed, memoryless description. We introduce a graph-recurrent neural network (GRNN) framework that learns such emergent dynamics directly from trajectories without specifying microscopic equations of motion. The framework integrates gated recurrent memory with graph message passing: local hidden states encode information beyond the instantaneous observables, while spatial coupling allows this information to propagate between sites. The observable and hidden fields evolve jointly through a shared local rule, providing a scalable representation of dynamics with spatial interactions and temporal memory. We demonstrate this framework on the emergent dynamics of charge-density-wave order in the semiclassical Holstein model, successfully reproducing temporal and spatial correlations and dynamical scaling during domain coarsening. By learning finite-time evolution directly, the GRNN ofers a route to eficient, data-driven descriptions of collective dynamics with unresolved microscopic degrees of freedom.

Pattern formation is a pervasive feature of correlated electron materials, where competing interactions generate magnetic textures, phase-separated clusters, and electronic stripes [1–5]. The evolution of these patterns poses a further challenge. Domains grow and defects move and annihilate across widely separated length and time scales, often exhibiting dynamical scaling during phase ordering [6, 7]. At the same time, their motion can depend on the electronic degrees of freedom underlying the competing orders. Phenomenological descriptions, including time-dependent Ginzburg–Landau and phasefield models, reach large scales eficiently but may omit these material-specific efects [8–10]. Microscopic simulations retain the electronic degrees of freedom, but must recalculate their response as the configuration evolves, restricting accessible sizes and times [11, 12]. Machine learning (ML) ofers a route to eficient, transferable dynamical models that connect microscopic physics with large-scale pattern formation.

A prominent example is the development of ML force fields for quantum-based molecular dynamics [13–19]. By replacing repeated electronic calculations with learned force predictions, these models have extended simulations to larger systems and longer times. This strategy has also been extended to condensed-matter lattice Hamiltonians, where learned forces drive the collective dynamics [20–23]. Its scalability draws on locality: the force at a site is inferred from its surrounding configuration, allowing the same predictor to be applied throughout a large lattice. Incorporating the relevant microscopic symmetries ensures consistent predictions for physically equivalent local environments. Such models have enabled large-scale studies of electronic phase separation and charge-density-wave (CDW), orbital, and magnetic ordering kinetics [24–29].

An alternative to force-field modeling is to learn a finite-time evolution map directly from order-parameter trajectories, relating a configuration to one at a later time. Such a model advances the field without evaluating microscopic forces and may permit larger efective time steps. In principle, it can also be trained on experimental trajectories without knowledge of the underlying Hamiltonian. Recurrent neural networks (RNNs) provide a natural framework for learning these time sequences, using an evolving hidden state to retain information from previous observations [30–32]. Architectures ranging from reservoir computing to long short term memory (LSTM) networks have been used to predict chaotic classical dynamics, molecular trajectories, and quantum evolution [33–42].

The need for temporal context also has a physical basis. Although microscopic dynamics are Markovian when expressed in terms of a complete state, eliminating electronic and other unobserved degrees of freedom can make the reduced dynamics non-Markovian [43–45]. Past observations may therefore contain information absent from the present one; under suitable conditions, Takens’ theorem shows how delayed observations can reconstruct the underlying dynamical state [46]. This motivates models that take a short history as input, including recent transformer approaches [47, 48].

An RNN instead compresses past observations into a recursively updated hidden state. The joint update of visible variables and hidden states is Markovian, while the visible dynamics alone can retain memory. From a coarse-graining perspective, the hidden states may encode the influence of unresolved microscopic degrees of freedom without identifying them individually [49–51]. Some recurrent models have addressed spatially extended dynamics [36, 52, 53], but interacting lattice systems pose a further design question: how can local recurrent memory be coupled through symmetry-aware GNN message passing to predict evolving textures across large lattices?

Here we introduce a graph-recurrent neural network (GRNN) that learns the evolution of spatially distributed observables directly from trajectories, without specifying their microscopic equations of motion. We formulate it as a message-passing extension of a gated recurrent unit (GRU), whose gates regulate how a hidden state incorporates new information and retains its past [35, 54]. On the lattice graph, messages from neighboring observables build a representation of the current spatial texture $[ 5 5 -$ 60]. Messages between neighboring hidden states also enter the recurrent update, allowing information accumulated over time to influence nearby sites [61, 62]. The resulting shared local rule advances the observable and hidden fields together and can be applied to lattices larger than those used for training.

![](images/bb57d3ae05e1febc1230826a22325079eb92d8c2d846ee6c2975a59643716c2a.jpg)  
FIG. 1. Schematic of the graph-recurrent update. The upper lattices represent successive physical configurations $\phi ( t )$ , and the lower lattices represent the corresponding site-resolved hidden states $h ( t )$ The labels $t - 1$ , t, and $t + 1$ denote successive prediction steps. Neighboring sites contribute to both the spatial representation of the physical field and the evolution of the hidden states. The colored arrows indicate the exchange of information between the physical and hidden states. The schematic shows the coupled evolution; the operations within one time step are given in the text.

We represent the evolving physical field by $\phi ( t ) \ =$ $\{ \phi _ { i } ( t ) \} _ { i \in V }$ , where V is the set of lattice sites, i labels a site and $\phi _ { i } ( t ) = ( \phi _ { i , 1 } ( t ) , \ldots , \phi _ { i , M } ( t ) )$ contains its M order-parameter components. In the GRNN update, $t + 1$ denotes the next sampled configuration, separated from t by the prediction interval $\Delta t _ { \mathrm { p r e d } }$ in physical time. At each site we also introduce a hidden vector $h _ { i } ( t )$ , which evolves alongside the physical field. As illustrated in Fig. 1, the GRNN learns a coupled update $( \phi ( t ) , h ( t ) ) \mapsto$ $( \phi ( t + 1 ) , h ( t + 1 ) )$ ), where $\pmb { h } ( t ) = \{ \pmb { h } _ { i } ( t ) \} _ { i \in V }$ . The hidden states are trained to retain information relevant to future evolution that is absent from the instantaneous physical configuration, including the influence of microscopic degrees of freedom excluded from the reduced description. They need not correspond individually to those microscopic variables; their role is to provide an evolving representation of their efects. The combined state can therefore admit a local-in-time update even when the dynamics of $\phi$ alone depend on its history.

The hidden states supply temporal information, but the evolution at a site also depends on the orderparameter texture around it. To incorporate this spatial dependence, we use graph message passing to build an embedding from the current physical configuration. Each successive layer combines information from neighboring sites, allowing the representation at site i to reflect a progressively larger region of the lattice. This gathering of spatial information is illustrated in the upper lattices of Fig. 1. Starting from ${ \pmb v } _ { i } ^ { ( 0 ) } ( t ) = \phi _ { i } ( t )$ , we define the mean neighbor features $\begin{array} { r } { \overline { { \pmb v } } _ { i } ^ { ( \ell ) } ( t ) = d _ { i } ^ { - 1 } \sum _ { j \in { \mathcal N } _ { i } } \pmb v _ { j } ^ { ( \ell ) } ( t ) } \end{array}$ , where ${ \mathcal { N } } _ { i }$ is the neighbor set and $d _ { i } = | { \mathcal { N } } _ { i } |$ . The features are updated according to

$$
\begin{array} { r } { \pmb { v } _ { i } ^ { ( \ell + 1 ) } ( t ) = \sigma _ { \mathrm { G } } \left[ W _ { \mathrm { s e l f } } ^ { ( \ell ) } \pmb { v } _ { i } ^ { ( \ell ) } ( t ) + W _ { \mathrm { n b r } } ^ { ( \ell ) } \pmb { \overline { { v } } } _ { i } ^ { ( \ell ) } ( t ) + \pmb { b } ^ { ( \ell ) } \right] , } \end{array}\tag{1}
$$

where $\sigma _ { \mathrm { G } }$ is the graph-layer activation, chosen here as the Gaussian error linear unit (GELU). The weight matrices and biases are learned and shared across sites. After $N _ { \mathrm { G } }$ layers, $z _ { i } ( t ) = \pmb { v } _ { i } ^ { ( N _ { \mathrm { G } } ) } ( t )$ encodes the surrounding texture for use in the recurrent update. This embedding is constructed from the configuration at time $t ;$ information from earlier times enters through the hidden states.

The embedding $z _ { i } ( t )$ describes the spatial texture at the current time. The recurrent update combines it with hidden states that carry information from earlier configurations, allowing the model’s use of that information to depend on the present texture. Because the history rele vant to a site can also originate nearby—for example, as a domain boundary moves toward it—we couple neighboring hidden states along the same lattice bonds. This exchange between hidden states is illustrated in the lower lattices of Fig. 1. Defining $\begin{array} { r } { \overline { { \pmb { h } } } _ { i } ( t ) = d _ { i } ^ { - 1 } \sum _ { j \in \mathcal { N } _ { i } } \pmb { h } _ { j } ( t ) } \end{array}$ , we construct the reset and update gates from $z _ { i } ( t )$ ， $h _ { i } ( t )$ and $\overline { { h } } _ { i } ( t )$

$$
\begin{array} { r l } & { { \mathbf { } } { \mathbf { } } _ { } ( t ) = \sigma _ { \mathrm { s } } \left[ W _ { r } z _ { i } ( t ) + U _ { r } { \mathbf { } } h _ { i } ( t ) + V _ { r } \overline { { h } } _ { i } ( t ) + b _ { r } \right] , } \\ & { { \mathbf { } } { \mathbf { } } _ { } u _ { i } ( t ) = \sigma _ { \mathrm { s } } \left[ W _ { u } z _ { i } ( t ) + U _ { u } { \mathbf { } } h _ { i } ( t ) + V _ { u } \overline { { h } } _ { i } ( t ) + b _ { u } \right] , } \end{array}\tag{2}
$$

where $\sigma _ { \mathrm { s } }$ is the logistic sigmoid. These gates determine how the stored information contributes to the new hidden state: the reset gate regulates the contribution of $h _ { i } ( t )$ to a candidate state, while the update gate controls how much of that candidate is incorporated. Specifically,

$$
\begin{array} { r l r } & { } & { \widetilde { \pmb { h } } _ { i } ( t + 1 ) = \operatorname { t a n h } \Bigl [ W _ { h } \pmb { z } _ { i } ( t ) + U _ { h } \big ( \pmb { r } _ { i } ( t ) \odot \pmb { h } _ { i } ( t ) \big ) } \\ & { } & { + V _ { h } \overline { { \pmb { h } } } _ { i } ( t ) + \pmb { b } _ { h } \Bigr ] , } \\ & { } & { \pmb { h } _ { i } ( t + 1 ) = \left[ \mathbf { 1 } - \pmb { u } _ { i } ( t ) \right] \odot \pmb { h } _ { i } ( t ) + \pmb { u } _ { i } ( t ) \odot \widetilde { \pmb { h } } _ { i } ( t + 1 ) , } \end{array}\tag{3}
$$

where denotes componentwise multiplication. The neighbor contribution enters the candidate directly, so information stored at one site can influence the hidden states nearby and propagate over successive time steps.

The next physical configuration is obtained from the spatial embedding and the updated hidden state:

$$
\begin{array} { r l r } & { } & { \pmb { \delta } _ { i } ( t ) = \mathrm { M L P } [ z _ { i } ( t ) \oplus \pmb { h } _ { i } ( t + 1 ) ] , } \\ & { } & { \phi _ { i } ( t + 1 ) = \mathcal { R } _ { \phi } [ \phi _ { i } ( t ) , \pmb { \delta } _ { i } ( t ) ] , \qquad } \end{array}\tag{4}
$$

![](images/49571739cf4531ed89f6e6b914b532885ead970cf51f88c2fe3a83bb9e476c65.jpg)  
FIG. 2. Normalized temporal autocorrelation $A ( \tau )$ of the CDW order parameter, comparing GRNN predictions (green) with reference adiabatic dynamics computed using KPM (magenta). The GRNN closely reproduces the decay of temporal correlations over the full interval $0 \leq \tau \leq 4 0 0$

where denotes concatenation and MLP denotes a multilayer perceptron. The readout $\mathcal { R } _ { \phi }$ specifies how a learned change acts on the physical variable and can be adapted to the allowed values and constraints of the order parameter. The complete update uses only $\phi ( t )$ and $h ( t )$ , but the evolution of ϕ alone can depend on earlier configurations through the hidden states. In this way, the network combines a local spatial description with a learned, site-dependent representation of memory.

We demonstrate the framework through CDW ordering in the semiclassical Holstein model, which describes itinerant electrons coupled to local lattice distortions [63]. For spinless electrons on a square lattice, the Hamiltonian is

$$
\begin{array} { r l r } {  { \mathcal { H } = t _ { \mathrm { n n } } \sum _ { \langle i j \rangle } ( c _ { i } ^ { \dagger } c _ { j } + \mathrm { H . c . } ) - g \sum _ { i } Q _ { i } \hat { n } _ { i } } } \\ & { } & { + \sum _ { i } ( \frac { P _ { i } ^ { 2 } } { 2 m } + \frac { k Q _ { i } ^ { 2 } } { 2 } ) + \kappa \sum _ { \langle i j \rangle } Q _ { i } Q _ { j } , } \end{array}\tag{5}
$$

where $\hat { n } _ { i } = c _ { i } ^ { \dagger } c _ { i } , Q _ { i }$ is a classical lattice displacement, and $P _ { i }$ is its conjugate momentum. The parameters $t _ { \mathrm { n n } }$ and g specify nearest-neighbor hopping and electron– lattice coupling, while k and κ describe the local and intersite elastic interactions.

We adopt the adiabatic approximation, in which the electrons equilibrate to the instantaneous displacement configuration $Q ( t )$ . The electronic subsystem thus has no independent dynamics, but its density $n _ { i } ( t ) = \langle \hat { n } _ { i } \rangle _ { Q ( t ) }$ determines the force acting on the lattice. The classical variables obey

$$
m \ddot { Q } _ { i } + \gamma \dot { Q } _ { i } = - k Q _ { i } - \kappa \sum _ { j \in \mathcal { N } _ { i } } Q _ { j } + g n _ { i } ( t ) + \eta _ { i } ( t ) ,\tag{6}
$$

where $\gamma$ is the damping coeficient and $\eta _ { i } ( t )$ represents thermal noise. Although the electronic Hamiltonian is quadratic at fixed $Q ,$ , the density must be recalculated as the lattice evolves. We generate reference trajectories using a Verlet-type integrator and evaluate the electronic densities and forces at each integration step with the kernel polynomial method (KPM) [11].

![](images/4c4a3effcec7837848156f66a6f6888079a34ad67475fd4953fbb85436c27705.jpg)  
FIG. 3. Equal-time CDW correlation function $C ( R , t )$ versus scaled separation $R = x / L ( t )$ at $t = 4 0$ , 80, 160, and 320, obtained from KPM reference dynamics (left) and GRNN predictions (right). The GRNN captures the spatial correlation profiles, while the collapse of curves at diferent times in both panels demonstrates dynamical scaling.

At half filling, the square-lattice Holstein model exhibits a strong tendency toward checkerboard charge order [64–66]. Its microscopic origin is particularly transparent for nearest-neighbor hopping, where perfect nest ing of the half-filled Fermi surface enhances the charge response to a checkerboard modulation. The resulting divergent charge response makes the uniform state unstable to a staggered lattice distortion for arbitrarily weak nonzero g at zero temperature in the classical lattice limit [67]. This distortion opens an electronic gap and produces alternating high- and low-density sites. Interchanging the two sublattices gives an equivalent ordered state, so the checkerboard CDW breaks a discrete $\mathbb { Z } _ { 2 }$ symmetry and undergoes an Ising transition [65, 67].

During phase ordering, regions of these two checkerboard patterns coexist and evolve through domain-wall motion. To describe this texture, we define the neighboraveraged density $\begin{array} { r } { { \overline { { n } } } _ { i } ( t ) \ = \ \frac { 1 } { 4 } \sum _ { j \in { \mathcal N } _ { i } } n _ { j } ( t ) } \end{array}$ and introduce the local order parameter

$$
\phi _ { i } ( t ) = ( - 1 ) ^ { x _ { i } + y _ { i } } \left[ n _ { i } ( t ) - \overline { { n } } _ { i } ( t ) \right] .\tag{7}
$$

Here $( x _ { i } , y _ { i } )$ are the integer lattice coordinates of site i. The density contrast measures the local charge modulation, while the sublattice factor removes its checkerboard alternation. The resulting field has approximately uniform sign within each domain, with opposite signs distinguishing the two ordered configurations.

The evolution of this emergent field is the quantity learned by the GRNN. In the adiabatic dynamics, the electronic density is an instantaneous function of the lattice displacements, $n _ { i } = n _ { i } ( \{ Q \} )$ . Since $\dot { Q } _ { k } = P _ { k } / m$ , the chain rule gives

$$
\dot { \phi } _ { i } ( t ) = ( - 1 ) ^ { x _ { i } + y _ { i } } \sum _ { k } \left[ \frac { \partial n _ { i } } { \partial Q _ { k } } - \frac { 1 } { 4 } \sum _ { j \in \mathcal { N } _ { i } } \frac { \partial n _ { j } } { \partial Q _ { k } } \right] \frac { P _ { k } ( t ) } { m } ,\tag{8}
$$

![](images/1690287dbf354c7a1bff4fee588b174545c613ed37d7de844d2d2a2074620203.jpg)  
FIG. 4. Autoregressive GRNN simulation of CDW domain coarsening on a $2 0 0 \times 2 0 0$ lattice at t = 0, 80, and 6400. Color indicates the local order parameter $\phi$ on the scale shown at right, from −1 to 1. The initially disordered field evolves into domains of the two symmetry-related charge-ordered states.

where the derivatives are evaluated at the instantaneous lattice configuration. This equation is not closed in ϕ: its right-hand side depends on the microscopic displacements and momenta, which are not determined by the instantaneous order-parameter field. Thus, even beyond the adiabatic approximation, retaining only $\phi$ produces efective non-Markovian dynamics; for example, states with the same displacements but diferent momenta can have the same $\phi$ and evolve diferently. The GRNN addresses this missing information through recurrent states that encode the influence of unresolved lattice motion, coupled through graph messages to account for spatial interactions. A central test of this learned description is whether it reproduces the temporal correlations of the emergent field. For the half-filled model, the numerical parameters and reference-trajectory protocol are summarized in Appendix B.

The dataset comprises 30 independent trajectories, divided into 16 for training, 4 for validation, and 10 for testing, with complete trajectories kept separate across these subsets. We use three graph layers with feature dimensions $1  1 2 8  9 6  6 4$ and a readout MLP with four hidden layers and GELU activations. The bounded readout, specified in Appendix D, uses $\epsilon = 1 0 ^ { - 5 }$ and a trainable α initialized to 0.1. Hidden states are initialized to zero and conditioned on reference configurations during an initial burn-in period. Training minimizes the trajectory mean squared error using truncated backpropagation through time and AdamW, with cosine annealing and gradient clipping. After burn-in, reference inputs are used with probability 0.2 during training; evaluation is fully autoregressive. The correlation benchmarks below use the held-out test trajectories.

We first examine the normalized single-site autocorrelation function, $A ( \tau ) = \langle \phi _ { i } ( t _ { 0 } ) \phi _ { i } ( t _ { 0 } + \tau ) \rangle / \langle \phi _ { i } ( t _ { 0 } ) ^ { 2 } \rangle$ where $\langle \cdots \rangle$ denotes an average over lattice sites i and reference times $t _ { 0 }$ for which both configurations are available. Here and below, $t , t _ { 0 } .$ and τ denote physical times. This quantity probes the temporal persistence of local CDW order and provides a direct test of the time sequences learned by the recurrent model. As shown in

Fig. 2, the GRNN closely reproduces the KPM reference results over the displayed time-lag range. While this agreement establishes the accuracy of local temporal correlations, characterizing collective phase ordering also requires examining how correlations develop across the lattice.

We therefore turn to the normalized equal-time spatial correlation function $C ( R , t )$ , which probes the organization of the CDW field into growing domains. Figure 3 compares the KPM reference dynamics and GRNN predictions at several stages of the evolution, with the separation x rescaled by the instantaneous correlation length L(t), so that the horizontal axis is $R = x / L ( t )$ The results demonstrate two complementary features. First, the GRNN reproduces the spatial correlation profiles of the microscopic dynamics, including their decay and weak anticorrelations at larger scaled separations. Second, the curves at diferent times collapse onto a common profile in both panels, consistent with dynamical scaling: the evolving domain structure is statistically selfsimilar when lengths are measured in units of L(t). Beyond capturing local temporal behavior, the GRNN thus reproduces the collective spatial correlations and scaling structure characteristic of CDW coarsening.

Finally, we apply the trained GRNN to a 200  200 lattice, demonstrating its ability to simulate extended CDW textures, as shown in Fig. 4. The evolving domain patterns are qualitatively consistent with those obtained in previous ML force-field simulations using descriptorbased and graph neural network models [26, 59]. These established features of CDW phase ordering provide a physical context for the large-scale demonstration, whose primary purpose here is to illustrate the scalability of the learned graph-recurrent dynamics. Combined with the quantitative temporal and spatial correlation benchmarks above, these results support the GRNN as an effective reduced description of collective ordering.

The dynamical-scaling analysis gives an algebraic growth law for the characteristic domain size, $L ( t ) \sim t ^ { \alpha }$ with $\alpha ~ < ~ 1 / 2$ , slower than the Allen–Cahn law for a nonconserved scalar order parameter [8]. This sub-

Allen–Cahn growth is consistent with previous ML forcefield simulations of CDW coarsening [26]. Its physical origin is a kinetic constraint imposed by microscopic electron-number conservation. Although staggered CDW order is nonconserved, nearly binary site occupations tie local domain-wall displacement to charge rearrangement; consequently, a wall segment generally cannot advance independently but must move cooperatively with compatible segments. This suppresses the independent curvature-driven motion underlying Allen–Cahn growth, while thermally induced partial occupations at the walls can relax the constraint and allow a crossover toward the Allen–Cahn regime [68, 69].

In summary, we have introduced a GRNN framework that combines local recurrent memory with graph message passing to learn the evolution of spatially distributed observables. Its hidden states encode information absent from the instantaneous physical field, while spatial coupling allows this information to influence the dynamics of neighboring sites. Learning finite-time evolution maps already allows the system to advance over intervals larger than the microscopic integration step; extending the prediction interval further could provide additional gains in computational eficiency. Because training requires trajectories of the selected observables rather than explicit microscopic equations, the framework is also suited to data-driven modeling from experimental measurements, where the underlying Hamiltonian or the relevant microscopic degrees of freedom may be incompletely known.

A particularly compelling direction is the extension to nonadiabatic dynamics. Although the present adiabatic setting already involves an emergent field with unresolved lattice motion, the electrons remain equilibrated to the instantaneous lattice configuration. In nonadiabatic or time-dependent Hartree–Fock dynamics, the electronic sector has its own evolution, and its feedback on the order parameter can depend on electronic excitations and coherences that are not determined by the instantaneous order. Learning only the order-parameter evolution would therefore require the recurrent states to encode the influence of this much larger unresolved sector. Such applications may demand larger hidden-state dimensions or more expressive memory architectures, although the required capacity is governed by the information relevant to collective evolution rather than the total number of microscopic variables. Extending the GRNN in this direction would provide a route to reduced dynamical models that retain electronic memory while avoiding explicit propagation of the full electronic state.

## ACKNOWLEDGMENTS

This work was supported by the U.S. Department of Energy, Ofice of Science, Ofice of Basic Energy Sciences, under Award No. DE-SC0020330. The authors acknowledge Research Computing at the University of Virginia for providing computational resources and technical support.

## APPENDIX

## Appendix A: Graph-recurrent update

At each lattice site $i \in V ,$ the GRNN associates the physical observable $\phi _ { i } ( t ) = ( \phi _ { i , 1 } ( t ) , \ldots , \phi _ { i , M } ( t ) )$ with a multicomponent hidden state $h _ { i } ( t )$ . The observable and hidden fields, ${ \phi } ( t ) = \{ \phi _ { i } ( t ) \} _ { i \in V }$ and $\pmb { h } ( t ) = \{ \pmb { h } _ { i } ( t ) \} _ { i \in V } ,$ evolve together through the coupled update $( \phi ( t ) , h ( t ) ) \mapsto ( \phi ( t + 1 ) , h ( t + 1 ) )$ . The hidden states retain information from earlier configurations, while graph messages couple neighboring sites. All sites share the same learned update rule.

This appendix gives a more detailed view of this update in Eqs. (1)–(4). Figure 5 follows one prediction step. Its upper path constructs spatial features from the present order-parameter configuration and advances the physical field; its lower path updates the site-resolved recurrent memory. The diagram uses the notation of the main text throughout: $t + 1$ is the next sampled time, a physical interval $\Delta t _ { \mathrm { p r e d } }$ after t. Thus $h _ { i } ( t + 1 )$ enters the prediction of $\phi _ { i } ( t + 1 )$ and is also carried into the following step.

For clarity, the neighbor averages are $\begin{array} { r } { \pmb { \overline { { v } } } _ { i } ^ { ( \ell ) } = d _ { i } ^ { - 1 } \sum _ { j \in { \mathcal { N } } _ { i } } { \pmb { v } } _ { j } ^ { ( \ell ) } } \end{array}$ and $\begin{array} { r } { \overline { { \pmb { h } } } _ { i } = d _ { i } ^ { - 1 } \sum _ { j \in \mathcal { N } _ { i } } \pmb { h } _ { j } } \end{array}$ , with $d _ { i } = | { \mathcal { N } } _ { i } |$ . Defining ${ \bf a } _ { i } ( t ) = z _ { i } ( t ) \oplus { \bf h } _ { i } ( t ) \oplus \overline { { { h } } } _ { i } ( t )$ allows the two gates in Eq. (2) to be written compactly as $\pmb { r } _ { i } = \sigma _ { \mathrm { s } } ( A _ { r } \pmb { a } _ { i } + \pmb { b } _ { r } )$ and ${ \pmb u } _ { i } = \sigma _ { \mathrm { s } } ( A _ { u } { \pmb a } _ { i } + { \pmb b } _ { u } )$ . The matrices $A _ { r }$ and $A _ { u }$ simply concatenate the W, U, and V matrices of the main text; this notation does not change the architecture. The candidate and updated memories are

$$
\begin{array} { r l } & { \widetilde { \pmb { h } } _ { i } ^ { + } = \operatorname { t a n h } \big [ { \pmb { W } } _ { h } { \pmb z } _ { i } + { U } _ { h } \big ( { \pmb r } _ { i } \odot { \pmb h } _ { i } \big ) + { V } _ { h } \overline { { \pmb { h } } } _ { i } + { \pmb b } _ { h } \big ] , } \\ & { { \pmb { h } } _ { i } ^ { + } = ( { \pmb 1 } - { \pmb u } _ { i } ) \odot { \pmb h } _ { i } + { \pmb u } _ { i } \odot \widetilde { { \pmb h } } _ { i } ^ { + } , } \end{array}\tag{A1}
$$

where quantities without an explicit time argument are evaluated at t and $h _ { i } ^ { + } = h _ { i } ( t + 1 )$ . The reset gate acts on the local previous memory in the candidate, while the neighboring memory contributes through the separate $V _ { h }$ term. An update at one site can therefore be influenced by memories stored at adjacent sites even if the current local observable happens to be the same.

![](images/b6cbadc8e954a284d2a2419101de0cbc1922b0a938495a1fbe95e26c5dc4af41.jpg)  
FIG. 5. One-step GRNN computation. Graph layers aggregate neighboring physical features into the embedding $z _ { i } ( t )$ ; the graph-coupled GRU combines this embedding with local and neighboring hidden states to produce $\pmb { h } _ { i } ^ { + } \equiv \pmb { h } _ { i } ( t + 1 )$ . The readout MLP uses $z _ { i } ( t )$ and the updated memory to predict $\delta _ { i } ( t )$ , which enters the physical update map $\mathcal { R } _ { \phi } .$ . The upper bypass carries the present physical field directly to this map. Spatial and recurrent aggregation are indicated by orange and violet paths, respectively. All learned parameters are shared across lattice sites.

With nearest-neighbor graph layers, $z _ { i } ( t )$ depends on observables out to $N _ { \mathrm { G } }$ graph hops at the same time. Repeated recurrent steps allow information in the hidden field to move farther across the lattice. The readout then gives $\pmb { \delta } _ { i } = \mathrm { M L P } ( z _ { i } \oplus \pmb { h } _ { i } ^ { + } )$ and $\phi _ { i } ( t + 1 ) = \mathcal { R } _ { \phi } [ \phi _ { i } ( t ) , \delta _ { i } ]$ . The map $\mathcal { R } _ { \phi }$ can be chosen to respect the range or geometric constraints of the physical field. These operations make the augmented $( \phi , h )$ update local in time without requiring the observable field alone to be Markovian.

## Appendix B: Holstein reference dynamics and CDW observables

We use the spinless square-lattice Hamiltonian of Eq. (5). At any fixed displacement configuration Q, the electronic part is a quadratic hopping problem with a site-dependent potential $- g Q _ { i } ;$ its density $n _ { i } [ Q ] = \langle \hat { n } _ { i } \rangle _ { Q }$ enters the force on the classical coordinate. Writing $P _ { i } = m { \dot { Q } } _ { i }$ , the adiabatic stochastic dynamics are equivalently

$$
\dot { Q } _ { i } = \frac { P _ { i } } { m } , \qquad \dot { P } _ { i } = - k Q _ { i } - \kappa \sum _ { j \in { \cal N } _ { i } } Q _ { j } + g n _ { i } [ Q ] - \frac { \gamma } { m } P _ { i } + \eta _ { i } ( t ) .\tag{B1}
$$

The force must be reevaluated as $Q$ changes. We compute its electronic contribution using the kernel polynomial method (KPM), which approximates spectral functions of the instantaneous one-electron Hamiltonian with a truncated Chebyshev expansion and evaluates local densities without full diagonalization [11, 70]. For the reference trajectories, we retain 1000 Chebyshev moments and use 100 colors in standard calculations or 400 in higher-precision evaluations (Table I). The classical evolution is integrated with a Verlet-type scheme.

For nearest-neighbor hopping on the square lattice, the electronic dispersion and nesting relation are

$$
\varepsilon _ { k } = 2 t _ { \mathrm { n n } } ( \cos k _ { x } + \cos k _ { y } ) , \qquad \varepsilon _ { k + q _ { \mathrm { C D W } } } = - \varepsilon _ { k } ,\tag{B2}
$$

where $\pmb q _ { \mathrm { C D W } } = ( \pi , \pi )$ and the lattice spacing is set to unity. The half-filled Fermi surface is therefore perfectly nested at the checkerboard ordering wave vector. This nesting underlies the instability toward staggered charge and lattice order discussed in the main text [67]. The two symmetry-related checkerboard patterns difer by an exchange of the sublattices. A local density contrast $\begin{array} { r } { \dot { n } _ { i } - \frac { 1 } { 4 } \sum _ { j \in \mathcal { N } _ { i } } n _ { j } } \end{array}$ emphasizes the modulation while suppressing a uniform density background. Multiplying by $( - 1 ) ^ { x _ { i } + y _ { i } }$ then removes the rapid checkerboard sign, giving the smooth Ising-like field $\phi _ { i }$ of Eq. (7). For a fully polarized ideal checkerboard with occupations 1 and 0, this definition yields $\phi _ { i } = + 1$ throughout one domain and $\phi _ { i } = - 1$ throughout the translated domain. Across an interface it interpolates between the two signs. Figure 6 illustrates how the strongly alternating microscopic pattern becomes a field with large, readily identifiable domains.

The equal-time structure factor provides a complementary momentum-space view of coarsening. For the staggered field we define

$$
\phi _ { \pmb q } ( t ) = \sum _ { i } \phi _ { i } ( t ) e ^ { - i \pmb q \cdot \pmb r _ { i } } , \qquad S ( \pmb q , t ) = \frac { 1 } { N } \left. \left| \phi _ { \pmb q } ( t ) \right| ^ { 2 } \right. ,\tag{B3}
$$

TABLE I. Model and reference-trajectory settings. The damping value refers to the integrator setting; no numerical value for γ in Eq. (B1) is inferred from it.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Hopping  $t _ { \mathrm { n n } } ;$  coupling g</td><td>-1; 3.5</td></tr><tr><td>Elastic constants k, κ; mass m</td><td>1, 0.18; 5</td></tr><tr><td>Filling; temperature  $k _ { \mathrm { B } } T$ </td><td> $\overline { { n } } = 1 / 2 ; 0 . 3$ </td></tr><tr><td>Reference lattice</td><td> $4 0 \times 4 0 ~ \mathrm { s i t e s }$ </td></tr><tr><td>Integration step  $\delta t ;$  integrator damping setting</td><td>0.04; 0.2</td></tr><tr><td>Steps per trajectory; final time</td><td> $1 0 ^ { 4 } ;$  400</td></tr><tr><td>KPM Chebyshev order</td><td>1000</td></tr><tr><td>KPM colors, standard / higher precision</td><td>100  / 400</td></tr><tr><td>Snapshot spacing; prediction interval  $\Delta t _ { \mathrm { p r e d } }$ </td><td>20 steps; 0.8</td></tr><tr><td>Independent trajectories, train / validation / test</td><td>16 / 4 / 10</td></tr></table>

![](images/83f5a173aa3ee0b5956d926b681ef5e2e610f90123f3a5b51ab71f8a1a3cc148.jpg)  
FIG. 6. Microscopic lattice pattern and emergent Ising-like CDW field in a representative 40 × 40 configuration. Left: displacement $Q _ { i }$ , whose alternating pattern tracks the local charge modulation through electron–lattice coupling. Right: staggered neighbor-density contrast $\begin{array} { r } { \bar { \phi _ { i } } \stackrel { - } { = } ( - 1 ) ^ { x _ { i } + y _ { i } } \big ( n _ { i } - \frac { 1 } { 4 } \sum _ { j \in \mathcal { N } _ { i } } n _ { j } \big ) } \end{array}$ evaluated for the same configuration. The staggered factor removes the checkerboard oscillation, so red and blue regions mark the two symmetry-related CDW domains; intermediate values occur near domain walls. The color bars show the values plotted in each panel.

where N is the number of lattice sites and the brackets indicate an average over independent trajectories. Because the checkerboard factor is already included in $\phi _ { i } .$ , the CDW peak occurs at $\pmb q = \mathbf 0$ in this representation; in the density structure factor the corresponding peak is at $( \pi , \pi )$ . Figure 7 compares the KPM and GRNN maps at three times. The broad initial peak becomes narrower as spatial correlations extend to larger distances, and the GRNN reproduces the evolution of its momentum-space shape.

## Appendix C: Kernel Polynomial Method for Reference Data Generation

The reference trajectories used for training and evaluating the GRNN are generated by integrating the Langevin equations of motion for the classical lattice degrees of freedom, with the electronic forces evaluated using the kerne polynomial method (KPM) [11, 70]. This approach avoids the computationally expensive diagonalization of the electronic Hamiltonian by approximating functions of the single-particle Hamiltonian through a Chebyshev polynomia expansion.

For a given configuration of lattice displacements, the single-particle Hamiltonian is rescaled such that its spectrum lies within the interval [ 1, 1]:

$$
\widetilde { \mathcal { H } } = \frac { \mathcal { H } - b } { a } ,\tag{C1}
$$

![](images/587e7d330d1bb877148fa9727a300f998fd132fb2650e5e9be1dee9bb7b2c69b.jpg)  
FIG. 7. Time-dependent structure factor $S ( \pmb q , t )$ of the staggered CDW field on a 40 × 40 lattice. The upper row shows KPM reference trajectories and the lower row shows autoregressive GRNN predictions at $t = 0 ,$ 40, and 400 (left to right). Momentum is measured relative to the checkerboard ordering wave vector, so the ordering peak of the staggered field is centered at $\pmb q = \mathbf 0$ Its narrowing as time increases reflects growing CDW domains; the GRNN reproduces the corresponding momentum-space pattern. Dashed squares outline the first Brillouin zone.

where a and b are determined by the spectral bounds. A function of the Hamiltonian can then be approximated as

$$
f ( \mathcal { H } ) \approx \sum _ { m = 0 } ^ { M - 1 } c _ { m } T _ { m } ( \widetilde { \mathcal { H } } ) ,\tag{C2}
$$

where $T _ { m }$ denotes the Chebyshev polynomial of order m, M is the expansion order, and $c _ { m }$ are the corresponding expansion coeficients, including any damping factors. The Chebyshev polynomials are evaluated recursively through

$$
T _ { m + 1 } ( \widetilde { \mathcal { H } } ) = 2 \widetilde { \mathcal { H } } T _ { m } ( \widetilde { \mathcal { H } } ) - T _ { m - 1 } ( \widetilde { \mathcal { H } } ) .\tag{C3}
$$

For noninteracting electrons coupled to classical lattice displacements, the electronic force acting on the displacement $x _ { i }$ can be expressed in terms of the electronic free energy $F _ { \mathrm { e l } }$ as

$$
F _ { i } ^ { \mathrm { e l } } = - \frac { \partial F _ { \mathrm { e l } } } { \partial x _ { i } } = - \mathrm { T r } \left[ f _ { \mathrm { F D } } ( \mathcal { H } ) \frac { \partial \mathcal { H } } { \partial x _ { i } } \right] ,\tag{C4}
$$

where $f _ { \mathrm { F D } }$ is the Fermi–Dirac distribution and the trace is over the single-particle Hilbert space. These forces are evaluated using KPM and incorporated into the Langevin equations governing the lattice dynamics.

Because the Chebyshev recurrence requires only sparse matrix-vector multiplications, the computational cost scales linearly with the number of lattice sites for fixed expansion order and stochastic sampling parameters[70]. The resulting microscopic trajectories are subsequently mapped onto the local charge-density-wave order parameter, providing the training data and reference benchmarks for the GRNN.

## Appendix D: Network implementation and training for the Holstein demonstration

For the Holstein demonstration, the input is one scalar $\phi _ { i }$ per lattice site. Three graph aggregation layers map channel dimensions $1  1 2 8  9 6  6 4$ , with GELU as the graph-layer nonlinearity. Their output $z _ { i }$ enters the graph-coupled GRU described above. A sitewise MLP with four hidden layers and GELU activations reads out a

scalar increment $\delta _ { i }$ from $z _ { i } \oplus h _ { i } ^ { + }$ . The same graph, recurrent, and readout parameters are used at every site, so their count does not increase with lattice area.

The CDW field defined in Eq. (7) is bounded by $- 1 \leq \phi _ { i } \leq 1$ . For this scalar observable $( M = 1 )$ , we specialize the general physical readout to

$$
\mathcal { R } _ { \phi } ( \phi , \delta ) = \operatorname { t a n h } \left[ \operatorname { a r c t a n h } ( \operatorname { c l i p } ( \phi , - 1 + \epsilon , 1 - \epsilon ) ) + \alpha \delta \right] ,\tag{D1}
$$

where $\epsilon = 1 0 ^ { - 5 }$ avoids the singular endpoints and α is a learned scale initialized to 0.1. This form applies the learned change in the transformed coordinate arctanh(ϕ) while keeping the resulting field within its allowed range. It is a choice specific to the bounded CDW observable; the general GRNN update does not require this readout form.

TABLE II. GRNN architecture and training settings used for the Holstein demonstration.
<table><tr><td>Component or procedure</td><td>Setting</td></tr><tr><td>Observable channels</td><td> $M = 1$ </td></tr><tr><td>Graph layers and channel sequence</td><td> $3 ; 1 \to 1 2 8 \to 9 6 \to 6 4$ </td></tr><tr><td>Graph activation and aggregation</td><td> $\operatorname { G E L U } ;$  mean over nearest neighbors</td></tr><tr><td>Recurrent unit</td><td>GRU with local and neighbor hidden inputs</td></tr><tr><td>Readout</td><td>MLP with four hidden layers, GELU</td></tr><tr><td>Bounded residual parameters</td><td> $\epsilon = 1 0 ^ { - 5 }$  ; α initialized at 0.1</td></tr><tr><td>Teacher forcing after burn-in</td><td>probability  $p = 0 . 2$ </td></tr><tr><td>Optimization</td><td>AdamW, cosine learning-rate annealing</td></tr><tr><td>Sequence gradients</td><td> $\mathrm { T B P T T } ,$  gradient clipping</td></tr><tr><td>Held-out evaluation</td><td>Autoregressive after burn-in</td></tr></table>

Each trajectory is initialized with $\mathbf { } h _ { i } = \mathbf { 0 }$ at every site. During an initial burn-in segment, the reference configurations are supplied as inputs so that the hidden states can encode information from the trajectory before the mode is trained to predict freely; these burn-in steps are excluded from the loss. After burn-in, training uses a mixture of reference and predicted configurations: at each step, the reference field is supplied with probability $p = 0 . 2 .$ , and otherwise the model’s previous prediction is used. This exposes the model to its own prediction errors during training while retaining some reference inputs. During validation and testing, reference configurations are used only during burn-in; subsequent evolution is fully autoregressive. Thus the correlations reported in the main text and Fig. 7 assess extended predictions on held-out trajectories, rather than repeated one-step predictions conditioned on exact intermediate fields.

Let $\phi _ { i } ^ { ( a ) } ( t )$ denote the reference field for trajectory a and $\widehat { \phi } _ { i } ^ { ( a ) } ( t + 1 )$ the corresponding GRNN prediction. The trajectory mean-squared-error objective, evaluated after burn-in, is

$$
\mathcal { L } _ { \mathrm { t r a j } } = \frac { 1 } { B N } \sum _ { a = 1 } ^ { B } \frac { 1 } { | \mathcal { T } _ { a } | } \sum _ { t \in \mathcal { T } _ { a } } \sum _ { i = 1 } ^ { N } \left[ \widehat { \phi } _ { i } ^ { ( a ) } ( t + 1 ) - \phi _ { i } ^ { ( a ) } ( t + 1 ) \right] ^ { 2 } ,\tag{D2}
$$

where B is the number of trajectories in the batch, $N$ is the number of lattice sites, and $\mathcal { T } _ { a }$ contains the prediction times retained after burn-in. The time average is taken separately for each trajectory before averaging over the batch, so each trajectory contributes equally to the objective.

Training uses truncated backpropagation through time (TBPTT) to limit the memory and computation required for long sequences. The forward evolution proceeds continuously across successive temporal chunks: the predicted field and hidden states at the end of one chunk provide the initial values for the next. At each chunk boundary, both are detached from the computational graph. Their numerical values therefore carry the trajectory history forward, while gradients propagate only within each chunk. Gradients are accumulated across chunks over the training interva before updating the parameters. We use AdamW with cosine learning-rate annealing and gradient clipping. Complete trajectories, rather than individual snapshots, are assigned to the training, validation, and test sets, preventing configurations from the same time sequence from appearing in diferent subsets.

## Appendix E: Ablation Study of Candidate Models

To assess the importance of the architectural components of the proposed graph-recurrent neural network (GRNN), we perform an ablation study involving five candidate models. In particular, we investigate the role of recurrent memory and the incorporation of spatially aggregated hidden states into the recurrent updates. The models are evaluated by comparing the temporal autocorrelation functions obtained from autoregressive simulations against the reference results computed using the kernel polynomial method (KPM).

![](images/66b29d067d79798f643ac7c32efa25cafa0513772a134d0b225a3bf22dc7f2cc.jpg)  
FIG. 8. Ablation study of the graph-recurrent neural network architecture. Temporal autocorrelation functions $A ( \tau )$ predicted by five candidate models $\left( C _ { 1 } { - } C _ { 5 } \right)$ are compared with the KPM reference. The candidates difer in their use of recurrent memory and spatially aggregated hidden states, as summarized in Table III. The full GRNN architecture $( C _ { 5 } )$ and the model with spatially coupled recurrent gates (C<sub>3</sub>) exhibit the closest agreement with the reference dynamics.

TABLE III. Comparison of the candidate architectures considered in the ablation study. All models use the same GNN backbone with feature dimensions $1 \to 1 2 8 \to 9 6 \to 6 4$ . Recurrent models employ a hidden-state dimension of 96. The RMSE quantifies the discrepancy between the predicted temporal autocorrelation function and the KPM reference.
<table><tr><td>Model</td><td>GRU</td><td> $\overline { { \mathbf { h } } } _ { i }$  in gates</td><td> $\overline { { \pmb { h } } } _ { i }$  in candidate</td><td>RMSE</td></tr><tr><td> $C _ { 1 }$ </td><td>No</td><td>一</td><td>一</td><td>0.032842</td></tr><tr><td> $C _ { 2 }$ </td><td>Yes</td><td>No</td><td>No</td><td>0.018433</td></tr><tr><td> $C _ { 3 }$ </td><td>Yes</td><td>Yes</td><td>No</td><td>0.006224</td></tr><tr><td> $C _ { 4 }$ </td><td>Yes</td><td>No</td><td>Yes</td><td>0.017147</td></tr><tr><td> $C _ { 5 }$ </td><td>Yes</td><td>Yes</td><td>Yes</td><td>0.006148</td></tr></table>

To ensure a consistent comparison, all models employ the same GNN backbone, consisting of three message-passing layers with feature dimensions $1 \to 1 2 8 \to 9 6 \to 6 4$ . For models containing recurrent units, the hidden-state dimension is fixed at 96. The training procedure is also kept identical across all configurations, using sequences of 50 time steps, a burn-in period of 20 steps, and a teacher-forcing probability of 0.2. All models are trained for 200 epochs with a batch size of 4 and an initial learning rate of $2 \times 1 0 ^ { - 4 }$ . For the memoryless model, the burn-in procedure does not initialize any recurrent hidden state.

The first candidate, denoted by $C _ { 1 }$ , consists of a conventional GNN without recurrent connections. This model performs spatial message passing and predicts the evolution of the order parameter autoregressively. The second candidate, $C _ { 2 } .$ combines the GNN backbone with a conventional GRU, whose gates and candidate hidden-state updates depend only on the local GNN embedding $z _ { i } ( t )$ and the local hidden state $h _ { i } ( t )$

The remaining architectures incorporate the spatially aggregated hidden state $\overline { { \mathbf { h } } } _ { i } ( t )$ into diferent components of the recurrent unit. In model $C _ { 3 } .$ , the aggregated hidden state enters the update and reset gates but is excluded from the candidate hidden-state update. Conversely, model $C _ { 4 }$ incorporates $\overline { { h } } _ { i } ( t )$ only into the candidate hidden-state update, while retaining conventional local gates. Finally, $C _ { 5 }$ corresponds to the full GRNN architecture introduced in Eqs. (2) and (3), in which the aggregated hidden state contributes to both the recurrent gates and the candidate hidden-state update.

The resulting temporal autocorrelation functions are shown in Fig. 8, and their root-mean-square errors (RMSEs) relative to the KPM reference are summarized in Table III. The memoryless GNN (C<sub>1</sub>) exhibits the largest discrepancy, with an RMSE of $3 . 2 8 \times 1 0 ^ { - 2 }$ . Introducing local recurrent memory $\left( C _ { 2 } \right)$ reduces this error to $1 . 8 4 \times 1 0 ^ { - 2 }$ , demonstrating the advantage of incorporating temporal information into the model.

A more substantial improvement is observed for $C _ { 3 }$ , which incorporates spatially aggregated hidden states into the recurrent gates and achieves an RMSE of $6 . 2 2 \times 1 0 ^ { - 3 }$ . In contrast, incorporating the same spatial information only into the candidate hidden-state update $\left( C _ { 4 } \right)$ yields an RMSE of $1 . 7 1 \times 1 0 ^ { - 2 }$ , representing only a modest improvement over the conventional GRU. These results indicate that spatial coupling in the update and reset gates plays a particularly important role in reproducing the temporal correlations of the microscopic dynamics.

The full GRNN architecture $\left( C _ { 5 } \right)$ achieves the lowest RMSE, $6 . 1 5 \times 1 0 ^ { - 3 }$ , although its improvement over $C _ { 3 }$ is relatively small. This observation suggests that the spatially coupled recurrent gates are primarily responsible for the improved accuracy, while the additional coupling in the candidate hidden-state update provides a smaller incremental benefit. Overall, the ablation study supports the use of spatially coupled recurrent memory for learning the efective dynamics of the collective order paramete

[1] A. Moreo, S. Yunoki, and E. Dagotto, Phase separation scenario for manganese oxides and related materials, Science 283, 2034 (1999).

[2] A. Moreo, M. Mayr, A. Feiguin, S. Yunoki, and E. Dagotto, Giant cluster coexistence in doped manganites and other compounds, Physical Review Letters 84, 5568 (2000).

[3] S. R. White and D. J. Scalapino, Density matrix renormalization group study of the striped phase in the 2D t-J model, Physical Review Letters 80, 1272 (1998).

[4] B.-X. Zheng, C.-M. Chung, P. Corboz, G. Ehlers, M.-P. Qin, R. M. Noack, H. Shi, S. R. White, S. Zhang, and G. K.-L. Chan, Stripe order in the underdoped region of the two-dimensional Hubbard model, Science 358, 1155 (2017).

[5] E. W. Huang, C. B. Mendl, S. Liu, S. Johnston, H.-C. Jiang, B. Moritz, and T. P. Devereaux, Numerical evidence of fluctuating stripes in the normal state of high-T<sub>c</sub> cuprate superconductors, Science 358, 1161 (2017).

[6] A. J. Bray, Theory of phase-ordering kinetics, Advances in Physics 43, 357 (1994).

[7] P. C. Hohenberg and B. I. Halperin, Theory of dynamic critical phenomena, Reviews of Modern Physics 49, 435 (1977).

[8] S. M. Allen and J. W. Cahn, A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening, Acta Metallurgica 27, 1085 (1979).

[9] J. W. Cahn and J. E. Hilliard, Free energy of a nonuniform system. I. interfacial free energy, The Journal of Chemical Physics 28, 258 (1958).

[10] L.-Q. Chen, Phase-field models for microstructure evolution, Annual Review of Materials Research 32, 113 (2002).

[11] K. Barros and Y. Kato, Eficient Langevin simulation of coupled classical fields and fermions, Physical Review B 88, 235101 (2013).

[12] G.-W. Chern, K. Barros, Z. Wang, H. Suwa, and C. D. Batista, Semiclassical dynamics of spin density waves, Physical Review B 97, 035120 (2018).

[13] J. Behler and M. Parrinello, Generalized neural-network representation of high-dimensional potential-energy surfaces, Physical Review Letters 98, 146401 (2007).

[14] A. P. Bart´ok, M. C. Payne, R. Kondor, and G. Cs´anyi, Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons, Physical Review Letters 104, 136403 (2010).

[15] L. Zhang, J. Han, H. Wang, R. Car, and W. E, Deep potential molecular dynamics: A scalable model with the accuracy of quantum mechanics, Physical Review Letters 120, 143001 (2018).

[16] O. T. Unke, S. Chmiela, H. E. Sauceda, M. Gastegger, I. Poltavsky, K. T. Sch¨utt, A. Tkatchenko, and K.-R. M¨uller, Machine learning force fields, Chemical Reviews 121, 10142 (2021).

[17] S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky, E(3)-equivariant graph neural networks for data-eficient and accurate interatomic potentials, Nature Communications 13, 2453 (2022).

[18] I. Batatia, D. P. Kov´acs, G. N. C. Simm, C. Ortner, and G. Cs´anyi, MACE: Higher order equivariant message passing neural networks for fast and accurate force fields, in Advances in Neural Information Processing Systems, Vol. 35 (2022) pp. 11423–11436.

[19] H. Suwa, J. S. Smith, N. Lubbers, C. D. Batista, G.- W. Chern, and K. Barros, Machine learning for molecular dynamics with strongly correlated electrons, Physical Review B 99, 161107 (2019).

[20] G.-W. Chern, Y. Fan, S. Zhang, and P. Zhang, Machinelearning modeling of magnetization dynamics in quasiequilibrium and driven metallic spin systems, Journal of Magnetism and Magnetic Materials 642, 173898 (2026).

[21] G.-W. Chern, Y. Fan, S. Zhang, and P. Zhang, Machinelearning force-field models for dynamical simulations of metallic magnets, AIP Advances 16, 025031 (2026).

[22] P. Zhang and G.-W. Chern, Machine learning nonequilibrium electron forces for spin dynamics of itinerant magnets, npj Computational Materials 9, 32 (2023).

[23] X. Cheng, S. Zhang, P. C. H. Nguyen, S. Azarfar, G.-W. Chern, and S. S. Baek, Convolutional neural networks for large-scale dynamical modeling of itinerant magnets, Physical Review Research 5, 033188 (2023).

[24] P. Zhang and G.-W. Chern, Arrested phase separation in double-exchange models: Large-scale simulation enabled by machine learning, Physical Review Letters 127, 146401 (2021).

[25] S. Zhang, P. Zhang, and G.-W. Chern, Anomalous phase separation in a correlated electron system: Machinelearning-enabled large-scale kinetic Monte Carlo simulations, Proceedings of the National Academy of Sciences 119, e2119957119 (2022).

[26] C. Cheng, S. Zhang, and G.-W. Chern, Machine learn-

ing for phase ordering dynamics of charge density waves, Physical Review B 108, 014301 (2023).

[27] S. Ghosh, S. Zhang, C. Cheng, and G.-W. Chern, Kinetics of orbital ordering in cooperative Jahn-Teller models: Machine-learning enabled large-scale simulations, Physical Review Materials 8, 123602 (2024).

[28] Y. Fan, S. Zhang, and G.-W. Chern, Coarsening of chiral domains in itinerant electron magnets: A machine learning force-field approach, Physical Review B 110, 245105 (2024).

[29] A. Tyberg, Y. Fan, and G.-W. Chern, Machine learning force field model for kinetic Monte Carlo simulations of itinerant Ising magnets, Physical Review B 111, 235132 (2025).

[30] J. L. Elman, Finding structure in time, Cognitive Science 14, 179 (1990).

[31] Y. Yu, X. Si, C. Hu, and J. Zhang, A review of recurrent neural networks: LSTM cells and network architectures, Neural Computation 31, 1235 (2019).

[32] A. Sherstinsky, Fundamentals of recurrent neural network (RNN) and long short-term memory (LSTM) network, Physica D: Nonlinear Phenomena 404, 132306 (2020).

[33] S. Hochreiter and J. Schmidhuber, Long short-term memory, Neural Computation 9, 1735 (1997).

[34] H. Jaeger and H. Haas, Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication, Science 304, 78 (2004).

[35] K. Cho, B. van Merri¨enboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, Learning phrase representations using RNN encoder–decoder for statistical machine translation, in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2014) pp. 1724–1734.

[36] J. Pathak, B. Hunt, M. Girvan, Z. Lu, and E. Ott, Modelfree prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach, Physical Review Letters 120, 024102 (2018).

[37] P. R. Vlachas, W. Byeon, Z. Y. Wan, T. P. Sapsis, and P. Koumoutsakos, Data-driven forecasting of highdimensional chaotic systems with long short-term memory networks, Proceedings of the Royal Society A 474, 20170844 (2018).

[38] A. Chattopadhyay, P. Hassanzadeh, and D. Subramanian, Data-driven predictions of a multiscale Lorenz 96 chaotic system using machine-learning methods: reservoir computing, artificial neural network, and long shortterm memory network, Nonlinear Processes in Geophysics 27, 373 (2020).

[39] S.-T. Tsai, E.-J. Kuo, and P. Tiwary, Learning molecular dynamics with simple language model built upon long short-term memory neural network, Nature Communications 11, 5115 (2020).

[40] L. Banchi, E. Grant, A. Rocchetto, and S. Severini, Modelling non-Markovian quantum processes with recurrent neural networks, New Journal of Physics 20, 123030 (2018).

[41] E. Flurin, L. S. Martin, S. Hacohen-Gourgy, and I. Siddiqi, Using a recurrent neural network to reconstruct quantum dynamics of a superconducting qubit from physical observations, Physical Review X 10, 011006 (2020).

[42] K. Lin, J. Peng, F. L. Gu, and Z. Lan, Simulation of open quantum dynamics with bootstrap-based long short-term

memory recurrent neural network, The Journal of Physical Chemistry Letters 12, 10225 (2021).

[43] R. Zwanzig, Memory efects in irreversible thermodynamics, Physical Review 124, 983 (1961).

[44] H. Mori, Transport, collective motion, and brownian motion, Progress of Theoretical Physics 33, 423 (1965).

[45] A. J. Chorin, O. H. Hald, and R. Kupferman, Optimal prediction and the Mori–Zwanzig representation of irreversible processes, Proceedings of the National Academy of Sciences USA 97, 2968 (2000).

[46] F. Takens, Detecting strange attractors in turbulence, in Dynamical Systems and Turbulence, Warwick 1980, Lecture Notes in Mathematics, Vol. 898, edited by D. Rand and L.-S. Young (Springer, Berlin, Heidelberg, 1981) pp. 366–381.

[47] H. Jang and G.-W. Chern, Transformer learning of chaotic collective dynamics in many-body systems (2026), arXiv:2601.19080 [physics.comp-ph].

[48] L. E. Herrera Rodr´ıguez and A. A. Kananenka, A short trajectory is all you need: A transformer-based model for long-time dissipative quantum dynamics, The Journal of Chemical Physics 161, 171101 (2024).

[49] G. Koppe, H. Toutounji, P. Kirsch, S. Lis, and D. Durstewitz, Identifying nonlinear dynamical systems via generative recurrent neural networks with applications to fMRI, PLOS Computational Biology 15, e1007263 (2019).

[50] P. R. Vlachas, G. Arampatzis, C. Uhler, and P. Koumoutsakos, Multiscale simulations of complex systems by learning their efective dynamics, Nature Machine Intelligence 4, 359 (2022).

[51] D. Durstewitz, G. Koppe, and M. I. Thurm, Reconstructing computational system dynamics from neural data with recurrent neural networks, Nature Reviews Neuroscience 24, 693 (2023).

[52] A. Ning, L. Yang, and G.-W. Chern, Recurrent convolutional neural networks for modeling nonadiabatic dynamics of quantum-classical systems, Physical Review E 113, 015307 (2026).

[53] C. Dinh, Y. Fan, and G.-W. Chern, Echo state network for coarsening dynamics of charge density waves, Physical Review E 112, 065302 (2025).

[54] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, Empirical evaluation of gated recurrent neural networks on sequence modeling (2014), arXiv:1412.3555 [cs.NE].

[55] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, The graph neural network model, IEEE Transactions on Neural Networks 20, 61 (2009).

[56] T. N. Kipf and M. Welling, Semi-supervised classification with graph convolutional networks, in International Conference on Learning Representations (ICLR) (2017) arXiv:1609.02907.

[57] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, Neural message passing for quantum chemistry, in Proceedings of the 34th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 70 (PMLR, 2017) pp. 1263–1272.

[58] A. Sanchez-Gonzalez, J. Godwin, T. Pfaf, R. Ying, J. Leskovec, and P. Battaglia, Learning to simulate complex physics with graph networks, in Proceedings of the 37th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 119 (PMLR, 2020) pp. 8459–8468.

[59] Y. Fan and G.-W. Chern, Graph neural network force fields for adiabatic dynamics of lattice hamiltonians, APL

Machine Learning 4, 036107 (2026).

[60] S. S. Bakshi, Y. Fan, and G.-W. Chern, Machine learning modeling of charge-density-wave recovery after laser melting, Physical Review B 114, 165110 (2026).

[61] Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel, Gated graph sequence neural networks, in International Conference on Learning Representations (ICLR) (2016) arXiv:1511.05493.

[62] Y. Li, R. Yu, C. Shahabi, and Y. Liu, Difusion convolutional recurrent neural network: Data-driven traffic forecasting, in International Conference on Learning Representations (ICLR) (2018) arXiv:1707.01926.

[63] T. Holstein, Studies of polaron motion: Part I. the molecular-crystal model, Annals of Physics 8, 325 (1959).

[64] R. T. Scalettar, N. E. Bickers, and D. J. Scalapino, Competition of pairing and peierls–charge-density-wave correlations in a two-dimensional electron-phonon model, Physical Review B 40, 197 (1989).

[65] R. M. Noack, D. J. Scalapino, and R. T. Scalettar, Charge-density-wave and pairing susceptibilities in a

two-dimensional electron-phonon model, Physical Review Letters 66, 778 (1991).

[66] M. Hohenadler and G. G. Batrouni, Dominant charge density wave correlations in the Holstein model on the half-filled square lattice, Physical Review B 100, 165114 (2019).

[67] M. Weber and M. Hohenadler, Two-dimensional Holstein–Hubbard model: Critical temperature, Ising universality, and bipolaron liquid, Physical Review B 98, 085405 (2018).

[68] H. Jang, Y. Yang, and G.-W. Chern, Anomalous coarsening and nonlinear difusion of kinks in a one-dimensional quasiclassical Holstein model, Physical Review E 114, 014225 (2026).

[69] A. Bora, S. S. Bakshi, H. Jang, and G.-W. Chern, Cooperative domain-wall dynamics in a two-dimensional quasiclassical Holstein model (2026), arXiv:2609.28436.

[70] A. Weiße, G. Wellein, A. Alvermann, and H. Fehske, The kernel polynomial method, Reviews of Modern Physics 78, 275 (2006).