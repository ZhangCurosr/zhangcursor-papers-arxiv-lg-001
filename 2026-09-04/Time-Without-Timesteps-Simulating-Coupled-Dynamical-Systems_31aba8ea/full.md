# Time Without Timesteps: Simulating Coupled Dynamical Systems via Self-Consistency

Liyu Zerihun

Mark Shinyoung Lee

## Abstract

Numerical simulation of dynamical systems is usually organized as a causal march through time: each state is computed from the previous state, and the full trajectory is obtained by repeated local updates. This paper explores a diferent computational formulation for coupled dynamical systems. We train neural surrogates that map full input trajectories and initial conditions directly to full output trajectories for individual subsystem types. Building on classical waveform relaxation, we replace the numerical integration performed within each subsystem update with these learned trajectory operators. Coupled systems are then assembled by enforcing self-consistency between the trajectories predicted by each subsystem and those supplied as inputs to its neighbors. Simulation therefore becomes a fixed-point problem over complete trajectories rather than an explicit timestep-by-timestep rollout.

We demonstrate the approach on van der Pol oscillators and Hodgkin–Huxley neuron networks. The sequential depth of the computation becomes the number of solver iterations rather than the number of timesteps: we recover coupled trajectories in 4–10 Newton iterations where the reference integrator takes 1500 steps. The gradient of the simulation likewise loses its time recursion: it is the solution of a linear system, not a backward pass through the integrator, and we solve it by GMRES at memory independent of solver depth. We show that a single scalar measured from the learned operator—the spectral radius of its Jacobian—predicts in advance where the coupled solve will converge, and that this prediction holds across both coupling strength and stifness. We also identify a regime in which the implicit gradient is the only one available: past the contraction boundary, unrolled backpropagation diverges and a Neumann adjoint fails, while the implicit gradient remains correct to 0.04% against finite diferences.

The method is not presented as a replacement for classical solvers. It is an early exploration of a trajectory-level simulation paradigm in which learned module operators are composed through self-consistency. We report regimes in which the approach succeeds and regimes in which surrogate error causes degradation, and we identify surrogate accuracy, fixed-point conditioning, and training-distribution coverage as the main practical bottlenecks.

## 1 Introduction

To simulate a diferential equation, we almost always march: given the current state, a numerical integrator estimates the next state, and repeating this local update produces a trajectory.

That procedure is powerful, mature, and often the right tool. But it is not the only way to view the computation. A diferential equation, together with its initial or boundary conditions, defines a set of admissible trajectories, and the physical solution is the one that satisfies the dynamics and all coupling constraints simultaneously. In this sense the trajectory is a global object selected by consistency; the order in which a solver happens to construct it is incidental.

This distinction matters most in coupled systems. When subsystems interact, each component’s trajectory depends on signals produced by its neighbors. If those driving signals were known, each subsystem could be solved independently. The dificulty is circular: the driving signals are themselves functions of the unknown trajectories. Standard solvers resolve this circularity locally in time by advancing all coupled states together. We instead ask whether the circularity can be resolved globally, at the level of whole trajectories.

That question has an established answer in numerical analysis. Waveform relaxation, introduced for circuit simulation [12], solves exactly this fixed point: each subsystem is integrated over the whole time window under an assumed input waveform, and the process is repeated until the waveforms agree. The method never became a general-purpose tool, and one reason is cost. Every relaxation sweep requires a full numerical integration of every subsystem, so each iteration is as expensive as a complete simulation, and the derivative of a sweep is expensive for the same reason.

Time Without Timesteps (TWT) asks what changes when that inner integration is replaced by a learned trajectory operator. For each subsystem type, we train a neural surrogate that maps a full driving input trajectory and initial condition to a full output trajectory. A coupled system is then assembled by requiring mutual agreement: each subsystem’s output must equal the trajectory predicted by its surrogate under the inputs induced by neighboring subsystems.

What motivates the construction is the structure of the computation, not raw speed.

One consequence is sequential depth. A conventional solve is sequential in the number of timesteps, because state at step n + 1 requires state at step n. A fixed-point solve is sequential in the number of iterations, and each iteration evaluates all module operators independently. In the experiments below the coupled solve converges in 4–10 Newton iterations on trajectories represented over 1500 timesteps.

The consequence that cuts deeper is diferentiation. Diferentiable simulators already exist, and gradients through coupled ODEs are routine: one either unrolls the integrator, at memory linear in the number of steps, or solves the continuous adjoint, at constant memory but still with a backward sweep across the full horizon. Both inherit the sequential structure of the forward march. At a fixed point the gradient is instead the solution of a linear system, $( I - J ^ { \top } ) \mathbf { v } = \nabla \mathcal { I }$ , an object with no time recursion in it at all. It can be solved by Krylov methods whose matrix-vector products are single applications of the module operators. Making those products cheap is precisely what the learned operator buys, and we show below that with an exact integrator in the same role the second-order solver is roughly 100× slower than simple relaxation, which is why the equilibrium view has not been practical before.

The method does not remove time from the representation. Trajectories are still represented on discrete grids; for the oscillator system we represent them in a Chebyshev coeficient basis, which decouples the model’s input dimension from the number of timesteps but does not eliminate the time axis. What changes is the outer computational structure: the solver no longer constructs the trajectory by applying a local update rule for every timestep.

This paper makes four contributions. First, it formulates coupled dynamical simulation as a trajectory-level self-consistency problem and situates it explicitly as waveform relaxation with learned subsystem operators. Second, it shows that the classical relaxation iteration is not the right solver for learned operators, and that a Jacobian-free Newton–Krylov method removes the contraction requirement that limits it. Third, it establishes that a single measured quantity—the spectral radius of the learned operator’s Jacobian—predicts the convergence boundary of the coupled solve in advance, across both coupling strength and stifness. Fourth, it demonstrates implicit diferentiation through the fixed point, validated three independent ways, including a regime where no other gradient is available.

We do not claim that TWT replaces classical numerical integration. Classical solvers remain the standard tool for accuracy, reliability, and broad applicability, and the systems studied here are small. The purpose of this work is to make a diferent computational paradigm precise: learn module-level trajectory operators, compose them through self-consistency, and diferentiate through

the resulting fixed point.

## 2 Background and Related Work

## 2.1 Waveform Relaxation

Waveform relaxation [12] decomposes a coupled system into subsystems and iterates on whole waveforms: given a current guess for every subsystem’s trajectory over [0, T], each subsystem is integrated independently under the inputs those trajectories induce, and the result becomes the next guess. The iteration converges superlinearly on finite windows for Lipschitz systems [17], with a contraction factor that degrades as the window lengthens—which is why practical implementations window the time axis. Parallel-in-time methods such as Parareal pursue the related goal of decoupling sequential depth from timestep count by decomposing the time axis itself [14, 6]; TWT instead decomposes by subsystem.

TWT is this algorithm with the inner integration replaced by a learned operator. The consequences go beyond cost. Because the learned operator is diferentiable and its Jacobian-vector products cost one network evaluation, solvers that are impractical for classical waveform relaxation become available: we use Newton–Krylov for the forward solve and GMRES for the adjoint, both of which need many Jacobian applications per step.

## 2.2 Trajectories as Global Objects

The view of trajectories as global objects has deep roots in physics. Hamilton’s principle selects physical trajectories by extremizing an action functional over a space of possible paths. Variational integrators discretize this principle directly and preserve geometric structure such as symplecticity and momentum maps [16]. Physics-informed neural networks (PINNs) also adopt a global view by parameterizing a solution function and minimizing diferential-equation residuals over a domain [18].

These approaches difer from TWT in important ways. Variational integrators still construct the trajectory through a discretized temporal scheme. PINNs optimize a new solution for each problem instance and often face ill-conditioned losses, spectral bias, and dificulty balancing residual and boundary terms [11, 21]. TWT instead amortizes subsystem solves into learned trajectory operators and uses self-consistency to compose them at inference time.

## 2.3 Neural Surrogates and Neural Operators

A growing body of work trains neural networks to approximate solution maps for diferential equations. Neural ordinary diferential equations learn a vector field and integrate it with a numerical solver, often using adjoint methods for memory-eficient diferentiation [4]. Neural operators such as DeepONet and the Fourier Neural Operator learn mappings between function spaces, allowing full solution fields to be predicted from inputs such as forcing functions or initial conditions [15, 13, 10].

TWT is closest in spirit to neural operator learning, because each module surrogate maps an input trajectory to an output trajectory. The diference is compositional: TWT uses trajectory operators as modules inside a coupled graph and enforces agreement across modules through a fixed-point solve. The problem is therefore not just learning an isolated solution operator, but composing several learned operators into a mutually consistent coupled simulation. As we show in Section 4.3, this makes a property of the operator that isolated-accuracy metrics do not measure—the norm of its Jacobian—directly relevant, and the two properties do not improve together.

## 2.4 Diferentiable Simulation and Equilibrium Models

Diferentiable simulators allow gradients to propagate through physical dynamics, enabling optimization of controls, parameters, and designs. Frameworks such as DifTaichi and Jaxley diferentiate through simulation programs or biophysical neuron models [8, 5]. These approaches diferentiate through the time-stepping computation itself, either by unrolling, which stores intermediate states, or by a continuous adjoint, which integrates backward across the horizon. In both cases the gradient computation inherits the sequential structure of the forward solve.

Deep Equilibrium Models (DEQs) define network outputs as fixed points of a learned transformation and use the implicit function theorem for memory-eficient diferentiation [1, 3]. TWT borrows the fixed-point view from equilibrium models, but applies it to physical simulation: the fixed point is not a hidden representation of a network layer but a set of subsystem trajectories that mutually satisfy coupling constraints, and—unlike a DEQ—it can be checked against an external ground truth.

## 3 Method

## 3.1 Problem Formulation

Consider a coupled system with K subsystems. The state of subsystem i evolves according to

$$
\begin{array} { r } { \dot { \mathbf { x } } _ { i } = f _ { i } ( \mathbf { x } _ { i } , \mathbf { u } _ { i } ( t ) , \boldsymbol { \theta } _ { i } ) , \qquad \mathbf { x } _ { i } ( 0 ) = \mathbf { x } _ { i , 0 } , } \end{array}\tag{1}
$$

where $\theta _ { i }$ denotes subsystem parameters and ${ \bf { u } } _ { i } ( t )$ is a driving input generated by neighboring subsystem states:

$$
{ \bf u } _ { i } ( t ) = g _ { i } \left( \{ { \bf x } _ { j } ( t ) \} _ { j \in \mathcal { N } ( i ) } \right) .\tag{2}
$$

The coupling structure is represented by a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where nodes are subsystems and edges indicate dependencies.

If the complete driving input ${ \bf { u } } _ { i } ( t )$ were known for each subsystem, each subsystem could be solved independently. The dificulty is circular: the driving inputs depend on the unknown trajectories.

## 3.2 Module-Level Trajectory Surrogates

For each module type, we train a surrogate $S _ { \phi }$ that approximates the isolated subsystem solution operator:

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { i } = S _ { \phi } ( \theta _ { i } , \mathbf { u } _ { i } , \mathbf { x } _ { i , 0 } ) , } \end{array}\tag{3}
$$

where $\mathbf { u } _ { i }$ is a discretized input trajectory and $\hat { \mathbf { x } } _ { i }$ the predicted output trajectory. The surrogate produces the entire trajectory in one forward evaluation, so time still appears in the representation but inference does not proceed by repeatedly applying a local update rule.

Trajectory representation. For the oscillator system we represent trajectories by their first N Chebyshev coeficients rather than by raw samples [20]. The transform is a fixed linear map in both directions, so it is diferentiable and commutes with any linear coupling: ${ \mathcal { C } } ( W \mathbf { X } ) = W { \mathcal { C } } ( \mathbf { X } )$ and coupling can be applied directly in coeficient space. On a 30-unit window, 256 coeficients reconstruct van der Pol trajectories to below 0.5% across the full stifness range studied, against 1500 raw samples. For the neuron system we use raw samples. We emphasize that the representation is an implementation choice orthogonal to the formulation; the two systems use diferent ones and the method is unchanged.

Training data. Data is generated by simulating isolated modules with known driving inputs using a conventional numerical solver. To expose the surrogate to inputs similar to those encountered during coupled simulation, we use a rolling-bufer strategy: early samples are generated from simple driving inputs and stored in a bufer, and later samples draw their inputs from weighted combinations of previously generated output trajectories.

This strategy requires care in two respects that we found to matter more than any architectural choice. First, the bufer can degenerate. Trajectories carrying no signal (a neuron that never fires) produce weaker composed drives, which produce further silent trajectories, and the bufer collapses. We discard degenerate samples before they enter the bufer and verify the written dataset. Second, and more consequentially, the bufer only covers the input distribution it happens to generate. Twice during this work an experiment probed a parameter range the generator had never sampled, and in both cases the resulting error was misattributed to the method before being traced to coverage. We return to this in Section 6.

The surrogate is a residual MLP with hidden dimension 768 and six residual blocks (7.5M parameters for the oscillator, 8.6M for the neuron). Input trajectories are concatenated with initial conditions and subsystem parameters and passed through the network, which regresses the full output trajectory. We train with plain MSE; Section 4.3 reports a negative result on Jacobian regularization.

## 3.3 Self-Consistency Solve

Given trained surrogates, the coupled solution is the fixed point

$$
{ \bf X } = F ( { \bf X } ) , \qquad F ( { \bf X } ) _ { i } = S _ { \phi } \bigl ( \theta _ { i } , g _ { i } \bigl ( \{ { \bf X } _ { j } \} _ { j \in N ( i ) } \bigr ) , { \bf x } _ { i , 0 } \bigr ) .\tag{4}
$$

At a zero-residual fixed point, every trajectory equals the one predicted by its module surrogate under the inputs induced by its neighbors.

Solvers and the contraction condition. The natural iteration is Picard, $\mathbf { X } \gets F ( \mathbf { X } )$ , which is what classical waveform relaxation performs. It converges only while

$$
\begin{array} { r } { \rho ( J _ { F } ) < 1 , \qquad J _ { F } = \frac { \partial S _ { \phi } } { \partial { \bf u } } \cdot W , } \end{array}\tag{5}
$$

where W is the linear coupling operator; for the all-to-all difusive coupling used below, $\| W \| = k _ { c }$ exactly. This condition is a property of the iteration, not of the formulation. We therefore also solve Eq. 4 by Newton’s method applied to $G ( \mathbf { X } ) = \mathbf { X } - F ( \mathbf { X } )$ , with the linear system handled by GMRES [19]—a Jacobian-free Newton–Krylov method [9]. The Jacobian is never formed: each GMRES iteration requires only $( I - J _ { F } ) { \bf v }$ , obtained by one forward-mode diferentiation of F. Newton converges wherever $I - J _ { F }$ is nonsingular, with no contraction requirement.

Implicit diferentiation. At the converged fixed point $\mathbf { X } ^ { * }$ , gradients of an objective J with respect to system parameters θ follow from the implicit function theorem [1, 3]:

$$
\left( I - J _ { F } ^ { \top } \right) \mathbf { v } = \nabla \mathbf { x } * \mathcal { I } , \qquad \frac { d \mathcal { I } } { d \theta } = \mathbf { v } ^ { \top } \frac { \partial F } { \partial \theta } .\tag{6}
$$

This linear system carries no dependence on how the forward solve was performed and no time recursion. Expanding $( I - J _ { F } ^ { \top } ) ^ { - 1 }$ as a Neumann series—the standard DEQ backward—reintroduces the condition $\rho ( J _ { F } ) < 1$ , and therefore fails exactly where Newton was introduced to succeed. We solve Eq. 6 by GMRES instead, which does not.

## 3.4 Exact-Surrogate Interpretation

Under an idealized exact-surrogate assumption, a zero-residual fixed point of $\operatorname { E q . }$ 4 coincides with the unique solution of the original coupled initial value problem, provided the underlying ODE satisfies standard existence and uniqueness conditions. At zero residual every subsystem trajectory equals the solution produced by its governing dynamics under the inputs induced by neighboring trajectories, so the assembled trajectories satisfy the coupled system; by uniqueness they are the physical solution.

For learned surrogates this becomes approximate, and the gap is measurable. We therefore report the self-consistency residual and the trajectory error separately throughout, and additionally report the error obtained by running the same fixed point with an exact integrator as the module operator. That third quantity is the floor attributable to the formulation and its discretization; the distance between it and the surrogate result is what learning costs.

The decomposition earned its keep during development. In an early version of this work the module simulator applied the drive at the end of each step while the coupled reference applied it at the start. Both are valid O(∆t) discretizations, but they define diferent fixed points, and the discrepancy, which grows from 0.8% to 10.4% with coupling strength, was indistinguishable from surrogate error until the exact-operator check isolated it. We report both conventions in Section 4.1.

## 4 Experiments

We study two systems. Coupled van der Pol oscillators, $\ddot { x } _ { i } = \mu ( 1 - x _ { i } ^ { 2 } ) \dot { x } _ { i } - k _ { i } x _ { i } + F _ { i } ( t )$ , are nonlinear with a stable limit cycle and, at large $\mu ,$ relaxation oscillations; a single surrogate spans $\mu \in [ 0 . 5 , 5 ]$ from quasi-harmonic to stif. Hodgkin–Huxley neurons [7] coupled through excitatory chemical synapses provide a second system with two properties the oscillators lack: the coupling is nonlinear, since the synaptic current depends on both pre- and postsynaptic state, and the surrogate is a compact model—voltage and gating variables are used to generate data but stay latent, and only the coupling-relevant variable s(t) is predicted.

Surrogate accuracy on held-out data is 3.05% median relative error for the oscillator (5.24% mean, 16.9% at the 95th percentile) and 2.61% median for the neuron (3.35% mean, 5.91% at the 95th percentile).

## 4.1 Validating the Fixed Point

Before introducing a surrogate we verify that the fixed point is the right object, by solving Eq. 4 with an exact RK4 module operator and comparing against a conventional coupled integration of the same system.

The formulation is exact, and the growth of Picard iteration counts with $k _ { c }$ is the contraction condition of Eq. 5 becoming binding.

This baseline also quantifies why the learned operator matters. Solving the same fixed point with Newton rather than Picard, using the exact integrator, takes 1548 s at $k _ { c } = 2$ against Picard’s 13.7 s, because every GMRES iteration is a full numerical integration of every subsystem. With the surrogate, the same Jacobian-vector product is one forward-mode pass.

## 4.2 Coupled Trajectories

We solve networks of N = 20 oscillators with all-to-all difusive coupling and detuned natural frequencies, sweeping coupling strength $k _ { c }$ and stifness $\mu ,$ with five random systems per cell. The

Table 1: Classical waveform relaxation with an exact module operator, $N = 2 0$ oscillators, $\mu = 1$ The fixed point reproduces the coupled reference exactly. The last column shows the same solve under a module discretization that difers from the reference by one step in where the coupling force is sampled: both are valid schemes, but they define diferent fixed points, and the discrepancy grows with coupling.
<table><tr><td> $k _ { c }$ </td><td>Feedforward err.</td><td>WR err.</td><td>WR err. (mismatched)</td><td>Picard iters.</td><td>Residual</td></tr><tr><td>0.25</td><td>95.6%</td><td>0.0%</td><td>0.8%</td><td>9</td><td> $1 . 1 \times 1 0 ^ { \cdot }$  -12</td></tr><tr><td>0.50</td><td>147.8%</td><td>0.0%</td><td>1.4%</td><td>13</td><td> $1 . 3 \times 1 0 ^ { - }$  -12</td></tr><tr><td>1.00</td><td>137.9%</td><td>0.0%</td><td>2.7%</td><td>21</td><td> $3 . 1 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>1.50</td><td>146.5%</td><td>0.0%</td><td>4.0%</td><td>30</td><td> $2 . 0 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td>2.00</td><td>140.6%</td><td>0.0%</td><td>10.4%</td><td>40</td><td> $3 . 6 \times 1 0 ^ { - 1 2 }$ </td></tr></table>

feedforward baseline is the same surrogate evaluated with the coupling disabled—the uncoupled view of the same network.

Table 2: Relative trajectory error against RK4 ground truth, $N = 2 0$ , five seeds per cell. Feedforward error is 97–153% throughout and is omitted per-cell. Parenthesized entries report seeds converged out of five where not all converged.
<table><tr><td rowspan="2">μ</td><td colspan="4">Coupling strength  $k _ { c }$ </td></tr><tr><td>0.25</td><td>0.50</td><td>1.00 1.50</td><td>2.00</td></tr><tr><td>0.5</td><td>1.7%</td><td>2.4%</td><td>5.8% 18.2%</td><td>34.8% (4/5)</td></tr><tr><td>1.0</td><td>1.0%</td><td>1.5% 4.0%</td><td>13.8%</td><td>31.9% (3/5)</td></tr><tr><td>2.0</td><td>1.1%</td><td>1.5%</td><td>3.2% 8.3%</td><td>31.2%</td></tr><tr><td>3.0</td><td>1.8%</td><td>2.3%</td><td>4.1% 8.1%</td><td>16.8%</td></tr><tr><td>5.0</td><td>3.8%</td><td>4.7%</td><td>7.0% 10.1%</td><td>15.0%</td></tr><tr><td>Newton iters.</td><td>4</td><td>4-5</td><td>6-8 8-10</td><td>10-57</td></tr></table>

For $k _ { c } \leq 1 . 5$ every configuration converges on all five seeds, with residuals below $1 0 ^ { - 1 1 }$ and errors between 1.0% and 18.2%, against a feedforward baseline that is wrong by more than 100% everywhere. The solve takes 4–10 Newton iterations on trajectories spanning 1500 timesteps.

## 4.3 Contraction Predicts Convergence

The pattern in the last column of Table 2 is not accidental. We estimate $\rho ( \partial S _ { \phi } / \partial \mathbf { u } )$ by power iteration on the learned operator, using only perturbation probes—no coupled system is involved. Combined with $\| W \| = k _ { c } ,$ , Eq. 5 predicts a convergence boundary $k _ { c } ^ { * } = 1 / \rho$ . Figure 1 overlays the predicted boundary on the observed sweep.

The ordering is exact: iteration count, convergence rate, and trajectory error all follow $\rho$ across the stifness range. The same measurement applied to the true operator gives $\rho \approx 0 . 5 4$ at $\mu = 1$ , so the learned operator is more expansive than the dynamics it approximates, and the coupled solve reaches a lower coupling strength than the physics would allow.

Accuracy and composability, in other words, are diferent properties, and they do not move together. The surrogate is least accurate in isolation at $\mu = 5$ (3.4% against 0.8% at $\mu = 1 )$ , yet the coupled solve is best there, because contraction dominates once coupling is strong. Across three separately trained surrogates we observe the same tension: better fit accompanied by larger $\rho .$

![](images/aba6b18cd291c21f643d92dfa009ec5f2ef48285eae5a5f1acde7785ac5e5ab2.jpg)  
Figure 1: The spectral radius of the learned operator, measured by power iteration on the isolated operator, predicts where the coupled solve converges. Cells: relative trajectory error against RK4 ground truth $( N = 2 0$ , five seeds per cell; parentheses give seeds converged where not all five did). Line: the predicted boundary $k _ { c } ^ { * } = 1 / \rho .$ . No coupled solve is involved in the prediction.

Table 3: The spectral radius of the learned operator, measured independently of any coupled solve, against the observed behavior at $k _ { c } = 2$
<table><tr><td> $\mu$ </td><td> $\rho$  (measured)</td><td>Predicted  $k _ { c } ^ { * }$ </td><td>Converged at  $k _ { c } = 2$ </td><td>Iters.</td></tr><tr><td>0.5</td><td>0.87</td><td>1.15</td><td> $4 / 5$ </td><td>57</td></tr><tr><td>1.0</td><td>0.64</td><td>1.56</td><td> $3 / 5$ </td><td>45</td></tr><tr><td>2.0</td><td>0.57</td><td>1.75</td><td> $5 / 5$ </td><td>11</td></tr><tr><td>3.0</td><td>0.54</td><td>1.85</td><td> $5 / 5$ </td><td>10</td></tr><tr><td>5.0</td><td>0.52</td><td>1.92</td><td> $5 / 5$ </td><td>10</td></tr></table>

The obvious remedy does not work naively, either. We trained a surrogate with an added penalty on the operator gain, in the spirit of Jacobian regularization for equilibrium models [2], estimated by random directional probes. The penalty reduced the probed quantity by half and made the coupled solve worse. The reason is that a random probe in n dimensions places only $1 / n$ of its energy along any one direction, so the estimate is the Frobenius norm rather than the spectral radius; constraining it crushed all directions while leaving the one that governs convergence slightly larger. We report this as a negative result: the correct quantity requires power iteration, not random probing.

## 4.4 Implicit Diferentiation

We diferentiate a scalar objective with respect to the coupling weights of a ring of oscillators and compare the implicit gradient of Eq. 6 against unrolled backpropagation and against central finite diferences.

Where the comparison is meaningful the gradients agree to five digits. Where it is not—past the contraction boundary—unrolled backpropagation diverges by a factor of 83 and the Neumann adjoint returns non-finite values, while the implicit gradient remains correct to 0.04% against finite diferences. This is the regime the Newton solver was introduced to reach, and the implicit gradient is the only one that survives it.

To confirm the gradients do useful work in an optimization, we recover the coupling weights of a six-oscillator ring from observed trajectories generated by RK4—the true physics, not the surrogate’s own output—by minimizing trajectory mismatch through the fixed point. Over ten random systems, parameter error falls from 83.2% to a median of 1.2% (worst case 2.1%) and output error from 105% to 1.3%. The implicit and unrolled gradients give identical results at this coupling strength; the implicit one uses memory independent of solver depth.

Table 4: Gradient agreement and memory. Unrolled memory grows with depth K; the implicit gradient does not. At coupling 0.5 the forward Newton solve converges in 14 iterations, but $\rho > 1$ and only the implicit gradient remains valid.
<table><tr><td></td><td>Coupling 0.1,  $N = 6$ </td><td>Coupling 0.1, N = 50</td><td>Coupling 0.5, N = 6</td></tr><tr><td>Cosine sim. vs. unrolled</td><td>1.0000</td><td>1.0000</td><td>-0.13 (unrolled diverges)</td></tr><tr><td>Rel. err. vs. unrolled</td><td> $1 . 4 \times 1 0 ^ { - 5 }$ </td><td> $7 . 9 \times 1 0 ^ { - 5 }$ </td><td>83.2 (unrolled diverges)</td></tr><tr><td>Rel. err. vs. finite diff.</td><td>0.06%</td><td></td><td>0.04%</td></tr><tr><td>Neumann adjoint</td><td>converges</td><td>converges</td><td>diverges (inf)</td></tr><tr><td>Unrolled memory,  $K { = } 2  1 6 0$ </td><td>102 → 146 MB</td><td> $1 0 8  {  } 4 7 2  { \mathrm { M B } }$ </td><td></td></tr><tr><td>Implicit memory</td><td>112MB</td><td>169 MB</td><td>112MB</td></tr></table>

## 4.5 Hodgkin–Huxley Neurons

An external neuron receives a transient current pulse and drives an internal neuron carrying a self-recurrent synapse, so the feedback loop closes at the internal neuron’s own output. We sweep the recurrent weight and measure on the post-pulse window, after the external drive is gone and the recurrent term is all that remains.

Table 5: Post-pulse relative error and mean activity. WR is the same fixed point with an exact module operator, and is the floor attributable to the formulation; the gap between SC and WR is surrogate error. Feedforward activity is independent of $w _ { \mathrm { r e c } }$ by construction.
<table><tr><td rowspan="2"> $w _ { \mathrm { r e c } }$ </td><td colspan="3">Error</td><td colspan="3">Mean activity</td><td rowspan="2">Iters.</td></tr><tr><td>FF</td><td>SC</td><td>WR</td><td>GT</td><td>FF</td><td>SC</td></tr><tr><td>0.5</td><td>5.4%</td><td>2.2%</td><td>0.9%</td><td>0.0527</td><td>0.0551</td><td>0.0527</td><td>3</td></tr><tr><td>1.0</td><td>8.2%</td><td>5.8%</td><td>1.0%</td><td>0.0508</td><td>0.0551</td><td>0.0502</td><td>3</td></tr><tr><td>2.0</td><td>17.9%</td><td>9.8%</td><td>1.5%</td><td>0.0474</td><td>0.0551</td><td>0.0472</td><td>4</td></tr><tr><td>3.0</td><td>35.4%</td><td>18.5%</td><td>1.7%</td><td>0.0436</td><td>0.0551</td><td>0.0439</td><td>4</td></tr><tr><td>4.0</td><td>56.3%</td><td>29.0%</td><td>1.9%</td><td>0.0397</td><td>0.0551</td><td>0.0404</td><td>9</td></tr><tr><td>5.0</td><td>54.3%</td><td>56.2%</td><td>1.6%</td><td>0.0405</td><td>0.0551</td><td>0.0413</td><td>(no conv.)</td></tr></table>

The activity columns carry the clearer statement. Feedforward predicts that recurrence has no efect at all: its activity is pinned at 0.0551 regardless of $w _ { \mathrm { r e c } } .$ . The true activity falls monotonically from 0.0527 to 0.0397 as recurrence strengthens—a consequence of the synaptic reversal potential, which makes strong input depolarizing at rest but shunting during a spike—and the fixed point tracks that dependence where the uncoupled view, by construction, cannot.

The error columns are more equivocal, and we report them plainly. Self-consistency roughly halves feedforward error, but the exact-operator floor sits near 1.9%, so most of the remaining error is the surrogate rather than the formulation. At $w _ { \mathrm { r e c } } = 5$ the solve does not converge for this surrogate. The neuron surrogate is comparably accurate to the oscillator one in isolation, so the gap reflects amplification through a fixed point whose conditioning is worse, consistent with Section 4.3.

## 5 Discussion

The experiments support the premise: coupled simulation can be reformulated as trajectory-leve self-consistency among learned module operators, and the reformulation changes the structure of both the forward computation and its derivative. The validation in Section 4.1 establishes that the fixed point is the correct object independently of any learning, and the exact-operator floor reported throughout separates what the formulation achieves from what the surrogate costs.

Two findings we did not anticipate seem worth emphasizing.

The first is that the classical relaxation iteration is the wrong solver for learned operators, and that this is not a detail. Picard is bound by $\rho ( \partial S _ { \phi } / \partial \mathbf { u } ) \left\| W \right\| < 1$ , and a learned operator is typically more expansive than the dynamics it approximates, so the bound binds earlier than the physics requires. Newton removes the condition entirely, and it is afordable only because the operator is learned: with an exact integrator in the same role it is roughly 100× slower than Picard. The equilibrium view and the learned operator are not independent choices; each is what makes the other practical.

The second is that operator accuracy and operator composability are distinct properties that can be measured separately and that do not improve together. A surrogate can fit better in isolation and compose worse, and we observe this across three independently trained models and across stifness within a single model. Standard surrogate evaluation reports only the first property. For any method that composes learned operators, the second appears to be equally predictive of end-to-end behavior, and—via power iteration on the Jacobian—it can be measured before any coupled solve is attempted.

The results also clarify what the method does not solve. TWT does not remove the need to represent trajectories, nor does it eliminate approximation error. The self-consistency residual measures agreement under the learned surrogate, not satisfaction of the original diferential equation, so a residual of $1 0 ^ { - 1 2 }$ can coexist with substantial physical error—as it does at $k _ { c } = 2$ . Reporting both, together with the exact-operator floor, is necessary for the numbers to be interpretable.

## 6 Limitations and Future Work

The present results are an early demonstration rather than a mature alternative to classical solvers. The systems studied here are small: 20 coupled oscillators and neuron pairs. Nothing in the formulation is specific to that scale, but we have not demonstrated it beyond it, and the structural claims about sequential depth would be considerably more convincing at $1 0 ^ { 3 }$ modules. Surrogate error of 1–4% is likewise far from solver-grade, and fixed-point conditioning amplifies it: a converged fixed point is only as meaningful as the operators from which it is built. The exact-operator floors reported here (0.0% for oscillators, 1.9% for neurons) bound what better surrogates could achieve within the same formulation.

For the same reason we report sequential depth—4–10 iterations against 1500 timesteps—and not wall-clock speedup. Neither our implementation nor the reference integrator is optimized, so the comparison would not be meaningful. The structural claim is that iteration count is not tied to trajectory length and that iterations are parallel across modules; the practical claim requires an optimized implementation we have not built. For reference, all reported experiments were trained and run in approximately one day on a single rented NVIDIA RTX 5090 (32 GB VRAM) instance with a 16-core AMD Ryzen 7 7800X3D host, roughly 24 GPU-hours in total; preliminary experimentation over the project’s duration used additional heterogeneous rented GPUs.

The dominant practical failure mode was training-distribution coverage. Twice, an experiment exercised a parameter range the data generator had never sampled (once in drive magnitude, once in recurrent weight), and in both cases the error was initially misattributed to the method. Composing learned operators makes this worse than in standard surrogate use, because the fixed-point iteration itself determines the inputs the operator sees, and those inputs are not known in advance. Generating training data that covers the distribution induced by coupling, rather than a distribution chosen beforehand, seems to us a central unsolved problem for this class of method.

Finally, the experiments cover ODE systems with linear or synaptic coupling; PDEs, stif multiscale systems, chaotic dynamics, discontinuities, and event-driven systems require separate investigation. Windowing the time axis, the classical remedy for slow waveform relaxation convergence, would also reduce ρ directly and is a natural next step we did not pursue. Beyond that, the most promising directions we see are learned preconditioners for the fixed-point solve, compressed or latent trajectory representations for long horizons, and training objectives that control the spectra radius directly rather than through the proxy we found insuficient.

## 7 Conclusion

Time Without Timesteps reframes coupled dynamical simulation as a fixed-point problem over trajectories, following classical waveform relaxation but replacing the inner numerical integration with learned trajectory operators. The reformulation changes the sequential structure of the forward solve, converging in 4–10 iterations on trajectories spanning 1500 timesteps, and it changes the structure of the gradient, replacing a backward march through the integrator with a linear system that can be solved at memory independent of solver depth. We show that the convergence boundary of the coupled solve is predicted in advance by a single measured property of the learned operator, that this property is distinct from isolated accuracy and does not improve alongside it, and that past the boundary the implicit gradient is the only one that remains valid. The result is not a universal alternative to numerical integration, but a computational paradigm for diferentiable, modular simulation whose failure modes we can now measure rather than merely observe.

## References

[1] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In Advances in Neural Information Processing Systems, volume 32, 2019.

[2] Shaojie Bai, Vladlen Koltun, and J. Zico Kolter. Stabilizing equilibrium models by Jacobian regularization. In International Conference on Machine Learning, 2021.

[3] Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe Llinares-L´opez, Fabian Pedregosa, and Jean-Philippe Vert. Eficient and modular implicit diferentiation. In Advances in Neural Information Processing Systems, volume 35, 2022.

[4] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary diferential equations. In Advances in Neural Information Processing Systems, volume 31, 2018.

[5] Michael Deistler, Kyra L. Kadhim, Matthijs Pals, Jonas Beck, Ziwei Huang, Manuel Gloeckler, Janne K. Lappalainen, Cornelius Schr¨oder, Philipp Berens, Pedro J. Gon¸calves, and Jakob H. Macke. Jaxley: Diferentiable simulation enables large-scale training of detailed biophysical models of neural dynamics. Nature Methods, 22(12):2649–2657, 2025.

[6] Martin J. Gander. 50 years of time parallel time integration. In Multiple Shooting and Time Domain Decomposition Methods, pages 69–113. Springer, 2015.

[7] Alan L. Hodgkin and Andrew F. Huxley. A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of Physiology, 117(4):500–544, 1952.

[8] Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, and Fr´edo Durand. DifTaichi: Diferentiable programming for physical simulation. In International Conference on Learning Representations, 2020.

[9] Dana A. Knoll and David E. Keyes. Jacobian-free Newton–Krylov methods: A survey of approaches and applications. Journal of Computational Physics, 193(2):357–397, 2004.

[10] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24(89):1–97, 2023.

[11] Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, and Michael W. Mahoney. Characterizing possible failure modes in physics-informed neural networks. In Advances in Neural Information Processing Systems, volume 34, 2021.

[12] Ekachai Lelarasmee, Albert E. Ruehli, and Alberto L. Sangiovanni-Vincentelli. The waveform relaxation method for time-domain analysis of large scale integrated circuits. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 1(3):131–145, 1982.

[13] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations. In International Conference on Learning Representations, 2021.

[14] Jacques-Louis Lions, Yvon Maday, and Gabriel Turinici. R´esolution d’EDP par un sch´ema en temps “parar´eel”. Comptes Rendus de l’Acad´emie des Sciences – Series I – Mathematics, 332(7):661–668, 2001.

[15] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, 2021.

[16] Jerrold E. Marsden and Matthew West. Discrete mechanics and variational integrators. Acta Numerica, 10:357–514, 2001.

[17] Ulla Miekkala and Olavi Nevanlinna. Convergence of dynamic iteration methods for initial value problems. SIAM Journal on Scientific and Statistical Computing, 8(4):459–482, 1987.

[18] Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019.

[19] Youcef Saad and Martin H. Schultz. GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. SIAM Journal on Scientific and Statistical Computing, 7(3):856–869, 1986.

[20] Lloyd N. Trefethen. Approximation Theory and Approximation Practice. SIAM, 2013.

[21] Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449:110768, 2022.