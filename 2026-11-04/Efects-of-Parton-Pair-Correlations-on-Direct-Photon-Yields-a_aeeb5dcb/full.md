# Efects of Parton Pair Correlations on Direct-Photon Yields and Elliptic Flow in Heavy-Ion Collisions

Xingjian Lu<sup>1,</sup> <sup>2,</sup> <sup>∗</sup> and Shuzhe Shi<sup>2,</sup> <sup>†</sup>

<sup>1</sup>School of Physics and Electronic Information, Yantai University, Yantai 264005, China <sup>2</sup>Department of Physics, Tsinghua University, Beijing 100084, China

Direct photons escape heavy-ion collisions with little rescattering, carrying information from the full emission history. Yet describing the measured direct-photon yield and the sizable elliptic flow v together remains a long-standing challenge. Here we explore how correlations between emitting partons may afect both observables. The local $2  2$ photon rate depends on the joint distribution of incoming partons, so a connected correlation between them can change photon emission without changing the one-particle distribution of medium particles. Once folded over the evolving medium in Au+Au and Pb+Pb collisions, the correlation raises the photon yield toward the fitted PHENIX and ALICE data and increases $v _ { 2 }$ mainly at intermediate $p _ { T }$ . The fitted temperature dependence accounts for most of the net flow increase in this region, suggesting that the correlation changes the relative contributions between hotter and cooler regions.

## I. INTRODUCTION

Photons are produced at every stage of a heavy-ion collision and then leave the medium without further interaction [1–6]. Their spectrum therefore carries an integrated record of the temperature and of the collective flow of the matter that emitted them. Two observables carry this record. The azimuthally averaged yield measures how much radiation the evolving medium produces, and the elliptic flow coeficient $v _ { 2 }$ measures how strongly this radiation follows the anisotropic expansion. See [7–9] for recent reviews and [10–33] for related studies. The PHENIX and STAR collaborations have measured direct photons in Au+Au collisions at $\sqrt { s _ { N N } } = 2 0 0$ GeV [11, 34–37] performed in the Relativistic Heavy-Ion Collider (RHIC), and the ALICE collaboration has measured them in $\sqrt { s _ { N N } } = 2 . 7 6$ TeV Pb+Pb collisions [14, 38] at Large Hadron Collider (LHC). In both systems the measured photon $v _ { 2 }$ is sizable; at RHIC it is of the same order as the hadron $v _ { 2 }$ [11], although a large part of the thermal radiation is expected from early, hot, and still weakly anisotropic stages. Describing a large yield and a large $v _ { 2 }$ with one model is the long-standing direct-photon problem [20, 25].

The central dificulty is to describe the direct-photon yield and $v _ { 2 }$ with the same calculation. The thermal photon signal mixes emission from a hot, weakly anisotropic early medium with emission from a cooler later medium in which stronger collective flow has developed. Increasing the early contribution can raise the yield but dilute v , while giving more weight to late emission can raise v but also change the yield and its momentum dependence. This interplay makes it dificult to describe both observables with one set of photon sources. A recent multistage calculation that combines prompt, pre-equilibrium, and hydrodynamic thermal photons [39] illustrates the dificulty in the data considered here. In the low-p<sub>T</sub> region, the calculation comes closest to the STAR yield but undershoots the PHENIX and ALICE yields. Its $v _ { 2 } ,$ , however, lies below the measured central values in nearly all fitted bins. Therefore, a viable mechanism must improve $v _ { 2 }$ and reduce the yield deficit in the bins where the baseline is insuficient, while preserving the agreement in the bins where the yield is already well described.

Most photon-rate calculations assume that the collid ing partons are uncorrelated, so their two-particle distribution is a product of one-particle distributions [25, 40]. In a dense, strongly interacting medium, this assump tion may fail [41]. In the 2 → 2 annihilation and Comp ton processes, a connected two-particle correlation enters the photon emission rate linearly and can produce an oscillatory correction to the local photon spectrum [42]. This correction can enhance emission at some photon momenta and suppress it at others. When folded over a flowing medium, this momentum-dependent correction can afect the yield and $v _ { 2 }$ diferently. In Ref. [42], we es tablished this local spectral efect but did not fold it over a realistic space-time evolution. It therefore left open whether the efect can describe measured yields and v<sub>2</sub>, and which parts of the correlation the data constrain.

Here we ask whether a local pair-correlation response can describe the direct-photon yield and $v _ { 2 }$ in Au+Au and Pb+Pb collisions when the other photon sources and hydrodynamic inputs are held fixed. We study the 0 − 20% and 20 − 40% centrality classes in both systems. We expand the correlation in a truncated momentumspace basis with two spatial features, giving 146 candidate coeficients, of which four are fitted, and reconstruct the coeficients separately for each class. We assess which combinations of coeficients the data constrain. We also test whether the total local photon emission rate is nonnegative in all cells above $T _ { \mathrm { s w } }$ at sampled photon energies, whether the total yield is positive in every experimental bin, whether the two-particle distribution is nonnegative in a weighted domain, and how these conditions afect the fit.

The result of the paper is organized as follows. Section II presents the model and method, Sec. III the results, and Sec. IV the conclusions.

## II. MODEL AND METHOD

## A. Local photon production rate

For completeness, we briefly review the correlation induced photon production calculation developed in Ref. [42]. For a tree-level process $a ( p _ { a } ) + b ( p _ { b } ) \ $ $c ( p _ { c } ) + \gamma ( k )$ in a medium cell at space-time point $X$ the invariant photon emission rate is

$$
\begin{array} { r l } { E _ { k } \frac { \mathrm { d } R } { \mathrm { d } ^ { 3 } k } ( X ) = \displaystyle \sum _ { a b  c \gamma } \frac { 1 } { 2 ( 2 \pi ) ^ { 3 } } \int \mathrm { d } \Pi _ { a } \mathrm { d } \Pi _ { b } \mathrm { d } \Pi _ { c } } & { } \\ { \times ( 2 \pi ) ^ { 4 } \delta ^ { ( 4 ) } ( p _ { a } + p _ { b } - p _ { c } - k ) } & { } \\ { \times \overline { { | \mathcal { M } | ^ { 2 } } } _ { a b  c \gamma } S _ { a b c } ( X ; p _ { a } , p _ { b } , p _ { c } ) , } \end{array}\tag{1}
$$

where $\mathrm { d } \Pi _ { i } ~ = ~ \mathrm { d } ^ { 3 } p _ { i } / [ ( 2 \pi ) ^ { 3 } 2 E _ { i } ]$ and $\overline { { | \mathcal { M } | ^ { 2 } } }$ contains the squared matrix element with the sums and averages over internal quantum numbers. For massless u, d, and s quarks the channel sum contains $q \bar { q }  g \gamma$ and $q ( \bar { q } ) g $ $q ( \bar { q } ) \gamma$ . The medium enters only through the statistical factor $S _ { a b c }$

To separate genuine correlations from single-particle efects, we write the two-particle distribution of species i and $j$ as

$$
{ \mathcal { F } } _ { i j } ( X ; p _ { i } , p _ { j } ) = f _ { i } ( X ; p _ { i } ) f _ { j } ( X ; p _ { j } ) + g _ { i j } ( X ; p _ { i } , p _ { j } ) ,\tag{2}
$$

and define the connected part $g _ { i j }$ by

$$
\begin{array} { c } { { \displaystyle { \int \mathrm { d } ^ { 3 } p _ { i } g _ { i j } = 0 , \qquad \int \mathrm { d } ^ { 3 } p _ { j } g _ { i j } = 0 , } } } \\ { { g _ { i j } ( X ; p _ { i } , p _ { j } ) = g _ { j i } ( X ; p _ { j } , p _ { i } ) . } } \end{array}\tag{3}
$$

The two zero-marginal conditions ensure that $g _ { i j }$ does not change the one-particle distributions. Without genuine three-particle correlations,

$$
\begin{array} { r } { S _ { a b c } = f _ { a } f _ { b } ( 1 + \sigma _ { c } f _ { c } ) + \big [ g _ { a b } ( 1 + \sigma _ { c } f _ { c } ) + \sigma _ { c } f _ { b } g _ { a c } + \sigma _ { c } f _ { a } g _ { b c } \big ] , } \\ { ( 4 ) } \end{array}
$$

with $\sigma _ { c } ~ = ~ + 1 ~ \left( - 1 \right)$ for a bosonic (fermionic) finalstate parton. We use the classical-statistics, leadinglogarithmic limit of Ref. [42], in which $S _ { a b c }  f _ { a } f _ { b } + g _ { a b }$ The correlation correction to Eq. (1) is then linear in $g _ { a b }$ We assume the same momentum-space correlation profile for all parton species.

In the local rest frame of a rotationally invariant cell, $g _ { a b }$ can depend only on the two energies scaled by the temperature, $\xi _ { i } = \stackrel { \cdot } { p _ { i } } \cdot u ( X ) / T ( X )$ , and on the relative angle, $z _ { a b } = \hat { p } _ { a } \cdot \hat { p } _ { b }$ . Rotational invariance removes any preferred absolute direction, but it leaves correlations in $z _ { a b }$ . We expand

$$
\begin{array} { r l r } {  { g _ { a b } ( X ; p _ { a } , p _ { b } ) } } \\ & { = \sum _ { n , m , \ell } \Psi _ { n m \ell } ( X ) f _ { a } ^ { \mathrm { r e f } } f _ { b } ^ { \mathrm { r e f } } L _ { n } ^ { ( 2 ) } ( \xi _ { a } ) L _ { m } ^ { ( 2 ) } ( \xi _ { b } ) P _ { \ell } ( z _ { a b } ) , \quad } & \end{array}\tag{5}
$$

with the Boltzmann reference distribution $f _ { i } ^ { \mathrm { r e f } } = e ^ { - \xi _ { i } }$ generalized Laguerre polynomials $L _ { n } ^ { ( 2 ) }$ , and Legendre polynomials $P _ { \ell }$

A mode in Eq. (5) is a connected correlation only if it satisfies Eq. (3). Integrating over $p _ { b }$ gives

$$
\begin{array} { r l } & { \quad \displaystyle \int \mathrm { d } ^ { 3 } p _ { b } f _ { b } ^ { \mathrm { r e f } } L _ { m } ^ { ( 2 ) } ( \xi _ { b } ) P _ { \ell } ( z _ { a b } ) } \\ & { = 4 \pi T ^ { 3 } \delta _ { \ell 0 } \int _ { 0 } ^ { \infty } \mathrm { d } \xi \xi ^ { 2 } e ^ { - \xi } L _ { m } ^ { ( 2 ) } ( \xi ) } \\ & { = 8 \pi T ^ { 3 } \delta _ { \ell 0 } \delta _ { m 0 } . } \end{array}\tag{6}
$$

The radial integral equals $2 \delta _ { m 0 }$ by the orthogonality of $L _ { m } ^ { ( 2 ) }$ with the weight $\xi ^ { 2 } e ^ { - \xi }$ . Every product mode with $\ell \geq 1$ therefore satisfies both zero-marginal conditions. For $\ell = 0$ , a single product mode satisfies both conditions when $n , m \geq 1$

The amplitude of each mode may vary across the medium. We write

$$
\Psi _ { n m \ell } ( X ) = \sum _ { A } \psi _ { A , n m \ell } q _ { A } ( X ) ,\tag{7}
$$

where $\psi _ { A , n m \ell }$ are global coeficients, $q _ { A } ( X )$ are fixed features of the hydrodynamic fields, and A labels the fea ture. We use two features: a constant that is independent of the hydrodynamic property, $q _ { \mathrm { C } } = 1$ , and the logarithm of the local temperature which captures the diference in correlation caused by the density related collision rates, $q _ { T } = \ln [ T ( X ) / 0 . 2 \ \mathrm { G e V } ]$ The reference value 0.2 GeV defines the zero of $q _ { T }$ . Its exact value is not very important, because changing it only adds a constant that can be absorbed into the corresponding $q _ { \mathrm { C } }$ coeficient. Both features are rotational scalars with the same definition in every hydrodynamic profile. With the 73 connected modes of the library (Fig. 1) and two features, the candidate set has $7 3 \times 2 = 1 4 6$ coeficients; for each class we fit four of them (Appendix A).

Because the correction is linear in $g _ { a b }$ , each mode produces a local spectral response $\mathcal { R } _ { n m \ell } ( x , T )$ , the change of the invariant rate per unit mode amplitude, where $x \equiv k \cdot u / T$ is the photon energy in the local rest frame in units of T. The correlation contribution to the local rate is

$$
\Delta \Gamma _ { \gamma } ^ { \mathrm { p a i r } } ( X ; k ) = \sum _ { A , n , m , \ell } \psi _ { A , n m \ell } q _ { A } ( X ) \mathcal { R } _ { n m \ell } \bigg ( \frac { k \cdot u } { T } , T \bigg ) .\tag{8}
$$

Instead of an incomplete leading-order rate in the demonstration study [42], we use the phenomenological additive model in this work

$$
\Gamma _ { \gamma } ^ { \mathrm { Q G P } } = \Gamma _ { \gamma } ^ { \mathrm { A M Y } } + \Delta \Gamma _ { \gamma } ^ { \mathrm { p a i r } } ,\tag{9}
$$

where $\Gamma _ { \gamma } ^ { \mathrm { A M Y } }$ is the complete leading-order equilibrium rate of Arnold, Moore, and Yafe, including the collinear bremsstrahlung and inelastic annihilation processes [40]. Both terms use the same flavor content $( u , \ d , \ s )$ , coupling prefactor $\kappa = 1 6 \pi ^ { 2 } \alpha _ { \mathrm { e m } } \alpha _ { s } .$ phase-space normalization, and units.

Our calculation has four steps. First, we define the connected pair correlation by its vanishing one-particle marginals and build a library of its local photon-spectrum responses. Second, we fold these responses over hydrodynamic profiles to obtain two quantities: $Y _ { 0 }$ , the azimuthally integrated photon yield, and $Y _ { 2 } ,$ the secondharmonic moment, with $\ v _ { 2 } \ = \ Y _ { 2 } / Y _ { 0 }$ We add fixed prompt, pre-equilibrium, and thermal photon contributions and compare with the measurements in their published bins. Third, we cast the reconstruction as a physics-constrained regression task. We select four response features from the candidate library and learn their coeficients separately for each collision class using the joint photon-yield and $v _ { 2 }$ data. Finally, we separate the correlation-induced changes in the yield and second harmonic and examine the response rank and local positivity.

Figure 1 shows the response library. Panel (a) shows the sign and shape of 79 stored responses and of the separately stored factorized reference without correlation correction, labeled $F ,$ on a common grid $x \equiv k \cdot u / T \in$ [0.01, 15]; each row is scaled by its own $\operatorname* { m a x } _ { x } \left| R _ { a } \right|$ . Panel (b) shows the stored amplitudes, which span a factor of $\dot { 4 } . \dot { 8 } \times 1 0 ^ { 5 }$ between modes, and the number of sign changes, which ranges from 2 to 8 across the non-reference responses. Every non-reference response changes sign. Each mode therefore gives a signed correction to the emission rate. Of the 79 non-reference modes, 64 have $\ell \geq 1$ and nine have $\ell = 0$ with $n , m \geq 1 ;$ all 73 satisfy the zero-marginal conditions and form the candidate pool of the fit. The remaining six $\ell = 0$ modes do not satisfy these conditions and are excluded from the paircorrelation inversion. Panel (c) shows one representative shape for each $\ell = 0 , \ldots , 4$ , selected among modes with zero marginals. The stored amplitudes in panel (b) are calculated before fitting the mode coeficients and are not physical mode weights. The figure illustrates the signs and shapes of the response modes.

## B. Hydrodynamic profiles and space-time convolution

The local rate is mapped to laboratory observables by a hydrodynamic profile. Each profile cell stores the proper time $\tau ,$ the transverse position (x, y), the temperature $T ,$ , and the transverse flow components $( v _ { x } , v _ { y } )$ Under longitudinal boost invariance the flow velocity is $u ^ { \mu } ~ = ~ \gamma _ { \perp }$ (cosh $\eta _ { s } , v _ { x } , v _ { y }$ , sinh η<sub>s</sub>) with $\gamma _ { \perp } ~ = ~ ( 1 - v _ { x } ^ { 2 } -$ $v _ { y } ^ { 2 } ) ^ { - 1 / 2 }$ , and the four-volume in Milne coordinates is $\mathrm { d } ^ { 4 } X = \tau \mathrm { d } \tau \mathrm { d } x \mathrm { d } y \mathrm { d } \eta _ { s }$ . A photon with transverse momentum $p _ { T }$ , rapidity $y ,$ , and azimuth $\phi _ { p }$ has the local energy

$$
E ^ { * } = k { \cdot } u = \gamma _ { \perp } p _ { T } \big [ \cosh ( y - \eta _ { s } ) - v _ { x } \cos \phi _ { p } - v _ { y } \sin \phi _ { p } \big ] .
$$

The QGP contribution at midrapidity follows from folding the local rate over all cells above the switching tem-

perature,

$$
\frac { \mathrm { d } N _ { \gamma } ^ { \mathrm { Q G P } } } { p _ { T } \mathrm { d } p _ { T } \mathrm { d } y \mathrm { d } \phi _ { p } } = \int _ { T ( X ) > T _ { \mathrm { s w } } } \mathrm { d } ^ { 4 } X \Gamma _ { \gamma } ^ { \mathrm { Q G P } } \big ( E ^ { \ast } = k \cdot u , T ( X ) \big ) ,\tag{11}
$$

with $T _ { \mathrm { s w } } = 1 4 5 ~ \mathrm { M e V } .$ This switching temperature is the same as that in Ref. [39], which provides the baseline contributions from the other stages in our study. The flow term in Eq. (10) is the only source of azimuthal dependence: a locally isotropic emission becomes anisotropic in the laboratory because cells moving along the reaction plane emit harder photons in that direction.

We use the azimuthal moments

$$
Y _ { n } ( p _ { T } ) = \int _ { 0 } ^ { 2 \pi } \mathrm { d } \phi _ { p } \cos [ n ( \phi _ { p } - \Psi _ { n } ) ] \frac { \mathrm { d } N _ { \gamma } } { p _ { T } \mathrm { d } p _ { T } \mathrm { d } y \mathrm { d } \phi _ { p } } ,\tag{12}
$$

for $n = 0 , 2 .$ , averaged over each published $p _ { T }$ bin i. Then $Y _ { 0 , i } / ( 2 \pi )$ is the invariant yield and $v _ { 2 , i } = Y _ { 2 , i } / Y _ { 0 , i }$ . Both $Y _ { 0 }$ and $Y _ { 2 }$ depend afinely on the coeficients of Eq. (8), whereas v is a ratio. This is why the inverse problem below is formulated for the moments, not for $v _ { 2 }$ directly.

The profiles used in this work are VISHydro [43] profile with MCGlb initial condition for the four collision classes, with initial time $\tau _ { 0 } = 0 . 6 \mathrm { f m } / c$ and grid spacings $\Delta \tau = 0 . 1$ fm/c and $\Delta x = \Delta y = 0 . 5$ fm. See Ref. [44] for more details of the medium profile in use. The convolution keeps cells with $T > 1 4 5 { \mathrm { ~ M e V } }$ , weights each cell by $\tau \Delta \tau \Delta x \Delta y$ . These profiles are not identical to the IP-Glasma+KøMPøST+music evolution of Ref. [39], but they are very similar.

## C. Other direct-photon sources

Direct photons are photons not produced by hadron decays. We write their moments as

$$
Y _ { n } ^ { \mathrm { t o t } } = \underbrace { P _ { n } + E _ { n } + H _ { n } } _ { B _ { n } } + C _ { n } , \qquad C _ { n } = \sum _ { \alpha } \psi _ { \alpha } Y _ { n , \alpha } .\tag{13}
$$

Here $P _ { n } , \ E _ { n } ,$ and $H _ { n }$ are the prompt, pre-equilibrium thermal, and hydrodynamic thermal contributions, respectively, taken from the published source files of Ref. [39]. Photons produced during the prompt stage have no elliptic flow, so their $v _ { 2 }$ vanishes, $\mathrm { i . e . , } P _ { 2 } = 0$ . We average these “three” contributions over the experimental bins in the same way as $C _ { n }$ and use their sum $B _ { n }$ as the internal fixed baseline. The correlation increment $C _ { n }$ is obtained by folding the local rate correction of Eq. (8) over the VISHydro profiles. Here ${ \alpha = ( A , n , m , \ell ) }$ labels the fitted coeficients, and $Y _ { n , \alpha }$ is the resulting moment at unit amplitude.

Three features of this construction limit its interpretation. First, $B _ { n }$ and $C _ { n }$ come from similar but not identical hydrodynamic evolutions. Second, the AMY rate of Eq. (9) is not added to the observables. The published thermal term $H _ { n }$ , which carries a slight viscous correction, plays its role, while $\Gamma _ { \gamma } ^ { \mathrm { A M Y } }$ enters only through the positivity condition on the local rate (Sec. III). We use $H _ { n }$ rather than the AMY rate in order to better assess the enhancement relative to the fit results of existing models. Third, the chemical-equilibration suppression studied in Ref. [39] is not applied to $C _ { n }$

![](images/0739d8f67dae89aa662d8a3b2fdc01dcd423c58e3c1ec9c498489d2736037df4.jpg)

(a)  
![](images/f512d40ff6d9cd4f57cd1221dae721442f78c54ad80af235e7afa7582cd34f4e.jpg)

(c)  
![](images/5fd1665769f15e7dfabb5aaddcc9b78502c3729117ab77c599c1304b0064b866.jpg)  
FIG. 1. Local response library for the pair-correlation basis of Eq. (5). (a) Stored responses $R _ { a } ( x ) \equiv y _ { n m \ell } ( x )$ in the order $( n , m , \ell )$ , with $\ell = 0 , \ldots , 4$ inside each $( n , m )$ block, as functions of $x \equiv k \cdot u / T .$ . Each row is divided by its own max $| R _ { a } |$ ; the color shows sign. The strip on the left marks in dark gray the 73 connected modes, the 64 modes with $\ell \geq 1$ and the nine $\ell = 0$ modes with $n , m \geq 1$ , which satisfy Eq. (3) and form the candidate pool of the fit; the six $\ell = 0$ modes with $n m = 0$ have nonzero marginals and are excluded (light gray). The top row is the factorized reference $F ,$ . The dotted line at $x = 0 . 3 1$ is the lower edge of the fitted response range. (b) Logarithm of the stored amplitude $A _ { a } = \operatorname* { m a x } _ { x } | R _ { a } |$ and number of sampled sign changes (threshold $1 0 ^ { - 6 } A _ { a } )$ . Open black, open gray, and filled symbols denote connected modes, excluded modes, and $F ,$ respectively. (c) For each $\ell = 0 , \ldots , 4$ , the connected mode whose normalized shape has the smallest mean squared distance to the other connected modes with the same $\ell ;$ the labels give $( n , m )$

## D. Experimental data products

Table I lists the measurements and their use. In $\mathrm { A u } { + } \mathrm { A u }$ collisions we use the PHENIX conversionmethod $v _ { 2 }$ [35] and the PHENIX direct-photon yield, formed as the published excess spectrum plus the $N _ { \mathrm { c o l l } ^ { - } }$ scaled $p + p$ reference [34]. STAR reports its direct virtual-photon yield at discrete p values but does not provide bin boundaries [36]. We therefore exclude the STAR data from the fit and compare the model with them at the published $p _ { T }$ values. In Pb+Pb collisions we use the ALICE direct-photon $v _ { 2 }$ [14] and yield [38].

TABLE I. Experimental inputs and their use. The centrality columns give the number of fitted bins or points out of all published ones. $^ a \mathrm { N o }$ baseline v<sub>2</sub> is available in these bins. <sup>b</sup>Compared with the model at the published $p _ { T }$ points; not fitted (Sec. II D). <sup>c</sup>Two 95% CL upper limits (UL) in the $2 0 - 4 0 \%$ class, used as one-sided constraints.
<table><tr><td>System</td><td>Observable</td><td>Measurement</td><td>0 -20%</td><td> $2 0 - 4 0 \%$ </td><td>Not fitted</td></tr><tr><td rowspan="2">Au+Au, 200 GeV</td><td> $v _ { 2 }$ </td><td>PHENIX, conversion [35]</td><td>10/11</td><td>10/11</td><td> $3 . 5 -- 5 . 0 \mathrm { ~ G e V } ^ { a }$ </td></tr><tr><td>yield</td><td> $\mathrm { P H E N I X } , \mathrm { e x c e s s } + N _ { \mathrm { c o l l } } p p \ [ 3 4 ]$ </td><td>11/11</td><td>11/11</td><td></td></tr><tr><td rowspan="3">Pb+Pb, 2.76 TeV</td><td>yield</td><td>STAR, virtual photons [36]</td><td>0/9</td><td>0/9</td><td> $\mathrm { a l l ~ p o i n t s } ^ { b }$ </td></tr><tr><td>v2</td><td>ALICE [14]</td><td>13/16</td><td>13/16</td><td> $4 . 1 - - 6 . 2 \mathrm { G e V } ^ { a }$ </td></tr><tr><td>yield</td><td>ALICE [38]</td><td>20/20</td><td> $1 8 / 2 0 + 2 ~ \mathrm { U L } ^ { c }$ </td><td></td></tr></table>

Every other comparison uses the published $p _ { T }$ bins. The highest PHENIX $v _ { 2 }$ bin and the three ALICE $v _ { 2 }$ bins above 4.1 GeV are not fitted, because the published source files used for $B _ { n }$ give no $v _ { 2 }$ there. The uncertainty of each point is the quadrature sum of the statistical and systematic uncertainties, or the published total when only a total is given, and it is symmetrized for the fit. The two lowest ALICE 20%–40% yield points are 95% confidence-level upper limits; they enter as one-sided constraints; see Appendix A for the exact treatment.

## III. RESULTS

To test whether the pair-correlation mechanism can improve both observables while satisfying the conditions for a physical local distribution and emission rate, Fig. 2 compares the measured direct-photon yields and $v _ { 2 }$ with three reconstructions in each of the four collision classes. The two gray styles show Ref. [39] in diferent forms: the thin light-gray dash-dotted curve reproduces its published total calculation, while the thick dark-gray bin segments show the corresponding values in the experimental bins. The binned values define the internal fixed baseline $B = P + E + H$ of $\operatorname { E q . }$ (13). We use this baseline to see how much the correlation term improves agreement with the experimental data.

Each colored curve is an in-sample fit using the same four response columns, with coeficients fitted separately for each class. For each class we fit the yield and flow together using the following joint loss function

$$
\begin{array} { r l r } {  { \chi ^ { 2 } ( \psi ) = \sum _ { i \in Y } \Big ( \frac { Y _ { i } ( \psi ) - Y _ { i } ^ { \mathrm { e x p } } } { \sigma _ { i } } \Big ) ^ { 2 } } } \\ & { } & { + \sum _ { i \in v _ { 2 } } \Big ( \frac { Y _ { 2 , i } ( \psi ) / Y _ { 0 , i } ( \psi ) - v _ { 2 , i } ^ { \mathrm { e x p } } } { \sigma _ { i } } \Big ) ^ { 2 } , } \end{array}\tag{14}
$$

so each fitted bin is weighted by its own uncertainty. The two ALICE yield upper limits are included as one-sided constraints. The resulting $\chi ^ { 2 } ( \psi )$ measures fit quality for the yield and flow data.

The three colored curves in Fig. 2 show diferent treatments of the following two physical constraints. Constraint A requires a nonnegative total local photon rate, $r _ { A } \equiv \Gamma _ { \gamma } ^ { \mathrm { Q G P } } / \Gamma _ { \gamma } ^ { \mathrm { A M Y } } \geq 0$ for $0 . 3 1 \leq x \leq 1 5$ in cells with $T \ > \ T _ { \mathrm { s w } } ^ { ' } .$ Constraint B requires a nonnegative twoparticle distribution, $r _ { B } \equiv \mathcal { F } _ { a b } / f _ { a } f _ { b } \ge 0$ for $\xi _ { a } , \xi _ { b } \le 1 2$ and all $z _ { a b }$ in those cells. For a physical description, both constraints must be satisfied everywhere. Given the limitations of our construction discussed in the preceding section, we also test a soft-constraint fit that permits negative values but penalizes their weighted mean-square violation. The three fits are referred to as “hard”, “soft”, and “unconstrained”. The hard-constraint fit enforces both constraints on the test grids. The soft-constraint fit, used as the main $\operatorname { f i t } ,$ adds $P = \tau ^ { - 2 } \langle \mathrm { m i n } ( 0 , r ) ^ { 2 } \rangle$ to Eq. (14) for each constraint, with the chosen penalty parameters $\tau _ { A } = \tau _ { B } = 0 . 1$ $\mathrm { A t } ~ \tau = 0 . 1$ , a root-mean-square violation of 10% costs one unit of $\chi ^ { 2 }$ . Both soft penalties give equal weight to hot cells. The A penalty weights the tested x points equally because it evaluates the local emission rate at each point, while the B penalty uses the thermal two-particle phase-space weight because it evaluates the two-particle distribution. The unconstrained fit sets $\tau _ { A } = \tau _ { B } = \infty$ and leaves both constraints unenforced. To keep $v _ { 2 } ~ = ~ Y _ { 2 } / Y _ { 0 }$ numerically stable, we impose $Y _ { 0 } ^ { \mathrm { t o t } } \ge 0 . 0 2 B _ { 0 }$ in the evaluated bins and points. This numerical floor is inactive in all three reported fits.

Figure 2 shows how the joint yield and $v _ { 2 }$ fits with a two-particle correlation term compare with the internal baseline and experimental data. In the $v _ { 2 }$ panels (a,c,e,g), the dark-gray baseline rises toward intermediate $p _ { T }$ and then falls, but lies below nearly all measurements. The colored reconstructions retain this broad momentum dependence, with a shallow low- $p _ { T }$ dip in $\mathrm { A u } { + } \mathrm { A u }$ before they rise toward a broad shoulder. They tend to flatten toward the upper end of the modeled range and generally lie above the baseline at intermediate and higher $p _ { T }$ . At the lowest momenta, however, their deviations from the baseline are small and can have either sign.

We next compare the three colored reconstructions. The gold hard-constraint curve rises above the baseline around the intermediate-p<sub>T</sub> shoulder, whereas the orange-red soft-constraint curve develops a higher shoul der and the pink unconstrained curve gives the largest and broadest enhancement. The three curves separate most clearly around the intermediate-p shoulder. In these fits, their separation shows how strongly the positivity constraints limit the flow added by the correlation term.

![](images/16b9443e470ec2074745b76f91e40d562b8f234f4d8d275f2d4d1520cfe7afc5.jpg)

![](images/399766149d8d67850e11ddfb43221aafdeabe9975ba6dda8f7985991d5d6b957.jpg)

![](images/521cbc87ea928629db031da802d2c3f07398d8761584cd0b090b3a49fe2a28a9.jpg)

![](images/b537e3f36d4d87d5be430f51e1a600aeb5897a762aa4a184c0cce7d2abd659ab.jpg)  
FIG. 2. Direct-photon $v _ { 2 }$ (left) and invariant yield (right) in Au+Au collisions at $\sqrt { s _ { N N } } = 2 0 0 \mathrm { G e V } , 0 { - } 2 0 \% \mathrm { ( a , b ) }$ and 20−40% (c,d), and in Pb+Pb collisions at $\sqrt { s _ { N N } } = 2 . 7 6 \mathrm { ~ T e V } , 0 - 2 0 \% \mathrm { ~ ( e , f ) }$ and $2 0 - 4 0 \% ( \mathrm { g , h ) ; }$ the data are listed in Table I. Vertical bars show the symmetrized total uncertainty used in the fit, horizontal bars the published bins. Black circles show PHENIX and ALICE measurements (filled when fitted, open otherwise), and gray open diamonds show STAR measurements at their published points; arrows are 95% CL upper limits. Models are drawn bin by bin, with thin connectors between adjacent bins within the same binning scheme. Short tick marks show the model values at the STAR points and at the ALICE limits. The thin light-gray dash-dotted curve is the public results of $B = P { + } E { + } H$ [Eq. (13)] baseline given in Ref. [39], and the thick darkgray dash-dotted curve is our treatment that recasts the continuous results therein into the same $p _ { T }$ bins of the experiemental data. The three colored curves show the three $B + C$ reconstructions: gold denotes hard contraint with no violation, orange-red denotes soft contraint with $\tau _ { A } = \tau _ { B } = 0 . 1$ , and pink denotes unconstrained with $\tau _ { A } = \tau _ { B } = \infty$ . The insets in panels (b,d,f,h) show $( Y _ { i } ^ { \mathrm { m o d e l } } - Y _ { i } ^ { \mathrm { e x p } } ) / \sigma _ { i }$ for the dark-gray baseline and three colored reconstructions in the fitted PHENIX/ALICE yield bins, with σ<sub>i</sub> the symmetrized experimental total uncertainty. The light-gray band indicating $- 1 \leq ( Y _ { i } ^ { \mathrm { m o d e l } } - \dot { Y } _ { i } ^ { \mathrm { e x p } } ) / \sigma _ { i } \leq 1$ . The open triangle in (a) marks a PHENIX data point outside the plotted range, $v _ { 2 } = - 0 . 3 5 \pm 0 . 4 8$ , which was not fitted because no corresponding baseline data exist.

The comparison with the measurements shows how these curve diferences afect the v fit. Most measured $v _ { 2 }$ points remain above the baseline and the hard-constraint curve. Around the broad measured enhancement, the soft-constraint fit moves closer to the data, and the unconstrained fit often comes closer still or reaches the measured central values. The gap persists at low $p _ { T }$ , especially in $\mathrm { A u } { + } \mathrm { A u }$ , and the unconstrained curve is not closest to every point at higher $p _ { T }$ . Across these three reconstructions, the in-sample joint fit describes $v _ { 2 }$ better as the constraints are relaxed. The cost of this better fit is that negative local photon rates and negative pair distributions become more frequent, as shown later in Fig. $^ \mathrm { 4 ( g , h ) }$

The yield fits improve more clearly than the $v _ { 2 }$ fits. In the logarithmic main panels of Fig. $^ { 2 ( \mathrm { b , d , f , h } ) }$ the baseline appears close to the measurements, but the insets reveal a mainly negative ofset, with several Au+Au points outside the 1σ band. Adding the pair correlation raises the yield toward the fitted data over low and intermediate $p _ { T }$ . This improvement is already clear for the hard-constraint fit, which enforces both positivity constraints. Below about 6 GeV, nearly all Pb+Pb and Au+Au 20 − 40% points in the hard-constraint fit lie within the 1σ band, and $\mathrm { A u + A u ~ 0 - 2 0 \% }$ also improves, although some points there remain outside the band. For all three $\mathrm { P b + P b }$ fits and for the unconstrained fit in $\mathrm { A u } { + } \mathrm { A u }$ , the residuals have both signs, although the hard and soft-constraint fits in Au+Au remain mostly below PHENIX. At high $p _ { T }$ in Pb+Pb, the reconstructions return to the baseline because the present pair-response library sets the correction to zero beyond $x = 1 5$

The remaining low $- p _ { T }$ diferences in yield and $v _ { 2 }$ motivate a closer look at the pair-correlation response. Here this response includes only $2  2$ Compton and annihilation processes in the classical-statistics, leading-logarithmic limit [42]. At complete leading order, the equilibrium rate also includes near-collinear bremsstrahlung and inelastic pair annihilation, with Landau–Pomeranchuk–Migdal suppression [40]. These channels enter the AMY rate used for the local positivity condition, but their response to the pair correlation is absent from the fitted correction. The classical limit also drops the final-state occupation factors in Eq. (4). A correlation-induced correction from these missing terms could change $Y _ { 0 }$ and $Y _ { 2 }$ diferently after integration over the medium and thus afect $v _ { 2 } = Y _ { 2 } / Y _ { 0 }$ and yield. Extending the pair response to complete leading order, and then assessing next-to-leading-order corrections, would test whether these approximations contribute to the remaining low-p diferences.

In this section, we examine the origin of the change in $v _ { 2 }$ . Relative to the baseline $v _ { 2 } ^ { B } \equiv B _ { 2 } ^ { - } / B _ { 0 }$ , the change in $v _ { 2 }$ is exactly

$$
v _ { 2 } ^ { \mathrm { t o t } } - v _ { 2 } ^ { B } = \frac { \mathcal { K } } { B _ { 0 } + \Delta Y _ { 0 } } , \qquad \mathcal { K } \equiv \Delta Y _ { 2 } - v _ { 2 } ^ { B } \Delta Y _ { 0 } .\tag{15}
$$

For a positive total yield, v rises if and only if $\kappa > 0$

The quantity K has two terms. The first term is the added second harmonic. The second term, $- v _ { 2 } ^ { B } \Delta Y _ { 0 }$ , is the negative of the second harmonic expected if the added yield had the baseline anisotropy. If the correlation added only a positive second harmonic without changing the yield, then $\Delta Y _ { 0 } = 0$ and $\mathcal { K } = \Delta Y _ { 2 } > 0$ . If the correlation added only a positive yield without changing the second harmonic, then $\Delta Y _ { 2 } ~ = ~ 0$ and $\kappa = - v _ { 2 } ^ { B } \Delta Y _ { 0 } ~ < ~ 0$ for $v _ { 2 } ^ { B } > 0$

The upper panels of Fig. 3 show $- v _ { 2 } ^ { B } \Delta Y _ { 0 } / B _ { 0 }$ in blue and the second-harmonic increment $\Delta Y _ { 2 } / B _ { 0 }$ in orangeyellow for the main fit. The green line shows the full $v _ { 2 }$ change $\kappa / B _ { 0 }$ , and the green line is the sum of the orange-yellow and blue lines. Around the intermediatep<sub>T</sub> enhancement, $v _ { 2 } ^ { B } \Delta Y _ { 0 }$ and $\Delta Y _ { 2 }$ are often of the same order, and the absolute value of the yield term is smaller where $\kappa > 0$ Near the shoulder, the pair correlation adds a second harmonic, but the accompanying increase in yield ofsets part of its efect on $v _ { 2 }$ . The resulting rise in v is smaller than one would infer from $\Delta Y _ { 2 }$ alone. In some low- $- p _ { T }$ bins, K is negative, so the correlation does not raise v<sub>2</sub> in every bin.

We next isolate the contribution of the two temperature-dependent columns while keeping all mainfit coeficients fixed. Let $\Delta Y _ { n } ^ { ( C ) }$ denote the contributions from the two constant-feature columns, with $\kappa _ { C } \equiv$ $\Delta Y _ { 2 } ^ { ( C ) } - v _ { 2 } ^ { B } \Delta Y _ { 0 } ^ { ( C ) }$ . The temperature-column contribution is then

$$
\Delta Y _ { n } ^ { ( T ) } \equiv \Delta Y _ { n } - \Delta Y _ { n } ^ { ( C ) } ,\tag{16}
$$

$$
\begin{array} { r } { \kappa _ { T } \equiv \kappa - \kappa _ { C } = \Delta Y _ { 2 } ^ { ( T ) } - v _ { 2 } ^ { B } \Delta Y _ { 0 } ^ { ( T ) } . } \end{array}\tag{17}
$$

In the lower panels of Fig. 3, the semi-transparent solid curves show $\kappa _ { C } / B _ { 0 } .$ , and the semi-transparent dashed curves show $\kappa _ { T } / B _ { 0 }$ obtained by subtracting them from the opaque solid curves. This is a decomposition of the main fit rather than a separate fit with only the temperature columns. At the intermediate-p<sub>T</sub> shoulder in these panels, $\kappa _ { T }$ supplies most of the positive remainder in all four collision classes, while the constant columns contribute more in the highest fitted bins.

This comparison suggests that the temperaturedependent correlation helps produce a positive K by changing the relative contributions of hotter and cooler regions. Without the pair correlation, the temperature weighting of each fluid cell’s thermal emission is constrained by the local one-particle distributions and the equation of state. The connected pair correlation has zero one-particle marginals [Eq. (3)], so it leaves the oneparticle distributions unchanged. Its amplitude can depend on $q _ { T } = \ln [ T / ( 0 . 2 \ \mathrm { G e V } ) ]$ ], giving the correlation correction an additional temperature-dependent weight across fluid cells.

The pattern is less uniform at low $p _ { T }$ , which may partly reflect the leading-logarithmic approximation used for the pair-correlation response, as discussed above.

We now examine whether the reconstructed correlation satisfies the two physical positivity conditions introduced above, the result in shown in Figure 4.

![](images/de0cac54e116ec04205a3f0c1c3eebab55070730d0a7f9586fbd7f64219a190d.jpg)  
FIG. 3. Decomposition for the main fit. From left to right, the columns correspond to Au+Au 0 − 20%, Au+Au $2 0 - 4 0 \%$ $\mathrm { P b + P b 0 - 2 0 \% }$ , and $\mathrm { P b + P b 2 0 - 4 0 \% } ,$ rspectively. Upper panels (a)–(d) show the full-fit $\Delta Y _ { 2 } / B _ { 0 }$ in orange-yellow, $- v _ { 2 } ^ { B } \Delta Y _ { 0 } / B _ { 0 }$ in blue, and their sum $\kappa / B _ { 0 }$ in green in the fitted v<sub>2</sub> bins. The green $\kappa / B _ { 0 }$ line is thicker than the orange-yellow and blue lines. Lower panels $\mathrm { { ( e ) - ( h ) } }$ show $\kappa / B _ { 0 }$ as an opaque dark green line, its constant-column part $\kappa _ { C } / B _ { 0 }$ as a semi-transparent solid line, and its temperature-column part $\kappa _ { T } / B _ { 0 }$ as a semi-transparent dashed line.

Panels (a)–(d) show the four-volume-weighted laboratory-frame QGP yield contribution from individual hot cells as a function of $p _ { T }$ . The solid curves are the means across cells, and the shaded regions show one standard deviation of the cell contributions. The means remain positive and fall toward zero with increasing $p _ { T }$ although a small fraction of cells contributes negatively at the shown momenta.

Panels $\mathrm { ( e , f ) }$ show two cuts of the local pair distribution (2), ${ \mathcal { F } } _ { a b ; }$ from the main fit. Panel (e) varies temperature-scaled-local-energy of particle $a , \xi _ { a } .$ , at fixed $\xi _ { b } = 2 . 5$ and relative overlap $z _ { a b } = 0 . 3 7 5 .$ and displays $e ^ { \xi _ { b } } \mathcal { F } _ { a b }$ . Panel (f) varies $\xi _ { b }$ at fixed $\xi _ { a } = 6$ and $z _ { a b } = 1$ and displays $e ^ { \xi _ { a } } \mathcal { F } _ { a b }$ . The solid curves are four-volumeweighted means across hot cells, and the shaded regions show one standard deviation. The means and shaded bands remain positive over both displayed ranges.

Panels (a)–(d) and $^ { ( \mathrm { e , f } ) }$ show averages or selected cuts, which do not establish positivity over the full local do main. The positivity conditions require a nonnegative local rate over the full range of x and a nonnegative pair distribution over the full ranges of $\xi _ { a } , \xi _ { b }$ , and $z _ { a b }$ . These result are shown in panels (g,h) of Fig. 4, which quantify the relation between the $v _ { 2 }$ fit quality and violations of local positivity. Both panels plot $\begin{array} { r } { \sum _ { i \in v _ { 2 } } \operatorname { p u l l } _ { v _ { 2 } , i } ^ { 2 } / n _ { v _ { 2 } } . } \end{array}$ , the mean squared pull over the fitted v<sub>2</sub> bins, on the vertical axis. Here pul $\mathrm { l } _ { v _ { 2 } , i } = ( v _ { 2 , i } ^ { \mathrm { m o d e l } } - v _ { 2 , i } ^ { \mathrm { e x p } } ) / \sigma _ { i } , \sigma _ { i }$ is the symmetrized total experimental uncertainty, and $n _ { v _ { 2 } }$ is the number of fitted $v _ { 2 }$ bins. The horizontal axes give the percentage of negative local-rate points in (g) and the thermal-pair-weighted percentage of negative pairdistribution values in (h). Each color identifies a colli sion class, and each dotted line connects its fits as the constraints are relaxed. Open squares mark the hardconstraint fit, filled diamonds the main soft-constraint fit with $\tau _ { A } = \tau _ { B } = 0 . 1$ , and open triangles the unconstrained fit. Filled circles mark the other soft-constraint fits, with $\tau _ { A } = \tau _ { B }$ set to 0.03, 0.3, or 1. From the hardconstraint fit through the main soft-constraint fit to the unconstrained fit, the mean squared $v _ { 2 }$ pull decreases in all four collision classes, while both negative fractions increase.

This trade-of motivates a future calculation of the baseline and pair-correlation response with the same hydrodynamic profile and extending the latter to complete leading order. These calculations would test whether the v<sub>2</sub> fit can improve while both positivity conditions are satisfied.

We next test whether or not our fitted coeficients are generalizable, i.e., whether those fitted for one collision class can be tranferred and describe another. Figure 5 compares the transferred yield and $v _ { 2 }$ curves with the target-class data. Purple and light purple curves use coeficients fitted to $\mathrm { A u + A u ~ 0 - 2 0 \% }$ and 20 − 40%, while dark and light green curves use coeficients fitted to the corresponding Pb+Pb classes. The border color identifies the target class, so the matching, heavier curve shows its own fit and the other colored curves show transfers. Figure 6 shows the $v _ { 2 }$ fit quality and positivity diagnostics for these transfers. The dark dashed curves in Fig. 5 and the dark-filled markers in Fig. 6 show two additional

![](images/329bcb6a7f2990a7ea2680acf627fd15207f78f1d6a7eb1ed9137c486054f098.jpg)

![](images/c5b2fbfc27934aa4eebb9559dd9a3e088be3a4e7b73d05e9a51ebf31e2f2568d.jpg)

![](images/2010aca4739bb2aee8eef814f198d6356db2fd9bc0a81a768311de411ae4a3e9.jpg)

![](images/5285869170523aa792428f6af7ebd95853fbdc7dddea98c969f7c224371bbf42.jpg)

![](images/06f72dd68ff47884a6d5624afc0222338f9b3d6246e4a628f00eaca5ad6e13b6.jpg)

![](images/76cfeb9a8fbf0f753c660539d7f275263cbcba8f332810fc952b4ecc52d81f0c.jpg)

![](images/c53769a5488247e809a9189aa96e6b5cfde6568ef84d156e1ce472bada2af326.jpg)

![](images/44d43cf976849459ec5c916cf6985cc3de6c8c8e53e66380bd8bbfd447654a71.jpg)  
FIG. 4. Diagnostics of the physical constraints. Panels (a)–(d) show the $p _ { T }$ dependence of the thermal-photon yield from local fluid cells in $\mathrm { A u + A u ~ 0 - 2 0 \% }$ $\mathrm { A u + A u ~ 2 0 - 4 0 \% }$ , Pb+Pb 0 − 20%, and Pb+Pb $2 0 - 4 0 \%$ , respectively. The solid curves are arithmetic means across all hot cells, and the shaded regions show one population standard deviation among cells. The vertical axes use a symmetric logarithmic scale that is linear for $| Y _ { 0 , c } ^ { \mathrm { Q G P } } | \leq \mathrm { 1 0 ^ { - 5 } ~ G e V ^ { - 2 } }$ and displays negative values. Panels (e,f) show two cuts of the main-fit local pair distribution ${ \mathcal { F } } _ { a b } .$ . Panel (e) varies $\xi _ { a }$ at fixed $\xi _ { b } = 2 . 5$ and $z _ { a b } = 0 . 3 7 5$ , with $e ^ { 2 . 5 } { \mathcal { F } } _ { a b }$ on the vertical axis. Panel (f) varies $\xi _ { b }$ at fixed $\xi _ { a } = 6$ and $z _ { a b } = 1$ , with $e ^ { 6 } \mathcal { F } _ { a b }$ on the vertical axis. The solid curves are four-volume-weighted means across hot cells, and the shaded regions show one standard deviation. The colors in (e)–(h) follow the collision-class order of $\mathrm { ( a ) - ( d ) }$ . Panels (g,h) show the summed squared $v _ { 2 }$ pulls per fitted bin on the vertical axis. Their horizontal axes show the percentage of negative local-rate test points in $\mathrm { ( g ) }$ and the percentage of negative pair-distribution values in (h). For each class, a dotted line joins the hard-constraint fit and the fits with $\tau _ { A } = \tau _ { B } = \tau \ \mathrm { a t } \ \tau = 0 . 0 3 , 0 . 1 , 0 . 3 , 1 .$ and ∞, in that order. Open squares mark the hard-constraint fit (A and B hard), filled diamonds the main soft-constraint fit $( \tau = 0 . 1 )$ , open triangles the unconstrained fit $( \tau = \infty )$ , and filled circles the other soft-constraint fits.

![](images/470a3501e7815871a55533258a62844f29f819fbdbd63db855fbfc02275d2969.jpg)

![](images/300520a036ad8ba107a232bcf32f324ab8edbc3da3d46a3b271e2b25792e8951.jpg)

![](images/0931d9d5ece7e94416bcd624b31631c5c6a9f1c5ed477ad69bfd4ff1d9dc0b8d.jpg)

![](images/2e971d9b27d9e73990c34e6a9d38a3a8c6d82fd3cb2e1017152a51d9d75354e7.jpg)

![](images/5bcb8f51b0732573b3342e762a224f0f1247e4793333cc918136eb6b74f35464.jpg)  
FIG. 5. Transfer of the four class-specific main-fit coeficient vectors and fits shared across the two centralities of each collision system. Panels (a,b), (c,d), (e,f), and $^ \mathrm { ( g , h ) }$ target Au+Au 0 − 20%, Au+Au 20 − 40%, Pb+Pb $0 - 2 0 \% ,$ , and Pb+Pb 20 − 40%, respectively. The left panel of each pair shows direct-photon $v _ { 2 }$ and the right panel the invariant yield. The colored border of each panel follows the same color in Fig. 4(a)–(d) and identifies its target collision class; curve color identifies the source class of the transferred coeficients. In each target pair, the heavier curve is the target’s own $\operatorname { f i t } .$ The dark dashed curve in each panel uses the coeficient vector fitted jointly to the two centralities of that collision system. The two gray dash-dotted curves show the same calculation from Ref. [39]: the thick dark-gray curve shows its binned form, used as the internal baseline $B ,$ while the thin light-gray curve shows its published continuous form. Filled circles show the fitted PHENIX or ALICE measurements with uncertainties, open v<sub>2</sub> circles denote published bins excluded from the fit, and open diamonds show STAR yields at thei published $p _ { T }$ values. Colored and dark short ticks at the STAR points and at the two ALICE 20 − 40% upper-limit bins mark the class-specific and shared-fit model predictions at those points. Downward arrows indicate the experimental 95% confidence-level upper limits.

fits, one for each collision system, in which both central- ities share one coeficient vector. The thick dark-gray

![](images/d473661ec941af545127aca0163348ca8e6fe281ad5ef006961aa0329c42aca9.jpg)  
FIG. 6. Fit quality and positivity diagnostics for transferred, shared, and baseline coeficients. Each of the four main-fit coeficient vectors is applied to every target class without refitting. Both panels show the summed squared v<sub>2</sub> pulls per fitted bin for the target class. Panel (a) gives the percentage of negative local-rate test points, while panel (b) gives the percentage of negative pair-distribution values. Marker shape identifies the target class: circles and squares represent $\mathrm { A u + A u ~ 0 - 2 0 \% }$ and $2 0 - 4 0 \%$ , and triangles and diamonds represent Pb+Pb 0 − 20% and $2 0 - 4 0 \%$ , respectively. For the transferred coeficients, outline color identifies the target class and fill color identifies the source class. Matching fill and outline colors identify a target class’s own fit. The larger dark-filled markers with target-colored outlines show the coeficients fitted jointly to the two centralities of the corresponding collision system. Gray-outlined open markers show the internal baseline $B ,$ for which both negative fractions are zero. Dotted lines connect the four source vectors for each target as a visual guide and do not represent a parameter scan.

![](images/cfd8de5429b2b1bbd90bbaa0ec7d2cba308a38dd3404f5b7bc53671aae226606.jpg)  
FIG. 7. Coeficients of the four dominating modes $\psi _ { \alpha } \equiv$ $\psi _ { A , n l m }$ with $A = C \ \mathrm { o r } \ T$ and $( n , l , m ) = ( 0 , 0 , 3 )$ or $( 1 , 0 , 2 ) .$ Diferent symbols represent diferent collision systems, and they have been shifted hotizontally for better visibility.

dash-dotted curves in Fig. 5 and the open gray-outlined markers at zero negative fraction in Fig. 6 show the internal baseline B.

The yield curves change less with the coeficient source than the $v _ { 2 }$ curves. Within each collision energy, coeficients from one centrality retain most of the joint yield and $v _ { 2 }$ improvement over the baseline in the other centrality. The transferred $v _ { 2 }$ pulls per fitted bin also remain close to those of the target class’s own fit. Summed over the fitted yield and $v _ { 2 }$ bins of both centralities, the squared pulls rise from 64.7 for the two separate Au+Au fits to 66.0 for their shared fit, and from 31.9 to 33.4 for Pb+Pb.

Across collision energies, the transfer is asymmetric. Compared with each Pb+Pb target’s own fit, Au+Au coeficients can give similar or smaller $v _ { 2 }$ pulls, but larger negative fractions for both the local rate and the pair distribution. Conversely, compared with each Au+Au target’s own fit, Pb+Pb coeficients reduce both negative fractions, but give larger $v _ { 2 }$ pulls and retain less of the improvement in the joint fit relative to the baseline. Figure 7 shows the fitted coeficients $\psi _ { \mathrm { { C } } , n m \ell }$ and $\psi _ { T , n m \ell }$ for the two selected momentum modes, $( n , m , \ell ) = ( 0 , 0 , 3 )$ and (1, 0, 2). For both modes, the $\mathrm { A u } { + } \mathrm { A u }$ fits give larger values of $\psi _ { T , n m \ell }$ than the Pb+Pb fits, so the corresponding mode amplitudes increase more rapidly with ln $T ,$ Applying the Au+Au coeficients to the hotter Pb+Pb profiles thus extends this temperature dependence beyond the range in the Au+Au profiles, which may contribute to the increased positivity violations. Applying the Pb+Pb coeficients to the Au+Au profiles weakens the temperature-dependent reweighting between hotter and cooler fluid cells, which may worsen the $v _ { 2 }$ fit.

Within each collision energy, the centrality-transfer and shared-fit tests support using one coeficient vector for the two centralities studied. Across energies, however, the transfer involves a trade-of between v fit quality and local positivity. Within the selected basis, the present results do not establish a common coeficient vector that describes both collision energies while satisfying both local positivity conditions.

## IV. CONCLUSIONS

With a forward model in which a connected twoparticle correlation of the emitting partons changes the local 2 → 2 photon rate developed by the Authors in Ref. [42], we ask whether there exists such a correlation can describe direct-photon yields and elliptic flow together with physically meaningful parameters. Connected modes are defined by zero marginals which leave the one-particle distributions unchanged, allowing us to isolate the pair-correlation response. We folded this response over hydrodynamic profiles, added it to a fixed direct-photon baseline, and fitted the measured yields and v while varying the enforcement of local nonnegativity.

The fitted correlation response changes both the yield and $v _ { 2 }$ appreciably relative to the baseline. The correlation response improves the fitted PHENIX and AL-ICE yields relative to the baseline even with both localpositivity conditions enforced, although it also increases the tension with the unfitted STAR yields in $\mathrm { A u } { + } \mathrm { A u }$ Under these hard conditions, the $v _ { 2 }$ fit also improves, though less than when the conditions are relaxed. Relaxing the conditions improves the in-sample $v _ { 2 }$ fit more strongly, but in the main fit $0 . 4 \% -- 4 . 4 \%$ of tested local rate points and at most 1.3% of the weighted pair domain are negative. In the unconstrained fits, further improvement accompanies negative local rates at $2 \% -- 1 5 \%$ of tested points.

[1] E. L. Feinberg, Direct Production of Photons and Dileptons in Thermodynamical Models of Multiple Hadron Production, Nuovo Cim. A 34, 391 (1976).

[2] E. V. Shuryak, Quark-Gluon Plasma and Hadronic Production of Leptons, Photons and Psions, Phys. Lett. B 78, 150 (1978).

[3] K. Kajantie and H. I. Miettinen, Temperature Measurement of Quark-Gluon Plasma Formed in High-Energy Nucleus-Nucleus Collisions, Z. Phys. C 9, 341 (1981).

[4] L. D. McLerran and T. Toimela, Photon and Dilepton Emission from the Quark - Gluon Plasma: Some General Considerations, Phys. Rev. D 31, 545 (1985).

[5] K. Kajantie, J. I. Kapusta, L. D. McLerran, and A. Mekjian, Dilepton Emission and the QCD Phase Transition in Ultrarelativistic Nuclear Collisions, Phys. Rev. D 34, 2746 (1986).

[6] I. Tserruya, Electromagnetic Probes, Landolt-Bornstein 23, 176 (2010), arXiv:0903.0415 [nucl-ex].

[7] C. Gale, Electromagnetic Radiation from High-Energy Nuclear Collisions (2025) arXiv:2502.13938 [nucl-th].

Within the selected basis, the two centralities at each collision energy are consistent with a shared coeficient vector, whereas transfer between energies is asymmetric.

The improved agreement may reflect a physical paircorrelation mechanism, although the use of diferent hydrodynamic evolutions for the baseline and response and the leading-logarithmic approximation for the response limit the conclusions that can be drawn. To address these limitations, we will compute the baseline and pair correlation response on the same hydrodynamic evolution with a consistent coupling. We will also extend the pair-correlation response to complete leading order and then assess next-to-leading-order corrections. These calculations will test whether a fit to both observables can satisfy both local positivity conditions, and they pave the way for a future study that incorporates connected correlations obtained by solving the transport equation under conditions relevant to heavy-ion collisions.

## ACKNOWLEDGMENTS

This work is supported by Yantai university under the grant 2226001. This work is also supported by NSFC by grant No. 12575143 and by Tsinghua University under grant Nos. 04200500123, 531205006, 533305009. We also acknowledge the support by center of high performance computing, Tsinghua University.

[8] G. Jackson, Theory overview on electroweak emission from heavy-ion collisions, EPJ Web Conf. 364, 01023 (2026), arXiv:2509.00262 [hep-ph].

[9] L. Du, Advancing multimessenger approaches in heavyion collisions: Insights from electromagnetic probes, EPJ Web Conf. 339, 01016 (2025), arXiv:2503.09045 [nuclth].

[10] A. Adare et al. (PHENIX), Enhanced production of direct photons in Au+Au collisions at $\sqrt { s _ { N N } } = 2 0 0 \ \mathrm { G e V }$ and implications for the initial temperature, Phys. Rev. Lett. 104, 132301 (2010), arXiv:0804.4168 [nucl-ex].

[11] A. Adare et al. (PHENIX), Observation of direct-photon <sub>collective flow in</sub> √<sub>sNN = 200 GeV Au+Au collisions,</sub> Phys. Rev. Lett. 109, 122302 (2012), arXiv:1105.4126 [nucl-ex].

[12] F.-M. Liu and S.-X. Liu, Quark-gluon plasma formation time and direct photons from heavy ion collisions, Phys. Rev. C 89, 034906 (2014), arXiv:1212.6587 [nucl-th].

[13] R. Rapp and H. van Hees, Thermal Dileptons as Fireball Thermometer and Chronometer, Phys. Lett. B 753, 586 (2016), arXiv:1411.4612 [hep-ph].

[14] S. Acharya et al. (ALICE), Direct photon elliptic flow in <sub>Pb-Pb collisions at</sub> √<sub>sNN = 2.76 TeV, Phys. Lett. B 789,</sub> 308 (2019), arXiv:1805.04403 [nucl-ex].

[15] P. Dasgupta, G.-L. Ma, R. Chatterjee, L. Yan, S. Zhang, and Y.-G. Ma, Thermal photons as a sensitive probe of α-cluster in C + Au collisions at the BNL Relativistic Heavy Ion Collider, Eur. Phys. J. A 57, 134 (2021), arXiv:2007.09543 [nucl-th].

[16] J.-A. Sun and L. Yan, Estimating the magnetic field strength in heavy-ion collisions via direct photon elliptic flow, Phys. Rev. C 109, 034917 (2024), arXiv:2311.03929 [nucl-th].

[17] P. Stankus, Direct photon production in relativistic heavy-ion collisions, Ann. Rev. Nucl. Part. Sci. 55, 517 (2005).

[18] F.-M. Liu and K. Werner, A Systematic study on direct photon production from central heavy ion collisions, J. Phys. G 36, 035101 (2009), arXiv:0712.3619 [hep-ph].

[19] I. Vitev and B.-W. Zhang, A Systematic study of direct photon production in heavy ion collisions, Phys. Lett. B 669, 337 (2008), arXiv:0804.3805 [hep-ph].

[20] G. David, Direct real photons in relativistic heavy ion collisions, Rept. Prog. Phys. 83, 046301 (2020), arXiv:1907.08893 [nucl-ex].

[21] J. F. Owens, Large Momentum Transfer Production of Direct Photons, Jets, and Particles, Rev. Mod. Phys. 59, 465 (1987).

[22] J. Churchill, L. Yan, S. Jeon, and C. Gale, Emission of electromagnetic radiation from the early stages of relativistic heavy-ion collisions, Phys. Rev. C 103, 024904 (2021), arXiv:2008.02902 [hep-ph].

[23] O. Garcia-Montero, A. Mazeliauskas, P. Plaschke, and S. Schlichting, Pre-equilibrium photons from the early stages of heavy-ion collisions, JHEP 03, 053, arXiv:2308.09747 [hep-ph].

[24] O. Garcia-Montero, P. Plaschke, and S. Schlichting, Scaling of pre-equilibrium dilepton production in QCD kinetic theory, Phys. Rev. D 111, 034036 (2025), arXiv:2403.04846 [hep-ph].

[25] J.-F. Paquet, C. Shen, G. S. Denicol, M. Luzum, B. Schenke, S. Jeon, and C. Gale, Production of photons in relativistic heavy-ion collisions, Phys. Rev. C 93, 044906 (2016), arXiv:1509.06738 [hep-ph].

[26] S. Turbide, R. Rapp, and C. Gale, Hadronic production of thermal photons, Phys. Rev. C 69, 014903 (2004), arXiv:hep-ph/0308085.

[27] R. J. Fries, B. Muller, and D. K. Srivastava, Highenergy photons from passage of jets through quark gluon plasma, Phys. Rev. Lett. 90, 132301 (2003), arXiv:nuclth/0208001.

[28] S. Turbide, C. Gale, S. Jeon, and G. D. Moore, Energy loss of leading hadrons and direct photon production in evolving quark-gluon plasma, Phys. Rev. C 72, 014906 (2005), arXiv:hep-ph/0502248.

[29] G.-Y. Qin, J. Ruppert, C. Gale, S. Jeon, and G. D. Moore, Jet energy loss, photon production, and photonhadron correlations at RHIC, Phys. Rev. C 80, 054909 (2009), arXiv:0906.3280 [hep-ph].

[30] X.-Y. Wu, H. Gao, B. Forster, C. Gale, G. Jackson, and S. Jeon, Thermal Dilepton Polarization and Dynamics of the QCD Plasma in Relativistic Heavy-Ion Collisions, Phys. Rev. Lett. 134, 242301 (2025), arXiv:2412.15052 [nucl-th].

[31] H. Gao, X.-Y. Wu, C. Gale, G. Jackson, and S. Jeon, The polarization of thermal dileptons emitted in high-energy heavy-ion collisions (2026), arXiv:2605.30180 [nucl-th].

[32] L. Du and U. Heinz, Electromagnetic Tomography of Radial Flow in the Quark-Gluon Plasma, Phys. Rev. Lett. 136, 102301 (2026), arXiv:2505.20752 [hep-ph].

[33] J. Churchill, L. Du, C. Gale, G. Jackson, and S. Jeon, Virtual Photons Shed Light on the Early Temperature of Dense QCD Matter, Phys. Rev. Lett. 132, 172301 (2024), arXiv:2311.06951 [nucl-th].

[34] A. Adare et al. (PHENIX), Centrality dependence of low-momentum direct-photon production in Au+Au collisions at $\sqrt { s _ { N N } } = 2 0 0$ GeV, Phys. Rev. C 91, 064904 (2015), arXiv:1405.3940 [nucl-ex].

[35] A. Adare et al. (PHENIX), Azimuthally anisotropic emission of low-momentum direct photons in Au+Au collisions at $\sqrt { s _ { { } _ { N N } } } = 2 0 0$ GeV, Phys. Rev. C 94, 064901 (2016), arXiv:1509.07758 [nucl-ex].

[36] L. Adamczyk et al. (STAR), Direct virtual photon pro-<sub>duction in Au+Au collisions at</sub> √<sub>sNN = 200 GeV, Phys.</sub> Lett. B 770, 451 (2017), arXiv:1607.01447 [nucl-ex].

[37] N. J. Abdulameer et al. (PHENIX), Nonprompt directphoton production in Au+Au collisions at sNN=200 GeV, Phys. Rev. C 109, 044912 (2024), arXiv:2203.17187 [nucl-ex].

[38] J. Adam et al. (ALICE), Direct photon production in Pb-<sub>Pb</sub> <sub>collisions</sub> <sub>at</sub> √<sub>sNN =</sub> <sub>2.76</sub> <sub>TeV,</sub> <sub>Phys.</sub> <sub>Lett.</sub> <sub>B</sub> <sub>754,</sub> 235 (2016), arXiv:1509.07324 [nucl-ex].

[39] C. Gale, J.-F. Paquet, B. Schenke, and C. Shen, Multimessenger heavy-ion collision physics, Phys. Rev. C 105, 014909 (2022), arXiv:2106.11216 [nucl-th].

[40] P. B. Arnold, G. D. Moore, and L. G. Yafe, Photon emission from quark gluon plasma: Complete leading order results, JHEP 12, 009, arXiv:hep-ph/0111107.

[41] X. Lu and S. Shi, Spectral Bogoliubov-Born-Green-Kirkwood-Yvon hierarchical approach: A scalable scheme for nonlinear Boltzmann and correlation kinetics, Phys. Rev. C 113, 034917 (2026), arXiv:2507.14243 [nucl-th].

[42] X. Lu and S. Shi, Probing Pair Correlations in QCD Matter with Photon Spectra (2026), arXiv:2606.02414 [hepph].

[43] C. Shen, Z. Qiu, H. Song, J. Bernhard, S. Bass, and U. Heinz, The iEBE-VISHNU code package for relativistic heavy-ion collisions, Comput. Phys. Commun. 199, 61 (2016), arXiv:1409.8164 [nucl-th].

[44] S. Shi, J. Liao, and M. Gyulassy, Global constraints from RHIC and LHC on transport properties of QCD fluids in CUJET/CIBJET framework, Chin. Phys. C 43, 044101 (2019), arXiv:1808.05461 [hep-ph].

## Appendix A: Four-column inverse fit

We select one set of four columns for all collision classes by forward selection. The candidate pool has 73 connected modes, each multiplied by either q<sub>C</sub> or $q _ { T }$ , for 146 columns. We perform the selection with both conditions A and B imposed as hard constraints. At each step, we screen the remaining candidates on a coarse constraint grid, refit the six best candidates on the full constraint grid, and add the one with the smallest sum of the four class objectives. The corrected Akaike criterion is smallest at two columns on this selection path, but we retain four so that one direction remains free after the three active hard constraints are imposed. The selected columns are the modes (0, 0, 3) and (1, 0, 2) with both q<sub>C</sub> and $q _ { T }$ We use this basis in every constraint setting to compare the fits directly. Within each setting, we fit the coefficients separately for each class by minimizing Eq. (14) with the corresponding positivity constraints or penalties and a small ridge regularizer.

The two lowest ALICE 20 − 40% yield points are 95% confidence-level upper limits, so they are omitted from the ordinary yield-residual sum in Eq. (14). They add the one-sided term

$$
\chi _ { \mathrm { U L } } ^ { 2 } ( \psi ) = \sum _ { i \in \mathrm { U L } } \left[ \frac { \operatorname* { m a x } \{ 0 , Y _ { i } ( \psi ) - Y _ { i } ^ { \mathrm { U L } } \} } { Y _ { i } ^ { \mathrm { U L } } / 1 . 6 4 5 } \right] ^ { 2 } .
$$

Here $Y _ { i } ^ { \mathrm { U L } }$ is the reported upper limit, and $Y _ { i } ^ { \mathrm { U L } } / 1 . 6 4 5$ is the adopted one-sided Gaussian width. The term vanishes when the predicted yield is below the limit and penalizes only predictions above it.