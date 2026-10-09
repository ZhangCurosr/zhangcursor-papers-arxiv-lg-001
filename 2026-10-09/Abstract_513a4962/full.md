# Com<sub>p</sub>osite Online-to-Noncon<sub>v</sub>ex Con<sub>v</sub>ersion <sub>w</sub>ith O<sub>p</sub>tim<sub>a</sub>l Or<sub>ac</sub>l<sub>e</sub> C<sub>o</sub>m<sub>p</sub>l<sub>e</sub>xit<sub>y</sub>

Min<sub>gy</sub>i Li<sup>∗</sup> T<sub>a</sub>ir<sub>a</sub> T<sub>suc</sub>hi<sub>ya</sub><sup>†</sup> Kenji Yamanis<sup>h</sup>i<sup>‡</sup>

O<sub>c</sub>t<sub>o</sub>b<sub>er</sub> 8<sub>,</sub> 2026

## Abstract

We consider stochastic nonsmooth nonconvex com<sub>p</sub>osite o<sub>p</sub>timization<sub>,</sub> which includes several im<sub>p</sub>ortant prob<sup>l</sup>ems suc<sup>h</sup> as constrained optimization and t<sup>h</sup>e regu<sup>l</sup>arized training o<sup>f</sup> neura<sup>l</sup> networ<sup>k</sup>s. T<sup>h</sup>e objective is th<sub>e sum o</sub>f <sub>a poss</sub>ibl<sub>y nonsmoo</sub>th <sub>nonconvex</sub> Li<sub>psc</sub>hit<sub>z</sub> f<sub>unc</sub>ti<sub>on an</sub>d <sub>a convex regu</sub>l<sub>ar</sub>i<sub>zer, an</sub>d th<sub>e</sub> f<sub>unc</sub>ti<sub>on</sub> i<sub>s accesse</sub>d th<sub>roug</sub>h <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c gra</sub>di<sub>en</sub>t<sub>s or</sub> f<sub>unc</sub>ti<sub>on va</sub>l<sub>ues.</sub> Th<sub>e goa</sub>l i<sub>s</sub> t<sub>o</sub> fi<sub>n</sub>d <sub>a po</sub>i<sub>n</sub>t th<sub>a</sub>t <sub>sa</sub>ti<sub>s</sub>fi<sub>es a</sub> Go<sup>l</sup>dstein-type stationarity condition designed <sup>f</sup>or composite objectives. To our <sup>k</sup>now<sup>l</sup>edge, no orac<sup>l</sup>e <sub>co</sub>m<sub>p</sub>l<sub>e</sub>xit<sub>y</sub> b<sub>ou</sub>nd f<sub>o</sub>r thi<sub>s se</sub>ttin<sub>g</sub> i<sub>s</sub> kn<sub>ow</sub>n <sub>u</sub>nd<sub>e</sub>r fir<sub>s</sub>t-<sub>o</sub>rd<sub>e</sub>r <sub>access, a</sub>nd <sub>e</sub>xi<sub>s</sub>tin<sub>g co</sub>m<sub>p</sub>l<sub>e</sub>xiti<sub>es u</sub>nd<sub>e</sub>r <sub>zero</sub>th<sub>-or</sub>d<sub>er access are su</sub>b<sub>op</sub>ti<sub>ma</sub>l<sub>.</sub> T<sub>o</sub> h<sub>an</sub>dl<sub>e</sub> thi<sub>s</sub> i<sub>ssue, we emp</sub>l<sub>oy</sub> th<sub>e</sub> f<sub>ramewor</sub>k <sub>o</sub>f <sub>on</sub>li<sub>ne-</sub>t<sub>o-nonconvex</sub> <sub>convers</sub>i<sub>on,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>c</sub>h<sub>ooses</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> di<sub>rec</sub>ti<sub>ons</sub> b<sub>y</sub> <sub>an</sub> <sub>on</sub>li<sub>ne</sub> l<sub>earner</sub> <sub>an</sub>d i<sub>s</sub> k<sub>nown</sub> t<sub>o</sub> <sub>ac</sub>hi<sub>eve</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>ra</sub>t<sub>es</sub> f<sub>or</sub> noncom<sub>p</sub>osite <sub>p</sub>roblems. We extend the framework to our com<sub>p</sub>osite scenario b<sub>y</sub> introducin<sub>g</sub> new losses f<sub>or</sub> th<sub>e</sub> l<sub>earner, w</sub>hi<sub>c</sub>h <sub>con</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e re u</sub>l<sub>ar</sub>i<sub>zer</sub> it<sub>se</sub>lf <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> it<sub>s</sub> li<sub>near</sub>i<sub>za</sub>ti<sub>on an</sub>d f<sub>or w</sub>hi<sub>c</sub>h <sub>a var</sub>i<sub>an</sub>t <sub>o</sub>f <sub>on</sub>li<sub>ne m</sub>i<sub>rror</sub> d<sub>escen</sub>t <sub>ac</sub>hi<sub>eves</sub> l<sub>ow regre</sub>t<sub>.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e resu</sub>lti<sub>ng a</sub>l<sub>gor</sub>ith<sub>m</sub> fi<sub>n</sub>d<sub>s suc</sub>h <sub>a po</sub>i<sub>n</sub>t <sub>w</sub>ith $O ( \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ stochastic <sub>g</sub>radient <sub>q</sub>ueries or $O ( d \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ function-value queries, where δ is the Goldstein radius, ε is the stationarity tolerance, and d is the dimension. These rates match the optimal ones for noncom<sub>p</sub>osite nonsmooth nonconvex o<sub>p</sub>timization<sub>,</sub> demonstratin<sub>g</sub> that the additional convex re<sub>g</sub>ularizer d<sub>oes no</sub>t <sub>worsen</sub> th<sub>e orac</sub>l<sub>e comp</sub>l<sub>ex</sub>it<sub>y.</sub> W<sub>e a</sub>l<sub>so g</sub>i<sub>ve ra</sub>t<sub>es</sub> f<sub>or</sub> th<sub>e smoo</sub>th <sub>case an</sub>d <sub>presen</sub>t <sub>numer</sub>i<sub>ca</sub>l ex<sub>p</sub>er<sup>i</sup>ments.

## 1 Introduction

We consider stochastic nonsmooth nonconvex com<sub>p</sub>osite o<sub>p</sub>timization<sub>,</sub>

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } \{ \Phi ( x ) : = f ( x ) + h ( x ) \} , \ f ( x ) : = \mathbb { E } _ { \xi } [ F ( x , \xi ) ] ,\tag{1}
$$

where ξ is a random vector, $f \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ i<sub>s</sub> l<sub>oca</sub>ll<sub>y</sub> Li<sub>psc</sub>hit<sub>z an</sub>d <sub>may</sub> b<sub>e nonconvex an</sub>d <sub>nonsmoo</sub>th<sub>, an</sub>d $h \colon \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ is a proper closed convex function. We study both first-order access to f through <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c gra</sub>di<sub>en</sub>t<sub>s an</sub>d <sub>zero</sub>th<sub>-or</sub>d<sub>er access</sub> th<sub>roug</sub>h <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> f<sub>unc</sub>ti<sub>on va</sub>l<sub>ues.</sub>

Composite optimization includes important problem classes such as ℓ<sub>1</sub>-regularized optimization, nuclearnorm-re<sub>g</sub>ularized o<sub>p</sub>timization, and constrained o<sub>p</sub>timization (Tibshirani, 1996; Mazumder et al., 2010; Parikh and Boyd, 2014). Most existing convergence analyses of composite optimization assume that f is smooth, that is, ∇f is Lipschitz continuous (Nesterov, 2013; Ghadimi et al., 2016). In this smooth setting, stochastic first-order and zeroth-order methods find an ε-stationary point of the objective Φ with $O ( \varepsilon ^ { - 4 } )$ <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>g</sub>radient <sub>q</sub>ueries and $O ( d \varepsilon ^ { - 4 } )$ function-value queries, respectively (Ghadimi et al., 2016), where ε is the stationarit<sub>y</sub> tolerance. In man<sub>y</sub> a<sub>pp</sub>lications<sub>,</sub> however<sub>,</sub> $f$ is nonsmooth<sub>,</sub> as in su<sub>pp</sub>ort vector machines with the hin<sub>g</sub>e loss (Cortes and Va<sub>p</sub>nik, 1995; Shalev-Shwartz et al., 2011) and the re<sub>g</sub>ularized trainin<sub>g</sub> of neural networks with ReLU activations (Krizhevsk<sub>y</sub> et al., 2012; Wan<sub>g</sub> et al., 2022).

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> O<sub>rac</sub>l<sub>e</sub> <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> f<sub>or</sub> fi<sub>n</sub>di<sub>ng</sub> <sub>a</sub> $( \gamma , \delta , \varepsilon )$ -PGSP (see Definition 3.3), which reduces to $( \delta , \varepsilon )$ <sub>-</sub>G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>i<sub>n</sub> stationarit<sub>y</sub> when $h = 0$ . The bounds show the dependence on d, δ, and $\varepsilon$ f<sub>or</sub> $\delta , \varepsilon \in ( 0 , 1 ]$
<table><tr><td>Reference</td><td>Problem class</td><td>Oracle</td><td>Complexity</td></tr><tr><td>Cutkosky et al. (2023)</td><td> $h = 0$ </td><td>First-order</td><td> $O ( \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ </td></tr><tr><td>Kornowski and Shamir (2024)</td><td> $h = 0$ </td><td>Zeroth-order</td><td> ${ \cal O } ( d \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ </td></tr><tr><td>Chen et al. (2025, Theorem 2)</td><td>Composite</td><td>Zeroth-order (minibatch)</td><td> $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 4 } )$ </td></tr><tr><td>Chen et al. (2025, Theorem 3)</td><td>Composite</td><td>Zeroth-order (variance reduction)</td><td> ${ \cal O } \dot { ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 3 } ) }$ </td></tr><tr><td>Pougkakiotis and Kalogerias (2026)</td><td>Composite</td><td>Zeroth-order</td><td> $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 4 } )$ </td></tr><tr><td>This work (Theorem 4.4)</td><td>Composite</td><td>First-order</td><td> $O ( \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ </td></tr><tr><td>This work (Theorem 4.6)</td><td>Composite</td><td>Zeroth-order</td><td> $O ( d \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ </td></tr></table>

With zeroth-order access, Liu et al. (2024b) stud<sub>y</sub> nonsmooth nonconvex o<sub>p</sub>timization over com<sub>p</sub>act convex sets, and Chen et al. (2025) extend this settin<sub>g</sub> to com<sub>p</sub>osite objectives. Their al<sub>g</sub>orithms require $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 4 } )$ function-value queries with minibatch estimation to find <sub>p</sub>roximal Goldstein stationar<sub>y p</sub>oints (PGSPs; see Definition 3.3), a Goldstein-type stationarity condition for composite objectives, where $\delta$ i<sub>s</sub> th<sub>e</sub> G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>i<sub>n</sub> radius and d is the dimension. Incorporating variance reduction further improves the bound to $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ T<sub>o ou</sub>r kn<sub>ow</sub>l<sub>e</sub>d<sub>ge,</sub> n<sub>o s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fir<sub>s</sub>t-<sub>o</sub>rd<sub>e</sub>r <sub>co</sub>m<sub>p</sub>l<sub>e</sub>xit<sub>y</sub> b<sub>ou</sub>nd f<sub>o</sub>r PGSP<sub>s</sub> i<sub>s</sub> kn<sub>ow</sub>n f<sub>o</sub>r <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>l Li<sub>psc</sub>hitz nonconvex $f$ and proper closed convex h.

In the noncom<sub>p</sub>osite settin<sub>g</sub> where $h = 0 ,$ the seminal framework of online-to-nonconvex conversion (O2NC), <sub>w</sub>hi<sub>c</sub>h <sub>c</sub>h<sub>ooses up</sub>d<sub>a</sub>t<sub>e</sub> di<sub>rec</sub>ti<sub>ons</sub> b<sub>y on</sub>li<sub>ne</sub> l<sub>earn</sub>i<sub>ng, a</sub>ll<sub>ows us</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e op</sub>ti<sub>ma</sub>l $O ( \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ stochastic <sub>g</sub>radient quer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> (Cutkosk<sub>y</sub> et al., 2023) and $O ( d \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ function-value quer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> (Kornowski and Shamir, 2024). Both rates are attained without variance reduction, and the zeroth-order bound is linear in d. This raises the following question:

Can simplefirst-order and zeroth-order methods attain these oracle complexitiesfor nonsmooth nonconvex composite optimization?

## 1.1 Contributions of this paper

We answer this <sub>q</sub>uestion afirmativel<sub>y</sub> with a new conversion from online re<sub>g</sub>ret to <sub>p</sub>roximal Goldstein <sub>s</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>rit<sub>y.</sub> Th<sub>e</sub> r<sub>esu</sub>ltin<sub>g</sub> m<sub>e</sub>th<sub>o</sub>d<sub>,</sub> C<sub>o</sub>mp<sub>os</sub>ite O2NC<sub>,</sub> find<sub>s a</sub> PGSP <sub>w</sub>ith $O ( \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ stochastic <sub>g</sub>radient <sub>q</sub>uer<sup>i</sup>es or $O ( d \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ function-value <sub>q</sub>ueries.

Both versions attain these rates without variance reduction (Theorems 4.4 and 4.6). To our knowled<sub>g</sub>e, our <sub>zero</sub>th<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d i<sub>s</sub> th<sub>e</sub> fi<sub>rs</sub>t t<sub>o</sub> fi<sub>n</sub>d PGSP<sub>s</sub> f<sub>or genera</sub>l <sub>nonsmoo</sub>th <sub>nonconvex compos</sub>it<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on w</sub>ith function-value <sub>q</sub>uer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> linear in $d ,$ im<sub>p</sub>rovin<sub>g</sub> the $d ^ { 3 / 2 }$ de<sub>p</sub>endence of Chen et al. (2025). For $h = 0$ the first-order rate matches the joint lower bound in δ and ε (Cutkosk et al., 2023), while the zeroth-order rate is o<sub>p</sub>timal in each of $d , \delta ,$ <sub>an</sub>d $\varepsilon$ individuall<sub>y</sub> (Kornowski and Shamir, 2024).<sup>1</sup> Since PGSP reduces to

Goldstein stationarit<sub>y w</sub>hen $h = 0$ <sub>,</sub> th<sub>ese</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s a</sub>l<sub>so app</sub>l<sub>y</sub> t<sub>o our compos</sub>it<sub>e pro</sub>bl<sub>em c</sub>l<sub>ass, an</sub>d <sub>our</sub> <sub>ra</sub>t<sub>es are op</sub>ti<sub>ma</sub>l i<sub>n</sub> th<sub>e same sense.</sub> T<sub>a</sub>bl<sub>e</sub> 1 <sub>summar</sub>i<sub>zes</sub> th<sub>e resu</sub>lt<sub>s.</sub>

O<sub>ur me</sub>th<sub>o</sub>d <sub>mo</sub>difi<sub>es</sub> th<sub>e</sub> l<sub>osses g</sub>i<sub>ven</sub> t<sub>o</sub> th<sub>e on</sub>li<sub>ne</sub> l<sub>earner</sub> i<sub>n</sub> O2NC<sub>.</sub> I<sub>n</sub> O2NC<sub>,</sub> th<sub>e on</sub>li<sub>ne</sub> l<sub>earner se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> di<sub>rec</sub>ti<sub>ons</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> li<sub>near</sub> l<sub>osses</sub> $u \mapsto \langle g _ { t } , u \rangle$ <sub>, w</sub>h<sub>ere</sub> $g _ { t }$ i<sub>s a</sub>n <sub>u</sub>nbi<sub>ase</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c g</sub>r<sub>a</sub>di<sub>e</sub>nt <sub>o</sub>f $f$ at a quer<sub>y p</sub>oint (Cutkosk<sub>y</sub> et al., 2023). This framework does not a<sub>pp</sub>l<sub>y</sub> to $\Phi = f + h$ directl , because Φ need not b<sub>e</sub> l<sub>oca</sub>ll<sub>y</sub> Li<sub>psc</sub>hit<sub>z w</sub>h<sub>en</sub> $h$ takes the value +∞. Moreover, linearizin<sub>g</sub> $h$ i<sub>g</sub>n<sub>o</sub>r<sub>es</sub> th<sub>e s</sub>tr<sub>uc</sub>t<sub>u</sub>r<sub>e o</sub>f $h ,$ <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> k<sub>nown exac</sub>tl<sub>y.</sub>

T<sub>o</sub> d<sub>ea</sub>l <sub>w</sub>ith thi<sub>s, we</sub> i<sub>ve</sub> th<sub>e</sub> l<sub>earner</sub> th<sub>e</sub> l<sub>osses</sub> $\ell _ { t } ( u ) = \langle g _ { t } , u \rangle + h ( x + u ) - h ( x )$ <sub>, w</sub>h<sub>ere</sub> $x \in$ dom h is th<sub>e cu</sub>rr<sub>e</sub>nt <sub>po</sub>int<sub>.</sub> Th<sub>e</sub> t<sub>e</sub>rm $h ( x + u ) - h ( x )$ l<sub>e</sub>t<sub>s</sub> th<sub>e</sub> l<sub>earner accoun</sub>t f<sub>or</sub> th<sub>e regu</sub>l<sub>ar</sub>i<sub>zer or cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>w</sub>h<sub>en</sub> choosin<sub>g</sub> a direction. For these losses, com<sub>p</sub>osite objective mirror descent (COMID; Duchi et al. 2010) attains th<sub>e sa</sub>m<sub>e</sub> r<sub>eg</sub>r<sub>e</sub>t b<sub>ou</sub>nd <sub>as</sub> f<sub>o</sub>r lin<sub>ea</sub>r l<sub>osses, a</sub>nd <sub>ou</sub>r <sub>co</sub>n<sub>ve</sub>r<sub>s</sub>i<sub>o</sub>n t<sub>u</sub>rn<sub>s</sub> thi<sub>s</sub> r<sub>eg</sub>r<sub>e</sub>t b<sub>ou</sub>nd int<sub>o a</sub> PGSP <sub>gua</sub>r<sub>a</sub>nt<sub>ee</sub> (Lemma 4.2). For zeroth-order access, we use two-<sub>p</sub>oint <sub>g</sub>radient estimates of a smoothed version of $f$ <sub>an</sub>d tr<sub>a</sub>n<sub>s</sub>f<sub>e</sub>r th<sub>e</sub> r<sub>esu</sub>ltin<sub>g gua</sub>r<sub>a</sub>nt<sub>ee</sub> t<sub>o</sub> $f + h$ usin<sub>g</sub> the smoothin<sub>g</sub> result of Kornowski and Shamir (2024).

O<sub>u</sub>r <sub>gua</sub>r<sub>a</sub>nt<sub>ee</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>o</sub>r <sub>eve</sub>r<sub>y</sub> <sub>p</sub>r<sub>o</sub>xim<sub>a</sub>l <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r $\gamma > 0$ <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y, an</sub>d <sub>ne</sub>ith<sub>er</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m nor</sub> it<sub>s</sub> <sub>o</sub>r<sub>ac</sub>l<sub>e co</sub>m<sub>p</sub>l<sub>e</sub>xit<sub>y</sub> d<sub>epe</sub>nd<sub>s o</sub>n $\gamma .$ <sub>.</sub> E<sub>x</sub>i<sub>s</sub>ti<sub>ng guaran</sub>t<sub>ees</sub> h<sub>o</sub>ld f<sub>or par</sub>ti<sub>cu</sub>l<sub>ar va</sub>l<sub>ues o</sub>f $\gamma ,$ <sub>w</sub>hi<sub>c</sub>h i<sub>s c</sub>h<sub>osen as an</sub> up<sup>d</sup>ate step size $( e . g .$ ., Ghadimi et al. 2016; Reddi et al. 2016; Pham et al. 2020; Chen et al. 2025). In contrast, <sub>ou</sub>r <sub>a</sub>l<sub>go</sub>rithm r<sub>e</sub>t<sub>u</sub>rn<sub>s</sub> $x + u ^ { * }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $u ^ { * }$ <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> th<sub>e</sub> <sub>average</sub>d <sub>on</sub>li<sub>ne</sub> l<sub>oss</sub> <sub>over</sub> th<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> di<sub>rec</sub>ti<sub>ons</sub> <sub>a</sub>t th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>anc</sub>h<sub>or</sub> $x ,$ <sub>a</sub>nd thi<sub>s po</sub>int <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> PGSP <sub>gua</sub>r<sub>a</sub>nt<sub>ee</sub> f<sub>o</sub>r <sub>eve</sub>r<sub>y</sub> $\gamma > 0$ (Lemma 4.2). Thus $\gamma$ i<sub>s use</sub>d onl<sub>y</sub> to state the stationarit<sub>y</sub> condition.

Wh<sub>en</sub> $f$ is smooth, we show that Composite O2NC with an optimistic online learner finds an ε-stationary <sub>p</sub>o<sup>i</sup>nt us<sup>i</sup>n<sub>g</sub> $O ( \varepsilon ^ { - 4 } )$ stochastic <sub>g</sub>radient <sub>q</sub>ueries or $O ( d \varepsilon ^ { - 4 } )$ function-value queries (Theorems 5.3 and 5.4). Th<sub>ese ra</sub>t<sub>es ma</sub>t<sub>c</sub>h <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> b<sub>oun</sub>d<sub>s un</sub>d<sub>er</sub> th<sub>e s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c orac</sub>l<sub>e, an</sub>d <sub>w</sub>ith d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c gra</sub>di<sub>en</sub>t<sub>s our resu</sub>lt <sub>recovers</sub> th<sub>e</sub> $O ( \varepsilon ^ { - 2 } )$ <sub>p</sub>roximal-<sub>g</sub>radient rate (Ghadimi et al., 2016) (Table 2 in A<sub>pp</sub>endix C.1).

N<sub>u</sub>m<sub>e</sub>ri<sub>ca</sub>l <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s o</sub>n n<sub>o</sub>n<sub>s</sub>m<sub>oo</sub>th <sub>co</sub>m<sub>pos</sub>it<sub>e p</sub>r<sub>o</sub>bl<sub>e</sub>m<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t C<sub>o</sub>mp<sub>os</sub>ite O2NC <sub>a</sub>tt<sub>a</sub>in<sub>s s</sub>m<sub>a</sub>ll<sub>e</sub>r <sub>va</sub>l<sub>ues</sub> of both the stationarit measure and the objective than the zeroth-order methods of Chen et al. (2025) for the same number of function-value queries (Section 6).

## 2 Related work

Composite optimization. Proximal-gradient methods are the standard approach to composite optimization (Nesterov, 2013; Parikh and Bo<sub>y</sub>d, 2014). Accelerated methods include FISTA for convex objectives (Beck and Teboulle, 2009) and extensions to nonconvex objectives (Li and Lin, 2015; Ghadimi and Lan, 2016; Li et al., 2017). For smooth nonconvex $f ,$ minibatch <sub>p</sub>roximal-<sub>g</sub>radient methods attain $O ( \varepsilon ^ { - 4 } )$ stochastic <sub>g</sub>radient quer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> (Ghadimi et al., 2016). Other conver<sub>g</sub>ence anal<sub>y</sub>ses rel<sub>y</sub> on the Kurd<sub>y</sub>ka–Łojasiewicz <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> (Attouch et al., 2013; Bolte et al., 2014).

With <sub>a</sub>dditi<sub>ona</sub>l <sub>smoo</sub>th<sub>ness assump</sub>ti<sub>ons on</sub> th<sub>e componen</sub>t <sub>or samp</sub>l<sub>e gra</sub>di<sub>en</sub>t<sub>s, var</sub>i<sub>ance-re</sub>d<sub>uce</sub>d <sub>me</sub>th<sub>o</sub>d<sub>s</sub> further im<sub>p</sub>ro<sub>v</sub>e stochastic <sub>g</sub>radient <sub>q</sub>uer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub>. Re<sub>p</sub>resentati<sub>v</sub>e methods include ProxSVRG and its <sub>v</sub>ariants (Xiao and Zhan<sub>g</sub>, 2014; Nitanda, 2014; Reddi et al., 2016; Li and Li, 2018), as well as ProxSAGA (Reddi et al., 2016). For stochastic ex<sub>p</sub>ectation <sub>p</sub>roblems, ProxS<sub>p</sub>iderBoost (Wan<sub>g</sub> et al., 2019), ProxSARAH (Pham et al., 2020), and h<sub>y</sub>brid SARAH-SGD methods (Tran-Dinh et al., 2022) attain $O ( \varepsilon ^ { - 3 } )$ stochastic <sub>g</sub>radient <sub>query comp</sub>l<sub>ex</sub>it<sub>y un</sub>d<sub>er</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>var</sub>i<sub>ance an</sub>d <sub>a</sub>dditi<sub>ona</sub>l <sub>smoo</sub>th<sub>ness con</sub>diti<sub>ons on</sub> th<sub>e samp</sub>l<sub>e gra</sub>di<sub>en</sub>t<sub>s.</sub>

E<sub>x</sub>t<sub>ens</sub>i<sub>ons</sub> t<sub>o</sub> <sub>nonsmoo</sub>th $f$ have also been widel<sub>y</sub> studied. Under weak convexit<sub>y</sub>, stochastic first-order (Davis and Grimmer, 2019; Davis and Drusv<sub>y</sub>atski<sub>y</sub>, 2019) and zeroth-order (Pou<sub>g</sub>kakiotis and Kalo<sub>g</sub>erias, 2023) met<sup>h</sup>ods provide Moreau-enve<sup>l</sup>ope stationarity guarantees. For <sup>f</sup>unctiona<sup>l</sup> constraints wit<sup>h</sup> Lipsc<sup>h</sup>itz objective and constraint functions, Jia and Grimmer (2025) establish Fritz–John and KKT <sub>g</sub>uarantees under weak convexit<sub>y</sub>, while Grimmer and Jia (2025) obtain Goldstein-t<sub>yp</sub>e counter<sub>p</sub>arts without weak convexit<sub>y</sub> usin<sub>g</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er orac</sub>l<sub>es.</sub> Th<sub>e</sub> l<sub>a</sub>tt<sub>er se</sub>tti<sub>ng over</sub>l<sub>aps w</sub>ith <sub>ours</sub> f<sub>or convex cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s,</sub> b<sub>u</sub>t <sub>we a</sub>l<sub>so</sub> h<sub>an</sub>dl<sub>e</sub> composite objectives wit<sup>h</sup> convex regu<sup>l</sup>arizers using more genera<sup>l</sup> stoc<sup>h</sup>astic <sup>fi</sup>rst-order orac<sup>l</sup>es.

F<sub>or smoo</sub>th fi<sub>n</sub>it<sub>e-sum compos</sub>it<sub>e pro</sub>bl<sub>ems, zero</sub>th<sub>-or</sub>d<sub>er var</sub>i<sub>an</sub>t<sub>s o</sub>f P<sub>rox</sub>SVRG<sub>,</sub> P<sub>rox</sub>SAGA<sub>, an</sub>d SPIDER re<sub>p</sub>lace <sub>g</sub>radients in the corres<sub>p</sub>ondin<sub>g</sub> first-order methods with function-value estimates (Huan<sub>g</sub> et al., 2019; Kazemi and Wan<sub>g</sub>, 2024). Be<sub>y</sub>ond the weakl<sub>y</sub> convex settin<sub>g</sub> discussed above, Liu et al. (2024b) stud<sub>y</sub> nonsmooth objectives over com<sub>p</sub>act convex sets, and Chen et al. (2025) extend this settin<sub>g</sub> to com<sub>p</sub>osite objectives. T<sup>h</sup>eir minibatc<sup>h</sup> and variance-reduced met<sup>h</sup>ods <sup>fi</sup>nd proxima<sup>l</sup> Go<sup>l</sup>dstein stationary points using ${ \cal O } ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 4 } )$ <sub>an</sub>d $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ function-value queries, res<sub>p</sub>ectivel<sub>y</sub>. Pou<sub>g</sub>kakiotis and Kalo<sub>g</sub>erias (2026) a<sup>l</sup>so study nonsmoot<sup>h</sup> nonconvex composite objectives using inexact <sup>f</sup>unction-va<sup>l</sup>ue queries. T<sup>h</sup>eir Moreauenvelo<sub>p</sub>e <sub>g</sub>uarantee can be converted into a <sub>p</sub>roximal Goldstein stationarit<sub>y</sub> <sub>g</sub>uarantee with $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 4 } )$ function-value quer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> (see A<sub>pp</sub>endix D). For this <sub>p</sub>roximal Goldstein criterion, our zeroth-order <sub>me</sub>th<sub>o</sub>d i<sub>mproves</sub> <sub>a</sub>ll <sub>o</sub>f th<sub>ese</sub> b<sub>oun</sub>d<sub>s</sub> t<sub>o</sub> $O ( d \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ , which is linear in d, without variance reduction.

Online-to-nonconvex conversion. The online-to-nonconvex conversion (O2NC) framework was introduced b<sub>y</sub> Cutkosk<sub>y</sub> et al. (2023) for stochastic nonsmooth nonconvex o<sub>p</sub>timization. It converts online re<sub>g</sub>ret bounds into Goldstein stationarit<sub>y g</sub>uarantees with o<sub>p</sub>timal stochastic <sub>g</sub>radient <sub>q</sub>uer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub>. Kornowski and Shamir (2024) combine the O2NC conversion with randomized smoothin<sub>g</sub> to obtain a bound linear in the dimension in the zeroth-order settin<sub>g</sub>.

Double o<sub>p</sub>timism further im<sub>p</sub>roves the <sub>g</sub>uarantees when both the <sub>g</sub>radient and Hessian are Li<sub>p</sub>schitz (Patitucci et al., 2026). Random scalin<sub>g</sub> and discounted re<sub>g</sub>ret connect O2NC to momentum (Zhan<sub>g</sub> and Cutkosk<sub>y</sub>, 2024) and Adam with model ex<sub>p</sub>onential movin<sub>g</sub> avera<sub>g</sub>es (Ahn and Cutkosk<sub>y</sub>, 2024). The broader online-learnin<sub>g</sub> <sub>p</sub>ers<sub>p</sub>ective also ex<sub>p</sub>lains Adam throu<sub>g</sub>h online learnin<sub>g</sub> of u<sub>p</sub>dates (Ahn et al., 2024) and <sub>y</sub>ields <sub>g</sub>uarantees for schedule-free SGD (Ahn et al., 2025).

For deterministic smooth objectives, Cutkosk<sub>y</sub> et al. (2023) recover the o<sub>p</sub>timal <sub>g</sub>radient quer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> usin<sub>g</sub> o<sub>p</sub>timistic online learners. Usin<sub>g</sub> momentum within discounted O2NC, Li and Tsuchi<sub>y</sub>a (2026) also attain the o<sub>p</sub>timal $O ( \varepsilon ^ { - 2 } )$ gradient query comp<sup>l</sup>exity <sup>f</sup>or deterministic smoot<sup>h</sup> objectives wit<sup>h</sup>out an optimistic <sub>on</sub>li<sub>ne</sub> l<sub>earner, us</sub>i<sub>ng a regre</sub>t d<sub>ecompos</sub>iti<sub>on</sub> b<sub>ase</sub>d <sub>on momen</sub>t<sub>um.</sub> Oth<sub>er ex</sub>t<sub>ens</sub>i<sub>ons</sub> h<sub>an</sub>dl<sub>e</sub> h<sub>eavy-</sub>t<sub>a</sub>il<sub>e</sub>d <sub>g</sub>radient noise (Liu et al., 2024a; Liu, 2026) and remove auxiliar<sub>y</sub> random inter<sub>p</sub>olation or scalin<sub>g</sub> under weak convexit<sub>y</sub> (Ji and Yuan, 2026). O2NC has also been a<sub>pp</sub>lied to nonsmooth objectives on Riemannian manifolds (Sahino<sub>g</sub>lu et al., 2025), decentralized stochastic o<sub>p</sub>timization (Sahino<sub>g</sub>lu and Shahram<sub>p</sub>our, 2024), and linearl constrained bilevel o timization (Kornowski et al., 2024).

We extend O2NC using composite online losses that retain h without linearization. Regret bounds from com<sub>p</sub>osite objective mirror descent (Duchi et al., 2010) then <sub>y</sub>ield <sub>p</sub>roximal Goldstein stationarit<sub>y</sub>, even when h takes the value +∞.

## 3 Preliminaries

T<sup>h</sup>is section states t<sup>h</sup>e assumptions on t<sup>h</sup>e objective and t<sup>h</sup>e stoc<sup>h</sup>astic orac<sup>l</sup>e and introduces t<sup>h</sup>e Go<sup>l</sup>dstein-type stationarit<sub>y</sub> measure used in our anal<sub>y</sub>sis.

Thr<sub>oug</sub>h<sub>ou</sub>t<sub>,</sub> <sub>we</sub> <sub>w</sub>rit<sub>e</sub> $\lVert \cdot \rVert$ <sub>an</sub>d $\langle \cdot , \cdot \rangle$ f<sub>or</sub> th<sub>e</sub> E<sub>uc</sub>lid<sub>ean</sub> <sub>norm</sub> <sub>an</sub>d i<sub>nner</sub> <sub>pro</sub>d<sub>uc</sub>t <sub>on</sub> $\mathbb { R } ^ { d } , \mathbb { B } ( x , r ) : = \{ y \in$ $\mathbb { R } ^ { d } \colon \| y - x \| \leq r \}$ f<sub>or</sub> th<sub>e c</sub>l<sub>ose</sub>d E<sub>uc</sub>lid<sub>ean</sub> b<sub>a</sub>ll<sub>,</sub> $\mathbb { S } ^ { d - 1 }$ for the unit sphere, Unif(S) for the uniform distribution on a set $S ,$ <sub>an</sub>d $[ n ] : = \{ 1 , \dots , n \}$ for a positive integer n. For a convex function $h ,$ , we write dom h and $\partial h$ f<sub>or</sub> it<sub>s e</sub>f<sub>ec</sub>ti<sub>ve</sub> d<sub>oma</sub>i<sub>n an</sub>d <sub>su</sub>bdif<sub>eren</sub>ti<sub>a</sub>l<sub>.</sub>

## 3.1 Problem setting

We consider the stochastic nonsmooth nonconvex com<sub>p</sub>osite o<sub>p</sub>timization <sub>p</sub>roblem formulated in (1). We <sub>assume</sub> th<sub>a</sub>t $\Delta : = \Phi ( x _ { 1 } ) - \mathrm { i n f } _ { x } \Phi ( x ) < \infty$ <sub>, w</sub>h<sub>ere</sub> $x _ { 1 }$ is the initial <sub>p</sub>oint. We stud<sub>y</sub> t<sub>w</sub>o oracle models for $f .$ Under first-order access<sub>,</sub> a <sub>q</sub>uer<sub>y</sub> at a <sub>p</sub>oint $\boldsymbol { y } \in \mathbb { R } ^ { d }$ r<sub>e</sub>t<sub>u</sub>rn<sub>s a s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c g</sub>r<sub>a</sub>di<sub>e</sub>nt <sub>o</sub>f $f$ at $y .$ <sub>.</sub> U<sub>n</sub>d<sub>er zero</sub>th<sub>-or</sub>d<sub>er</sub> <sup>access</sup>, <sup>a</sup> q<sup>uer</sup>y <sup>at</sup> $y$ <sub>re</sub>t<sub>urns</sub> th<sub>e</sub> f<sub>unc</sub>ti<sub>on va</sub>l<sub>ue</sub> $F ( y , \xi )$ f<sub>or a samp</sub>l<sub>e</sub> $\xi ,$ <sub>w</sub>h<sub>ere</sub> th<sub>e same samp</sub>l<sub>e can</sub> b<sub>e use</sub>d <sub>a</sub>t two <sub>p</sub>oints as in the two-<sub>p</sub>oint feedback model (A<sub>g</sub>arwal et al., 2010; Ghadimi and Lan, 2013; Duchi et al., 2015; Shamir, 2017; Kornowski and Shamir, 2024). We impose the following assumptions on the objective and the first-order oracle, followin<sub>g</sub> Cutkosk<sub>y</sub> et al. (2023).

Assumption 3.1. The function $f$ i<sub>s</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e.</sub> F<sub>or some</sub> $\sigma \geq 0$ <sub>an</sub>d $G > 0$ , a <sub>q</sub>uer<sub>y</sub> to t<sup>h</sup>e orac<sup>l</sup>e at <sup>an</sup>y p<sup>oint</sup> $\boldsymbol { y } \in \mathbb { R } ^ { d }$ is inde<sub>p</sub>endent of the <sub>p</sub>recedin<sub>g</sub> histor<sub>y</sub> and returns a stochastic <sub>g</sub>radient $g$ satisf<sub>y</sub>in<sub>g</sub> 0 $\mathrm { i } ) \mathbb { E } [ g \mid y ] = \nabla f ( y ) , ( \mathrm { i i } ) \mathbb { E } [ \| g - \nabla f ( y ) \| ^ { 2 } \mid y ] \leq \sigma ^ { 2 }$ <sub>,</sub> <sub>an</sub>d $( \operatorname { i i i } ) \mathbb { E } [ \| g \| ^ { 2 } \mid y ] \leq G ^ { 2 }$

Sin<sub>ce</sub> $f$ is locally Lipschitz, its diferentiability implies that f is well-behaved (Cutkosky et al., 2023, Proposition 2), that is, $\begin{array} { r } { f ( y ) - f ( x ) = \int _ { 0 } ^ { 1 } \langle \nabla f \big ( x + t ( y - x ) \big ) } \end{array}$ , y − x⟩ dt for all $x , y \in \mathbb { R } ^ { d }$

Wh<sub>en</sub> $f$ i<sub>s</sub> n<sub>o</sub>t dif<sub>e</sub>r<sub>e</sub>nti<sub>a</sub>bl<sub>e, we</sub> r<sub>ep</sub>l<sub>ace</sub> it b<sub>y</sub> $f _ { \rho } ( x ) : = \mathbb { E } _ { z \sim \mathsf { U n i f } ( \mathbb { B } ( 0 , 1 ) ) } [ f ( x + \rho z ) ]$ with a smoothin<sub>g</sub> radius $\rho >$ 0 and carry out the analysis on $f _ { \rho } + h$ <sub>.</sub> Th<sub>en</sub> $f _ { \rho }$ i<sub>s</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e w</sub>ith $\nabla f _ { \rho } ( x ) = \mathbb { E } _ { z \sim \mathsf { U n i f } ( \mathbb { B } ( 0 , 1 ) ) } [ \nabla f ( x + \rho z ) ]$ <sub>w</sub>h<sub>ere</sub> $f$ i<sub>s</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e a</sub>t $x + \rho z$ <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y, an</sub>d <sub>a s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c gra</sub>di<sub>en</sub>t <sub>o</sub>f $f$ at $x + \rho z$ is a stochastic <sub>g</sub>r<sub>a</sub>di<sub>e</sub>nt <sub>o</sub>f $f _ { \rho }$ at x with the same second-moment bound (Cutkosky et al., 2023, Proposition 2). Moreover, if $f$ is L-Lipschitz, it holds that $| f _ { \rho } ( x ) - f ( x ) | \leq \rho L$ <sup>f</sup>or ever<sub>y</sub> $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . Stationarit<sub>y g</sub>uarantees for $f _ { \rho } + h$ are trans<sup>f</sup>erred to t<sup>h</sup>e objective $f + h$ b<sub>y</sub> L<sub>emma</sub> 3<sub>.</sub>4<sub>,</sub> <sub>w</sub>ith th<sub>e</sub> G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>i<sub>n</sub> <sub>ra</sub>di<sub>us</sub> <sub>en</sub>l<sub>arge</sub>d b<sub>y</sub> $\rho .$

F<sub>or zero</sub>th<sub>-or</sub>d<sub>er access, we rep</sub>l<sub>ace</sub> A<sub>ssump</sub>ti<sub>on</sub> 3<sub>.</sub>1 <sub>w</sub>ith th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng assump</sub>ti<sub>on on</sub> th<sub>e samp</sub>l<sub>e</sub> f<sub>unc</sub>ti<sub>ons,</sub> followin<sub>g</sub> Kornowski and Shamir (2024).

Assumption 3.2. For any ξ, the function $F ( \cdot , \xi )$ is $L ( \xi ) { \mathrm { - I } }$ i<sub>p</sub>schitz<sub>,</sub> where $\mathbb { E } _ { \xi } \big [ L ( \xi ) ^ { 2 } \big ] \leq L _ { 0 } ^ { 2 } .$

## 3.2 Goldstein-type stationarity

H<sub>ere, we</sub> fi<sub>rs</sub>t d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>i<sub>n su</sub>bdif<sub>eren</sub>ti<sub>a</sub>l <sub>an</sub>d th<sub>en use</sub> it t<sub>o</sub> d<sub>e</sub>fi<sub>ne prox</sub>i<sub>ma</sub>l G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>i<sub>n s</sub>t<sub>a</sub>ti<sub>onary</sub> points <sup>f</sup>or t<sup>h</sup>e composite objective $f + h$ , followin<sub>g</sub> Chen et al. (2025). For a locall<sub>y</sub> Li<sub>p</sub>schitz $f ,$ th<sub>e</sub> Cl<sub>ar</sub>k<sub>e</sub> <sub>g</sub>eneralized directional derivative is $\begin{array} { r } { f ^ { \circ } ( x ; v ) : = \operatorname* { l i m } \operatorname* { s u p } _ { y \to x , t \downarrow 0 } \frac { f ( y + t v ) - f ( y ) } { t } } \end{array}$ <sub>an</sub>d th<sub>e</sub> Cl<sub>ar</sub>k<sub>e su</sub>bdif<sub>eren</sub>ti<sub>a</sub>l i<sub>s</sub> $\partial f ( \boldsymbol { x } ) : = \big \{ g \in \mathbb { R } ^ { d } \colon \langle g , v \rangle \leq f ^ { \circ } ( \boldsymbol { x } ; v )$ f<sub>or a</sub>ll $v \in \mathbb { R } ^ { d } \}$ (Clarke, 1990). Then, for $\delta \geq 0$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>i<sub>n</sub> δ-subdiferential (Goldstein, 1977) by

$$
\partial _ { \delta } f ( x ) : = \mathrm { c o n v } \Bigg \{ \bigcup _ { y \in \mathbb { B } ( x , \delta ) } \partial f ( y ) \Bigg \} .
$$

We next define the <sub>p</sub>roximal o<sub>p</sub>erator and the <sub>p</sub>roximal-<sub>g</sub>radient ma<sub>pp</sub>in<sub>g</sub> (Nesterov, 2013; Parikh and Bo<sub>y</sub>d, 2014; Ghadimi et al., 2016). For $\gamma > 0 , z \in \mathbb { R } ^ { d }$ , x ∈ dom h, and $\boldsymbol { g } \in \mathbb { R } ^ { d }$ <sub>,</sub> th<sub>e p</sub>r<sub>o</sub>xim<sub>a</sub>l <sub>ope</sub>r<sub>a</sub>t<sub>o</sub>r <sub>o</sub>f $h ,$ $\mathrm { p r o x } _ { \gamma h } ( z )$ <sub>,</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub>

$$
\operatorname { p r o x } _ { \gamma h } ( z ) : = \arg \operatorname* { m i n } _ { y \in \mathbb { R } ^ { d } } \bigg \{ h ( y ) + \frac { 1 } { 2 \gamma } \| y - z \| ^ { 2 } \bigg \} ,\tag{2}
$$

and the <sub>p</sub>roximal-<sub>g</sub>radient ma<sub>pp</sub>in<sub>g</sub> $G _ { \gamma h } ( x , g )$ as

$$
G _ { \gamma h } ( x , g ) : = \frac { 1 } { \gamma } \big ( x - \mathrm { p r o x } _ { \gamma h } ( x - \gamma g ) \big ) .\tag{3}
$$

Note that the minimizer in (2) is unique since $\begin{array} { r } { h ( y ) + \frac { 1 } { 2 \gamma } \lVert y - z \rVert ^ { 2 } } \end{array}$ is strong<sup>l</sup>y convex.

We use t<sup>h</sup>e <sup>f</sup>o<sup>ll</sup>owing extension o<sup>f</sup> Go<sup>l</sup>dstein stationarity to composite objectives $f + h$ (Chen et al., 2025).

Definition 3.3 (PGSP, Chen et al. 2025). A point x is said to be a $( \gamma , \delta , \varepsilon )$ -<sub>p</sub>roximal Goldstein stationar<sub>y</sub> <sub>p</sub>oint (PGSP) if

$$
\operatorname* { m i n } _ { g \in \partial _ { \delta } f ( x ) } \| G _ { \gamma h } ( x , g ) \| \leq \varepsilon .\tag{4}
$$

Thi<sub>s re</sub>l<sub>a</sub>t<sub>es</sub> t<sub>o o</sub>th<sub>er s</sub>t<sub>a</sub>ti<sub>onar</sub>it<sub>y con</sub>diti<sub>ons as</sub> f<sub>o</sub>ll<sub>ows:</sub>

(i) In (noncom<sub>p</sub>osite) nonsmooth nonconvex o<sub>p</sub>timization, $h = 0$ <sub>g</sub><sup>i</sup>ves $G _ { \gamma h } ( x , g ) = g .$ . Thus, (4) is the $( \delta , \varepsilon )$ -Goldstein stationarity condition dist $( 0 , \partial _ { \delta } f ( x ) ) \leq \varepsilon .$ , independently of γ (Zhang et al., 2020).

(ii) In smooth nonconvex o<sub>p</sub>timization, where $\nabla f$ is $L _ { \mathrm { 1 } ^ { - } } \mathrm { I }$ Li<sub>p</sub>schitz continuous<sub>,</sub> we have $\partial f ( x ) = \{ \nabla f ( x ) \}$ I<sub>n</sub> thi<sub>s</sub> <sub>case,</sub> $\| G _ { \gamma h } ( x , \nabla f ( x ) ) \|$ f<sub>or a</sub> fi<sub>xe</sub>d $\gamma > 0$ is the standard stationarit<sub>y</sub> measure (Nesterov, 2013; Ghadimi et al., 2016), and we call $x \in$ dom h an ε-stationary point of Φ if $\| G _ { \gamma h } ( x , \nabla f ( x ) ) \| \leq \varepsilon .$ <sub>.</sub> A<sub>s</sub> shown b<sub>y</sub> Chen et al. (2025, Pro<sub>p</sub>osition 2), $\mathrm { ~ a ~ } ( \gamma , \varepsilon / ( 2 L _ { 1 } ) , \varepsilon / 2 )$ -PGSP is an ε-stationary point of $\Phi _ { ; }$ and conversely, an ε-stationary point of $\Phi$ is a $( \gamma , \delta , \varepsilon ) \ – \mathrm { P G S P }$ <sup>f</sup>or ever<sub>y</sub> $\delta \geq 0$

(iii) In constrained nonsmooth nonconvex optimization, where h in (1) is zero on Ω and $+ \infty$ outside it<sub>,</sub> t<sup>h</sup>e o<sub>p</sub>erator $\operatorname { p r o x } _ { \gamma h }$ becomes the Euclidean projection onto Ω, and Definition 3.3 coincides with the $( \gamma , \delta , \varepsilon )$ -<sub>g</sub>eneralized Goldstein stationarit<sub>y</sub> condition of Liu et al. (2024b, Definition 4.2).

(iv) In weakl<sub>y</sub> convex o<sub>p</sub>timization, stationarit<sub>y</sub> is commonl<sub>y</sub> measured b<sub>y</sub> the <sub>g</sub>radient norm of the Moreau enve<sup>l</sup>ope, w<sup>h</sup>ic<sup>h</sup> a<sup>l</sup>so re<sup>l</sup>ates to PGSP. For t<sup>h</sup>e smoot<sup>h</sup>ed composite objective $\Phi _ { \delta } = f _ { \delta } + h$ <sub>,</sub> l<sub>e</sub>t $L _ { \delta }$ b<sub>e</sub> a Li<sub>p</sub>schitz constant of $\nabla f _ { \delta }$ <sub>an</sub>d <sub>c</sub>h<sub>oose</sub> $0 < \gamma < 1 / L _ { \delta }$ . A point x satisfying $\| \nabla e _ { \gamma } \Phi _ { \delta } ( x ) \| \le \varepsilon$ is a $( \gamma , \delta , ( 1 + \gamma L _ { \delta } ) \varepsilon ) – \mathrm { P G S P }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $e _ { \gamma } \Phi _ { \delta }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> M<sub>oreau enve</sub>l<sub>ope o</sub>f $\Phi _ { \delta }$ (see Pro<sub>p</sub>osition D.1 in A<sub>pp</sub>endix D).

F<sub>or</sub> <sub>ran</sub>d<sub>om</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>s,</sub> <sub>s</sub>t<sub>a</sub>ti<sub>onar</sub>it<sub>y</sub> <sub>guaran</sub>t<sub>ees</sub> b<sub>oun</sub>d th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>res</sub>id<sub>ua</sub>l<sub>.</sub>

T<sup>h</sup>e <sup>f</sup>o<sup>ll</sup>owing <sup>l</sup>emma trans<sup>f</sup>ers a PGSP guarantee <sup>f</sup>rom t<sup>h</sup>e smoot<sup>h</sup>ed objective $f _ { \rho } + h$ to $f + h$ b<sub>y</sub> increasin<sub>g</sub> th<sub>e</sub> G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>i<sub>n ra</sub>di<sub>us</sub> b<sub>y</sub> $\rho ,$ and directl<sub>y</sub> follows from Kornowski and Shamir (2024, Lemma 4).

Lemma 3.4. For any $\rho , \nu \geq 0$ and any x, $\partial _ { \nu } f _ { \rho } ( x ) \subseteq \partial _ { \rho + \nu } f ( x )$ . Consequently, for any $\gamma > 0 ,$ , if x is a $( \gamma , \nu , \varepsilon ) – P G S P$ for $f _ { \rho } + h _ { \cdot }$ , then it is a $( \gamma , \rho + \nu , \varepsilon ) – P G S P f o r \ f + h$

Thi<sub>s</sub> l<sub>emma</sub> l<sub>e</sub>t<sub>s</sub> <sub>us</sub> <sub>ana</sub>l<sub>yze</sub> th<sub>e</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e</sub> f<sub>unc</sub>ti<sub>on</sub> $f _ { \rho }$ <sub>even</sub> <sub>w</sub>h<sub>en</sub> $f$ i<sub>s</sub> <sub>nonsmoo</sub>th <sub>an</sub>d t<sub>rans</sub>f<sub>er</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> g<sup>uarantee</sup> <sup>to</sup> $f + h$ <sup>at</sup> <sup>an</sup>y <sup>tar</sup>g<sup>et</sup> $\delta \geq \rho + \nu$

## 4 Composite O2NC

This section establishes our conversion framework, com<sub>p</sub>osite online-to-nonconvex conversion (Composite O2NC), and <sub>p</sub>rovides the oracle com<sub>p</sub>lexit<sub>y g</sub>uarantees for the first-order and zeroth-order settin<sub>g</sub>.

## 4.1 Proposed conversion framework

H<sub>e</sub>r<sub>e we p</sub>r<sub>ov</sub>id<sub>e</sub> C<sub>o</sub>mp<sub>os</sub>ite O2NC<sub>, su</sub>mm<sub>a</sub>riz<sub>e</sub>d in Al<sub>go</sub>rithm 1 <sub>a</sub>nd ill<sub>us</sub>tr<sub>a</sub>t<sub>e</sub>d in Fi<sub>gu</sub>r<sub>e</sub> 1<sub>.</sub> Th<sub>e ve</sub>r<sub>s</sub>i<sub>o</sub>n<sub>s w</sub>ith fi<sub>rs</sub>t<sub>-or</sub>d<sub>er access an</sub>d <sub>zero</sub>th<sub>-or</sub>d<sub>er access</sub> dif<sub>er on</sub>l<sub>y</sub> i<sub>n</sub> th<sub>e query orac</sub>l<sub>e</sub> $\mathsf { Q }$ <sub>use</sub>d i<sub>n</sub> Li<sub>ne</sub> 6<sub>.</sub> I<sub>n w</sub>h<sub>a</sub>t f<sub>o</sub>ll<sub>ows, we</sub>

Algorithm 1: Composite O2NC   
Require: initial point $x _ { 1 } \in$ dom $h ,$ r<sub>a</sub>di<sub>us</sub> $D > 0 ,$ bl<sub>oc</sub>k l<sub>eng</sub>th $T ,$ <sub>,</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f bl<sub>oc</sub>k<sub>s</sub> $K ,$ , q<sup>uer</sup>y <sup>oracle</sup> $\mathsf { Q } ,$   
<sub>on</sub>li<sub>ne</sub> l<sub>earner.</sub>   
1 for $k = 1 , \ldots , K$ do   
2 I<sub>n</sub>iti<sub>a</sub>li<sub>ze</sub> th<sub>e on</sub>li<sub>ne</sub> l<sub>earner.</sub>   
3 for $t = 1 , \dots , T$ do   
4 Onlin<sub>e</sub> l<sub>ea</sub>rn<sub>e</sub>r <sub>ou</sub>t<sub>pu</sub>t<sub>s</sub> $u _ { k , t } \in \mathcal { U } _ { D } ( x _ { k } )$   
5 D<sub>raw</sub> $s _ { k , t } \sim \mathsf { U n i f } [ 0 , 1 ]$ <sub>an</sub>d <sub>se</sub>t $y _ { k , t } = x _ { k } + s _ { k , t } u _ { k , t } .$   
6 Obt<sub>a</sub>in $g _ { k , t }$ <sub>us</sub>i<sub>ng</sub> th<sub>e ava</sub>il<sub>a</sub>bl<sub>e orac</sub>l<sub>e:</sub>   
7 (i) First-order access: $g _ { k , t } = \mathsf Q _ { \mathrm { f o } } ( y _ { k , t } )$ <sub>,</sub> satisf<sub>y</sub>in<sub>g</sub> Assum<sub>p</sub>tion 3.1.   
8 (ii) Zeroth-order access: $g _ { k , t } = \mathsf { Q } _ { \mathsf { z o } } ( y _ { k , t } ) = \widehat { g } _ { \rho } ( y _ { k , t } ; \xi , w )$ , defined in (9).   
9 Give the learner the convex loss in (5): $\ell _ { k , t } ( u ) = \langle g _ { k , t } , u \rangle + h ( x _ { k } + u ) - h ( x _ { k } )$   
10 D<sub>e</sub>fi<sub>ne</sub> $\begin{array} { r } { \bar { g } _ { k } = T ^ { - 1 } \sum _ { t = 1 } ^ { T } g _ { k , t } . } \end{array}$   
11 D<sub>raw</sub> $I _ { k } \sim \mathsf { U n i f } \{ 1 , \dots , T \}$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y an</sub>d <sub>se</sub>t $x _ { k + 1 } = x _ { k } + u _ { k , I _ { k } }$   
12 Draw R ∼ Unif $\{ 1 , \ldots , K \}$ ind<sub>epe</sub>nd<sub>e</sub>ntl<sub>y.</sub>   
<sub>13</sub> Shift th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>anc</sub>h<sub>or</sub> b<sub>y</sub> $u _ { R } ^ { * } \in \arg$ min $\mathop { \cdot u } \in \mathcal { U } _ { D } ( x _ { R } )  { \left\{ \left. { \bar { g } } _ { R } , u \right. + h ( x _ { R } + u ) - h ( x _ { R } ) \right\} } $   
Return: the shifted point $y _ { R } = x _ { R } + u _ { R } ^ { * } .$

d<sub>escr</sub>ib<sub>e</sub> Al<sub>gor</sub>ith<sub>m</sub> 1 i<sub>n</sub> d<sub>e</sub>t<sub>a</sub>il<sub>.</sub>

The al orithm runs K blocks of T rounds, and in each block $k \in [ K ]$ th<sub>e u</sub> d<sub>a</sub>t<sub>e</sub> i<sub>s cen</sub>t<sub>ere</sub>d <sub>a</sub>t <sub>an anc</sub>h<sub>or</sub> $x _ { k } . ~ \mathrm { A t }$ <sub>eac</sub>h <sub>anc</sub>h<sub>or, an on</sub>li<sub>ne</sub> l<sub>earner c</sub>h<sub>ooses a</sub> di<sub>rec</sub>ti<sub>on</sub> $u _ { k , t }$ f<sub>rom</sub> $\mathcal { U } _ { D } ( x _ { k } ) : = \left\{ u \in \mathbb { R } ^ { d } \colon \| u \| \leq D , x _ { k } + u \in \mathrm { d o m } h \right\}$ in each round t (Line 4). The algorithm then draws $s _ { k , t } \sim \mathsf { U n i f } [ 0 , 1 ]$ , sets $y _ { k , t } = x _ { k } + s _ { k , t } u _ { k , t }$ <sub>,</sub> and <sub>q</sub>ueries $g _ { k , t } = \mathsf Q ( y _ { k , t } )$ (Lines 5 and 6). With first-order access, Q returns an unbiased stochastic <sub>g</sub>radient of $f$ at $y _ { k , t } ,$ as in Assum<sub>p</sub>tion 3.1. With zeroth-order access, Q uses function values at two nearb<sub>y p</sub>oints to construct an <sub>u</sub>nbi<sub>ase</sub>d <sub>es</sub>tim<sub>a</sub>t<sub>o</sub>r <sub>o</sub> $\nabla f _ { \rho } ( y _ { k , t } )$ , as defined in (9). The al<sub>g</sub>orithm then <sub>g</sub>ives the learner the loss

$$
\ell _ { k , t } ( u ) = \langle g _ { k , t } , u \rangle + h ( x _ { k } + u ) - h ( x _ { k } ) ,\tag{5}
$$

as in Line 9. This loss is convex in u since $h$ is con<sub>v</sub>ex.

L<sub>e</sub>t $\mathcal { F } _ { k , t }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory</sub> b<sub>e</sub>f<sub>ore</sub> $s _ { k , t }$ i<sub>s</sub> d<sub>rawn,</sub> <sub>w</sub>hi<sub>c</sub>h d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> $u _ { k , t }$ . <sup>S</sup>u<sub>pp</sub>ose t<sup>h</sup>at $\mathbb { E } [ g _ { k , t } \mid y _ { k , t } ] = \nabla f ( y _ { k , t } )$ <sub>w</sub>hi<sub>c</sub>h h<sub>o</sub>ld<sub>s un</sub>d<sub>er</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er access</sub> b<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 3<sub>.</sub>1 <sub>an</sub>d <sub>un</sub>d<sub>er zero</sub>th<sub>-or</sub>d<sub>er access w</sub>ith $f _ { \rho }$ in <sub>p</sub>l<sub>ace o</sub>f $f$ (Section 4.3). Then, the conditional ex<sub>p</sub>ectation of the loss can be rewritten as

$$
\begin{array} { l l } { \mathbb { E } [ \ell _ { k , t } ( u _ { k , t } ) \mid \mathcal { F } _ { k , t } ] = \displaystyle \int _ { 0 } ^ { 1 } \langle \nabla f ( x _ { k } + s u _ { k , t } ) , u _ { k , t } \rangle \mathrm { d } s + h ( x _ { k } + u _ { k , t } ) - h ( x _ { k } ) } \\ { \displaystyle \qquad = \Phi ( x _ { k } + u _ { k , t } ) - \Phi ( x _ { k } ) , } \end{array}\tag{6}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e un</sub>bi<sub>ase</sub>d<sub>ness o</sub>f $g _ { k , t }$ <sub>an</sub>d $s _ { k , t } \sim \mathsf { U n i f } [ 0 , 1 ]$ <sub>, an</sub>d th<sub>e secon</sub>d <sub>equa</sub>lit<sub>y</sub> f<sub>rom</sub> th<sub>e we</sub>ll<sub>-</sub>b<sub>e</sub>h<sub>ave</sub>d<sub>ness o</sub>f $f .$

Aft<sub>er</sub> $T$ <sub>roun</sub>d<sub>s,</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> d<sub>raws</sub> $I _ { k }$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>an</sub>d <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> f<sub>rom</sub> $\{ 1 , \ldots , T \}$ <sub>an</sub>d <sub>se</sub>t<sub>s</sub> $x _ { k + 1 } =$ $x _ { k } + u _ { k , I _ { k } }$ (Line 11). Due to this u<sub>p</sub>date rule, avera<sub>g</sub>in<sub>g</sub> over $I _ { k }$ and usin<sub>g</sub> (6) <sub>g</sub>ives

$$
\mathbb { E } [ \Phi ( x _ { k + 1 } ) - \Phi ( x _ { k } ) ] = \mathbb { E } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u _ { k , t } ) \right] .
$$

![](images/2a29abce6a89b3c2ca1ac42fa098c34d20c387a703c37f8f80aa36bd61f42b03.jpg)

Figure 1: Query points and iterate updates in Composite O2NC. Each anchor (black dot) is fixed for T rounds.   
Th<sub>e</sub> d<sub>as</sub>h<sub>e</sub>d <sub>c</sub>i<sub>rc</sub>l<sub>e</sub> d<sub>eno</sub>t<sub>es</sub> $\mathbb { B } ( x _ { k } , D )$ .

Th<sub>us, we can es</sub>ti<sub>ma</sub>t<sub>e</sub> th<sub>e</sub> f<sub>unc</sub>ti<sub>on</sub> dif<sub>erence</sub> $\Phi ( x _ { k } + u _ { k , t } ) - \Phi ( x _ { k } )$ b<sub>y</sub> the cumulative loss based on (5), <sub>w</sub>hi<sub>c</sub>h i<sub>s connec</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e regre</sub>t <sub>as we w</sub>ill <sub>see</sub> b<sub>e</sub>l<sub>ow.</sub>

After the K blocks, the algorithm draws R uniformly from $[ K ]$ (Line 12) and returns the shifted <sub>p</sub>oint $y _ { R } = x _ { R } + u _ { R } ^ { * } ,$ <sub>w</sub>h<sub>ere</sub> $u _ { R } ^ { * }$ i<sub>s</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer</sub> i<sub>n</sub> Li<sub>ne</sub> 13<sub>.</sub> Th<sub>e ana</sub>l<sub>yses</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>ons</sub> 4<sub>.</sub>2 <sub>an</sub>d 4<sub>.</sub>3 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h PGSP <sub>g</sub>uarantees at the returned <sub>p</sub>oint $y _ { R }$

We measure the online learner’s performance in block k by comparing its cumulative loss with that of the best fi<sub>xe</sub>d di<sub>rec</sub>ti<sub>on</sub> i<sub>n</sub> $\mathcal { U } _ { D } ( x _ { k } )$ . We use a common u<sub>pp</sub>er bound $R _ { T }$ on this expected regret over all blocks in [K] t<sub>o</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>guaran</sub>t<sub>ees</sub> b<sub>e</sub>l<sub>ow,</sub>

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u _ { k , t } ) - \operatorname* { m i n } _ { u \in \mathcal { U } _ { D } ( x _ { k } ) } \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u ) \right] \leq R _ { T } .\tag{7}
$$

Euclidean com<sub>p</sub>osite objective mirror descent (COMID) (Duchi et al., 2010, Theorem 2) achieves the followin<sub>g</sub> b<sub>oun</sub>d <sub>on</sub> $R _ { T }$ .

Theorem 4.1. Suppose that $\mathbb { E } \big [ \| g _ { k , t } \| ^ { 2 } \mid y _ { k , t } \big ] \le G ^ { 2 }$ for every k, t. Then, COMID achieves $R _ { T } \leq 2 D G { \sqrt { T } } .$ To relate the best fixed loss in (7) to PGSP, we define the local com<sub>p</sub>osite <sub>g</sub>a<sub>p</sub> for $x \in$ dom h, $\boldsymbol { g } \in \mathbb { R } ^ { d }$ <sub>,</sub> <sub>an</sub>d $D > 0$ <sup>b</sup><sub>y</sub>

$$
\mathcal { C } _ { D } ( x , g ) : = - \frac { 1 } { D } \operatorname* { m i n } _ { u \in \mathcal { U } _ { D } ( x ) } \{ \langle g , u \rangle + h ( x + u ) - h ( x ) \} .
$$

It is nonne<sub>g</sub>ati<sub>v</sub>e since $u = 0 \in \mathcal { U } _ { D } ( x )$ gives the value 0. Let $\begin{array} { r } { \bar { g } _ { k } : = T ^ { - 1 } \sum _ { t = 1 } ^ { T } g _ { k , t } } \end{array}$ b<sub>e</sub> th<sub>e</sub> <sub>average</sub> <sub>o</sub>f th<sub>e</sub> gradient estimates returned in block k, as in Line 10. Then the best fixed average loss is expressed in terms of the local com<sub>p</sub>osite <sub>g</sub>a<sub>p</sub> $\mathcal { C } _ { D } ( x , g )$ <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

$$
\begin{array} { l } { \displaystyle \frac { 1 } { T } \operatorname* { m i n } _ { u \in \mathcal { U } _ { D } ( x _ { k } ) } \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u ) = \operatorname* { m i n } _ { u \in \mathcal { U } _ { D } ( x _ { k } ) } \{ \langle \bar { g } _ { k } , u \rangle + h ( x _ { k } + u ) - h ( x _ { k } ) \} } \\ { = - D \mathcal { C } _ { D } ( x _ { k } , \bar { g } _ { k } ) . } \end{array}\tag{8}
$$

At Li<sub>ne</sub> 13<sub>,</sub> Al<sub>gor</sub>ith<sub>m</sub> 1 <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>compos</sub>it<sub>e mo</sub>d<sub>e</sub>l <sub>aroun</sub>d th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>anc</sub>h<sub>or</sub> $x _ { R }$ <sub>an</sub>d <sub>re</sub>t<sub>urns</sub> th<sub>e</sub> <sub>s</sub>hift<sub>e</sub>d <sub>po</sub>int $y _ { R }$ <sub>.</sub> Thi<sub>s s</sub>hift <sub>a</sub>ll<sub>ows us</sub> t<sub>o o</sub>bt<sub>a</sub>in th<sub>e co</sub>n<sub>ve</sub>r<sub>ge</sub>n<sub>ce gua</sub>r<sub>a</sub>nt<sub>ee</sub> t<sub>o</sub> th<sub>e</sub> PGSP <sub>s</sub>im<sub>u</sub>lt<sub>a</sub>n<sub>eous</sub>l<sub>y</sub> f<sub>o</sub>r <sup>ever</sup>y $\gamma > 0$ <sub>, as</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> l<sub>emma s</sub>h<sub>ows.</sub>

Lemma 4.2. Fix $x \in$ dom h, $D \ > \ 0 .$ , and $\nu \geq 2 D$ . Let $g \ \in \ \partial _ { D } f ( x )$ and $\widehat { \boldsymbol { g } } _ { \mathbf { \lambda } } \in \mathbb { R } ^ { d }$ . Let $u ^ { * } \in$ arg $\begin{array} { r } { \operatorname* { m i n } _ { u \in \mathcal { U } _ { D } ( x ) } \{ \langle \widehat { g } , u \rangle + h ( x + u ) - h ( x ) \} } \end{array}$ } and let $y = x + u ^ { * }$ . Then, for every $\gamma > 0 ;$

$$
\operatorname* { m i n } _ { v \in \partial _ { \nu } f ( y ) } \| G _ { \gamma h } ( y , v ) \| \leq \mathcal { C } _ { D } ( x , \widehat { g } ) + \| g - \widehat { g } \| .
$$

As discussed in the introduction, existin<sub>g</sub> a<sub>pp</sub>roaches (Nesterov, 2013; Parikh and Bo<sub>y</sub>d, 2014; Ghadimi et al., 2016; Chen et al., 2025) use the <sub>p</sub>arameter of the <sub>p</sub>roximal ma<sub>p</sub> as an u<sub>p</sub>date ste<sub>p</sub> size. B<sub>y</sub> contrast, our algorithm does not require tuning γ, and the complexity for finding a PGSP is independent of $\gamma$

## 4.2 First-order complexity

W<sub>e</sub> fi<sub>rs</sub>t <sub>ana</sub>l<sub>yze</sub> Al<sub>gor</sub>ith<sub>m</sub> 1 <sub>w</sub>ith fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> <sub>access</sub> t<sub>o</sub> $f ,$ sett<sup>i</sup>n<sub>g</sub> $\mathsf Q = \mathsf Q _ { \mathrm { f o } }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\mathsf Q _ { \mathrm { f o } } ( y )$ <sub>ca</sub>ll<sub>s</sub> th<sub>e s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>gra</sub>di<sub>en</sub>t <sub>orac</sub>l<sub>e o</sub>f A<sub>ssump</sub>ti<sub>on</sub> 3<sub>.</sub>1 <sub>a</sub>t $y .$

L<sub>e</sub>t $\begin{array} { r } { \widetilde { g } _ { k } : = T ^ { - 1 } \sum _ { t = 1 } ^ { T } \nabla f ( y _ { k , t } ) } \end{array}$ b<sub>e</sub> th<sub>e average gra</sub>di<sub>en</sub>t i<sub>n</sub> bl<sub>oc</sub>k $k .$ Since all <sub>q</sub>uer<sub>y p</sub>oints lie in $\mathbb { B } ( x _ { k } , D )$ , we h<sub>ave</sub> $\widetilde { g } _ { k } \in \partial _ { D } f ( x _ { k } )$ <sub>.</sub> B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> 4<sub>.</sub>2<sub>,</sub> it <sub>su</sub>fi<sub>ces</sub> t<sub>o</sub> <sub>uppe</sub>r b<sub>ou</sub>nd th<sub>e</sub> l<sub>oca</sub>l <sub>co</sub>m<sub>pos</sub>it<sub>e</sub> <sub>gap</sub> $\mathcal { C } _ { D } ( x _ { R } , \bar { g } _ { R } )$ <sub>an</sub>d th<sub>e</sub> <sub>g</sub>radient estimation error $\| \bar { g } _ { R } - \widetilde { g } _ { R } \|$ <sub>.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> l<sub>emma g</sub>i<sub>ves</sub> th<sub>ese</sub> b<sub>oun</sub>d<sub>s.</sub>

Lemma 4.3. Suppose that f is well-behaved and that the oracle satisfies conditions (i) and (ii) of Assumption 3.1. Ifan online learner satisfies (7), then

$$
\mathbb { E } [ \mathcal { C } _ { D } ( x _ { R } , \bar { g } _ { R } ) ] \leq \frac { \Delta } { D K } + \frac { R _ { T } } { D T } , \quad \mathbb { E } [ \| \bar { g } _ { R } - \widetilde { g } _ { R } \| ] \leq \frac { \sigma } { \sqrt { T } } .
$$

Proof. By (7) and (8), we have

$$
D \mathbb { E } [ \mathcal { C } _ { D } ( x _ { k } , \bar { g } _ { k } ) ] \leq \frac { R _ { T } } { T } - \mathbb { E } \Bigg [ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u _ { k , t } ) \Bigg ] = \frac { R _ { T } } { T } + \mathbb { E } [ \Phi ( x _ { k } ) - \Phi ( x _ { k + 1 } ) ] ,
$$

where the last inequalit<sub>y</sub> uses (6). Summin<sub>g</sub> over $k \in [ K ]$ and usin<sub>g</sub> $R \sim \mathsf { U n i f } \{ 1 , \dots , K \}$ <sub>g</sub><sup>i</sup>ves

$$
\mathbb { E } [ \mathcal { C } _ { D } ( x _ { R } , \bar { g } _ { R } ) ] \le \frac { R _ { T } } { D T } + \frac { \Phi ( x _ { 1 } ) - \mathbb { E } [ \Phi ( x _ { K + 1 } ) ] } { D K } \le \frac { R _ { T } } { D T } + \frac { \Delta } { D K } ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f th<sub>e</sub> fi<sub>rs</sub>t <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>.</sub>

F<sub>o</sub>r th<sub>e</sub> <sub>seco</sub>nd <sub>s</sub>t<sub>a</sub>t<sub>e</sub>m<sub>e</sub>nt<sub>,</sub> b<sub>y</sub> J<sub>e</sub>n<sub>se</sub>n’<sub>s</sub> in<sub>equa</sub>lit<sub>y,</sub>

$$
\mathbb { E } [ \| \bar { g } _ { R } - \widetilde { g } _ { R } \| ] \le \sqrt { \mathbb { E } [ \| \bar { g } _ { R } - \widetilde { g } _ { R } \| ^ { 2 } ] } = \sqrt { \frac { 1 } { K T ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| g _ { k , t } - \nabla f ( y _ { k , t } ) \| ^ { 2 } ] } \le \frac { \sigma } { \sqrt { T } } ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $R \sim \mathsf { U n i f } \{ 1 , \dots , K \}$ <sub>an</sub>d th<sub>e con</sub>diti<sub>ona</sub>l <sub>un</sub>bi<sub>ase</sub>d<sub>ness o</sub>f $g _ { k , t }$ f<sub>or</sub> $\nabla f ( y _ { k , t } )$ <sub>removes</sub> th<sub>e cross</sub> t<sub>erms, an</sub>d th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e var</sub>i<sub>ance</sub> b<sub>oun</sub>d i<sub>n</sub> A<sub>ssump</sub>ti<sub>on</sub> 3<sub>.</sub>1<sub>.</sub> □

C<sub>o</sub>mbinin<sub>g</sub> L<sub>e</sub>mm<sub>as</sub> 4<sub>.</sub>2 <sub>a</sub>nd 4<sub>.</sub>3 <sub>w</sub>ith th<sub>e</sub> COMID r<sub>eg</sub>r<sub>e</sub>t b<sub>ou</sub>nd in Th<sub>eo</sub>r<sub>e</sub>m 4<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>in<sub>g</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> g<sup>radient</sup> q<sup>uer</sup>y <sup>com</sup>p<sup>lexit</sup>y<sup>.</sup>

Theorem 4.4. Suppose that Assumption 3.1 holds, fix $\delta > 0$ and $0 < \varepsilon \le$ min $\{ G + \sigma , \Delta / \delta \}$ , and let $D = \delta / 2 , T = \operatorname* { m a x } \{ 1 , \lceil 4 ( 2 G + \sigma ) ^ { 2 } / \varepsilon ^ { 2 } \rceil \}$ , and $K = \mathrm { m a x } \{ 1 , \lceil 2 \Delta / ( D \varepsilon ) \rceil \}$ . Then Algorithm 1 with the COMID learner ofTheorem 4.1 and $\mathsf Q = \mathsf Q _ { \mathrm { f o } }$ outputs $y _ { R }$ that is a $( \gamma , \delta , \varepsilon ) – P G S P$ in expectation for every $\gamma > 0 ,$ , using $\begin{array} { r } { O \left( \frac { \Delta ( G + \sigma ) ^ { 2 } } { \delta \varepsilon ^ { 3 } } \right) } \end{array}$ stochastic gradient queries.

Thi<sub>s</sub> b<sub>oun</sub>d <sub>ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e</sub> $O ( \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ com<sub>p</sub>lexit<sub>y</sub> of Cutkosk<sub>y</sub> et al. (2023) for $h = 0$ <sub>.</sub> Sin<sub>ce</sub> $h = 0$ is a s<sub>p</sub>ecial case o<sup>f</sup> our setting, t<sup>h</sup>eir joint <sup>l</sup>ower bound $\Omega ( \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ also a<sub>pp</sub>lies to our com<sub>p</sub>osite <sub>p</sub>roblem class<sub>,</sub> establishin<sub>g</sub> optimal dependence on δ and ε.

Proof. Since $\widetilde { g } _ { R } \in \partial _ { D } f ( x _ { R } )$ <sub>an</sub>d $\delta = 2 D$ <sub>, us</sub>in<sub>g</sub> L<sub>e</sub>mm<sub>a</sub> 4<sub>.</sub>2 <sub>w</sub>ith $x = x _ { R } , g = \widetilde { g } _ { R }$ <sub>,</sub> <sub>an</sub>d $\widehat { \boldsymbol { g } } = \bar { \boldsymbol { g } } _ { R }$ <sub>g</sub>ives<sub>,</sub> for <sup>ever</sup>y $\gamma > 0$

$$
\begin{array} { r l } & { \mathbb { E } \bigg [ \underset { g \in \partial \delta f ( y _ { R } ) } { \operatorname* { m i n } } \| G _ { \gamma h } ( y _ { R } , g ) \| \bigg ] \leq \mathbb { E } [ \mathcal { C } _ { D } ( x _ { R } , \bar { g } _ { R } ) + \| \bar { g } _ { R } - \widetilde { g } _ { R } \| ] } \\ & { \quad \quad \quad \quad \quad \quad \quad \leq \frac { \Delta } { D K } + \frac { R _ { T } } { D T } + \frac { \sigma } { \sqrt { T } } } \\ & { \quad \quad \quad \quad \quad \leq \frac { \Delta } { D K } + \frac { 2 G + \sigma } { \sqrt { T } } \leq \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 2 } = \varepsilon , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> L<sub>emma</sub> 4<sub>.</sub>3 <sub>an</sub>d th<sub>e</sub> thi<sub>r</sub>d i<sub>nequa</sub>lit<sub>y uses</sub> $R _ { T } \leq 2 D G \sqrt { T }$ f<sub>rom</sub> Th<sub>eorem</sub> 4<sub>.</sub>1<sub>.</sub> Si<sub>nce</sub> $\varepsilon \le \Delta / \delta$ <sub>an</sub>d $\varepsilon \leq G + \sigma$ <sub>,</sub> the <sub>p</sub>arameter choices <sub>g</sub>ive $K = { O } \left( { \Delta } / { ( \delta \varepsilon ) } \right)$ <sub>an</sub>d $T =$ $O \big ( ( G + \sigma ) ^ { 2 } / \varepsilon ^ { 2 } \big )$ <sub>.</sub> Th<sub>e</sub>ir <sub>p</sub>r<sub>o</sub>d<sub>uc</sub>t $K T$ <sub>g</sub>i<sub>ves</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d n<sub>u</sub>mb<sub>e</sub>r <sub>o</sub>f <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c g</sub>r<sub>a</sub>di<sub>e</sub>nt <sub>que</sub>ri<sub>es, w</sub>hi<sub>c</sub>h <sub>co</sub>m<sub>p</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e proo</sub>f<sub>.</sub> □

## 4.3 Zeroth-order complexity

H<sub>ere, we ana</sub>l<sub>yze</sub> Al<sub>gor</sub>ith<sub>m</sub> 1 <sub>w</sub>ith <sub>zero</sub>th<sub>-or</sub>d<sub>er access un</sub>d<sub>er</sub> A<sub>ssump</sub>ti<sub>on</sub> 3<sub>.</sub>2<sub>.</sub> F<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> K<sub>ornows</sub>ki <sub>an</sub>d Shamir (2024), we a<sub>pp</sub>l<sub>y</sub> the al<sub>g</sub>orithm to $\Phi _ { \rho } : = f _ { \rho } + h$ <sub>, w</sub>h<sub>ere</sub> $f _ { \rho }$ i<sub>s</sub> th<sub>e un</sub>if<sub>orm</sub>l<sub>y smoo</sub>th<sub>e</sub>d f<sub>unc</sub>ti<sub>on</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>o</sub>n 3<sub>.</sub>1 <sub>w</sub>ith r<sub>a</sub>di<sub>us</sub> $\rho > 0$ . Write $\Delta _ { \rho } : = \Phi _ { \rho } ( x _ { 1 } ) - \mathrm { i n f } _ { x } \Phi _ { \rho } ( x )$ for its initial <sub>g</sub>a<sub>p</sub>.

At each query point y, draw independent samples $\xi$ <sub>an</sub>d $w \sim \mathsf { U n i f } ( \mathbb { S } ^ { d - 1 } )$ <sub>, an</sub>d <sub>se</sub>t $\mathsf Q _ { \mathrm { z o } } ( y ) : = \widehat { g } _ { \rho } ( y ; \xi , w )$ <sub>w</sub>h<sub>ere</sub>

$$
\widehat { g } _ { \rho } ( y ; \xi , w ) : = \frac { d } { 2 \rho } ( F ( y + \rho w , \xi ) - F ( y - \rho w , \xi ) ) w .\tag{9}
$$

E<sub>ac</sub>h <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub> <sub>ca</sub>ll <sub>uses</sub> t<sub>wo</sub> f<sub>unc</sub>ti<sub>on-va</sub>l<sub>ue</sub> <sub>quer</sub>i<sub>es.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> l<sub>emma</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>mean</sub> <sub>an</sub>d <sub>a</sub> <sub>secon</sub>d<sub>-momen</sub>t b<sub>oun</sub>d f<sub>or</sub> thi<sub>s es</sub>ti<sub>ma</sub>t<sub>or.</sub>

Lemma 4.5. Under Assumption 3.2, set $G _ { \mathrm { z o } } : = 4 ( 2 \pi ) ^ { 1 / 4 } L _ { 0 } \sqrt { d } .$ . Then f is L<sub>0</sub>-Lipschitz, and for every $\boldsymbol { y } \in \mathbb { R } ^ { d } .$ the estimator (9) satisfies

$$
\mathbb { E } _ { \xi , w } [ \widehat { g } _ { \rho } ( y ; \xi , w ) \mid y ] = \nabla f _ { \rho } ( y ) ,\tag{10}
$$

$$
\begin{array} { r } { \mathbb { E } _ { \xi , w } \left[ \| \widehat { g } _ { \rho } ( y ; \xi , w ) \| ^ { 2 } \mid y \right] \leq G _ { \mathrm { z o } } ^ { 2 } . } \end{array}\tag{11}
$$

Consequently, $\mathsf { Q } _ { \mathrm { z o } }$ satisfies Assumption 3.1 for $f _ { \rho }$ with $G = \sigma = G _ { \mathrm { z o } }$

U<sub>s</sub>i<sub>ng</sub> th<sub>e</sub> <sub>argumen</sub>t <sub>o</sub>f Th<sub>eorem</sub> 4<sub>.</sub>4 f<sub>or</sub> $\Phi _ { \rho }$ <sub>a</sub>nd L<sub>e</sub>mm<sub>a</sub> 3<sub>.</sub>4 <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>gua</sub>r<sub>a</sub>nt<sub>ee</sub> f<sub>o</sub>r $f + h$ <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d b<sub>e</sub>l<sub>ow.</sub>

Theorem 4.6. Suppose that Assumption 3.2 holds, fix $\delta > 0$ and $0 < \varepsilon \le L _ { 0 } ,$ , and let $D = \rho = \delta / 3$ $T = \mathrm { m a x } \{ 1 , \left\lceil 5 7 6 \sqrt { 2 \pi } d L _ { 0 } ^ { 2 } / \varepsilon ^ { 2 } \right\rceil \}$ , and $K = \mathrm { m a x } \{ 1 , \lceil 2 ( \Delta + \delta L _ { 0 } ) / ( D \varepsilon ) \rceil \}$ . Then Algorithm 1 with the COMID learner ofTheorem 4.1 and $\mathsf Q = \mathsf Q _ { \mathrm { z o } }$ outputs $y _ { R }$ that is $a \left( \gamma , \delta , \varepsilon \right) \ – P G S P$ in expectation for every $\gamma > 0$ , using $O \left( \frac { d L _ { 0 } ^ { 2 } ( \Delta + \delta L _ { 0 } ) } { \delta \varepsilon ^ { 3 } } \right)$ function-value queries.

Thi<sub>s</sub> b<sub>oun</sub>d i<sub>mproves on</sub> th<sub>e</sub> $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 4 } )$ <sub>an</sub>d $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ com<sub>p</sub>lexities obtained b<sub>y</sub> Chen et al. (2025) with minibatch estimation and variance reduction<sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>. It also im<sub>p</sub>roves on the $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 4 } )$

function-value quer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> for PGSPs obtained from Pou<sub>g</sub>kakiotis and Kalo<sub>g</sub>erias (2026) under the <sub>orac</sub>l<sub>e con</sub>diti<sub>ons</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> D<sub>.</sub> It <sub>ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e</sub> $O ( d \delta ^ { - 1 } \varepsilon ^ { - 3 } )$ <sub>co</sub>m<sub>p</sub>l<sub>e</sub>xit<sub>y o</sub>f K<sub>o</sub>rn<sub>ows</sub>ki <sub>a</sub>nd Sh<sub>a</sub>mir (2024) for $h = 0$ <sub>a</sub>nd i<sub>s</sub> <sub>op</sub>tim<sub>a</sub>l in <sub>eac</sub>h <sub>o</sub>f $d , \delta ,$ , and ε individually.

## 5 Smooth setting

This section derives oracle complexity bounds for finding an ε-stationary point of Φ when f is smooth. In <sub>p</sub>articular<sub>,</sub> this section assumes that $f$ is $L _ { \mathrm { 1 } } \mathrm { - s m o o t h } .$ <sub>,</sub> th<sub>a</sub>t i<sub>s,</sub> $\| \nabla f ( x ) - \nabla f ( y ) \| \leq L _ { 1 } \| x - y \|$ f<sub>or a</sub>ll $x , y \in \mathbb { R } ^ { d }$ Th<sub>e proo</sub>f<sub>s o</sub>f thi<sub>s sec</sub>ti<sub>on are</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>2<sub>.</sub>

The <sub>p</sub>roximal-<sub>g</sub>radient ma<sub>pp</sub>in<sub>g</sub> $G _ { \gamma h } ( x , \nabla f ( x ) )$ is commonly used to define ε-stationarity for smooth com <sub>p</sub>osite objectives (Nesterov, 2013; Ghadimi et al., 2016), as in Section 3. The followin<sub>g</sub> lemma relates ε-stationarity to the PGSP condition.

Lemma 5.1. When f is $L _ { 1 }$ -smooth, for every $x \in$ dom $h , \gamma > 0 .$ , and $\delta \geq 0$ , it holds that $\| G _ { \gamma h } ( x , \nabla f ( x ) ) \| \leq$ $\begin{array} { r } { \operatorname* { m i n } _ { g \in \partial _ { \delta } f ( x ) } \| G _ { \gamma h } ( x , g ) \| + L _ { 1 } \delta . } \end{array}$

C<sub>om</sub>bi<sub>n</sub>i<sub>ng</sub> L<sub>emma</sub> 5<sub>.</sub>1 <sub>an</sub>d Th<sub>eorem</sub> 4<sub>.</sub>4 <sub>w</sub>ith $\delta = \varepsilon / ( 2 L _ { 1 } ) \operatorname { g i v e s } O ( \varepsilon ^ { - 4 } )$ stochastic <sub>g</sub>radient <sub>q</sub>uer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> for finding an ε-stationary point of Φ.

To obtain a shar<sub>p</sub>er rate in the deterministic case $\sigma = 0$ <sub>, w</sub>e incor<sub>p</sub>orate the idea of o<sub>p</sub>timistic online mirror descent (Chian<sub>g</sub> et al., 2012; Rakhlin and Sridharan, 2013) into COMID (see A<sub>pp</sub>endix A for the u<sub>p</sub>date and the <sub>p</sub>roof). The o<sub>p</sub>timistic COMID learner uses the <sub>p</sub>revious stochastic <sub>g</sub>radient as a hint. To initialize the hi<sub>n</sub>t i<sub>n eac</sub>h bl<sub>oc</sub>k<sub>,</sub> l<sub>e</sub>t $g _ { k , 0 }$ be an inde<sub>p</sub>endent stochastic <sub>g</sub>radient <sub>q</sub>ueried at $x _ { k }$ . Since the <sub>q</sub>uer<sub>y p</sub>oints in each bl<sub>oc</sub>k li<sub>e</sub> i<sub>n</sub> $\mathbb { B } ( x _ { k } , D )$ <sub>, smoo</sub>th<sub>ness an</sub>d th<sub>e var</sub>i<sub>ance</sub> b<sub>oun</sub>d <sub>g</sub>i<sub>ve</sub> $\mathbb { E } \big [ \| g _ { k , t } - g _ { k , t - 1 } \| ^ { 2 } \big ] \leq 1 2 ( L _ { 1 } D + \sigma ) ^ { 2 }$ . Usin<sub>g</sub> thi<sub>s</sub> b<sub>oun</sub>d i<sub>n</sub> th<sub>e regre</sub>t <sub>ana</sub>l<sub>ys</sub>i<sub>s y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng guaran</sub>t<sub>ee w</sub>ith<sub>ou</sub>t th<sub>e secon</sub>d<sub>-momen</sub>t b<sub>oun</sub>d i<sub>n con</sub>diti<sub>on</sub> (iii) of Assum<sub>p</sub>tion 3.1.

Theorem 5.2. Suppose that f is $L _ { 1 }$ -smooth and that conditions (i) and (ii) ofAssumption 3.1 hold. Then, optimistic COMID achieves $R _ { T } \leq 2 \sqrt { 3 } ( L _ { 1 } D + \sigma ) D \sqrt { T }$

N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t <sub>op</sub>timi<sub>s</sub>ti<sub>c</sub> COMID i<sub>s</sub> n<sub>o</sub>t limit<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> <sub>s</sub>m<sub>oo</sub>th <sub>se</sub>ttin<sub>g:</sub> th<sub>e</sub> <sub>gua</sub>r<sub>a</sub>nt<sub>ee</sub> <sub>o</sub>f Th<sub>eo</sub>r<sub>e</sub>m 4<sub>.</sub>4 <sub>a</sub>l<sub>so</sub> h<sub>o</sub>ld<sub>s</sub> <sub>w</sub>ith thi<sub>s</sub> l<sub>ea</sub>rn<sub>e</sub>r in th<sub>e</sub> n<sub>o</sub>n<sub>s</sub>m<sub>oo</sub>th <sub>se</sub>ttin<sub>g.</sub>

W<sub>e</sub> <sub>now</sub> d<sub>er</sub>i<sub>ve</sub> <sub>smoo</sub>th <sub>coun</sub>t<sub>erpar</sub>t<sub>s</sub> <sub>o</sub>f Th<sub>eorems</sub> 4<sub>.</sub>4 <sub>an</sub>d 4<sub>.</sub>6 f<sub>or</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> <sub>an</sub>d <sub>zero</sub>th<sub>-or</sub>d<sub>er</sub> <sub>access.</sub>

Theorem 5.3. Suppose that f is L<sub>1</sub>-smooth and that the oracle satisfies conditions (i) and (ii) ofAssumption 3.1, $\mathit { f i x } \varepsilon > 0 ,$ , and let $D = \varepsilon / ( 1 2 L _ { 1 } ) , T = \operatorname* { m a x } \{ 1 , \lceil 4 0 0 \sigma ^ { 2 } / \varepsilon ^ { 2 } \rceil \}$ , and $K = \mathrm { m a x } \{ 1 , \lceil 4 8 L _ { 1 } \Delta / \varepsilon ^ { 2 } \rceil \}$ . Then Algorithm 1 with the optimistic COMID learner ofTheorem 5.2 and $\mathsf Q = \mathsf Q _ { \mathrm { f o } }$ outputs a point y<sub>R</sub> that is an ε- stationary point of Φ for every $\gamma > 0 ,$ , that is, $\mathbb { E } [ \| G _ { \gamma h } ( y _ { R } , \nabla f ( y _ { R } ) ) \| ] \le \varepsilon ,$ , using $\begin{array} { r } { O \Big ( \operatorname* { m a x } \Big \{ \frac { L _ { 1 } \Delta } { \varepsilon ^ { 2 } } , \frac { \sigma ^ { 2 } } { \varepsilon ^ { 2 } } , \frac { L _ { 1 } \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \Big \} \Big ) } \end{array}$ stochastic gradient queries.

In th<sub>e</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c case</sub> $\sigma = 0$ <sub>,</sub> Th<sub>eorem</sub> 5<sub>.</sub>3 <sub>uses</sub> $O ( L _ { 1 } \Delta / \varepsilon ^ { 2 } )$ gradient queries to find an ε-stationary point of Φ. The com lexit in Theorem 5.3 matches that of roximal stochastic radient methods (Ghadimi et al., 2016). Its $\sigma ^ { 2 } \varepsilon ^ { - 4 }$ d<sub>e en</sub>d<sub>ence a</sub>l<sub>so ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d f<sub>or un</sub>bi<sub>ase</sub>d b<sub>oun</sub>d<sub>e</sub>d<sub>-var</sub>i<sub>ance orac</sub>l<sub>es, even w</sub>h<sub>en</sub> $h = 0 ( \mathrm { A r j e v a n i }$ et al., 2023). Table 2 in A<sub>pp</sub>endix C.1 summarizes these com<sub>p</sub>arisons.

Theorem 5.4. Suppose that Assumption 3.2 holds and that f is $L _ { 1 }$ -smooth,fix $\varepsilon > 0 ,$ , and let $\rho = \varepsilon / ( 2 L _ { 1 } )$ $D = \varepsilon / ( 2 4 L _ { 1 } ) , T = \operatorname* { m a x } \{ 1 , \left\lceil 1 6 0 0 G _ { \mathrm { z o } } ^ { 2 } / \varepsilon ^ { 2 } \right\rceil \}$ , and $K = \lceil 1 9 2 L _ { 1 } \Delta / \varepsilon ^ { 2 } + 4 8 \rceil$ , where $G _ { \mathrm { z o } } = 4 ( 2 \pi ) ^ { 1 / 4 } L _ { 0 } \sqrt { d }$ as in Lemma 4.5. Then Algorithm 1 with the optimistic COMID learner ofTheorem 5.2 and $\mathsf Q = \mathsf Q _ { \mathrm { z o } }$ outputs a point y that is an ε-stationary point of Φ for every $\gamma > 0$ , that is, $\mathbb { E } [ \| G _ { \gamma h } ( y _ { R } , \nabla f ( y _ { R } ) ) \| ] \le \varepsilon$ , using $\begin{array} { r } { O \Big ( \operatorname* { m a x } \Big \{ \frac { L _ { 1 } \Delta } { \varepsilon ^ { 2 } } , \frac { d L _ { 0 } ^ { 2 } } { \varepsilon ^ { 2 } } , \frac { d L _ { 0 } ^ { 2 } L _ { 1 } \Delta } { \varepsilon ^ { 4 } } \Big \} \Big ) } \end{array}$ function-value queries.

![](images/8f6541b62bedf6c5b8db59172b3fffb09cc6c005046f8feb8684b58b0c1785bc.jpg)  
Figure 2: Stationarity measure mi $1 _ { g \in \partial _ { \delta } f ( x ) } \| G _ { \gamma h } ( x , g ) \|$ (left) and objective values (ri<sub>g</sub>ht) for absolute loss for ReLU regression at d = 16 (top), d = 256 (middle), and $d = 1 0 2 4$ (bottom). Curves and shaded bands <sub>s</sub>h<sub>ow means an</sub>d <sub>one s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on over</sub> t<sub>en</sub> t<sub>r</sub>i<sub>a</sub>l<sub>s.</sub>

Th<sub>e</sub> l<sub>ea</sub>di<sub>ng</sub> t<sub>erm</sub> i<sub>n</sub> Th<sub>eorem</sub> 5<sub>.</sub>4 i<sub>s</sub> $O ( d L _ { 0 } ^ { 2 } L _ { 1 } \Delta \varepsilon ^ { - 4 } )$ for finding an ε-stationary point of Φ. This has the same $d \varepsilon ^ { - 4 }$ de<sub>p</sub>endence as the bound for smooth com<sub>p</sub>osite objectives in Ghadimi et al. (2016, Corollar<sub>y</sub> 8).

## 6 Numerical experiments

T<sub>o va</sub>lid<sub>a</sub>t<sub>e our</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>resu</sub>lt<sub>s, we eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e query e</sub>fi<sub>c</sub>i<sub>ency o</sub>f C<sub>omposite</sub> O2NC <sub>on nonsmoo</sub>th n<sub>o</sub>n<sub>co</sub>n<sub>ve</sub>x <sub>co</sub>m<sub>pos</sub>it<sub>e p</sub>r<sub>o</sub>bl<sub>e</sub>m<sub>s.</sub> W<sub>e co</sub>nd<sub>uc</sub>t z<sub>e</sub>r<sub>o</sub>th-<sub>o</sub>rd<sub>e</sub>r <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s o</sub>n <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> l<sub>oss</sub> f<sub>o</sub>r R<sub>e</sub>LU r<sub>eg</sub>r<sub>ess</sub>i<sub>o</sub>n at $d \in \{ 1 6 , 2 5 6 , 1 0 2 4 \}$ <sub>, w</sub>ith $\ell _ { 1 }$ <sub>an</sub>d $\ell _ { 2 }$ re<sub>g</sub>ularization and box constraints. We com<sub>p</sub>are Composite O2NC with the minibatch (o<sub>p</sub>tion G1) and variance-reduced (o<sub>p</sub>tion G2) methods of Chen et al. (2025), usin<sub>g</sub> the same data and initia<sup>l</sup>ization <sup>f</sup>or a<sup>ll</sup> met<sup>h</sup>ods wit<sup>h</sup>in eac<sup>h</sup> tria<sup>l</sup>. T<sup>h</sup>e objective and a<sup>ll</sup> experimenta<sup>l</sup> parameters are in A<sub>pp</sub>endix E.

Figure 2 p<sup>l</sup>ots t<sup>h</sup>e stationarity measure and composite objective va<sup>l</sup>ue against t<sup>h</sup>e number o<sup>f f</sup>unction-va<sup>l</sup>ue queries. We evaluate stationarity using a numerical upper bound on the measure with δ = 0.01 and γ = 1. A<sub>cross a</sub>ll th<sub>ree</sub> di<sub>mens</sub>i<sub>ons,</sub> C<sub>omposite</sub> O2NC <sub>ou</sub>t<sub>per</sub>f<sub>orms</sub> th<sub>e o</sub>th<sub>er me</sub>th<sub>o</sub>d<sub>s</sub> i<sub>n</sub> t<sub>erms o</sub>f b<sub>o</sub>th th<sub>e s</sub>t<sub>a</sub>ti<sub>onar</sub>it<sub>y</sub> measure and t<sup>h</sup>e composite objective va<sup>l</sup>ue. Additiona<sup>l</sup> resu<sup>l</sup>ts <sup>f</sup>or ramp <sup>l</sup>oss are given in Appendix E.

## AI use statement

W<sub>e use</sub>d GPT<sub>-</sub>5<sub>.</sub>5<sub>,</sub> GPT<sub>-</sub>5<sub>.</sub>6<sub>-</sub>S<sub>o</sub>l<sub>,</sub> GPT<sub>-</sub>6<sub>.</sub>1<sub>-</sub>S<sub>o</sub>l<sub>, an</sub>d Cl<sub>au</sub>d<sub>e</sub> O<sub>pus</sub> 5<sub>.</sub>5 t<sub>o ass</sub>i<sub>s</sub>t <sub>w</sub>ith <sub>proo</sub>f d<sub>eve</sub>l<sub>opmen</sub>t <sub>an</sub>d <sub>c</sub>h<sub>ec</sub>ki<sub>ng,</sub> lit<sub>era</sub>t<sub>ure searc</sub>h<sub>es, an</sub>d <sub>e</sub>diti<sub>ng.</sub> W<sub>e rev</sub>i<sub>ewe</sub>d <sub>a</sub>ll AI<sub>-ass</sub>i<sub>s</sub>t<sub>e</sub>d <sub>wor</sub>k <sub>an</sub>d t<sub>a</sub>k<sub>e respons</sub>ibilit<sub>y</sub> f<sub>or</sub> th<sub>e</sub> p<sup>a</sup>p<sup>er.</sup>

## Acknowledgements

TT i<sub>s</sub> <sub>suppo</sub>rt<sub>e</sub>d b<sub>y</sub> JSPS KAKENHI Gr<sub>a</sub>nt N<sub>u</sub>mb<sub>e</sub>r JP26K21297 <sub>a</sub>nd JST BOOST Gr<sub>a</sub>nt N<sub>u</sub>mb<sub>e</sub>r JP-MJBY25D4<sub>,</sub> KY i<sub>s</sub> <sub>pa</sub>rti<sub>a</sub>ll<sub>y</sub> <sub>suppo</sub>rt<sub>e</sub>d b<sub>y</sub> JSPS KAKENHI Gr<sub>a</sub>nt N<sub>u</sub>mb<sub>e</sub>r JP24H00703<sub>.</sub>

## References

Al<sub>e</sub>kh A<sub>ga</sub>r<sub>wa</sub>l<sub>,</sub> Of<sub>e</sub>r D<sub>e</sub>k<sub>e</sub>l<sub>,</sub> <sub>a</sub>nd Lin Xi<sub>ao.</sub> O<sub>p</sub>tim<sub>a</sub>l <sub>a</sub>l<sub>go</sub>rithm<sub>s</sub> f<sub>o</sub>r <sub>o</sub>nlin<sub>e</sub> <sub>co</sub>n<sub>ve</sub>x <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n <sub>w</sub>ith m<sub>u</sub>lti-<sub>po</sub>int bandit feedback. In Proceedings of the 23rd Conference on Learning Theory, pages 28–40, 2010. 5

Kwangjun A<sup>h</sup>n and As<sup>h</sup>o<sup>k</sup> Cut<sup>k</sup>os<sup>k</sup>y. Adam wit<sup>h</sup> mode<sup>l</sup> exponentia<sup>l</sup> moving average is e<sup>f</sup>ective <sup>f</sup>or nonconvex optimization. In Advances in Neural Information Processing Systems, volume 37, pages 94909–94933, 2024<sub>.</sub> 4

Kwangjun A<sup>h</sup>n, Z<sup>h</sup>iyu Z<sup>h</sup>ang, Yunbum Koo<sup>k</sup>, and Yan Dai. Understanding Adam optimizer via on<sup>l</sup>ine <sup>l</sup>earning of updates: Adam is FTRL in disguise. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235, pages 619–640, 2024. 4

Kwan jun A<sup>h</sup>n, Ga i<sup>k</sup> Ma a<sup>k</sup> an, and As<sup>h</sup>o<sup>k</sup> Cut<sup>k</sup>os<sup>k</sup> . Genera<sup>l f</sup>ramewor<sup>k f</sup>or on<sup>l</sup>ine-to-nonconvex con version: Schedule-free SGD is also efective for nonconvex optimization. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 772–795, 2025. 4

Yossi Arjevani, Yair Carmon, Jo<sup>h</sup>n C. Duc<sup>h</sup>i, Dy<sup>l</sup>an J. Foster, Nat<sup>h</sup>an Srebro, and B<sup>l</sup>a<sup>k</sup>e Woodwort<sup>h</sup>. Lower bounds for non-convex stochastic optimization. Mathematical Programming, 199(1–2):165–214, 2023. 11

H<sub>e</sub>d<sub>y</sub> Att<sub>ouc</sub>h<sub>,</sub> Jé<sub>r</sub>ô<sub>me</sub> B<sub>o</sub>lt<sub>e, an</sub>d B<sub>enar</sub> F<sub>ux</sub> S<sub>va</sub>it<sub>er.</sub> C<sub>onvergence o</sub>f d<sub>escen</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>or sem</sub>i<sub>-a</sub>l<sub>ge</sub>b<sub>ra</sub>i<sub>c an</sub>d t<sub>ame pro</sub>bl<sub>ems:</sub> P<sub>rox</sub>i<sub>ma</sub>l <sub>a</sub>l<sub>gor</sub>ith<sub>ms,</sub> f<sub>orwar</sub>d<sub>–</sub>b<sub>ac</sub>k<sub>war</sub>d <sub>sp</sub>litti<sub>ng, an</sub>d <sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub>d G<sub>auss–</sub>S<sub>e</sub>id<sub>e</sub>l <sub>me</sub>th<sub>o</sub>d<sub>s.</sub> Mathematical Programming, 137:91–129, 2013. 3

A<sub>m</sub>i<sub>r</sub> B<sub>ec</sub>k <sub>an</sub>d M<sub>arc</sub> T<sub>e</sub>b<sub>ou</sub>ll<sub>e.</sub> A f<sub>as</sub>t it<sub>era</sub>ti<sub>ve</sub> <sub>s</sub>h<sub>r</sub>i<sub>n</sub>k<sub>age-</sub>th<sub>res</sub>h<sub>o</sub>ldi<sub>ng</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> f<sub>or</sub> li<sub>near</sub> i<sub>nverse</sub> <sub>pro</sub>bl<sub>ems.</sub> SIAM Journal on Imaging Sciences, 2(1):183–202, 2009. 3

Jérôm<sub>e</sub> B<sub>o</sub>lt<sub>e,</sub> Sh<sub>o</sub>h<sub>a</sub>m S<sub>a</sub>b<sub>ac</sub>h<sub>, a</sub>nd M<sub>a</sub>r<sub>c</sub> T<sub>e</sub>b<sub>ou</sub>ll<sub>e.</sub> Pr<sub>o</sub>xim<sub>a</sub>l <sub>a</sub>lt<sub>e</sub>rn<sub>a</sub>tin<sub>g</sub> lin<sub>ea</sub>riz<sub>e</sub>d minimiz<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r n<sub>o</sub>n<sub>co</sub>n<sub>ve</sub>x and nonsmooth problems. Mathematical Programming, 146(1–2):459–494, 2014. 3

Zi<sub>y</sub>i Ch<sub>e</sub>n<sub>,</sub> P<sub>e</sub>ir<sub>a</sub>n Y<sub>u, a</sub>nd H<sub>e</sub>n<sub>g</sub> H<sub>ua</sub>n<sub>g.</sub> Z<sub>e</sub>r<sub>o</sub>th-<sub>o</sub>rd<sub>e</sub>r m<sub>e</sub>th<sub>o</sub>d<sub>s</sub> f<sub>o</sub>r <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> n<sub>o</sub>n<sub>co</sub>n<sub>ve</sub>x n<sub>o</sub>n<sub>s</sub>m<sub>oo</sub>th <sub>co</sub>m<sub>pos</sub>it<sub>e</sub> optimization. arXiv preprint arXiv:2510.04446, 2025. 2, 3, 4, 5, 6, 9, 10, 12, 27

Ch<sub>ao</sub>-K<sub>a</sub>i Chi<sub>a</sub>n<sub>g,</sub> Ti<sub>a</sub>nb<sub>ao</sub> Y<sub>a</sub>n<sub>g,</sub> Chi<sub>a</sub>-J<sub>u</sub>n<sub>g</sub> L<sub>ee,</sub> M<sub>e</sub>hrd<sub>a</sub>d M<sub>a</sub>hd<sub>av</sub>i<sub>,</sub> Chi-J<sub>e</sub>n L<sub>u,</sub> R<sub>o</sub>n<sub>g</sub> Jin<sub>,</sub> <sub>a</sub>nd Sh<sub>e</sub>n<sub>g</sub>h<sub>uo</sub> Zh<sub>u.</sub>

Online optimization with gradual variations. In Proceedings of the 25th Annual Conference on Learning Theory, volume 23, pages 6.1–6.20, 2012. 11

Frank H. Clarke. Optimization and Nonsmooth Analysis, volume 5. Society for Industrial and Applied M<sub>a</sub>th<sub>e</sub>m<sub>a</sub>ti<sub>cs,</sub> 1990<sub>.</sub> 5

Corinna Cortes and Vladimir Vapnik. Support-vector networks. Machine Learning, 20:273–297, 1995. 2

A<sub>s</sub>h<sub>o</sub>k C<sub>u</sub>tk<sub>os</sub>k<sub>y,</sub> H<sub>a</sub>r<sub>s</sub>h M<sub>e</sub>ht<sub>a, a</sub>nd Fr<sub>a</sub>n<sub>cesco</sub> Or<sub>a</sub>b<sub>o</sub>n<sub>a.</sub> O<sub>p</sub>tim<sub>a</sub>l <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> n<sub>o</sub>n-<sub>s</sub>m<sub>oo</sub>th n<sub>o</sub>n-<sub>co</sub>n<sub>ve</sub>x <sub>op</sub>timiz<sub>a</sub> tion through online-to-non-convex conversion. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202, pages 6643–6670, 2023. 2, 3, 4, 5, 10

D<sub>a</sub>m<sub>e</sub>k D<sub>av</sub>i<sub>s a</sub>nd Dmitri<sub>y</sub> Dr<sub>usvya</sub>t<sub>s</sub>ki<sub>y.</sub> St<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> m<sub>o</sub>d<sub>e</sub>l-b<sub>ase</sub>d minimiz<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f <sub>wea</sub>kl<sub>y co</sub>n<sub>ve</sub>x f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n<sub>s.</sub> SIAM Journal on Optimization, 29(1):207–239, 2019. 4

Dame<sup>k</sup> Davis and Benjamin Grimmer. Proxima<sup>ll</sup>y guided stoc<sup>h</sup>astic subgradient met<sup>h</sup>od <sup>f</sup>or nonsmoot<sup>h</sup>, nonconvex problems. SIAM Journal on Optimization, 29(3):1908–1930, 2019. 3

J<sub>o</sub>hn D<sub>uc</sub>hi<sub>,</sub> El<sub>a</sub>d H<sub>a</sub>z<sub>a</sub>n<sub>, a</sub>nd Y<sub>o</sub>r<sub>a</sub>m Sin<sub>ge</sub>r<sub>.</sub> Ad<sub>ap</sub>ti<sub>ve su</sub>b<sub>g</sub>r<sub>a</sub>di<sub>e</sub>nt m<sub>e</sub>th<sub>o</sub>d<sub>s</sub> f<sub>o</sub>r <sub>o</sub>nlin<sub>e</sub> l<sub>ea</sub>rnin<sub>g a</sub>nd <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> optimization. Journal ofMachine Learning Research, 12(61):2121–2159, 2011. 27

Jo<sup>h</sup>n C. Duc<sup>h</sup>i, S<sup>h</sup>ai S<sup>h</sup>a<sup>l</sup>ev-S<sup>h</sup>wartz, Yoram Singer, and Ambuj Tewari. Composite objective mirror descent. In Proceedings ofthe 23rd Annual Conference on Learning Theory, pages 14–26, 2010. 3, 4, 8, 18

J<sub>o</sub>hn C<sub>.</sub> D<sub>uc</sub>hi<sub>,</sub> Mi<sub>c</sub>h<sub>ae</sub>l I<sub>.</sub> J<sub>o</sub>rd<sub>a</sub>n<sub>,</sub> M<sub>a</sub>rtin J<sub>.</sub> W<sub>a</sub>in<sub>w</sub>ri<sub>g</sub>ht<sub>, a</sub>nd Andr<sub>e</sub> Wibi<sub>so</sub>n<sub>o.</sub> O<sub>p</sub>tim<sub>a</sub>l r<sub>a</sub>t<sub>es</sub> f<sub>o</sub>r z<sub>e</sub>r<sub>o</sub>-<sub>o</sub>rd<sub>e</sub>r convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory, 61 (5):2788–2806, 2015. 5

S<sub>aee</sub>d Gh<sub>a</sub>di<sub>m</sub>i <sub>an</sub>d G<sub>uang</sub>h<sub>u</sub>i L<sub>an.</sub> St<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> fi<sub>rs</sub>t<sub>- an</sub>d <sub>zero</sub>th<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>or nonconvex s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> programming. SIAM Journal on Optimization, 23(4):2341–2368, 2013. 5

S<sub>aee</sub>d Gh<sub>a</sub>di<sub>m</sub>i <sub>an</sub>d G<sub>uang</sub>h<sub>u</sub>i L<sub>an.</sub> A<sub>cce</sub>l<sub>era</sub>t<sub>e</sub>d <sub>gra</sub>di<sub>en</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>or nonconvex non</sub>li<sub>near an</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> programming. Mathematical Programming, 156:59–99, 2016. 3

S<sub>aee</sub>d Gh<sub>a</sub>dimi<sub>,</sub> G<sub>ua</sub>n<sub>g</sub>h<sub>u</sub>i L<sub>a</sub>n<sub>, a</sub>nd H<sub>o</sub>n<sub>gc</sub>h<sub>ao</sub> Zh<sub>a</sub>n<sub>g.</sub> Mini-b<sub>a</sub>t<sub>c</sub>h <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c app</sub>r<sub>o</sub>xim<sub>a</sub>ti<sub>o</sub>n m<sub>e</sub>th<sub>o</sub>d<sub>s</sub> f<sub>o</sub>r nonconvex stochastic composite optimization. Mathematical Programming, 155(1–2):267–305, 2016. 1, 2, 3<sub>,</sub> 5<sub>,</sub> 6<sub>,</sub> 9<sub>,</sub> 11<sub>,</sub> 12<sub>,</sub> 23

A. A. Goldstein. Optimization of Lipschitz continuous functions. Mathematical Programming, 13(1):14–22, 1977<sub>.</sub> 5

Benjamin Grimmer and Zhichao Jia. Goldstein stationarity in Lipschitz constrained optimization. Optimization Letters, 19(2):425–435, 2025. 4

F<sub>e</sub>ih<sub>u</sub> H<sub>ua</sub>n<sub>g,</sub> Bin G<sub>u,</sub> Zh<sub>ouyua</sub>n H<sub>uo,</sub> S<sub>o</sub>n<sub>gca</sub>n Ch<sub>e</sub>n<sub>, a</sub>nd H<sub>e</sub>n<sub>g</sub> H<sub>ua</sub>n<sub>g.</sub> F<sub>as</sub>t<sub>e</sub>r <sub>g</sub>r<sub>a</sub>di<sub>e</sub>nt-fr<sub>ee p</sub>r<sub>o</sub>xim<sub>a</sub>l stochastic methods for nonconvex nonsmooth optimization. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pages 1503–1510, 2019. 4

F<sub>a</sub>nf<sub>a</sub>n Ji <sub>a</sub>nd Xi<sub>ao</sub>-T<sub>o</sub>n<sub>g</sub> Y<sub>ua</sub>n<sub>.</sub> D<sub>e</sub>r<sub>a</sub>nd<sub>o</sub>miz<sub>e</sub>d <sub>o</sub>nlin<sub>e</sub>-t<sub>o</sub>-n<sub>o</sub>n-<sub>co</sub>n<sub>ve</sub>x <sub>co</sub>n<sub>ve</sub>r<sub>s</sub>i<sub>o</sub>n f<sub>o</sub>r <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c wea</sub>kl<sub>y co</sub>n<sub>ve</sub>x optimization. In International Conference on Learning Representations, pages 125389–125414, 2026. 4

Z<sup>h</sup>ic<sup>h</sup>ao Jia and Benjamin Grimmer. First-order met<sup>h</sup>ods <sup>f</sup>or nonsmoot<sup>h</sup> nonconvex <sup>f</sup>unctiona<sup>l</sup> constrained optimization with or without Slater points. SIAM Journal on Optimization, 35(2):1300–1329, 2025. 4

Eh<sub>sa</sub>n K<sub>a</sub>z<sub>e</sub>mi <sub>a</sub>nd Li<sub>q</sub>i<sub>a</sub>n<sub>g</sub> W<sub>a</sub>n<sub>g.</sub> Efi<sub>c</sub>i<sub>e</sub>nt z<sub>e</sub>r<sub>o</sub>th-<sub>o</sub>rd<sub>e</sub>r <sub>p</sub>r<sub>o</sub>xim<sub>a</sub>l <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> m<sub>e</sub>th<sub>o</sub>d f<sub>o</sub>r n<sub>o</sub>n<sub>co</sub>n<sub>ve</sub>x n<sub>o</sub>n<sub>s</sub>m<sub>oo</sub>th

black-box problems. Machine Learning, 113:97–120, 2024. 4

G<sub>uy</sub> K<sub>ornows</sub>ki <sub>an</sub>d Oh<sub>a</sub>d Sh<sub>am</sub>i<sub>r.</sub> A<sub>n a</sub>l<sub>gor</sub>ith<sub>m w</sub>ith <sub>op</sub>ti<sub>ma</sub>l di<sub>mens</sub>i<sub>on-</sub>d<sub>epen</sub>d<sub>ence</sub> f<sub>or zero-or</sub>d<sub>er nonsmoo</sub>th nonconvex stochastic optimization. Journal of Machine Learning Research, 25(122):1–14, 2024. 2, 3, 4, 5, 6<sub>,</sub> 10<sub>,</sub> 11<sub>,</sub> 22

G<sub>uy</sub> K<sub>o</sub>rn<sub>ows</sub>ki<sub>,</sub> S<sub>wa</sub>ti P<sub>a</sub>dm<sub>a</sub>n<sub>a</sub>bh<sub>a</sub>n<sub>,</sub> K<sub>a</sub>i W<sub>a</sub>n<sub>g,</sub> Jimm<sub>y</sub> Zh<sub>a</sub>n<sub>g, a</sub>nd S<sub>uv</sub>rit Sr<sub>a.</sub> Fir<sub>s</sub>t-<sub>o</sub>rd<sub>e</sub>r m<sub>e</sub>th<sub>o</sub>d<sub>s</sub> f<sub>o</sub>r linearly constrained bilevel optimization. In Advances in Neural Information Processing Systems, volume 37, <sub>pages</sub> 141417<sub>–</sub>141460<sub>,</sub> 2024<sub>.</sub> 4

Al<sub>e</sub>x Krizh<sub>evs</sub>k<sub>y,</sub> Il<sub>ya</sub> S<sub>u</sub>t<sub>s</sub>k<sub>eve</sub>r<sub>, a</sub>nd G<sub>eo</sub>fr<sub>ey</sub> E<sub>.</sub> Hint<sub>o</sub>n<sub>.</sub> Im<sub>age</sub>N<sub>e</sub>t <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>o</sub>n <sub>w</sub>ith d<sub>eep co</sub>n<sub>vo</sub>l<sub>u</sub>ti<sub>o</sub>n<sub>a</sub>l neural networks. In Advances in Neural Information Processing Systems, volume 25, 2012. 2

Huan Li and Zhouchen Lin. Accelerated proximal gradient methods for nonconvex programming. In Advances in Neural Information Processing Systems, volume 28, 2015. 3

Min<sub>gy</sub>i Li <sub>a</sub>nd T<sub>a</sub>ir<sub>a</sub> T<sub>suc</sub>hi<sub>ya.</sub> M<sub>uo</sub>n <sub>w</sub>ith finit<sub>e</sub> N<sub>ew</sub>t<sub>o</sub>n–S<sub>c</sub>h<sub>u</sub>lz<sub>:</sub> Th<sub>e</sub> <sub>s</sub>m<sub>oo</sub>thin<sub>g</sub> b<sub>e</sub>n<sub>e</sub>fit in n<sub>o</sub>n<sub>s</sub>m<sub>oo</sub>th nonconvex optimization. arXiv preprint arXiv:2608.26288, 2026. 4

Qunwei Li, Yi Zhou, Yin<sub>g</sub>bin Lian<sub>g</sub>, and Pramod K. Varshne<sub>y</sub>. Conver<sub>g</sub>ence anal<sub>y</sub>sis of proximal <sub>g</sub>radient with momentum for nonconvex optimization. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pages 2111–2119, 2017. 3

Zhize Li and Jian Li. A sim<sub>p</sub>le <sub>p</sub>roximal stochastic <sub>g</sub>radient method for nonsmooth nonconvex o<sub>p</sub>timization. In Advances in Neural Information Processing Systems, volume 31, 2018. 3

Langqi Liu, Yibo Wang, and Lijun Z<sup>h</sup>ang. Hig<sup>h</sup>-probabi<sup>l</sup>ity bound <sup>f</sup>or non-smoot<sup>h</sup> non-convex stoc<sup>h</sup>astic optimization with heavy tails. In Proceedings of the 41st International Conference on Machine Learning, <sub>vo</sub>l<sub>ume</sub> 235<sub>, pages</sub> 32122<sub>–</sub>32138<sub>,</sub> 2024<sub>a.</sub> 4

Zh<sub>ua</sub>n<sub>g</sub>h<sub>ua</sub> Li<sub>u,</sub> Ch<sub>e</sub>n<sub>g</sub> Ch<sub>e</sub>n<sub>,</sub> L<sub>uo</sub> L<sub>uo, a</sub>nd Br<sub>ya</sub>n Ki<sub>a</sub>n H<sub>s</sub>i<sub>a</sub>n<sub>g</sub> L<sub>ow.</sub> Z<sub>e</sub>r<sub>o</sub>th-<sub>o</sub>rd<sub>e</sub>r m<sub>e</sub>th<sub>o</sub>d<sub>s</sub> f<sub>o</sub>r <sub>co</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d nonconvex nonsmooth stochastic optimization. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 30842–30872, 2024b. 2, 4, 6

Zijian Liu. On<sup>l</sup>ine convex optimization wit<sup>h</sup> <sup>h</sup>eavy tai<sup>l</sup>s: O<sup>l</sup>d a<sup>l</sup>gorit<sup>h</sup>ms, new regrets, and app<sup>l</sup>ications. In Proceedings of the 37th International Conference on Algorithmic Learning Theory, volume 313, pages 1<sub>–</sub>47<sub>,</sub> 2026<sub>.</sub> 4

R<sub>a</sub>h<sub>u</sub>l M<sub>a</sub>z<sub>u</sub>md<sub>e</sub>r<sub>,</sub> Tr<sub>evo</sub>r H<sub>as</sub>ti<sub>e, a</sub>nd R<sub>o</sub>b<sub>e</sub>rt Tib<sub>s</sub>hir<sub>a</sub>ni<sub>.</sub> S<sub>pec</sub>tr<sub>a</sub>l r<sub>egu</sub>l<sub>a</sub>riz<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>l<sub>go</sub>rithm<sub>s</sub> f<sub>o</sub>r l<sub>ea</sub>rnin<sub>g</sub> l<sub>a</sub>r<sub>ge</sub> incomplete matrices. Journal ofMachine Learning Research, 11(80):2287–2322, 2010. 1

Yurii Nesterov. Gradient methods for minimizing composite functions. Mathematical Programming, 140(1): 125<sub>–</sub>161<sub>,</sub> 2013<sub>.</sub> 1<sub>,</sub> 3<sub>,</sub> 5<sub>,</sub> 6<sub>,</sub> 9<sub>,</sub> 11<sub>,</sub> 23

Atsushi Nitanda. Stochastic proximal gradient descent with acceleration techniques. In Advances in Neural Information Processing Systems, volume 27, 2014. 3

Neal Parikh and Stephen Boyd. Proximal algorithms. Foundations and Trends in Optimization, 1(3):127–239, 2014<sub>.</sub> 1<sub>,</sub> 3<sub>,</sub> 5<sub>,</sub> 9

Francisco Patitucci<sub>,</sub> Ruichen Jian<sub>g,</sub> and Ar<sub>y</sub>an Mokhtari. Im<sub>p</sub>rovin<sub>g</sub> online-to-nonconvex conversion for smooth optimization via double optimism. In International Conference on Learning Representations, pages 26553<sub>–</sub>26581<sub>,</sub> 2026<sub>.</sub> 4

Nhan H. Pham, Lam M. N<sub>g</sub>u<sub>y</sub>en, Dzun<sub>g</sub> T. Phan, and Quoc Tran-Dinh. ProxSARAH: An eficient al<sub>g</sub>orithmic

framework for stochastic composite nonconvex optimization. Journal of Machine Learning Research, 21 (110):1–48, 2020. 3, 23

S<sub>py</sub>rid<sub>o</sub>n P<sub>oug</sub>k<sub>a</sub>ki<sub>o</sub>ti<sub>s</sub> <sub>a</sub>nd Di<sub>o</sub>n<sub>ys</sub>i<sub>s</sub> K<sub>a</sub>l<sub>oge</sub>ri<sub>as.</sub> A z<sub>e</sub>r<sub>o</sub>th-<sub>o</sub>rd<sub>e</sub>r <sub>p</sub>r<sub>o</sub>xim<sub>a</sub>l <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>g</sub>r<sub>a</sub>di<sub>e</sub>nt m<sub>e</sub>th<sub>o</sub>d f<sub>o</sub>r weakly convex stochastic optimization. SIAM Journal on Scientific Computing, 45(5):A2679–A2702, 2023. 4

S<sub>py</sub>rid<sub>o</sub>n P<sub>oug</sub>k<sub>a</sub>ki<sub>o</sub>ti<sub>s a</sub>nd Di<sub>o</sub>n<sub>ys</sub>i<sub>s</sub> K<sub>a</sub>l<sub>oge</sub>ri<sub>as.</sub> In<sub>e</sub>x<sub>ac</sub>t z<sub>e</sub>r<sub>o</sub>th-<sub>o</sub>rd<sub>e</sub>r n<sub>o</sub>n<sub>s</sub>m<sub>oo</sub>th <sub>a</sub>nd n<sub>o</sub>n<sub>co</sub>n<sub>ve</sub>x <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> composite optimization and applications. Journal ofNonlinear and Variational Analysis, 10(2):287–311, 2026<sub>.</sub> 2 4 11 24 25

Alexander Rakhlin and Karthik Sridharan. Online learning with predictable sequences. In Proceedings of the 26th Annual Conference on Learning Theory, volume 30, pages 993–1019, 2013. 11

S<sub>as</sub>h<sub>an</sub>k J<sub>.</sub> R<sub>e</sub>ddi S<sub>uvr</sub>it S<sub>ra</sub> B<sub>arna</sub>bá<sub>s</sub> Pó<sub>czos an</sub>d Al<sub>exan</sub>d<sub>er</sub> J<sub>.</sub> S<sub>mo</sub>l<sub>a.</sub> P<sub>rox</sub>i<sub>ma</sub>l <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>or</sub> nonsmooth nonconvex finite-sum optimization. In Advances in Neural Information Processing Systems, <sub>vo</sub>l<sub>ume</sub> 29<sub>,</sub> 2016<sub>.</sub> 3

Emr<sub>e</sub> S<sub>a</sub>hin<sub>og</sub>l<sub>u</sub> <sub>a</sub>nd Sh<sub>a</sub>hin Sh<sub>a</sub>hr<sub>a</sub>m<sub>pou</sub>r<sub>.</sub> An <sub>o</sub>nlin<sub>e</sub> <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n <sub>pe</sub>r<sub>spec</sub>ti<sub>ve</sub> <sub>o</sub>n fir<sub>s</sub>t-<sub>o</sub>rd<sub>e</sub>r <sub>a</sub>nd z<sub>e</sub>r<sub>o</sub>-<sub>o</sub>rd<sub>e</sub>r decentralized nonsmooth nonconvex stochastic optimization. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 43043–43059, 2024. 4

Emr<sub>e</sub> S<sub>a</sub>hin<sub>og</sub>l<sub>u,</sub> Y<sub>ou</sub>b<sub>a</sub>n<sub>g</sub> S<sub>u</sub>n<sub>, a</sub>nd Sh<sub>a</sub>hin Sh<sub>a</sub>hr<sub>a</sub>m<sub>pou</sub>r<sub>.</sub> Finit<sub>e</sub>-tim<sub>e a</sub>n<sub>a</sub>l<sub>ys</sub>i<sub>s o</sub>f <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> n<sub>o</sub>n<sub>co</sub>n<sub>ve</sub>x nonsmooth optimization on the Riemannian manifolds. In Advances in Neural Information Processing Systems, volume 38, pages 134166–134186, 2025. 4

Sh<sub>a</sub>i Sh<sub>a</sub>l<sub>ev</sub>-Sh<sub>wa</sub>rtz<sub>,</sub> Y<sub>o</sub>r<sub>a</sub>m Sin<sub>ge</sub>r<sub>,</sub> N<sub>a</sub>th<sub>a</sub>n Sr<sub>e</sub>br<sub>o,</sub> <sub>a</sub>nd Andr<sub>ew</sub> C<sub>o</sub>tt<sub>e</sub>r<sub>.</sub> P<sub>egasos:</sub> Prim<sub>a</sub>l <sub>es</sub>tim<sub>a</sub>t<sub>e</sub>d <sub>su</sub>bgradient solver for SVM. Mathematical Programming, 127:3–30, 2011. 2

Oh<sub>a</sub>d Sh<sub>am</sub>i<sub>r.</sub> A<sub>n op</sub>ti<sub>ma</sub>l <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> f<sub>or</sub> b<sub>an</sub>dit <sub>an</sub>d <sub>zero-or</sub>d<sub>er convex op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on w</sub>ith t<sub>wo-po</sub>i<sub>n</sub>t f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> Journal ofMachine Learning Research, 18(52):1–11, 2017. 5

Robert Tibshirani. Regression shrinkage and selection via the Lasso. Journal ofthe Royal Statistical Society: Series B (Methodological), 58(1):267–288, 1996. 1

Quoc Tran-Dinh, Nhan H. Pham, Dzun<sub>g</sub> T. Phan, and Lam M. N<sub>g</sub>u<sub>y</sub>en. A h<sub>y</sub>brid stochastic optimization framework for composite nonconvex optimization. Mathematical Programming, 191:1005–1071, 2022. 3, 23

Yif<sub>e</sub>i W<sub>a</sub>n<sub>g,</sub> J<sub>o</sub>n<sub>a</sub>th<sub>a</sub>n L<sub>aco</sub>tt<sub>e,</sub> <sub>a</sub>nd M<sub>e</sub>rt Pil<sub>a</sub>n<sub>c</sub>i<sub>.</sub> Th<sub>e</sub> hidd<sub>e</sub>n <sub>co</sub>n<sub>ve</sub>x <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n l<sub>a</sub>nd<sub>scape</sub> <sub>o</sub>f r<sub>egu</sub>l<sub>a</sub>riz<sub>e</sub>d two-layer ReLU networks: An exact characterization of optimal solutions. In International Conference on Learning Representations, 2022. 2

Zh<sub>e</sub> W<sub>ang,</sub> K<sub>a</sub>i<sub>y</sub>i Ji<sub>,</sub> Yi Zh<sub>ou,</sub> Yi<sub>ng</sub>bi<sub>n</sub> Li<sub>ang,</sub> <sub>an</sub>d V<sub>a</sub>hid T<sub>aro</sub>kh<sub>.</sub> S<sub>p</sub>id<sub>er</sub>B<sub>oos</sub>t <sub>an</sub>d <sub>momen</sub>t<sub>um:</sub> F<sub>as</sub>t<sub>er</sub> <sub>var</sub>i<sub>ance</sub> reduction algorithms. In Advances in Neural Information Processing Systems, volume 32, 2019. 3, 23

Lin Xiao and Tong Zhang. A proximal stochastic gradient method with progressive variance reduction. SIAM Journal on Optimization, 24(4):2057–2075, 2014. 3

H<sub>a</sub>ih<sub>a</sub>n Zh<sub>a</sub>n<sub>g,</sub> W<sub>e</sub>nd<sub>ao</sub> W<sub>u,</sub> Ch<sub>e</sub>nh<sub>e</sub>n<sub>g</sub> Zh<sub>a</sub>n<sub>g,</sub> Y<sub>a</sub>n<sub>y</sub>i Li<sub>,</sub> Ch<sub>u</sub>n<sub>yua</sub>n Zh<sub>e</sub>n<sub>g,</sub> C<sub>o</sub>n<sub>g</sub> F<sub>a</sub>n<sub>g,</sub> H<sub>ao</sub>x<sub>ua</sub>n Li<sub>, a</sub>nd Zhouchen Lin. Joint lower bounds for zeroth-order nonconvex optimization on Euclidean balls. arXiv preprint arXiv:2610.00275, 2026. 2

Jin<sub>g</sub>zh<sub>ao</sub> Zh<sub>a</sub>n<sub>g,</sub> H<sub>o</sub>n<sub>g</sub>zh<sub>ou</sub> Lin<sub>,</sub> St<sub>e</sub>f<sub>a</sub>ni<sub>e</sub> J<sub>ege</sub>lk<sub>a,</sub> S<sub>uv</sub>rit Sr<sub>a,</sub> <sub>a</sub>nd Ali J<sub>a</sub>db<sub>a</sub>b<sub>a</sub>i<sub>e.</sub> C<sub>o</sub>m<sub>p</sub>l<sub>e</sub>xit<sub>y</sub> <sub>o</sub>f findin<sub>g</sub>

stationary points of nonconvex nonsmooth functions. In Proceedings of the 37th International Conference on Machine Learning, volume 119, pages 11173–11182, 2020. 6

Qinzi Zhan<sub>g</sub> and Ashok Cutkosk<sub>y</sub>. Random scalin<sub>g</sub> and momentum for non-smooth non-convex optimization. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 58780–58799, 2024<sub>.</sub> 4

## A Deferred proofs of the regret bounds for COMID

We describe how com<sub>p</sub>osite objective mirror descent (COMID) and o<sub>p</sub>timistic COMID <sub>g</sub>enerate the directions $u _ { k , t }$ usin<sub>g</sub> (12) and (15), res<sub>p</sub>ectivel<sub>y</sub>, and <sub>p</sub>rove their re<sub>g</sub>ret bounds.

Proof of Theorem 4.1. Fix a block k and apply Duchi et al. (2010), in their notation, with

$$
\Omega = \mathcal { U } _ { D } ( x _ { k } ) , \qquad f _ { k , t } ( u ) = \langle g _ { k , t } , u \rangle , \qquad r _ { k } ( u ) = h ( x _ { k } + u ) - \operatorname* { m i n } _ { v \in \mathcal { U } _ { D } ( x _ { k } ) } h ( x _ { k } + v ) .
$$

Initialize $u _ { k , 1 } \in \arg \operatorname* { m i n } _ { v \in \mathcal { U } _ { D } ( x _ { k } ) } h ( x _ { k } + v )$ <sub>,</sub> <sub>so</sub> th<sub>a</sub>t $r _ { k } ( u ) \ge 0$ for all u and $r _ { k } ( u _ { k , 1 } ) = 0$ . Usin<sub>g</sub> the Euclidean Bre<sub>g</sub>man diver<sub>g</sub>ence, the COMID u<sub>p</sub>date (Duchi et al., 2010, Eq. (3)) s<sub>p</sub>ecializes to

$$
u _ { k , t + 1 } \in \underset { u \in \mathcal { U } _ { D } ( x _ { k } ) } { \arg \operatorname* { m i n } } \biggl \{ \frac { 1 } { 2 } \| u - u _ { k , t } \| ^ { 2 } + \eta ( \langle g _ { k , t } , u \rangle + h ( x _ { k } + u ) - h ( x _ { k } ) ) \biggr \} ,
$$

<sub>w</sub>h<sub>ere</sub> $\eta > 0$ i<sub>s</sub> th<sub>e</sub> <sub>s</sub>t<sub>ep</sub> <sub>s</sub>i<sub>ze.</sub> Th<sub>en,</sub> b<sub>y</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f $\ell _ { k , t }$ in (5),

$$
u _ { k , t + 1 } \in \underset { u \in \mathcal { U } _ { D } ( x _ { k } ) } { \arg \operatorname* { m i n } } \bigg \{ \frac { 1 } { 2 } \| u - u _ { k , t } \| ^ { 2 } + \eta \ell _ { k , t } ( u ) \bigg \} .\tag{12}
$$

<sup>F</sup>or an<sub>y</sub> $u \in \mathcal { U } _ { D } ( x _ { k } )$ , Theorem 2 of Duchi et al. (2010) now <sub>g</sub>ives

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u _ { k , t } ) - \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u ) = \sum _ { t = 1 } ^ { T } ( f _ { k , t } ( u _ { k , t } ) + r _ { k } ( u _ { k , t } ) - f _ { k , t } ( u ) - r _ { k } ( u ) ) } } \\ & { } & { \leq \frac { \| u - u _ { k , 1 } \| ^ { 2 } } { 2 \eta } + r _ { k } ( u _ { k , 1 } ) + \frac { \eta } { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| g _ { k , t } \| ^ { 2 } . } \end{array}\tag{13}
$$

Takin<sub>g</sub> the minimum over $u ,$ then ex<sub>p</sub>ectation<sub>,</sub> and usin<sub>g</sub> $\| u - u _ { k , 1 } \| \leq 2 D$ <sub>an</sub>d th<sub>e assume</sub>d <sub>momen</sub>t b<sub>oun</sub>d <sub>g</sub><sup>i</sup>ves

$$
\mathbb { E } \Bigg [ \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u _ { k , t } ) - \operatorname* { m i n } _ { u \in \mathcal { U } _ { D } ( x _ { k } ) } \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u ) \Bigg ] \leq \frac { 2 D ^ { 2 } } { \eta } + \frac { \eta T G ^ { 2 } } { 2 } .
$$

Choosin<sub>g</sub> $\eta = 2 D / ( G \sqrt { T } )$ <sub>ma</sub>k<sub>es</sub> th<sub>e r</sub>i<sub>g</sub>ht<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> $2 D G \sqrt { T }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub>

ProofofTheorem 5.2. Fix a block k and set $r _ { k } ( u ) : = h ( x _ { k } + u ) - h ( x _ { k } )$ <sub>.</sub> L<sub>e</sub>t $g _ { k , 0 }$ b<sub>e an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>g</sub>radient <sub>q</sub>ueried at $x _ { k }$ . For a ste<sub>p</sub> size $\eta > 0$ <sub>,</sub> initialize $v _ { k , 1 } = 0$ <sub>an</sub>d

$$
u _ { k , 1 } \in \underset { u \in \mathcal { U } _ { D } ( x _ { k } ) } { \arg \operatorname* { m i n } } \bigg \{ \frac { 1 } { 2 } \| u \| ^ { 2 } + \eta ( \langle g _ { k , 0 } , u \rangle + r _ { k } ( u ) ) \bigg \} .
$$

We use $g _ { k , t }$ <sub>as</sub> th<sub>e</sub> hi<sub>n</sub>t <sub>vec</sub>t<sub>or</sub> f<sub>or roun</sub>d $t + 1$ and a<sub>pp</sub>l<sub>y</sub> the COMID u<sub>p</sub>date (12) in the <sub>p</sub>roof of Theorem 4.1 <sub>as</sub> f<sub>o</sub>ll<sub>ows</sub>

$$
v _ { k , t + 1 } \in \underset { u \in \mathcal { U } _ { D } ( x _ { k } ) } { \arg \operatorname* { m i n } } \bigg \{ \frac { 1 } { 2 } \| u - v _ { k , t } \| ^ { 2 } + \eta \ell _ { k , t } ( u ) \bigg \} ,\tag{14}
$$

$$
u _ { k , t + 1 } \in \underset { u \in \mathcal { U } _ { D } ( x _ { k } ) } { \arg \operatorname* { m i n } } \bigg \{ \frac { 1 } { 2 } \| u - v _ { k , t + 1 } \| ^ { 2 } + \eta ( \langle g _ { k , t } , u \rangle + r _ { k } ( u ) ) \bigg \} .\tag{15}
$$

<sup>F</sup>or an<sub>y</sub> $u \in \mathcal { U } _ { D } ( x _ { k } )$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } & { \ell _ { k , t } ( u _ { k , t } ) - \ell _ { k , t } ( u ) = \ell _ { k , t } ( v _ { k , t + 1 } ) - \ell _ { k , t } ( u ) + \ell _ { k , t } ( u _ { k , t } ) - \ell _ { k , t } ( v _ { k , t + 1 } ) } \\ & { \qquad = \ell _ { k , t } ( v _ { k , t + 1 } ) - \ell _ { k , t } ( u ) + \langle g _ { k , t } , u _ { k , t } - v _ { k , t + 1 } \rangle + r _ { k } ( u _ { k , t } ) - r _ { k } ( v _ { k , t + 1 } ) } \\ & { \qquad = \ell _ { k , t } ( v _ { k , t + 1 } ) - \ell _ { k , t } ( u ) + \langle g _ { k , t - 1 } , u _ { k , t } - v _ { k , t + 1 } \rangle } \\ & { \qquad + r _ { k } ( u _ { k , t } ) - r _ { k } ( v _ { k , t + 1 } ) + \langle g _ { k , t } - g _ { k , t - 1 } , u _ { k , t } - v _ { k , t + 1 } \rangle . } \end{array}\tag{16}
$$

T<sub>o</sub> b<sub>ou</sub>nd thi<sub>s</sub> dif<sub>e</sub>r<sub>e</sub>n<sub>ce,</sub> <sub>we</sub> d<sub>e</sub>ri<sub>ve</sub> th<sub>e</sub> <sub>op</sub>tim<sub>a</sub>lit<sub>y</sub> <sub>co</sub>nditi<sub>o</sub>n<sub>s.</sub> F<sub>o</sub>r <sub>a</sub>n<sub>y</sub> $w \in \mathcal { U } _ { D } ( x _ { k } )$ <sub>an</sub>d $0 < \lambda \leq 1$ <sub>,</sub> l<sub>e</sub>t $w _ { \lambda } = v _ { k , t + 1 } + \lambda ( w - v _ { k , t + 1 } ) \in \mathcal { U } _ { D } ( x _ { k } )$ <sub>.</sub> Th<sub>en,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } & { 0 \leq \displaystyle \frac 1 2 \| w _ { \lambda } - v _ { k , t } \| ^ { 2 } - \frac 1 2 \| v _ { k , t + 1 } - v _ { k , t } \| ^ { 2 } + \eta ( \ell _ { k , t } ( w _ { \lambda } ) - \ell _ { k , t } ( v _ { k , t + 1 } ) ) } \\ & { \quad = \lambda \langle v _ { k , t + 1 } - v _ { k , t } , w - v _ { k , t + 1 } \rangle + \displaystyle \frac { \lambda ^ { 2 } } { 2 } \| w - v _ { k , t + 1 } \| ^ { 2 } + \eta ( \ell _ { k , t } ( w _ { \lambda } ) - \ell _ { k , t } ( v _ { k , t + 1 } ) ) } \\ & { \quad \leq \lambda \langle v _ { k , t + 1 } - v _ { k , t } , w - v _ { k , t + 1 } \rangle + \displaystyle \frac { \lambda ^ { 2 } } { 2 } \| w - v _ { k , t + 1 } \| ^ { 2 } + \eta \lambda ( \ell _ { k , t } ( w ) - \ell _ { k , t } ( v _ { k , t + 1 } ) ) , } \end{array}
$$

where the first inequalit<sub>y</sub> follows from (14) and the last inequalit<sub>y</sub> follows from convexit<sub>y</sub> of $\ell _ { k , t }$ <sub>,</sub> which <sub>g</sub>ives $\ell _ { k , t } ( w _ { \lambda } ) \leq ( 1 - \lambda ) \ell _ { k , t } ( v _ { k , t + 1 } ) + \lambda \ell _ { k , t } ( w )$ . Dividing by λ and letting $\lambda \downarrow 0 .$ <sub>, we o</sub>bt<sub>a</sub>in

$$
0 \leq \langle v _ { k , t + 1 } - v _ { k , t } , w - v _ { k , t + 1 } \rangle + \eta ( \ell _ { k , t } ( w ) - \ell _ { k , t } ( v _ { k , t + 1 } ) ) .\tag{17}
$$

Similarl<sub>y</sub>, a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> this ar<sub>g</sub>ument to (15) <sub>g</sub>ives

$$
0 \leq \langle u _ { k , t } - v _ { k , t } , w - u _ { k , t } \rangle + \eta ( \langle g _ { k , t - 1 } , w - u _ { k , t } \rangle + r _ { k } ( w ) - r _ { k } ( u _ { k , t } ) ) .\tag{18}
$$

Thus, (17) and (18) <sub>g</sub>ive

$$
\ell _ { k , t } ( v _ { k , t + 1 } ) - \ell _ { k , t } ( u ) \leq \frac { \langle v _ { k , t } - v _ { k , t + 1 } , v _ { k , t + 1 } - u \rangle } { \eta } = \frac { \| u - v _ { k , t } \| ^ { 2 } - \| u - v _ { k , t + 1 } \| ^ { 2 } - \| v _ { k , t + 1 } - v _ { k , t } \| ^ { 2 } } { 2 \eta }
$$

<sub>an</sub>d

$$
\begin{array} { r l } & { \langle g _ { k , t - 1 } , u _ { k , t } - v _ { k , t + 1 } \rangle + r _ { k } ( u _ { k , t } ) - r _ { k } ( v _ { k , t + 1 } ) \leq \frac { \langle v _ { k , t } - u _ { k , t } , u _ { k , t } - v _ { k , t + 1 } \rangle } { \eta } } \\ & { \qquad = \frac { \| v _ { k , t + 1 } - v _ { k , t } \| ^ { 2 } - \| u _ { k , t } - v _ { k , t + 1 } \| ^ { 2 } - \| u _ { k , t } - v _ { k , t } \| ^ { 2 } } { 2 \eta } . } \end{array}
$$

Substitutin<sub>g</sub> these bounds into (16) <sub>g</sub>ives

$$
\begin{array} { r l r } { \ell _ { k , t } ( u _ { k , t } ) - \ell _ { k , t } ( u ) \le \frac { \| u - v _ { k , t } \| ^ { 2 } - \| u - v _ { k , t + 1 } \| ^ { 2 } - \| v _ { k , t + 1 } - v _ { k , t } \| ^ { 2 } } { 2 \eta } } \\ & { } & { \qquad + \frac { \| v _ { k , t + 1 } - v _ { k , t } \| ^ { 2 } - \| u _ { k , t } - v _ { k , t + 1 } \| ^ { 2 } - \| u _ { k , t } - v _ { k , t } \| ^ { 2 } } { 2 \eta } } \\ & { } & { \qquad + \left. g _ { k , t } - g _ { k , t - 1 } , u _ { k , t } - v _ { k , t + 1 } \right. } \\ & { } & { \qquad \le \frac { \| u - v _ { k , t } \| ^ { 2 } - \| u - v _ { k , t + 1 } \| ^ { 2 } } { 2 \eta } - \frac { \| u _ { k , t } - v _ { k , t + 1 } \| ^ { 2 } + \| u _ { k , t } - v _ { k , t } \| ^ { 2 } } { 2 \eta } } \\ & { } & { \qquad + \frac { \eta } { 2 } \| g _ { k , t } - g _ { k , t - 1 } \| ^ { 2 } + \frac { \| u _ { k , t } - v _ { k , t + 1 } \| ^ { 2 } } { 2 \eta } \qquad \mathrm { ( b y ~ Y o u n g ' s ~ i n e q u a l i t y ) } } \end{array}
$$

$$
\begin{array} { l } { \displaystyle = \frac { \| \boldsymbol { u } - \boldsymbol { v } _ { k , t } \| ^ { 2 } - \| \boldsymbol { u } - \boldsymbol { v } _ { k , t + 1 } \| ^ { 2 } } { 2 \eta } - \frac { \| \boldsymbol { u } _ { k , t } - \boldsymbol { v } _ { k , t } \| ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \| \boldsymbol { g } _ { k , t } - \boldsymbol { g } _ { k , t - 1 } \| ^ { 2 } } \\ { \displaystyle \le \frac { \| \boldsymbol { u } - \boldsymbol { v } _ { k , t } \| ^ { 2 } - \| \boldsymbol { u } - \boldsymbol { v } _ { k , t + 1 } \| ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \| \boldsymbol { g } _ { k , t } - \boldsymbol { g } _ { k , t - 1 } \| ^ { 2 } . } \end{array}
$$

Summing over t and using $\lVert u \rVert \leq D$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u _ { k , t } ) - \sum _ { t = 1 } ^ { T } \ell _ { k , t } ( u ) \leq \frac { D ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \sum _ { t = 1 } ^ { T } \lVert g _ { k , t } - g _ { k , t - 1 } \rVert ^ { 2 } .\tag{19}
$$

With the con<sub>v</sub>ention $y _ { k , 0 } = x _ { k }$ <sub>, we</sub> h<sub>ave</sub> f<sub>or a</sub>ll $t \geq 1$

$$
\begin{array} { r l } & { \mathbb { E } [ \| g _ { k , t } - g _ { k , t - 1 } \| ^ { 2 } ] } \\ & { = \mathbb { E } \big [ \| ( g _ { k , t } - \nabla f ( y _ { k , t } ) ) + ( \nabla f ( y _ { k , t } ) - \nabla f ( y _ { k , t - 1 } ) ) + ( \nabla f ( y _ { k , t - 1 } ) - g _ { k , t - 1 } ) \| ^ { 2 } \big ] } \\ & { \leq 3 \mathbb { E } \big [ \| g _ { k , t } - \nabla f ( y _ { k , t } ) \| ^ { 2 } \big ] + 3 \mathbb { E } \big [ \| \nabla f ( y _ { k , t } ) - \nabla f ( y _ { k , t - 1 } ) \| ^ { 2 } \big ] + 3 \mathbb { E } \big [ \| g _ { k , t - 1 } - \nabla f ( y _ { k , t - 1 } ) \| ^ { 2 } \big ] } \\ & { \leq 6 \sigma ^ { 2 } + 3 L _ { 1 } ^ { 2 } \mathbb { E } \big [ \| y _ { k , t } - y _ { k , t - 1 } \| ^ { 2 } \big ] } \\ & { \leq 1 2 L _ { 1 } ^ { 2 } D ^ { 2 } + 6 \sigma ^ { 2 } \leq 1 2 ( L _ { 1 } D + \sigma ) ^ { 2 } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e secon</sub>d i<sub>nequa</sub>lit<sub>y uses</sub> th<sub>e</sub> $L _ { 1 }$ -smoothness of f and condition (ii) of Assumption 3.1.

This bound and (19) <sub>g</sub>ive

$$
R _ { T } \leq \frac { D ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \sum _ { t = 1 } ^ { T } \mathbb { E } \big [ \| g _ { k , t } - g _ { k , t - 1 } \| ^ { 2 } \big ] \leq \frac { D ^ { 2 } } { 2 \eta } + 6 \eta T ( L _ { 1 } D + \sigma ) ^ { 2 } .
$$

Choosin<sub>g</sub> $\eta = D / ( 2 \sqrt { 3 T } ( L _ { 1 } D + \sigma ) )$ then <sub>g</sub>ives $R _ { T } \leq 2 \sqrt { 3 } ( L _ { 1 } D + \sigma ) D \sqrt { T } .$

## B Deferred proofs for Section 4

W<sub>e prove</sub> L<sub>emmas</sub> 4<sub>.</sub>2 <sub>an</sub>d 4<sub>.</sub>5 <sub>an</sub>d Th<sub>eorem</sub> 4<sub>.</sub>6 i<sub>n</sub> thi<sub>s sec</sub>ti<sub>on.</sub>

Lemma B.1. For $x \in$ dom h, $g \in \mathbb { R } ^ { d } , \gamma > 0 .$ , and every $s \in \partial h ( x ) , \| G _ { \gamma h } ( x , g ) \| \leq \| g + s \|$

Proof of Lemma B.1. By (2), $\mathrm { p r o x } _ { \gamma h } ( x - \gamma g )$ minimizes $\begin{array} { r } { h ( v ) + \frac { 1 } { 2 \gamma } \| v - ( x - \gamma g ) \| ^ { 2 } } \end{array}$ over $v \in \mathbb { R } ^ { d }$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub> its first-order o<sub>p</sub>timalit<sub>y</sub> condition is

$$
0 \in \partial h \bigl ( \operatorname { p r o x } _ { \gamma h } ( x - \gamma g ) \bigr ) + \frac { 1 } { \gamma } \bigl ( \operatorname { p r o x } _ { \gamma h } ( x - \gamma g ) - x + \gamma g \bigr ) .
$$

B<sub>y</sub> (3), the second term is $- G _ { \gamma h } ( x , g ) + g$ <sub>.</sub> H<sub>ence</sub>

$$
G _ { \gamma h } ( x , g ) - g \in \partial h \big ( \mathrm { p r o x } _ { \gamma h } ( x - \gamma g ) \big ) .
$$

Sin<sub>ce</sub> $s \in \partial h ( x )$ <sub>,</sub> the two sub<sub>g</sub>radient ine<sub>q</sub>ualities are

$$
\begin{array} { r l } & { h \big ( \mathrm { p r o x } _ { \gamma h } ( x - \gamma g ) \big ) \geq h ( x ) + \big \langle s , \mathrm { p r o x } _ { \gamma h } ( x - \gamma g ) - x \big \rangle , } \\ & { \qquad h ( x ) \geq h \big ( \mathrm { p r o x } _ { \gamma h } ( x - \gamma g ) \big ) + \big \langle G _ { \gamma h } ( x , g ) - g , x - \mathrm { p r o x } _ { \gamma h } ( x - \gamma g ) \big \rangle . } \end{array}
$$

Addin<sub>g</sub> th<sub>e</sub>m <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r l } & { 0 \leq \big \langle G _ { \gamma h } ( x , g ) - g - s , \mathrm { p r o x } _ { \gamma h } ( x - \gamma g ) - x \big \rangle } \\ & { \quad = - \gamma \big \langle G _ { \gamma h } ( x , g ) - g - s , G _ { \gamma h } ( x , g ) \big \rangle , } \end{array}
$$

where the equalit<sub>y</sub> follows from (3). Therefore,

$$
\| G _ { \gamma h } ( x , g ) \| ^ { 2 } \leq \langle g + s , G _ { \gamma h } ( x , g ) \rangle \leq \| g + s \| \| G _ { \gamma h } ( x , g ) \| .
$$

$\mathrm { I f } \ \| G _ { \gamma h } ( x , g ) \| > 0$ <sub>,</sub> division b<sub>y</sub> this norm com<sub>p</sub>letes the <sub>p</sub>roof. Otherwise<sub>,</sub> the claim is immediate. □

Lemma B.2. Fix $x \in$ dom h, $\begin{array} { r } { g \in \mathbb { R } ^ { d } , } \end{array}$ , and $D > 0$ . Let $u ^ { * }$ be a minimizer of $\langle g , u \rangle + h ( x + u ) - h ( x )$ over $u \in \mathcal { U } _ { D } ( x )$ , and let $y = x + u ^ { * }$ . Then, for any $\gamma > 0 , \| G _ { \gamma h } ( y , g ) \| \leq \mathcal { C } _ { D } ( x , g )$

ProofofLemma B.2. The Lagrangian for $\| u \| ^ { 2 } / 2 \le D ^ { 2 } / 2$ <sub>g</sub><sup>i</sup>ves some $\lambda \geq 0$ <sub>suc</sub>h th<sub>a</sub>t

$$
0 \in g + \partial h ( x + u ^ { * } ) + \lambda u ^ { * } , \qquad \lambda \big ( \| u ^ { * } \| ^ { 2 } - D ^ { 2 } \big ) = 0 .
$$

Sin<sub>ce</sub> $y = x + u ^ { * }$ <sub>,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $s ^ { * } \in \partial h ( y )$ <sub>suc</sub>h th<sub>a</sub>t

$$
g + s ^ { * } + \lambda ( y - x ) = 0 , \qquad \lambda ( \lVert y - x \rVert - D ) = 0 .
$$

These conditions im<sub>p</sub>l<sub>y</sub>

$$
\langle g + s ^ { * } , y - x \rangle = - \lambda \| y - x \| ^ { 2 } = - D \lambda \| y - x \| = - D \| g + s ^ { * } \| .\tag{20}
$$

The sub<sub>g</sub>radient ine<sub>q</sub>ualit<sub>y</sub> $h ( y ) - h ( x ) \leq \langle s ^ { * } , y - x \rangle$ <sub>an</sub>d $u ^ { * } = y - x$ <sub>g</sub><sup>i</sup>ve

$$
\begin{array} { l } { \displaystyle \mathcal { C } _ { D } ( \boldsymbol { x } , \boldsymbol { g } ) = - \frac { 1 } { D } ( \langle \boldsymbol { g } , \boldsymbol { u } ^ { * } \rangle + h ( \boldsymbol { y } ) - h ( \boldsymbol { x } ) ) } \\ { \displaystyle \ge - \frac { 1 } { D } \langle \boldsymbol { g } + \boldsymbol { s } ^ { * } , \boldsymbol { y } - \boldsymbol { x } \rangle } \\ { \displaystyle = \| \boldsymbol { g } + \boldsymbol { s } ^ { * } \| } \\ { \displaystyle \ge \| \boldsymbol { G } _ { \gamma h } ( \boldsymbol { y } , \boldsymbol { g } ) \| , } \end{array}\tag{b<sub>y</sub> (20)}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> L<sub>emma</sub> B<sub>.</sub>1<sub>.</sub>

Proof of Lemma 4.2. Since $\| y - x \| \leq D$ <sub>an</sub>d $\nu \geq 2 D , \mathbb { B } ( x , D ) \subseteq \mathbb { B } ( y , 2 D ) \subseteq \mathbb { B } ( y , \nu )$ <sub>an</sub>d $g \in \partial _ { D } f ( x ) \subseteq$ $\partial _ { \nu } f ( \boldsymbol { y } )$ . Then, b<sub>y</sub> (3),

$$
\begin{array} { l l } { \displaystyle \| G _ { \gamma h } ( y , g ) - G _ { \gamma h } ( y , \widehat g ) \| = \frac { 1 } { \gamma } \big \| \mathrm { p r o x } _ { \gamma h } ( y - \gamma \widehat g ) - \mathrm { p r o x } _ { \gamma h } ( y - \gamma g ) \big \| } \\ { \displaystyle \leq \frac { 1 } { \gamma } \| ( y - \gamma \widehat g ) - ( y - \gamma g ) \| } \\ { \displaystyle = \| g - \widehat g \| } \end{array}\tag{21}
$$

where the inequality follows from the nonexpansiveness of prox<sub>γh</sub>.

Th<sub>ere</sub>f<sub>ore,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } { \underset { v \in \partial _ { \nu } f ( y ) } { \operatorname* { m i n } } \lVert G _ { \gamma h } ( y , v ) \rVert \leq \lVert G _ { \gamma h } ( y , g ) \rVert } & { } \\ { } & { \leq \lVert G _ { \gamma h } ( y , \widehat g ) \rVert + \lVert g - \widehat g \rVert } \\ { } & { \leq { \mathcal C } _ { D } ( x , \widehat g ) + \lVert g - \widehat g \rVert . } \end{array}\tag{b<sub>y</sub> (21)}
$$

(b<sub>y</sub> Lemma B.2)

Proof of Lemma 4.5. By Jensen’s and the Cauchy–Schwarz inequalities and Assumption 3.2, for all $x , y ,$ , we h<sub>ave</sub>

$$
| f ( x ) - f ( y ) | \leq \mathbb { E } _ { \xi } [ | F ( x , \xi ) - F ( y , \xi ) | ] \leq \mathbb { E } _ { \xi } [ L ( \xi ) ] \| x - y \| \leq L _ { 0 } \| x - y \| .
$$

The claims in (10) and (11) follow directl<sub>y</sub> from Kornowski and Shamir (2024, Lemma 7).

ProofofTheorem 4.6. We use the proof of Theorem 4.4 for $f _ { \rho } + h$ <sub>w</sub>ith t<sub>a</sub>r<sub>ge</sub>t G<sub>o</sub>ld<sub>s</sub>t<sub>e</sub>in r<sub>a</sub>di<sub>us</sub> $2 D$ <sub>,</sub> th<sub>en use</sub> Lemma 3.4. By Lemma 4.5, f is $L _ { 0 } – \mathbf { L }$ i<sub>p</sub>schitz and $| f _ { \rho } ( x ) - f ( x ) | \leq \rho L _ { 0 }$ for every x. Thus

$$
\Delta _ { \rho } \leq \Delta + 2 \rho L _ { 0 } \leq \Delta + \delta L _ { 0 } .
$$

With $G = \sigma = G _ { \mathrm { z o } }$ f<sub>rom</sub> L<sub>emma</sub> 4<sub>.</sub>5<sub>,</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ces</sub> i<sub>n</sub> Th<sub>eorem</sub> 4<sub>.</sub>6 <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>

$$
T \geq \frac { 4 ( 2 G _ { \mathrm { z o } } + G _ { \mathrm { z o } } ) ^ { 2 } } { \varepsilon ^ { 2 } } , K \geq \frac { 2 \Delta _ { \rho } } { D \varepsilon } .
$$

C<sub>onsequen</sub>tl<sub>y,</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 4<sub>.</sub>4 <sub>g</sub>i<sub>ves</sub>

$$
\mathbb { E } \bigg [ \operatorname* { m i n } _ { \substack { g \in \partial _ { 2 D } f _ { \rho } ( y _ { R } ) } } \big \| G _ { \gamma h } ( y _ { R } , g ) \big \| \bigg ] \leq \varepsilon
$$

<sup>f</sup>or ever<sub>y</sub> $\gamma > 0$ <sub>.</sub> Sin<sub>ce</sub> $\rho + 2 D = \delta \quad$ <sub>,</sub> L<sub>e</sub>mm<sub>a</sub> 3<sub>.</sub>4 im<sub>p</sub>li<sub>es</sub> $\partial _ { 2 D } f _ { \rho } ( y _ { R } ) \subseteq \partial _ { \delta } f ( y _ { R } )$ <sub>,</sub> <sub>p</sub>r<sub>ov</sub>in<sub>g</sub> th<sub>e</sub> PGSP <sub>gua</sub>r<sub>a</sub>nt<sub>ee.</sub> Th<sub>e</sub> r<sub>esu</sub>ltin<sub>g</sub> n<sub>u</sub>mb<sub>e</sub>r <sub>o</sub>f f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n-<sub>va</sub>l<sub>ue</sub> <sub>que</sub>ri<sub>es</sub> i<sub>s</sub>

$$
N = 2 K T = O \bigg ( \frac { d L _ { 0 } ^ { 2 } ( \Delta + \delta L _ { 0 } ) } { \delta \varepsilon ^ { 3 } } \bigg ) .
$$

## C Comparison and deferred proofs for Section 5

## C.1 Comparison with existing bounds

Table 2 compares Theorems 5.3 and 5.4 with existing oracle complexity bounds for finding an ε-stationary <sub>p</sub>oint of $\Phi = f + h$ <sub>w</sub>ith $L _ { 1 } .$ <sub>-smoo</sub>th $f ,$ that is, a point x with $\mathbb { E } [ \| G _ { \gamma h } ( x , \nabla f ( x ) ) \| ] \le \varepsilon$ <sub>.</sub> B<sub>oun</sub>d<sub>s s</sub>t<sub>a</sub>t<sub>e</sub>d f<sub>or</sub> $\mathbb { E } \big [ \| G _ { \gamma h } ( x , \nabla f ( x ) ) \| ^ { 2 } \big ] \ \leq \ \varepsilon ^ { 2 }$ in the ori<sub>g</sub>inal <sub>p</sub>a<sub>p</sub>ers im<sub>p</sub>l<sub>y</sub> this condition b<sub>y</sub> Jensen’s ine<sub>q</sub>ualit<sub>y</sub>. The <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> b<sub>oun</sub>d<sub>s</sub> h<sub>o</sub>ld f<sub>or</sub> th<sub>e</sub> <sub>s</sub>t<sub>ep</sub> <sub>s</sub>i<sub>ze</sub> $\gamma$ <sub>use</sub>d b<sub>y</sub> th<sub>e</sub> <sub>respec</sub>ti<sub>ve</sub> <sub>me</sub>th<sub>o</sub>d<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>our</sub> b<sub>oun</sub>d<sub>s</sub> h<sub>o</sub>ld f<sub>or</sub> <sub>every</sub> $\gamma > 0$ <sub>.</sub> M<sub>ean-square</sub>d <sub>smoo</sub>th<sub>ness re</sub>f<sub>ers</sub> t<sub>o</sub> th<sub>e con</sub>diti<sub>on</sub> $\mathbb { E } _ { \xi } \big [ \| \nabla F ( x , \xi ) - \nabla F ( y , \xi ) \| ^ { 2 } \big ] \le L ^ { 2 } \| x - y \| ^ { 2 }$ f<sub>or</sub> <sub>a</sub>ll $x , y \in \mathbb { R } ^ { d }$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s s</sub>t<sub>ronger</sub> th<sub>an</sub> th<sub>e smoo</sub>th<sub>ness o</sub>f $f$

Table 2: Oracle complexity for finding an ε-stationary point of $\Phi = f + h$ <sub>w</sub>ith $L _ { 1 }$ <sub>-smoo</sub>th $f .$ Th<sub>e</sub> b<sub>oun</sub>d<sub>s</sub> show the dependence on d and ε.
<table><tr><td>Reference</td><td>Oracle</td><td>Access</td><td>Assumption on the oracle</td><td>Complexity</td></tr><tr><td>Nesterov (2013); Ghadimi et al. (2016)</td><td>Deterministic</td><td>First-order</td><td></td><td> $O ( \varepsilon ^ { - 2 } )$ </td></tr><tr><td>Ghadimi et al. (2016)</td><td>Stochastic</td><td>First-order</td><td>Bounded variance</td><td> $O ( \varepsilon ^ { - 4 } )$ </td></tr><tr><td>Wang et al. (2019); Pham et al. (2020);</td><td>Stochastic</td><td>First-order</td><td>Bounded variance,</td><td> $O ( \varepsilon ^ { - 3 } )$ </td></tr><tr><td>Tran-Dinh et al. (2022) Ghadimi et al. (2016)</td><td>Stochastic</td><td>Zeroth-order</td><td>mean-squared smoothness Each  $F ( \cdot , \xi )$  is smooth</td><td> $O ( d \varepsilon ^ { - 4 } )$ </td></tr><tr><td>This work (Theorem 5.3)</td><td>Deterministic</td><td>First-order</td><td></td><td> $O ( \varepsilon ^ { - 2 } )$ </td></tr><tr><td>This work (Theorem 5.3)</td><td>Stochastic</td><td>First-order</td><td>Bounded variance</td><td> $O ( \varepsilon ^ { - 4 } )$ </td></tr><tr><td>This work (Theorem 5.4)</td><td>Stochastic</td><td>Zeroth-order</td><td>Each  $F ( \cdot , \xi )$  is Lipschitz</td><td> $O ( d \varepsilon ^ { - 4 } )$ </td></tr></table>

## C.2 Deferred proofs

W<sub>e prove</sub> L<sub>emma</sub> 5<sub>.</sub>1 <sub>an</sub>d th<sub>e orac</sub>l<sub>e comp</sub>l<sub>ex</sub>it<sub>y resu</sub>lt<sub>s</sub> f<sub>rom</sub> S<sub>ec</sub>ti<sub>on</sub> $5 .$

ProofofLemma 5.1. For every $g \in \partial _ { \delta } f ( x )$ , (21) and the $L _ { 1 }$ <sub>-smoo</sub>th<sub>ness</sub> <sub>o</sub>f $f$ <sub>g</sub><sup>i</sup>ve

$$
\begin{array} { r } { \| G _ { \gamma h } ( x , \nabla f ( x ) ) \| \le \| G _ { \gamma h } ( x , g ) \| + \| \nabla f ( x ) - g \| \le \| G _ { \gamma h } ( x , g ) \| + L _ { 1 } \delta . } \end{array}
$$

Takin<sub>g</sub> the minimum over $g \in \partial _ { \delta } f ( x )$ <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub>

Proof of Theorem 5.3. Combining Lemmas 4.2 and 4.3, Lemma 5.1 with $\delta = 2 D$ , <sup>f</sup>or ever<sub>y</sub> $\gamma > 0$

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| G _ { \gamma h } ( y _ { R } , \nabla f ( y _ { R } ) ) \| ] \leq \frac { \Delta } { D K } + \frac { R _ { T } } { D T } + \frac { \sigma } { \sqrt { T } } + 2 L _ { 1 } D } } \\ & { } & { \leq \frac { \Delta } { D K } + 2 L _ { 1 } D + \frac { 2 \sqrt { 3 } ( L _ { 1 } D + \sigma ) + \sigma } { \sqrt { T } } } \\ & { } & { \leq \frac { \Delta } { D K } + 6 L _ { 1 } D + \frac { 5 \sigma } { \sqrt { T } } . } \end{array}\tag{b<sub>y</sub> Theorem 5.2}
$$

B<sub>y</sub> th<sub>e pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> in Th<sub>eo</sub>r<sub>e</sub>m 5<sub>.</sub>3<sub>,</sub> $6 L _ { 1 } D = \varepsilon / 2 , \Delta / ( D K ) \le \varepsilon / 4$ <sub>,</sub> <sub>an</sub>d $5 \sigma / \sqrt { T } \ \leq \ \varepsilon / 4$ <sub>g</sub><sup>i</sup>ves $\mathbb { E } [ \| G _ { \gamma h } ( y _ { R } , \nabla f ( y _ { R } ) ) \| ] \le \varepsilon$ . The number of stochastic <sub>g</sub>radient <sub>q</sub>ueries is

$$
\begin{array} { r } { N = K ( T + 1 ) = O \bigg ( \operatorname* { m a x } \bigg \{ \cfrac { L _ { 1 } \Delta } { \varepsilon ^ { 2 } } , \cfrac { \sigma ^ { 2 } } { \varepsilon ^ { 2 } } , \cfrac { L _ { 1 } \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \bigg \} \bigg ) . } \end{array}
$$

Proof of Theorem 5.4. By $\nabla f _ { \rho } ( x ) = \mathbb { E } _ { z \sim \mathsf { U n i f } ( \mathbb { B } ( 0 , 1 ) ) } [ \nabla f ( x + \rho z ) ]$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } & { \| \nabla f _ { \rho } ( x ) - \nabla f _ { \rho } ( y ) \| \leq \mathbb { E } _ { z } [ \| \nabla f ( x + \rho z ) - \nabla f ( y + \rho z ) \| ] \leq L _ { 1 } \| x - y \| , } \\ & { \| \nabla f _ { \rho } ( x ) - \nabla f ( x ) \| \leq \mathbb { E } _ { z } [ \| \nabla f ( x + \rho z ) - \nabla f ( x ) \| ] \leq L _ { 1 } \rho , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> b<sub>o</sub>th b<sub>oun</sub>d<sub>s use</sub> th<sub>e</sub> $L _ { 1 }$ <sub>-smoo</sub>th<sub>ness o</sub>f $f$ <sub>an</sub>d th<sub>e secon</sub>d <sub>uses</sub> $\| z \| \leq 1$ <sub>.</sub> Sin<sub>ce</sub> $\mathbb { E } _ { z } [ z ] = 0$ <sub>,</sub> th<sub>e</sub> <sub>qua</sub>dr<sub>a</sub>ti<sub>c</sub> T<sub>ay</sub>l<sub>o</sub>r b<sub>ou</sub>nd <sub>g</sub>i<sub>ves</sub>

$$
| f _ { \rho } ( x ) - f ( x ) | = | \mathbb { E } _ { z } [ f ( x + \rho z ) - f ( x ) - \rho \langle \nabla f ( x ) , z \rangle ] | \leq \frac { L _ { 1 } \rho ^ { 2 } } { 2 } \mathbb { E } _ { z } \big [ \| z \| ^ { 2 } \big ] \leq \frac { L _ { 1 } \rho ^ { 2 } } { 2 } .
$$

Thus the initial <sub>g</sub>a<sub>p</sub> satisfies

$$
\Delta _ { \rho } \leq \Delta + L _ { 1 } \rho ^ { 2 } = \Delta + \frac { \varepsilon ^ { 2 } } { 4 L _ { 1 } } .
$$

With $\sigma = G _ { \mathrm { z o } }$ f<sub>rom</sub> L<sub>emma</sub> 4<sub>.</sub>5<sub>,</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ces</sub> i<sub>n</sub> Th<sub>eorem</sub> 5<sub>.</sub>4 <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>

$$
T \geq \frac { 1 6 0 0 G _ { \mathrm { z o } } ^ { 2 } } { \varepsilon ^ { 2 } } , \qquad K \geq \frac { 1 9 2 L _ { 1 } \Delta } { \varepsilon ^ { 2 } } + 4 8 \geq \frac { 1 9 2 L _ { 1 } \Delta _ { \rho } } { \varepsilon ^ { 2 } } .
$$

C<sub>o</sub>n<sub>seque</sub>ntl<sub>y,</sub> th<sub>e</sub> <sub>p</sub>r<sub>oo</sub>f <sub>o</sub>f Th<sub>eo</sub>r<sub>e</sub>m 5<sub>.</sub>3 <sub>g</sub>i<sub>ves,</sub> f<sub>o</sub>r <sub>eve</sub>r<sub>y</sub> $\gamma > 0$

$$
\begin{array} { r l r } & {  { \mathbb { E } [ \| G _ { \gamma h } ( y _ { R } , \nabla f _ { \rho } ( y _ { R } ) ) \| ] \le \frac { \Delta _ { \rho } } { D K } + 6 L _ { 1 } D + \frac { 5 G _ { \mathrm { z o } } } { \sqrt { T } } } } \\ & { } & { \le \displaystyle \frac { \varepsilon } { 8 } + \frac { \varepsilon } { 4 } + \frac { \varepsilon } { 8 } = \frac { \varepsilon } { 2 } . \qquad } \end{array}
$$

Usin<sub>g</sub> (21) and the <sub>g</sub>radient bias bound above, we obtain

$$
\begin{array} { r l } & { \mathbb { E } [ \| G _ { \gamma h } ( y _ { R } , \nabla f ( y _ { R } ) ) \| ] \le \mathbb { E } [ \| G _ { \gamma h } ( y _ { R } , \nabla f _ { \rho } ( y _ { R } ) ) \| ] + \mathbb { E } [ \| \nabla f _ { \rho } ( y _ { R } ) - \nabla f ( y _ { R } ) \| ] } \\ & { \qquad \le \frac { \varepsilon } { 2 } + L _ { 1 } \rho = \varepsilon . } \end{array}
$$

E<sub>ac</sub>h bl<sub>oc</sub>k <sub>uses</sub> $2 ( T + 1 )$ function-value <sub>q</sub>ueries<sub>,</sub> so

$$
\begin{array} { r } { N = 2 K ( T + 1 ) \le 2 \bigg ( 4 9 + \frac { 1 9 2 L _ { 1 } \Delta } { \varepsilon ^ { 2 } } \bigg ) \bigg ( 2 + \frac { 1 6 0 0 G _ { \mathrm { z o } } ^ { 2 } } { \varepsilon ^ { 2 } } \bigg ) } \\ { = O \bigg ( \operatorname* { m a x } \bigg \{ \frac { L _ { 1 } \Delta } { \varepsilon ^ { 2 } } , \frac { d L _ { 0 } ^ { 2 } } { \varepsilon ^ { 2 } } , \frac { d L _ { 0 } ^ { 2 } L _ { 1 } \Delta } { \varepsilon ^ { 4 } } \bigg \} \bigg ) , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t <sub>equa</sub>lit<sub>y</sub> <sub>uses</sub> $G _ { \mathrm { z o } } ^ { 2 } = 1 6 \sqrt { 2 \pi } d L _ { 0 } ^ { 2 }$ <sub>.</sub> Thi<sub>s</sub> <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub>

## D Relation between Moreau-envelope stationarity and PGSP

L<sub>e</sub>t $L _ { \delta } > 0$ b<sub>e</sub> <sub>a</sub> Li<sub>psc</sub>hitz <sub>co</sub>n<sub>s</sub>t<sub>a</sub>nt <sub>o</sub>f $\nabla f _ { \delta }$ <sub>.</sub> F<sub>or</sub> $0 < \gamma < 1 / L _ { \delta }$ <sub>,</sub> th<sub>e</sub> M<sub>oreau</sub> <sub>enve</sub>l<sub>ope</sub> <sub>o</sub>f $\Phi _ { \delta } = f _ { \delta } + h$ is

$$
e _ { \gamma } \Phi _ { \delta } ( x ) : = \operatorname* { m i n } _ { z \in \mathbb { R } ^ { d } } \biggl \{ \Phi _ { \delta } ( z ) + \frac { \| z - x \| ^ { 2 } } { 2 \gamma } \biggr \} .
$$

Followin<sub>g</sub> Pou<sub>g</sub>kakiotis and Kalo<sub>g</sub>erias (2026, Definition 3.1), a <sub>p</sub>oint $x \in$ dom $h$ is an ε-stationary point of $\Phi _ { \delta }$ in th<sub>e</sub> M<sub>o</sub>r<sub>eau</sub>-<sub>e</sub>n<sub>ve</sub>l<sub>ope se</sub>n<sub>se</sub> if $\| \nabla e _ { \gamma } \Phi _ { \delta } ( x ) \| \le \varepsilon$

Proposition D.1. For $0 < \gamma < 1 / L _ { \delta }$ and $x \in$ dom $h ,$

$$
( 1 - \gamma L _ { \delta } ) \| \nabla e _ { \gamma } \Phi _ { \delta } ( x ) \| \le \| G _ { \gamma h } ( x , \nabla f _ { \delta } ( x ) ) \| \le ( 1 + \gamma L _ { \delta } ) \| \nabla e _ { \gamma } \Phi _ { \delta } ( x ) \| ,\tag{22}
$$

$$
\operatorname* { m i n } _ { g \in \partial _ { \delta } f ( x ) } \| G _ { \gamma h } ( x , g ) \| \leq ( 1 + \gamma L _ { \delta } ) \| \nabla e _ { \gamma } \Phi _ { \delta } ( x ) \| .\tag{23}
$$

Thus, every ε-stationary point of $\Phi _ { \delta }$ in the Moreau-envelope sense is a $( \gamma , \delta , ( 1 + \gamma L _ { \delta } ) \varepsilon ) – P G S P$ of the original objective $\Phi = f + h$

Proof. The $\begin{array} { r } { L _ { \delta ^ { - } } \mathbf { I } } \end{array}$ Li<sub>p</sub>schitz continuit<sub>y</sub> of $\nabla f _ { \delta }$ <sub>a</sub>nd <sub>co</sub>n<sub>ve</sub>xit<sub>y</sub> <sub>o</sub>f $h$ im<sub>p</sub>l<sub>y</sub> that $\Phi _ { \delta }$ is $L _ { \delta }$ <sub>-wea</sub>kl<sub>y</sub> <sub>convex.</sub> Th<sub>us,</sub> th<sub>e</sub> minimization definin<sub>g</sub> $e _ { \gamma } \Phi _ { \delta } ( x )$ is stron<sub>g</sub>l<sub>y</sub> convex and has a uni<sub>q</sub>ue minimizer $z ,$ <sub>w</sub>ith

$$
\nabla e _ { \gamma } \Phi _ { \delta } ( x ) = \frac { x - z } { \gamma } .
$$

The optimality condition for z is

$$
0 \in \nabla f _ { \delta } ( z ) + \partial h ( z ) + \frac { z - x } { \gamma } = \partial h ( z ) + \frac { z - ( x - \gamma \nabla f _ { \delta } ( z ) ) } { \gamma } .
$$

Hence, b<sub>y</sub> (2),

$$
z = \mathrm { p r o x } _ { \gamma h } ( \boldsymbol { x } - \gamma \nabla f _ { \delta } ( z ) ) .
$$

Usin<sub>g</sub> (3) and (21) and the $L _ { \delta }$ -Li<sub>p</sub>schitz continuit<sub>y</sub> of $\nabla f _ { \delta }$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>in

$$
\begin{array} { r l } { \| G _ { \gamma h } ( x , \nabla f _ { \delta } ( x ) ) - \nabla e _ { \gamma } \Phi _ { \delta } ( x ) \| = \| G _ { \gamma h } ( x , \nabla f _ { \delta } ( x ) ) - G _ { \gamma h } ( x , \nabla f _ { \delta } ( z ) ) \| } & { } \\ { \le \| \nabla f _ { \delta } ( x ) - \nabla f _ { \delta } ( z ) \| } & { } \\ { \le L _ { \delta } \| x - z \| = \gamma L _ { \delta } \| \nabla e _ { \gamma } \Phi _ { \delta } ( x ) \| . } \end{array}
$$

Thus, (22) follows from the trian<sub>g</sub>le inequalities, and (23) follows from $\nabla f _ { \delta } ( x ) \in \partial _ { \delta } f ( x )$ b<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> 3<sub>.</sub>4<sub>.</sub>

For exact sam<sub>p</sub>led function values, Pou<sub>g</sub>kakiotis and Kalo<sub>g</sub>erias (2026, Theorem 4.1) attain $\mathbb { E } \big [ \| \nabla e _ { \gamma } \Phi _ { \delta } ( X ) \| ^ { 2 } \big ] \leq$ $\varepsilon ^ { 2 }$ for their output X using $O ( d ^ { 3 / 2 } \delta ^ { - 1 } \varepsilon ^ { - 4 } )$ function-value <sub>q</sub>ueries<sub>,</sub> with $L _ { \delta } = c L _ { 0 } \sqrt { d } / \delta$ <sub>an</sub>d $\gamma = ( 2 L _ { \delta } ) ^ { - 1 }$ <sub>w</sub>h<sub>ere</sub> $c > 0$ is a uni<sub>v</sub>ersal constant. Since $\gamma L _ { \delta } = 1 / 2$ , (23) and Jensen’s inequalit<sub>y</sub> <sub>g</sub>ive

$$
\begin{array} { r l } & { \mathbb { E } \bigg [ \underset { g \in \partial _ { \delta } f ( X ) } { \operatorname* { m i n } } \| G _ { \gamma h } ( X , g ) \| \bigg ] \leq \frac { 3 } { 2 } \mathbb { E } [ \| \nabla e _ { \gamma } \Phi _ { \delta } ( X ) \| ] } \\ & { \qquad \leq \frac { 3 } { 2 } \sqrt { \mathbb { E } [ \| \nabla e _ { \gamma } \Phi _ { \delta } ( X ) \| ^ { 2 } ] } \leq \frac { 3 } { 2 } \varepsilon . } \end{array}
$$

Th<sub>us,</sub> th<sub>e same num</sub>b<sub>er o</sub>f f<sub>unc</sub>ti<sub>on-va</sub>l<sub>ue quer</sub>i<sub>es y</sub>i<sub>e</sub>ld<sub>s</sub> $\iota \left( \gamma , \delta , 3 \varepsilon / 2 \right) \mathrm { - P G S P }$

## E Experimental details

Thi<sub>s sec</sub>ti<sub>o</sub>n <sub>p</sub>r<sub>ov</sub>id<sub>es</sub> th<sub>e e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>a</sub>l d<sub>e</sub>t<sub>a</sub>il<sub>s o</sub>f S<sub>ec</sub>ti<sub>o</sub>n 6<sub>.</sub>

Problem setting. For the experiments, we use absolute loss for ReLU regression and ramp loss with $n = 5 1 2$ <sub>an</sub>d $d \in \{ 1 6 , 2 5 6 , 1 0 2 4 \}$ <sub>.</sub> Th<sub>e</sub> l<sub>osses an</sub>d <sub>common regu</sub>l<sub>ar</sub>i<sub>zer are</sub>

$$
f _ { \mathrm { R e L U } } ( W ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Biggl | \sum _ { j = 1 } ^ { \sqrt { d } } v _ { j } \operatorname* { m a x } \{ 0 , \langle W _ { j } , a _ { i } \rangle \} - b _ { i } \Biggr | ,
$$

$$
f _ { \mathrm { R a m p } } ( x ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , 1 - b _ { i } \langle a _ { i } , x \rangle \} \} ,
$$

$$
h ( x ) = \frac { 0 . 0 2 } { \sqrt { d } } \lVert x \rVert _ { 1 } + \frac { 0 . 0 1 } { 2 } \lVert x \rVert ^ { 2 } + \mathbb { 1 } _ { [ - 0 . 5 , 0 . 5 ] ^ { d } } ( x ) .
$$

![](images/3b2c8cd03761910c4f15d44870682746a5f2790d5f8a08f04ea11351901cc18b.jpg)  
Fi<sub>g</sub>ure 3: Stationarit<sub>y</sub> measure $\begin{array} { r } { \operatorname* { m i n } _ { g \in \partial _ { \delta } f ( x ) } \| G _ { \gamma h } ( x , g ) | } \end{array}$ (left) and com<sub>p</sub>osite objective values (ri<sub>g</sub>ht) for ram<sub>p</sub> loss at d = 16 (top), d = 256 (middle), and $d = 1 0 2 4$ (bottom). Curves and shaded bands show means and <sub>one s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on over</sub> t<sub>en</sub> t<sub>r</sub>i<sub>a</sub>l<sub>s.</sub>

Bot<sup>h</sup> prob<sup>l</sup>ems <sup>h</sup>ave objective $\Phi = f + h$ . Both sample losses are 1-Lipschitz. The indicator function $\mathbb { 1 } _ { C }$ t<sub>a</sub>k<sub>es</sub> the value 0 on C and +∞ outside C. For absolute loss for ReLU regression, $x = \operatorname { v e c } ( W )$ <sub>w</sub>ith $W \in \mathbb { R } ^ { \sqrt { d } \times \sqrt { d } }$ <sub>an</sub>d fi<sub>xe</sub>d <sub>ou</sub>t<sub>pu</sub>t <sub>we</sub>i<sub>g</sub>ht<sub>s</sub> $v _ { j } = ( - 1 ) ^ { j - 1 } / d ^ { 1 / 4 }$ <sub>, w</sub>ith<sub>ou</sub>t bi<sub>ases.</sub>

For data <sub>g</sub>eneration<sub>,</sub> $a _ { i }$ are normalized Gaussian vectors with norm at most 1. We draw the entries of $\bar { W }$ <sub>an</sub>d w¯ uniformly from $[ - 0 . 8 , 0 . 8 ]$ , then zero each entry of W<sup>¯</sup> independently with probability $1 / 2$ <sub>an</sub>d <sub>a ran</sub>d<sub>om</sub>l<sub>y</sub> chosen half of the entries of w¯. For absolute loss for ReLU regression, we set $\begin{array} { r } { b _ { i } = \sum _ { j = 1 } ^ { \sqrt { d } } v _ { j } } \end{array}$ max $\{ 0 , \langle \bar { W } _ { j } , a _ { i } \rangle \} +$ $\xi _ { i } ,$ <sub>, w</sub>h<sub>ere</sub> $\xi _ { i }$ are independent and uniform on [−0.02, 0.02]. For ramp loss, we set $b _ { i } = \mathrm { s i g n } ( \langle a _ { i } , \bar { w } \rangle )$ <sub>an</sub>d independently flip each sign with probability 0.1. Each dataset is fixed during optimization. Initial points are sampled by normalizing Gaussian vectors to norm 0.5. We evaluate Composite O2NC at its current anchor <sub>a</sub>ft<sub>er</sub> <sub>eac</sub>h bl<sub>oc</sub>k <sub>an</sub>d Ch<sub>en</sub>’<sub>s</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>a</sub>t th<sub>e</sub>i<sub>r</sub> l<sub>a</sub>t<sub>es</sub>t <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>d it<sub>era</sub>t<sub>es.</sub> W<sub>e</sub> <sub>use</sub> <sub>a</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>upper</sub> b<sub>oun</sub>d <sub>on</sub> the stationarity measure mi $\mathrm { a } _ { g \in \partial _ { \delta } f ( x ) } \| G _ { \gamma h } ( x , g ) \|$ <sub>w</sub>ith $\gamma = 1$ and t<sup>h</sup>e composite objective va<sup>l</sup>ue as eva<sup>l</sup>uation <sub>measures.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>quer</sub>i<sub>es</sub> <sub>are</sub> <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d f<sub>rom</sub> th<sub>e</sub> <sub>query</sub> <sub>coun</sub>t<sub>s.</sub>

Algorithm parameters. We fix $\delta = 0 . 0 1$ <sub>an</sub>d <sub>se</sub>t th<sub>e</sub> bl<sub>oc</sub>k l<sub>eng</sub>th <sub>o</sub>f C<sub>omposite</sub> O2NC t<sub>o</sub> $T = 2 5 0 0$ <sub>.</sub> F<sub>or</sub> Ch<sub>en</sub>’<sub>s me</sub>th<sub>o</sub>d<sub>s,</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>s</sub>i<sub>zes are</sub> $B = 2 5 0 0$ f<sub>or</sub> G1 <sub>an</sub>d $B _ { 0 } = 2 5 0 0 , B _ { 1 } = 5 0$ f<sub>or</sub> G2<sub>, w</sub>ith <sub>re</sub>f<sub>res</sub>h <sub>per</sub>i<sub>o</sub>d $p = 5 0$ <sub>.</sub> F<sub>o</sub>r C<sub>o</sub>mp<sub>os</sub>ite O2NC<sub>, we se</sub>t $D = \rho = \delta / 3$ . We use an ada<sub>p</sub>tive ste<sub>p</sub> size like AdaGrad (Duchi et al., 2011) based on the COMID re<sub>g</sub>ret bound in (13),

$$
\eta _ { k , t } = \frac { \sqrt { 2 } D } { \sqrt { 1 + \sum _ { s = 1 } ^ { t } \lVert g _ { k , s } \rVert ^ { 2 } } } .
$$

This ste<sub>p</sub> size <sub>g</sub>ives $R _ { T } = O ( D G { \sqrt { T } } )$ <sub>,</sub> retainin<sub>g</sub> the order of the function-value <sub>q</sub>uer<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> in The-<sub>orem</sub> 4<sub>.</sub>6<sub>.</sub> F<sub>or</sub> Ch<sub>en</sub>’<sub>s me</sub>th<sub>o</sub>d<sub>s, we se</sub>t $G = c = 1$ <sub>an</sub>d <sub>use</sub> th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>s</sub>t<sub>ep s</sub>i<sub>zes</sub> $\begin{array} { r } { \gamma _ { \mathrm { G 1 } } ~ = ~ \frac { \delta } { \sqrt { d } } } \end{array}$ <sub>an</sub>d $\begin{array} { r } { \gamma _ { \mathrm { G 2 } } = \frac { \delta } { 2 ( d + \sqrt { d } ) } } \end{array}$ from Chen et al. (2025, Theorems 2 and 3). The ex<sub>p</sub>eriments were conducted usin<sub>g</sub> P<sub>y</sub>thon 3<sub>.</sub>12 <sub>on</sub> <sub>an</sub> A<sub>pp</sub>l<sub>e</sub> M1 P<sub>ro</sub> <sub>w</sub>ith 8 CPU <sub>cores</sub> <sub>an</sub>d 16 GB <sub>o</sub>f <sub>memory.</sub> F<sub>or</sub> <sub>ramp</sub> l<sub>oss,</sub> Fi<sub>gure</sub> 3 lik<sub>ew</sub>i<sub>se</sub> <sub>s</sub>h<sub>ows</sub> <sup>l</sup>ower mean composite objective va<sup>l</sup>ues <sup>f</sup>or Composite O2NC over t<sup>h</sup>e course o<sup>f</sup> optimization at a<sup>ll</sup> t<sup>h</sup>ree di<sub>mens</sub>i<sub>ons.</sub> It<sub>s mean s</sub>t<sub>a</sub>ti<sub>onar</sub>it<sub>y measure a</sub>l<sub>so</sub> b<sub>ecomes</sub> l<sub>ower</sub> th<sub>an</sub> th<sub>ose o</sub>f b<sub>o</sub>th Ch<sub>en me</sub>th<sub>o</sub>d<sub>s as op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>p</sub>rocee<sup>d</sup>s.

Computational cost and limitations. In our algorithm, the online learner solves convex subproblems at each round. The u<sub>p</sub>date rules are <sub>g</sub>iven b<sub>y</sub> (12) and (15) with fixed learnin<sub>g</sub> rates or b<sub>y</sub> their variants with <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> l<sub>ea</sub>rnin<sub>g</sub> r<sub>a</sub>t<sub>es.</sub> C<sub>o</sub>m<sub>pa</sub>r<sub>e</sub>d <sub>w</sub>ith minib<sub>a</sub>t<sub>c</sub>h m<sub>e</sub>th<sub>o</sub>d<sub>s,</sub> <sub>ou</sub>r <sub>a</sub>l<sub>go</sub>rithm in<sub>cu</sub>r<sub>s</sub> <sub>co</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>ove</sub>rh<sub>ea</sub>d from solvin<sub>g</sub> convex sub<sub>p</sub>roblems more fre<sub>q</sub>uentl<sub>y</sub>. Reducin<sub>g</sub> com<sub>p</sub>utational costs and tunin<sub>g</sub> learnin<sub>g</sub> rates for <sub>p</sub>ractical use are interestin<sub>g</sub> directions. Since our main focus is im<sub>p</sub>rovin<sub>g</sub> oracle com<sub>p</sub>lexit<sub>y,</sub> we leave th<sub>em</sub> f<sub>or</sub> f<sub>u</sub>t<sub>ure wor</sub>k<sub>.</sub>