RESEARCH PREPRINT

# Exact Limits of Random Projections for Preserving Geometry: Distance Recovery, Nearest-Neighbor Rankings, and Covariance Shape in Gaussian Models

Pi<sub>y</sub>us<sup>h</sup> Sao

C<sub>omputer</sub> S<sub>c</sub>i<sub>ence an</sub>d M<sub>at</sub>h<sub>emat</sub>i<sub>cs</sub> Di<sub>v</sub>i<sub>s</sub>i<sub>on,</sub> O<sub>a</sub>k Rid<sub>ge</sub> N<sub>at</sub>i<sub>ona</sub>l L<sub>a</sub>b<sub>oratory,</sub> O<sub>a</sub>k Rid<sub>ge,</sub> T<sub>ennessee</sub> 37830<sub>,</sub> USA <sup>\*</sup>Corres<sub>p</sub>on<sup>d</sup>in<sub>g</sub> aut<sup>h</sup>or. Emai<sup>l</sup>: sao<sub>p</sub><sup>k</sup>@orn<sup>l</sup>.<sub>g</sub>ov

## Abstract

T<sup>h</sup>e Jo<sup>h</sup>nson–Lin<sup>d</sup>enstrauss (JL) <sup>l</sup>emma guarantees t<sup>h</sup>at a ran<sup>d</sup>om projection o<sup>f</sup> n points to $m = O ( \varepsilon ^ { - 2 } \log n )$ <sup>d</sup>imensions <sub>p</sub>reserves <sub>p</sub>airwise s<sub>q</sub>uare<sup>d d</sup>istances wit<sup>h</sup>in re<sup>l</sup>ative error ε wit<sup>h h</sup>i<sub>g</sub><sup>h</sup> <sub>p</sub>ro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>it<sub>y</sub>, an<sup>d</sup> t<sup>h</sup>is <sup>d</sup>imension <sub>or</sub>d<sub>er</sub> i<sub>s asymptot</sub>i<sub>ca</sub>ll<sub>y opt</sub>i<sub>ma</sub>l<sub>.</sub> I<sub>n</sub> hi<sub>g</sub>h di<sub>mens</sub>i<sub>ons,</sub> h<sub>owever,</sub> di<sub>stances concentrate aroun</sub>d <sub>a</sub> b<sub>ase</sub>li<sub>ne w</sub>hil<sub>e</sub> <sup>k</sup>ey geometr<sup>i</sup>c <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on <sup>li</sup>es <sup>i</sup>n muc<sup>h</sup> sma<sup>ll</sup>er <sup>fl</sup>uctuat<sup>i</sup>ons. We s<sup>h</sup>ow t<sup>h</sup>at t<sup>h</sup>e J<sup>L b</sup>oun<sup>d</sup> can t<sup>h</sup>ere<sup>f</sup>ore <sup>b</sup>e <sub>un</sub>i<sub>n</sub>f<sub>ormat</sub>i<sub>ve</sub> <sub>a</sub>b<sub>out</sub> <sub>reta</sub>i<sub>ne</sub>d <sub>geometry:</sub> <sub>an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> G<sub>auss</sub>i<sub>an</sub> <sub>rep</sub>l<sub>acement</sub> <sub>map</sub> <sub>can</sub> <sub>sat</sub>i<sub>s</sub>f<sub>y</sub> i<sub>t</sub> <sub>even</sub> <sub>t</sub>h<sub>oug</sub>h <sub>t</sub>h<sub>e rep</sub>l<sub>acement c</sub>l<sub>ou</sub>d i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent o</sub>f <sub>t</sub>h<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l d<sub>ata.</sub>

We then ask how well an<sub>y</sub> decoder can recover a featuref(D) of a squared distance D from a linear sketch. U<sub>n</sub>d<sub>er</sub> <sub>square</sub>d<sub>-error</sub> l<sub>oss,</sub> <sub>t</sub>h<sub>e</sub> <sub>opt</sub>i<sub>ma</sub>l d<sub>eco</sub>d<sub>er</sub> i<sub>s</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>expectat</sub>i<sub>on,</sub> <sub>so</sub> <sub>recovery</sub> d<sub>e</sub>fi<sub>nes</sub> <sub>a</sub> li<sub>near</sub> <sub>operator</sub> <sub>w</sub>h<sub>ose</sub> <sub>s</sub>i<sub>ngu</sub>l<sub>ar</sub> <sub>va</sub>l<sub>ues</sub> <sub>quant</sub>if<sub>y</sub> f<sub>eature</sub> <sub>recovery.</sub> F<sub>or</sub> i<sub>sotrop</sub>i<sub>c</sub> G<sub>auss</sub>i<sub>an</sub> d<sub>ata</sub> $\scriptstyle ( \sum \mathbf { \sigma } _ { } = { \dot { \sigma } } ^ { 2 } I _ { d } )$ <sub>,</sub> <sub>we</sub> di<sub>agona</sub>li<sub>ze</sub> <sub>t</sub>hi<sub>s</sub> operator in closed form. For fixed k with $m , d - m  \infty ,$ , its kth sin<sub>g</sub>ular value satisfies $\ell _ { k } \approx ( m / d ) ^ { \bar { k } / 2 }$

This yields three sharp consequences. A rank-m sketch retains at most an m/d fraction of the variance of any feature ofone s uared distance. Ifm → ∞ and $m / d \to 0 ,$ <sub>t</sub>h<sub>e ex ecte</sub>d K<sub>en</sub>d<sub>a</sub>ll <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> i<sub>s</sub> $\frac { 2 } { \pi } \sqrt { m / d } ( 1 + o ( 1 ) ) ;$ f<sub>or</sub> <sup>fi</sup>xe<sup>d</sup> q, nearest-neig<sup>hb</sup>or agreement ten<sup>d</sup>s to 1/q. Yet one projection can satis<sup>f</sup>y t<sup>h</sup>e JL <sup>b</sup>oun<sup>d</sup> w<sup>h</sup>i<sup>l</sup>e mean Ken<sup>d</sup>a<sup>ll</sup> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>van</sub>i<sub>s</sub>h<sub>es</sub> <sub>w</sub>h<sub>en</sub> l<sub>og</sub> $n \ll m \ll d .$ Af<sub>ter remov</sub>i<sub>ng sca</sub>l<sub>e,</sub> H<sub>aar-average</sub>d <sub>reta</sub>i<sub>ne</sub>d <sub>covar</sub>i<sub>ance-s</sub>h<sub>ape</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> i<sub>s</sub> $( m / d ) ^ { 2 }$ . <sup>Th</sup>us J<sup>L di</sup>stance preservat<sup>i</sup>on <sup>d</sup>oes not quant<sup>if</sup>y t<sup>h</sup>e geometry ava<sup>il</sup>a<sup>bl</sup>e <sup>f</sup>or compar<sup>i</sup>son or i<sub>n</sub>f<sub>erence.</sub>

Keywords: random projection; Johnson–Lindenstrauss lemma; distance recovery; nearest-neighbor ranking; maximal correlation; covariance s<sup>h</sup>a<sub>p</sub>e

MSC Codes: 60D05; 62H12; 68W20

## 1. Introduction

Hi<sub>g</sub>h<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l d<sub>ata</sub> <sub>p</sub>i<sub>pe</sub>li<sub>nes</sub> <sub>rout</sub>i<sub>ne</sub>l<sub>y</sub> <sub>rep</sub>l<sub>ace</sub> <sub>raw</sub> d<sub>ata</sub> <sub>w</sub>i<sub>t</sub>h <sub>compresse</sub>d <sub>representat</sub>i<sub>ons.</sub> A<sub>pprox</sub>i<sub>mate-</sub> <sub>nearest-ne</sub>i<sub>g</sub>hb<sub>or</sub> <sub>systems</sub> <sub>generate</sub> <sub>can</sub>did<sub>ates</sub> i<sub>n</sub> <sub>a</sub> <sub>compresse</sub>d d<sub>oma</sub>i<sub>n</sub> <sub>an</sub>d <sub>reran</sub>k <sub>t</sub>h<sub>em</sub> <sub>w</sub>i<sub>t</sub>h <sub>r</sub>i<sub>c</sub>h<sub>er</sub> <sub>co</sub>d<sub>es</sub> or full-<sub>p</sub>recision distances [1<sub>,</sub> 2]. Sin<sub>g</sub>le-cell methods summarize hi<sub>g</sub>h-dimensional measurements in l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>v</sub>i<sub>sua</sub>li<sub>zat</sub>i<sub>ons, even as</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s warn t</sub>h<sub>at t</sub>h<sub>e recovere</sub>d <sub>or</sub>d<sub>er</sub>i<sub>ngs can</sub> b<sub>e unsta</sub>bl<sub>e or</sub> distorted [3<sub>,</sub> 4<sub>,</sub> 5]. The decisions this forces on the <sub>p</sub>ractitioner—when to rerank<sub>,</sub> how far to com<sub>p</sub>ress<sub>,</sub> how <sub>many</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>s</sub>k<sub>etc</sub>h<sub>es</sub> <sub>to</sub> d<sub>raw—are</sub> <sub>current</sub>l<sub>y</sub> <sub>ma</sub>d<sub>e</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y,</sub> b<sub>ecause</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>guarantees</sub> <sub>may</sub> <sub>not</sub> <sub>capture</sub> <sub>t</sub>h<sub>e</sub> <sub>tas</sub>k<sub>-re</sub>l<sub>evant</sub> <sub>not</sub>i<sub>on</sub> <sub>o</sub>f <sub>qua</sub>li<sub>ty,</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub>i<sub>r</sub> <sub>worst-case</sub> b<sub>oun</sub>d<sub>s</sub> <sub>can</sub> b<sub>e</sub> <sub>too</sub> <sub>conservat</sub>i<sub>ve</sub> <sub>to</sub> <sub>g</sub>uide <sub>p</sub>arameter selection [6]. The <sub>g</sub>oal of this <sub>p</sub>a<sub>p</sub>er is to <sub>q</sub>uantif<sub>y</sub> that retained <sub>g</sub>eometr<sub>y</sub> for the random <sub>p</sub>rojection<sub>,</sub> the com<sub>p</sub>ressed re<sub>p</sub>resentation with the stron<sub>g</sub>est formal <sub>g</sub>uarantees [7<sub>,</sub> 8].

The <sub>g</sub>uarantee behind random <sub>p</sub>rojection is the Johnson–Lindenstrauss (JL) lemma [9]: for finite sets, a <sub>su</sub>i<sub>ta</sub>bl<sub>e</sub> <sub>map</sub> i<sub>nto</sub> di<sub>mens</sub>i<sub>on</sub> $m = { \dot { O ( \varepsilon ^ { - 2 } \log n ) } }$ preserves every pairwise squared distance among n points to re<sup>l</sup>ative error ε. For an in<sup>d</sup>exe<sup>d</sup> sam<sub>p</sub><sup>l</sup>e $x _ { 1 } , \ldots , x _ { n }$ , we say that a map T satisfies the $\varepsilon - J L$ bound when

$$
\left( 1 - \varepsilon \right) \left\| x _ { i } - x _ { j } \right\| ^ { 2 } \leq \left\| T ( x _ { i } ) - T ( x _ { j } ) \right\| ^ { 2 } \leq \left( 1 + \varepsilon \right) \left\| x _ { i } - x _ { j } \right\| ^ { 2 } \qquad { \mathrm { f o r ~ e v e r y ~ } } i < j .
$$

If the sample or map is random, we call the occurrence of this bound the JL event.

For concentrated high-dimensional data, concentration ofmeasure causes the pairwise squared distances controlled by the JL bound to be dominated by a common population mean, which we call the baseline. C<sub>onsequent</sub>l<sub>y,</sub> <sub>t</sub>h<sub>ese</sub> b<sub>oun</sub>d<sub>s</sub> b<sub>ecome</sub> l<sub>ess</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>ve</sub> <sub>a</sub>b<sub>out</sub> <sub>spec</sub>ifi<sub>c</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> di<sub>stances,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>nstea</sub>d <sub>appear</sub> as smaller centered fluctuations around this baseline. Downstream tasks depend on this fine scale: which di<sub>stance</sub> i<sub>s</sub> <sub>sma</sub>ll<sub>er,</sub> <sub>an</sub>d h<sub>ow</sub> f<sub>ar</sub> <sub>a</sub> <sub>po</sub>i<sub>nt</sub> di<sub>str</sub>ib<sub>ut</sub>i<sub>on</sub> d<sub>eparts</sub> f<sub>rom</sub> <sub>sp</sub>h<sub>er</sub>i<sub>c</sub>i<sub>ty.</sub> T<sub>wo</sub> <sub>quest</sub>i<sub>ons</sub> f<sub>o</sub>ll<sub>ow,</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> answers ne<sup>i</sup>t<sup>h</sup>er. <sup>D</sup>oes t<sup>h</sup>e J<sup>L</sup> event <sup>i</sup>tse<sup>lf</sup> cert<sup>if</sup>y t<sup>h</sup>at t<sup>h</sup>e geometry carry<sup>i</sup>ng t<sup>hi</sup>s <sup>fi</sup>ne sca<sup>l</sup>e surv<sup>i</sup>ve<sup>d</sup> the sketch? And how much of the fluctuations can any decoder recover from a rank-m sketch?

Th<sub>e</sub> <sub>structure</sub> <sub>o</sub>f <sub>pr</sub>i<sub>or</sub> <sub>wor</sub>k i<sub>n</sub>di<sub>cates</sub> <sub>w</sub>h<sub>y</sub> b<sub>ot</sub>h <sub>quest</sub>i<sub>ons</sub> <sub>are</sub> <sub>open.</sub> L<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> <sub>s</sub>h<sub>ow</sub> <sub>t</sub>h<sub>at</sub> <sub>t</sub>h<sub>e</sub> dimension order in the JL lemma is essentiall<sub>y</sub> unim<sub>p</sub>rovable: Alon [10] showed that near-e<sub>q</sub>uidistant <sub>p</sub>oint sets alread<sub>y</sub> force nearl<sub>y</sub> the same dimension order for ever<sub>y</sub> ma<sub>p,</sub> linear or not<sub>,</sub> and Larsen and Nelson [11] s<sup>h</sup>arpene<sup>d</sup> t<sup>hi</sup>s <sup>l</sup>ower <sup>b</sup>oun<sup>d</sup> to matc<sup>h</sup> t<sup>h</sup>e J<sup>L di</sup>mens<sup>i</sup>on or<sup>d</sup>er <sup>f</sup>or worst-case construct<sup>i</sup>ons. <sup>B</sup>ecause t<sup>h</sup>e l<sub>emma</sub> <sub>atta</sub>i<sub>ns</sub> <sub>t</sub>h<sub>e</sub> <sub>opt</sub>i<sub>ma</sub>l di<sub>mens</sub>i<sub>on</sub> <sub>or</sub>d<sub>er</sub> f<sub>or</sub> <sub>t</sub>h<sub>e</sub> <sub>guarantee</sub> i<sub>t</sub> <sub>states,</sub> <sub>t</sub>h<sub>e</sub> <sub>gap</sub> <sub>cannot</sub> b<sub>e</sub> <sub>c</sub>l<sub>ose</sub>d b<sub>y</sub> <sub>s</sub>h<sub>arpen</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> l<sub>emma</sub> i<sub>tse</sub>lf<sub>;</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>veness</sub> <sub>a</sub>b<sub>out</sub> <sub>concentrate</sub>d d<sub>ata</sub> <sub>requ</sub>i<sub>res</sub> <sub>a</sub> dif<sub>erent</sub> <sub>cr</sub>i<sub>ter</sub>i<sub>on.</sub> Th<sub>e</sub> li<sub>terature</sub> <sub>on</sub> concentration addresses the ori<sub>g</sub>inal s<sub>p</sub>ace rather than the sketch: Be<sub>y</sub>er et al. [12] <sub>g</sub>ave a suficient condition for distance concentration<sub>,</sub> and the Durrant–Kabán converse [13] established the corres<sub>p</sub>ondin<sub>g</sub> necessar<sub>y</sub> con<sup>di</sup>t<sup>i</sup>on an<sup>d id</sup>ent<sup>ifi</sup>e<sup>d</sup> w<sup>h</sup>en nearest ne<sup>i</sup>g<sup>hb</sup>ors rema<sup>i</sup>n mean<sup>i</sup>ng<sup>f</sup>u<sup>l</sup>. <sup>A</sup> t<sup>hi</sup>r<sup>d li</sup>ne c<sup>h</sup>aracter<sup>i</sup>zes project<sup>i</sup>on outputs: Dasgupta [14, 15] established what we call Dasgupta’s sphericalization phenomenon, the rounding of eccentric Gaussian com<sub>p</sub>onents toward s<sub>p</sub>hericit<sub>y;</sub> Das<sub>g</sub>u<sub>p</sub>ta<sub>,</sub> Hsu<sub>,</sub> and Verma [16] and Diaconis and Freedman [17] obtained related structural results for broader distributions<sub>,</sub> includin<sub>g</sub> a<sub>pp</sub>roximation b<sub>y</sub> scale mixtures of s<sub>p</sub>herical Gaussians<sub>;</sub> and Li and Malik [18] <sub>g</sub>ave order <sub>p</sub>reservation <sub>g</sub>uarantees for fixed vector pa<sup>i</sup>rs at prescr<sup>ib</sup>e<sup>d l</sup>engt<sup>h</sup> rat<sup>i</sup>os. <sup>N</sup>one o<sup>f</sup> t<sup>h</sup>ese resu<sup>l</sup>ts tests w<sup>h</sup>et<sup>h</sup>er J<sup>L</sup> sat<sup>i</sup>s<sup>f</sup>act<sup>i</sup>on cert<sup>ifi</sup>es reta<sup>i</sup>ne<sup>d</sup> <sub>geometry,</sub> <sub>an</sub>d <sub>none</sub> <sub>quant</sub>ifi<sub>es</sub> h<sub>ow</sub> <sub>muc</sub>h di<sub>stance</sub> <sub>var</sub>i<sub>at</sub>i<sub>on</sub> <sub>a</sub> d<sub>eco</sub>d<sub>er</sub> <sub>can</sub> <sub>recover</sub> f<sub>rom</sub> <sub>a</sub> <sub>s</sub>k<sub>etc</sub>h<sub>.</sub>

<sup>O</sup>ur approac<sup>h</sup> <sup>i</sup>ntro<sup>d</sup>uces t<sup>h</sup>e m<sup>i</sup>ss<sup>i</sup>ng o<sup>b</sup>ject: an exp<sup>li</sup>c<sup>i</sup>t <sup>d</sup>eco<sup>d</sup>er. <sup>W</sup>e mo<sup>d</sup>e<sup>l</sup> recovery as est<sup>i</sup>mat<sup>i</sup>on <sub>o</sub>f <sub>a</sub> f<sub>eature</sub> <sub>o</sub>f <sub>a</sub> <sub>square</sub>d di<sub>stance</sub> f<sub>rom</sub> <sub>t</sub>h<sub>e</sub> <sub>s</sub>k<sub>etc</sub>h <sub>un</sub>d<sub>er</sub> <sub>square</sub>d<sub>-error</sub> l<sub>oss.</sub> Th<sub>e</sub> <sub>opt</sub>i<sub>ma</sub>l <sub>est</sub>i<sub>mate</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>expectat</sub>i<sub>on,</sub> <sub>so</sub> <sub>recovery</sub> <sub>acts</sub> <sub>as</sub> <sub>a</sub> li<sub>near</sub> <sub>operator,</sub> <sub>an</sub>d i<sub>ts</sub> <sub>s</sub>i<sub>ngu</sub>l<sub>ar</sub> <sub>va</sub>l<sub>ues</sub> <sub>measure</sub> h<sub>ow</sub> <sub>muc</sub>h <sub>o</sub>f <sub>eac</sub>h f<sub>eature</sub> <sub>surv</sub>i<sub>ves</sub> <sub>compress</sub>i<sub>on.</sub> Thi<sub>s</sub> f<sub>ormu</sub>l<sub>at</sub>i<sub>on</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> li<sub>m</sub>i<sub>ts</sub> <sub>t</sub>h<sub>at</sub> <sub>t</sub>h<sub>e</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>error</sub> cr<sup>i</sup>ter<sup>i</sup>on o<sup>f</sup> t<sup>h</sup>e J<sup>L</sup> <sup>l</sup>emma cannot express. We ma<sup>k</sup>e t<sup>h</sup>ree ma<sup>i</sup>n contr<sup>ib</sup>ut<sup>i</sup>ons. <sup>Fi</sup>rst, we prove t<sup>h</sup>at t<sup>h</sup>e error allowed b<sub>y</sub> the JL bound can exceed the fluctuation scale of a concentrated Gaussian cloud (section 2). We <sub>t</sub>h<sub>en s</sub>h<sub>ow t</sub>h<sub>at an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> G<sub>auss</sub>i<sub>an rep</sub>l<sub>acement map can sat</sub>i<sub>s</sub>f<sub>y t</sub>h<sub>e same</sub> b<sub>oun</sub>d <sub>at t</sub>h<sub>e</sub> di<sub>mens</sub>i<sub>on</sub> <sub>p</sub>rescribed b<sub>y</sub> the JL lemma even thou<sub>g</sub>h its re<sub>p</sub>lacement cloud is inde<sub>p</sub>endent of the data (section 3); sat<sup>i</sup>s<sup>f</sup>y<sup>i</sup>ng t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> t<sup>h</sup>ere<sup>f</sup>ore nee<sup>d</sup> not cert<sup>if</sup>y reta<sup>i</sup>ne<sup>d</sup> geometry. <sup>S</sup>econ<sup>d</sup>, we <sup>id</sup>ent<sup>if</sup>y t<sup>h</sup>ree recovery targets t<sup>h</sup>at t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> <sup>d</sup>oes not separate<sup>l</sup>y contro<sup>l</sup>—<sup>di</sup>stance va<sup>l</sup>ues, ne<sup>i</sup>g<sup>hb</sup>or ran<sup>ki</sup>ngs, an<sup>d</sup> covar<sup>i</sup>ance sha<sub>p</sub>e—and derive their exact Gaussian recover<sub>y</sub> limits (sections 4 to 8). For isotro<sub>p</sub>ic distance features, the recoverable variance ceilin<sub>g</sub> is m/d; nei<sub>g</sub>hbor rankin<sub>g</sub>s var<sub>y</sub> on the associated correlation scale m/d; <sub>an</sub>d dif<sub>use</sub> <sub>covar</sub>i<sub>ance</sub> <sub>s</sub>h<sub>ape</sub> <sub>contracts</sub> <sub>at</sub> <sub>or</sub>d<sub>er</sub> $( m / d ) ^ { 2 }$ <sub>.</sub> Thi<sub>r</sub>d<sub>,</sub> f<sub>or</sub> <sub>a</sub> <sub>genera</sub>l k<sub>nown</sub> <sub>map,</sub> <sub>we</sub> <sub>separate</sub> <sub>w</sub>h<sub>at</sub> d<sub>epen</sub>d<sub>s on ran</sub>k<sub>, s</sub>i<sub>ngu</sub>l<sub>ar spectrum, an</sub>d <sub>numer</sub>i<sub>ca</sub>l <sub>prec</sub>i<sub>s</sub>i<sub>on, t</sub>h<sub>en quant</sub>if<sub>y w</sub>h<sub>at ensem</sub>bl<sub>es o</sub>f i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> sketches can recover (sections 9 and 10).

Th<sub>ese</sub> l<sub>aws</sub> <sub>g</sub>i<sub>ve</sub> <sub>t</sub>h<sub>e</sub> <sub>pract</sub>i<sub>t</sub>i<sub>oner</sub>’<sub>s</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>a</sub> <sub>quant</sub>i<sub>tat</sub>i<sub>ve</sub> <sub>sca</sub>l<sub>e.</sub> Wi<sub>t</sub>hi<sub>n</sub> <sub>t</sub>h<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> b<sub>enc</sub>h<sub>-</sub> <sub>mar</sub>k<sub>, t</sub>h<sub>e</sub> f<sub>ract</sub>i<sub>on o</sub>f di<sub>stance var</sub>i<sub>ance reta</sub>i<sub>ne</sub>d<sub>, t</sub>h<sub>e corre</sub>l<sub>at</sub>i<sub>on sca</sub>l<sub>e o</sub>f <sub>ne</sub>i<sub>g</sub>hb<sub>or ran</sub>ki<sub>ngs, an</sub>d <sub>t</sub>h<sub>e</sub> contraction ofcovariance shape are each determined b<sub>y</sub> the ratio m/d, so the cost ofcompressin<sub>g</sub> farther and <sub>t</sub>h<sub>e</sub> b<sub>ene</sub>fi<sub>t</sub> <sub>o</sub>f <sub>a</sub>ddi<sub>ng</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>s</sub>k<sub>etc</sub>h<sub>es</sub> <sub>can</sub> b<sub>e</sub> <sub>rea</sub>d <sub>o</sub>f b<sub>e</sub>f<sub>ore</sub> <sub>any</sub> <sub>exper</sub>i<sub>ment.</sub> Th<sub>e</sub> <sub>rep</sub>l<sub>acement-map</sub> resu<sup>l</sup>t a<sup>dd</sup>s a caut<sup>i</sup>on t<sup>h</sup>at no emp<sup>i</sup>r<sup>i</sup>ca<sup>l</sup> protoco<sup>l</sup> supp<sup>li</sup>es: a map can sat<sup>i</sup>s<sup>f</sup>y t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> w<sup>hil</sup>e <sup>i</sup>ts output i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> d<sub>ata.</sub> B<sub>eyon</sub>d <sub>t</sub>h<sub>e</sub> <sub>spec</sub>ifi<sub>c</sub> G<sub>auss</sub>i<sub>an</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>we</sub> <sub>expect</sub> <sub>t</sub>h<sub>ese</sub> <sub>tec</sub>h<sub>n</sub>i<sub>ques</sub> <sub>to</sub> <sub>exten</sub>d <sub>to</sub> <sub>ot</sub>h<sub>er</sub> di<sub>str</sub>ib<sub>ut</sub>i<sub>ons</sub> <sub>an</sub>d <sub>to</sub> <sub>more</sub> <sub>comp</sub>l<sub>ex</sub> f<sub>eatures;</sub> <sub>sect</sub>i<sub>on</sub> 13 d<sub>eve</sub>l<sub>ops</sub> <sub>t</sub>h<sub>e</sub> <sub>resu</sub>l<sub>t</sub>i<sub>ng</sub> <sub>researc</sub>h di<sub>rect</sub>i<sub>ons.</sub> E<sub>xcept</sub> <sub>w</sub>h<sub>ere</sub> <sub>a</sub> <sub>comp</sub>l<sub>ete</sub> <sub>argument</sub> <sub>appears</sub> di<sub>rect</sub>l<sub>y</sub> <sub>a</sub>f<sub>ter</sub> <sub>a</sub> <sub>statement,</sub> <sub>proo</sub>f<sub>s</sub> <sub>are</sub> <sub>co</sub>ll<sub>ecte</sub>d i<sub>n</sub> <sub>t</sub>h<sub>e</sub> <sub>supp</sub>l<sub>ement</sub> (section B); the discussion followin<sub>g</sub> a statement in the text records the mechanism of the <sub>p</sub>roof.

## 2. The Scale Problem: Baseline versus Fluctuations

We <sup>b</sup>eg<sup>i</sup>n <sup>b</sup>y quant<sup>if</sup>y<sup>i</sup>ng w<sup>h</sup>en t<sup>h</sup>e re<sup>l</sup>at<sup>i</sup>ve error a<sup>ll</sup>owe<sup>d</sup> <sup>b</sup>y t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> <sup>i</sup>s too coarse to reso<sup>l</sup>ve <sup>di</sup>stance fl<sub>uctuat</sub>i<sub>ons.</sub> L<sub>et</sub> $X , X ^ { \prime } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( \mu , \Sigma )$ i<sub>n</sub> $\mathbb { R } ^ { d }$ , w<sup>h</sup>ere t<sup>h</sup>e ei<sub>g</sub>enva<sup>l</sup>ues o<sup>f</sup> Σ are $\lambda _ { 1 } \geq \cdots \geq \lambda _ { d } \geq 0$ <sub>.</sub> L<sub>e</sub>t $L : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ b<sub>e</sub> <sub>any</sub> li<sub>near</sub> <sub>map</sub> <sub>o</sub>f <sub>ran</sub>k <sub>at</sub> <sub>most</sub> $m ,$ random or deterministic, ob<sup>l</sup>ivious or Σ-aware; no <sub>statement</sub> b<sub>e</sub>l<sub>ow</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> h<sub>ow</sub> <sub>t</sub>h<sub>e</sub> <sub>map</sub> i<sub>s</sub> <sub>c</sub>h<sub>osen.</sub> W<sub>r</sub>i<sub>te</sub>

$$
D = \left\| X - X ^ { \prime } \right\| ^ { 2 } , \qquad \mathcal { O } = \left( L X , L X ^ { \prime } \right)
$$

for the squared distance and the observed sketch. Whenever L is random, we condition on its realization <sub>an</sub>d <sub>revea</sub>l i<sub>t</sub> <sub>to</sub> <sub>t</sub>h<sub>e</sub> d<sub>eco</sub>d<sub>er;</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ent</sub>l<sub>y,</sub> <sub>t</sub>h<sub>e</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>statements</sub> b<sub>e</sub>l<sub>ow</sub> <sub>are</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>on</sub> <sub>t</sub>h<sub>e</sub> <sub>samp</sub>l<sub>e</sub>d <sup>ma</sup>p<sup>. For a</sup> q<sup>uer</sup>y $X _ { 0 }$ <sub>an</sub>d <sub>two can</sub>did<sub>ates</sub> $X _ { 1 } , X _ { 2 }$ , define the neighbor contrast

$$
S = \| X _ { 0 } - X _ { 1 } \| ^ { 2 } - \| X _ { 0 } - X _ { 2 } \| ^ { 2 } .
$$

I<sub>ts</sub> <sub>s</sub>i<sub>gn</sub> id<sub>ent</sub>ifi<sub>es</sub> <sub>t</sub>h<sub>e</sub> <sub>nearer</sub> <sub>can</sub>did<sub>ate,</sub> <sub>so</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>a</sub> <sub>two-can</sub>did<sub>ate</sub> <sub>ran</sub>ki<sub>ng</sub> i<sub>s</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ent</sub> <sub>to</sub> <sub>preserv</sub>i<sub>ng</sub> the sign of S.

Th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> fl<sub>uctuat</sub>i<sub>ons</sub> <sub>appear</sub> i<sub>n</sub> <sub>t</sub>h<sub>e</sub> fi<sub>rst</sub> <sub>two</sub> <sub>moments:</sub>

$$
\begin{array} { r } { \mathbb { E } D = 2 \operatorname { t r } \Sigma , \qquad \operatorname { V a r } ( D ) = 8 \operatorname { t r } ( \Sigma ^ { 2 } ) . } \end{array}\tag{1}
$$

The mean E D is the common population baseline for i.i.d. pairs, while the centered deviations $D - \mathbb { E } D$ are the fluctuations that carry comparative geometry. Diagonalizing Σ expresses D as a weighted sum of i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>c</sub>hi<sub>-square</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>var</sub>i<sub>ances</sub> <sub>a</sub>dd<sub>,</sub> <sub>so</sub> <sub>t</sub>h<sub>e</sub> fl<sub>uctuat</sub>i<sub>on</sub> <sub>sca</sub>l<sub>e</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> <sub>t</sub>h<sub>e</sub> <sub>square</sub>d <sub>e</sub>i<sub>genva</sub>l<sub>ues t</sub>h<sub>roug</sub>h $\operatorname { t r } ( \Sigma ^ { 2 } )$ <sub>.</sub> Th<sub>e re</sub>l<sub>at</sub>i<sub>ve sca</sub>l<sub>e o</sub>f <sub>t</sub>h<sub>e</sub> fl<sub>uctuat</sub>i<sub>ons</sub> i<sub>s</sub>

$$
\begin{array} { r } { r _ { 2 } ( \Sigma ) = \displaystyle \frac { ( \mathrm { t r } \Sigma ) ^ { 2 } } { \mathrm { t r } ( \Sigma ^ { 2 } ) } , \qquad \displaystyle \frac { \mathrm { s d } ( D ) } { \mathbb { E } D } = \sqrt { \frac { 2 } { r _ { 2 } ( \Sigma ) } } . } \end{array}\tag{2}
$$

<sup>The</sup> q<sup>uantit</sup>y $r _ { 2 } ( \Sigma )$ i<sub>s</sub> <sub>a</sub> <sub>square</sub>d<sub>-trace</sub> <sub>e</sub>f<sub>ect</sub>i<sub>ve</sub> <sub>ran</sub>k<sub>:</sub> l<sub>arger</sub> <sub>va</sub>l<sub>ues</sub> <sub>mean</sub> <sub>t</sub>i<sub>g</sub>h<sub>ter</sub> <sub>re</sub>l<sub>at</sub>i<sub>ve</sub> <sub>concentrat</sub>i<sub>on.</sub> F<sub>or</sub> <sup>an isotro</sup>p<sup>ic covariance matrix</sup>, $r _ { 2 } ( \Sigma ) = d ,$ <sub>so re</sub>l<sub>at</sub>i<sub>ve</sub> fl<sub>uctuat</sub>i<sub>ons are o</sub>f <sub>or</sub>d<sub>er</sub> $d ^ { - 1 / 2 } \left[ 1 2 \right]$

Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>propos</sub>i<sub>t</sub>i<sub>on</sub> <sub>states</sub> <sub>exact</sub>l<sub>y</sub> <sub>w</sub>h<sub>en</sub> <sub>a</sub> <sub>re</sub>l<sub>at</sub>i<sub>ve-error</sub> b<sub>oun</sub>d fi<sub>xes</sub> <sub>t</sub>h<sub>e</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f <sub>a</sub> <sub>compar</sub>i<sub>son.</sub> H<sub>ere</sub> $\widetilde { D } _ { 1 }$ <sub>an</sub>d ${ \widetilde { D } } _ { 2 }$ <sub>are</sub> <sub>any</sub> di<sub>storte</sub>d <sub>va</sub>l<sub>ues</sub> <sub>sat</sub>i<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d i<sub>nequa</sub>li<sub>t</sub>i<sub>es;</sub> l<sub>ater</sub> <sub>sect</sub>i<sub>ons</sub> <sub>reserve</sub> <sub>t</sub>h<sub>e</sub> notation De for the s<sub>q</sub>uared distance com<sub>p</sub>uted from a sketch.

Proposition 2.1 (When a relative-error bound preserves order). Let $0 < D _ { 1 } < D _ { 2 } ,$ , and suppose

$$
( 1 - \varepsilon ) D _ { j } \leq \widetilde { D } _ { j } \leq ( 1 + \varepsilon ) D _ { j } , \qquad j = 1 , 2 .
$$

These inequalities guarantee $\widetilde { D } _ { 1 } < \widetilde { D } _ { 2 }$ for every admissible pair if and only if

$$
{ \frac { D _ { 2 } - D _ { 1 } } { D _ { 1 } + D _ { 2 } } } > \varepsilon .
$$

Proof. The worst admissible distortions preserve the order exactly when $( 1 + \varepsilon ) D _ { 1 } < ( 1 - \varepsilon ) D _ { 2 }$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ent</sub> <sub>to</sub> <sub>t</sub>h<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>con</sub>di<sub>t</sub>i<sub>on;</sub> if <sub>t</sub>hi<sub>s</sub> i<sub>nequa</sub>li<sub>ty</sub> f<sub>a</sub>il<sub>s,</sub> <sub>t</sub>h<sub>e</sub> <sub>extreme</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> <sub>errors</sub> <sub>reverse</sub> <sub>or</sub> <sub>t</sub>i<sub>e</sub> <sub>t</sub>h<sub>e</sub> <sub>or</sub>d<sub>er.</sub> □

Th<sub>e</sub> <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub> <sub>contrast</sub> i<sub>s</sub> <sub>one</sub> <sub>suc</sub>h <sub>compar</sub>i<sub>son,</sub> <sub>w</sub>i<sub>t</sub>h $D _ { j } = \left. X _ { 0 } - X _ { j } \right. ^ { 2 }$ <sub>:</sub> <sub>cert</sub>if<sub>y</sub>i<sub>ng</sub> <sub>w</sub>hi<sub>c</sub>h <sub>can</sub>did<sub>ate</sub> i<sub>s</sub> nearer is certi<sup>f</sup><sub>y</sub>in<sub>g</sub> t<sup>h</sup>e si<sub>g</sub>n o<sup>f</sup> $S = \pm ( D _ { 2 } - D _ { 1 } )$ <sub>.</sub> C<sub>ompar</sub>i<sub>sons</sub> <sub>w</sub>i<sub>t</sub>h <sub>sma</sub>ll<sub>er</sub> <sub>marg</sub>i<sub>ns</sub> <sub>may</sub> <sub>surv</sub>i<sub>ve,</sub> b<sub>ut</sub> <sub>t</sub>h<sub>e</sub> b<sub>oun</sub>d d<sub>oes</sub> <sub>not</sub> <sub>guarantee</sub> <sub>t</sub>h<sub>em.</sub> S<sub>ect</sub>i<sub>on</sub> 7 <sub>quant</sub>ifi<sub>es</sub> <sub>t</sub>h<sub>e</sub>i<sub>r</sub> <sub>average</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> f<sub>or</sub> <sub>t</sub>h<sub>e</sub> <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub> <sub>contrast</sub> <sub>un</sub>d<sub>er</sub> ran<sup>d</sup>om project<sup>i</sup>on.

The allowed absolute error in a typical squared distance is of order ε E D. If $\varepsilon \gg r _ { 2 } ( \Sigma ) ^ { - 1 / 2 }$ <sub>, t</sub>hi<sub>s error</sub> exceeds the fluctuation scale sd(D). In this re<sub>g</sub>ime, the JL bound can still hold, but it does not b<sub>y</sub> itself certif<sub>y</sub> <sub>ne</sub>i<sub>g</sub>hb<sub>or compar</sub>i<sub>sons.</sub> If i<sub>nstea</sub>d $\varepsilon \asymp r _ { 2 } ( \Sigma ) ^ { - 1 / 2 }$ , t<sup>h</sup>e J<sup>L</sup> <sup>l</sup>emma g<sup>i</sup>ves $m = O ( r _ { 2 } ( \Sigma ) \log n )$ . For isotro<sub>p</sub>ic <sup>d</sup>ata, this scales as d lo<sub>g</sub> n and therefore no lon<sub>g</sub>er <sub>g</sub>uarantees a tar<sub>g</sub>et dimension below $\bar { d . }$

Th<sub>e</sub> <sub>a</sub>ll<sub>owe</sub>d <sub>error</sub> <sub>can</sub> <sub>t</sub>h<sub>ere</sub>f<sub>ore</sub> b<sub>e</sub> <sub>too</sub> l<sub>arge</sub> <sub>to</sub> <sub>reso</sub>l<sub>ve</sub> <sub>comparat</sub>i<sub>ve</sub> <sub>geometry.</sub> A <sub>coarser</sub> <sub>to</sub>l<sub>erance</sub> <sub>st</sub>ill l<sub>eaves</sub> <sub>a</sub> di<sub>st</sub>i<sub>nct</sub> <sub>quest</sub>i<sub>on:</sub> <sub>w</sub>h<sub>et</sub>h<sub>er</sub> <sub>pass</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> <sub>test</sub> <sub>cert</sub>ifi<sub>es</sub> <sub>anyt</sub>hi<sub>ng</sub> <sub>a</sub>b<sub>out</sub> <sub>t</sub>h<sub>e</sub> <sub>reta</sub>i<sub>ne</sub>d <sub>geometry.</sub>

## 3. The JL Bound Can Hold without Information

Does satisfying the JL bound imply retention of any sample-specific structure? Section 2 s<sup>h</sup>owe<sup>d</sup> t<sup>h</sup>at t<sup>h</sup>e error a<sup>ll</sup>owe<sup>d b</sup>y t<sup>h</sup>e J<sup>L b</sup>oun<sup>d</sup> can excee<sup>d</sup> t<sup>h</sup>e <sup>fl</sup>uctuat<sup>i</sup>on sca<sup>l</sup>e. <sup>Th</sup>at o<sup>b</sup>servat<sup>i</sup>on concerns t<sup>h</sup>e to<sup>l</sup>erance, <sub>not</sub> <sub>t</sub>h<sub>e</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>reta</sub>i<sub>ne</sub>d b<sub>y</sub> <sub>a</sub> <sub>part</sub>i<sub>cu</sub>l<sub>ar</sub> <sub>map:</sub> <sub>a</sub> l<sub>oose</sub> b<sub>oun</sub>d <sub>may</sub> <sub>coex</sub>i<sub>st</sub> <sub>w</sub>i<sub>t</sub>h i<sub>n</sub>f<sub>ormat</sub>i<sub>ve</sub> <sub>maps.</sub> W<sub>e</sub> t<sup>h</sup>ere<sup>f</sup>ore as<sup>k</sup>: <sup>d</sup>oes sat<sup>i</sup>s<sup>f</sup>y<sup>i</sup>ng t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> guarantee any samp<sup>l</sup>e-spec<sup>ifi</sup>c structure<sup>?</sup>

We answer negatively by constructing a map. An independent Gaussian replacement map assigns each in<sup>d</sup>exe<sup>d</sup> in<sub>p</sub>ut <sub>p</sub>oint $X _ { i }$ <sub>a</sub> f<sub>res</sub>h G<sub>auss</sub>i<sub>an</sub> <sub>output</sub> $Y _ { i } ,$ <sub>samp</sub>l<sub>e</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub>l<sub>y</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> d<sub>ata,</sub> <sub>so</sub> <sub>t</sub>h<sub>at</sub> $I ( X ^ { n } ; Y ^ { n } ) = 0$ and no decoder can extract any feature of the input from the output. Its image is the replacement cloud; the <sub>s</sub>h<sub>are</sub>d i<sub>n</sub>di<sub>ces</sub> <sub>pa</sub>i<sub>r</sub> <sub>eac</sub>h i<sub>nput</sub> di<sub>stance</sub> $\left\| { X } _ { i } - { X } _ { j } \right\|$ <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>output</sub> di<sub>stance</sub> $\left\| Y _ { i } - Y _ { j } \right\|$ <sub>.</sub> Th<sub>e</sub> <sub>q</sub>uestion is whether this zero-information ma<sub>p</sub> can nonetheless satisf<sub>y</sub> the ε-JL bound on all (<sup>n</sup><sub>2</sub>) <sub>p</sub>airs at the stan<sup>d</sup>ar<sup>d</sup> J<sup>L</sup> <sup>di</sup>mens<sup>i</sup>on.

## 3.1 The independent Gaussian replacement map

Proposition 3.1 (A zero-information map can satisfy the JL bound). Let $n \geq 2 , m , d \in \mathbb { N } , \sigma > 0 ,$ and $0 < \varepsilon , \delta < 1$ . Let $X _ { 1 } , \dots , X _ { n } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( \mu , \sigma ^ { 2 } I _ { d } )$ and $Y _ { 1 } , \dots , Y _ { n } \overset { \mathrm { i i d } } { \sim } \mathcal { N } \big ( 0 , \sigma ^ { 2 } \frac { d } { m } I _ { m } \big )$ be independent, and define the index map $T _ { \mathrm { r e p } } ( X _ { i } ) = Y _ { i } .$ If min $\{ m , d \} \ge 7 2 \varepsilon ^ { - 2 } \log \bigl ( 4 { \binom { n } { 2 } } / \delta \bigr )$ , then, with probability at least $1 - \delta _ { \cdot }$

$$
\begin{array} { r } { ( 1 - \varepsilon ) \left\| X _ { i } - X _ { j } \right\| ^ { 2 } \leq \left\| T _ { \mathrm { r e p } } ( X _ { i } ) - T _ { \mathrm { r e p } } ( X _ { j } ) \right\| ^ { 2 } \leq ( 1 + \varepsilon ) \left\| X _ { i } - X _ { j } \right\| ^ { 2 } \qquad f o r \ a l l \ i < j . } \end{array}
$$

In particular, in the compression regime $m \leq d , T _ { \mathrm { r e p } }$ can satisfy the $\varepsilon - J L$ bound on the sample at the standard JL dimension order $m = O \bigl ( \varepsilon ^ { - 2 } \log ( n / \delta ) \bigr )$ , while its output carries no information about the data: $I ( X ^ { n } ; Y ^ { n } ) = 0$

Proof. For every $i < j ,$

$$
\frac { \left\| X _ { i } - X _ { j } \right\| ^ { 2 } } { 2 \sigma ^ { 2 } d } \overset { d } { = } \frac { x _ { d } ^ { 2 } } { d } , \qquad \frac { \left\| Y _ { i } - Y _ { j } \right\| ^ { 2 } } { 2 \sigma ^ { 2 } d } \overset { d } { = } \frac { \chi _ { m } ^ { 2 } } { m } .
$$

S<sub>e</sub>t $a = \varepsilon / { \bigl ( } 2 + \varepsilon { \bigr ) }$ <sub>.</sub> Th<sub>e</sub> Ch<sub>erno</sub>f b<sub>oun</sub>d <sub>g</sub>i<sub>ves</sub> $\operatorname* { P r } \bigr ( \biggr | x _ { k } ^ { 2 } / k - 1 \biggr | > a \bigr ) \le 2 e ^ { - k a ^ { 2 } / 8 }$ <sub>.</sub> A <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>over t</sub>h<sub>e</sub> $2 { \binom { n } { 2 } }$ <sub>norma</sub>li<sub>ze</sub>d <sub>square</sub>d di<sub>stances p</sub>l<sub>aces a</sub>ll <sub>o</sub>f <sub>t</sub>h<sub>em</sub> i<sub>n</sub> $[ 1 - a , \dot { 1 } + a ]$ <sub>w</sub>i<sub>t</sub>h <sub>pro</sub>b<sub>a</sub>bili<sub>ty</sub> <sub>at</sub> l<sub>east</sub>

$$
1 - 4 { \binom { n } { 2 } } \exp \left( - { \frac { \operatorname* { m i n } \{ m , d \} a ^ { 2 } } { 8 } } \right) \geq 1 - \delta ,
$$

<sub>w</sub>h<sub>ere</sub> <sub>t</sub>h<sub>e</sub> l<sub>ast</sub> i<sub>nequa</sub>li<sub>ty</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> <sub>t</sub>h<sub>e</sub> di<sub>mens</sub>i<sub>on</sub> h<sub>ypot</sub>h<sub>es</sub>i<sub>s</sub> <sub>an</sub>d $8 ( 2 + \varepsilon ) ^ { 2 } < 7 2 \nonumber$ . On t<sup>h</sup>is event, ever<sub>y</sub> <sup>out</sup>p<sup>ut-to-in</sup>p<sup>ut</sup> <sup>ratio</sup> <sup>lies</sup> <sup>in</sup>

$$
{ \bigg [ } { \frac { 1 - a } { 1 + a } } , { \frac { 1 + a } { 1 - a } } { \bigg ] } = { \bigg [ } { \frac { 1 } { 1 + \varepsilon } } , 1 + \varepsilon { \bigg ] } \subseteq { \big [ } 1 - \varepsilon , 1 + \varepsilon { \big ] } ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>proves</sub> <sub>t</sub>h<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m.</sub>

<sup>Th</sup>e mec<sup>h</sup>an<sup>i</sup>sm <sup>i</sup>s t<sup>h</sup>at t<sup>h</sup>e J<sup>L</sup> event <sup>f</sup>actors: <sup>i</sup>nput-c<sup>l</sup>ou<sup>d</sup> concentrat<sup>i</sup>on an<sup>d</sup> output-c<sup>l</sup>ou<sup>d</sup> concentrat<sup>i</sup>on <sub>aroun</sub>d <sub>a</sub> <sub>s</sub>h<sub>are</sub>d b<sub>ase</sub>li<sub>ne</sub> <sub>su</sub>fi<sub>ce,</sub> <sub>an</sub>d <sub>no</sub> <sub>cross-con</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>structure</sub> i<sub>s</sub> <sub>nee</sub>d<sub>e</sub>d<sub>.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>remar</sub>k <sub>recor</sub>d<sub>s</sub> <sub>t</sub>h<sub>e constants an</sub>d <sub>t</sub>h<sub>e two-stage</sub> f<sub>orm.</sub>

Remark 3.2 (Sharp constants, two-stage form, and scope). Let $\mathcal { E } _ { \varepsilon } ^ { \mathrm { r e p } }$ <sup>d</sup>enote t<sup>h</sup>e J<sup>L</sup> event <sup>i</sup>n propos<sup>i</sup>t<sup>i</sup>on 3.1, <sub>an</sub>d<sub>,</sub> f<sub>or</sub> $0 < a < 1$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\mathcal G _ { X } ( a ) = \left\{ \left| \frac { \left. X _ { i } - X _ { j } \right. ^ { 2 } } { 2 \sigma ^ { 2 } d } - 1 \right| \leq a \mathrm { f o r } \mathrm { a l l } i < j \right\} .
$$

(a) In the com<sub>p</sub>ression re<sub>g</sub>ime $m \leq d ,$ the hypothesis is a condition on m alone, with the dimension or<sup>d</sup>er prescr<sup>ib</sup>e<sup>d</sup> <sup>b</sup>y t<sup>h</sup>e J<sup>L</sup> <sup>l</sup>emma: $m = O \bigl ( \varepsilon ^ { - 2 } \log ( n / \delta ) \bigr )$ .

(b) The <sub>p</sub>roof <sub>g</sub>ives the shar<sub>p</sub> form: with $a _ { \varepsilon } = \varepsilon / { \left( 2 + \varepsilon \right) }$

$$
\mathrm { P r } ( \mathcal { E } _ { \varepsilon } ^ { \mathrm { r e p } } ) \geq 1 - 2 N _ { p } e ^ { - d a _ { \varepsilon } ^ { 2 } / 8 } - 2 N _ { p } e ^ { - m a _ { \varepsilon } ^ { 2 } / 8 } ,
$$

<sub>an</sub>d <sub>m</sub>i<sub>n</sub> $\{ m , d \} \ge 8 ( 2 + \varepsilon ) ^ { 2 } \varepsilon ^ { - 2 } \log ( 4 N _ { p } / \delta )$ <sub>su</sub>fi<sub>ces.</sub>

(c) The same <sub>p</sub>roof <sub>y</sub>ields the two-sta<sub>g</sub>e bounds

$$
\operatorname* { P r } _ { X } \left( \mathcal { G } _ { X } ( a _ { \varepsilon } ) \right) \geq 1 - 2 N _ { p } e ^ { - d a _ { \varepsilon } ^ { 2 } / 8 } , \qquad \operatorname* { i n f } _ { x ^ { n } \in \mathcal { G } _ { X } ( a _ { \varepsilon } ) } \operatorname* { P r } _ { Y } \left( \mathcal { E } _ { \varepsilon } ^ { \mathrm { r e p } } \mid X ^ { n } = x ^ { n } \right) \geq 1 - 2 N _ { p } e ^ { - m a _ { \varepsilon } ^ { 2 } / 8 } ,\tag{3}
$$

<sub>so most</sub> fi<sub>xe</sub>d i<sub>nput c</sub>l<sub>ou</sub>d<sub>s</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>ll<sub>y a</sub>d<sub>m</sub>i<sub>t an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> G<sub>auss</sub>i<sub>an rep</sub>l<sub>acement map.</sub> Th<sub>e p</sub>h<sub>e-</sub> <sup>nomenon</sup> <sup>is</sup> <sup>therefore</sup> <sup>a</sup> p<sup>ro</sup>p<sup>ert</sup>y <sup>of</sup> <sup>concentration</sup>, <sup>not</sup> <sup>of</sup> <sup>avera</sup>g<sup>in</sup>g <sup>over</sup> <sup>random</sup> <sup>in</sup>p<sup>uts.</sup>

(d) Inde<sub>p</sub>endence concerns the unconditional joint law; conditionin<sub>g</sub> on the success event induces d<sub>epen</sub>d<sub>ence.</sub>

Figure 1 i<sup>ll</sup>ustrates t<sup>h</sup>e separation. For comparison, we use a resca<sup>l</sup>e<sup>d</sup> ran<sup>k</sup>-m ort<sup>h</sup>ogona<sup>l</sup> projection whose row s<sub>p</sub>ace is uniforml<sub>y</sub> random (Haar-distributed, in the terminolo<sub>gy</sub> of section 4.3).

(c) Replacement

![](images/b1c25a9971a107515b7e4b5e09deef1c1b32e1ddf69d09543a4beae69a83b1cd.jpg)

![](images/e1c21b0847f86c840834596d4a95a7573316d9d15e270a4c8171e35ea2aee3a7.jpg)

![](images/d382d41a9c228379120ca1ea79a74c4f49839c9e7b2e26c8cfdf93fe515a9f1c.jpg)  
Figure 1. Satisfying the JL bound does not certify retained information. Panel (a) reports the median maximum relative error over all $\textstyle { \binom { 1 0 0 } { 2 } }$ pairs for a rescaled uniformly random rank-m orthogonal projection and an independent Gaussian replacement map at $\stackrel { \_ } { d } = 8 1 9 2 ;$ bands span the 10th–90th percentiles over 30 trials, and the dashed line marks $\varepsilon = 0 . 2$ . Panels (b)–(c) show one trial at m/d = 1/4: projected squared distances remain correlated with the original squared-distance fluctuations, whereas replacement squared distances do not. The annotated maximum errors show that both displayed maps satisfy the JL bound.

<sup>Th</sup>ree natura<sup>l</sup> o<sup>b</sup>ject<sup>i</sup>ons <sup>d</sup>e<sup>li</sup>m<sup>i</sup>t t<sup>h</sup>e scope o<sup>f</sup> t<sup>hi</sup>s separat<sup>i</sup>on: w<sup>h</sup>at t<sup>h</sup>e output actua<sup>ll</sup>y te<sup>ll</sup>s a pract<sup>i</sup>t<sup>i</sup>oner, <sub>w</sub>h<sub>et</sub>h<sub>er</sub> <sub>t</sub>h<sub>e</sub> <sub>construct</sub>i<sub>on</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> <sub>popu</sub>l<sub>at</sub>i<sub>on</sub> <sub>sca</sub>li<sub>ng,</sub> <sub>an</sub>d <sub>w</sub>h<sub>et</sub>h<sub>er</sub> i<sub>t</sub> <sub>requ</sub>i<sub>res</sub> <sub>an</sub> <sub>overs</sub>i<sub>ze</sub>d di<sub>mens</sub>i<sub>on.</sub> Th<sub>e</sub> <sub>same</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>g</sub>i<sub>ves</sub> <sub>an</sub> <sub>exact</sub> <sub>answer</sub> <sub>to</sub> <sub>t</sub>h<sub>e</sub> fi<sub>rst.</sub>

Corollary 3.3 (Ranking consequences of the replacement map). Under the assumptions of proposition 3.1, $i f n \geq 3 ,$ define the mean Kendall statistic for the replacement map by

$$
\overline { { \mathfrak { T } } } _ { n } ^ { \mathrm { r e p } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( { n - 1 \atop 2 } \right) ^ { - 1 } \sum _ { j < k } \mathrm { s g n } \Big [ ( D _ { i j } - D _ { i k } ) \big ( D _ { i j } ^ { \mathrm { r e p } } - D _ { i k } ^ { \mathrm { r e p } } \big ) \Big ] .
$$

Then

$$
\mathbb { E } \overline { { \mathfrak { T } } } _ { n } ^ { \mathrm { r e p } } = 0 , \qquad \operatorname* { P r } \left( \left| \overline { { \mathfrak { T } } } _ { n } ^ { \mathrm { r e p } } \right| > t \right) \leq 2 \exp \left( - \frac { \lfloor n / 3 \rfloor t ^ { 2 } } { 2 } \right) .
$$

In the analogous setting with one query and $q \geq 2$ candidates, let $\mathcal { N } _ { k }$ and $\mathcal { N } _ { k } ^ { \mathrm { r e p } }$ be the top-k sets for the original and replacement clouds, respectively, where $1 \leq k < q .$ Then

$$
\mathbb { E } \frac { \left| \mathcal { N } _ { k } \cap \mathcal { N } _ { k } ^ { \mathrm { r e p } } \right| } { k } = \frac { k } { q } , \qquad \operatorname* { P r } \left( \mathcal { N } _ { 1 } = \mathcal { N } _ { 1 } ^ { \mathrm { r e p } } \right) = \frac { 1 } { q } .
$$

Proof. Each Kendall summand is a product of two independent signs. Exchangeability of the candidate l<sub>a</sub>b<sub>e</sub>l<sub>s ma</sub>k<sub>es one s</sub>i<sub>gn symmetr</sub>i<sub>c, so t</sub>h<sub>e</sub> k<sub>erne</sub>l h<sub>as mean zero.</sub> Si<sub>nce</sub> $( X _ { i } , Y _ { i } )$ are i.i.<sup>d</sup>. <sub>p</sub>airs across $i , \overline { { \tau } } _ { n } ^ { \mathrm { r e p } }$ i<sub>s</sub> <sub>a</sub> bounded order-three U-statistic. Hoefdin<sub>g</sub> [19] <sub>p</sub>roved the re<sub>q</sub>uired blockin<sub>g</sub> ine<sub>q</sub>ualit<sub>y;</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> it <sub>g</sub>ives <sub>t</sub>h<sub>e state</sub>d <sub>ta</sub>il b<sub>oun</sub>d<sub>.</sub> Th<sub>e ne</sub>i<sub>g</sub>hb<sub>or set</sub> f<sub>rom t</sub>h<sub>e rep</sub>l<sub>acement c</sub>l<sub>ou</sub>d i<sub>s a</sub> f<sub>unct</sub>i<sub>on o</sub>f <sub>t</sub>h<sub>at c</sub>l<sub>ou</sub>d <sub>a</sub>l<sub>one;</sub> b<sub>y</sub> exchan<sub>g</sub>eabilit<sub>y</sub> it is a uniform k-subset independent of $\mathcal { N } _ { k } .$ <sub>.</sub> Ti<sub>es</sub> h<sub>ave</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>ty</sub> <sub>zero.</sub> □

## 3.2 Calibration

Empirical calibration. Em<sub>p</sub>irica<sup>l</sup> ca<sup>l</sup>i<sup>b</sup>ration can re<sub>p</sub><sup>l</sup>ace <sub>p</sub>o<sub>p</sub>u<sup>l</sup>ation sca<sup>l</sup>in<sub>g</sub>, w<sup>h</sup>ic<sup>h</sup> uses <sup>k</sup>now<sup>l</sup>e<sup>d</sup><sub>g</sub>e o<sup>f</sup> $\sigma ^ { 2 }$ <sub>t</sub>h<sub>at</sub> <sub>a</sub> <sub>pract</sub>i<sub>t</sub>i<sub>oner</sub> <sub>may</sub> <sub>not</sub> h<sub>ave.</sub> Wi<sub>t</sub>h $\overline { { D } } _ { X } = N _ { p } ^ { - 1 } \sum _ { i < j } D _ { i j }$ <sub>an</sub>d $\overline { { D } } _ { \mathrm { r e p } } = \mathrm { \overline { { N } } } _ { p } ^ { - 1 } \sum _ { i < j } D _ { i j } ^ { \mathrm { r e p } }$ <sub>, t</sub>h<sub>e resca</sub>l<sub>e</sub>d <sub>c</sub>l<sub>ou</sub>d $Y _ { i } ^ { \star } = ( \overline { { D } } _ { X } / \overline { { D } } _ { \mathrm { r e p } } ) ^ { 1 / 2 } Y _ { i }$ <sub>matc</sub>h<sub>es</sub> <sub>t</sub>h<sub>e</sub> <sub>rea</sub>li<sub>ze</sub>d <sub>mean</sub> <sub>square</sub>d di<sub>stance</sub> <sub>exact</sub>l<sub>y.</sub> Th<sub>e</sub> <sub>same</sub> <sub>proo</sub>f h<sub>o</sub>ld<sub>s</sub> <sub>w</sub>i<sub>t</sub>h $b _ { \varepsilon } = ( \sqrt { 1 + \varepsilon } - 1 ) / ( \sqrt { 1 + \varepsilon } + 1 ) \sim \varepsilon / 4$ i<sub>n</sub> <sub>p</sub>l<sub>ace</sub> <sub>o</sub>f $a _ { \varepsilon } ,$ <sub>,</sub> b<sub>ecause</sub> <sub>on</sub> <sub>t</sub>h<sub>e</sub> b<sub>an</sub>d <sub>event</sub> b<sub>ot</sub>h <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>means</sub> <sub>a</sub>l<sub>so</sub> li<sub>e</sub> i<sub>n t</sub>h<sub>e</sub> b<sub>an</sub>d <sub>an</sub>d <sub>t</sub>h<sub>e worst rat</sub>i<sub>o</sub> i<sub>s</sub> $( ( 1 + \dot { b _ { \varepsilon } } ) / ( 1 - b _ { \varepsilon } ) ) ^ { 2 } = 1 + \varepsilon$ <sub>.</sub> Gl<sub>o</sub>b<sub>a</sub>l <sub>resca</sub>li<sub>ng</sub> <sub>preserves</sub> <sub>a</sub>ll <sub>or</sub>d<sub>er</sub>i<sub>ngs,</sub> <sub>so</sub> <sub>t</sub>h<sub>e</sub> <sub>conc</sub>l<sub>us</sub>i<sub>ons</sub> <sub>o</sub>f <sub>coro</sub>ll<sub>ary</sub> 3<sub>.</sub>3 f<sub>or</sub> <sub>ran</sub>ki<sub>ngs</sub> <sub>an</sub>d <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d<sub>s</sub> <sub>rema</sub>i<sub>n</sub> <sub>exact.</sub> If $\mathcal { E } _ { \varepsilon } ^ { \star }$ <sup>d</sup>enotes t<sup>h</sup>e ca<sup>lib</sup>rate<sup>d</sup> J<sup>L</sup> <sub>event,</sub> <sub>t</sub>h<sub>en</sub>

$$
\begin{array} { r } { \operatorname* { P r } ( \mathcal { E } _ { \varepsilon } ^ { \star } ) \geq 1 - 2 N _ { p } e ^ { - d b _ { \varepsilon } ^ { 2 } / 8 } - 2 N _ { p } e ^ { - m b _ { \varepsilon } ^ { 2 } / 8 } . } \end{array}
$$

Th<sub>e</sub> <sub>ca</sub>lib<sub>rate</sub>d <sub>output</sub> i<sub>s</sub> <sub>not</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> i<sub>nput.</sub> I<sub>ts</sub> d<sub>ata-</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>part</sub> i<sub>s</sub> <sub>c</sub>h<sub>aracter</sub>i<sub>ze</sub>d <sub>prec</sub>i<sub>se</sub>l<sub>y</sub> b<sub>y</sub>

$$
X ^ { n } \longrightarrow D _ { X } \longrightarrow Y ^ { \star n } , \qquad \overline { { { D } } } _ { Y ^ { \star } } = \overline { { { D } } } _ { X } \quad \mathrm { a l m o s t \ s u r e l y . }
$$

Th<sub>us</sub> <sub>t</sub>h<sub>e</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l l<sub>aw</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> <sub>output</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> $X ^ { n }$ <sub>on</sub>l<sub>y</sub> <sub>t</sub>h<sub>roug</sub>h $\overline { { D } } _ { X }$ <sub>,</sub> <sub>an</sub>d <sub>t</sub>h<sub>at</sub> <sub>stat</sub>i<sub>st</sub>i<sub>c</sub> i<sub>s</sub> <sub>recovera</sub>bl<sub>e</sub> f<sub>rom</sub> <sub>t</sub>h<sub>e</sub> <sub>output:</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>ca</sub>lib<sub>rat</sub>i<sub>on</sub> l<sub>ea</sub>k<sub>s</sub> <sub>exact</sub>l<sub>y</sub> <sub>one</sub> <sub>sca</sub>l<sub>ar</sub> <sub>an</sub>d <sub>no</sub> <sub>ran</sub>ki<sub>ng</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on.</sub>

## 3.3 Sharpness

<sup>Th</sup>e t<sup>hi</sup>r<sup>d</sup> o<sup>b</sup>ject<sup>i</sup>on <sup>i</sup>s t<sup>h</sup>at t<sup>h</sup>e rep<sup>l</sup>acement map m<sup>i</sup>g<sup>h</sup>t succee<sup>d</sup> on<sup>l</sup>y <sup>i</sup>n a <sup>di</sup>mens<sup>i</sup>on so <sup>l</sup>arge t<sup>h</sup>at t<sup>h</sup>e J<sup>L</sup> b<sub>oun</sub>d b<sub>ecomes</sub> <sub>tr</sub>i<sub>v</sub>i<sub>a</sub>l <sub>to</sub> <sub>sat</sub>i<sub>s</sub>f<sub>y.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>propos</sub>i<sub>t</sub>i<sub>on</sub> <sub>ru</sub>l<sub>es</sub> <sub>t</sub>hi<sub>s</sub> <sub>out.</sub>

Proposition 3.4 (Sharp dimension order for independent Gaussian replacement maps). In the setting of proposition 3.1, let $r = \lfloor n / 2 \rfloor$ and $0 < \varepsilon \le 1$ . There are universal constants $c , C > 0$ such that

$$
\mathrm { P r } ( \mathcal { E } _ { \varepsilon } ^ { \mathrm { r e p } } ) \leq \exp \left( - c r e ^ { - C m \varepsilon ^ { 2 } } \right) .
$$

Consequently, i $f \mathrm { P r } ( \mathcal { E } _ { \varepsilon } ^ { \mathrm { r e p } } ) \geq 1 - \delta$ , then

$$
m \geq \frac { 1 } { C \varepsilon ^ { 2 } } \log \left( \frac { c r } { - \log \left( 1 - \delta \right) } \right)
$$

whenever the logarithm is positive. In particular, for $\delta \leq 1 / 2$ and n/δ suficiently large, $m = \Omega \left( \varepsilon ^ { - 2 } \log ( n / \delta ) \right)$

Th<sub>e proo</sub>f i<sub>n sect</sub>i<sub>on</sub> B <sub>uses t</sub>h<sub>e</sub> i<sub>.</sub>i<sub>.</sub>d<sub>.</sub> $F _ { m , d }$ rat<sup>i</sup>os assoc<sup>i</sup>ate<sup>d</sup> w<sup>i</sup>t<sup>h</sup> <sup>di</sup>sjo<sup>i</sup>nt pa<sup>i</sup>rs an<sup>d</sup> t<sup>h</sup>e <sup>Zh</sup>ang–<sup>Zh</sup>ou lower bound [20<sub>,</sub> Cor. 3] for the u<sub>pp</sub>er chi-s<sub>q</sub>uare tail <sub>p</sub>robabilit<sub>y</sub>. To<sub>g</sub>ether with <sub>p</sub>ro<sub>p</sub>osition 3.1<sub>,</sub> this shows t<sup>h</sup>at t<sup>h</sup>e <sup>di</sup>mens<sup>i</sup>on or<sup>d</sup>er prescr<sup>ib</sup>e<sup>d</sup> <sup>b</sup>y t<sup>h</sup>e J<sup>L</sup> <sup>l</sup>emma <sup>i</sup>s <sup>b</sup>ot<sup>h</sup> su<sup>fi</sup>c<sup>i</sup>ent an<sup>d</sup> necessary <sup>f</sup>or t<sup>h</sup>e <sup>i</sup>n<sup>d</sup>epen<sup>d</sup>ent <sup>Gaussian</sup> <sup>re</sup>p<sup>lacement</sup> <sup>ma</sup>p<sup>.</sup>

<sup>Th</sup>e conc<sup>l</sup>us<sup>i</sup>on <sup>i</sup>s t<sup>h</sup>at t<sup>h</sup>e cert<sup>ifi</sup>cate an<sup>d</sup> t<sup>h</sup>e <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on are separate o<sup>b</sup>jects. <sup>S</sup>at<sup>i</sup>s<sup>f</sup>y<sup>i</sup>ng t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> <sup>d</sup>oes not <sup>b</sup>y <sup>i</sup>tse<sup>lf</sup> cert<sup>if</sup>y reta<sup>i</sup>ne<sup>d</sup> geometry; w<sup>h</sup>at rep<sup>l</sup>aces t<sup>h</sup>e cert<sup>ifi</sup>cate <sup>i</sup>s t<sup>h</sup>e su<sup>b</sup>ject o<sup>f</sup> t<sup>h</sup>e next sect<sup>i</sup>on.

## 4. Recovery as an Operator Problem

## 4.1 Sketch, target, and decoder

<sup>Th</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> c<sup>h</sup>ec<sup>k</sup>s <sup>if</sup> a map preserves or<sup>i</sup>g<sup>i</sup>na<sup>l</sup> <sup>di</sup>stances w<sup>i</sup>t<sup>hi</sup>n a sma<sup>ll</sup> re<sup>l</sup>at<sup>i</sup>ve error. <sup>I</sup>n contrast, recovery <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> fi<sub>xes</sub> <sub>an</sub> <sub>o</sub>b<sub>servat</sub>i<sub>on</sub> <sub>an</sub>d d<sub>eterm</sub>i<sub>nes</sub> <sub>w</sub>h<sub>at</sub> <sub>any</sub> d<sub>eco</sub>d<sub>er</sub> <sub>can</sub> i<sub>n</sub>f<sub>er</sub> f<sub>rom</sub> i<sub>t.</sub> S<sub>ect</sub>i<sub>on</sub> 2 <sub>an</sub>d <sub>sect</sub>i<sub>on</sub> 3 <sub>s</sub>h<sub>ow</sub> w<sup>h</sup>y t<sup>hi</sup>s <sup>di</sup>st<sup>i</sup>nct<sup>i</sup>on matters: t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> can <sup>h</sup>o<sup>ld</sup> even w<sup>h</sup>en t<sup>h</sup>e mappe<sup>d</sup> po<sup>i</sup>nts conta<sup>i</sup>n no <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on a<sup>b</sup>out t<sup>h</sup>e ori<sub>g</sub>ina<sup>l</sup> <sub>g</sub>eometr<sub>y</sub>.

Central question. Which features of the original geometry can a decoder recover from the <sub>s</sub>k<sub>etc</sub>h<sub>, an</sub>d h<sub>ow muc</sub>h <sub>o</sub>f <sub>eac</sub>h?

T<sub>o ana</sub>l<sub>yze a s</sub>i<sub>ng</sub>l<sub>e</sub> di<sub>stance, we</sub> f<sub>ormu</sub>l<sub>ate t</sub>h<sub>e quest</sub>i<sub>on as</sub> f<sub>o</sub>ll<sub>ows.</sub> A k<sub>nown</sub> li<sub>near map</sub> $L : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ <sup>acts on a</sup> p<sup>air</sup> $X , X ^ { \prime } .$ <sub>.</sub> Th<sub>e target</sub> di<sub>stance</sub> i<sub>s</sub> $D = \left\| X - X ^ { \prime } \right\| ^ { 2 }$ <sub>,</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> <sub>o</sub>b<sub>servat</sub>i<sub>on</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>s</sub>k<sub>etc</sub>h $\mathcal { O } = \left( L X , L X ^ { \prime } \right)$ A feature is a square-integrable scalar $f ( D )$ wit<sup>h</sup> <sub>p</sub>ositive variance $\mathrm { V a r } ( f ( D ) ) > 0 .$ A decoder is any measurable f<sub>unct</sub>i<sub>on</sub> $g ( \mathcal { O } )$

U<sub>n</sub>d<sub>er</sub> <sub>square</sub>d<sub>-error</sub> l<sub>oss,</sub> <sub>t</sub>h<sub>e</sub> <sub>opt</sub>i<sub>ma</sub>l d<sub>eco</sub>d<sub>er</sub> f<sub>or</sub> <sub>a</sub> f<sub>eature</sub> i<sub>s</sub> i<sub>ts</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>expectat</sub>i<sub>on.</sub> B<sub>y</sub> <sub>t</sub>h<sub>e</sub> l<sub>aw</sub> <sub>o</sub>f <sub>tota</sub>l <sub>var</sub>i<sub>ance,</sub> <sub>t</sub>hi<sub>s</sub> <sub>opt</sub>i<sub>ma</sub>l d<sub>eco</sub>d<sub>er</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> <sub>t</sub>h<sub>e</sub> <sub>mean</sub> <sub>square</sub>d <sub>error:</sub>

$$
\begin{array} { r } { g _ { f } ^ { \star } ( \mathcal { O } ) = \mathbb { E } [ f ( D ) \mid \mathcal { O } ] , \qquad \operatorname* { i n f } _ { g } \mathbb { E } \left[ ( f ( D ) - g ( \mathcal { O } ) ) ^ { 2 } \right] = \operatorname { V a r } ( f ( D ) ) - \operatorname { V a r } \big ( \mathbb { E } [ f ( D ) \mid \mathcal { O } ] \big ) . } \end{array}
$$

<sup>Th</sup>e con<sup>di</sup>t<sup>i</sup>ona<sup>l</sup> expectat<sup>i</sup>on ort<sup>h</sup>ogona<sup>ll</sup>y projects $f ( D )$ <sub>onto t</sub>h<sub>e c</sub>l<sub>ose</sub>d <sub>su</sub>b<sub>space o</sub>f <sub>square-</sub>i<sub>ntegra</sub>bl<sub>e</sub> <sup>f</sup>unct<sup>i</sup>ons o<sup>f</sup> t<sup>h</sup>e s<sup>k</sup>etc<sup>h</sup>. <sup>Thi</sup>s project<sup>i</sup>on sp<sup>li</sup>ts t<sup>h</sup>e <sup>f</sup>eature<sup>’</sup>s var<sup>i</sup>ance <sup>i</sup>nto a recovere<sup>d</sup> part an<sup>d</sup> an ort<sup>h</sup>ogona<sup>l</sup> residual. We measure this recovery using the recoverable variance fraction:

$$
\mathsf { R e c } ( f ; \mathcal { O } ) = \frac { \mathrm { V a r } \left( \mathbb { E } [ f ( D ) \mid \mathcal { O } ] \right) } { \mathrm { V a r } ( f ( D ) ) } \in [ 0 , 1 ] .
$$

Thi<sub>s</sub> f<sub>ract</sub>i<sub>on</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> <sub>t</sub>h<sub>e</sub> f<sub>eature</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> <sub>o</sub>b<sub>servat</sub>i<sub>on,</sub> <sub>not</sub> <sub>on</sub> <sub>t</sub>h<sub>e</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f <sub>recovery</sub> <sub>a</sub>l<sub>gor</sub>i<sub>t</sub>h<sub>m.</sub>

<sup>Th</sup>e cr<sup>i</sup>ter<sup>i</sup>on separates w<sup>h</sup>at t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> con<sup>fl</sup>ates. <sup>F</sup>or t<sup>h</sup>e rep<sup>l</sup>acement c<sup>l</sup>ou<sup>d</sup> o<sup>f</sup> propos<sup>i</sup>t<sup>i</sup>on 3.1, in<sup>d</sup>e<sub>p</sub>en<sup>d</sup>ence <sub>g</sub>ives $\dot { \mathbb { E } } [ f ( D ) \mid Y ^ { n } ] = \mathbb { E } \breve { f } ( D )$ , <sup>h</sup>ence Rec $( f ; Y ^ { n } ) = 0$ <sup>f</sup>or every <sup>f</sup>eature: t<sup>h</sup>e J<sup>L</sup> event can coex<sup>i</sup>st w<sup>i</sup>t<sup>h</sup> not<sup>hi</sup>ng recovera<sup>bl</sup>e. <sup>A</sup> <sup>li</sup>near s<sup>k</sup>etc<sup>h</sup> o<sup>f</sup> t<sup>h</sup>e <sup>d</sup>ata <sup>i</sup>nstea<sup>d</sup> y<sup>i</sup>e<sup>ld</sup>s a nonzero project<sup>i</sup>on. <sup>Th</sup>e supremum o<sup>f</sup> $\mathsf { R e c } ( f ; { \mathcal { O } } )$ over <sup>f</sup>eatures <sup>i</sup>s t<sup>h</sup>ere<sup>f</sup>ore t<sup>h</sup>e o<sup>b</sup>ject o<sup>f</sup> <sup>i</sup>nterest, an<sup>d</sup> <sup>i</sup>t <sup>i</sup>s an operator norm.

## 4.2 The distance-recovery operator

Gathering all centered, square-integrable features of D into the space $L _ { 0 } ^ { 2 } ( D )$ <sup>frames</sup> <sup>recover</sup>y <sup>as</sup> <sup>an</sup> <sup>o</sup>p<sup>erator</sup> problem. We define the recovery operator

$$
\begin{array} { r } { \mathcal { A } : L _ { 0 } ^ { 2 } ( D ) \longrightarrow L _ { 0 } ^ { 2 } ( \mathcal { O } ) , \qquad \mathcal { A } f = \mathbb { E } [ f ( D ) \mid \mathcal { O } ] , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $L _ { 0 } ^ { 2 } ( \mathcal { O } )$ denotes the centered, s<sub>q</sub>uare-inte<sub>g</sub>rable functions of the observation. Because A is a restricted ort<sup>h</sup>ogona<sup>l</sup> project<sup>i</sup>on, <sup>i</sup>t never <sup>i</sup>ncreases t<sup>h</sup>e $L ^ { 2 }$ norm, <sub>y</sub>ie<sup>ld</sup>in<sub>g</sub>

$$
\mathsf { R e c } ( f ; \mathcal { O } ) = \frac { \| \mathcal { A } f \| _ { 2 } ^ { 2 } } { \| f \| _ { 2 } ^ { 2 } } .
$$

T<sup>h</sup>e s<sub>q</sub>uare<sup>d</sup> o<sub>p</sub>erator norm $\| \mathcal { A } \| _ { \mathrm { o p } } ^ { 2 }$ <sub>represents t</sub>h<sub>e max</sub>i<sub>mum recovera</sub>bl<sub>e</sub> f<sub>ract</sub>i<sub>on over a</sub>ll f<sub>eatures.</sub> Thi<sub>s</sub> norm also e<sub>q</sub>uals the s<sub>q</sub>uared Hirschfeld–Gebelein–Rén<sub>y</sub>i (HGR) maximal correlation [21, 22], which <sub>measures</sub> <sub>t</sub>h<sub>e</sub> <sub>max</sub>i<sub>mum</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>o</sub>b<sub>ta</sub>i<sub>na</sub>bl<sub>e</sub> b<sub>y</sub> <sub>trans</sub>f<sub>orm</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> <sub>target</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> <sub>o</sub>b<sub>servat</sub>i<sub>on:</sub>

$$
\| A \| _ { \mathrm { o p } } ^ { 2 } = \operatorname* { s u p } _ { f \in L _ { \mathsf { f } } ^ { 2 } ( D ) } \frac { \| A f \| _ { 2 } ^ { 2 } } { \| f \| _ { 2 } ^ { 2 } } = \mathsf { \rho } _ { \mathrm { H G R } } ^ { 2 } ( D ; \mathcal { O } ) , \qquad \mathsf { \rho } _ { \mathsf { H G R } } ( Y ; Z ) = \operatorname* { s u p } _ { f , g } \left| \mathrm { C o r r } ( f ( Y ) , g ( Z ) ) \right| .\tag{4}
$$

H<sub>ere,</sub> <sub>t</sub>h<sub>e</sub> l<sub>ast</sub> <sub>supremum</sub> i<sub>s</sub> <sub>over</sub> <sub>a</sub>ll <sub>measura</sub>bl<sub>e,</sub> <sub>nonconstant,</sub> <sub>square-</sub>i<sub>ntegra</sub>bl<sub>e</sub> <sub>sca</sub>l<sub>ar</sub> f<sub>unct</sub>i<sub>ons.</sub> O<sub>pt</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> first over g <sub>y</sub>ields the normalized conditional variance<sub>,</sub> which confirms this identit<sub>y</sub> [21]. Conse<sub>q</sub>uentl<sub>y,</sub> no estimator can recover a variance fraction of any feature of D that exceeds $\| \mathcal { A } \| _ { \mathrm { o p } } ^ { 2 }$

Th<sub>e</sub> <sub>operator</sub> d<sub>escr</sub>ib<sub>es</sub> <sub>recovery</sub> <sub>at</sub> <sub>t</sub>h<sub>ree</sub> l<sub>eve</sub>l<sub>s:</sub>

$$
\begin{array} { r l } & { \mathrm { o n e ~ f e a t u r e } ; \qquad \quad \lVert A f \rVert _ { 2 } ^ { 2 } / \left. f \right. _ { 2 } ^ { 2 } , } \\ & { \mathrm { t h e ~ b e s t ~ f e a t u r e } ; \qquad \left. A \right. _ { \mathrm { o p } } ^ { 2 } , } \\ & { \mathrm { a l l ~ f e a t u r e s ~ a t ~ o n c e } { \mathrm { : } } \quad \mathrm { t h e ~ s i n g u l a r ~ s y s t e m ~ o f } \ : A . } \end{array}
$$

C<sub>onstant</sub> f<sub>eatures</sub> <sub>are</sub> <sub>recovere</sub>d <sub>per</sub>f<sub>ect</sub>l<sub>y</sub> <sub>an</sub>d f<sub>orm</sub> <sub>a</sub> <sub>tr</sub>i<sub>v</sub>i<sub>a</sub>l <sub>mo</sub>d<sub>e.</sub> C<sub>enter</sub>i<sub>ng</sub> <sub>removes</sub> <sub>t</sub>hi<sub>s</sub> <sub>mo</sub>d<sub>e,</sub> <sub>ensur</sub>i<sub>ng</sub> <sub>t</sub>h<sub>at t</sub>h<sub>e operator on</sub> $L _ { 0 } ^ { 2 } ( D )$ <sub>captures</sub> <sub>on</sub>l<sub>y</sub> <sub>t</sub>h<sub>e</sub> <sub>geometr</sub>i<sub>c</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on.</sub> Whil<sub>e</sub> <sub>t</sub>h<sub>e</sub> <sub>operator</sub> <sub>norm</sub> i<sub>s</sub> <sub>a</sub>l<sub>ways</sub> d<sub>e</sub>fi<sub>ne</sub>d<sub>,</sub> <sub>a</sub> di<sub>screte</sub> <sub>s</sub>i<sub>ngu</sub>l<sub>ar</sub> <sub>system</sub> <sub>ex</sub>i<sub>sts</sub> <sub>on</sub>l<sub>y</sub> f<sub>or</sub> <sub>spec</sub>ifi<sub>c</sub> <sub>c</sub>h<sub>anne</sub>l<sub>s.</sub> Th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>c</sub>h<sub>anne</sub>l <sub>we</sub> <sub>construct</sub> <sub>next</sub> <sub>a</sub>d<sub>m</sub>i<sub>ts suc</sub>h <sub>a system</sub> i<sub>n c</sub>l<sub>ose</sub>d f<sub>orm.</sub>

## 4.3 The Gaussian distance channel

We identify the concrete operator by reducing the sketch to a scalar channel. The first step, row orthonormalization, is a reversible output transformation that preserves all information in a noiseless linear observation by a known map. Let a rank-r map L have thin singular value decomposition $\ b { L } = \ b { U _ { r } } \ b { S _ { r } } \ b { V _ { r } } ^ { \top }$ an<sup>d</sup> <sub>p</sub>ut $R = V _ { r } ^ { \top }$ Di<sub>scar</sub>di<sub>ng</sub> <sub>re</sub>d<sub>un</sub>d<sub>ant</sub> <sub>output</sub> <sub>coor</sub>di<sub>nates</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
L x = U _ { r } S _ { r } R x , \qquad R x = S _ { r } ^ { - 1 } U _ { r } ^ { \top } L x .
$$

B<sub>ecause</sub> $S _ { r }$ is invertible, Lx and Rx uni uel determine each other on the nonredundant out ut. If L has full <sub>row ran</sub>k<sub>, we can</sub> i<sub>nstea</sub>d <sub>ta</sub>k<sub>e</sub> $R = ( L L ^ { \top } ) ^ { - 1 / 2 } L$ . Here r denotes the rank of $\dot { L } ;$ f<sub>or</sub> f<sub>u</sub>ll<sub>-row-ran</sub>k <sub>s</sub>k<sub>etc</sub>h<sub>es</sub> with m rows, $r = m$

L<sub>e</sub>t $\Sigma = \sigma ^ { 2 } I _ { d } .$ , and let L have rank $0 < r < d .$ U<sub>n</sub>d<sub>er</sub> <sub>t</sub>hi<sub>s</sub> i<sub>sotrop</sub>i<sub>c</sub> <sub>covar</sub>i<sub>ance,</sub> <sub>row</sub> <sub>ort</sub>h<sub>onorma</sub>li<sub>zat</sub>i<sub>on</sub> replaces the sketch with the equivalent pair (ΠX, ΠX<sup>′</sup>), where Π is a rank-r orthogonal projector, and by rotationa<sup>l</sup> invariance t<sup>h</sup>e <sup>l</sup>aw o<sup>f</sup> t<sup>h</sup>is <sub>p</sub>air <sup>d</sup>e<sub>p</sub>en<sup>d</sup>s on Π on<sup>l</sup><sub>y</sub> t<sup>h</sup>rou<sub>g</sub><sup>h</sup> its ran<sup>k</sup>. For a ran<sup>d</sup>om o<sup>bl</sup>ivious ma<sub>p</sub>, the row space is therefore all that matters. A random r-dimensional subspace of $\mathbb { R } ^ { d }$ is Haar distributed if i<sub>t</sub> i<sub>s un</sub>if<sub>orm over t</sub>h<sub>e</sub> G<sub>rassmann</sub>i<sub>an, t</sub>h<sub>e set o</sub>f <sub>a</sub>ll <sub>suc</sub>h <sub>su</sub>b<sub>spaces; represent</sub> i<sub>t</sub> b<sub>y an</sub> $r \times d$ matrix R with ort<sup>h</sup>onorma<sup>l</sup> rows an<sup>d</sup> projector $\Pi = R ^ { \top } R$ <sub>.</sub> O<sub>rt</sub>h<sub>onorma</sub>li<sub>z</sub>i<sub>ng t</sub>h<sub>e rows o</sub>f <sub>an</sub> i<sub>.</sub>i<sub>.</sub>d<sub>. stan</sub>d<sub>ar</sub>d G<sub>auss</sub>i<sub>an</sub> <sub>matr</sub>i<sub>x y</sub>i<sub>e</sub>ld<sub>s t</sub>hi<sub>s same</sub> H<sub>aar row-space</sub> di<sub>str</sub>ib<sub>ut</sub>i<sub>on, a stan</sub>d<sub>ar</sub>d <sub>construct</sub>i<sub>on</sub> i<sub>n ran</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>numer</sub>i<sub>ca</sub>l li<sub>near</sub> al<sub>g</sub>ebra [7]: ri<sub>g</sub>ht multi<sub>p</sub>lication b<sub>y</sub> a fixed ortho<sub>g</sub>onal matrix <sub>p</sub>reserves the Gaussian law<sub>,</sub> which makes the <sub>row space un</sub>if<sub>orm on t</sub>h<sub>e</sub> G<sub>rassmann</sub>i<sub>an.</sub>

S<sub>tan</sub>d<sub>ar</sub>di<sub>ze t</sub>h<sub>e</sub> dif<sub>erence as</sub>

$$
Z = \frac { X - X ^ { \prime } } { \sqrt { 2 } \sigma } \sim { \mathcal { N } } ( 0 , I _ { d } ) , \qquad T = \| Z \| ^ { 2 } , \qquad U = \| \Pi Z \| ^ { 2 } , \qquad V = { \mathcal { T } } - U .
$$

Th<sub>en</sub> $U \sim \chi _ { r } ^ { 2 } , V \sim \chi _ { d - r } ^ { 2 } , U \perp V ,$ <sub>,</sub> <sub>an</sub>d $D = 2 \sigma ^ { 2 } \mathcal { T }$

Three independence properties reduce the full sketch to the scalar U. First, the Gaussian midpoint $\left( X + X ^ { \prime } \right) / 2$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent o</sub>f ${ \bar { X } } - X ^ { \prime }$ , so its rojection carries no in<sup>f</sup>ormation a<sup>b</sup>out D. Secon<sup>d</sup>, rotationa<sup>l</sup> invariance makes the direction of ΠZ uniform on the unit s<sub>p</sub>here of ran<sub>g</sub>e(Π) and inde<sub>p</sub>endent of its len<sub>g</sub>th $U ^ { 1 / 2 }$ . Third, the orthogonal Gaussian components ΠZ and $( I - \Pi ) Z$ are independent, so V is independent <sub>o</sub>f <sub>t</sub>h<sub>e</sub> <sub>ent</sub>i<sub>re</sub> <sub>reta</sub>i<sub>ne</sub>d <sub>vector.</sub> Th<sub>ere</sub>f<sub>ore,</sub> <sub>t</sub>h<sub>e</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l l<sub>aw</sub> <sub>o</sub>f $\tau$ <sub>g</sub>i<sub>ven</sub> <sub>t</sub>h<sub>e</sub> f<sub>u</sub>ll <sub>s</sub>k<sub>etc</sub>h d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> $U ;$ i<sub>n</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>st</sub>i<sub>c</sub> <sub>notat</sub>i<sub>on,</sub> $D \bot \mathcal { O } \mid U$ <sub>.</sub> Fi<sub>gure</sub> 2 <sub>summar</sub>i<sub>zes t</sub>hi<sub>s re</sub>d<sub>uct</sub>i<sub>on.</sub>

S<sub>tan</sub>d<sub>ar</sub>d <sub>gamma-sum</sub> id<sub>ent</sub>i<sub>t</sub>i<sub>es</sub> <sub>now</sub> <sub>spec</sub>if<sub>y</sub> <sub>t</sub>h<sub>e</sub> <sub>c</sub>h<sub>anne</sub>l<sub>.</sub> If $U \sim \mathrm { G a m m a } ( a , \theta )$ <sub>an</sub>d $V \sim { \mathrm { G a m m a } } ( b , \theta )$ <sub>are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent,</sub> <sub>t</sub>h<sub>en</sub>

$$
\mathcal { T } = U + V \sim \mathrm { G a m m a } ( a + b , \theta ) , \qquad \frac { U } { \mathcal { T } } \sim \mathrm { B e t a } ( a , b ) , \qquad \frac { U } { \mathcal { T } } \perp \mathcal { T } .
$$

![](images/ee4d70477bc6af7c6ccfe978ba6eb92472d8080ab3be7b8480273c454b94c549.jpg)  
Figure 2. The distance channel. A rank-r sketch splits the squared norm of a standardized Gaussian diference into observed and omitted parts. Given U, the rest of the sketch provides no further information about the distance $D = 2 \sigma ^ { 2 } ( U + V )$

H<sub>ere</sub> $a = r / 2 , b = ( d - r ) / 2$ <sub>,</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> <sub>common</sub> <sub>sca</sub>l<sub>e</sub> i<sub>s</sub> $\theta = 2$ for the chi-squared laws above. Observing U f<sub>rom t</sub>h<sub>e</sub> l<sub>atent tota</sub>l ${ \mathcal { T } } = U + V$ forms the beta–gamma channel.

Th<sub>e</sub> di<sub>stance-recovery pro</sub>bl<sub>em</sub> i<sub>s now re</sub>d<sub>uce</sub>d <sub>to t</sub>h<sub>e con</sub>di<sub>t</sub>i<sub>ona</sub>l<sub>-expectat</sub>i<sub>on operator o</sub>f <sub>t</sub>h<sub>e</sub> b<sub>eta–</sub> <sub>gamma</sub> <sub>c</sub>h<sub>anne</sub>l<sub>:</sub> <sub>s</sub>i<sub>nce</sub> $D \perp \mathcal { O } \mid U$ <sub>an</sub>d $D = 2 \sigma ^ { 2 } T$ , the o<sub>p</sub>erator A acts throu<sub>g</sub>h $h ( \mathcal { \bar { T } } ) \mapsto \mathbb { E } [ h ( \mathcal { T } ) \ |$ | U]. <sup>Section</sup> <sup>5</sup> <sup>com</sup>p<sup>utes</sup> <sup>its</sup> <sup>o</sup>p<sup>erator</sup> <sup>norm</sup> <sup>and</sup> <sup>its</sup> <sup>com</sup>p<sup>lete</sup> <sup>sin</sup>g<sup>ular</sup> <sup>s</sup>p<sup>ectrum.</sup>

## 4.4 How the framework recurs

Th<sub>e ran</sub>ki<sub>ng an</sub>d <sub>covar</sub>i<sub>ance-s</sub>h<sub>ape pro</sub>bl<sub>ems rep</sub>l<sub>ace t</sub>h<sub>e pa</sub>i<sub>r</sub> $( L _ { 0 } ^ { 2 } ( D ) , L _ { 0 } ^ { 2 } ( { \mathcal { O } } ) )$ b<sub>y ot</sub>h<sub>er target an</sub>d <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> spaces. <sup>O</sup>ne <sup>f</sup>ormu<sup>l</sup>at<sup>i</sup>on—con<sup>di</sup>t<sup>i</sup>ona<sup>l</sup> expectat<sup>i</sup>on as ort<sup>h</sup>ogona<sup>l</sup> project<sup>i</sup>on—covers a<sup>ll</sup> t<sup>h</sup>ree pro<sup>bl</sup>ems; t<sup>h</sup>e operator itself difers with each tar<sub>g</sub>et. Throu<sub>g</sub>hout, d is the ambient dimension, m the output dimension, and r the map rank; n is the sample size and q the number of candidates compared to a query; D denotes a squared distance, S a shared-query distance contrast, and O the observed sketch. Three derived quantities recur: ρ<sub>HGR</sub> is t<sup>h</sup>e maxima<sup>l</sup> corre<sup>l</sup>ation <sup>b</sup>etween a tar<sub>g</sub>et an<sup>d</sup> an o<sup>b</sup>servation, $r _ { 2 } ( \Sigma ) = ( \mathrm { t r } \Sigma ) ^ { 2 } / \mathrm { t r } ( \Sigma ^ { 2 } )$ i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>e</sub>f<sub>ect</sub>i<sub>ve</sub> <sub>ran</sub>k <sub>o</sub>f <sub>t</sub>h<sub>e</sub> <sub>spectrum,</sub> <sub>an</sub>d $\ell _ { k }$ i<sub>n</sub>d<sub>exes</sub> <sub>t</sub>h<sub>e</sub> <sub>s</sub>i<sub>ngu</sub>l<sub>ar</sub> <sub>va</sub>l<sub>ues</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> di<sub>stance-recovery</sub> <sub>operator.</sub> S<sub>ect</sub>i<sub>ons</sub> 6 <sub>an</sub>d b<sub>eyon</sub>d i<sub>ntro</sub>d<sub>uce</sub> <sub>t</sub>h<sub>e</sub> <sub>recovery</sub> f<sub>ract</sub>i<sub>ons</sub> $\alpha _ { m } ( \Sigma )$ <sub>an</sub>d $\alpha _ { \Pi } ( \Sigma )$ <sub>an</sub>d <sub>t</sub>h<sub>e</sub> bl<sub>oc</sub>k <sub>retent</sub>i<sub>on rat</sub>i<sub>os</sub> $\theta _ { g }$ <sub>w</sub>h<sub>ere</sub> <sub>t</sub>h<sub>ey</sub> <sub>are</sub> fi<sub>rst</sub> <sub>use</sub>d<sub>.</sub>

<sup>F</sup>or <sup>di</sup>stance <sup>f</sup>eatures, sect<sup>i</sup>on 5 so<sup>l</sup>ves t<sup>h</sup>e operator just constructe<sup>d</sup>: t<sup>h</sup>e centere<sup>d</sup> <sup>di</sup>stance atta<sup>i</sup>ns t<sup>h</sup>e <sup>o</sup>p<sup>erator</sup> <sup>norm</sup> ${ \sqrt { r / d } } ,$ <sub>, an</sub>d <sub>t</sub>h<sub>e</sub> f<sub>u</sub>ll <sub>s</sub>i<sub>ngu</sub>l<sub>ar system</sub> i<sub>s a genera</sub>li<sub>ze</sub>d L<sub>aguerre</sub> f<sub>am</sub>il<sub>y.</sub> S<sub>ect</sub>i<sub>on</sub> 6 <sub>t</sub>h<sub>en treats</sub> <sub>genera</sub>l <sub>covar</sub>i<sub>ance,</sub> <sub>w</sub>h<sub>ere</sub> <sub>recover</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> di<sub>stance</sub> <sub>va</sub>l<sub>ue</sub> <sub>an</sub>d <sub>recover</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> b<sub>est</sub> <sub>non</sub>li<sub>near</sub> f<sub>eature</sub> <sub>can</sub> dif<sub>er.</sub> The recover<sub>y</sub> scale is a variance fraction of order m/d, a <sub>q</sub>uantit<sub>y</sub> the JL bound does not control (section 2).

F<sub>or</sub> <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub> <sub>ran</sub>ki<sub>ngs,</sub> <sub>t</sub>h<sub>e</sub> <sub>target</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>s</sub>h<sub>are</sub>d<sub>-query</sub> <sub>contrast</sub> $S = D _ { 0 1 } - D _ { 0 2 }$ , or its si<sub>g</sub>n, an<sup>d</sup> we <sub>o</sub>b<sub>serve</sub> $\mathcal { O } = \left( L X _ { 0 } , L X _ { 1 } , L \bar { X } _ { 2 } \right)$ <sub>.</sub> Th<sub>e</sub> <sub>target</sub> i<sub>s</sub> <sub>a</sub> f<sub>unct</sub>i<sub>on</sub> <sub>o</sub>f <sub>two</sub> d<sub>epen</sub>d<sub>ent</sub> di<sub>stances,</sub> <sub>so</sub> <sub>t</sub>h<sub>e</sub> <sub>sca</sub>l<sub>ar</sub> <sub>operator</sub> i<sub>s</sub> rep<sup>l</sup>ace<sup>d</sup> <sup>b</sup>y a project<sup>i</sup>on o<sup>f</sup> t<sup>h</sup>e jo<sup>i</sup>nt contrast; we ca<sup>l</sup>cu<sup>l</sup>ate t<sup>h</sup>e exact s<sup>i</sup>gn <sup>l</sup>aw <sup>f</sup>rom a con<sup>di</sup>t<sup>i</sup>ona<sup>l</sup> <sup>G</sup>auss<sup>i</sup>an calculation (section 7). Si<sub>g</sub>n statistics inherit the correlation scale m/d rather than the variance-fraction scale m/d.

F<sub>or</sub> <sub>covar</sub>i<sub>ance</sub> <sub>s</sub>h<sub>ape,</sub> <sub>t</sub>h<sub>e</sub> <sub>re</sub>l<sub>evant</sub> Hilb<sub>ert</sub> <sub>space</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>score</sub> <sub>space,</sub> <sub>an</sub>d <sub>remov</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> <sub>un</sub>k<sub>nown</sub> overa<sup>ll</sup> sca<sup>l</sup>e <sup>i</sup>s an ort<sup>h</sup>ogona<sup>l</sup> project<sup>i</sup>on o<sup>f</sup> t<sup>h</sup>e s<sup>h</sup>ape score onto t<sup>h</sup>e ort<sup>h</sup>ogona<sup>l</sup> comp<sup>l</sup>ement o<sup>f</sup> t<sup>h</sup>e nu<sup>i</sup>sance <sub>sca</sub>l<sub>e</sub> <sub>score.</sub> Th<sub>e</sub> <sub>resu</sub>l<sub>t</sub>i<sub>ng</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ent</sub> Fi<sub>s</sub>h<sub>er</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>contracts</sub> <sub>at</sub> <sub>or</sub>d<sub>er</sub> $( m / d ) ^ { 2 }$ f<sub>or</sub> dif<sub>use</sub> <sub>s</sub>h<sub>ape</sub> di<sub>rect</sub>i<sub>ons</sub> (section 8).

<sup>Th</sup>e exponents <sup>dif</sup>er <sup>b</sup>ecause t<sup>h</sup>e t<sup>h</sup>ree rows measure <sup>dif</sup>erent o<sup>b</sup>jects. <sup>V</sup>ar<sup>i</sup>ance recovery squares an $L ^ { 2 }$ <sub>corre</sub>l<sub>at</sub>i<sub>on, ran</sub>k <sub>compar</sub>i<sub>sons</sub> d<sub>epen</sub>d <sub>on a s</sub>i<sub>gn pro</sub>b<sub>a</sub>bili<sub>ty an</sub>d <sub>t</sub>h<sub>ere</sub>f<sub>ore on t</sub>h<sub>e corre</sub>l<sub>at</sub>i<sub>on</sub> i<sub>tse</sub>lf<sub>, an</sub>d <sub>a</sub> covar<sup>i</sup>ance pertur<sup>b</sup>at<sup>i</sup>on <sup>i</sup>s compresse<sup>d</sup> on <sup>b</sup>ot<sup>h</sup> s<sup>id</sup>es <sup>b</sup>y t<sup>h</sup>e ran<sup>d</sup>om projector.

## 5. Exact Solution of the Gaussian Distance Operator

For the beta–<sub>g</sub>amma channel of section 4.3, the recover<sub>y</sub> o<sub>p</sub>erator A of section 4.2 acts throu<sub>g</sub>h the scalar <sub>p</sub>air (U, T ). This section com<sub>p</sub>utes its o<sub>p</sub>erator norm, then its com<sub>p</sub>lete sin<sub>g</sub>ular s<sub>y</sub>stem, and derives the

Table 1. Guide to the three recovery laws. Each law is stated in the units of its recovery quantity.
<table><tr><td>Target</td><td>Recovery quantity</td><td>Recovery law</td></tr><tr><td>Distance features, isotropic recoverable variance fraction</td><td></td><td> $m / d$ </td></tr><tr><td>Neighbor rankings</td><td>excess pairwise agreement over chance</td><td> $\pi ^ { - 1 } { \sqrt { m / d } } \left( 1 + o { \left( 1 \right) } \right)$ </td></tr><tr><td>Diffuse covariance shape</td><td>Haar-mean efficient-information fraction</td><td> $\frac { ( m - 1 ) ( m + 2 ) } { ( d - 1 ) ( d + 2 ) } \sim ( m / d ) ^ { 2 }$ </td></tr></table>

<sub>c</sub>h<sub>anne</sub>l’<sub>s</sub> <sub>mutua</sub>l i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>an</sub>d id<sub>ent</sub>i<sub>t</sub>i<sub>es</sub> f<sub>or</sub> <sub>t</sub>h<sub>e</sub> di<sub>stance.</sub>

## 5.1 The operator norm

The norm follows from a result on <sub>p</sub>artial sums. Dembo<sub>,</sub> Ka<sub>g</sub>an<sub>,</sub> and She<sub>pp</sub> [22] <sub>p</sub>roved that the maximal correlation between a partial sum of r i.i.d. nondegenerate finite-variance variables and a containing sum of d such variables is ${ \sqrt { r / d } } .$ <sub>.</sub> Th<sub>e c</sub>hi<sub>-square var</sub>i<sub>a</sub>bl<sub>es o</sub>f <sub>sect</sub>i<sub>on</sub> 4<sub>.</sub>3 <sub>sat</sub>i<sub>s</sub>f<sub>y t</sub>h<sub>ese</sub> h<sub>ypot</sub>h<sub>eses.</sub> O<sub>r</sub>di<sub>nary</sub> li<sub>near</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> <sub>equa</sub>l<sub>s</sub> ${ \sqrt { r / d } } ;$ <sub>t</sub>h<sub>e t</sub>h<sub>eorem says t</sub>h<sub>at non</sub>li<sub>near trans</sub>f<sub>ormat</sub>i<sub>ons o</sub>f <sub>t</sub>h<sub>e part</sub>i<sub>a</sub>l <sub>an</sub>d <sub>tota</sub>l <sup>sums</sup> <sup>cannot</sup> <sup>im</sup>p<sup>rove</sup> <sup>it.</sup>

Theorem 5.1 (Operator norm and maximal correlation). For the isotropic component and any rank-r linear observation,

$$
\left\| \mathbf { \mathcal { A } } \right\| _ { \mathrm { o p } } = \rho _ { \mathrm { H G R } } ( D ; \mathcal { O } ) = \sqrt { \frac { r } { d } } .
$$

Consequently, $\mathrm { V a r } ( \mathbb { E } [ f ( D ) \mid \mathcal { O } ] ) \leq ( r / d ) \mathrm { V a r } ( f ( D ) ) f o r e \nu e r \gamma f \in L ^ { 2 } ,$ and

$$
\operatorname* { i n f } _ { g } \mathbb { E } [ ( f ( D ) - g ( \mathcal { O } ) ) ^ { 2 } ] \geq ( 1 - r / d ) \operatorname { V a r } ( f ( D ) ) .
$$

The <sub>p</sub>roof(section B) writes T as a sum ofd inde<sub>p</sub>endent $\chi _ { 1 } ^ { 2 }$ variables and U as the partial sum ofthe first r. The Dembo–Kagan–Shepp theorem gives maximal correlation $\sqrt { r / d }$ for this pair. Because U is suficient for T —the rest of the sketch carries no further information about the distance—the identit<sub>y</sub> transfers to the f<sub>u</sub>ll <sub>s</sub>k<sub>etc</sub>h<sub>.</sub> Th<sub>us</sub> <sub>t</sub>h<sub>res</sub>h<sub>o</sub>ld<sub>s,</sub> <sub>quant</sub>il<sub>es,</sub> <sub>c</sub>li<sub>ppe</sub>d di<sub>stances,</sub> <sub>an</sub>d <sub>every</sub> <sub>ot</sub>h<sub>er</sub> <sub>square-</sub>i<sub>ntegra</sub>bl<sub>e</sub> <sub>sca</sub>l<sub>ar</sub> f<sub>eature</sub> <sub>o</sub>b<sub>ey</sub> <sub>t</sub>h<sub>e</sub> <sub>same</sub> d<sub>eco</sub>d<sub>er-</sub>f<sub>ree</sub> <sub>ce</sub>ili<sub>ng.</sub> M<sub>utua</sub>l i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>comp</sub>l<sub>ements</sub> <sub>t</sub>hi<sub>s</sub> <sub>ce</sub>ili<sub>ng</sub> b<sub>y</sub> <sub>measur</sub>i<sub>ng</sub> <sub>tota</sub>l de<sub>p</sub>endence [23]<sub>,</sub> but neither <sub>q</sub>uantit<sub>y</sub> resolves recover<sub>y</sub> mode b<sub>y</sub> mode<sub>;</sub> the sin<sub>g</sub>ular s<sub>y</sub>stem does.

## 5.2 The singular spectrum

Th<sub>e gamma</sub> l<sub>aw</sub> i<sub>s t</sub>h<sub>e ort</sub>h<sub>ogona</sub>li<sub>ty measure</sub> f<sub>or t</sub>h<sub>e genera</sub>li<sub>ze</sub>d L<sub>aguerre po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>s, an</sub>d G<sub>r</sub>ifi<sub>t</sub>h<sub>s</sub> [24] dia<sub>g</sub>onalized the <sub>g</sub>amma conditional-ex<sub>p</sub>ectation o<sub>p</sub>erator $h ( \mathcal { T } ) \mapsto \mathbb { E } [ h ( \mathcal { T } ) \mid \dot { U } ]$ i<sub>n t</sub>h<sub>at</sub> b<sub>as</sub>i<sub>s.</sub> F<sub>or t</sub>h<sub>e</sub> b<sub>eta–gamma</sub> <sub>pa</sub>i<sub>r</sub> <sub>o</sub>f <sub>sect</sub>i<sub>on</sub> 4<sub>.</sub>3<sub>,</sub> <sub>t</sub>hi<sub>s</sub> <sub>system</sub> <sub>reso</sub>l<sub>ves</sub> <sub>recovera</sub>bl<sub>e</sub> <sub>var</sub>i<sub>ance</sub> <sub>one</sub> fl<sub>uctuat</sub>i<sub>on</sub> <sub>component</sub> <sub>at</sub> <sub>a</sub> ti<sub>me.</sub>

Theorem 5.2 (Laguerre spectrum). The conditional-expectation operator $f ( \mathcal { T } ) \mapsto \mathbb { E } [ f ( \mathcal { T } ) \mid$ U] has generalized Laguerre singular functions and singular values

$$
\ell _ { k } = \left( \frac { \left( r / 2 \right) _ { k } } { \left( d / 2 \right) _ { k } } \right) ^ { 1 / 2 } , \qquad k = 0 , 1 , 2 , \ldots ,
$$

where $( a ) _ { k }$ is the rising factorial [24]. $\begin{array} { r } { I f f ( T ) - \mathbb { E } f ( T ) = \sum _ { k \geq 1 } c _ { k } \Phi _ { k } ( T ) } \end{array}$ is the orthonormal Laguerre expansion,

then

$$
\operatorname { V a r } ( \mathbb { E } [ f ( \mathcal { T } ) \mid \mathcal { O } ] ) = \sum _ { k \geq 1 } \ell _ { k } ^ { 2 } c _ { k } ^ { 2 } .
$$

For fixed $k ,$ as r → ∞ and $d - r  \infty$

$$
\ell _ { k } = \left( \frac { r } { d } \right) ^ { k / 2 } \left[ 1 + O _ { k } \left( \frac { 1 } { r } + \frac { 1 } { d } \right) \right] .
$$

Thi<sub>s</sub> i<sub>s</sub> G<sub>r</sub>ifi<sub>t</sub>h<sub>s</sub>’<sub>s</sub> di<sub>agona</sub>li<sub>zat</sub>i<sub>on</sub> <sub>spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>to</sub> <sub>t</sub>h<sub>e</sub> <sub>gamma</sub> <sub>parameters</sub> $a = r / 2$ <sub>an</sub>d $a + b = d / 2 ;$ <sub>t</sub>h<sub>e</sub> <sub>proo</sub>f<sub>,</sub> <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>at</sub> <sub>o</sub>f <sub>t</sub>h<sub>eorem</sub> $5 . 3 .$ <sub>,</sub> <sub>appears</sub> i<sub>n</sub> <sub>sect</sub>i<sub>on</sub> B<sub>.</sub> Th<sub>e</sub> fi<sub>rst</sub> <sub>nonconstant</sub> <sub>mo</sub>d<sub>e</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>centere</sub>d <sub>tota</sub>l <sub>energy</sub> $\tau _ { - \\mathbb { E } } \tau$ <sub>an</sub>d h<sub>as</sub> $\ell _ { 1 } = \sqrt { r / d } = \| \mathcal { A } \| _ { \mathrm { o p } } .$ <sub>,</sub> <sub>so</sub> <sub>t</sub>h<sub>e</sub> <sub>operator</sub> <sub>norm</sub> <sub>o</sub>f <sub>t</sub>h<sub>eorem</sub> 5<sub>.</sub>1 i<sub>s</sub> <sub>atta</sub>i<sub>ne</sub>d b<sub>y</sub> <sub>t</sub>h<sub>e</sub> <sub>centere</sub>d di<sub>stance</sub> i<sub>tse</sub>lf<sub>.</sub> Th<sub>e</sub> <sub>secon</sub>d <sub>mo</sub>d<sub>e</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>ort</sub>h<sub>ogona</sub>li<sub>ze</sub>d <sub>qua</sub>d<sub>rat</sub>i<sub>c</sub> fl<sub>uctuat</sub>i<sub>on</sub> <sub>o</sub>f $\tau .$ . Thus k indexes increasin<sub>g</sub>l<sub>y</sub> <sub>non</sub>li<sub>near</sub> fl<sub>uctuat</sub>i<sub>on</sub> <sub>components,</sub> <sub>not</sub> <sub>a</sub>ddi<sub>t</sub>i<sub>ona</sub>l <sub>spat</sub>i<sub>a</sub>l <sub>coor</sub>di<sub>nates.</sub> A<sub>t</sub> l<sub>ea</sub>di<sub>ng</sub> <sub>or</sub>d<sub>er,</sub> <sub>a</sub>d<sub>vanc</sub>i<sub>ng</sub> f<sub>rom</sub> mode k to mode $k + 1$ <sub>costs an a</sub>ddi<sub>t</sub>i<sub>ona</sub>l f<sub>actor o</sub>f ${ \sqrt { r / d } } .$

## 5.3 Mutual information and identities for the distance

Th<sub>e</sub> <sub>spectrum</sub> <sub>measures</sub> <sub>recovera</sub>bl<sub>e</sub> <sub>var</sub>i<sub>ance</sub> f<sub>eature</sub> b<sub>y</sub> f<sub>eature.</sub> F<sub>rom</sub> <sub>t</sub>h<sub>e</sub> <sub>gamma</sub> <sub>entrop</sub>i<sub>es</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>an</sub>d <sub>om</sub>i<sub>tte</sub>d <sub>square</sub>d <sub>norms,</sub> <sub>we</sub> <sub>a</sub>l<sub>so</sub> <sub>o</sub>b<sub>ta</sub>i<sub>n</sub> <sub>t</sub>h<sub>e</sub> <sub>tota</sub>l <sub>mutua</sub>l i<sub>n</sub>f<sub>ormat</sub>i<sub>on.</sub>

Theorem 5.3 (Exact mutual information). With $h _ { \Gamma } ( k )$ the entropy of a gamma variable of shape k at any fixed scale,

$$
\begin{array} { c } { { I ( D ; \mathcal { O } ) = h _ { \Gamma } \displaystyle \left( \frac { d } { 2 } \right) - h _ { \Gamma } \displaystyle \left( \frac { d - r } { 2 } \right) \longrightarrow - \frac 1 2 \log ( 1 - \alpha ) } } \\ { { \displaystyle i f r / d \to \alpha < 1 \ a n d \ d - r \to \infty . I f r / d \to 0 . t h e n I ( D ; \mathcal { O } ) = r / ( 2 d ) + O ( r ^ { 2 } / d ^ { 2 } + 1 / d ) . } } \end{array}
$$

The <sub>p</sub>roof (section B) evaluates the <sub>g</sub>amma entro<sub>p</sub>ies of the observed and omitted s<sub>q</sub>uared norms; the li<sub>m</sub>i<sub>ts</sub> f<sub>o</sub>ll<sub>ow</sub> f<sub>rom</sub> S<sub>t</sub>i<sub>r</sub>li<sub>ng</sub> <sub>expans</sub>i<sub>ons.</sub>

<sup>F</sup>or t<sup>h</sup>e ran<sup>k</sup>-r projector, <sup>d</sup>e<sup>fi</sup>ne t<sup>h</sup>e resca<sup>l</sup>e<sup>d</sup> projecte<sup>d</sup> <sup>di</sup>stance <sup>b</sup>y

$$
\widetilde { D } = \frac { d } { r } \left\| \Pi ( X - X ^ { \prime } ) \right\| ^ { 2 } .
$$

Th<sub>e</sub> <sub>same</sub> d<sub>ecompos</sub>i<sub>t</sub>i<sub>on</sub> <sub>g</sub>i<sub>ves</sub> <sub>t</sub>h<sub>ree</sub> id<sub>ent</sub>i<sub>t</sub>i<sub>es</sub> f<sub>or</sub> <sub>t</sub>h<sub>e</sub> di<sub>stance:</sub>

$$
\operatorname { C o r r } ( D , \widetilde { D } ) = \sqrt { r / d } , \qquad \frac { \operatorname* { i n f } _ { f } \mathbb { E } [ ( D - f ( \widetilde { D } ) ) ^ { 2 } ] } { \operatorname { V a r } ( D ) } = 1 - \frac { r } { d } ,
$$

<sub>an</sub>d

$$
\frac { \mathbb { E } [ ( ( \widetilde { D } - \mathbb { E } \widetilde { D } ) - ( D - \mathbb { E } D ) ) ^ { 2 } ] } { \operatorname { V a r } ( D ) } = \frac { d - r } { r } .
$$

R<sub>esca</sub>li<sub>n preserves t</sub>h<sub>e common</sub> b<sub>ase</sub>li<sub>ne,</sub> b<sub>ut t</sub>h<sub>e</sub> fl<sub>uctuat</sub>i<sub>on no</sub>i<sub>se o</sub>f $\widetilde { D }$ h<sub>as re</sub>l<sub>at</sub>i<sub>ve or</sub>d<sub>er</sub> $r ^ { - 1 / 2 }$ <sub>rat</sub>h<sub>er t</sub>h<sub>an</sub> <sub>t</sub>h<sub>e</sub> d<sub>ata</sub>’<sub>s</sub> $\dot { d } ^ { - 1 / 2 }$

For one pair under a rank-m linear sketch,

$$
\mathsf { \rho _ { H G R } } \left( D _ { i j } ; \left( L X _ { i } , L X _ { j } \right) \right) = \sqrt { m / d }
$$

<sub>w</sub>h<sub>ereas</sub> <sub>t</sub>h<sub>e</sub> <sub>rep</sub>l<sub>acement</sub> <sub>c</sub>l<sub>ou</sub>d $Y ^ { n }$ <sup>of</sup> p<sup>ro</sup>p<sup>osition</sup> <sup>3.1</sup> g<sup>ives</sup>

$$
\mathsf { p } _ { \mathrm { H G R } } ( D _ { i j } ; Y ^ { n } ) = 0 .
$$

<sup>Th</sup>us, t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> <sup>i</sup>s compat<sup>ibl</sup>e <sup>b</sup>ot<sup>h</sup> w<sup>i</sup>t<sup>h</sup> t<sup>h</sup>e max<sup>i</sup>ma<sup>l</sup> <sup>d</sup>epen<sup>d</sup>ence $\sqrt { m / d }$ attainable from a rank-m linear <sub>s</sub>k<sub>etc</sub>h <sub>o</sub>f <sub>one</sub> di<sub>stance</sub> <sub>an</sub>d <sub>w</sub>i<sub>t</sub>h <sub>zero</sub> d<sub>epen</sub>d<sub>ence.</sub>

Th<sub>e</sub> i<sub>sotrop</sub>i<sub>c</sub> <sub>operator</sub> i<sub>s</sub> <sub>now</sub> <sub>so</sub>l<sub>ve</sub>d<sub>.</sub> S<sub>ect</sub>i<sub>on</sub> 6 <sub>exten</sub>d<sub>s</sub> di<sub>stance</sub> <sub>recovery</sub> <sub>to</sub> <sub>genera</sub>l <sub>covar</sub>i<sub>ance,</sub> <sub>w</sub>h<sub>ere</sub> <sub>t</sub>h<sub>e</sub> <sub>operator</sub> <sub>norm</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> <sub>recovery</sub> f<sub>ract</sub>i<sub>on</sub> f<sub>or</sub> <sub>t</sub>h<sub>e</sub> di<sub>stance</sub> <sub>can</sub> <sub>separate.</sub>

## 6. Distance Recovery Beyond Isotropy

Th<sub>e</sub> i<sub>sotrop</sub>i<sub>c so</sub>l<sub>ut</sub>i<sub>on</sub> i<sub>s comp</sub>l<sub>ete: t</sub>h<sub>eorems</sub> 5<sub>.</sub>1 <sub>an</sub>d 5<sub>.</sub>2 <sub>g</sub>i<sub>ve t</sub>h<sub>e operator norm an</sub>d <sub>t</sub>h<sub>e</sub> f<sub>u</sub>ll <sub>s</sub>i<sub>ngu</sub>l<sub>ar</sub> <sub>spectrum.</sub> A<sub>n</sub>i<sub>sotropy</sub> <sub>separates</sub> <sub>two</sub> <sub>quest</sub>i<sub>ons</sub> <sub>t</sub>h<sub>at</sub> <sub>co</sub>i<sub>nc</sub>id<sub>e</sub> f<sub>or</sub> <sub>a</sub> fl<sub>at</sub> <sub>spectrum:</sub> <sub>recovery</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> di<sub>stance</sub> <sub>va</sub>l<sub>ue</sub> <sub>an</sub>d <sub>recovery</sub> <sub>o</sub>f <sub>an</sub> <sub>ar</sub>bi<sub>trary</sub> <sub>non</sub>li<sub>near</sub> f<sub>eature.</sub>

## 6.1 General covariance: the recovery fraction for the distance

Th<sub>eorem</sub> 5<sub>.</sub>1 i<sub>s an</sub> i<sub>sotrop</sub>i<sub>c statement a</sub>b<sub>out every</sub> f<sub>eature.</sub> F<sub>or genera</sub>l <sub>covar</sub>i<sub>ance, recovery o</sub>f <sub>t</sub>h<sub>e</sub> di<sub>stance</sub> value itself admits an exact answer for every spectrum and every rank-m map. Because eigendirection j <sub>contr</sub>ib<sub>utes</sub> $8 \lambda _ { j } ^ { 2 }$ to $\mathrm { V a r } ( D )$ , an optimal m-dimensional observation targets the largest squared eigenvalues. We therefore define the recovery fraction for the distance D by

$$
\alpha _ { m } ( \Sigma ) = \frac { \sum _ { j \leq m } \lambda _ { j } ^ { 2 } } { \sum _ { j \leq d } \lambda _ { j } ^ { 2 } } \in ( 0 , 1 ] .\tag{5}
$$

<sup>We</sup> <sup>must</sup> <sup>se</sup>p<sup>arate</sup> <sup>two</sup> <sup>recover</sup>y q<sup>uestions.</sup> <sup>First</sup>, $\alpha _ { m } ( \Sigma )$ i<sub>s t</sub>h<sub>e</sub> l<sub>argest</sub> f<sub>ract</sub>i<sub>on o</sub>f $\mathrm { V a r } ( D )$ <sub>recovera</sub>bl<sub>e</sub> by a rank-m map. Second, for a fixed observation, $\rho _ { \mathrm { H G R } } ^ { 2 } ( D ; \mathcal { O } )$ is the lar<sub>g</sub>est recoverable fraction over all square-integrable features of D. These quantities coincide at m/d in the isotropic case. Under anisotropy <sub>t</sub>h<sub>ey</sub> <sub>can</sub> dif<sub>er;</sub> b<sub>a</sub>l<sub>ance</sub>d <sub>spectra</sub>l <sub>retent</sub>i<sub>on</sub> i<sub>s</sub> <sub>a</sub> <sub>su</sub>fi<sub>c</sub>i<sub>ent</sub> <sub>con</sub>di<sub>t</sub>i<sub>on</sub> <sub>t</sub>h<sub>at</sub> <sub>restores</sub> <sub>equa</sub>li<sub>ty.</sub>

Theorem 6.1 (Sharp rank-m limit for distance recovery). For every rank-m linear L and every measurable estimator $g ,$

$$
\frac { \mathrm { V a r } \big ( \mathbb { E } \big [ D \mid \mathcal O \big ] \big ) } { \mathrm { V a r } ( D ) } \leq \alpha _ { m } ( \Sigma ) , \qquad \mathbb { E } \big [ ( D - g ( \mathcal O ) ) ^ { 2 } \big ] \geq 8 \sum _ { j > m } \lambda _ { j } ^ { 2 } .
$$

The variance bound is attained when the row space of L is the top-m eigenspace of Σ; for this map, the mean-square bound is attained $b \gamma g = \mathbb { E } [ D \mid { \mathcal { O } } ]$ . The same statements, with $\mathrm { V a r } ( \stackrel {  } { S } ) = \mathrm { \ i } 2 \operatorname { t r } \stackrel {  } { ( } \Sigma ^ { 2 } )$ , hold for the neighbor contrast $S = \left\| X _ { 0 } - X _ { 1 } \right\| ^ { 2 } - \left\| X _ { 0 } - X _ { 2 } \right\| ^ { 2 } g i \nu e n \left( L X _ { 0 } , L X _ { 1 } , L X _ { 2 } \right)$

The <sub>p</sub>roof (section B) combines the Gaussian conditional mean, a <sub>p</sub>rojector identit<sub>y</sub>, and ei<sub>g</sub>envalue i<sub>nter</sub>l<sub>ac</sub>i<sub>ng</sub> f<sub>or t</sub>h<sub>e compresse</sub>d <sub>covar</sub>i<sub>ance.</sub> Th<sub>e</sub> b<sub>oun</sub>d<sub>s are un</sub>if<sub>orm: no non</sub>li<sub>near est</sub>i<sub>mator o</sub>f <sub>t</sub>h<sub>e</sub> di<sub>stance</sub> value and no Σ-aware choice of L exceeds the squared-eigenvalue limit. The mean-square bound also follows f<sub>rom</sub> <sub>t</sub>h<sub>e</sub> l<sub>aw</sub> <sub>o</sub>f <sub>tota</sub>l <sub>var</sub>i<sub>ance:</sub> <sub>su</sub>b<sub>tract</sub>i<sub>ng</sub> <sub>at</sub> <sub>most</sub> $8 \textstyle \sum _ { j \leq m } \lambda _ { j } ^ { 2 }$ <sub>recovere</sub>d <sub>var</sub>i<sub>ance</sub> f<sub>rom</sub> $\begin{array} { r } { \mathrm { V a r } ( D ) = 8 \sum _ { j \leq d } \lambda _ { j } ^ { 2 } } \end{array}$ l<sub>eaves at</sub> l<sub>east</sub> $8 \textstyle \sum _ { j > m } \lambda _ { j } ^ { 2 }$

## 6.2 Anisotropy: divergence and reconciliation

<sup>The isotro</sup>p<sup>ic identit</sup>y <sup>mi</sup>g<sup>ht su</sup>gg<sup>est that</sup> $\sqrt { \alpha _ { m } ( \Sigma ) }$ b<sub>oun</sub>d<sub>s</sub> HGR <sub>un</sub>d<sub>er</sub> <sub>an</sub>i<sub>sotropy,</sub> b<sub>ut</sub> <sub>t</sub>hi<sub>s</sub> b<sub>oun</sub>d f<sub>a</sub>il<sub>s</sub> i<sub>n</sub> <sub>genera</sub>l<sub>.</sub> Th<sub>e</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> i<sub>s</sub> <sub>unequa</sub>l <sub>retent</sub>i<sub>on</sub> <sub>across</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>spectra</sub>l bl<sub>oc</sub>k<sub>s:</sub> <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> <sub>one</sub> bl<sub>oc</sub>k <sub>comp</sub>l<sub>ete</sub>l<sub>y w</sub>hil<sub>e</sub> di<sub>scar</sub>di<sub>ng anot</sub>h<sub>er can</sub> l<sub>et a non</sub>li<sub>near trans</sub>f<sub>ormat</sub>i<sub>on rewe</sub>i<sub>g</sub>h<sub>t t</sub>h<sub>e</sub>i<sub>r contr</sub>ib<sub>ut</sub>i<sub>ons.</sub> A <sub>two-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>counterexamp</sub>l<sub>e</sub> ill<sub>ustrates t</sub>hi<sub>s mec</sub>h<sub>an</sub>i<sub>sm.</sub>

Theorem 6.2 (Anisotropic counterexample). Let $D = 2 G _ { 1 } ^ { 2 } + G _ { 2 } ^ { 2 } .$ for independent standard normals $G _ { 1 } , G _ { 2 }$ and observe $U = 2 G _ { 1 } ^ { 2 }$ . Although $\alpha _ { 1 } = 4 / 5 ,$ quadratic polynomial features already give

$$
\rho _ { \mathrm { H G R } } ( D ; U ) \geq \sqrt { \frac { 2 4 6 + 2 \sqrt { 2 0 1 } } { 3 1 1 } } \approx 0 . 9 3 9 2 > \sqrt { 4 / 5 } .
$$

Wh<sub>en</sub> $\rho _ { \mathrm { H G R } } ^ { 2 } ( D ; \mathcal { O } ) > \alpha _ { m } ( \Sigma )$ , some nonlinear feature of D is more recoverable than the distance value itself. We call this gap nonlinear excess. Here $\sqrt { 4 / 5 } \approx 0 . 8 9 4 4$ i<sub>s t</sub>h<sub>e</sub> li<sub>near corre</sub>l<sub>at</sub>i<sub>on w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>e</sub> di<sub>stance.</sub> Th<sub>e</sub> <sub>p</sub>roof (section B) evaluates the canonical correlation of the <sub>q</sub>uadratic features, which makes the dis<sub>p</sub>la<sub>y</sub>ed va<sup>l</sup>ue a <sup>l</sup>ower <sup>b</sup>oun<sup>d</sup> on ρ<sub>HGR</sub>. For $D _ { \lambda } = \lambda G _ { 1 } ^ { 2 } + G _ { 2 } ^ { 2 }$ <sub>an</sub>d $U _ { \lambda } = \lambda G _ { 1 } ^ { 2 }$ , Fi<sub>g</sub>ure 3 re<sub>p</sub>orts canonica<sup>l</sup>-corre<sup>l</sup>ation l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>rom</sub> <sub>neste</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l f<sub>eature</sub> <sub>spaces.</sub> Th<sub>ese</sub> <sub>va</sub>l<sub>ues</sub> <sub>are</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>o</sub>b<sub>servat</sub>i<sub>on,</sub> <sub>not</sub> <sub>t</sub>h<sub>e</sub> <sub>unrestr</sub>i<sub>cte</sub>d HGR <sub>va</sub>l<sub>ue</sub> <sub>or</sub> <sub>an</sub> <sub>opt</sub>i<sub>m</sub>i<sub>zat</sub>i<sub>on</sub> <sub>over</sub> <sub>o</sub>b<sub>servat</sub>i<sub>ons.</sub>

N<sub>on</sub>li<sub>near</sub> <sub>excess</sub> <sub>can</sub> <sub>occur</sub> b<sub>ut</sub> i<sub>s</sub> <sub>not</sub> <sub>un</sub>i<sub>versa</sub>l<sub>.</sub> A <sub>su</sub>fi<sub>c</sub>i<sub>ent</sub> <sub>con</sub>di<sub>t</sub>i<sub>on</sub> f<sub>or</sub> i<sub>ts</sub> <sub>a</sub>b<sub>sence</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>y</sub> <sub>wr</sub>i<sub>t</sub>i<sub>ng</sub>

$$
\begin{array} { l } { \displaystyle { \sum = \sum _ { g = 1 } ^ { G } \lambda _ { g } P _ { g } , \qquad r _ { g } = \mathrm { r a n k } P _ { g } , } } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> <sub>t</sub>h<sub>e</sub> $\lambda _ { g } > 0$ <sub>are</sub> di<sub>st</sub>i<sub>nct</sub> <sub>an</sub>d $P _ { g }$ projects onto the associated eigenspace, or spectral block. Let Π satisfy $\Pi { \boldsymbol { \Sigma } } = { \boldsymbol { \Sigma } } \Pi$ <sub>,</sub> <sub>so</sub> i<sub>t</sub> <sub>preserves</sub> <sub>every</sub> bl<sub>oc</sub>k<sub>,</sub> <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
s _ { g } = \mathrm { r a n k } \bigl ( \Pi P _ { g } \bigr ) , \qquad \theta _ { g } = \frac { s _ { g } } { r _ { g } } , \qquad \alpha _ { \Pi } \bigl ( \Sigma \bigr ) = \frac { \mathrm { t r } \bigl ( \Pi \Sigma ^ { 2 } \bigr ) } { \mathrm { t r } \bigl ( \Sigma ^ { 2 } \bigr ) } = \frac { \sum _ { g } \lambda _ { g } ^ { 2 } s _ { g } } { \sum _ { g } \lambda _ { g } ^ { 2 } r _ { g } } .
$$

Th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> di<sub>stance</sub> <sub>t</sub>h<sub>en</sub> <sub>sp</sub>li<sub>ts</sub> <sub>across</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> b<sub>eta–gamma</sub> <sub>c</sub>h<sub>anne</sub>l<sub>s,</sub> <sub>one</sub> <sub>per</sub> <sub>spectra</sub>l bl<sub>oc</sub>k<sub>.</sub> M<sub>ax</sub>i<sub>ma</sub>l correlation tensorizes over such inde<sub>p</sub>endent <sub>p</sub>airs [25]: for nonde<sub>g</sub>enerate $( Y _ { j } , Z _ { j } )$ i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>across</sub> $j ,$

$$
\mathsf { \rho } _ { \mathsf { H G R } } \left( ( Y _ { j } ) _ { j } ; ( Z _ { j } ) _ { j } \right) = \operatorname* { m a x } _ { j } \mathsf { \rho } _ { \mathsf { H G R } } ( Y _ { j } ; Z _ { j } ) .
$$

Theorem 6.3 (Spectral-block HGR bounds and the balanced regime). For independent X, $X ^ { \prime } \sim { \mathcal { N } } ( \mu , \Sigma )$ let $D = \left. X - X ^ { \prime } \right. ^ { 2 }$ and $\mathcal { O } = \left( \Pi X , \Pi X ^ { \prime } \right)$ . Under $\Pi { \boldsymbol { \Sigma } } = { \boldsymbol { \Sigma } } \Pi$

$$
\sqrt { \alpha _ { \Pi } ( \Sigma ) } \leq \mathsf { \rho } _ { \mathrm { H G R } } ( D ; \mathcal { O } ) \leq \sqrt { \operatorname* { m a x } _ { g } \theta _ { g } } .
$$

If $s _ { g } = \theta r _ { g }$ for every nonzero spectral block, then

$$
\begin{array} { r } { \rho _ { \mathrm { H G R } } ( D ; \mathcal { O } ) = \sqrt { \theta } = \sqrt { \alpha _ { \Pi } ( \Sigma ) } . } \end{array}
$$

Consequently, Va $\cdot ( \mathbb { E } [ f ( D ) \mid \mathcal { O } ] ) \leq \theta \operatorname { V a r } ( f ( D ) )$ for every $f \in L ^ { 2 } ( D )$ , and the centered distance paired with the centered retained squared norm attains maximal correlation.

Wi<sub>tsen</sub>h<sub>ausen tensor</sub>i<sub>zat</sub>i<sub>on</sub> b<sub>oun</sub>d<sub>s t</sub>hi<sub>s pro</sub>d<sub>uct c</sub>h<sub>anne</sub>l b<sub>y</sub> i<sub>ts</sub> l<sub>argest</sub> bl<sub>oc</sub>k <sub>corre</sub>l<sub>at</sub>i<sub>on,</sub> $\sqrt { \operatorname* { m a x } _ { g } \theta _ { g } } ;$ a centered linear witness su<sub>pp</sub>lies the lower bound (section B contains the full ar<sub>g</sub>ument). Une<sub>q</sub>ual ei<sub>g</sub>envalues <sub>a</sub>l<sub>one t</sub>h<sub>ere</sub>f<sub>ore</sub> d<sub>o not</sub> b<sub>rea</sub>k <sub>t</sub>h<sub>e</sub> b<sub>a</sub>l<sub>ance</sub>d id<sub>ent</sub>i<sub>ty.</sub> U<sub>nequa</sub>l <sub>retent</sub>i<sub>on perm</sub>i<sub>ts non</sub>li<sub>near excess</sub> b<sub>ut</sub> d<sub>oes not</sub> f<sub>orce</sub> i<sub>t;</sub> <sub>equa</sub>l f<sub>ract</sub>i<sub>ona</sub>l <sub>retent</sub>i<sub>on</sub> <sub>ma</sub>k<sub>es</sub> <sub>t</sub>h<sub>e</sub> <sub>excess</sub> <sub>van</sub>i<sub>s</sub>h<sub>.</sub>

Th<sub>e</sub> fl<sub>at</sub> <sub>spectrum</sub> <sub>c</sub>l<sub>oses</sub> <sub>t</sub>h<sub>e</sub> <sub>gap</sub> b<sub>etween</sub> <sub>t</sub>h<sub>e</sub> <sub>two</sub> <sub>recovery</sub> <sub>quant</sub>i<sub>t</sub>i<sub>es.</sub> If $\Sigma = \lambda P$ h<sub>as</sub> fl<sub>at</sub> <sub>support</sub> <sub>o</sub>f <sub>ran</sub>k r, an observation retaining rank s within range(P) has $\rho _ { \mathrm { H G R } } ^ { 2 } = \bar { s } / r ;$ an optimal rank-at-most-m observation <sub>c</sub>h<sub>ooses</sub> i<sub>ts</sub> <sub>row</sub> <sub>space</sub> i<sub>ns</sub>id<sub>e</sub> <sub>t</sub>h<sub>e</sub> <sub>support,</sub> <sub>ta</sub>k<sub>es</sub> $s = \operatorname* { m i n } \{ m , r \}$ <sub>, an</sub>d <sub>atta</sub>i<sub>ns</sub> $\rho _ { \mathrm { H G R } } ^ { 2 } = \alpha _ { m } \mathopen { } \mathclose \bgroup \left( \Sigma \aftergroup \egroup \right)$ . More <sub>g</sub>enera<sup>ll</sup><sub>y</sub>, f<sub>or</sub> <sub>one</sub> di<sub>stance,</sub> $\alpha _ { m } ( \Sigma )$ <sub>governs</sub> <sub>t</sub>h<sub>e</sub> di<sub>stance</sub> <sub>va</sub>l<sub>ue</sub> f<sub>or</sub> <sub>every</sub> <sub>spectrum,</sub> <sub>an</sub>d $\alpha _ { \Pi } ( \Sigma )$ i<sub>s</sub> <sub>a</sub>l<sub>so</sub> <sub>t</sub>h<sub>e</sub> <sub>ce</sub>ili<sub>ng</sub> f<sub>or</sub> every <sup>f</sup>eature un<sup>d</sup>er a <sup>b</sup>a<sup>l</sup>ance<sup>d</sup> a<sup>li</sup>gne<sup>d</sup> project<sup>i</sup>on. <sup>R</sup>an<sup>ki</sup>ngs compare severa<sup>l</sup> <sup>d</sup>epen<sup>d</sup>ent <sup>di</sup>stances s<sup>h</sup>ar<sup>i</sup>ng a <sub>query</sub> <sub>an</sub>d f<sub>o</sub>ll<sub>ow</sub> <sub>t</sub>h<sub>e</sub> <sub>square-root</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>sca</sub>l<sub>e</sub> <sub>assoc</sub>i<sub>ate</sub>d <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>e</sub> <sub>same</sub> <sub>ran</sub>k <sub>rat</sub>i<sub>o.</sub>

## 7. Rankings Under Random Projection

<sup>S</sup>ect<sup>i</sup>ons 5 an<sup>d</sup> 6 quant<sup>ifi</sup>e<sup>d</sup> t<sup>h</sup>e recovery o<sup>f</sup> one <sup>di</sup>stance. <sup>R</sup>an<sup>ki</sup>ngs <sup>i</sup>nstant<sup>i</sup>ate t<sup>h</sup>e project<sup>i</sup>on temp<sup>l</sup>ate o<sup>f</sup> sect<sup>i</sup>on 4.4 w<sup>i</sup>t<sup>h</sup> t<sup>h</sup>e s<sup>h</sup>are<sup>d</sup>-query contrast as target an<sup>d</sup> t<sup>h</sup>e t<sup>h</sup>ree projecte<sup>d</sup> po<sup>i</sup>nts as o<sup>b</sup>servat<sup>i</sup>on. <sup>B</sup>ecause <sub>ran</sub>ki<sub>ngs</sub> <sub>compare</sub> <sub>severa</sub>l d<sub>epen</sub>d<sub>ent</sub> di<sub>stances,</sub> <sub>we</sub> fi<sub>rst</sub> d<sub>e</sub>fi<sub>ne</sub> <sub>t</sub>h<sub>e</sub> <sub>ran</sub>k f<sub>unct</sub>i<sub>ona</sub>l <sub>an</sub>d id<sub>ent</sub>if<sub>y</sub> <sub>t</sub>h<sub>e</sub> <sub>ran</sub>ki<sub>ng</sub> <sub>sca</sub>l<sub>e</sub> b<sub>e</sub>f<sub>ore</sub> <sub>comput</sub>i<sub>ng</sub> i<sub>ts</sub> <sub>exact</sub> l<sub>aw.</sub>

F<sub>or</sub> <sub>two</sub> <sub>t</sub>i<sub>e-</sub>f<sub>ree</sub> <sub>score</sub> <sub>vectors</sub> $a , b \in \mathbb { R } ^ { q }$ <sub>,</sub> K<sub>en</sub>d<sub>a</sub>ll’<sub>s</sub> <sub>ran</sub>k <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> i<sub>s</sub>

$$
\tau ( a , b ) = \binom { q } { 2 } ^ { - 1 } \sum _ { j < k } \mathrm { s g n } \big [ ( a _ { j } - a _ { k } ) ( b _ { j } - b _ { k } ) \big ] .\tag{6}
$$

This statistic is the fraction of concordant <sub>p</sub>airs minus the fraction of discordant <sub>p</sub>airs [26]. For inde<sub>p</sub>endent <sub>cont</sub>i<sub>nuous</sub> <sub>ran</sub>ki<sub>ngs,</sub> <sub>t</sub>h<sub>e</sub> <sub>expecte</sub>d K<sub>en</sub>d<sub>a</sub>ll <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> i<sub>s</sub> <sub>zero.</sub>

## 7.1 Why m/d: sign statistics at the linear mode

W<sub>e</sub> <sub>can</sub> id<sub>ent</sub>if<sub>y</sub> <sub>t</sub>h<sub>e</sub> <sub>ran</sub>ki<sub>ng</sub> <sub>sca</sub>l<sub>e</sub> b<sub>e</sub>f<sub>ore</sub> <sub>t</sub>h<sub>e</sub> <sub>exact</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>at</sub>i<sub>on.</sub> F<sub>or</sub> <sub>a</sub> <sub>query</sub> $X _ { 0 }$ <sub>an</sub>d <sub>can</sub>did<sub>ates</sub> $X _ { 1 } , X _ { 2 }$ <sub>,</sub> <sub>wr</sub>it<sub>e</sub>

$$
S = D _ { 0 1 } - D _ { 0 2 } = a ^ { \top } b , \qquad a = X _ { 1 } - X _ { 2 } , \qquad b = X _ { 1 } + X _ { 2 } - 2 X _ { 0 } .
$$

For i.i.d. isotropic Gaussians, a and b are independent. Under the rescaled rank-m projector, the contrast is $\widetilde { S } = ( d / m ) a ^ { \top } \Pi b$ . Conditional on a, the pair $( \bar { S , { \widetilde S } } )$ is centered jointl<sub>y</sub> Gaussian in b with correlation

$$
\rho ( a ) = { \frac { \Vert \Pi a \Vert } { \Vert a \Vert } } , \qquad \rho ( a ) ^ { 2 } \sim \mathrm { B e t a } \Bigg ( { \frac { m } { 2 } } , { \frac { d - m } { 2 } } \Bigg ) .
$$

The squared correlation has mean m/d.

For a centered jointl<sub>y</sub> Gaussian <sub>p</sub>air<sub>,</sub> She<sub>pp</sub>ard [27] showed that the si<sub>g</sub>n-a<sub>g</sub>reement <sub>p</sub>robabilit<sub>y</sub> is $\frac { 1 } { 2 } + \pi ^ { - 1 }$ arcsin ρ. $\mathsf { A s } \rho  0$ <sub>,</sub> <sub>t</sub>hi<sub>s</sub> <sub>agreement</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>ty</sub> <sub>excee</sub>d<sub>s</sub> <sub>c</sub>h<sub>ance</sub> b<sub>y</sub> $\rho / \pi + O \big ( \rho ^ { 3 } \big )$ <sub>.</sub> Th<sub>us t</sub>h<sub>e reta</sub>i<sub>ne</sub>d<sub>-</sub> ener<sub>gy</sub> fraction m/d becomes a rankin<sub>g</sub> excess of order $\pi ^ { - 1 } { \sqrt { m / d } }$ <sub>on</sub> <sub>t</sub>h<sub>e</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>sca</sub>l<sub>e.</sub> A<sub>verag</sub>i<sub>ng</sub> Sh<sub>eppar</sub>d’<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub> <sub>over</sub> <sub>t</sub>h<sub>e</sub> B<sub>eta-</sub>di<sub>str</sub>ib<sub>ute</sub>d <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>t</sub>h<sub>en</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>exact</sub> l<sub>aw.</sub>

## 7.2 Exact pairwise order agreement

L<sub>e</sub>t $X _ { 0 } , X _ { 1 } , X _ { 2 } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( \mu , \sigma ^ { 2 } I _ { d } )$ in<sup>d</sup>epen<sup>d</sup>ent<sup>l</sup>y o<sup>f</sup> a Haar ran<sup>k</sup>-m projection, w<sup>h</sup>ere $0 < m < d .$ L<sub>e</sub>t $D _ { 0 j }$ <sub>an</sub>d $\widetilde { D } _ { 0 j }$ <sup>b</sup>e t<sup>h</sup>e or<sup>i</sup>g<sup>i</sup>na<sup>l</sup> an<sup>d</sup> resca<sup>l</sup>e<sup>d</sup> projecte<sup>d</sup> square<sup>d</sup> <sup>di</sup>stances.

Theorem 7.1 (Exact pairwise order agreement). The probability that the comparisons agree is

$$
p _ { m , d } = \operatorname* { P r } \left[ \bigl ( D _ { 0 1 } - D _ { 0 2 } \bigr ) \bigl ( \widetilde { D } _ { 0 1 } - \widetilde { D } _ { 0 2 } \bigr ) > 0 \right] = \frac { 1 } { 2 } + \frac { 1 } { \pi } \mathbb { E } \left[ \arcsin \sqrt { B } \right] , \qquad B \sim \operatorname { B e t a } \left( \frac { m } { 2 } , \frac { d - m } { 2 } \right) .
$$

Hence $\begin{array} { r } { p _ { m , d } \leq \frac { 1 } { 2 } + \frac { 1 } { 2 } \sqrt { m / d } , } \end{array}$ and

$$
p _ { m , d } = { \frac { 1 } { 2 } } + { \frac { 1 } { \pi } } { \sqrt { \frac { m } { d } } } \left( 1 + O { \left( { \frac { 1 } { m } } + { \frac { m } { d } } \right) } \right)
$$

when $m \to \infty$ and m/d → 0. For one query ranking any number of candidates, the expected Kendall correlation is $\mathbb { E } \tau = 2 p _ { m , d } - 1$

The <sub>p</sub>roof (section B) conditions on a in the <sub>p</sub>recedin<sub>g</sub> reduction and avera<sub>g</sub>es She<sub>pp</sub>ard’s formula over <sub>t</sub>h<sub>e</sub> B<sub>eta</sub> l<sub>aw.</sub> Th<sub>us, w</sub>h<sub>en</sub> $m = o ( d )$ , <sup>the</sup> <sup>ex</sup>p<sup>ected</sup> p<sup>airwise</sup> <sup>a</sup>g<sup>reement</sup> <sup>is</sup> <sup>onl</sup>y $\Theta ( \sqrt { m / d } )$ <sub>a</sub>b<sub>ove c</sub>h<sub>ance.</sub>

Anisotropic bounds

![](images/21d87736de4a7969cc1f57cbe57dc1fc5e558beeffee0632ee8a5190aa6d1ac0.jpg)  
Pairwise ordering

![](images/fe64fbc0dd604b078dea6668f210fc713911c68a17831aa69673cbcfa702da39.jpg)  
Figure 3. Polynomial lower bounds for $D _ { \lambda } \mapsto U _ { \lambda }$ . Higher degrees reveal nonlinear dependence beyond $\sqrt { \alpha _ { 1 } } ;$ the vertical line marks the counterexample $\lambda = 2 .$  
Figure 4. Pairwise order agreement: the exact Beta–arcsine law, its asymptote, and Monte Carlo simulations at $d = 2 0 0$ with 120,000 samples per marker.

## 7.3 Nearest-neighbor and top-k overlap

Pairwise a<sub>g</sub>reement does not determine whether the nearest candidate or the full top-k set is preserved. Let $X _ { 0 } , X _ { 1 } , \dots , X _ { q } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , I _ { d } )$ <sub>,</sub> <sub>w</sub>i<sub>t</sub>h $X _ { 0 }$ <sup>the</sup> q<sup>uer</sup>y<sup>.</sup> <sup>Let</sup> $\boldsymbol { R } \in \mathbb { R } ^ { m \times d }$ h<sub>ave ort</sub>h<sub>onorma</sub>l <sub>rows an</sub>d b<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> <sub>samp</sub>l<sub>e;</sub> i<sub>sotropy</sub> <sub>ma</sub>k<sub>es</sub> i<sub>ts</sub> <sub>or</sub>i<sub>entat</sub>i<sub>on</sub> i<sub>rre</sub>l<sub>evant.</sub> L<sub>et</sub> $\widetilde { \mathcal { N } } _ { k }$ <sub>an</sub>d $\mathcal { N } _ { k }$ <sup>b</sup>e t<sup>h</sup>e projecte<sup>d</sup> an<sup>d</sup> or<sup>i</sup>g<sup>i</sup>na<sup>l</sup> sets o<sup>f</sup> k nearest candidate indices.

W<sub>r</sub>it<sub>e</sub>

$$
U _ { j } = \left\| R ( X _ { j } - X _ { 0 } ) \right\| ^ { 2 } , \qquad V _ { j } = \left\| ( I - \Pi ) ( X _ { j } - X _ { 0 } ) \right\| ^ { 2 } , \qquad \Pi = R ^ { \top } R .
$$

<sup>Th</sup>e projecte<sup>d</sup> ran<sup>ki</sup>ng <sup>i</sup>s <sup>d</sup>eterm<sup>i</sup>ne<sup>d b</sup>y $U = ( U _ { j } ) _ { j } ,$ <sub>, w</sub>hil<sub>e t</sub>h<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>ran</sub>ki<sub>ng</sub> i<sub>s</sub> d<sub>eterm</sub>i<sub>ne</sub>d b<sub>y</sub> $U + V ,$ <sub>.</sub> F<sub>or</sub> one coordinate, Gaussian fourth moments <sub>g</sub>ive, for distinct candidates j and k,

$$
\mathrm { V a r } \Big ( \big ( x _ { j } - x _ { 0 } \big ) ^ { 2 } \Big ) = 8 , \qquad \mathrm { C o v } \Big ( \big ( x _ { j } - x _ { 0 } \big ) ^ { 2 } , \big ( x _ { k } - x _ { 0 } \big ) ^ { 2 } \Big ) = 2 .
$$

Th<sub>e</sub> <sub>centere</sub>d <sub>an</sub>d <sub>norma</sub>li<sub>ze</sub>d <sub>vectors</sub> $( U - 2 m 1 ) / \sqrt { m }$ <sub>an</sub>d $( V - 2 ( d - m ) 1 ) / \surd d - m$ <sub>t</sub>h<sub>ere</sub>f<sub>ore converge</sub> <sub>to</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> G<sub>auss</sub>i<sub>an score vectors w</sub>i<sub>t</sub>h <sub>covar</sub>i<sub>ance</sub> $6 I _ { q } + 2 1 1 ^ { \top }$ <sub>.</sub> Th<sub>e</sub> <sub>ran</sub>k<sub>-one</sub> <sub>term</sub> i<sub>n</sub> <sub>t</sub>hi<sub>s</sub> <sub>covar</sub>i<sub>ance</sub> <sub>represents</sub> <sub>a</sub> <sub>ran</sub>d<sub>om</sub> <sub>s</sub>hif<sub>t</sub> <sub>s</sub>h<sub>are</sub>d b<sub>y</sub> <sub>every</sub> <sub>can</sub>did<sub>ate.</sub> R<sub>emov</sub>i<sub>ng</sub> <sub>t</sub>hi<sub>s</sub> <sub>s</sub>hif<sub>t</sub> l<sub>eaves</sub> <sub>t</sub>h<sub>e</sub> <sub>ran</sub>k<sub>s</sub> <sub>unc</sub>h<sub>ange</sub>d <sub>an</sub>d <sub>g</sub>ives corre<sup>l</sup>ation $\sqrt { m / d }$ b<sub>etween</sub> <sub>t</sub>h<sub>e</sub> <sub>reta</sub>i<sub>ne</sub>d <sub>an</sub>d f<sub>u</sub>ll <sub>scores.</sub>

F<sub>or</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>stan</sub>d<sub>ar</sub>d <sub>norma</sub>l <sub>vectors</sub> $G , H \in \mathbb { R } ^ { q }$ <sub>,</sub> l<sub>et</sub> $\mathcal { K } _ { \boldsymbol { k } } ( \boldsymbol { z } )$ return the indices of the k smallest <sub>coor</sub>di<sub>nates an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
\mathcal { R } _ { q , k } ( \boldsymbol { \rho } ) = \frac { 1 } { k } \mathbb { E } \left| \mathcal { K } _ { k } ( G ) \cap \mathcal { K } _ { k } \left( \boldsymbol { \rho } G + \sqrt { 1 - \boldsymbol { \rho ^ { 2 } } } H \right) \right| .
$$

Th<sub>e</sub> <sub>nearest-ne</sub>i<sub>g</sub>hb<sub>or</sub> <sub>case</sub> i<sub>s</sub> $p _ { q } ( \boldsymbol { \mathbf { \rho } } ) = \mathcal { R } _ { q , 1 } ( \boldsymbol { \mathbf { \rho } } )$ <sub>,</sub> <sub>t</sub>h<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>ty</sub> <sub>t</sub>h<sub>at</sub> <sub>t</sub>h<sub>e</sub> <sub>two</sub> <sub>score</sub> <sub>vectors</sub> h<sub>ave</sub> <sub>t</sub>h<sub>e</sub> <sub>same</sub> <sub>sma</sub>ll<sub>est coor</sub>di<sub>nate.</sub> F<sub>or</sub> $G _ { \rho } = \rho G + \sqrt { 1 - \rho ^ { 2 } } H .$ <sub>,</sub> d<sub>e</sub>fi<sub>ne t</sub>h<sub>e m</sub>i<sub>n</sub>i<sub>mum ce</sub>ll<sub>s</sub>

$$
A _ { i } = \big \{ x \in \mathbb { R } ^ { q } : x _ { i } = \operatorname* { m i n } _ { j } x _ { j } \big \} , \qquad i = 1 , \dots , q .
$$

Th<sub>en</sub>

$$
\sum _ { i = 1 } ^ { q } \operatorname* { P r } ( G \in A _ { i } , G _ { \boldsymbol { \rho } } \in A _ { i } ) = \operatorname* { P r } \left( { \arg \operatorname* { m i n } G _ { i } = \arg \operatorname* { m i n } _ { i } ( G _ { \boldsymbol { \rho } } ) _ { i } } \right) .\tag{7}
$$

Th<sub>e sum over a</sub>ll $q$ <sub>ce</sub>ll<sub>s</sub> i<sub>s essent</sub>i<sub>a</sub>l<sub>; one ce</sub>ll h<sub>as pro</sub>b<sub>a</sub>bili<sub>ty sma</sub>ll<sub>er</sub> b<sub>y a</sub> f<sub>actor o</sub>f $\dot { \mathbf { \zeta } } _ { q }$ <sup>b</sup>y <sup>s</sup>y<sup>mmetr</sup>y<sup>. Let</sup> $\Phi _ { \rho }$ b<sub>e</sub> <sub>t</sub>h<sub>e stan</sub>d<sub>ar</sub>d bi<sub>var</sub>i<sub>ate-norma</sub>l d<sub>ens</sub>i<sub>ty an</sub>d l<sub>et</sub> $\dot { \overline { { \Phi } } } _ { \rho } ( x , \gamma ) \dot { = } \operatorname* { P r } ( Z _ { 1 } > \dot { x } , Z _ { 2 } > \gamma )$ <sup>b</sup>e <sup>i</sup>ts jo<sup>i</sup>nt surv<sup>i</sup>va<sup>l</sup> <sup>f</sup>unct<sup>i</sup>on. W<sub>e a</sub>l<sub>so wr</sub>i<sub>te</sub> ${ \overline { { \Phi } } } ( x ) = \operatorname* { P r } ( Z > x )$ f<sub>or t</sub>h<sub>e un</sub>i<sub>var</sub>i<sub>ate stan</sub>d<sub>ar</sub>d<sub>-norma</sub>l <sub>surv</sub>i<sub>va</sub>l f<sub>unct</sub>i<sub>on.</sub> C<sub>on</sub>di<sub>t</sub>i<sub>on</sub>i<sub>ng on t</sub>h<sub>e</sub> <sup>common</sup> <sup>minimum</sup> g<sup>ives</sup>

$$
p _ { q } ( \boldsymbol { \rho } ) = q \int \int _ { \mathbb { R } ^ { 2 } } \Phi _ { \rho } ( x , \gamma ) \overline { { \Phi } } _ { \rho } ( x , \gamma ) ^ { q - 1 } d x d \gamma .\tag{8}
$$

E<sub>ac</sub>h <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> G<sub>auss</sub>i<sub>an</sub> <sub>pa</sub>i<sub>r</sub> <sub>must</sub> <sub>excee</sub>d b<sub>ot</sub>h <sub>con</sub>di<sub>t</sub>i<sub>one</sub>d <sub>va</sub>l<sub>ues.</sub> B<sub>y</sub> <sub>centra</sub>l <sub>symmetry</sub> <sub>t</sub>h<sub>e</sub> <sub>same</sub> i<sub>ntegra</sub>l may <sup>b</sup>e wr<sup>i</sup>tten w<sup>i</sup>t<sup>h</sup> t<sup>h</sup>e <sup>l</sup>ower jo<sup>i</sup>nt <sup>CDF</sup> a<sup>f</sup>ter rep<sup>l</sup>ac<sup>i</sup>ng t<sup>h</sup>e common m<sup>i</sup>n<sup>i</sup>mum <sup>b</sup>y a common max<sup>i</sup>mum.

Theorem 7.2 (Gaussian score limit for fixed candidate count). Fix $q$ and $1 \ \leq \ k \ < \ q .$ . If m $ \infty ,$ $d - m  \infty ,$ and m/d $ \alpha \in [ 0 , 1 ]$ , then

$$
\mathbb { E } \frac { \left| \mathcal { N } _ { k } \cap \widetilde { \mathcal { N } } _ { k } \right| } { k } \longrightarrow \mathcal { R } _ { q , k } ( \sqrt { \alpha } ) .
$$

For $k = 1$

$$
\mathrm { P r } (  { N _ { 1 } } =  { \widetilde { \mathcal { N } } } _ { 1 } ) = p _ { q } \left( \sqrt { m / d } \right) + O _ { q } \left( m ^ { - 1 / 2 } + ( d - m ) ^ { - 1 / 2 } \right) .
$$

In particular, $p _ { q } ( 0 ) = 1 / q$ and $\mathcal { R } _ { q , k } ( 0 ) = k / q .$ . Thus, for fixed $q ,$ k and m/d → 0, the probability that the nearest neighbors agree tends to $1 / q ,$ and the expected normalized top-k overlap tends to $k / q$

The <sub>p</sub>roof (section B) a<sub>pp</sub>lies Bentkus’s multivariate Berr<sub>y</sub>–Esseen bound for convex sets [28] to the in<sup>d</sup>epen<sup>d</sup>ent coor<sup>d</sup>inate sums <sup>f</sup>orming t<sup>h</sup>e projecte<sup>d</sup> an<sup>d</sup> omitte<sup>d</sup> scores; <sup>b</sup>ecause q is <sup>fi</sup>xe<sup>d</sup>, summing over t<sup>h</sup>e q <sup>d</sup>isjoint c<sup>h</sup>oices o<sup>f</sup> common minimizer gives t<sup>h</sup>e state<sup>d</sup> $O _ { q } ( m ^ { - 1 / 2 } + ( d - m ) ^ { - 1 / 2 } )$ <sub>trans</sub>f<sub>er.</sub>

M<sub>ore genera</sub>ll<sub>y,</sub> l<sub>et</sub> $R _ { r : q }$ be the rank of the second-coordinate value paired with the rth ordered firstcoordinate value. This paired value is the concomitant of the rth order statistic. Then

$$
\mathcal { R } _ { q , k } ( \rho ) = \frac { 1 } { k } \sum _ { r = 1 } ^ { k } \operatorname* { P r } ( R _ { r : q } \leq k ) .
$$

The distribution of ranks in a finite sam<sub>p</sub>le is classical [29]. We reduce nei<sub>g</sub>hborhoods after random project<sup>i</sup>on to t<sup>h</sup>ese <sup>k</sup>erne<sup>l</sup>s at corre<sup>l</sup>at<sup>i</sup>on ${ \sqrt { m / d } }$ <sub>,</sub> <sub>esta</sub>bli<sub>s</sub>h <sub>t</sub>h<sub>e</sub> fi<sub>n</sub>i<sub>te-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>trans</sub>f<sub>er,</sub> <sub>an</sub>d d<sub>er</sub>i<sub>ve</sub> <sub>t</sub>h<sub>e</sub> top-k limit.

<sup>F</sup>r<sup>i</sup>eze an<sup>d</sup> Jerrum <sup>d</sup>er<sup>i</sup>ve<sup>d</sup> t<sup>h</sup>e sma<sup>ll</sup>-corre<sup>l</sup>at<sup>i</sup>on expans<sup>i</sup>on o<sup>f</sup> t<sup>h</sup>e same <sup>G</sup>auss<sup>i</sup>an common-argmax <sup>k</sup>erne<sup>l</sup> [30]. In common-minimum notation it <sub>g</sub>ives the followin<sub>g</sub> result; the <sub>p</sub>roof, b<sub>y</sub> diferentiatin<sub>g</sub> e<sub>q</sub>uation (8) at $\rho = 0 ;$ , <sup>a</sup>pp<sup>ears</sup> <sup>in</sup> <sup>section</sup> <sup>B.</sup>

Proposition 7.3 (Classical small-correlation expansion). For fixed $q \geq 2 ,$

$$
p _ { q } ( \boldsymbol { \rho } ) = \frac { 1 } { q } + c _ { q } \boldsymbol { \rho } + O _ { q } ( \boldsymbol { \rho } ^ { 2 } ) , \qquad c _ { q } = q ^ { 2 } ( q - 1 ) \left[ \int _ { - \infty } ^ { \infty } \Phi ( x ) ^ { 2 } \overline { { \Phi } } ( x ) ^ { q - 2 } d x \right] ^ { 2 } .
$$

In particular, $c _ { 2 } = 1 / \pi ,$ recovering the small-correlation expansion of Sheppard’s formula. Hence

$$
\operatorname* { P r } ( \mathcal { N } _ { 1 } = \widetilde { \mathcal { N } } _ { 1 } ) = \frac { 1 } { q } + c _ { q } \sqrt { \frac { m } { d } } + O _ { q } \left( \frac { m } { d } + \frac { 1 } { \sqrt { m } } + \frac { 1 } { \sqrt { d - m } } \right) .
$$

## 7.4 The JL bound can hold while rankings collapse

<sup>S</sup>ect<sup>i</sup>on 3 s<sup>h</sup>owe<sup>d</sup> t<sup>h</sup>at t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> a<sup>l</sup>one <sup>f</sup>orces no reta<sup>i</sup>ne<sup>d</sup> geometry: an <sup>i</sup>n<sup>d</sup>epen<sup>d</sup>ent <sup>G</sup>auss<sup>i</sup>an rep<sup>l</sup>acement map can sat<sup>i</sup>s<sup>f</sup>y <sup>i</sup>t. <sup>Th</sup>e same separat<sup>i</sup>on a<sup>l</sup>so <sup>h</sup>o<sup>ld</sup>s <sup>f</sup>or a <sup>li</sup>near project<sup>i</sup>on. <sup>Al</sup>t<sup>h</sup>oug<sup>h</sup> t<sup>h</sup>e pa<sup>i</sup>rw<sup>i</sup>se <sup>l</sup>aw suggests t<sup>h</sup>at ran<sup>ki</sup>ngs <sup>d</sup>egra<sup>d</sup>e un<sup>d</sup>er t<sup>hi</sup>n project<sup>i</sup>on, <sup>i</sup>t <sup>d</sup>oes not s<sup>h</sup>ow w<sup>h</sup>et<sup>h</sup>er one <sup>fi</sup>xe<sup>d</sup> map can sat<sup>i</sup>s<sup>f</sup>y t<sup>h</sup>e J<sup>L</sup> b<sub>oun</sub>d <sub>on</sub> <sub>t</sub>h<sub>e</sub> <sub>same</sub> <sub>samp</sub>l<sub>e</sub> <sub>w</sub>hil<sub>e</sub> i<sub>ts</sub> <sub>ran</sub>ki<sub>ngs</sub> <sub>co</sub>ll<sub>apse.</sub> W<sub>e</sub> <sub>prove</sub> <sub>t</sub>h<sub>at</sub> <sub>t</sub>h<sub>ese</sub> <sub>events</sub> <sub>coex</sub>i<sub>st.</sub>

L<sub>e</sub>t $A = \sqrt { d / m } R .$ , where R has Haar-distributed orthonormal rows. For distinct x, y, z, define

$$
s _ { A } ( x ; \gamma , z ) = { \mathrm { s g n } } \left[ \left( \left\| x - \gamma \right\| ^ { 2 } - \left\| x - z \right\| ^ { 2 } \right) \left( \left\| A ( x - \gamma ) \right\| ^ { 2 } - \left\| A ( x - z ) \right\| ^ { 2 } \right) \right] .
$$

For n data points, the average Kendall correlation across queries is

$$
\overline { { \tau } } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( { n - 1 \atop 2 } \right) ^ { - 1 } \sum _ { j < k \atop j , k \ne i } s _ { A } ( X _ { i } ; X _ { j } , X _ { k } ) ,
$$

w<sup>h</sup>ose ex<sub>p</sub>ectation invo<sup>l</sup>ves t<sup>h</sup>e sin<sub>g</sub><sup>l</sup>e-<sub>q</sub>uer<sub>y</sub> constant

$$
\tau _ { m , d } = \frac 2 \pi \mathbb E \arcsin \sqrt { B } , \qquad B \sim \mathrm { B e t a } \left( \frac m 2 , \frac { d - m } 2 \right) :
$$

b<sub>y</sub> <sub>t</sub>h<sub>eorem</sub> $7 . 1 , \tau _ { m , d }$ is t<sup>h</sup>e expecte<sup>d</sup> Ken<sup>d</sup>a<sup>ll</sup> corre<sup>l</sup>ation <sup>f</sup>or one query un<sup>d</sup>er a ran<sup>k</sup>-m Haar projection.

Theorem 7.4 (One map can satisfy the JL bound while rankings collapse). Let $X _ { 1 } , \dotsc , X _ { n } \stackrel { i . i . d . } { \sim } { \mathcal { N } } ( 0 , I _ { d } )$ independently ofA, with $n \geq 3 , 0 < m < d , 0 < \varepsilon < 1$ , and $0 < \delta , \mathfrak { n } < 1$

(i) Conditional concentration. Conditionally on $A , \mathbb { E } [ \overline { { \tau } } _ { n } \mid A ] = \tau _ { m , d } a n d , f o r t > 0 _ { \iota }$

$$
\operatorname* { P r } \left( \left| \overline { \tau } _ { n } - \tau _ { m , d } \right| > t \mid A \right) \leq 2 \exp \left( - \frac { \lfloor n / 3 \rfloor t ^ { 2 } } { 2 } \right) .
$$

(ii) JL event. A universal $C > 0$ exists such that, $i f m \geq C \varepsilon ^ { - 2 } \log ( n / \delta )$ , then A satisfies the $\varepsilon - J L$ bound on the sample with probability at least $1 - \delta$

(iii) Simultaneous conclusion. With probability at least $1 - \delta - \eta _ { \ast }$ , the JL event in part $( i i )$ and

$$
\overline { { \tau } } n \leq \sqrt { \frac { m } { d } } + \sqrt { \frac { 2 \log ( 2 / \mathfrak { n } ) } { \lfloor n / 3 \rfloor } }
$$

hold simultaneously.

The <sub>p</sub>roof (section B) s<sub>y</sub>mmetrizes the Kendall kernel and a<sub>pp</sub>lies the blockin<sub>g</sub> ine<sub>q</sub>ualit<sub>y</sub> of Hoefdin<sub>g</sub> [19] for bounded order-three U-statistics<sub>,</sub> <sub>p</sub>artitionin<sub>g</sub> the sam<sub>p</sub>le into disjoint tri<sub>p</sub>les to restore inde<sub>p</sub>endence an<sup>d</sup> g<sup>i</sup>ve con<sup>di</sup>t<sup>i</sup>ona<sup>l</sup> concentrat<sup>i</sup>on; a un<sup>i</sup>on <sup>b</sup>oun<sup>d</sup> g<sup>i</sup>ves t<sup>h</sup>e J<sup>L</sup> event. <sup>If</sup> $n \to \infty$ <sub>an</sub>d l<sub>og</sub> $n \ll m \ll d ,$ <sub>t</sub>h<sub>e</sub> same samp<sup>l</sup>e<sup>d</sup> map can satis<sup>f</sup><sub>y</sub> t<sup>h</sup>e JL <sup>b</sup>oun<sup>d</sup> <sup>f</sup>or <sup>fi</sup>xe<sup>d</sup> ε w<sup>h</sup>i<sup>l</sup>e $\overline { { \tau } } _ { n } \to 0$ i<sub>n</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>ty.</sub>

Remark 7.5 (Haar projection versus replacement map). The normalized replacement distance has law $\chi _ { m } ^ { 2 } / m ,$ exact<sup>l</sup>y t<sup>h</sup>e <sup>di</sup>stort<sup>i</sup>on <sup>l</sup>aw <sup>f</sup>or one vector un<sup>d</sup>er an <sup>i</sup>.<sup>i</sup>.<sup>d</sup>. <sup>G</sup>auss<sup>i</sup>an J<sup>L</sup> matr<sup>i</sup>x. <sup>F</sup>or t<sup>h</sup>e <sup>H</sup>aar map <sup>i</sup>n <sub>t</sub>h<sub>eorem</sub> 7<sub>.</sub>4<sub>,</sub>

$$
{ \frac { \left\| A \nu \right\| ^ { 2 } } { \left\| \nu \right\| ^ { 2 } } } { \frac { d } { d } } { \frac { d } { m } } B , \qquad B \sim \mathrm { B e t a } \left( { \frac { m } { 2 } } , { \frac { d - m } { 2 } } \right) , \qquad \mathrm { V a r } \left( { \frac { \left\| A \nu \right\| ^ { 2 } } { \left\| \nu \right\| ^ { 2 } } } \right) = { \frac { 2 ( d - m ) } { m ( d + 2 ) } } .
$$

This variance agrees with 2/m only when $m / d \to 0$ <sub>.</sub> E<sub>ven t</sub>h<sub>ere, a</sub>i<sub>rw</sub>i<sub>se</sub> di<sub>stances are</sub> d<sub>e en</sub>d<sub>ent</sub> i<sub>n</sub> <sup>dif</sup>erent ways: every projecte<sup>d</sup> pa<sup>i</sup>r uses t<sup>h</sup>e same project<sup>i</sup>on, w<sup>h</sup>ereas every rep<sup>l</sup>acement pa<sup>i</sup>r <sup>b</sup>e<sup>l</sup>ongs <sub>to t</sub>h<sub>e same rep</sub>l<sub>acement c</sub>l<sub>ou</sub>d<sub>.</sub> F<sub>or one</sub> di<sub>stance, t</sub>h<sub>e two construct</sub>i<sub>ons t</sub>h<sub>ere</sub>f<sub>ore occupy t</sub>h<sub>e oppos</sub>i<sub>te</sub> endpoints recorded in section 5.3: the maximal dependence m/d attainable from a rank-m linear <sub>s</sub>k<sub>etc</sub>h<sub>, an</sub>d <sub>no</sub> d<sub>epen</sub>d<sub>ence at a</sub>ll<sub>.</sub>

O<sub>ur</sub> <sub>resu</sub>l<sub>ts</sub> <sub>s</sub>h<sub>ow</sub> <sub>t</sub>h<sub>at</sub> <sub>ran</sub>ki<sub>ng</sub> <sub>agreement</sub> f<sub>o</sub>ll<sub>ows</sub> <sub>t</sub>h<sub>e</sub> $\sqrt { m / d }$ corre<sup>l</sup>at<sup>i</sup>on sca<sup>l</sup>e, w<sup>h</sup>ereas t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> <sup>d</sup>oes <sub>not</sub> <sub>constra</sub>i<sub>n</sub> i<sub>t.</sub> W<sub>e</sub> <sub>next</sub> <sub>stu</sub>d<sub>y</sub> <sub>c</sub>l<sub>uster</sub>i<sub>ng</sub> <sub>an</sub>d <sub>est</sub>i<sub>mat</sub>i<sub>on,</sub> <sub>w</sub>hi<sub>c</sub>h d<sub>epen</sub>d <sub>on</sub> h<sub>ow</sub> <sub>we</sub>ll <sub>a</sub> <sub>s</sub>k<sub>etc</sub>h <sub>preserves</sub> <sub>t</sub>h<sub>e</sub> <sub>a</sub>bili<sub>ty</sub> <sub>to</sub> di<sub>st</sub>i<sub>ngu</sub>i<sub>s</sub>h <sub>near</sub>b<sub>y</sub> di<sub>str</sub>ib<sub>ut</sub>i<sub>ons.</sub>

## 8. Covariance-Shape Information

Di<sub>stance</sub> <sub>ran</sub>ki<sub>ngs</sub> <sub>concern</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e-samp</sub>l<sub>e</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>es.</sub> Cl<sub>uster</sub>i<sub>ng</sub> <sub>an</sub>d <sub>est</sub>i<sub>mat</sub>i<sub>on</sub> i<sub>nstea</sub>d <sub>as</sub>k h<sub>ow</sub> <sub>we</sub>ll projecte<sup>d d</sup>ata <sup>di</sup>st<sup>i</sup>ngu<sup>i</sup>s<sup>h</sup> near<sup>b</sup>y <sup>di</sup>str<sup>ib</sup>ut<sup>i</sup>ons. <sup>W</sup>e use <sup>Fi</sup>s<sup>h</sup>er <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on to measure t<sup>h</sup>at <sup>l</sup>oca<sup>l di</sup>st<sup>i</sup>nct<sup>i</sup>on <sub>an</sub>d <sub>to remove overa</sub>ll <sub>sca</sub>l<sub>e</sub> b<sub>e</sub>f<sub>ore measur</sub>i<sub>ng covar</sub>i<sub>ance s</sub>h<sub>ape.</sub>

For a regular parametric family with density p<sub>θ</sub>, the Fisher information matrix is the second-moment <sub>matr</sub>i<sub>x o</sub>f <sub>t</sub>h<sub>e score,</sub>

$$
\mathcal { T } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { \theta } } \left[ \nabla _ { \boldsymbol { \theta } } \log p _ { \boldsymbol { \Theta } } ( \boldsymbol { X } ) \big ( \nabla _ { \boldsymbol { \theta } } \log p _ { \boldsymbol { \Theta } } ( \boldsymbol { X } ) \big ) ^ { \top } \right] .\tag{9}
$$

U<sub>n</sub>d<sub>er</sub> <sub>t</sub>h<sub>e</sub> <sub>usua</sub>l <sub>regu</sub>l<sub>ar</sub>i<sub>ty</sub> <sub>con</sub>di<sub>t</sub>i<sub>ons</sub> i<sub>t</sub> <sub>a</sub>l<sub>so</sub> <sub>equa</sub>l<sub>s</sub> $- \mathbb { E } _ { \theta } [ \nabla _ { \theta } ^ { 2 } \log p _ { \theta } ( X ) ]$ <sub>.</sub> Fi<sub>s</sub>h<sub>er</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>g</sub>i<sub>ves</sub> <sub>t</sub>h<sub>e</sub> l<sub>oca</sub>l q<sup>uadratic</sup> <sup>a</sup>pp<sup>roximation</sup>

$$
D _ { \mathrm { K L } } \big ( P _ { \boldsymbol { \theta } } | | o P _ { \boldsymbol { \theta } + \boldsymbol { \delta } } \big ) = \frac { 1 } { 2 } \boldsymbol { \delta } ^ { \top } \mathcal { Z } ( \boldsymbol { \theta } ) \boldsymbol { \delta } + o \big ( \| \boldsymbol { \delta } \| ^ { 2 } \big )
$$

[31]. It therefore measures local statistical distin<sub>g</sub>uishabilit<sub>y;</sub> throu<sub>g</sub>h the Cramér–Rao bound<sub>,</sub> its inverse <sub>a</sub>l<sub>so</sub> <sub>contro</sub>l<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>var</sub>i<sub>ance</sub> <sub>o</sub>f <sub>regu</sub>l<sub>ar</sub> <sub>est</sub>i<sub>mators.</sub>

W<sup>h</sup>en a <sub>p</sub>arameter o<sup>f</sup> interest ϑ is accom<sub>p</sub>anied b<sub>y</sub> a nuisance <sub>p</sub>arameter ζ, t<sup>h</sup>e e<sup>fi</sup>cient in<sup>f</sup>ormation <sub>removes</sub> <sub>score</sub> <sub>var</sub>i<sub>at</sub>i<sub>on</sub> <sub>exp</sub>l<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> <sub>t</sub>h<sub>e</sub> <sub>nu</sub>i<sub>sance</sub> di<sub>rect</sub>i<sub>on.</sub> If <sub>t</sub>h<sub>e</sub> <sub>nu</sub>i<sub>sance</sub> bl<sub>oc</sub>k i<sub>s</sub> <sub>nons</sub>i<sub>ngu</sub>l<sub>ar,</sub> <sub>t</sub>h<sub>e</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ent</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> S<sub>c</sub>h<sub>ur</sub> <sub>comp</sub>l<sub>ement</sub>

$$
\mathcal { T } _ { \mathrm { e f f } } = \mathcal { T } _ { \vartheta \vartheta } - \mathcal { T } _ { \vartheta \zeta } \mathcal { T } _ { \zeta \zeta } ^ { - 1 } \mathcal { T } _ { \zeta \vartheta } .\tag{10}
$$

<sup>E</sup>qu<sup>i</sup>va<sup>l</sup>ent<sup>l</sup>y, <sup>i</sup>t <sup>i</sup>s t<sup>h</sup>e <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on <sup>i</sup>n t<sup>h</sup>e res<sup>id</sup>ua<sup>l</sup> score a<sup>f</sup>ter project<sup>i</sup>ng out t<sup>h</sup>e span o<sup>f</sup> t<sup>h</sup>e nu<sup>i</sup>sance score. <sup>Thi</sup>s <sup>i</sup>s t<sup>h</sup>e project<sup>i</sup>on <sup>i</sup>n score space ant<sup>i</sup>c<sup>i</sup>pate<sup>d</sup> <sup>i</sup>n sect<sup>i</sup>on 4.4: t<sup>h</sup>e target space <sup>i</sup>s spanne<sup>d</sup> <sup>b</sup>y scores rat<sup>h</sup>er t<sup>h</sup>an <sup>b</sup>y <sup>f</sup>eatures o<sup>f</sup> a <sup>di</sup>stance, an<sup>d</sup> ort<sup>h</sup>ogona<sup>l</sup> project<sup>i</sup>on aga<sup>i</sup>n <sup>d</sup>eterm<sup>i</sup>nes w<sup>h</sup>at t<sup>h</sup>e o<sup>b</sup>servat<sup>i</sup>on reta<sup>i</sup>ns.

## 8.1 Fisher information under projection

Let R have Haar-distributed orthonormal rows and write $\Pi = R ^ { \dagger } R$ <sub>.</sub> F<sub>or</sub> $X \sim { \mathcal { N } } ( \mu , \sigma ^ { 2 } I _ { d } )$ <sub>o</sub>b<sub>serve</sub>d <sub>t</sub>h<sub>roug</sub>h t<sup>hi</sup>s <sup>k</sup>nown project<sup>i</sup>on, t<sup>h</sup>e <sup>Fi</sup>s<sup>h</sup>er <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on <sup>f</sup>or t<sup>h</sup>e mean <sup>i</sup>s exact<sup>l</sup>y

$$
\mathbb { Z } _ { \mu } ( R X ) = \frac { 1 } { \sigma ^ { 2 } } \Pi .\tag{11}
$$

It is the identity on the observed m-space and zero on the kernel. For a scalar submodel $\mu = \theta \nu ,$ <sub>t</sub>h<sub>e reta</sub>i<sub>ne</sub>d f<sub>ract</sub>i<sub>on</sub> $\nu ^ { \top } \Pi \nu / \Vert \nu \Vert ^ { 2 }$ i<sub>s</sub> B<sub>e</sub>t<sub>a</sub> $\scriptstyle ( { \frac { m } { 2 } } , { \frac { d - m } { 2 } } )$ -distributed with mean m/d, and the Cramér–Rao variance inflation in direction v is its reciprocal.

## 8.2 Mean, scale, and shape contraction

A l<sub>oca</sub>l <sub>pertur</sub>b<sub>at</sub>i<sub>on</sub> <sub>o</sub>f <sub>a</sub> G<sub>auss</sub>i<sub>an</sub> <sub>component</sub> d<sub>ecomposes</sub> i<sub>nto</sub> <sub>a</sub> <sub>c</sub>h<sub>ange</sub> <sub>o</sub>f <sub>mean,</sub> <sub>a</sub> <sub>sca</sub>l<sub>ar</sub> <sub>c</sub>h<sub>ange</sub> <sub>o</sub>f l<sub>og-sca</sub>l<sub>e, an</sub>d <sub>a trace</sub>l<sub>ess c</sub>h<sub>ange o</sub>f <sub>covar</sub>i<sub>ance s</sub>h<sub>ape.</sub> T<sub>o</sub> d<sub>e</sub>fi<sub>ne s</sub>h<sub>ape</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub>l<sub>y o</sub>f <sub>sca</sub>l<sub>e, cons</sub>id<sub>er</sub> $\Sigma _ { \zeta , \epsilon } ^ { - } = e ^ { \zeta } ( I _ { d } + \epsilon H )$ <sub>w</sub>i<sub>t</sub>h <sub>tr</sub> $H = 0 .$ . Here ζ contro<sup>l</sup>s overa<sup>ll</sup> sca<sup>l</sup>e and ϵ t<sup>h</sup>e ma<sub>g</sub>nitude o<sup>f</sup> t<sup>h</sup>e trace<sup>l</sup>ess s<sup>h</sup>a<sub>p</sub>e <sup>d</sup>irection H. Treating ζ as a nuisance parameter projects t<sup>h</sup>e s<sup>h</sup>ape score $R H R ^ { \top }$ <sub>away</sub> f<sub>rom</sub> <sub>t</sub>h<sub>e</sub> <sub>sca</sub>l<sub>e</sub> <sub>score,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s proport</sub>i<sub>ona</sub>l <sub>to</sub> $I _ { m } .$ <sub>.</sub> Th<sub>e e</sub>fi<sub>c</sub>i<sub>ent s</sub>h<sub>ape score</sub> i<sub>s t</sub>h<sub>ere</sub>f<sub>ore t</sub>h<sub>e trace</sub>l<sub>ess part o</sub>f $R H R ^ { \top }$

$$
R H R ^ { \top } - \frac { \mathrm { t r } ( R H R ^ { \top } ) } { m } I _ { m } .
$$

<sup>Haar</sup> <sup>avera</sup>g<sup>in</sup>g g<sup>ives</sup> <sup>a</sup> <sup>se</sup>p<sup>arate</sup> <sup>contraction</sup> <sup>factor</sup> <sup>for</sup> <sup>each</sup> p<sup>erturbation</sup> <sup>t</sup>yp<sup>e.</sup>

Table 2. Haar-averaged fractions of local Gaussian Fisher information, equivalently of local Kullback–Leibler divergence. The shape fraction is computed after eliminating unknown overall scale; unlike the log-scale fraction, it varies across row spaces, and the table reports its Haar mean.
<table><tr><td>Perturbation</td><td>Retained fraction</td><td>Dependence on row space</td></tr><tr><td>Mean direction</td><td>m/d</td><td>Beta-distributed around its mean</td></tr><tr><td>Scalar log-scale</td><td>m/d</td><td>Deterministic</td></tr><tr><td>Diffuse traceless shape</td><td> $\frac { ( m - 1 ) ( m + 2 ) } { ( d - 1 ) ( d + 2 ) } \sim ( m / d ) ^ { 2 }$ </td><td>Random; displayed fraction is the Haar mean</td></tr></table>

Theorem 8.1 (Local contraction taxonomy). A Haar rank-m projection retains the three expected fractions summarized in Table 2; the shape row uses eficient Fisher information after eliminating the nuisance scale ζ. For any symmetric traceless H, the exact moment identity is

$$
\mathbb { E } _ { R } \left. R H R ^ { \top } - \frac { \mathrm { t r } ( R H R ^ { \top } ) } { m } I _ { m } \right. _ { F } ^ { 2 } = \frac { ( m - 1 ) ( m + 2 ) } { ( d - 1 ) ( d + 2 ) } \left. H \right. _ { F } ^ { 2 } .
$$

At t<sup>h</sup>e isotropic mo<sup>d</sup>e<sup>l</sup>, t<sup>h</sup>e unprojecte<sup>d</sup> e<sup>fi</sup>cient Fis<sup>h</sup>er in<sup>f</sup>ormation <sup>f</sup>or ϵ is $\textstyle { \frac { 1 } { 2 } } \left\| H \right\| _ { F } ^ { 2 }$ , w<sup>hil</sup>e t<sup>h</sup>e projecte<sup>d</sup> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> i<sub>s</sub> <sub>one</sub> h<sub>a</sub>lf <sub>o</sub>f <sub>t</sub>h<sub>e</sub> <sub>square</sub>d <sub>norm</sub> i<sub>ns</sub>id<sub>e</sub> <sub>t</sub>h<sub>e</sub> <sub>expectat</sub>i<sub>on.</sub> Th<sub>e</sub> H<sub>aar</sub> <sub>moment</sub> id<sub>ent</sub>i<sub>ty</sub> <sub>t</sub>h<sub>ere</sub>f<sub>ore</sub> <sub>g</sub>i<sub>ves</sub> <sub>t</sub>h<sub>e</sub> <sub>contract</sub>i<sub>on</sub> f<sub>actor</sub> f<sub>or</sub> <sub>s</sub>h<sub>ape</sub> di<sub>rect</sub>l<sub>y;</sub> <sub>t</sub>h<sub>e</sub> <sub>proo</sub>f<sub>,</sub> b<sub>ase</sub>d <sub>on</sub> <sub>secon</sub>d<sub>-or</sub>d<sub>er</sub> H<sub>aar</sub> i<sub>ntegrat</sub>i<sub>on,</sub> <sub>appears</sub> i<sub>n</sub> <sub>sect</sub>i<sub>on</sub> B<sub>.</sub> A dif<sub>use</sub> <sub>s</sub>h<sub>ape</sub> <sub>pertur</sub>b<sub>at</sub>i<sub>on</sub> di<sub>str</sub>ib<sub>utes</sub> i<sub>ts</sub> <sub>square</sub>d F<sub>ro</sub>b<sub>en</sub>i<sub>us</sub> <sub>norm</sub> <sub>across</sub> <sub>many</sub> <sub>e</sub>i<sub>gen</sub>di<sub>rect</sub>i<sub>ons</sub> rat<sup>h</sup>er t<sup>h</sup>an concentrat<sup>i</sup>ng <sup>i</sup>t <sup>i</sup>n a <sup>f</sup>ew sp<sup>ik</sup>es. <sup>S</sup>uc<sup>h</sup> a pertur<sup>b</sup>at<sup>i</sup>on <sup>l</sup>oses <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on tw<sup>i</sup>ce. <sup>P</sup>roject<sup>i</sup>on removes <sub>coor</sub>di<sub>nates an</sub>d <sub>a</sub>l<sub>so re</sub>d<sub>uces contrast w</sub>i<sub>t</sub>hi<sub>n t</sub>h<sub>e reta</sub>i<sub>ne</sub>d <sub>coor</sub>di<sub>nates.</sub> Fi<sub>gure</sub> 5 <sub>compares t</sub>h<sub>e exact constant</sub> <sub>w</sub>i<sub>t</sub>h <sub>s</sub>i<sub>mu</sub>l<sub>at</sub>i<sub>on.</sub>

F<sub>or</sub> <sub>two</sub> <sub>we</sub>ll<sub>-separate</sub>d G<sub>auss</sub>i<sub>an</sub> <sub>components</sub> <sub>w</sub>i<sub>t</sub>h <sub>common</sub> i<sub>sotrop</sub>i<sub>c</sub> <sub>covar</sub>i<sub>ance,</sub> <sub>t</sub>h<sub>e</sub> l<sub>ea</sub>di<sub>ng</sub> l<sub>ogar</sub>i<sub>t</sub>h<sub>m</sub> o<sup>f</sup> t<sup>h</sup>e <sup>B</sup>ayes error <sup>i</sup>s proport<sup>i</sup>ona<sup>l</sup> to t<sup>h</sup>e square<sup>d</sup> <sup>di</sup>stance <sup>b</sup>etween t<sup>h</sup>e<sup>i</sup>r means. <sup>A</sup> <sup>H</sup>aar project<sup>i</sup>on mu<sup>l</sup>t<sup>i</sup>p<sup>li</sup>es that ex<sub>p</sub>onent b<sub>y</sub> the Beta-distributed factor in e<sub>q</sub>uation (11), which has mean m/d and concentrates as both dimensions <sub>g</sub>row. Das<sub>g</sub>u<sub>p</sub>ta [14] obtained lo<sub>g</sub>arithmic <sub>p</sub>rojection dimensions for learnin<sub>g</sub> se<sub>p</sub>arated <sub>m</sub>i<sub>xtures.</sub> O<sub>ur</sub> l<sub>oca</sub>l i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>rate</sub> i<sub>s</sub> <sub>cons</sub>i<sub>stent</sub> <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>at</sub> <sub>resu</sub>l<sub>t</sub> b<sub>ut</sub> i<sub>s</sub> <sub>not</sub> <sub>a</sub> <sub>comp</sub>l<sub>ete</sub> <sub>c</sub>l<sub>uster</sub>i<sub>ng</sub> <sub>guarantee.</sub>

## 8.3 Compression toward sphericity and the spike exception

Th<sub>e</sub> l<sub>oca</sub>l <sub>ca</sub>l<sub>cu</sub>l<sub>at</sub>i<sub>on</sub> d<sub>escr</sub>ib<sub>es</sub> i<sub>n</sub>fi<sub>n</sub>i<sub>tes</sub>i<sub>ma</sub>l <sub>c</sub>h<sub>anges.</sub> T<sub>o connect</sub> i<sub>t w</sub>i<sub>t</sub>h <sub>a g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>covar</sub>i<sub>ance,</sub> l<sub>et</sub> $A = \sqrt { d / m } R$ d<sub>e</sub>fi<sub>ne</sub> $C _ { A } = A \Sigma A ^ { \top } , \operatorname { p u t } \bar { \lambda } = \mathrm { t r } \big ( \Sigma \big ) / d ,$ <sub>an</sub>d <sub>set</sub> $\Delta = \Sigma - \bar { \lambda } I .$ <sub>.</sub> Th<sub>en</sub>

$$
C _ { A } - \frac { { \mathrm { t r } } \Sigma } { m } I _ { m } = \frac { d } { m } R \Delta R ^ { \top } .
$$

S<sub>tan</sub>d<sub>ar</sub>d b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> <sub>ran</sub>d<sub>om</sub> <sub>compress</sub>i<sub>on</sub> id<sub>ent</sub>if<sub>y</sub> <sub>two</sub> di<sub>mens</sub>i<sub>on</sub>l<sub>ess</sub> <sub>quant</sub>i<sub>t</sub>i<sub>es</sub> <sub>t</sub>h<sub>at</sub> <sub>contro</sub>l <sub>convergence</sub> <sub>to</sub> a scalar matrix in o<sub>p</sub>erator norm [14<sub>,</sub> 15]. The <sub>q</sub>uantit<sub>y</sub> $\sqrt { m } \ \| \Delta \| _ { F } / \mathrm { t r } \mathbf { \bar { Z } }$ <sub>contro</sub>l<sub>s t</sub>h<sub>e</sub> di<sub>str</sub>ib<sub>ute</sub>d <sub>an</sub>i<sub>sotrop</sub>i<sub>c</sub> bulk: the square-root factor reflects accumulation across many weak directions. The quantity m $\| \Delta \| _ { \mathrm { o p } } / \mathrm { t r } \Sigma$ <sub>contro</sub>l<sub>s</sub> <sub>t</sub>h<sub>e</sub> l<sub>argest</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l di<sub>rect</sub>i<sub>on,</sub> f<sub>or</sub> <sub>w</sub>hi<sub>c</sub>h <sub>no</sub> <sub>suc</sub>h <sub>averag</sub>i<sub>ng</sub> <sub>occurs.</sub> Th<sub>us</sub> <sub>t</sub>h<sub>e</sub> <sub>compresse</sub>d <sub>covar</sub>i<sub>ance</sub> <sub>approac</sub>h<sub>es</sub> <sub>a</sub> <sub>sca</sub>l<sub>ar</sub> <sub>matr</sub>i<sub>x</sub> i<sub>n</sub> <sub>reg</sub>i<sub>mes</sub> <sub>w</sub>h<sub>ere</sub> b<sub>ot</sub>h <sub>quant</sub>i<sub>t</sub>i<sub>es</sub> <sub>van</sub>i<sub>s</sub>h<sub>.</sub> Th<sub>e</sub> <sub>resu</sub>l<sub>t</sub>i<sub>ng</sub> <sub>roun</sub>di<sub>ng</sub> <sub>towar</sub>d <sub>a</sub> <sub>sca</sub>l<sub>ar</sub> covariance is Das<sub>g</sub>u<sub>p</sub>ta’s s<sub>p</sub>hericalization <sub>p</sub>henomenon. Das<sub>g</sub>u<sub>p</sub>ta [14<sub>,</sub> 15] established the <sub>p</sub>henomenon for eccentric Gaussians<sub>,</sub> and Das<sub>g</sub>u<sub>p</sub>ta<sub>,</sub> Hsu<sub>,</sub> and Verma [16] extended it to more <sub>g</sub>eneral distributions.

A <sub>su</sub>fi<sub>c</sub>i<sub>ent</sub>l<sub>y</sub> l<sub>arge</sub> <sub>spectra</sub>l <sub>sp</sub>ik<sub>e</sub> <sub>v</sub>i<sub>o</sub>l<sub>ates</sub> <sub>t</sub>h<sub>e</sub> <sub>operator-norm</sub> <sub>con</sub>di<sub>t</sub>i<sub>on</sub> <sub>an</sub>d <sub>can</sub> <sub>rema</sub>i<sub>n</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e</sub> <sub>a</sub>f<sub>ter</sub> <sub>compress</sub>i<sub>on.</sub> I<sub>ts</sub> d<sub>etect</sub>i<sub>on</sub> <sub>t</sub>h<sub>res</sub>h<sub>o</sub>ld d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> <sub>t</sub>h<sub>e</sub> <sub>popu</sub>l<sub>at</sub>i<sub>on</sub> <sub>spectrum,</sub> <sub>samp</sub>l<sub>e</sub> <sub>s</sub>i<sub>ze,</sub> <sub>an</sub>d <sub>stat</sub>i<sub>st</sub>i<sub>ca</sub>l <sub>tas</sub>k<sub>,</sub> <sub>so</sub> <sub>t</sub>h<sub>e rate</sub> d<sub>er</sub>i<sub>ve</sub>d f<sub>or</sub> dif<sub>use s</sub>h<sub>ape s</sub>h<sub>ou</sub>ld <sub>not</sub> b<sub>e app</sub>li<sub>e</sub>d <sub>to</sub> fi<sub>n</sub>i<sub>te-ran</sub>k <sub>sp</sub>ik<sub>es w</sub>i<sub>t</sub>h<sub>out an a</sub>ddi<sub>t</sub>i<sub>ona</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>.</sub>

The quadratic contraction completes the taxonom<sub>y</sub>: mean and scale contract at m/d, difuse traceless <sub>s</sub>h<sub>ape</sub> <sub>at</sub> <sub>or</sub>d<sub>er</sub> $( m / d ) ^ { 2 }$ <sub>,</sub> <sub>an</sub>d <sub>a</sub> <sub>su</sub>fi<sub>c</sub>i<sub>ent</sub>l<sub>y</sub> l<sub>arge</sub> <sub>sp</sub>ik<sub>e</sub> <sub>escapes</sub> <sub>t</sub>h<sub>e</sub> dif<sub>use</sub> <sub>rate.</sub> A k<sub>nown</sub> <sub>no</sub>i<sub>se</sub>l<sub>ess</sub> <sub>map</sub> <sub>can</sub> b<sub>e row ort</sub>h<sub>onorma</sub>li<sub>ze</sub>d <sub>w</sub>i<sub>t</sub>h<sub>out c</sub>h<sub>ang</sub>i<sub>ng</sub> i<sub>ts</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on, w</sub>h<sub>ereas est</sub>i<sub>mators t</sub>h<sub>at use t</sub>h<sub>e output metr</sub>i<sub>c</sub> <sup>directl</sup>y <sup>or</sup> <sup>invert</sup> <sup>nois</sup>y <sup>measurements</sup> <sup>remain</sup> <sup>sensitive</sup> <sup>to</sup> <sup>its</sup> <sup>sin</sup>g<sup>ular</sup> <sup>s</sup>p<sup>ectrum.</sup>

## 9. General Maps: Rank, Spectrum, and Precision

<sup>O</sup>rt<sup>h</sup>ogona<sup>l</sup> project<sup>i</sup>ons <sup>l</sup>ose <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on on<sup>l</sup>y <sup>b</sup>y <sup>di</sup>scar<sup>di</sup>ng <sup>di</sup>rect<sup>i</sup>ons: t<sup>h</sup>ey are <sup>i</sup>sometr<sup>i</sup>es on t<sup>h</sup>e<sup>i</sup>r reta<sup>i</sup>ne<sup>d</sup> <sub>row</sub> <sub>spaces.</sub> A <sub>genera</sub>l li<sub>near</sub> <sub>map,</sub> b<sub>y</sub> <sub>contrast,</sub> <sub>can</sub> <sub>a</sub>l<sub>so</sub> di<sub>stort</sub> <sub>t</sub>h<sub>e</sub> <sub>geometry</sub> i<sub>t</sub> k<sub>eeps.</sub> Th<sub>e</sub> <sub>re</sub>l<sub>evant</sub> <sub>spectra</sub>l <sub>summary</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub> <sub>t</sub>h<sub>e</sub> d<sub>ownstream</sub> <sub>tas</sub>k<sub>.</sub> T<sub>a</sub>bl<sub>e</sub> 3 <sub>prev</sub>i<sub>ews</sub> <sub>t</sub>h<sub>e</sub> <sub>t</sub>h<sub>ree</sub> <sub>cases.</sub>

Table 3. Three downstream uses of a fixed known rank-r map T, with $M = T ^ { \top }$ T and nonzero singular values $s _ { 1 } , \ldots , s _ { r } .$ . The first two rows take $Z \sim \mathcal { N } ( 0 , I _ { d } ) ;$ the third takes $\gamma = T x + \varepsilon \eta$ for independent $x \sim \mathcal { N } ( 0 , I _ { d } )$ and $\boldsymbol \eta \sim \bar { \mathcal { N } } ( 0 , I _ { m } )$
<table><tr><td>Downstream use</td><td>Governing summary Exact law</td><td></td></tr><tr><td>Optimal nonlinear inference</td><td>Rank r</td><td> $\operatorname { p _ { H G R } } ( \left\| Z \right\| ^ { 2 } ; T Z ) = { \sqrt { r / d } }$ </td></tr><tr><td>Unwhitened squared-norm estimate Trace ratio</td><td> $r _ { 2 } ( M )$ </td><td> $\mathrm { C o r r } ( \Vert Z \Vert ^ { 2 } , \Vert T Z \Vert ^ { 2 } ) = \sqrt { r _ { 2 } ( M ) / d }$ </td></tr><tr><td>Inversion with additive noise</td><td>Spectrum near zero</td><td> $\begin{array} { r } { \mathbf { M M S E } ( x \mid T x + \varepsilon \eta , T ) = d - \sum _ { i = 1 } ^ { r } s _ { i } ^ { 2 } / ( s _ { i } ^ { 2 } + \varepsilon ^ { 2 } ) } \end{array}$ </td></tr></table>

## 9.1 The rank–spectrum–precision trichotomy

Theorem 9.1 (Trichotomy). Let $Z \sim \mathcal { N } ( 0 , I _ { d } )$ , let $\boldsymbol { T } \in \mathbb { R } ^ { m \times d }$ be a fixed known map of rank r with nonzero singular values $s _ { i } ,$ and write $D = \left\| Z \right\| ^ { 2 } , Q = \left\| T Z \right\| ^ { 2 } .$ , and $M = T ^ { \top } T$

(i) Best-<sub>p</sub>ossible inference is <sub>g</sub>overned b<sub>y</sub> rank alone: $\mathsf { \rho } _ { \mathrm { H G R } } \bigl ( D ; T Z \bigr ) = \mathsf { \sqrt { \mathstrut r / d } } ,$ so for every $f \in L ^ { 2 } .$ V<sub>a</sub> $\cdot ( \vec { \mathbb { E } } \tilde { [ } f ( D ) \mid T Z ] ) \leq ( r / d ) \breve { \operatorname { V a r } } ( f ( D ) )$

(ii) The unwhitened s<sub>q</sub>uared-norm estimate is <sub>g</sub>overned b<sub>y</sub> the spectrum:

$$
\mathrm { C o r r } ( D , { \mathrm { Q } } ) = { \frac { \mathrm { t r } M } { \sqrt { d \ \mathrm { t r } ( M ^ { 2 } ) } } } = \sqrt { \frac { r _ { 2 } ( M ) } { d } } , \qquad r _ { 2 } ( M ) = { \frac { ( \mathrm { t r } M ) ^ { 2 } } { \mathrm { t r } ( M ^ { 2 } ) } } .
$$

$I f \kappa = s _ { \mathrm { m a x } } / s _ { \mathrm { m i n } }$ over the nonzero singular values, then

$$
r \frac { 4 \kappa ^ { 2 } } { ( \kappa ^ { 2 } + 1 ) ^ { 2 } } \leq r _ { 2 } ( M ) \leq r .
$$

The lower bound is asymptotic to $4 r / \kappa ^ { 2 } \ a s \ \kappa  \infty .$

(iii) Recovery under additive noise is governed by the spectrum near zero: $i f x \sim \mathcal { N } ( 0 , I _ { d } )$ and $\boldsymbol \eta \sim \mathcal { N } ( \dot { 0 } , I _ { m } )$ are independent and $\gamma = T x + \varepsilon \eta$ , then

$$
\operatorname { t r } \operatorname { V a r } ( \mathbb { E } [ x \mid \gamma , T ] ) = \sum _ { i = 1 } ^ { r } { \frac { s _ { i } ^ { 2 } } { s _ { i } ^ { 2 } + \varepsilon ^ { 2 } } } , \qquad \operatorname { M M S E } ( x \mid \gamma , T ) = d - \sum _ { i = 1 } ^ { r } { \frac { s _ { i } ^ { 2 } } { s _ { i } ^ { 2 } + \varepsilon ^ { 2 } } } .
$$

The <sub>p</sub>roof (section B) treats the three <sub>p</sub>arts se<sub>p</sub>aratel<sub>y</sub>. Part (i) follows b<sub>y</sub> orthonormalizin<sub>g</sub> the rows of T as in section 4.3. The singular value decomposition shows that observing TZ is equivalent to observing r <sub>coor</sub>di<sub>nates o</sub>f <sub>a rotate</sub>d <sub>stan</sub>d<sub>ar</sub>d G<sub>auss</sub>i<sub>an.</sub> Wi<sub>t</sub>h $W = ( T T ^ { \top } ) ^ { \dagger / 2 } T$ <sub>, one</sub> h<sub>as</sub> $\mathrm { C o r r } ( \bar { D _ { } } , \| W Z \| ^ { 2 } ) = \sqrt { r / d }$ . If T i<sub>s</sub> i<sub>nvert</sub>ibl<sub>e, t</sub>h<sub>en</sub> $D = ( T Z ) ^ { \top } ( T T ^ { \top } ) ^ { - 1 } ( T Z )$ exact<sup>l</sup><sub>y</sub>.

Part (ii) anal<sub>y</sub>zes $Q = \| T Z \| ^ { 2 }$ without row-orthonormalizing T. We bound its correlation with D using <sub>t</sub>h<sub>e</sub> K<sub>antorov</sub>i<sub>c</sub>h i<sub>nequa</sub>li<sub>ty:</sub> f<sub>or</sub> <sub>pos</sub>i<sub>t</sub>i<sub>ve</sub> $x _ { i } \in [ x _ { \operatorname* { m i n } } , x _ { \operatorname* { m a x } } ] .$

$$
\frac { ( \sum _ { i = 1 } ^ { r } x _ { i } ) ^ { 2 } } { r \sum _ { i = 1 } ^ { r } x _ { i } ^ { 2 } } \geq \frac { 4 x _ { \operatorname* { m i n } } x _ { \operatorname* { m a x } } } { ( x _ { \operatorname* { m i n } } + x _ { \operatorname* { m a x } } ) ^ { 2 } } .
$$

Ta<sup>k</sup>in<sub>g</sub> $x _ { i } = s _ { i } ^ { 2 }$ <sub>an</sub>d $x _ { \mathrm { m a x } } / x _ { \mathrm { m i n } } = \kappa ^ { 2 }$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>t</sub>h<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>trace-rat</sub>i<sub>o</sub> b<sub>oun</sub>d<sub>.</sub>

In <sub>p</sub>art (iii), we ask whether invertin<sub>g</sub> T remains stable under additive noise. The factor $s _ { i } ^ { 2 } / \bigl ( s _ { i } ^ { 2 } + \varepsilon ^ { 2 } \bigr )$ i<sub>s</sub> the variance recovered in singular direction i by the Gaussian posterior mean, or equivalently the Wienerfil<sub>ter</sub> <sub>ga</sub>i<sub>n.</sub> Wh<sub>en</sub> $s _ { i } \ll \varepsilon ,$ the recovered fraction in direction i is negligible even if the map is invertible. I<sub>nvert</sub>ibili<sub>ty</sub> <sub>a</sub>l<sub>one</sub> d<sub>oes</sub> <sub>not</sub> i<sub>mp</sub>l<sub>y</sub> <sub>t</sub>h<sub>at</sub> <sub>t</sub>h<sub>e</sub> <sub>output</sub> E<sub>uc</sub>lid<sub>ean</sub> <sub>metr</sub>i<sub>c</sub> <sub>preserves</sub> di<sub>stances</sub> <sub>or</sub> <sub>t</sub>h<sub>at</sub> <sub>no</sub>i<sub>sy</sub> i<sub>nvers</sub>i<sub>on</sub> i<sub>s sta</sub>bl<sub>e.</sub>

## 9.2 Gaussian-map examples

G<sub>auss</sub>i<sub>an</sub> <sub>maps</sub> i<sub>so</sub>l<sub>ate</sub> <sub>revers</sub>ibl<sub>e</sub> <sub>spectra</sub>l di<sub>stort</sub>i<sub>on</sub> f<sub>rom</sub> <sub>ran</sub>k l<sub>oss.</sub> A <sub>square</sub> i<sub>.</sub>i<sub>.</sub>d<sub>.</sub> G<sub>auss</sub>i<sub>an</sub> <sub>map</sub> i<sub>s</sub> f<sub>u</sub>ll <sub>ran</sub>k<sub>,</sub> <sub>yet</sub> i<sub>ts unw</sub>hi<sub>tene</sub>d <sub>centere</sub>d<sub>-</sub>di<sub>stance corre</sub>l<sub>at</sub>i<sub>on ten</sub>d<sub>s to</sub> $1 / { \sqrt { 2 } } ;$ <sub>row ort</sub>h<sub>onorma</sub>li<sub>zat</sub>i<sub>on restores t</sub>h<sub>e</sub> H<sub>aar</sub> va<sup>l</sup>ue. Fi<sub>g</sub>ure 6 s<sup>h</sup>ows t<sup>h</sup>e one-sta<sub>g</sub>e com<sub>p</sub>arison.

![](images/2847f1e325d0816bbe05ff2e22b8d6d9b23f9e6a89d16eb1c3d0873439c92f41.jpg)  
Figure 5. Local information contraction. Mean and log-scale directions retain m/d; traceless shape retains $( m - 1 ) ( m + \bar { 2 } ) / ( ( d -$ $1 ) ( d + 2 ) )$ ). Markers use $d = 6 0$ and 1,500 Haar frames per m.

![](images/d8200f8cbca862ae1d31e69f9919dcc5768bb98d7b2f73e23802ff780f631786.jpg)  
Figure 6. Correlation between original and mapped centered squared distances after one Haar or i.i.d. Gaussian stage. Markers use the chi-square representation at $d = 2 0 0 ,$ , with 150,000 samples per m.

A <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>map</sub> <sub>exposes</sub> <sub>t</sub>h<sub>e</sub> di<sub>st</sub>i<sub>nct</sub>i<sub>on</sub> <sub>among</sub> <sub>ran</sub>k l<sub>oss,</sub> <sub>revers</sub>ibl<sub>e</sub> <sub>spectra</sub>l di<sub>stort</sub>i<sub>on,</sub> <sub>an</sub>d i<sub>nsta</sub>bili<sub>ty</sub> <sub>un</sub>d<sub>er</sub> <sub>no</sub>i<sub>se.</sub> W<sub>e next as</sub>k h<sub>ow mu</sub>l<sub>t</sub>i<sub>p</sub>l<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent s</sub>k<sub>etc</sub>h<sub>es c</sub>h<sub>ange t</sub>h<sub>e ava</sub>il<sub>a</sub>bl<sub>e ran</sub>k<sub>.</sub>

## 10. Ensembles

W<sub>e</sub> <sub>now</sub> <sub>cons</sub>id<sub>er</sub> <sub>mu</sub>l<sub>t</sub>i<sub>p</sub>l<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>s</sub>k<sub>etc</sub>h<sub>es,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>re</sub>d<sub>uce</sub> <sub>t</sub>h<sub>e</sub> <sub>var</sub>i<sub>a</sub>bili<sub>ty</sub> <sub>o</sub>f <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>s</sub>k<sub>etc</sub>h<sub>.</sub> B<sub>y</sub> <sub>stac</sub>ki<sub>ng</sub> <sub>t</sub>h<sub>em,</sub> <sub>we</sub> <sub>o</sub>b<sub>ta</sub>i<sub>n</sub> <sub>t</sub>h<sub>e</sub>i<sub>r</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>ce</sub>ili<sub>ng;</sub> b<sub>y</sub> <sub>averag</sub>i<sub>ng</sub> <sub>t</sub>h<sub>em,</sub> <sub>we</sub> <sub>o</sub>b<sub>ta</sub>i<sub>n</sub> <sub>a</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> d<sub>eco</sub>d<sub>er</sub> <sub>w</sub>h<sub>ose</sub> fi<sub>n</sub>i<sub>te-s</sub>i<sub>ze</sub> g<sup>a</sup>p <sup>can</sup> <sup>be</sup> <sup>com</sup>p<sup>uted</sup> <sup>exactl</sup>y<sup>.</sup>

Theorem 10.1 (Ensemble ceiling and averaging). Let $A _ { 1 } , \ldots , A _ { k } \in \mathbb { R } ^ { m \times d }$ be sketches applied to the same data.

(i) Converse. Stacking gives one linear map ofrank at most $p = \operatorname* { m i n } \{ k m , d \}$ . For any joint decoder, the recoverable centered-distance variance is at most $\alpha _ { p } ( \Sigma )$ . In the isotropic case, ρ<sub>HGR</sub> $\leq { \sqrt { p / d } }$ , with equality for independent Gaussian blocks and the optimal joint decoder.

(ii) Explicit averaging decoder. If the $A _ { i }$ are independent Gaussian maps and we average the k rescaled squared distances they produce, then the correlation, jointly over the maps and data, is

$$
{ \sqrt { \frac { k m } { k m + d + 2 } } } .
$$

The ratio of this correlation to the HGR ceiling is

$$
Y _ { k , m , d } = \sqrt { \frac { k m d } { ( k m + d + 2 ) \operatorname* { m i n } \{ k m , d \} } } = \left\{ \sqrt { d / ( k m + d + 2 ) } , \quad k m \leq d , \pm \right.
$$

For the avera<sub>g</sub>in<sub>g</sub> decoder, the <sub>p</sub>roof (section B) conditions on the standardized diference to obtain an <sub>est</sub>i<sub>mate o</sub>f <sub>t</sub>h<sub>e</sub> f<sub>orm</sub> $\| Z \| ^ { 2 } C ,$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $C \sim \chi _ { \mathit { k m } } ^ { 2 } / ( k m )$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>o</sub>f $Z .$ Th<sub>e</sub> <sub>resu</sub>l<sub>t</sub>i<sub>ng</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>exact e</sub>f<sub>ect o</sub>f <sub>an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent mu</sub>l<sub>t</sub>i<sub>p</sub>li<sub>cat</sub>i<sub>ve c</sub>hi<sub>-square</sub> fl<sub>uctuat</sub>i<sub>on,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng t</sub>h<sub>e</sub> fi<sub>n</sub>i<sub>te</sub> $d + 2$ <sub>correc</sub>ti<sub>on:</sub> avera<sub>g</sub>in<sub>g</sub> k sketches leaves the distance estimate intact and shrinks this fluctuation b<sub>y</sub> a factor $\sqrt { k }$ i<sub>n stan</sub>d<sub>ar</sub>d

d<sub>ev</sub>i<sub>at</sub>i<sub>on.</sub>

Inde<sub>p</sub>endent sketches add rank, but the<sub>y</sub> never exceed the rank-km ceilin<sub>g</sub>. The converse in <sub>p</sub>art (i) is <sub>t</sub>h<sub>e</sub> <sub>stac</sub>k<sub>e</sub>d<sub>-ran</sub>k <sub>spec</sub>i<sub>a</sub>l <sub>case</sub> <sub>o</sub>f <sub>t</sub>h<sub>eorem</sub> 6<sub>.</sub>1<sub>;</sub> <sub>t</sub>h<sub>e</sub> <sub>averag</sub>i<sub>ng</sub> id<sub>ent</sub>i<sub>ty</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e-stage</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> <sub>concatenate</sub>d $k m \times \bar { d }$ Gaussian ma<sub>p</sub> (section B). Sim<sub>p</sub>le avera<sub>g</sub>in<sub>g</sub> has the correct scalin<sub>g</sub> and a<sub>pp</sub>roaches this ceilin<sub>g</sub> when km $\ll d$ or $k m \gg d ;$ <sup>its</sup> <sup>lar</sup>g<sup>est</sup> <sup>constant-factor</sup> g<sup>a</sup>p <sup>occurs</sup> <sup>near</sup> $k m \approx d ,$ <sub>w</sub>h<sub>ere</sub> $\gamma _ { k , m , d } \approx 1 / \sqrt { 2 }$ Before km reaches d, added sketches expand the available rank; afterward, the<sub>y</sub> onl<sub>y</sub> reduce distortion in the averag<sup>i</sup>ng <sup>d</sup>eco<sup>d</sup>er. <sup>C</sup>orrect<sup>i</sup>ng t<sup>h</sup>e jo<sup>i</sup>nt metr<sup>i</sup>c w<sup>h</sup>en $k m \leq d ,$ <sub>or</sub> i<sub>nvert</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> <sub>stac</sub>k<sub>e</sub>d <sub>map</sub> <sub>w</sub>h<sub>en</sub> $k m \geq d ,$ <sub>removes</sub> <sub>t</sub>hi<sub>s</sub> <sub>revers</sub>ibl<sub>e</sub> <sub>spectra</sub>l di<sub>stort</sub>i<sub>on</sub> <sub>an</sub>d <sub>reac</sub>h<sub>es</sub> <sub>t</sub>h<sub>e</sub> i<sub>sotrop</sub>i<sub>c</sub> <sub>ce</sub>ili<sub>ng.</sub>

Th<sub>ese</sub> id<sub>ent</sub>i<sub>t</sub>i<sub>es</sub> <sub>are</sub> <sub>exact</sub> <sub>popu</sub>l<sub>at</sub>i<sub>on</sub> <sub>statements.</sub> W<sub>e</sub> <sub>ver</sub>if<sub>y</sub> <sub>representat</sub>i<sub>ve</sub> fi<sub>n</sub>i<sub>te-s</sub>i<sub>ze</sub> <sub>c</sub>l<sub>a</sub>i<sub>ms</sub> i<sub>n</sub> <sub>sect</sub>i<sub>on</sub> 11<sub>.</sub>

## 11. Numerical Validation

Th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> <sub>resu</sub>l<sub>ts</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>exact</sub> <sub>popu</sub>l<sub>at</sub>i<sub>on</sub> id<sub>ent</sub>i<sub>t</sub>i<sub>es,</sub> <sub>asymptot</sub>i<sub>c</sub> li<sub>m</sub>i<sub>ts,</sub> <sub>an</sub>d <sub>pro</sub>b<sub>a</sub>bili<sub>ty</sub> <sub>statements</sub> <sub>at</sub> finite d and m. We use numerical validation for two purposes: to check independent implementations of the closed-form expressions and to measure how closel<sub>y</sub> experiments at finite d and m follow their limitin<sub>g</sub> <sub>pre</sub>di<sub>ct</sub>i<sub>ons.</sub> W<sub>e</sub> <sub>use</sub> <sub>exact</sub> G<sub>auss</sub>i<sub>an</sub> <sub>moments</sub> f<sub>or</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l id<sub>ent</sub>i<sub>t</sub>i<sub>es,</sub> d<sub>eterm</sub>i<sub>n</sub>i<sub>st</sub>i<sub>c</sub> <sub>qua</sub>d<sub>rature</sub> f<sub>or</sub> i<sub>ntegra</sub>l formulas, and Monte Carlo samplin<sub>g</sub> for stochastic claims at finite d and m. We report one standard error <sup>f</sup>or <sup>M</sup>onte <sup>C</sup>ar<sup>l</sup>o est<sup>i</sup>mates an<sup>d Wil</sup>son 95<sup>% i</sup>nterva<sup>l</sup>s <sup>f</sup>or <sup>f</sup>requenc<sup>i</sup>es o<sup>f</sup>jo<sup>i</sup>nt events. <sup>Th</sup>ese ca<sup>l</sup>cu<sup>l</sup>at<sup>i</sup>ons <sub>support</sub> <sub>t</sub>h<sub>e</sub> f<sub>ormu</sub>l<sub>as</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub>i<sub>r</sub> i<sub>nterpretat</sub>i<sub>on</sub> <sub>at</sub> fi<sub>n</sub>i<sub>te</sub> di<sub>mens</sub>i<sub>ons;</sub> <sub>t</sub>h<sub>ey</sub> d<sub>o</sub> <sub>not</sub> <sub>rep</sub>l<sub>ace</sub> <sub>t</sub>h<sub>e</sub> <sub>proo</sub>f<sub>s.</sub>

Th<sub>e</sub> <sub>comp</sub>l<sub>ete</sub> <sub>repro</sub>d<sub>uc</sub>ibili<sub>ty</sub> <sub>mater</sub>i<sub>a</sub>l<sub>s</sub> <sub>are</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>at</sub>

## <sup>h</sup>ttps:<sup>//</sup>g<sup>i</sup>t<sup>h</sup>u<sup>b</sup>.com<sup>/</sup>p<sup>i</sup>yus<sup>h</sup>314<sup>/</sup>ran<sup>d</sup>om-project<sup>i</sup>on-geometry.

Th<sub>e</sub> <sub>repro</sub>d<sub>uc</sub>ibili<sub>ty</sub> <sub>gu</sub>id<sub>e</sub> i<sub>n</sub> <sub>sect</sub>i<sub>on</sub> E <sub>recor</sub>d<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>teste</sub>d <sub>vers</sub>i<sub>on</sub> <sub>an</sub>d <sub>exp</sub>l<sub>a</sub>i<sub>ns</sub> h<sub>ow</sub> <sub>to</sub> <sub>set</sub> <sub>up,</sub> <sub>execute,</sub> <sub>an</sub>d <sub>generate</sub> <sub>t</sub>h<sub>e</sub> <sub>art</sub>if<sub>acts.</sub>

<sup>Th</sup>e <sup>i</sup>n<sup>d</sup>epen<sup>d</sup>ent <sup>G</sup>auss<sup>i</sup>an rep<sup>l</sup>acement exper<sup>i</sup>ment tests w<sup>h</sup>et<sup>h</sup>er t<sup>h</sup>e J<sup>L b</sup>oun<sup>d di</sup>st<sup>i</sup>ngu<sup>i</sup>s<sup>h</sup>es a genu<sup>i</sup>ne s<sup>h</sup>are<sup>d</sup> project<sup>i</sup>on <sup>f</sup>rom a c<sup>l</sup>ou<sup>d</sup> t<sup>h</sup>at <sup>i</sup>s samp<sup>l</sup>e<sup>d</sup> <sup>i</sup>n<sup>d</sup>epen<sup>d</sup>ent<sup>l</sup>y o<sup>f</sup> t<sup>h</sup>e <sup>d</sup>ata. <sup>I</sup>t compares a norma<sup>li</sup>ze<sup>d</sup> <sup>H</sup>aar project<sup>i</sup>on, an <sup>i</sup>.<sup>i</sup>.<sup>d</sup>. <sup>G</sup>auss<sup>i</sup>an map, an<sup>d</sup> t<sup>h</sup>e <sup>i</sup>n<sup>d</sup>epen<sup>d</sup>ent <sup>G</sup>auss<sup>i</sup>an rep<sup>l</sup>acement map at t<sup>h</sup>e same $( n , d , m , \varepsilon )$ <sub>va</sub>l<sub>ues.</sub> F<sub>or</sub> $n = 1 0 0 , d = 8 1 9 2 , m = 1 0 2 4$ <sub>,</sub> <sub>an</sub>d $\varepsilon = 0 . 2 ,$ <sub>a</sub>ll 30 H<sub>aar</sub> <sub>tr</sub>i<sub>a</sub>l<sub>s</sub> <sub>an</sub>d <sub>a</sub>ll 30 G<sub>auss</sub>i<sub>an-map</sub> <sub>tr</sub>i<sub>a</sub>l<sub>s</sub> <sub>sat</sub>i<sub>s</sub>fi<sub>e</sub>d t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup>, as <sup>did</sup> 28 o<sup>f</sup> 30 rep<sup>l</sup>acement tr<sup>i</sup>a<sup>l</sup>s. $\mathrm { A t } m / d = 1 / 4$ <sub>,</sub> <sub>t</sub>h<sub>e</sub> P<sub>earson</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>o</sub>f <sub>centere</sub>d <sub>square</sub>d <sup>di</sup>stances <sup>i</sup>n <sup>Fi</sup>gure 1 was 0.55 <sup>f</sup>or t<sup>h</sup>e <sup>H</sup>aar project<sup>i</sup>on an<sup>d</sup> 0.01 <sup>f</sup>or t<sup>h</sup>e rep<sup>l</sup>acement map; <sup>b</sup>ot<sup>h</sup> <sup>di</sup>sp<sup>l</sup>aye<sup>d</sup> maps sat<sup>i</sup>s<sup>fi</sup>e<sup>d</sup> t<sup>h</sup>e same J<sup>L</sup> <sup>b</sup>oun<sup>d</sup>. <sup>Fi</sup>gure 1 reports t<sup>h</sup>e manuscr<sup>i</sup>pt v<sup>i</sup>ew, w<sup>hil</sup>e t<sup>h</sup>e repro<sup>d</sup>uc<sup>ibili</sup>ty repos<sup>i</sup>tory recor<sup>d</sup>s t<sup>h</sup>e <sup>f</sup>u<sup>ll</sup> ran<sup>ki</sup>ng, nearest-ne<sup>i</sup>g<sup>hb</sup>or, ca<sup>lib</sup>rat<sup>i</sup>on, an<sup>d</sup> <sup>di</sup>sjo<sup>i</sup>nt-pa<sup>i</sup>r <sup>di</sup>agnost<sup>i</sup>cs.

E<sub>very</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>at</sub>i<sub>on</sub> b<sub>ase</sub>d <sub>on</sub> <sub>exact</sub> <sub>moments</sub> <sub>or</sub> <sub>qua</sub>d<sub>rature</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 4 <sub>agrees</sub> <sub>w</sub>i<sub>t</sub>h i<sub>ts</sub> <sub>re</sub>f<sub>erence</sub> <sub>va</sub>l<sub>ue</sub> <sub>to</sub> <sub>t</sub>h<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>prec</sub>i<sub>s</sub>i<sub>on,</sub> <sub>an</sub>d <sub>every</sub> M<sub>onte</sub> C<sub>ar</sub>l<sub>o</sub> <sub>est</sub>i<sub>mate</sub> <sub>agrees</sub> <sub>w</sub>i<sub>t</sub>hi<sub>n</sub> <sub>two</sub> <sub>stan</sub>d<sub>ar</sub>d <sub>errors.</sub> T<sub>wo</sub> <sub>rows</sub> <sub>requ</sub>i<sub>re</sub> <sub>separate</sub> i<sub>nterpretat</sub>i<sub>on.</sub> Th<sub>e</sub> <sub>quarter-c</sub>i<sub>rc</sub>l<sub>e</sub> <sub>entry</sub> i<sub>s</sub> <sub>t</sub>h<sub>e</sub> li<sub>m</sub>i<sub>t</sub>i<sub>ng</sub> <sub>mean</sub> <sub>o</sub>f <sub>a</sub> <sub>poster</sub>i<sub>or</sub> <sub>var</sub>i<sub>ance</sub> f<sub>ract</sub>i<sub>on.</sub> Th<sub>e</sub> entr<sub>y</sub> <sup>f</sup>or $r _ { 2 }$ i<sub>s a</sub> d<sub>eterm</sub>i<sub>n</sub>i<sub>st</sub>i<sub>c equ</sub>i<sub>va</sub>l<sub>ent</sub> f<sub>orme</sub>d f<sub>rom a rat</sub>i<sub>o o</sub>f <sub>moments, not an</sub> id<sub>ent</sub>i<sub>ty</sub> f<sub>or a</sub> fi<sub>n</sub>i<sub>te samp</sub>l<sub>e.</sub>

T<sub>a</sub>bl<sub>e</sub> 5 <sub>va</sub>lid<sub>ates</sub> <sub>t</sub>h<sub>e</sub> <sub>resu</sub>l<sub>ts</sub> f<sub>or</sub> b<sub>a</sub>l<sub>ance</sub>d bl<sub>oc</sub>k<sub>s,</sub> <sub>t</sub>h<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>common</sub> <sub>m</sub>i<sub>n</sub>i<sub>mum,</sub> <sub>nearest</sub> <sub>ne</sub>i<sub>g</sub>hb<sub>ors,</sub> <sub>an</sub>d $\mathrm { { J L - K } }$ <sub>en</sub>d<sub>a</sub>ll <sub>coex</sub>i<sub>stence.</sub> Th<sub>e</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>at</sub>i<sub>on</sub> f<sub>or</sub> b<sub>a</sub>l<sub>ance</sub>d bl<sub>oc</sub>k<sub>s</sub> <sub>uses</sub> <sub>exact</sub> <sub>moments</sub> <sub>to</sub> <sub>c</sub>h<sub>ec</sub>k<sub>,</sub> <sub>w</sub>i<sub>t</sub>h<sub>out</sub> M<sub>onte</sub> C<sub>ar</sub>l<sub>o</sub> <sub>error,</sub> <sub>t</sub>h<sub>e</sub> d<sub>egree-two</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> <sub>canon</sub>i<sub>ca</sub>l <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>an</sub>d h<sub>ence</sub> <sub>on</sub> <sub>non</sub>li<sub>near</sub> <sub>excess.</sub> The nearest-nei<sub>g</sub>hbor row compares a simulation at finite d and m, 0.137020, with its Gaussian score li<sub>m</sub>i<sub>t,</sub> 0<sub>.</sub>138325<sub>, at</sub> $m = 2 0$ <sub>.</sub> H<sub>ere t</sub>h<sub>e</sub> $m ^ { - 1 / 2 }$ B<sub>err –</sub>E<sub>sseen rema</sub>i<sub>n</sub>d<sub>er can excee</sub>d <sub>t</sub>h<sub>e</sub> fi<sub>rst-or</sub>d<sub>er s</sub>i <sub>na</sub>l<sub>,</sub> so simulation provides the check at finite d and m. The final row tests the coexistence phenomenon of t<sup>h</sup>eorem 7.4. <sup>I</sup>ts jo<sup>i</sup>nt rate, 0.990, <sup>i</sup>s t<sup>h</sup>e emp<sup>i</sup>r<sup>i</sup>ca<sup>l</sup> <sup>f</sup>requency w<sup>i</sup>t<sup>h</sup> max<sup>i</sup>mum samp<sup>l</sup>e <sup>di</sup>stort<sup>i</sup>on at most 15<sup>%</sup> <sub>an</sub>d <sub>mean</sub> K<sub>en</sub>d<sub>a</sub>ll <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>magn</sub>i<sub>tu</sub>d<sub>e</sub> <sub>at</sub> <sub>most</sub> 0<sub>.</sub>1<sub>.</sub>

W<sub>e</sub> i<sub>nterpret</sub> <sub>two</sub> <sub>c</sub>h<sub>ec</sub>k<sub>s</sub> i<sub>n</sub> <sub>more</sub> d<sub>eta</sub>il b<sub>ecause</sub> <sub>t</sub>h<sub>ey</sub> <sub>revea</sub>l fi<sub>n</sub>i<sub>te-samp</sub>l<sub>e</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> <sub>t</sub>h<sub>at</sub> <sub>asymptot</sub>i<sub>c</sub> <sub>notat</sub>i<sub>on</sub> <sub>can</sub> <sub>o</sub>b<sub>scure.</sub> Fi<sub>rst,</sub> <sub>we</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>ate</sub> <sub>exact</sub> G<sub>auss</sub>i<sub>an</sub> <sub>moments</sub> <sub>at</sub> $\lambda = 2$ <sub>to repro</sub>d<sub>uce t</sub>h<sub>e qua</sub>d<sub>rat</sub>i<sub>c</sub> <sub>canon</sub>i<sub>ca</sub>l <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> i<sub>n t</sub>h<sub>eorem</sub> 6<sub>.</sub>2<sub>.</sub> W<sub>e t</sub>h<sub>en sweep t</sub>h<sub>e po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>canon</sub>i<sub>ca</sub>l <sub>corre</sub>l<sub>at</sub>i<sub>on o</sub>f $D _ { \lambda } = \dot { \lambda } G _ { 1 } ^ { 2 } + G _ { 2 } ^ { 2 }$ <sub>an</sub>d $U _ { \lambda } = \lambda G _ { 1 } ^ { 2 }$ <sup>to</sup> p<sup>roduce Fi</sup>g<sup>ure 3. De</sup>g<sup>ree one a</sup>g<sup>rees with</sup> $\sqrt { \alpha _ { 1 } }$ <sub>, w</sub>hil<sub>e neste</sub>d d<sub>egree-two t</sub>h<sub>roug</sub>h d<sub>egree-</sub>f<sub>our spaces y</sub>i<sub>e</sub>ld l<sub>arger</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s away</sub> f<sub>rom</sub> i<sub>sotropy.</sub> Th<sub>ese va</sub>l<sub>ues concern one o</sub>b<sub>servat</sub>i<sub>on</sub> <sub>an</sub>d fi<sub>n</sub>i<sub>te</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>c</sub>l<sub>asses;</sub> <sub>t</sub>h<sub>ey</sub> <sub>ne</sub>i<sub>t</sub>h<sub>er</sub> <sub>eva</sub>l<sub>uate</sub> <sub>unrestr</sub>i<sub>cte</sub>d HGR <sub>nor</sub> <sub>opt</sub>i<sub>m</sub>i<sub>ze</sub> <sub>t</sub>h<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>su</sub>b<sub>space.</sub>

Table 4. Numerical validation summary. Monte Carlo (MC) estimates are estimate ± one standard error; exact-moment and quadrature checks have no sampling error. Both isotropic mean-square error rows are normalized by Var(D).
<table><tr><td>Claim</td><td>Reference value Computed value</td><td></td><td>Method</td></tr><tr><td>Isotropic  $\mathrm { C o r r } ( D , \widetilde { D } )$ </td><td>0.2236</td><td> $0 . 2 2 2 8 \pm 0 . 0 0 1 5$ </td><td>MC, independent chi-square split, N = 400,000</td></tr><tr><td>Normalized minimum mean-square error</td><td>0.9500</td><td> $0 . 9 5 1 9 \pm 0 . 0 0 2 2$ </td><td>MC, conditional mean,  $N = 4 0 0 { , } 0 0 0$ </td></tr><tr><td>Centered estimation MSE / Var(D)</td><td>19.000</td><td> $1 9 . 1 0 0 \pm 0 . 0 5 3$ </td><td>MC, rescaled projection, N = 400,000</td></tr><tr><td>Quadratic canonical correlation at λ=2</td><td>0.939239</td><td>0.939239</td><td>exact Gaussian moments</td></tr><tr><td>Beta-arcsine p5,50 Diffuse traceless covariance-shape</td><td>0.59829</td><td> $0 . 5 9 9 0 7 \pm 0 . 0 0 0 7 7$ </td><td>MC, conditional Gaussian,  $N = 4 0 0 { , } 0 0 0$ </td></tr><tr><td>contraction (8/40)</td><td>0.04274</td><td> $0 . 0 4 2 9 7 \pm 0 . 0 0 0 1 8$ </td><td>MC, Haar frames, N = 3,000</td></tr><tr><td>One Gaussian stage (200 → 200)</td><td>0.70535</td><td> $0 . 7 0 4 4 2 \pm 0 . 0 0 0 8 0$ </td><td>MC, chi-square stage,  $N = 4 0 0 { , } 0 0 0$ </td></tr><tr><td>Gaussian chain  $( 3 0 0  1 5 0  1 0 0 )$ </td><td>0.4058</td><td> $0 . 4 0 5 0 \pm 0 . 0 0 1 3$ </td><td>MC, independent chi-square stages, N = 400,000</td></tr><tr><td>Wishart E tr(M²) Ratio-of-moments proxy vs.</td><td>243.00</td><td> $2 4 3 . 1 4 \pm 0 . 3 8$ </td><td>MC, Gaussian maps,  $N = 3 { , } 0 0 0$ </td></tr><tr><td>Monte Carlo mean of r2</td><td>14.8148</td><td> $1 4 . 8 4 1 3 \pm 0 . 0 0 7 0$ </td><td>MC; proxy is not an identity,  $N = 3 { , } 0 0 0$ </td></tr><tr><td>Quarter-circle mean posterior-variance fraction  $\scriptstyle ( \varepsilon = 0 . 2 )$ </td><td>0.180998</td><td>0.180998</td><td>deterministic quadrature</td></tr><tr><td>Ensemble average (km=100, d=400)</td><td>0.44632</td><td>0.44618 ± 0.00080 MC, stacked Gaussian map,</td><td> $N = 1 , 0 0 0 , 0 0 0$ </td></tr></table>

Table 5. Independent checks of the incorporated results. Monte Carlo uncertainty is one standard error; the JL–Kendall coexistence event has Wilson 95% interval [0.9710, 0.9966].
<table><tr><td>Claim and setting</td><td>Prediction</td><td>Computation</td></tr><tr><td>Balanced blocks,  $\theta = 0 . 2 5$ </td><td> $\rho _ { \mathrm { H G R } } = 0 . 5$ </td><td>degree-two CCA lower bound 0.500000000000</td></tr><tr><td> $p _ { 8 } \big ( \sqrt { 2 0 / 1 0 ^ { 4 } } \big )$ </td><td>integral in equation (8)</td><td>0.138324998</td></tr><tr><td> $\mathrm { k N N } , ( d , m , q ) = ( 1 0 ^ { 4 } , 2 0 , 8 )$ </td><td>Gaussian score limit 0.138325</td><td> $0 . 1 3 7 0 2 0 \pm 0 . 0 0 0 5 4 4$ </td></tr><tr><td>JL-Kendall coexistence,  $( n , d , m ) = ( 1 0 0 , 1 0 ^ { 6 } , 2 0 0 0 )$ </td><td> $\mathbb { E } \tau = 0 . 0 2 8 4 7 6$ </td><td> $0 . 0 2 8 9 9 6 \pm 0 . 0 0 1 2 5 4 ; \mathrm { j o i n t r a t e } \ 0 . 9 9 0$ </td></tr></table>

S<sub>econ</sub>d<sub>,</sub> <sub>t</sub>h<sub>e</sub> <sub>rows</sub> f<sub>or</sub> G<sub>auss</sub>i<sub>an</sub> <sub>stages</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 4 di<sub>st</sub>i<sub>ngu</sub>i<sub>s</sub>h <sub>an</sub> <sub>exact</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>average</sub>d <sub>over</sub> <sub>map</sub> <sub>an</sub>d d<sub>ata</sub> <sub>ran</sub>d<sub>omness</sub> f<sub>rom</sub> <sub>a</sub> d<sub>eterm</sub>i<sub>n</sub>i<sub>st</sub>i<sub>c</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ent</sub> f<sub>or</sub> <sub>a</sub> <sub>typ</sub>i<sub>ca</sub>l <sub>map.</sub> W<sub>e</sub> <sub>compare</sub> <sub>one-</sub> <sub>an</sub>d <sub>two-stage</sub> <sub>s</sub>i<sub>mu</sub>l<sub>at</sub>i<sub>ons</sub> <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>e</sub> fi<sub>n</sub>i<sub>te</sub> <sub>pro</sub>d<sub>ucts</sub> i<sub>n</sub> <sub>t</sub>h<sub>eorem</sub> A<sub>.</sub>1<sub>,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>t</sub>h<sub>e</sub>i<sub>r</sub> <sub>+</sub>2 <sub>correct</sub>i<sub>ons.</sub> A <sub>separate</sub> Wi<sub>s</sub>h<sub>art</sub> <sub>exper</sub>i<sub>ment</sub> <sub>c</sub>h<sub>ec</sub>k<sub>s</sub> E $\mathrm { t r } \big ( M ^ { 2 } \big ) = d \big ( m + d + 1 \big ) / m$ <sup>.</sup> <sup>B</sup>y <sup>contrast</sup>, $m d / { \left( m + d + 1 \right) }$ i<sub>s</sub> <sub>a</sub> <sub>rat</sub>i<sub>o</sub> <sub>o</sub>f <sub>trace</sub> <sub>moments,</sub> <sub>not</sub> E[r<sub>2</sub>]; the adjacent row records the finite-sam<sub>p</sub>le diference rather than treatin<sub>g</sub> the <sub>p</sub>rox<sub>y</sub> as an identit<sub>y</sub>.

H<sub>av</sub>i<sub>ng</sub> <sub>c</sub>h<sub>ec</sub>k<sub>e</sub>d <sub>t</sub>h<sub>e</sub> <sub>popu</sub>l<sub>at</sub>i<sub>on</sub> f<sub>ormu</sub>l<sub>as</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub>i<sub>r</sub> fi<sub>n</sub>i<sub>te-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>approx</sub>i<sub>mat</sub>i<sub>ons,</sub> <sub>we</sub> <sub>turn</sub> <sub>to</sub> <sub>t</sub>h<sub>e</sub>i<sub>r</sub> <sub>p</sub>ractica<sup>l</sup> im<sub>p</sub><sup>l</sup>ications.

## 12. Discussion: Implications for Practice

Th<sub>e</sub> di<sub>st</sub>i<sub>nct preservat</sub>i<sub>on</sub> l<sub>aws</sub> i<sub>mp</sub>l<sub>y</sub> dif<sub>erent responses</sub> i<sub>n pract</sub>i<sub>ce.</sub> R<sub>eran</sub>ki<sub>ng repa</sub>i<sub>rs a coarse ne</sub>i<sub>g</sub>hb<sub>or</sub> <sub>stage,</sub> <sub>tas</sub>k <sub>a</sub>li<sub>gnment</sub> <sub>c</sub>h<sub>anges</sub> <sub>w</sub>h<sub>at</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> <sub>matters,</sub> <sub>an</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>s</sub>k<sub>etc</sub>h<sub>es</sub> <sub>tra</sub>d<sub>e</sub> <sub>a</sub>dd<sub>e</sub>d <sub>ran</sub>k <sub>aga</sub>i<sub>nst</sub> di<sub>stort</sub>i<sub>on</sub> i<sub>n an unw</sub>hi<sub>tene</sub>d <sub>est</sub>i<sub>mate.</sub>

Candidate generation and reranking Loca<sup>l</sup>it<sub>y</sub>-sensitive <sup>h</sup>as<sup>h</sup>in<sub>g</sub> an<sup>d</sup> com<sub>p</sub>resse<sup>d</sup> in<sup>d</sup>exes can su<sub>pp</sub>ort a coarse candidate sta<sub>g</sub>e [8<sub>,</sub> 32]. Theorem 7.2 ex<sub>p</sub>lains wh<sub>y</sub> evaluatin<sub>g</sub> a short list at full <sub>p</sub>recision can still matter: at fixed q, the projected top choice approaches an independent uniform choice when m/d → 0. Thi<sub>s</sub> <sub>t</sub>h<sub>eorem</sub> d<sub>oes</sub> <sub>not</sub> <sub>prescr</sub>ib<sub>e</sub> <sub>an</sub> <sub>eng</sub>i<sub>neer</sub>i<sub>ng</sub> <sub>t</sub>h<sub>res</sub>h<sub>o</sub>ld b<sub>ecause</sub> <sub>rea</sub>l <sub>marg</sub>i<sub>ns,</sub> <sub>can</sub>did<sub>ate</sub> <sub>counts,</sub> <sub>an</sub>d d<sub>ata</sub> di<sub>str</sub>ib<sub>ut</sub>i<sub>ons</sub> dif<sub>er</sub> f<sub>rom</sub> <sub>t</sub>h<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>.</sub> I<sub>t</sub> d<sub>oes</sub> <sub>rep</sub>l<sub>ace</sub> <sub>a</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> h<sub>eur</sub>i<sub>st</sub>i<sub>c</sub> <sub>w</sub>i<sub>t</sub>h <sub>a</sub> <sub>comp</sub>l<sub>ete</sub> <sub>ran</sub>ki<sub>ng</sub> l<sub>aw</sub> f<sub>or</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>can</sub>did<sub>ates.</sub>

Objectives governed by the baseline rather than fluctuations Su<sup>b</sup>s<sub>p</sub>ace em<sup>b</sup>e<sup>dd</sup>in<sub>g</sub>s, s<sup>k</sup>etc<sup>h</sup>e<sup>d</sup> re<sub>g</sub>ression, <sub>an</sub>d <sub>ran</sub>d<sub>om</sub>i<sub>ze</sub>d l<sub>ow-ran</sub>k <sub>approx</sub>i<sub>mat</sub>i<sub>on</sub> <sub>target</sub> <sub>res</sub>id<sub>ua</sub>l <sub>norms</sub> <sub>or</sub> d<sub>om</sub>i<sub>nant</sub> <sub>su</sub>b<sub>spaces</sub> <sub>rat</sub>h<sub>er</sub> <sub>t</sub>h<sub>an</sub> <sub>t</sub>h<sub>e</sub> orderin<sub>g</sub> of nearl<sub>y</sub> e<sub>q</sub>ual within-cloud distances [7]. Randomized sin<sub>g</sub>ular value decom<sub>p</sub>osition is desi<sub>g</sub>ned <sub>to</sub> <sub>reta</sub>i<sub>n</sub> <sub>spectra</sub>l <sub>sp</sub>ik<sub>es,</sub> <sub>not</sub> dif<sub>use</sub> <sub>an</sub>i<sub>sotropy.</sub> I<sub>ts</sub> <sub>success</sub> i<sub>s</sub> <sub>t</sub>h<sub>ere</sub>f<sub>ore</sub> <sub>compat</sub>ibl<sub>e</sub> <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>e</sub> <sub>recovery</sub> li<sub>m</sub>i<sub>ts</sub> d<sub>er</sub>i<sub>ve</sub>d h<sub>ere.</sub>

Coding and task alignment Pro<sup>d</sup>uct <sub>q</sub>uantization c<sup>h</sup>an<sub>g</sub>es t<sup>h</sup>e resource <sup>f</sup>rom out<sub>p</sub>ut <sup>d</sup>imension to a <sup>fi</sup>nite bit rate [33]. For a Gaussian source, reverse water-filling assigns rate only to covariance modes above a distortion threshold and allocates more rate to lar<sub>g</sub>er variances [23]. Learned rotations can then redistribute distortion across coordinates [34]. Supervised nested-representation training, often called Matryoshka training, takes a diferent route. It trains each leading coordinate block to remain predictive, so label-relevant information a<sub>pp</sub>ears earl<sub>y</sub> in the re<sub>p</sub>resentation [35]. Both a<sub>pp</sub>roaches add assum<sub>p</sub>tions absent from an o<sup>bl</sup>ivious ran<sup>k</sup>-m projection. T<sup>h</sup>e re<sup>l</sup>evant resource is tas<sup>k</sup> in<sup>f</sup>ormation, not tota<sup>l</sup> wit<sup>h</sup>in-c<sup>l</sup>ou<sup>d d</sup>istance <sup>recover</sup>y<sup>.</sup>

Cannings–Samworth ensembles Cannin<sub>g</sub>s an<sup>d</sup> Samwort<sup>h</sup> [36] a<sub>gg</sub>re<sub>g</sub>ate c<sup>l</sup>assi<sup>fi</sup>ers traine<sup>d</sup> on mu<sup>l</sup>ti<sub>p</sub><sup>l</sup>e ran<sup>d</sup>om project<sup>i</sup>ons. <sup>O</sup>ur popu<sup>l</sup>at<sup>i</sup>on ca<sup>l</sup>cu<sup>l</sup>at<sup>i</sup>on <sup>i</sup>so<sup>l</sup>ates t<sup>h</sup>e geometr<sup>i</sup>c <sup>b</sup>ene<sup>fi</sup>t o<sup>f</sup> t<sup>h</sup>e para<sup>ll</sup>e<sup>l</sup> s<sup>k</sup>etc<sup>h</sup>es: <sup>b</sup>e<sup>f</sup>ore $k m = d ,$ <sub>more</sub> <sub>s</sub>k<sub>etc</sub>h<sub>es</sub> <sub>ra</sub>i<sub>se</sub> b<sub>ot</sub>h <sub>ran</sub>k <sub>an</sub>d <sub>averag</sub>i<sub>ng</sub> <sub>accuracy;</sub> <sub>a</sub>f<sub>terwar</sub>d<sub>,</sub> <sub>t</sub>h<sub>ey</sub> i<sub>mprove</sub> <sub>on</sub>l<sub>y</sub> <sub>t</sub>h<sub>e</sub> <sub>c</sub>h<sub>osen</sub> estimate (theorem 10.1). Related cluster ensembles use the same <sub>p</sub>arallel strate<sub>gy</sub> [37]. Task <sub>g</sub>ains still de<sub>p</sub>end <sub>on</sub> <sub>marg</sub>i<sub>ns</sub> <sub>an</sub>d d<sub>ownstream</sub> l<sub>earn</sub>i<sub>ng.</sub>

Limits of transfer T<sup>h</sup>ese <sub>p</sub>ractica<sup>l</sup> res<sub>p</sub>onses re<sup>l</sup><sub>y</sub> on exact Gaussian <sup>l</sup>aws w<sup>h</sup>ose trans<sup>f</sup>er <sup>b</sup>e<sub>y</sub>on<sup>d</sup> t<sup>h</sup>e b<sub>enc</sub>h<sub>mar</sub>k <sub>requ</sub>i<sub>res</sub> <sub>care.</sub> G<sub>auss</sub>i<sub>an</sub>i<sub>ty</sub> <sub>ma</sub>k<sub>es</sub> <sub>t</sub>h<sub>e</sub> <sub>c</sub>hi<sub>-square</sub> <sub>c</sub>h<sub>anne</sub>l <sub>an</sub>d <sub>score</sub> <sub>covar</sub>i<sub>ance</sub> <sub>exact.</sub> Th<sub>e</sub> <sub>re</sub>l<sub>at</sub>i<sub>ve-error cr</sub>i<sub>ter</sub>i<sub>on</sub> i<sub>n propos</sub>i<sub>t</sub>i<sub>on</sub> 2<sub>.</sub>1 i<sub>s</sub> di<sub>str</sub>ib<sub>ut</sub>i<sub>on-</sub>f<sub>ree, an</sub>d <sub>t</sub>h<sub>e proo</sub>f <sub>mec</sub>h<sub>an</sub>i<sub>sms suggest extens</sub>i<sub>ons</sub> <sub>un</sub>d<sub>er</sub> i<sub>nvar</sub>i<sub>ance</sub> <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>es</sub> <sub>or</sub> <sub>e</sub>lli<sub>pt</sub>i<sub>ca</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> b<sub>ut</sub> <sub>t</sub>h<sub>e</sub> <sub>constants</sub> <sub>nee</sub>d <sub>not</sub> <sub>surv</sub>i<sub>ve.</sub> Ali<sub>gne</sub>d <sub>an</sub>i<sub>sotropy</sub> i<sub>s</sub> now covered by theorem 6.3; noncommuting maps remain open. Fixed-q neighbor overlap is solved for <sup>i</sup>sotrop<sup>i</sup>c <sup>G</sup>auss<sup>i</sup>an <sup>d</sup>ata an<sup>d</sup> an <sup>i</sup>n<sup>d</sup>epen<sup>d</sup>ent ort<sup>h</sup>ogona<sup>l</sup> project<sup>i</sup>on, w<sup>hil</sup>e grow<sup>i</sup>ng ne<sup>i</sup>g<sup>hb</sup>or<sup>h</sup>oo<sup>d</sup>s, <sup>h</sup>u<sup>b</sup>ness, d<sub>ens</sub>i<sub>ty</sub> <sub>est</sub>i<sub>mates,</sub> <sub>an</sub>d <sub>non-</sub>G<sub>auss</sub>i<sub>an</sub> <sub>un</sub>i<sub>versa</sub>li<sub>ty</sub> <sub>requ</sub>i<sub>re</sub> <sub>a</sub> <sub>t</sub>h<sub>eory</sub> f<sub>or</sub> di<sub>stances</sub> <sub>t</sub>h<sub>at</sub> <sub>s</sub>h<sub>are</sub> <sub>a</sub> <sub>query.</sub>

## 13. Research Directions

Th<sub>e</sub> <sub>operator</sub> <sub>too</sub>lki<sub>t</sub> d<sub>eve</sub>l<sub>ope</sub>d h<sub>ere—con</sub>di<sub>t</sub>i<sub>ona</sub>l<sub>-expectat</sub>i<sub>on</sub> <sub>c</sub>h<sub>anne</sub>l<sub>s,</sub> <sub>s</sub>i<sub>ngu</sub>l<sub>ar</sub> <sub>systems,</sub> <sub>an</sub>d <sub>s</sub>h<sub>arp</sub> G<sub>auss</sub>i<sub>an</sub> li<sub>m</sub>i<sub>ts—exten</sub>d<sub>s to sett</sub>i<sub>ngs t</sub>h<sub>e sca</sub>l<sub>ar</sub> i<sub>sotrop</sub>i<sub>c c</sub>h<sub>anne</sub>l d<sub>oes not</sub> di<sub>rect</sub>l<sub>y reac</sub>h<sub>.</sub> Thi<sub>s sect</sub>i<sub>on</sub> f<sub>ormu</sub>l<sub>ates t</sub>h<sub>e</sub> resu<sup>l</sup>t<sup>i</sup>ng c<sup>h</sup>a<sup>ll</sup>enges as prec<sup>i</sup>se pro<sup>bl</sup>ems, eac<sup>h</sup> anc<sup>h</sup>ore<sup>d</sup> to a t<sup>h</sup>eorem or conjecture t<sup>h</sup>e <sup>f</sup>ramewor<sup>k</sup> a<sup>l</sup>rea<sup>d</sup>y <sub>p</sub>ro<sup>d</sup>uces.

Research Direction 13.1 (General anisotro ic maximal correlation). For $D = \textstyle \sum _ { i } \lambda _ { i } G _ { i } ^ { 2 }$ <sub>o</sub>b<sub>serve</sub>d through a rank-m map, determine $\rho _ { \mathrm { H G R } } ( D ; \mathcal { O } )$ <sub>an</sub>d <sub>t</sub>h<sub>e opt</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng o</sub>b<sub>servat</sub>i<sub>on.</sub> W<sub>e so</sub>l<sub>ve t</sub>h<sub>e pro</sub>bl<sub>em</sub> <sup>f</sup>or commut<sup>i</sup>ng projectors w<sup>i</sup>t<sup>h b</sup>a<sup>l</sup>ance<sup>d</sup> spectra<sup>l</sup> retent<sup>i</sup>on <sup>i</sup>n t<sup>h</sup>eorem 6.3, w<sup>hi</sup>c<sup>h</sup> a<sup>l</sup>so <sup>b</sup>oun<sup>d</sup>s every <sub>commut</sub>i<sub>ng</sub> bl<sub>oc</sub>k <sub>a</sub>ll<sub>ocat</sub>i<sub>on.</sub> Th<sub>eorem</sub> 6<sub>.</sub>2 <sub>s</sub>h<sub>ows t</sub>h<sub>at an un</sub>b<sub>a</sub>l<sub>ance</sub>d <sub>a</sub>ll<sub>ocat</sub>i<sub>on can excee</sub>d $\sqrt { \alpha _ { m } } .$ F<sub>or</sub> a noncommutin<sub>g</sub> ma<sub>p</sub>, t<sup>h</sup>e o<sup>b</sup>serve<sup>d</sup> <sub>q</sub>ua<sup>d</sup>ratic <sup>f</sup>orm no <sup>l</sup>on<sub>g</sub>er s<sub>p</sub><sup>l</sup>its a<sup>l</sup>on<sub>g</sub> covariance ei<sub>g</sub>ens<sub>p</sub>aces, <sub>prevent</sub>i<sub>ng</sub> di<sub>rect use o</sub>f <sub>t</sub>h<sub>e</sub> bl<sub>oc</sub>k<sub>w</sub>i<sub>se</sub> b<sub>eta–gamma re</sub>d<sub>uct</sub>i<sub>on or</sub> i<sub>ts tensor</sub>i<sub>zat</sub>i<sub>on.</sub> Th<sub>e unrestr</sub>i<sub>cte</sub>d <sub>pro</sub>bl<sub>em</sub> <sub>coup</sub>l<sub>es</sub> <sub>a</sub> <sub>we</sub>i<sub>g</sub>h<sub>te</sub>d<sub>-gamma</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l<sub>-expectat</sub>i<sub>on</sub> <sub>operator</sub> <sub>w</sub>i<sub>t</sub>h <sub>an</sub> <sub>opt</sub>i<sub>m</sub>i<sub>zat</sub>i<sub>on</sub> <sub>over</sub> <sub>row</sub> <sup>s</sup>p<sup>aces.</sup>

Research Direction 13.2 (Uniform concomitant asymptotics and growing top-k sets). For fixed $\rho \in \left( 0 , 1 \right)$ <sub>,</sub> G<sub>auss</sub>i<sub>an p</sub>l<sub>ura</sub>li<sub>ty sta</sub>bili<sub>ty measures t</sub>h<sub>e pro</sub>b<sub>a</sub>bili<sub>ty t</sub>h<sub>at two corre</sub>l<sub>ate</sub>d G<sub>auss</sub>i<sub>an score</sub> <sub>vectors</sub> <sub>se</sub>l<sub>ect</sub> <sub>t</sub>h<sub>e</sub> <sub>same</sub> <sub>extrema</sub>l <sub>coor</sub>di<sub>nate.</sub> Th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> li<sub>terature</sub> <sub>g</sub>i<sub>ves</sub> <sub>t</sub>h<sub>e</sub> <sub>c</sub>l<sub>ass</sub>i<sub>ca</sub>l <sub>asymptot</sub>i<sub>c</sub> [38<sub>,</sub> 39]

$$
p _ { q } ( \boldsymbol { \mathsf { \rho } } ) \sim \frac { \Gamma \left( 1 / ( 1 + \boldsymbol { \mathsf { \rho } } ) \right) ^ { 2 } } { \sqrt { 1 - \boldsymbol { \mathsf { \rho } } ^ { 2 } } } ( q - 1 ) ^ { - ( 1 - \boldsymbol { \mathsf { \rho } } ) / ( 1 + \boldsymbol { \mathsf { \rho } } ) } \left( 4 \pi \log ( q - 1 ) \right) ^ { - \boldsymbol { \mathsf { \rho } } / ( 1 + \boldsymbol { \mathsf { \rho } } ) } .
$$

When the correlation varies with q, existing uniform Gaussian-tail control and Borell’s bound imply [38]

$$
\operatorname * { l i m } _ { q \to \infty } \operatorname * { s u p } _ { q p _ { q } ( \mathsf { p } _ { q } ) } \leq e ^ { 2 \lambda } \qquad \mathrm { w h e n } \qquad \mathsf { \ p } _ { q } \log q \to \lambda \in [ 0 , \infty ) .
$$

A <sub>matc</sub>hi<sub>ng</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>,</sub> id<sub>ea</sub>ll<sub>y w</sub>i<sub>t</sub>h <sub>a un</sub>if<sub>orm error est</sub>i<sub>mate, rema</sub>i<sub>ns open</sub> i<sub>n t</sub>hi<sub>s tr</sub>i<sub>angu</sub>l<sub>ar</sub> re<sub>g</sub>ime, w<sup>h</sup>ere $q \to \infty$ <sub>an</sub>d $\rho _ { q } \log q  \lambda$ <sub>.</sub> Th<sub>e case</sub> $\lambda = 0$ <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom t</sub>h<sub>e upper</sub> b<sub>oun</sub>d <sub>an</sub>d $p _ { q } ( \boldsymbol { \rho } ) \geq 1 / q$ f<sub>or</sub> $\rho \geq 0$ <sub>.</sub> M<sub>ore genera</sub>ll<sub>y,</sub> d<sub>eterm</sub>i<sub>ne over</sub>l<sub>ap w</sub>h<sub>en</sub> $q , k ,$ <sub>an</sub>d $\rho ^ { - 1 }$ grow jo<sup>i</sup>nt<sup>l</sup>y. <sup>Th</sup>e <sub>g</sub>oal is to establish a sharp trian<sub>g</sub>ular-re<sub>g</sub>ime and top-k extension of the Gaussian stabilit<sub>y</sub> kernel. <sup>D</sup>eterm<sup>i</sup>n<sup>i</sup>st<sup>i</sup>c or<sup>di</sup>na<sup>l</sup> capac<sup>i</sup>ty an<sup>d</sup> t<sup>h</sup>e correspon<sup>di</sup>ng <sup>G</sup>auss<sup>i</sup>an-c<sup>l</sup>ou<sup>d</sup> conjecture are recor<sup>d</sup>e<sup>d</sup> <sup>i</sup>n t<sup>h</sup>eorem <sup>D</sup>.1 an<sup>d</sup> conjecture <sup>D</sup>.2.

Conjecture 13.3 (Thin-compression $( m / d  0 )$ spike phase diagram). Consider m/d → 0 with centered covariance variation spread across many eigendirections and with no dominant eigenvalue. First, identify conditions under which the normalized compressed traceless bulk has a universal limiting spectrum. Second, determine when a finite-rank population spike becomes distinguishable from that bulk. A semicircle transition is plausible only under additional assumptions. Under free compression—the free-probability model for compression by a generic random projection—the compressed spectral law still depends on the population law. For example, compressing $\sigma ^ { 2 } I + \lambda \dot { \nu } \dot { \bar { \nu } }$ produces a scalar matrix plus rank one, with no population-level random bulk. A detection threshold must therefore depend jointly on projection dimension, sample size, and covariance, as in deformed random-matrix models, which study random spectral bulks perturbed by low-rank structure [40].

Research Direction 13.4 (Growing-neighborhood statistics). Determine the leading components of d<sub>ens</sub>i<sub>ty,</sub> h<sub>u</sub>b<sub>ness-magn</sub>i<sub>tu</sub>d<sub>e,</sub> <sub>an</sub>d i<sub>ntr</sub>i<sub>ns</sub>i<sub>c-</sub>di<sub>mens</sub>i<sub>on</sub> <sub>est</sub>i<sub>mators</sub> <sub>w</sub>h<sub>en</sub> <sub>t</sub>h<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f di<sub>stances</sub> <sub>s</sub>h<sub>ar</sub>i<sub>ng</sub> a query grows. Here hubness is the tendency of a few points to be nearest neighbors of many queries [41]<sub>;</sub> hub ma<sub>g</sub>nitude counts how often this occurs<sub>,</sub> whereas hub identit<sub>y</sub> records which <sub>p</sub>oints become hubs. Althou<sub>g</sub>h theorem 7.2 resolves fixed-q top-k overlap, the sin<sub>g</sub>le-distance La<sub>g</sub>uerre decomposition i<sub>n</sub> <sub>t</sub>h<sub>eorem</sub> 5<sub>.</sub>2 d<sub>oes</sub> <sub>not</sub> <sub>tensor</sub>i<sub>ze</sub> <sub>across</sub> <sub>a</sub> <sub>s</sub>h<sub>are</sub>d <sub>query.</sub> A <sub>comp</sub>l<sub>ete</sub> <sub>t</sub>h<sub>eory</sub> <sub>o</sub>f <sub>t</sub>h<sub>ese</sub> <sub>stat</sub>i<sub>st</sub>i<sub>cs</sub> <sub>must</sub> <sub>t</sub>h<sub>ere</sub>f<sub>ore</sub> <sub>cover</sub> <sub>non-</sub>G<sub>auss</sub>i<sub>an</sub> <sub>score</sub> li<sub>m</sub>i<sub>ts</sub> <sub>an</sub>d di<sub>st</sub>i<sub>ngu</sub>i<sub>s</sub>h <sub>c</sub>h<sub>anges</sub> i<sub>n</sub> h<sub>u</sub>b <sub>magn</sub>i<sub>tu</sub>d<sub>e</sub> f<sub>rom</sub> <sub>c</sub>h<sub>anges</sub> i<sub>n</sub> h<sub>u</sub>b id<sub>ent</sub>i<sub>ty.</sub>

Research Direction 13.5 (Supervised task-recovery limit). Define $\alpha _ { m } ^ { \mathrm { t a s k } }$ <sub>as t</sub>h<sub>e</sub> Fi<sub>s</sub>h<sub>er</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>on</sub> about a task arameter retained under the best rank-m linear ma . We seek to determine when su<sub>p</sub>ervise<sup>d</sup> trainin<sub>g</sub> can ma<sup>k</sup>e $\alpha _ { m } ^ { \mathrm { t a s k } }$ close to one at small m while total-covariance recovery remains sma<sup>ll</sup>. <sup>Thi</sup>s <sup>i</sup>nc<sup>l</sup>u<sup>d</sup>es o<sup>b</sup>ject<sup>i</sup>ves t<sup>h</sup>at tra<sup>i</sup>n neste<sup>d</sup> pre<sup>fi</sup>xes o<sup>f</sup> a representat<sup>i</sup>on to rema<sup>i</sup>n pre<sup>di</sup>ct<sup>i</sup>ve, suc<sup>h</sup> as Matr<sub>y</sub>oshka-st<sub>y</sub>le trainin<sub>g</sub> [35]. Characterize the resultin<sub>g</sub> trade-of with nuisance <sub>g</sub>eometr<sub>y</sub>.

## 14. Conclusion

<sup>R</sup>an<sup>d</sup>om project<sup>i</sup>ons <sup>d</sup>o not preserve a s<sup>i</sup>ng<sup>l</sup>e, un<sup>dif</sup>erent<sup>i</sup>ate<sup>d</sup> not<sup>i</sup>on o<sup>f</sup> geometry. <sup>O</sup>n t<sup>h</sup>e <sup>G</sup>auss<sup>i</sup>an benchmark, the retention laws reviewed in Table 1 share one parameter, the rank ratio m/d, <sub>y</sub>et act on dif<sub>erent</sub> <sub>sca</sub>l<sub>es:</sub> <sub>a</sub> <sub>var</sub>i<sub>ance</sub> f<sub>ract</sub>i<sub>on</sub> f<sub>or</sub> di<sub>stance</sub> f<sub>eatures,</sub> <sub>a</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> f<sub>or</sub> <sub>t</sub>h<sub>e</sub>i<sub>r</sub> <sub>s</sub>i<sub>gns,</sub> <sub>an</sub>d <sub>a</sub> <sub>square</sub>d <sub>rat</sub>i<sub>o</sub> f<sub>or</sub> dif<sub>use</sub> <sub>s</sub>h<sub>ape.</sub> U<sub>n</sub>d<sub>er</sub> <sub>genera</sub>l <sub>covar</sub>i<sub>ance,</sub> $\alpha _ { m } ( \Sigma )$ replaces m/d for the distance value, and si<sub>g</sub>n statistics inherit <sub>t</sub>h<sub>e</sub> <sub>assoc</sub>i<sub>ate</sub>d <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>sca</sub>l<sub>e,</sub> <sub>so</sub> <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d<sub>s</sub> <sub>w</sub>i<sub>t</sub>h <sub>a</sub> fi<sub>xe</sub>d <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>can</sub>did<sub>ates</sub> <sub>approac</sub>h <sub>c</sub>h<sub>ance</sub> <sub>as</sub> m/d → 0.

Th<sub>e</sub> dif<sub>erences</sub> <sub>among</sub> <sub>t</sub>h<sub>ese</sub> <sub>sca</sub>l<sub>es</sub> h<sub>ave</sub> di<sub>rect</sub> <sub>consequences.</sub> B<sub>ecause</sub> <sub>s</sub>h<sub>ape</sub> <sub>contracts</sub> f<sub>aster</sub> <sub>t</sub>h<sub>an</sub> <sub>mean</sub> <sub>an</sub>d <sub>sca</sub>l<sub>e,</sub> <sub>compress</sub>i<sub>on</sub> <sub>sp</sub>h<sub>er</sub>i<sub>ca</sub>li<sub>zes</sub> di<sub>str</sub>ib<sub>ute</sub>d <sub>an</sub>i<sub>sotropy</sub> <sub>w</sub>hil<sub>e</sub> <sub>concentrate</sub>d <sub>sp</sub>ik<sub>es</sub> <sub>can</sub> <sub>rema</sub>i<sub>n</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e.</sub> <sup>B</sup>ecause ran<sup>ki</sup>ngs <sup>d</sup>epen<sup>d</sup> on t<sup>h</sup>e corre<sup>l</sup>at<sup>i</sup>on sca<sup>l</sup>e, a project<sup>i</sup>on can sat<sup>i</sup>s<sup>f</sup>y t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> on a <sup>fi</sup>n<sup>i</sup>te samp<sup>l</sup>e <sub>w</sub>hil<sub>e</sub> i<sub>ts</sub> <sub>mean</sub> <sub>ran</sub>ki<sub>ng</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>van</sub>i<sub>s</sub>h<sub>es—an</sub>d <sub>t</sub>h<sub>e</sub> <sub>rep</sub>l<sub>acement-map</sub> <sub>construct</sub>i<sub>on</sub> <sub>s</sub>h<sub>ows</sub> <sub>t</sub>h<sub>e</sub> b<sub>oun</sub>d i<sub>s</sub> compat<sup>ibl</sup>e w<sup>i</sup>t<sup>h</sup> output ent<sup>i</sup>re<sup>l</sup>y <sup>i</sup>n<sup>d</sup>epen<sup>d</sup>ent o<sup>f</sup> t<sup>h</sup>e <sup>d</sup>ata. <sup>Th</sup>e J<sup>L b</sup>oun<sup>d</sup> a<sup>l</sup>one t<sup>h</sup>ere<sup>f</sup>ore <sup>d</sup>oes not <sup>f</sup>orce <sup>d</sup>ata-<sup>d</sup>e<sub>p</sub>en<sup>d</sup>ent <sub>g</sub>eometr<sub>y</sub>.

F<sub>or</sub> <sub>a</sub> <sub>genera</sub>l k<sub>nown</sub> <sub>map,</sub> <sub>ran</sub>k <sub>governs</sub> <sub>opt</sub>i<sub>ma</sub>l d<sub>eco</sub>di<sub>ng,</sub> <sub>spectra</sub>l <sub>sprea</sub>d <sub>governs</sub> <sub>an</sub> <sub>unw</sub>hi<sub>tene</sub>d <sub>square</sub>d<sub>-norm est</sub>i<sub>mate, an</sub>d <sub>t</sub>h<sub>e sma</sub>ll<sub>est s</sub>i<sub>ngu</sub>l<sub>ar va</sub>l<sub>ues govern no</sub>i<sub>sy</sub> i<sub>nvers</sub>i<sub>on.</sub> M<sub>u</sub>l<sub>t</sub>i<sub>p</sub>l<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>s</sub>k<sub>etc</sub>h<sub>es</sub> <sub>can</sub> <sub>ra</sub>i<sub>se</sub> <sub>t</sub>h<sub>e</sub> <sub>com</sub>bi<sub>ne</sub>d <sub>ran</sub>k<sub>,</sub> b<sub>ut</sub> <sub>on</sub>l<sub>y</sub> <sub>up</sub> <sub>to</sub> <sub>t</sub>h<sub>e</sub> <sub>am</sub>bi<sub>ent</sub> di<sub>mens</sub>i<sub>on.</sub> Th<sub>ese</sub> <sub>exact</sub> G<sub>auss</sub>i<sub>an</sub> l<sub>aws</sub> <sub>s</sub>h<sub>ow</sub> <sub>w</sub>h<sub>ere</sub> <sub>reran</sub>ki<sub>ng,</sub> <sub>metr</sub>i<sub>c</sub> <sub>correct</sub>i<sub>on,</sub> <sub>or</sub> <sub>a</sub>ddi<sub>t</sub>i<sub>ona</sub>l <sub>s</sub>k<sub>etc</sub>h<sub>es</sub> <sub>are</sub> <sub>necessary.</sub> Th<sub>e</sub> <sub>same</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l<sub>-expectat</sub>i<sub>on</sub> <sub>too</sub>lki<sub>t exten</sub>d<sub>s</sub> b<sub>eyon</sub>d <sub>t</sub>h<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>; sect</sub>i<sub>on</sub> 13 d<sub>eve</sub>l<sub>ops t</sub>h<sub>e resu</sub>l<sub>t</sub>i<sub>ng researc</sub>h di<sub>rect</sub>i<sub>ons,</sub> f<sub>rom</sub> <sub>an</sub>i<sub>sotrop</sub>i<sub>c max</sub>i<sub>ma</sub>l <sub>corre</sub>l<sub>at</sub>i<sub>on an</sub>d <sub>grow</sub>i<sub>ng ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d<sub>s to compresse</sub>d <sub>covar</sub>i<sub>ance p</sub>h<sub>ase</sub> di<sub>agrams an</sub>d su<sub>p</sub>ervise<sup>d</sup> tas<sup>k</sup> recover<sub>y</sub>.

## Acknowledgments

Thi<sub>s</sub> <sub>wor</sub>k <sub>was</sub> <sub>supporte</sub>d b<sub>y</sub> <sub>t</sub>h<sub>e</sub> A<sub>pp</sub>li<sub>e</sub>d M<sub>at</sub>h<sub>emat</sub>i<sub>cs</sub> <sub>program</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> U<sub>.</sub>S<sub>.</sub> D<sub>epartment</sub> <sub>o</sub>f E<sub>nergy,</sub> Of<sub>-</sub> fice of Science, Ofice of Advanced Scientific Com<sub>p</sub>utin<sub>g</sub> Research (ASCR), throu<sub>g</sub>h the SPARSITUTE Mathematical Multifaceted Integrated Capability Center (MMICC), A Mathematical Institute for Sparse Computations in Science and Engineering (sparsitute.lbl.gov). We thank David Rabson, ASCR program manager; R<sub>ama</sub>k<sub>r</sub>i<sub>s</sub>h<sub>nan</sub> K<sub>annan,</sub> <sub>act</sub>i<sub>ng</sub> di<sub>rector</sub> <sub>o</sub>f <sub>t</sub>h<sub>e</sub> SPARSITUTE MMICC<sub>;</sub> <sub>an</sub>d Mi<sub>c</sub>h<sub>ae</sub>l L<sub>.</sub> P<sub>ar</sub>k<sub>s,</sub> <sub>t</sub>h<sub>e</sub> O<sub>a</sub>k Rid<sub>ge</sub> N<sub>at</sub>i<sub>ona</sub>l L<sub>a</sub>b<sub>oratory</sub> <sub>po</sub>i<sub>nt</sub> <sub>o</sub>f <sub>contact</sub> f<sub>or</sub> ASCR A<sub>pp</sub>li<sub>e</sub>d M<sub>at</sub>h<sub>emat</sub>i<sub>cs,</sub> f<sub>or</sub> <sub>t</sub>h<sub>e</sub>i<sub>r</sub> l<sub>ea</sub>d<sub>ers</sub>hi<sub>p</sub> <sub>an</sub>d <sub>support.</sub>

## Appendix A. Gaussian Chains and Hard-Edge Precision

Hanin and Nica [42] showed that the harmonic sum $\begin{array} { r } { H _ { L } = \sum _ { \ell = 1 } ^ { L } n _ { \ell } ^ { - 1 } } \end{array}$ <sub>o</sub>f i<sub>nverse</sub> l<sub>ayer</sub> <sub>w</sub>id<sub>t</sub>h<sub>s</sub> <sub>governs</sub> <sub>norm</sub> fl<sub>uctuat</sub>i<sub>ons</sub> i<sub>n</sub> l<sub>ong</sub> <sub>ran</sub>d<sub>om-matr</sub>i<sub>x</sub> <sub>pro</sub>d<sub>ucts</sub> <sub>app</sub>li<sub>e</sub>d <sub>to</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>vector.</sub> F<sub>or</sub> <sub>a</sub> <sub>ran</sub>d<sub>om</sub> G<sub>auss</sub>i<sub>an</sub> i<sub>nput,</sub> <sub>we</sub> <sub>a</sub>dd <sub>t</sub>h<sub>e</sub> i<sub>nput-</sub>fl<sub>uctuat</sub>i<sub>on term</sub> $n _ { 0 } ^ { - 1 } = d ^ { - 1 }$ . The s<sub>p</sub>ectrum law in theorem 9.1(ii) then <sub>g</sub>ives the followin<sub>g</sub> <sub>exact</sub> fi<sub>n</sub>i<sub>te-c</sub>h<sub>a</sub>i<sub>n</sub> id<sub>ent</sub>i<sub>ty.</sub>

Theorem A.1 (Gaussian chains). Let $P _ { L } = G _ { L } \cdot \cdot \cdot G _ { 1 }$ be a chain of independent Gaussian stages $G _ { \ell } \in$ $\mathbb { R } ^ { n _ { \ell } \times n _ { \ell - 1 } }$ with entries $N \big ( 0 , 1 / n _ { \ell } \big )$ , where $n _ { \mathrm { 0 } } = d .$ Apply the chain to $Z \sim \mathcal { N } ( 0 , 2 \sigma ^ { 2 } I _ { d } )$ . With correlation taken jointly over Z and the stages,

$$
\mathrm { C o r r } ^ { 2 } \big ( \| Z \| ^ { 2 } , \| P _ { L } Z \| ^ { 2 } \big ) = \frac { 2 / d } { \prod _ { \ell = 0 } ^ { L } \big ( 1 + 2 / n _ { \ell } \big ) - 1 }
$$

at every finite size. A single stage of width m therefore gives $\mathrm { C o r r } = \sqrt { m / \left( m + d + 2 \right) }$ . For a typical sampled map,

$$
r _ { 2 } \big ( P _ { L } ^ { \top } P _ { L } \big ) = \big ( 1 + o _ { \mathbb { P } } ( 1 ) \big ) \left( \sum _ { \ell = 0 } ^ { L } \frac { 1 } { n _ { \ell } } \right) ^ { - 1 } .
$$

For one stage, the deterministic equivalent is $\left( 1 / m + 1 / d \right) ^ { - 1 }$ . The finite ratio of moments is md $/ ( m + d + 1 )$ because E tr M = d and E tr $M ^ { 2 } = d \big ( m + d + 1 \big ) / m$

C<sub>on</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>on</sub> i<sub>ts</sub> i<sub>nput,</sub> <sub>eac</sub>h <sub>stage</sub> <sub>mu</sub>l<sub>t</sub>i<sub>p</sub>li<sub>es</sub> <sub>square</sub>d <sub>norm</sub> b<sub>y</sub> <sub>an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> $\chi _ { n _ { \ell } } ^ { 2 } / n _ { \ell }$ f<sub>actor, so re</sub>l<sub>at</sub>i<sub>ve</sub> secon<sup>d</sup> moments mu<sup>l</sup>t<sup>i</sup>p<sup>l</sup>y. <sup>Th</sup>e <sup>fi</sup>n<sup>i</sup>te <sup>id</sup>ent<sup>i</sup>ty concerns t<sup>h</sup>e jo<sup>i</sup>nt <sup>l</sup>aw o<sup>f</sup> stages an<sup>d</sup> <sup>i</sup>nput, w<sup>h</sup>ereas t<sup>h</sup>e $r _ { 2 }$ <sub>statement</sub> <sub>concerns</sub> <sub>a</sub> <sub>typ</sub>i<sub>ca</sub>l fi<sub>xe</sub>d <sub>map</sub> <sub>asymptot</sub>i<sub>ca</sub>ll<sub>y.</sub> Th<sub>e</sub> <sub>ver</sub>ifi<sub>cat</sub>i<sub>on</sub> i<sub>n</sub> <sub>sect</sub>i<sub>on</sub> 11 di<sub>st</sub>i<sub>ngu</sub>i<sub>s</sub>h<sub>es</sub> <sub>t</sub>h<sub>e</sub> <sub>exact</sub> id<sub>ent</sub>i<sub>ty</sub> f<sub>rom</sub> i<sub>ts</sub> d<sub>eterm</sub>i<sub>n</sub>i<sub>st</sub>i<sub>c</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ent.</sub>

Th<sub>e</sub> <sub>po</sub>l<sub>ar</sub> d<sub>ecompos</sub>i<sub>t</sub>i<sub>on</sub> f<sub>actors</sub> <sub>one</sub> G<sub>auss</sub>i<sub>an</sub> <sub>stage</sub> i<sub>nto</sub> <sub>a</sub> H<sub>aar-</sub>di<sub>str</sub>ib<sub>ute</sub>d <sub>row-ort</sub>h<sub>onorma</sub>l f<sub>actor</sub> <sub>an</sub>d <sub>an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> Wi<sub>s</sub>h<sub>art</sub> f<sub>actor.</sub> Th<sub>e</sub> fi<sub>rst</sub> i<sub>mposes</sub> <sub>t</sub>h<sub>e</sub> <sub>ran</sub>k b<sub>ott</sub>l<sub>enec</sub>k<sub>;</sub> <sub>t</sub>h<sub>e</sub> <sub>secon</sub>d <sub>a</sub>dd<sub>s</sub> <sub>a</sub> <sub>mu</sub>l<sub>t</sub>i<sub>p</sub>li<sub>cat</sub>i<sub>ve</sub> $\chi _ { d } ^ { 2 } / d$ <sub>norm</sub> fl<sub>uctuat</sub>i<sub>on.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>pro</sub>d<sub>uct</sub> h<sub>as</sub> <sub>t</sub>h<sub>e</sub> f<sub>u</sub>ll $\dot { x } _ { m } ^ { 2 } / m$ fl<sub>uctuat</sub>i<sub>on.</sub> Th<sub>us</sub> <sub>a</sub> <sub>square</sub> G<sub>auss</sub>i<sub>an</sub> <sub>stage</sub> i<sub>s</sub> f<sub>u</sub>ll <sub>ran</sub>k b<sub>ut</sub> h<sub>as unw</sub>hi<sub>tene</sub>d <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> ${ \sqrt { d / ( 2 d + 2 ) } }  1 / { \sqrt { 2 } } .$ <sub>.</sub> I<sub>n contrast, a square ort</sub>h<sub>ogona</sub>l <sub>stage</sub> d<sub>oes not</sub> di<sub>stort</sub> Euclidean distance. For L square Gaussian stages of width $n ,$ <sub>t</sub>h<sub>e</sub> H<sub>an</sub>i<sub>n–</sub>Ni<sub>ca</sub> h<sub>armon</sub>i<sub>c</sub> <sub>sum</sub> <sub>contr</sub>ib<sub>utes</sub> L/n and the random input contributes 1/n. Hence $r _ { 2 } \sim n / { \left( L + 1 \right) }$ <sub>an</sub>d <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> <sub>ten</sub>d<sub>s</sub> <sub>to</sub> $1 / \sqrt { L + 1 }$

Sequential composition difers from parallel concatenation. Concatenatin<sub>g</sub> k independent m-row <sub>s</sub>k<sub>etc</sub>h<sub>es</sub> <sub>g</sub>i<sub>ves</sub> $1 / r _ { 2 } \stackrel { - } { \sim } 1 / d + 1 / ( k m )$ <sub>, as</sub> i<sub>n t</sub>h<sub>eorem</sub> 10<sub>.</sub>1<sub>.</sub> Th<sub>e c</sub>h<sub>a</sub>i<sub>n equ</sub>i<sub>va</sub>l<sub>ent</sub> i<sub>s spec</sub>ifi<sub>c to t</sub>h<sub>e</sub> i<sub>.</sub>i<sub>.</sub>d<sub>.</sub> G<sub>auss</sub>i<sub>an</sub> <sub>stages</sub> <sub>cons</sub>id<sub>ere</sub>d <sub>a</sub>b<sub>ove</sub> <sub>an</sub>d <sub>to</sub> <sub>ensem</sub>bl<sub>es</sub> <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>e</sub> <sub>same</sub> li<sub>m</sub>i<sub>t</sub>i<sub>ng</sub> <sub>trace</sub> <sub>moments;</sub> <sub>structure</sub>d <sub>trans</sub>f<sub>orms</sub> <sub>can</sub> h<sub>ave ot</sub>h<sub>er spectra</sub>l l<sub>aws.</sub> M<sub>oreover,</sub> $r _ { 2 }$ contains no sin<sub>g</sub>u<sup>l</sup>ar-vector in<sup>f</sup>ormation, so an anisotro<sub>p</sub>ic <sup>d</sup>ia<sub>g</sub>nostic <sub>a</sub>l<sub>so</sub> <sub>nee</sub>d<sub>s</sub> <sub>an</sub> <sub>a</sub>li<sub>gnment</sub> <sub>quant</sub>i<sub>ty</sub> <sub>suc</sub>h <sub>as</sub>

$$
\mathsf { \rho } _ { \Sigma } ( T ) ^ { 2 } = \frac { [ \mathrm { t r } \bigl ( M \Sigma ^ { 2 } \bigr ) ] ^ { 2 } } { \mathrm { t r } \bigl ( \Sigma ^ { 2 } \bigr ) \mathrm { t r } \bigl ( M \Sigma M \Sigma \bigr ) } .
$$

## A.1 Precision and the hard edge

Marchenko and Pastur [43] derived the limitin<sub>g</sub> law that<sub>,</sub> for a normalized s<sub>q</sub>uare Gaussian ma<sub>p,</sub> <sub>g</sub>ives the <sub>quarter-c</sub>i<sub>rc</sub>l<sub>e</sub> <sub>s</sub>i<sub>ngu</sub>l<sub>ar-va</sub>l<sub>ue</sub> d<sub>ens</sub>i<sub>ty</sub>

$$
\rho ( s ) = \frac { 1 } { \pi } \sqrt { 4 - s ^ { 2 } } , \qquad 0 \leq s \leq 2 .
$$

The origin is the hard edge: singular values are constrained to be nonnegative, and the limiting density remains nonzero there. Under the additive-noise model in theorem 9.1(iii), the fraction of directions with

si<sub>g</sub>na<sup>l</sup>-to-noise ratio <sup>b</sup>e<sup>l</sup>ow one is a<sub>pp</sub>roximate<sup>l</sup><sub>y</sub> $( 2 / \pi ) \varepsilon$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> <sub>t</sub>h<sub>e</sub> <sub>average</sub> <sub>poster</sub>i<sub>or</sub> l<sub>oss</sub> h<sub>as</sub> <sub>t</sub>h<sub>e</sub> <sub>exact</sub> li<sub>m</sub>i<sub>t</sub>

$$
\frac { 1 } { d } \ M \ M \mathrm { S E } ( x \mid \gamma , T ) \longrightarrow \Psi _ { \mathrm { e d g e } } ( \varepsilon ) = \frac { \varepsilon } { 2 } \left( \sqrt { \varepsilon ^ { 2 } + 4 } - \varepsilon \right) = \varepsilon - \frac { \varepsilon ^ { 2 } } { 2 } + O ( \varepsilon ^ { 3 } ) .
$$

Edelman [44] showed that $s _ { \operatorname* { m i n } } = \Theta _ { \mathbb { P } } \bigl ( 1 / d \bigr )$ <sub>a</sub>f<sub>ter</sub> <sub>norma</sub>li<sub>zat</sub>i<sub>on.</sub> Th<sub>us,</sub> if $\varepsilon \approx 2 ^ { - b }$ , <sup>recoverin</sup>g <sup>ever</sup>y <sup>direction</sup> <sup>re</sup>q<sup>uires</sup> $b \gtrsim \log _ { 2 } d .$ Th<sub>e</sub> <sub>average</sub> <sub>unrecovere</sub>d f<sub>ract</sub>i<sub>on</sub> i<sub>s</sub> i<sub>nstea</sub>d <sub>approx</sub>i<sub>mate</sub>l<sub>y</sub> $2 ^ { - b }$ <sub>,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>o</sub>f $\mathrm { . } d .$ Th<sub>ese</sub> <sub>statements</sub> <sub>ca</sub>lib<sub>rate</sub> <sub>a</sub>ddi<sub>t</sub>i<sub>ve</sub> G<sub>auss</sub>i<sub>an</sub> <sub>no</sub>i<sub>se;</sub> fl<sub>oat</sub>i<sub>ng-po</sub>i<sub>nt</sub> <sub>roun</sub>d<sub>o</sub>f i<sub>s</sub> <sub>not</sub> i<sub>tse</sub>lf i<sub>sotrop</sub>i<sub>c</sub> <sub>a</sub>ddi<sub>t</sub>i<sub>ve</sub> <sub>no</sub>i<sub>se.</sub>

## Appendix B. Proofs

Th<sub>roug</sub>h<sub>out,</sub> l<sub>et</sub> $Z = X - X ^ { \prime }$ , so $Z \sim { \mathcal { N } } ( 0 , 2 \Sigma )$ <sub>an</sub>d $D = \| Z \| ^ { 2 }$

## B.1 Proof of proposition 3.4

L<sub>e</sub>t $r = \lfloor n / 2 \rfloor$ as in <sub>p</sub>ro<sub>p</sub>osition 3.4. For the disjoint <sub>p</sub>airs (2ℓ − 1, 2ℓ), define

$$
R _ { \ell } = { \frac { D _ { 2 \ell - 1 , 2 \ell } ^ { \mathrm { r e p } } } { D _ { 2 \ell - 1 , 2 \ell } } } .
$$

Th<sub>e</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> $R _ { 1 } , \ldots , R _ { r }$ <sub>are</sub> i<sub>.</sub>i<sub>.</sub>d<sub>.,</sub> <sub>an</sub>d

$$
R _ { \ell } \stackrel { d } { = } \frac { \chi _ { m } ^ { 2 } / m } { \chi _ { d } ^ { 2 } / d } \sim F _ { m , d } .
$$

L<sub>e</sub>t $U = \chi _ { m } ^ { 2 } / m$ <sub>an</sub>d $V = \chi _ { d } ^ { 2 } / d$ be inde<sub>p</sub>endent. Chen and Rubin [45] showed that the median of a <sub>g</sub>amma <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> i<sub>s sma</sub>ll<sub>er t</sub>h<sub>an</sub> i<sub>ts mean, so</sub> $\operatorname* { P r } ( V \leq 1 ) \geq 1 / 2$ <sub>.</sub> M<sub>oreover, t</sub>h<sub>e</sub> Zh<sub>ang–</sub>Zh<sub>ou</sub> l<sub>ower</sub> b<sub>oun</sub>d f<sub>or t</sub>h<sub>e</sub> $x ^ { 2 }$ u<sub>pp</sub>er-tail <sub>p</sub>robabilit<sub>y</sub> [20<sub>,</sub> Cor. 3] <sub>g</sub>ives<sub>,</sub> uniforml<sub>y</sub> for $0 < \varepsilon \le 1$

$$
\begin{array} { r } { \mathrm { P r } \mathopen { } \mathclose \bgroup \left( U \geq 1 + \varepsilon \aftergroup \egroup \right) \geq c _ { 0 } e ^ { - C _ { 0 } m \varepsilon ^ { 2 } } . } \end{array}
$$

Th<sub>e</sub> <sub>event</sub> <sub>on</sub> <sub>t</sub>h<sub>e</sub> l<sub>e</sub>f<sub>t</sub> <sub>toget</sub>h<sub>er</sub> <sub>w</sub>i<sub>t</sub>h $V \leq 1$ im<sub>p</sub><sup>l</sup>ies $U / V \geq 1 + \varepsilon$ <sub>.</sub> Th<sub>us,</sub> if

$$
p _ { m , d } ^ { \mathrm { r e p } } ( \varepsilon ) = \mathrm { P r } \left( F _ { m , d } \notin \left[ 1 - \varepsilon , 1 + \varepsilon \right] \right) ,
$$

<sub>t</sub>h<sub>en</sub> $p _ { m , d } ^ { \mathrm { r e p } } ( \varepsilon ) \geq c e ^ { - C m \varepsilon ^ { 2 } }$ . <sup>Th</sup>e J<sup>L</sup> event <sup>i</sup>mp<sup>li</sup>es t<sup>h</sup>at every <sup>di</sup>sjo<sup>i</sup>nt-pa<sup>i</sup>r rat<sup>i</sup>o <sup>li</sup>es <sup>i</sup>n t<sup>h</sup>e prescr<sup>ib</sup>e<sup>d</sup> <sup>i</sup>nterva<sup>l</sup>, h<sub>ence</sub>

$$
\begin{array} { r } { \mathrm { P r } ( \mathcal { E } _ { \varepsilon } ^ { \mathrm { r e p } } ) \leq \left( 1 - p _ { m , d } ^ { \mathrm { r e p } } ( \varepsilon ) \right) ^ { r } \leq \exp \left( - c r e ^ { - C m \varepsilon ^ { 2 } } \right) . } \end{array}
$$

If $\mathrm { P r } ( \mathcal { E } _ { \varepsilon } ^ { \mathrm { r e p } } ) \geq 1 - \delta$ <sub>,</sub> <sub>ta</sub>ki<sub>ng</sub> l<sub>ogar</sub>i<sub>t</sub>h<sub>ms</sub> <sub>an</sub>d <sub>rearrang</sub>i<sub>ng</sub> <sub>proves</sub> <sub>t</sub>h<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>necessary</sub> <sub>con</sub>di<sub>t</sub>i<sub>on</sub> i<sub>n</sub> p<sup>ro</sup>p<sup>osition</sup> <sup>3.4.</sup> <sup>Finall</sup>y, $- \log ( 1 - \delta ) \leq 2 \delta$ f<sub>or</sub> $\delta \leq 1 / 2$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> <sub>t</sub>h<sub>e</sub> <sub>state</sub>d <sub>or</sub>d<sub>er.</sub> □

## B.2 Moment matching for the replacement cloud

In t<sup>h</sup>e settin<sub>g</sub> o<sup>f</sup> <sub>p</sub>ro<sub>p</sub>osition 3.1, <sup>l</sup>et $N _ { p } = { \binom { n } { 2 } }$ <sub>an</sub>d $a _ { \varepsilon } = \varepsilon / { \left( 2 + \varepsilon \right) }$ <sub>.</sub> Th<sub>e</sub> G<sub>auss</sub>i<sub>an rep</sub>l<sub>acement c</sub>l<sub>ou</sub>d <sub>matc</sub>h<sub>es t</sub>h<sub>e</sub> <sub>popu</sub>l<sub>at</sub>i<sub>on</sub> <sub>mean</sub> <sub>square</sub>d di<sub>stance</sub> b<sub>ut</sub> i<sub>n</sub>fl<sub>ates</sub> i<sub>ts</sub> <sub>var</sub>i<sub>ance:</sub> $\mathrm { V a r } \big ( D _ { i j } ^ { \mathrm { r e p } } \big ) = 8 \sigma ^ { 4 } d ^ { 2 } / m _ { \mathrm { \Omega } }$ <sub>,</sub> <sub>w</sub>h<sub>ereas</sub> $\mathrm { V a r } \bigl ( D _ { i j } \bigr ) = 8 \sigma ^ { 4 } d .$ <sup>Th</sup>e rep<sup>l</sup>acement map sat<sup>i</sup>s<sup>fi</sup>es t<sup>h</sup>e J<sup>L b</sup>oun<sup>d</sup> t<sup>h</sup>roug<sup>h</sup> concentrat<sup>i</sup>on rat<sup>h</sup>er t<sup>h</sup>an moment matc<sup>hi</sup>ng. <sup>M</sup>ore

<sub>genera</sub>ll<sub>y,</sub> if <sub>two</sub> di<sub>stance</sub> f<sub>am</sub>ili<sub>es</sub> h<sub>ave</sub> <sub>common</sub> <sub>mean</sub> $\mu _ { D }$ <sub>an</sub>d <sub>var</sub>i<sub>ances</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> $\nu _ { X }$ <sub>an</sub>d $\nu _ { Y }$ <sub>,</sub> Ch<sub>e</sub>b<sub>ys</sub>h<sub>ev</sub>’<sub>s</sub> i<sub>nequa</sub>li<sub>ty</sub> <sub>an</sub>d <sub>a</sub> <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>g</sub>i<sub>ve</sub>

$$
\mathrm { P r } ( \mathrm { f a i l u r e } ) \leq { \frac { N _ { p } ( \nu _ { X } + \nu _ { Y } ) } { a _ { \varepsilon } ^ { 2 } \mu _ { D } ^ { 2 } } } .
$$

F<sub>or</sub> <sub>t</sub>h<sub>e</sub> <sub>two</sub> G<sub>auss</sub>i<sub>an</sub> <sub>c</sub>l<sub>ou</sub>d<sub>s,</sub>

$$
\frac { \nu _ { X } } { \mu _ { D } ^ { 2 } } = \frac { 2 } { d } , \qquad \frac { \nu _ { Y } } { \mu _ { D } ^ { 2 } } = \frac { 2 } { m } , \qquad \mathrm { P r } ( \mathrm { f a i l u r e } ) \leq \frac { 2 N _ { p } } { a _ { \varepsilon } ^ { 2 } } \left( \frac { 1 } { d } + \frac { 1 } { m } \right) .
$$

Ch<sub>e</sub>b<sub>ys</sub>h<sub>ev contro</sub>l <sub>a</sub>l<sub>one t</sub>h<sub>ere</sub>f<sub>ore requ</sub>i<sub>res m</sub>i<sub>n</sub> $\{ m , d \} \ = \ \Omega \big ( n ^ { 2 } / \big ( \varepsilon ^ { 2 } \delta \big ) \big )$ <sub>.</sub> Th<sub>e</sub> l<sub>ogar</sub>i<sub>t</sub>h<sub>m</sub>i<sub>c</sub> di<sub>mens</sub>i<sub>on</sub> i<sub>n</sub> <sub>propos</sub>i<sub>t</sub>i<sub>on</sub> 3<sub>.</sub>1 <sub>comes</sub> f<sub>rom</sub> <sub>su</sub>b<sub>-exponent</sub>i<sub>a</sub>l <sub>concentrat</sub>i<sub>on,</sub> <sub>not</sub> <sub>mere</sub>l<sub>y</sub> f<sub>rom</sub> <sub>t</sub>h<sub>e</sub> fi<sub>rst</sub> <sub>two</sub> <sub>moments.</sub> E<sub>xact</sub> <sub>matc</sub>hi<sub>ng</sub> <sub>o</sub>f <sub>t</sub>h<sub>ose</sub> <sub>moments</sub> i<sub>mposes</sub> <sub>a</sub> dif<sub>erent</sub> di<sub>mens</sub>i<sub>ona</sub>l <sub>o</sub>b<sub>struct</sub>i<sub>on.</sub>

Proposition B.1 (Exact first-two-moment matching requires half dimension). Let $Y , Y ^ { \prime }$ be i.i.d. random vectors in $\mathbb { R } ^ { m }$ with finite fourth moments and covariance C. Then

$$
\mathbb { E } \left. Y - Y ^ { \prime } \right. ^ { 2 } = 2 \operatorname { t r } C
$$

and

$$
\operatorname { V a r } ( \left\| Y - Y ^ { \prime } \right\| ^ { 2 } ) = 2 \operatorname { V a r } ( \left\| Y - \mathbb { E } Y \right\| ^ { 2 } ) + 4 \operatorname { t r } ( C ^ { 2 } ) .
$$

If these moments equal 2σ<sup>2</sup>d and $8 \sigma ^ { 4 } d ,$ respectively, then m $\geq d / 2$ . This lower bound is sharp when d is even: for $m = d / 2$ and $U \sim \operatorname { U n i f } ( S ^ { m - 1 } )$ , the replacement $Y = \sigma { \sqrt { d } } U$ matches both moments exactly.

Centering Y does not change pairwise distances, so assume $\mathbb { E } Y = 0$ <sup>.</sup> <sup>Ex</sup>p<sup>andin</sup>g

$$
\left\| Y - Y ^ { \prime } \right\| ^ { 2 } = \left\| Y \right\| ^ { 2 } + \left\| Y ^ { \prime } \right\| ^ { 2 } - 2 Y ^ { \top } Y ^ { \prime }
$$

an<sup>d</sup> usin<sub>g</sub> in<sup>d</sup>e<sub>p</sub>en<sup>d</sup>ence <sub>g</sub>ives

$$
\begin{array} { r } { \mathbb { E } \left\| Y - Y ^ { \prime } \right\| ^ { 2 } = 2 \operatorname { t r } C , \qquad \operatorname { V a r } ( \left\| Y - Y ^ { \prime } \right\| ^ { 2 } ) = 2 \operatorname { V a r } ( \left\| Y \right\| ^ { 2 } ) + 4 \mathbb { E } ( Y ^ { \top } Y ^ { \prime } ) ^ { 2 } . } \end{array}
$$

<sup>The</sup> <sup>remainin</sup>g <sup>ex</sup>p<sup>ectation</sup> <sup>is</sup> $\operatorname { t r } ( C ^ { 2 } )$ <sub>.</sub> M<sub>oment</sub> <sub>matc</sub>hi<sub>ng</sub> <sub>t</sub>h<sub>ere</sub>f<sub>ore</sub> i<sub>mp</sub>li<sub>es</sub> <sub>tr</sub> $C = \sigma ^ { 2 } d$ <sub>an</sub>d

$$
8 \sigma ^ { 4 } d \geq 4 \mathrm { t r } ( C ^ { 2 } ) \geq \frac { 4 ( \mathrm { t r } C ) ^ { 2 } } { m } = \frac { 4 \sigma ^ { 4 } d ^ { 2 } } { m } ,
$$

where the middle inequality is Cauchy–Schwarz on the eigenvalues of C. Thus $m \geq d / 2$

F<sub>or t</sub>h<sub>e c</sub>l<sub>a</sub>i<sub>me</sub>d <sub>extrem</sub>i<sub>zer,</sub> $\| Y \| ^ { 2 } = \sigma ^ { 2 } d$ i<sub>s</sub> d<sub>eterm</sub>i<sub>n</sub>i<sub>st</sub>i<sub>c an</sub>d $\smash {  { \mathrm { C o v } } ( Y ) = ( \sigma ^ { 2 } d / m ) I _ { m } }$ <sub>.</sub> H<sub>ence</sub>

$$
\mathbb { E } \left\| Y - Y ^ { \prime } \right\| ^ { 2 } = 2 \sigma ^ { 2 } d , \quad \quad \mathrm { V a r } ( \left\| Y - Y ^ { \prime } \right\| ^ { 2 } ) = 4 \operatorname { t r } ( \mathrm { C o v } ( Y ) ^ { 2 } ) = \frac { 4 \sigma ^ { 4 } d ^ { 2 } } { m } = 8 \sigma ^ { 4 } d .
$$

Th<sub>ese</sub> <sub>matc</sub>hi<sub>ng</sub> <sub>moments</sub> <sub>prove</sub> <sub>s</sub>h<sub>arpness.</sub> Th<sub>us</sub> <sub>exact</sub> fi<sub>rst-two-moment</sub> <sub>matc</sub>hi<sub>ng</sub> <sub>requ</sub>i<sub>res</sub> <sub>a</sub> li<sub>near</sub> <sub>s</sub>h<sub>are</sub> o<sup>f</sup> t<sup>h</sup>e am<sup>bi</sup>ent <sup>di</sup>mens<sup>i</sup>on, even t<sup>h</sup>oug<sup>h</sup> t<sup>h</sup>e rep<sup>l</sup>acement map <sup>i</sup>n propos<sup>i</sup>t<sup>i</sup>on 3.1 sat<sup>i</sup>s<sup>fi</sup>es t<sup>h</sup>e J<sup>L</sup> <sup>b</sup>oun<sup>d</sup> <sup>i</sup>n l<sub>ogar</sub>i<sub>t</sub>h<sub>m</sub>i<sub>c</sub> di<sub>mens</sub>i<sub>on</sub> <sub>w</sub>i<sub>t</sub>h<sub>out</sub> <sub>matc</sub>hi<sub>ng</sub> <sub>t</sub>h<sub>e</sub> <sub>var</sub>i<sub>ance.</sub> □

## B.3 Proofoftheorem 5.1

Af<sub>ter row ort</sub>h<sub>onorma</sub>li<sub>zat</sub>i<sub>on, t</sub>h<sub>e s</sub>k<sub>etc</sub>h i<sub>s</sub> i<sub>n</sub>f<sub>ormat</sub>i<sub>ona</sub>ll<sub>y equ</sub>i<sub>va</sub>l<sub>ent to</sub> $( \Pi X , \Pi X ^ { \prime } )$ with Π a rank-r projector. <sup>S</sup>et $Z = ( X - X ^ { \prime } ) / ( \sqrt { 2 } \sigma ) , \mathcal { T } = \| Z \| ^ { 2 }$ <sub>,</sub> <sub>an</sub>d $U = \| \Pi Z \| ^ { 2 }$ <sub>.</sub> Th<sub>e</sub> G<sub>auss</sub>i<sub>an m</sub>id<sub>po</sub>i<sub>nt</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent o</sub>f $Z ,$ and the conditional law of T given ΠZ depends on the projected vector only through U. Hence $U \sim \chi _ { r } ^ { 2 }$ $V = \mathcal { T } - U \sim \chi _ { d - r } ^ { 2 } , U \perp V ,$ <sub>an</sub>d $\tau \perp \mathcal { O } \mid \hat { U }$ <sub>.</sub> S<sub>u</sub>fi<sub>c</sub>i<sub>ency</sub> <sub>t</sub>h<sub>en</sub> <sub>g</sub>i<sub>ves</sub> $\rho _ { \mathrm { H G R } } ( \dot { D } ; \mathcal { O } ) = \rho _ { \mathrm { H G R } } ( T ; U )$

W<sub>r</sub>it<sub>e</sub> $\textstyle { \mathcal { T } } = \sum _ { i = 1 } ^ { d } W _ { i }$ <sub>w</sub>i<sub>t</sub>h $W _ { i } \ i . \mathrm { i . d . } \ \chi _ { 1 } ^ { 2 }$ <sub>an</sub>d<sub>,</sub> b<sub>y rotat</sub>i<sub>on,</sub> $\begin{array} { r } { U = \sum _ { i \leq r } W _ { i } } \end{array}$ <sub>.</sub> Th<sub>e part</sub>i<sub>a</sub>l<sub>-sum t</sub>h<sub>eorem o</sub>f Dembo<sub>,</sub> Ka<sub>g</sub>an<sub>,</sub> and She<sub>pp</sub> [22] <sub>y</sub>ields $\rho _ { \mathrm { H G R } } \backsimeq \sqrt { r / d } ,$ <sub>, prov</sub>i<sub>ng t</sub>h<sub>eorem</sub> 5<sub>.</sub>1<sub>.</sub> Th<sub>e var</sub>i<sub>ance an</sub>d MMSE <sub>coro</sub>ll<sub>ar</sub>i<sub>es</sub> f<sub>o</sub>ll<sub>ow</sub> b<sub>ecause</sub> $\bar { \operatorname { V a r } } ( \mathbb { E } [ \hat { f } \mid \mathcal { O } ] ) / \operatorname { V a r } ( f ) \leq \rho _ { \mathrm { H G R } } ^ { 2 }$ □

## B.4 Proofoftheorems 5.2 and 5.3

Retain the variables U, V, and $\tau$ from the <sub>p</sub>recedin<sub>g</sub> <sub>p</sub>roof. For theorem 5.2, (U, T ) is the classical <sub>g</sub>amma <sub>p</sub>air wit<sup>h</sup> $\begin{array} { r } { U / \mathcal { T } \sim \mathrm { B e t a } \big ( \frac { r } { 2 } , \frac { d - r } { 2 } \big ) } \end{array}$ i<sub>n</sub>d<sub>epen</sub>d<sub>ent o</sub>f $\tau .$ . Grifiths [24] identifies the canonical (sin<sub>g</sub>ular) s<sub>y</sub>stem of <sub>t</sub>h<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l<sub>-expectat</sub>i<sub>on</sub> <sub>operator</sub> <sub>as</sub> <sub>genera</sub>li<sub>ze</sub>d L<sub>aguerre</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>s</sub> <sub>w</sub>i<sub>t</sub>h <sub>s</sub>i<sub>ngu</sub>l<sub>ar</sub> <sub>va</sub>l<sub>ues</sub> $\ell _ { k } = [ \left( r / 2 \right) _ { k } / ( d / 2 ) _ { k } ] ^ { 1 / 2 }$ <sub>,</sub> <sub>an</sub>d P<sub>arseva</sub>l’<sub>s</sub> id<sub>ent</sub>i<sub>ty</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>var</sub>i<sub>ance</sub> d<sub>ecompos</sub>i<sub>t</sub>i<sub>on.</sub> Th<sub>e</sub> <sub>state</sub>d <sub>asymptot</sub>i<sub>cs</sub> f<sub>o</sub>ll<sub>ow</sub> f<sub>rom</sub> l<sub>og</sub> $\begin{array} { r } { \ell _ { k } ^ { 2 } = \sum _ { j < k } \log \frac { r / 2 + j } { d / 2 + j } } \end{array}$

Fi<sub>na</sub>ll<sub>y,</sub> f<sub>or t</sub>h<sub>eorem</sub> $5 . 3 , I ( D ; \mathcal { O } ) = I ( \mathcal { T } ; U ) = h ( \mathcal { T } ) - h ( \mathcal { T } \mid U ) = h ( \mathcal { T } ) - h ( V )$ b<sub>y</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>state</sub>d <sub>gamma-entropy</sub> dif<sub>erence;</sub> <sub>t</sub>h<sub>e</sub> li<sub>m</sub>i<sub>ts</sub> f<sub>o</sub>ll<sub>ow</sub> f<sub>rom</sub> S<sub>t</sub>i<sub>r</sub>li<sub>ng</sub> <sub>expans</sub>i<sub>ons</sub> <sub>o</sub>f $\ln _ { \Gamma }$ . □

## B.5 Proofoftheorem 6.1

Th<sub>e</sub> <sub>con</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>mean</sub> ${ \widehat { X } } = \mathbb { E } [ X \mid L X ]$ i<sub>s</sub> G<sub>auss</sub>i<sub>an w</sub>i<sub>t</sub>h <sub>covar</sub>i<sub>ance</sub> $B = \Sigma L ^ { \top } ( L \Sigma L ^ { \top } ) ^ { \dagger } L \Sigma = \Sigma ^ { 1 / 2 } P \Sigma ^ { 1 / 2 }$ <sub>w</sub>h<sub>ere t</sub>h<sub>e pseu</sub>d<sub>o</sub>i<sub>nverse</sub> id<sub>ent</sub>i<sub>ty</sub> i<sub>mp</sub>li<sub>es t</sub>h<sub>at</sub> $P = \Sigma ^ { 1 / 2 } L ^ { \top } ( L \Sigma L ^ { \top } ) ^ { \dagger } L \Sigma ^ { 1 / 2 }$ is idempotent of rank at most m. Conditionin<sub>g</sub> on O <sub>g</sub>ives $X - X ^ { \prime } \sim \mathcal { N } ( \widehat { X } - \widehat { X } ^ { \prime } , 2 ( \Sigma - B ) )$ , so

$$
\operatorname { \mathbb { E } } [ D \mid { \mathcal { O } } ] = \left\| { \widehat { X } } - { \widehat { X } } ^ { \prime } \right\| ^ { 2 } + 2 \operatorname { t r } ( \Sigma - B ) .
$$

B<sub>ecause</sub> $\widehat { X } - \widehat { X } ^ { \prime } \sim { \mathcal { N } } ( 0 , 2 B )$

$$
\operatorname { V a r } ( \mathbb { E } [ D \mid { \mathcal { O } } ] ) = 8 \operatorname { t r } ( B ^ { 2 } ) .
$$

Si<sub>nce</sub> $\operatorname { t r } \left( B ^ { 2 } \right) = \operatorname { t r } \left( \left( P \Sigma P \right) ^ { 2 } \right)$ , Poincaré separation implies that the eigenvalues of the compression PΣP are d<sub>om</sub>i<sub>nate</sub>d <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> b<sub>y</sub> $\lambda _ { 1 } , \ldots , \lambda _ { m }$ <sub>.</sub> H<sub>ence</sub> t $\begin{array} { r } { \operatorname { r } ( B ^ { 2 } ) \leq \sum _ { j \leq m } \lambda _ { j } ^ { 2 } } \end{array}$ , with e<sub>q</sub>ualit<sub>y</sub> when ran<sub>g</sub>e(P) is the to<sub>p</sub>-m ei<sub>g</sub>ens<sub>p</sub>ace. We derive the minimum mean-s<sub>q</sub>uare error (MMSE) claim from the law of total variance:

$$
\operatorname* { i n f } _ { g } \mathbb { E } \left[ ( D - g ( \mathcal { O } ) ) ^ { 2 } \right] = \mathbb { E } \operatorname { V a r } ( D \mid \mathcal { O } ) = \operatorname { V a r } ( D ) - \operatorname { V a r } ( \mathbb { E } \left[ D \mid \mathcal { O } \right] ) \geq 8 \sum _ { j > m } \lambda _ { j } ^ { 2 } .
$$

F<sub>or t</sub>h<sub>e contrast, set</sub> $a = X _ { 1 } - X _ { 2 } , b = X _ { 1 } + X _ { 2 } - 2 X _ { 0 }$ <sub>,</sub> <sub>an</sub>d $c = X _ { 0 } + X _ { 1 } + X _ { 2 }$ . <sup>C</sup>ovar<sup>i</sup>ance com<sub>p</sub>utat<sup>i</sup>on <sub>s</sub>h<sub>ows</sub> <sub>t</sub>h<sub>at</sub> $a , b ,$ c are mutually independent, $S = a ^ { \top } b ,$ <sub>, an</sub>d $\mathrm { V a r } \big ( S \big ) = \mathrm { t r } \big ( 6 \Sigma \cdot 2 \Sigma \big ) = 1 2 \mathrm { t r } \big ( \Sigma ^ { 2 } \big )$ <sub>.</sub> Th<sub>e o</sub>b<sub>servat</sub>i<sub>ons</sub> are a <sup>li</sup>near <sup>bi</sup>ject<sup>i</sup>on o<sup>f</sup> $( L a , L b , L c )$ , so a, b remain conditionall<sub>y</sub> independent <sub>g</sub>iven O. Hence $\mathbb { E } [ S | \mathcal { O } ] = \widehat { \boldsymbol { a } } ^ { \top } \widehat { \boldsymbol { b } }$ <sub>an</sub>d $\operatorname { V a r } ( \widehat { a } ^ { \top } \widehat { b } ) = \operatorname { t r } ( 2 B \cdot 6 B ) = 1 2 \operatorname { t r } ( B ^ { 2 } )$ <sub>,</sub> <sub>y</sub>i<sub>e</sub>ldi<sub>ng</sub> <sub>t</sub>h<sub>e</sub> <sub>same</sub> <sub>rat</sub>i<sub>o.</sub> □

## B.6 Proofoftheorem 6.2

L<sub>e</sub>t $A = G _ { 1 } ^ { 2 }$ <sub>an</sub>d $B = G _ { 2 } ^ { 2 }$ <sub>,</sub> <sub>w</sub>h<sub>ose</sub> <sub>moments</sub> <sub>are</sub> 1<sub>,</sub> 3<sub>,</sub> 15<sub>,</sub> 105<sub>.</sub> Th<sub>e</sub> <sub>covar</sub>i<sub>ances</sub> <sub>among</sub> $\{ D , D ^ { 2 } \}$ <sub>an</sub>d $\{ U , U ^ { 2 } \}$ <sub>are</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>s</sub> i<sub>n</sub> <sub>t</sub>h<sub>ese</sub> <sub>moments:</sub>

$$
\mathrm { V a r } D = 1 0 , \quad \mathrm { C o v } ( D , D ^ { 2 } ) = 1 3 2 , \qquad \mathrm { V a r } D ^ { 2 } = 2 2 4 0 ,
$$

$$
\mathrm { V a r } U = 8 , \mathrm { C o v } ( U , U ^ { 2 } ) = 9 6 , \mathrm { V a r } U ^ { 2 } = 1 5 3 6 ,
$$

$$
\mathrm { C o v } ( D , U ) = 8 , ~ \mathrm { C o v } ( D , U ^ { 2 } ) = 9 6 , ~ \mathrm { C o v } ( D ^ { 2 } , U ) = 1 1 2 , ~ \mathrm { C o v } ( D ^ { 2 } , U ^ { 2 } ) = 1 7 2 8 .
$$

Th<sub>e</sub> l<sub>argest</sub> <sub>e</sub>i<sub>genva</sub>l<sub>ue</sub> <sub>o</sub>f $\cdot \Sigma _ { D D } ^ { - 1 } \Sigma _ { D U } \Sigma _ { U U } ^ { - 1 } \Sigma _ { U D } \mathrm { i s } \left( 2 4 6 + 2 \sqrt { 2 0 1 } \right) / 3 1 1 \approx 0 . 8 8 2 1 7 0$ <sub>,</sub> <sub>an</sub>d i<sub>ts</sub> <sub>square</sub> <sub>root</sub> <sub>excee</sub>d<sub>s</sub> $\sqrt { 4 / 5 }$ <sub>.</sub> R<sub>estr</sub>i<sub>ct</sub>i<sub>ng</sub> <sub>to</sub> <sub>qua</sub>d<sub>rat</sub>i<sub>c</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>s</sub> <sub>ma</sub>k<sub>es</sub> <sub>t</sub>hi<sub>s</sub> <sub>va</sub>l<sub>ue</sub> <sub>a</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> $\rho _ { \mathrm { H G R } }$ □

## B.7 Proofoftheorem 6.3

R<sub>emove</sub> <sub>t</sub>h<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> G<sub>auss</sub>i<sub>an</sub> <sub>m</sub>id<sub>po</sub>i<sub>nt</sub> <sub>an</sub>d <sub>w</sub>hi<sub>ten</sub> i<sub>ns</sub>id<sub>e</sub> <sub>eac</sub>h <sub>spectra</sub>l bl<sub>oc</sub>k<sub>.</sub> Th<sub>en</sub>

$$
D = 2 \sum _ { g = 1 } ^ { G } \lambda _ { g } T _ { g } , \qquad T _ { g } = U _ { g } + V _ { g } ,
$$

<sub>w</sub>h<sub>ere</sub> <sub>t</sub>h<sub>e</sub> bl<sub>oc</sub>k <sub>pa</sub>i<sub>rs</sub> <sub>are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>an</sub>d

$$
U _ { g } \sim \chi _ { s _ { g } } ^ { 2 } , \qquad V _ { g } \sim \chi _ { r _ { g } - s _ { g } } ^ { 2 } , \qquad U _ { g } \perp V _ { g } .
$$

For estimating D, the vector $( U _ { g } ) _ { g }$ i<sub>s su</sub>fi<sub>c</sub>i<sub>ent</sub> f<sub>or t</sub>h<sub>e</sub> f<sub>u</sub>ll <sub>s</sub>k<sub>etc</sub>h<sub>.</sub> Th<sub>e</sub> b<sub>eta–gamma c</sub>h<sub>anne</sub>l $T _ { g } \mapsto U _ { g }$ h<sub>as</sub> <sub>nontr</sub>i<sub>v</sub>i<sub>a</sub>l <sub>max</sub>i<sub>ma</sub>l <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> $\sqrt { s _ { g } / r _ { g } }$ . Witsenhausen [25] <sub>p</sub>roved the tensorization theorem<sub>;</sub> to<sub>g</sub>ether with <sup>data</sup> p<sup>rocessin</sup>g, <sup>it</sup> g<sup>ives</sup>

$$
\mathsf { p } _ { \mathrm { H G R } } ( D ; \mathcal { O } ) \leq \operatorname* { m a x } _ { g } \sqrt { s _ { g } / r _ { g } } .
$$

F<sub>or t</sub>h<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>, ta</sub>k<sub>e</sub> $\begin{array} { r } { G ( U ) = 2 \sum _ { g } \lambda _ { g } ( U _ { g } - s _ { g } ) } \end{array}$ <sub>.</sub> Th<sub>e</sub> <sub>om</sub>i<sub>tte</sub>d <sub>we</sub>i<sub>g</sub>h<sub>te</sub>d <sub>sum</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>o</sub>f $G ( U )$ , so

$$
\operatorname { C o v } ( D , G ( U ) ) = \operatorname { V a r } ( G ( U ) ) = 8 \sum _ { g } \lambda _ { g } ^ { 2 } s _ { g } , \qquad \operatorname { V a r } ( D ) = 8 \sum _ { g } \lambda _ { g } ^ { 2 } r _ { g } .
$$

Th<sub>us</sub> $\mathrm { C o r r } ( D , G ( U ) ) = \sqrt { \alpha _ { \Pi } ( \Sigma ) }$ . I<sup>f</sup> ever<sub>y</sub> $s _ { g } / r _ { g }$ e<sub>q</sub>ua<sup>l</sup>s θ, t<sup>h</sup>e bounds coincide. T<sup>h</sup>e <sup>fl</sup>at-su<sub>pp</sub>ort statement is <sub>t</sub>h<sub>e one-</sub>bl<sub>oc</sub>k <sub>spec</sub>i<sub>a</sub>li<sub>zat</sub>i<sub>on.</sub> □

## B.8 Proofoftheorem 7.1

Wi<sub>t</sub>h $a = X _ { 1 } - X _ { 2 }$ <sub>an</sub>d $b = X _ { 1 } + X _ { 2 } - 2 X _ { 0 }$ in<sup>d</sup>e<sub>p</sub>en<sup>d</sup>ent isotro<sub>p</sub>ic Gaussians, $D _ { 0 1 } - D _ { 0 2 } = a ^ { \top } b$ <sub>an</sub>d $\begin{array} { r } { \widetilde { D } _ { 0 1 } - \widetilde { D } _ { 0 2 } = \frac { d } { m } a ^ { \top } \Pi b . } \end{array}$ <sub>.</sub> C<sub>on</sub>di<sub>t</sub>i<sub>ona</sub>l <sub>on</sub> $^ { a , }$ these diferences are centered jointl<sub>y</sub> Gaussian in b with correlation $\sqrt { a ^ { \top } \Pi a / a ^ { \top } a } ,$ w<sup>h</sup>ose s<sub>q</sub>uare is $\begin{array} { r } { \mathrm { B e t a } \big ( \frac { m } { 2 } , \frac { d - m } { 2 } \big ) } \end{array}$ b<sub>y</sub> rotation invariance. She<sub>pp</sub>ard [27] showed that the si<sub>g</sub>na<sub>g</sub>reement <sub>p</sub>ro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>it<sub>y</sub> is $\frac { 1 } { 2 } + \frac { 1 } { \pi }$ arcsin ρ; averaging over a yields the law. Applying arcsin $\begin{array} { r } { \sqrt { B } \le \frac { \pi } { 2 } \sqrt { B } } \end{array}$ <sub>an</sub>d E $\sqrt { B } \leq \sqrt { m / d }$ gives the bound, whereas concentration of B at m/d gives the expansion. Kendall’s τ over one q<sup>uer</sup>y<sup>’s</sup> <sup>rankin</sup>g <sup>avera</sup>g<sup>es</sup> p<sup>airwise</sup> <sup>si</sup>g<sup>n</sup> <sup>a</sup>g<sup>reements</sup>, <sup>each</sup> <sup>with</sup> <sup>mar</sup>g<sup>inal</sup> p<sup>robabilit</sup>y $p _ { m , d }$ b<sub>y</sub> <sub>exc</sub>h<sub>angea</sub>bili<sub>ty;</sub> h<sub>ence</sub> $\mathbb { E } [ \tau ] = \breve { 2 } p _ { m , d } - 1$ □

## B.9 Proof of theorem 7.2 and proposition 7.3

By rotational invariance, take R to select the first m coordinates and put $r = d - m$ . S<sub>p</sub><sup>l</sup>it $X _ { j } = \left( Y _ { j } , Z _ { j } \right)$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
U _ { j } = \left\| Y _ { j } - Y _ { 0 } \right\| ^ { 2 } , \qquad V _ { j } = \left\| Z _ { j } - Z _ { 0 } \right\| ^ { 2 } .
$$

<sup>Th</sup>e projecte<sup>d</sup> ran<sup>ki</sup>ng <sup>i</sup>s t<sup>h</sup>at o<sup>f</sup> $( U _ { j } ) _ { j }$ <sub>, t</sub>h<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>ran</sub>ki<sub>ng</sub> i<sub>s t</sub>h<sub>at o</sub>f $( U _ { j } + V _ { j } ) _ { j }$ , and the vectors U and V are i<sub>n</sub>d<sub>epen</sub>d<sub>ent.</sub>

F<sub>or</sub> <sub>one</sub> <sub>coor</sub>di<sub>nate,</sub> <sub>set</sub>

$$
W = ( ( X _ { 1 } - X _ { 0 } ) ^ { 2 } - 2 , \ldots , ( X _ { q } - X _ { 0 } ) ^ { 2 } - 2 ) .
$$

<sup>Di</sup>rect <sup>G</sup>auss<sup>i</sup>an moments <sub>g</sub><sup>i</sup>ve $\operatorname { C o v } ( W ) = 6 I _ { q } + 2 1 1 ^ { \top } = : C _ { q } .$ <sub>.</sub> I<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub>l<sub>y,</sub>

$$
\frac { U - 2 m 1 } { \sqrt { m } } \Rightarrow G ^ { ( 1 ) } , \qquad \frac { V - 2 r 1 } { \sqrt { r } } \Rightarrow G ^ { ( 2 ) } ,
$$

<sub>w</sub>h<sub>ere</sub> $G ^ { ( 1 ) } , G ^ { ( 2 ) } \sim \mathcal { N } ( 0 , C _ { q } )$ . <sup>W</sup>e ma<sub>y</sub> wr<sup>i</sup>te

$$
G ^ { ( 1 ) } = \sqrt { 6 } G + \sqrt { 2 } \xi 1 , \qquad G ^ { ( 2 ) } = \sqrt { 6 } H + \sqrt { 2 } \eta 1 .
$$

<sup>Th</sup>e common s<sup>hif</sup>ts <sup>d</sup>o not a<sup>f</sup>ect ran<sup>ki</sup>ngs. <sup>Th</sup>e <sup>li</sup>m<sup>i</sup>t<sup>i</sup>ng projecte<sup>d</sup> scores are t<sup>h</sup>ere<sup>f</sup>ore $G ,$ <sub>an</sub>d <sub>t</sub>h<sub>e</sub> li<sub>m</sub>i<sub>t</sub>i<sub>ng</sub> ori<sub>g</sub>ina<sup>l</sup> scores are $\sqrt { \alpha } G + \sqrt { 1 - \alpha } H$ . Ties <sup>h</sup>ave <sub>p</sub>ro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>it<sub>y</sub> zero, so t<sup>h</sup>e continuous ma<sub>pp</sub>in<sub>g</sub> t<sup>h</sup>eorem <sub>p</sub>roves the top-k limit.

F<sub>or</sub> $k = 1$ <sub>,</sub> <sub>t</sub>h<sub>e</sub> <sub>event</sub> <sub>t</sub>h<sub>at</sub> <sub>one</sub> fi<sub>xe</sub>d i<sub>n</sub>d<sub>ex</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> b<sub>ot</sub>h <sub>ran</sub>ki<sub>ngs</sub> i<sub>s</sub> <sub>an</sub> i<sub>ntersect</sub>i<sub>on</sub> <sub>o</sub>f li<sub>near</sub> h<sub>a</sub>lf<sub>spaces</sub> i<sub>n</sub> <sub>t</sub>h<sub>e</sub> $2 q$ <sub>norma</sub>li<sub>ze</sub>d <sub>score coor</sub>di<sub>nates.</sub> A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> B<sub>ent</sub>k<sub>us</sub>’<sub>s mu</sub>l<sub>t</sub>i<sub>var</sub>i<sub>ate</sub> B<sub>erry–</sub>E<sub>sseen</sub> b<sub>oun</sub>d f<sub>or convex sets</sub> [28] to the m-coordinate sum and then to the inde<sub>p</sub>endent r-coordinate sum <sub>g</sub>ives error $O _ { q } \left( m ^ { - 1 / 2 } + r ^ { - 1 / 2 } \right)$ Summin<sub>g</sub> over t<sup>h</sup>e $q$ <sup>di</sup>sjo<sup>i</sup>nt common m<sup>i</sup>n<sup>i</sup>m<sup>i</sup>zers g<sup>i</sup>ves t<sup>h</sup>e state<sup>d</sup> rate. <sup>C</sup>on<sup>di</sup>t<sup>i</sup>on<sup>i</sup>ng on t<sup>h</sup>e score pa<sup>i</sup>r o<sup>f</sup> the common minimizer <sub>y</sub>ields e<sub>q</sub>uation (8): all other <sub>p</sub>airs must lie above it, so the joint survival function <sup>a</sup>pp<sup>ears. This</sup> p<sup>roves theorem 7.2.</sup>

For <sub>p</sub>ro<sub>p</sub>osition 7.3, diferentiate e<sub>q</sub>uation (8). At $\rho = 0$

$$
\partial _ { \boldsymbol { \rho } } \Phi _ { \boldsymbol { \rho } } ( x , \gamma ) = x \gamma \Phi ( x ) \Phi ( \gamma ) , \qquad \partial _ { \boldsymbol { \rho } } \overline { { \Phi } } _ { \boldsymbol { \rho } } ( x , \gamma ) = \Phi ( x ) \Phi ( \gamma ) .
$$

S<sub>e</sub>t $\begin{array} { r } { a _ { q } = \int \Phi ( x ) ^ { 2 } \overline { { \Phi } } ( x ) ^ { q - 2 } } \end{array}$ dx. Inte<sub>g</sub>ration b<sub>y</sub> parts <sub>g</sub>ives

$$
\int x \Phi ( x ) { \overline { { \Phi } } } ( x ) ^ { q - 1 } d x = - ( q - 1 ) a _ { q } .
$$

Th<sub>e</sub> fi<sub>rst</sub> dif<sub>erent</sub>i<sub>ate</sub>d <sub>term</sub> <sub>conta</sub>i<sub>ns</sub> <sub>t</sub>h<sub>e</sub> <sub>square</sub> <sub>o</sub>f <sub>t</sub>hi<sub>s</sub> i<sub>ntegra</sub>l<sub>,</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> <sub>secon</sub>d <sub>contr</sub>ib<sub>utes</sub> $q ( q - 1 ) a _ { q } ^ { 2 }$ H<sub>ence</sub> $p _ { q } ^ { \prime } ( 0 ) = q ^ { 2 } ( q - 1 ) a _ { q } ^ { 2 } .$ <sup>.</sup> <sup>A</sup> <sup>Ta</sup>y<sup>lor</sup> <sup>ex</sup>p<sup>ansion</sup> p<sup>roves</sup> <sup>the</sup> p<sup>ro</sup>p<sup>osition.</sup> □

## B.10 Proofoftheorem 7.4

For a <sup>fi</sup>xe<sup>d</sup> ran<sup>k</sup>-m projector an<sup>d</sup> an isotropic Gaussian trip<sup>l</sup>e, t<sup>h</sup>eorem 7.1 gives $\mathbb { E } s _ { A } ( X _ { 1 } ; X _ { 2 } , X _ { 3 } ) = \tau _ { m , d } .$ independently of the orientation of A. Define the symmetric kernel

$$
h _ { A } ( x , \gamma , z ) = \frac { 1 } { 3 } \big ( s _ { A } ( x ; \gamma , z ) + s _ { A } ( \gamma ; x , z ) + s _ { A } ( z ; x , \gamma ) \big ) .
$$

<sup>A</sup> <sup>countin</sup>g <sup>identit</sup>y g<sup>ives</sup>

$$
\overline { { \tau } } _ { n } = \binom { n } { 3 } ^ { - 1 } \sum _ { 1 \leq i < j < k \leq n } h _ { A } ( X _ { i } , X _ { j } , X _ { k } ) .
$$

Th<sub>e</sub> k<sub>erne</sub>l <sub>ta</sub>k<sub>es va</sub>l<sub>ues</sub> i<sub>n</sub> $[ - 1 , 1 ]$ . Hoefdin<sub>g</sub> [19] <sub>p</sub>roved the blockin<sub>g</sub> ine<sub>q</sub>ualit<sub>y</sub> for bounded order-three <sup>U-statistics</sup>; <sup>a</sup>pp<sup>l</sup>y<sup>in</sup>g <sup>it</sup> y<sup>ields</sup>

$$
\operatorname* { P r } \left( \left| \overline { \tau } _ { n } - \tau _ { m , d } \right| > t \mid A \right) \leq 2 \exp \left( - \frac { \lfloor n / 3 \rfloor t ^ { 2 } } { 2 } \right) .
$$

F<sub>or eac</sub>h fi<sub>xe</sub>d <sub>nonzero</sub> $\nu ,$ <sub>t</sub>h<sub>e rat</sub>i<sub>o</sub> $\left\| A \nu \right\| ^ { 2 } / \left\| \nu \right\| ^ { 2 }$ h<sub>as a sca</sub>l<sub>e</sub>d B<sub>eta</sub> l<sub>aw an</sub>d <sub>sat</sub>i<sub>s</sub>fi<sub>es</sub>

$$
\operatorname* { P r } \left( \left| \frac { \left\| A \nu \right\| ^ { 2 } } { \left\| \nu \right\| ^ { 2 } } - 1 \right| > \varepsilon \right) \leq 2 e ^ { - c \varepsilon ^ { 2 } m } .
$$

Conditionin<sub>g</sub> on the data and takin<sub>g</sub> a union bound over the (<sup>n</sup>) diferences <sub>p</sub>roves the stated JL event. Fi<sub>na</sub>ll<sub>y,</sub> <sub>t</sub>h<sub>eorem</sub> 7<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> $\tau _ { m , d } \leq \sqrt { m / d }$ . Substituting the stated t into the U-statistic bound and applying a un<sup>i</sup>on <sup>b</sup>oun<sup>d</sup> w<sup>i</sup>t<sup>h</sup> t<sup>h</sup>e J<sup>L</sup> event comp<sup>l</sup>etes t<sup>h</sup>e proo<sup>f</sup>. □

## B.11 Proofoftheorem 8.1

Mean direction. The Kullback–Leibler divergence $D _ { \mathrm { K L } }$ b<sub>etween</sub> <sub>s</sub>hif<sub>te</sub>d i<sub>sotrop</sub>i<sub>c</sub> G<sub>auss</sub>i<sub>ans</sub> i<sub>s</sub> $\scriptstyle \| \delta \| ^ { 2 } / 2 \sigma ^ { 2 }$ <sup>b</sup>e<sup>f</sup>ore project<sup>i</sup>on an<sup>d</sup> $\| R \boldsymbol { \delta } \| ^ { 2 } / 2 \sigma ^ { 2 }$ <sub>a</sub>f<sub>terwar</sub>d<sub>.</sub> Th<sub>e rat</sub>i<sub>o</sub> $\| R \boldsymbol { \delta } \| ^ { 2 } / \| \boldsymbol { \delta } \| ^ { 2 }$ h<sub>as</sub> di<sub>str</sub>ib<sub>ut</sub>i<sub>on</sub> B<sub>eta</sub> $\textstyle \left( { \frac { m } { 2 } } , { \frac { d - m } { 2 } } \right)$ <sub>an</sub>d mean m/d.

Scalar log-scale. The divergence $\begin{array} { r } { D _ { \mathrm { K L } } = \frac { d } { 2 } \big ( r - 1 - \log r \big ) } \end{array}$ <sub>maps</sub> d<sub>eterm</sub>i<sub>n</sub>i<sub>st</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>to</sub> $\begin{array} { r } { { \frac { m } { 2 } } { \big ( } r - 1 - \log r { \big ) } } \end{array}$ b<sub>ecause</sub> $R \big ( \sigma ^ { 2 } I \big ) R ^ { \top } = \sigma ^ { 2 } I _ { m }$

Difuse traceless covariance shape. Consider $\Sigma _ { \zeta , \varepsilon } ~ = ~ e ^ { \zeta } ( I + \varepsilon H )$ with H symmetric traceless and with $\zeta$ <sub>an</sub> <sub>un</sub>k<sub>nown</sub> <sub>nu</sub>i<sub>sance</sub> <sub>sca</sub>l<sub>e.</sub> A<sub>t</sub> <sub>t</sub>h<sub>e</sub> i<sub>sotrop</sub>i<sub>c</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>t</sub>h<sub>e</sub> <sub>covar</sub>i<sub>ance-tangent</sub> Fi<sub>s</sub>h<sub>er</sub> i<sub>nner</sub> <sub>pro</sub>d<sub>uct</sub> i<sub>s</sub> $\langle A , B \rangle = { \textstyle { \frac { 1 } { 2 } } } \operatorname { t r } ( A B )$ . The original shape tangent H is orthogonal to the scale tangent $I _ { d } .$ . <sup>Af</sup>ter project<sup>i</sup>on, removin<sub>g</sub> t<sup>h</sup>e com<sub>p</sub>onent o<sup>f</sup> $R H R ^ { \dagger }$ <sub>para</sub>ll<sub>e</sub>l <sub>to</sub> <sub>t</sub>h<sub>e</sub> <sub>nu</sub>i<sub>sance</sub> <sub>tangent</sub> $I _ { m }$ <sub>g</sub>i<sub>ves</sub> <sub>t</sub>h<sub>e</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ent</sub> <sub>s</sub>h<sub>ape</sub> <sub>tangent</sub> $\begin{array} { r } { R H R ^ { \top } - \frac { \mathrm { t r } \left( R H R ^ { \top } \right) } { m } I _ { m } . } \end{array}$ <sub>.</sub> F<sub>or</sub> t<sub>r</sub> $H = 0$ , secon<sup>d</sup>-or<sup>d</sup>er Haar inte<sub>g</sub>ration <sub>g</sub>ives

$$
\mathbb E \operatorname { t r } \Bigl ( ( R H R ^ { \top } ) ^ { 2 } \Bigr ) = \frac { m \bigl ( ( m + 1 ) d - 2 \bigr ) } { d { \bigl ( d - 1 \bigr ) } { \bigl ( d + 2 \bigr ) } } \ \| H \| _ { F } ^ { 2 } , \qquad \mathbb E \bigl [ \operatorname { t r } ( R H R ^ { \top } ) ^ { 2 } \bigr ] = \frac { 2 m ( d - m ) } { d { \bigl ( d - 1 \bigr ) } { \bigl ( d + 2 \bigr ) } } \ \| H \| _ { F } ^ { 2 } .
$$

Subtracting 1/m times the second identity from the first yields the traceless moment in theorem 8.1. Dividing t<sup>h</sup>e projecte<sup>d</sup> e<sup>fi</sup>c<sup>i</sup>ent <sup>Fi</sup>s<sup>h</sup>er <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on <sup>b</sup>y $\textstyle { \frac { 1 } { 2 } } \left\| H \right\| _ { F } ^ { 2 }$ <sub>g</sub>ives the exact ratio. E<sub>q</sub>uation (11) follows from the standard Gaussian Fisher-information calculation with known R. □

## B.12 Proofoftheorems A.1, 9.1 and 10.1

Rank. By the singular value decomposition, TZ is an invertible function of $( s _ { i } Z _ { i } ^ { \prime } ) _ { i \leq 1 }$ f<sub>or a rotate</sub>d <sub>stan</sub>d<sub>ar</sub>d G<sub>auss</sub>i<sub>an</sub> $Z ^ { \prime }$ <sub>.</sub> B<sub>ecause</sub> <sub>eac</sub>h $s _ { i }$ is nonzero, o<sup>b</sup>servin<sub>g</sub> $T Z$ is informationally equivalent to observing r <sub>coor</sub>di<sub>nates</sub> <sub>o</sub>f $Z ^ { \prime }$ , so theorem 5.1 applies with r in place of $m .$

Unwhitened squared-norm estimate. With $M = T ^ { \top } T$ <sub>,</sub> G<sub>auss</sub>i<sub>an</sub> <sub>qua</sub>d<sub>rat</sub>i<sub>c-</sub>f<sub>orm</sub> id<sub>ent</sub>i<sub>t</sub>i<sub>es</sub> <sub>g</sub>i<sub>ve</sub> $\mathrm { C o v } ( D , Q ) =$ $2 \operatorname { t r } ( M )$ <sub>an</sub>d $\mathrm { V a r } ( Q ) ~ = ~ 2 \operatorname { t r } ( M ^ { 2 } )$ f<sub>or</sub> $Z \sim \mathcal { N } ( 0 , I _ { d } )$ <sub>.</sub> H<sub>ence</sub> C<sub>orr</sub> $= { \mathrm { t r } } M / { \sqrt { d \mathrm { t r } ( M ^ { 2 } ) } }$ <sub>.</sub> Th<sub>e</sub> K<sub>antorov</sub>i<sub>c</sub>h <sup>ine</sup>q<sup>ualit</sup>y <sup>a</sup>pp<sup>lied</sup> <sup>to</sup> <sup>the</sup> p<sup>ositive</sup> <sup>ei</sup>g<sup>envalues</sup> $s _ { i } ^ { 2 }$ <sub>g</sub>i<sub>ves</sub> <sub>t</sub>h<sub>e</sub> <sub>state</sub>d <sub>two-s</sub>id<sub>e</sub>d b<sub>oun</sub>d<sub>,</sub> <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>e</sub> <sub>extrem</sub>i<sub>zer</sub> p<sup>lacin</sup>g <sup>mass</sup> $\approx 1 / \bigl ( \kappa ^ { 2 } + 1 \bigr )$ at $s _ { \mathrm { m a x } }$ <sub>.</sub> Whi<sub>ten</sub>i<sub>ng</sub> <sub>ma</sub>k<sub>es</sub> $W ^ { \top } W$ t<sup>h</sup>e row-space projector, so $r _ { 2 } ( W ^ { \top } W ) = r .$

Additive-noise recovery. Gaussian conditioning gives recovered variance $s _ { i } ^ { 2 } / \big ( s _ { i } ^ { 2 } + \varepsilon ^ { 2 } \big )$ an<sup>d</sup> <sub>p</sub>osterior variance $\varepsilon ^ { 2 } / \bigl ( s _ { i } ^ { 2 } + \varepsilon ^ { 2 } \bigr )$ i<sub>n</sub> <sub>eac</sub>h <sub>r</sub>i<sub>g</sub>h<sub>t-s</sub>i<sub>ngu</sub>l<sub>ar</sub> di<sub>rect</sub>i<sub>on.</sub> A<sub>verag</sub>i<sub>ng</sub> <sub>t</sub>h<sub>e</sub> <sub>poster</sub>i<sub>or</sub> <sub>var</sub>i<sub>ance</sub> <sub>over</sub> <sub>t</sub>h<sub>e</sub> <sub>quarter-c</sub>i<sub>rc</sub>l<sub>e</sub> l<sub>aw</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> $\Psi _ { \mathrm { e d g e } } ( \varepsilon ) = { \textstyle \frac { \varepsilon } { 2 } } { \big ( } { \sqrt { \varepsilon ^ { 2 } + 4 } } - \varepsilon { \big ) }$

Finite joint identity. Conditional on its input $\gamma ,$ a Gaussian sta<sub>g</sub>e wit<sup>h</sup> entries $N \big ( 0 , 1 / n _ { \ell } \big )$ <sup>out</sup>p<sup>uts</sup> $G _ { \ell } \gamma \sim$ $\mathcal { N } ( 0 , \| \gamma \| ^ { 2 } / n _ { \ell } I _ { n _ { \ell } } )$ <sub>.</sub> Th<sub>us</sub> $\| G _ { \ell } \gamma \| ^ { 2 } = \| \gamma \| ^ { 2 } \cdot \chi _ { n _ { \ell } } ^ { 2 } / n _ { \ell }$ <sub>,</sub> <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>e</sub> $x ^ { 2 }$ f<sub>actor</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>o</sub>f <sub>everyt</sub>hi<sub>ng</sub> <sub>pr</sub>i<sub>or.</sub> H<sub>ence</sub> E $\left[ \left. P _ { L } Z \right. \right. ^ { 2 } \left. \ Z \right] = \left. \left. Z \right. \right. ^ { 2 }$ <sub>,</sub> <sub>an</sub>d <sub>secon</sub>d <sub>moments</sub> <sub>mu</sub>l<sub>t</sub>i<sub>p</sub>l<sub>y:</sub>

$$
\mathbb { E } \left\| P _ { L } Z \right\| ^ { 4 } = \mathbb { E } \left\| Z \right\| ^ { 4 } \prod _ { \ell \geq 1 } ( 1 + 2 / n _ { \ell } ) .
$$

Wi<sub>t</sub>h E $\left\| Z \right\| ^ { 4 } = ( \mathbb { E } \left\| Z \right\| ^ { 2 } ) ^ { 2 } ( 1 + 2 / d )$ , t<sup>h</sup>ese moments <sub>g</sub>ive

$$
\mathrm { V a r } \bigl ( \| P _ { L } Z \| ^ { 2 } \bigr ) = \bigl ( \mathbb { E } \| Z \| ^ { 2 } \bigr ) ^ { 2 } \left[ \prod _ { \ell \geq 0 } \bigl ( 1 + 2 / n _ { \ell } \bigr ) - 1 \right] , \qquad n _ { 0 } = d .
$$

M<sub>oreover,</sub> $\operatorname { C o v } ( \left\| Z \right\| ^ { 2 } , \left\| P _ { L } Z \right\| ^ { 2 } ) = \operatorname { V a r } ( \left\| Z \right\| ^ { 2 } ) = ( \mathbb { E } \left\| Z \right\| ^ { 2 } ) ^ { 2 }$ · 2/d. Their ratio is the stated identit<sub>y</sub>, and $L = 1$ g<sup>ives</sup> $m / { \left( m + d + 2 \right) }$

Typical-map deterministic equivalent. The identities E tr $M = d$ <sub>an</sub>d E <sub>tr</sub> $M ^ { 2 } = d ( m + d + 1 ) / m$ <sub>g</sub>ive t<sup>h</sup>e ratio <sub>o</sub>f<sub>moments</sub> $m \bar { d } / \bigl ( m + d + 1 \bigr )$ ). Concentration of both traces <sub>y</sub>ields $r _ { 2 } = \big ( 1 + o _ { \mathbb { P } } ( 1 ) \big )$ md/(m + d). For multiple l<sub>ayers, t</sub>h<sub>e recurs</sub>i<sub>on</sub>

$$
\mathbb { E } \bigl [ \mathrm { t r } \bigl ( P _ { \ell } ^ { \top } P _ { \ell } \bigr ) ^ { 2 } \mid P _ { \ell - 1 } \bigr ] = \left( 1 + \frac { 1 } { n _ { \ell } } \right) \mathrm { t r } \bigl ( P _ { \ell - 1 } ^ { \top } P _ { \ell - 1 } \bigr ) ^ { 2 } + \frac { 1 } { n _ { \ell } } \left( \mathrm { t r } P _ { \ell - 1 } ^ { \top } P _ { \ell - 1 } \right) ^ { 2 }
$$

<sub>a</sub>dd<sub>s</sub> $1 / n _ { \ell }$ <sub>per</sub> l<sub>ayer</sub> <sub>at</sub> l<sub>ea</sub>di<sub>ng</sub> <sub>or</sub>d<sub>er.</sub> R<sub>an</sub>k<sub>-one</sub> <sub>maps</sub> $( r _ { 2 } \equiv 1 )$ <sub>s</sub>h<sub>ow</sub> <sub>t</sub>h<sub>at</sub> <sub>t</sub>hi<sub>s</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ent</sub> i<sub>s</sub> <sub>not</sub> <sub>a</sub> fi<sub>n</sub>i<sub>te</sub> id<sub>ent</sub>i<sub>ty.</sub> Polar decomposition. Write

$$
\boldsymbol { G } / \sqrt { m } = \left( \boldsymbol { G } \boldsymbol { G } ^ { \top } / d \right) ^ { 1 / 2 } \cdot \sqrt { d / m } \boldsymbol { R } ,
$$

where R is Haar and independent of the Wishart factor. For a fixed x, the Haar stage gives $\begin{array} { r } { { \frac { d } { m } } B _ { \pmb { \imath } } } \end{array}$ <sub>w</sub>i<sub>t</sub>h $B \sim \mathrm { B e t a } { \left( \frac { m } { 2 } , \frac { d - m } { 2 } \right) }$ . <sup>Al</sup>ong t<sup>h</sup>e projecte<sup>d di</sup>rect<sup>i</sup>on, t<sup>h</sup>e square <sup>f</sup>actor contr<sup>ib</sup>utes $u ^ { \top } ( G G ^ { \top } / d ) u \sim \chi _ { d } ^ { 2 } / d .$ <sub>.</sub> Th<sub>e</sub> b<sub>eta–gamma</sub> <sub>a</sub>l<sub>ge</sub>b<sub>ra</sub> <sub>t</sub>h<sub>en</sub> <sub>g</sub>i<sub>ves</sub> $\textstyle { \frac { d } { m } } B \cdot \chi _ { d } ^ { 2 } / d ^ { \frac { d } { = } } \chi _ { m } ^ { 2 } / m$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>t</sub>h<sub>e</sub> f<sub>u</sub>ll <sub>map</sub>’<sub>s</sub> <sub>no</sub>i<sub>se.</sub>

Ensembles. Stacking is a linear map ofrank ≤ min{km, d}, so theorem 6.1 for general Σ and theorem 9.1(i) <sub>app</sub>l<sub>y</sub> <sub>w</sub>i<sub>t</sub>h <sub>t</sub>h<sub>at</sub> <sub>ran</sub>k<sub>.</sub> I<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> G<sub>auss</sub>i<sub>an</sub> bl<sub>oc</sub>k<sub>s</sub> <sub>concatenate</sub> i<sub>nto</sub> <sub>one</sub> G<sub>auss</sub>i<sub>an</sub> <sub>matr</sub>i<sub>x</sub> <sub>w</sub>h<sub>ose</sub> <sub>row</sub> <sub>space</sub> i<sub>s</sub> H<sub>aar, g</sub>i<sub>v</sub>i<sub>ng equa</sub>li<sub>ty a</sub>f<sub>ter w</sub>hi<sub>ten</sub>i<sub>ng.</sub> Th<sub>e average o</sub>f <sub>per-s</sub>k<sub>etc</sub>h <sub>est</sub>i<sub>mates equa</sub>l<sub>s t</sub>h<sub>e unw</sub>hi<sub>tene</sub>d squared-norm estimate from the concatenated km $\times \ d$ Gaussian ma<sub>p</sub>, wit<sup>h</sup> entries $N \big ( \dot { 0 } , 1 / \big ( k m \big ) \big )$ <sub>a</sub>f<sub>ter</sub> <sub>sca</sub>li<sub>ng.</sub> Th<sub>e s</sub>i<sub>ng</sub>l<sub>e-stage</sub> id<sub>ent</sub>i<sub>ty t</sub>h<sub>ere</sub>f<sub>ore g</sub>i<sub>ves corre</sub>l<sub>at</sub>i<sub>on</sub> $\sqrt { { k m } / { \left( { k m + d + 2 } \right) } }$ <sub>over</sub> b<sub>ot</sub>h <sub>map</sub> <sub>an</sub>d d<sub>ata</sub> <sub>ran</sub>d<sub>omness.</sub> Di<sub>v</sub>idi<sub>ng</sub> <sub>t</sub>hi<sub>s</sub> <sub>corre</sub>l<sub>at</sub>i<sub>on</sub> b<sub>y</sub> <sub>t</sub>h<sub>e</sub> HGR <sub>ce</sub>ili<sub>ng</sub> $\sqrt { \operatorname* { m i n } \{ k m , d \} / d }$ <sub>g</sub>i<sub>ves</sub> <sub>t</sub>h<sub>e</sub> <sub>rat</sub>i<sub>o</sub> <sub>state</sub>d i<sub>n</sub> <sub>t</sub>h<sub>eorem</sub> 10<sub>.</sub>1<sub>.</sub>

## Appendix C. A Nonasymptotic Neighborhood Coupling

Th<sub>e</sub> G<sub>auss</sub>i<sub>an score</sub> li<sub>m</sub>i<sub>t</sub> f<sub>or</sub> fi<sub>xe</sub>d <sub>can</sub>did<sub>ate count</sub> i<sub>n t</sub>h<sub>e ma</sub>i<sub>n art</sub>i<sub>c</sub>l<sub>e assumes t</sub>h<sub>at</sub> b<sub>ot</sub>h <sub>coor</sub>di<sub>nate</sub> bl<sub>oc</sub>k<sub>s</sub> grow. In contrast, theorem C.1 covers fixed m and gives an explicit, conservative dependence on the <sub>can</sub>did<sub>ate count.</sub>

Theorem C.1 (Nonasymptotic chance-neighborhood coupling). In the setting of theorem 7.2, let $r =$   
$d - m \geq 2$ and $a = \log ( \dot { 2 } q )$ . Define   
c<sub>r</sub> = √<sub>12π Γ</sub> <sub>(r/2)</sub> 1 Γ ((r − 1)/2) ≍ r<sup>−1/2</sup>   
and   
<sup>∆</sup>q,m,r <sup>=</sup> <sup>min</sup> <sup>q</sup><sub>2</sub>cr (12<sup>√</sup>ma + 16a) .   
There are independent uniformly random k-subsets $S _ { U } , S _ { V } \ o f \{ 1 , . . . , q \}$ , with $S _ { U } = \widetilde { \mathcal { N } } _ { k } .$ , such that   
Pr(N<sub>k</sub> ̸= S<sub>V</sub>) ≤ ∆<sub>q,m,r</sub>.   
Consequently,   
dTVL(Ne<sub>k</sub>, N<sub>k</sub>), L(S<sub>U</sub>, S<sub>V</sub>) ≤ ∆q<sub>,</sub>m<sub>,</sub>r,   
<sup></sup><sub></sub>N<sub>k</sub> ∩ Ne<sub>k</sub><sup></sup><sub></sub> k   
<sup></sup><sub>E</sub> <sup>≤</sup> <sup>∆</sup>q,m,r,   
k q   
and, for k = 1,   
<sup></sup><sub></sub>Pr(N<sub>1</sub> = Ne<sub>1</sub>) − <sup>1</sup><sub>q</sub> <sup></sup><sub></sub> <sup>≤</sup> <sup>∆</sup>q,m,r<sup>.</sup>   
For fixed q and k, $m = o ( d )$ implies $\Delta _ { q , m , d - m }  0 .$

Proof. We decompose $D _ { j } = U _ { j } + V _ { j }$ <sub>as</sub> i<sub>n t</sub>h<sub>e proo</sub>f <sub>o</sub>f <sub>t</sub>h<sub>eorem</sub> 7<sub>.</sub>2<sub>.</sub> L<sub>et</sub> $S _ { U }$ <sub>an</sub>d $S _ { V }$ be the indices of the k smallest coordinates of U and V. They are independent uniform k-subsets. Set

$$
W _ { U } = \operatorname * { m a x } _ { j } U _ { j } - \operatorname * { m i n } _ { j } U _ { j } , \qquad G _ { V , k } = V _ { ( k + 1 ) } - V _ { ( k ) } .
$$

If $G _ { V , k } > W _ { U }$ , then even after adding U, every index in $S _ { V }$ <sub>rema</sub>i<sub>ns</sub> <sub>a</sub>h<sub>ea</sub>d <sub>o</sub>f <sub>every</sub> i<sub>n</sub>d<sub>ex</sub> <sub>outs</sub>id<sub>e</sub> $S _ { V }$ , so $\mathcal { N } _ { k } = \overset { \cdot } { S } _ { V } .$

F<sub>or</sub> $i \neq j ,$

$$
V _ { i } - V _ { j } = \left( Z _ { i } - Z _ { j } \right) ^ { \top } \left( Z _ { i } + Z _ { j } - 2 Z _ { 0 } \right) \stackrel { d } { = } \sqrt { 1 2 } \sum _ { \ell = 1 } ^ { r } g _ { \ell } h _ { \ell } ,
$$

<sub>w</sub>h<sub>ere</sub> $g _ { \ell }$ <sub>an</sub>d $h _ { \ell }$ <sub>are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> <sub>stan</sub>d<sub>ar</sub>d <sub>norma</sub>l <sub>var</sub>i<sub>a</sub>bl<sub>es.</sub> Th<sub>e</sub> <sub>c</sub>h<sub>aracter</sub>i<sub>st</sub>i<sub>c</sub> f<sub>unct</sub>i<sub>on</sub> <sub>o</sub>f <sub>t</sub>hi<sub>s</sub> dif<sub>erence</sub> i<sub>s</sub> $( 1 + 1 2 t ^ { 2 } ) ^ { - r / 2 }$ <sub>.</sub> F<sub>our</sub>i<sub>er</sub> i<sub>nvers</sub>i<sub>on o</sub>f <sub>t</sub>hi<sub>s</sub> f<sub>unct</sub>i<sub>on t</sub>h<sub>ere</sub>f<sub>ore g</sub>i<sub>ves</sub>

$$
\operatorname* { s u p } _ { x } f _ { V _ { i } - V _ { j } } ( x ) \leq \frac { 1 } { 2 \sqrt { 1 2 \pi } } \frac { \Gamma ( ( r - 1 ) / 2 ) } { \Gamma ( r / 2 ) } .
$$

Thi<sub>s</sub> d<sub>ens</sub>i<sub>ty</sub> b<sub>oun</sub>d i<sub>mp</sub>li<sub>es</sub> <sub>t</sub>h<sub>at</sub>

$$
\left. \operatorname* { P r } ( \left| V _ { i } - V _ { j } \right| \leq t ) \leq c _ { r } t , \right. \qquad \operatorname* { P r } ( G _ { V , k } \leq t ) \leq \binom { q } { 2 } c _ { r } t .
$$

Since U and V are independent, conditioning on $W _ { U }$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\begin{array} { r } { \operatorname* { P r } ( G _ { V , k } \leq W _ { U } ) \leq \binom { q } { 2 } c _ { r } \mathbb E W _ { U } . } \end{array}
$$

Mar<sub>g</sub>ina<sup>ll</sup><sub>y</sub>, $U _ { j } \sim 2 \chi _ { m } ^ { 2 }$ . Laurent and Massart [46] <sub>p</sub>roved the concentration ine<sub>q</sub>ualit<sub>y</sub> used here<sub>;</sub> a union b<sub>oun</sub>d <sub>an</sub>d <sub>ta</sub>il i<sub>ntegrat</sub>i<sub>on g</sub>i<sub>ve</sub>

$$
\mathbb { E } W _ { U } \le 1 2 \sqrt { m \log ( 2 q ) } + 1 6 \log ( 2 q ) .
$$

These bounds prove the couplin<sub>g</sub>. Because two independent uniform k-subsets have expected normalized over<sup>l</sup>a<sub>p</sub> $k / q$ <sub>,</sub> <sub>t</sub>h<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>statements</sub> f<sub>o</sub>ll<sub>ow.</sub> □

## Appendix D. Classical Deterministic Ordinal-Capacity Bounds

Whil<sub>e our pro</sub>b<sub>a</sub>bili<sub>st</sub>i<sub>c t</sub>h<sub>eorems</sub> i<sub>n t</sub>h<sub>e ma</sub>i<sub>n art</sub>i<sub>c</sub>l<sub>e concern ran</sub>d<sub>om maps an</sub>d G<sub>auss</sub>i<sub>an c</sub>l<sub>ou</sub>d<sub>s, a c</sub>l<sub>ass</sub>i<sub>ca</sub>l d<sub>eterm</sub>i<sub>n</sub>i<sub>st</sub>i<sub>c</sub> <sub>quest</sub>i<sub>on</sub> <sub>as</sub>k<sub>s</sub> <sub>w</sub>h<sub>et</sub>h<sub>er</sub> <sub>any</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> <sub>can</sub> <sub>rea</sub>li<sub>ze</sub> <sub>an</sub> <sub>ar</sub>bi<sub>trary</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> i<sub>n</sub> l<sub>ow</sub> di<sub>mens</sub>i<sub>on.</sub> F<sub>or</sub> <sub>monotone</sub> E<sub>uc</sub>lid<sub>ean</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ngs,</sub> Bil<sub>u</sub> <sub>an</sub>d Li<sub>n</sub>i<sub>a</sub>l <sub>use</sub>d <sub>s</sub>i<sub>gn-pattern</sub> <sub>met</sub>h<sub>o</sub>d<sub>s</sub> <sub>to</sub> <sub>s</sub>h<sub>ow</sub> <sub>t</sub>h<sub>at</sub> <sub>a</sub>l<sub>most</sub> <sub>every</sub> finite metric re<sub>q</sub>uires linear dimension [47]. We record the corres<sub>p</sub>ondin<sub>g</sub> number of full distance orders <sub>an</sub>d <sub>a</sub> <sub>un</sub>i<sub>versa</sub>l di<sub>mens</sub>i<sub>on</sub> b<sub>oun</sub>d f<sub>or</sub> <sub>rea</sub>li<sub>z</sub>i<sub>ng</sub> <sub>t</sub>h<sub>em.</sub>

L<sub>e</sub>t $\mathcal { O } _ { n , m }$ b<sub>e t</sub>h<sub>e set o</sub>f <sub>str</sub>i<sub>ct tota</sub>l <sub>or</sub>d<sub>ers o</sub>f <sub>t</sub>h<sub>e</sub> $N = { \binom { n } { 2 } }$ pairwise distances induced by n labeled points in $\mathbb { R } ^ { m }$ . Eac<sup>h</sup> com<sub>p</sub>arison $\left\| x _ { i } - x _ { j } \right\| ^ { 2 } > \| x _ { k } - x _ { \ell } \| ^ { 2 }$ is a degree-two polynomial inequality in mn variables, and <sub>t</sub>h<sub>ere</sub> <sub>are</sub> $\binom { N } { 2 } = O \big ( n ^ { 4 } \big )$ <sub>suc</sub>h <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>s.</sub>

Theorem D.1 (Ordinal capacity). (i) The sign-pattern bounds of Warren [48] and Milnor $\left[ 4 9 \right] g i \nu e$

$$
\left. \mathcal { O } _ { n , m } \right. \leq \exp \bigl ( O ( m n \log n ) \bigr ) .
$$

(ii) Almendra-Hernández and Martínez-Sandoval [50] proved that every strict total ordering of the N pairwise distances is realizable in $\mathbb { R } ^ { n - 2 }$ . Their February 2026 correction, prompted by a construction of Maldonado, Raggi Pérez, and Roldán-Pensado [51], leaves open whether every ordering is realizable in $\mathbb { R } ^ { n - 3 }$ . We call the minimum dimension realizing a specified n-point ordering its Almendra-Hernández– Martínez-Sandoval (AHMS) dimension. Thus every such ordering has AHMS dimension at most $n - 2 .$

(iii) Consequently, a uniformly random abstract ordering satisfies

$$
\begin{array} { r } { \operatorname* { P r } ( r e a l i z a b l e i n \mathbb { R } ^ { m } ) \leq \exp [ - \Theta ( n ^ { 2 } \log n ) + O ( m n \log n ) ] , } \end{array}
$$

which is superexponentially small when $m = o ( n )$

Th<sub>e</sub> <sub>max</sub>i<sub>ma</sub>l AHMS di<sub>mens</sub>i<sub>on</sub> <sub>t</sub>h<sub>ere</sub>f<sub>ore</sub> h<sub>as</sub> <sub>or</sub>d<sub>er</sub> $\Theta ( n )$ <sub>,</sub> <sub>a</sub>l<sub>t</sub>h<sub>oug</sub>h i<sub>ts</sub> <sub>exact</sub> <sub>va</sub>l<sub>ue</sub> <sub>rema</sub>i<sub>ns</sub> <sub>open.</sub>

## D.1 Proof of theorem D.1

For <sub>p</sub>art 1, t<sup>h</sup>e $O ( n ^ { 4 } )$ comparison polynomials have degree two in mn variables. Warren’s sign-pattern bound [48] <sub>g</sub>ives $( C s \cdot \dot { 2 } / \dot { ( } m n ) ) ^ { m } = \exp \dot { ( } O \dot { ( } m n \log n ) )$ <sub>rea</sub>li<sub>za</sub>bl<sub>e patterns.</sub> F<sub>or part</sub> 2<sub>,</sub> Al<sub>men</sub>d<sub>ra-</sub>H<sub>ern</sub>á<sub>n</sub>d<sub>ez an</sub>d Martínez-Sandoval [50] established the universal $n - 2$ <sub>rea</sub>li<sub>zat</sub>i<sub>on</sub> b<sub>oun</sub>d<sub>.</sub> F<sub>or</sub> <sub>part</sub> 3<sub>,</sub> $N ! = \exp ( \Theta ( n ^ { 2 } \log n ) )$ abstract orderings are possible; dividing the sign-pattern count by N! proves the probability bound.

Conjecture D.2 (Typical AHMS dimension). For Gaussian point clouds whose ambient dimension is at least proportional to n, the AHMS dimension of the induced strict distance ordering is Ω(n) with high probability. Thus Gaussian orderings would have the same linear-dimensional order as the worst case. More quantitatively, the minimum Kendall distance to $\mathcal { O } _ { n , m }$ should interpolate between the classical counting obstruction and the pairwise m/d law. The counting ratio cannot prove this distribution-specific statement because Gaussian distance orderings are not uniform.

## Appendix E. Reproducibility and Artifact Generation

W<sub>e</sub> h<sub>ost</sub> <sub>t</sub>h<sub>e</sub> <sub>manuscr</sub>i<sub>pt</sub>’<sub>s</sub> <sub>executa</sub>bl<sub>e</sub> <sub>compan</sub>i<sub>on</sub> <sub>at</sub>

## <sup>h</sup>ttps:<sup>//</sup>g<sup>i</sup>t<sup>h</sup>u<sup>b</sup>.com<sup>/</sup>p<sup>i</sup>yus<sup>h</sup>314<sup>/</sup>ran<sup>d</sup>om-project<sup>i</sup>on-geometry.

Thi<sub>s</sub> <sub>compan</sub>i<sub>on</sub> <sub>conta</sub>i<sub>ns</sub> <sub>a</sub> <sub>reusa</sub>bl<sub>e</sub> P<sub>yt</sub>h<sub>on</sub> <sub>pac</sub>k<sub>age,</sub> <sub>a</sub> <sub>t</sub>h<sub>eorem-</sub>l<sub>eve</sub>l <sub>ver</sub>ifi<sub>cat</sub>i<sub>on</sub> h<sub>arness,</sub> <sub>paper-spec</sub>ifi<sub>c</sub> <sub>repro</sub>d<sub>uct</sub>i<sub>on</sub> <sub>programs,</sub> <sub>exper</sub>i<sub>ment</sub> <sub>contracts,</sub> <sub>tutor</sub>i<sub>a</sub>l<sub>s,</sub> <sub>note</sub>b<sub>oo</sub>k<sub>s,</sub> <sub>an</sub>d <sub>a</sub> d<sub>ocumentat</sub>i<sub>on</sub> <sub>s</sub>i<sub>te.</sub> T<sub>o</sub> <sub>ensure</sub> <sub>exact</sub> <sub>repro</sub>d<sub>uc</sub>ibili<sub>ty,</sub> <sub>our</sub> <sub>manuscr</sub>i<sub>pt</sub> <sub>repos</sub>i<sub>tory</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>es</sub> <sub>t</sub>h<sub>e</sub> <sub>compan</sub>i<sub>on</sub> <sub>as</sub> <sub>a</sub> <sub>vers</sub>i<sub>on-p</sub>i<sub>nne</sub>d Gi<sub>t</sub> <sub>su</sub>b<sub>mo</sub>d<sub>u</sub>l<sub>e</sub> un<sup>d</sup>er companion/. <sup>Th</sup>e recor<sup>d</sup>e<sup>d</sup> su<sup>b</sup>mo<sup>d</sup>u<sup>l</sup>e comm<sup>i</sup>t <sup>id</sup>ent<sup>ifi</sup>es t<sup>h</sup>e exact executa<sup>bl</sup>e vers<sup>i</sup>on use<sup>d</sup> <sup>f</sup>or t<sup>h</sup>e <sub>reporte</sub>d <sub>resu</sub>l<sub>ts.</sub>

## E.1 Environment and theorem checks

T<sub>o</sub> i<sub>n</sub>i<sub>t</sub>i<sub>a</sub>li<sub>ze t</sub>h<sub>e p</sub>i<sub>nne</sub>d <sub>compan</sub>i<sub>on an</sub>d i<sub>nsta</sub>ll i<sub>ts</sub> d<sub>eve</sub>l<sub>opment</sub> d<sub>epen</sub>d<sub>enc</sub>i<sub>es, run t</sub>h<sub>ese comman</sub>d<sub>s</sub> f<sub>rom t</sub>h<sub>e</sub> <sup>manuscri</sup>p<sup>t</sup> <sup>root:</sup>

git submodule update --init --recursive

python -m pip install -e "companion[dev]"

T<sub>o run t</sub>h<sub>e t</sub>h<sub>eorem-</sub>l<sub>eve</sub>l h<sub>arness, execute</sub>

python companion/verification/run\_all.py

Th<sub>e</sub> h<sub>arness</sub> <sub>eva</sub>l<sub>uates</sub> <sub>eac</sub>h <sub>pr</sub>i<sub>nc</sub>i<sub>pa</sub>l <sub>t</sub>h<sub>eorem</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub>l<sub>y</sub> <sub>an</sub>d <sub>reports</sub> <sub>a</sub> <sub>pass</sub>/f<sub>a</sub>il <sub>summary.</sub> I<sub>ts</sub> <sub>c</sub>h<sub>ec</sub>k<sub>s</sub> <sub>com</sub>bi<sub>ne</sub> <sub>sym</sub>b<sub>o</sub>li<sub>c</sub> <sub>or</sub> <sub>exact-moment</sub> id<sub>ent</sub>i<sub>t</sub>i<sub>es,</sub> d<sub>eterm</sub>i<sub>n</sub>i<sub>st</sub>i<sub>c</sub> <sub>numer</sub>i<sub>ca</sub>l i<sub>ntegrat</sub>i<sub>on,</sub> <sub>an</sub>d <sub>see</sub>d<sub>e</sub>d M<sub>onte</sub> C<sub>ar</sub>l<sub>o</sub> <sub>exper</sub>i<sub>ments,</sub> <sub>accor</sub>di<sub>ng</sub> <sub>to</sub> <sub>t</sub>h<sub>e</sub> <sub>type</sub> <sub>o</sub>f <sub>c</sub>l<sub>a</sub>i<sub>m.</sub>

## E.2 Manuscript tables and figures

R<sub>egenerate</sub> <sub>t</sub>h<sub>e</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>rows</sub> <sub>use</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>es</sub> 4 <sub>an</sub>d 5 f<sub>rom</sub> <sub>t</sub>h<sub>e</sub> <sub>manuscr</sub>i<sub>pt</sub> <sub>root</sub> b<sub>y</sub> <sub>runn</sub>i<sub>ng</sub>

python companion/reproduction/paper/verify\_claims.py \

--seed 42 --full --output-dir data

python companion/reproduction/paper/verify\_new\_results.py \

--seed 20260812 --full --output-dir data

Th<sub>e</sub> fi<sub>rst</sub> <sub>program</sub> <sub>c</sub>h<sub>ec</sub>k<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>sca</sub>l<sub>ar</sub> <sub>c</sub>h<sub>anne</sub>l<sub>,</sub> <sub>an</sub>i<sub>sotropy,</sub> <sub>an</sub>d G<sub>auss</sub>i<sub>an</sub> <sub>maps.</sub> Th<sub>e</sub> <sub>secon</sub>d <sub>c</sub>h<sub>ec</sub>k<sub>s</sub> b<sub>a</sub>l<sub>ance</sub>d spectra<sup>l</sup> <sup>bl</sup>oc<sup>k</sup>s, t<sup>h</sup>e <sup>G</sup>auss<sup>i</sup>an <sup>i</sup>ntegra<sup>l</sup> <sup>f</sup>or a common m<sup>i</sup>n<sup>i</sup>mum, nearest-ne<sup>i</sup>g<sup>hb</sup>or over<sup>l</sup>ap, an<sup>d</sup> J<sup>L</sup>–<sup>K</sup>en<sup>d</sup>a<sup>ll</sup> <sub>coex</sub>i<sub>stence.</sub> Th<sub>e</sub> <sub>–full</sub> <sub>opt</sub>i<sub>on</sub> <sub>se</sub>l<sub>ects</sub> <sub>t</sub>h<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>s</sub>i<sub>zes</sub> <sub>reporte</sub>d i<sub>n</sub> <sub>t</sub>h<sub>e</sub> <sub>paper;</sub> <sub>om</sub>i<sub>tt</sub>i<sub>ng</sub> i<sub>t</sub> <sub>g</sub>i<sub>ves</sub> <sub>a</sub> f<sub>aster</sub> <sub>smo</sub>k<sub>e</sub> <sub>test.</sub> T<sub>oget</sub>h<sub>er,</sub> <sub>t</sub>h<sub>ese</sub> <sub>programs</sub> <sub>wr</sub>i<sub>te</sub> <sub>t</sub>h<sub>e</sub> <sub>mac</sub>hi<sub>ne-rea</sub>d<sub>a</sub>bl<sub>e</sub> <sub>resu</sub>l<sub>ts.</sub> Th<sub>e</sub> fi<sub>rst</sub> <sub>a</sub>l<sub>so</sub> <sub>generates</sub> <sub>t</sub>h<sub>e</sub> LAT<sub>E</sub>X <sub>rows</sub> <sub>use</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 4<sub>;</sub> <sub>t</sub>h<sub>e</sub> <sub>compact</sub> <sub>summary</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 5 i<sub>s</sub> <sub>ma</sub>i<sub>nta</sub>i<sub>ne</sub>d i<sub>n</sub> <sub>t</sub>h<sub>e</sub> <sub>manuscr</sub>i<sub>pt</sub> <sub>source.</sub>

R<sub>un</sub> <sub>t</sub>h<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ent</sub> G<sub>auss</sub>i<sub>an</sub> <sub>rep</sub>l<sub>acement</sub> <sub>exper</sub>i<sub>ment,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>ca</sub>lib<sub>rat</sub>i<sub>on</sub> <sub>an</sub>d <sub>t</sub>h<sub>e</sub> <sub>exact</sub> <sup>di</sup>sjo<sup>i</sup>nt-pa<sup>i</sup>r $F _ { m , d }$ <sup>com</sup>p<sup>arison</sup> <sup>used</sup> <sup>in</sup> p<sup>ro</sup>p<sup>osition</sup> <sup>3.4</sup>, <sup>with</sup>

python companion/experiments/zero\_information\_jl/run.py \

--profile full --output companion/artifacts/zero\_information\_jl

Thi<sub>s</sub> <sub>scr</sub>i<sub>pt</sub> <sub>wr</sub>i<sub>tes</sub> <sub>t</sub>h<sub>e</sub> <sub>paper</sub> fi<sub>gure,</sub> <sub>t</sub>h<sub>e</sub> <sub>comp</sub>l<sub>ete</sub> <sub>s</sub>i<sub>x-pane</sub>l di<sub>agnost</sub>i<sub>c,</sub> CSV fil<sub>es</sub> f<sub>or</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>tr</sub>i<sub>a</sub>l<sub>s,</sub> <sub>t</sub>h<sub>e</sub> comparison of F ratios, and machine-readable metadata. The commands that build the archival paper bundle <sub>an</sub>d <sub>pu</sub>bli<sub>cat</sub>i<sub>on</sub> fi<sub>gures</sub> i<sub>nvo</sub>k<sub>e</sub> <sub>t</sub>h<sub>e</sub> <sub>same</sub> <sub>exper</sub>i<sub>ment</sub> <sub>automat</sub>i<sub>ca</sub>ll<sub>y.</sub>

T<sub>o</sub> <sub>regenerate</sub> <sub>t</sub>h<sub>e</sub> fi<sub>ve</sub> <sub>pu</sub>bli<sub>cat</sub>i<sub>on</sub> <sub>p</sub>l<sub>ots</sub> di<sub>rect</sub>l<sub>y</sub> i<sub>n</sub> <sub>t</sub>h<sub>e</sub> <sub>manuscr</sub>i<sub>pt</sub> <sub>tree,</sub> <sub>run</sub>

python companion/scripts/generate\_paper\_assets.py \

--output figures/plots

Th<sub>e compan</sub>i<sub>on repos</sub>i<sub>tory a</sub>l<sub>so prov</sub>id<sub>es repro</sub>d<sub>uct</sub>i<sub>on note</sub>b<sub>oo</sub>k<sub>s t</sub>h<sub>at expose t</sub>h<sub>e same ca</sub>l<sub>cu</sub>l<sub>at</sub>i<sub>ons</sub> i<sub>nterac-</sub> <sub>t</sub>i<sub>ve</sub>l<sub>y.</sub> I<sub>ts</sub> d<sub>ocumentat</sub>i<sub>on</sub> <sub>recor</sub>d<sub>s</sub> <sub>t</sub>h<sub>e</sub> <sub>expecte</sub>d <sub>outputs,</sub> <sub>exper</sub>i<sub>ment</sub> <sub>contracts,</sub> d<sub>epen</sub>d<sub>ency</sub> <sub>vers</sub>i<sub>ons,</sub> <sub>an</sub>d <sub>cont</sub>i<sub>nuous-</sub>i<sub>ntegrat</sub>i<sub>on</sub> <sub>c</sub>h<sub>ec</sub>k<sub>s.</sub>

## References

[1] JefJohnson, Matthijs Douze, and Hervé Jégou. “Billion-Scale Similarity Search with GPUs”. In: IEEE Transactions on Big Data 7.3 (2021), pp. 535–547. DOI: 10.1109/TBDATA.2019.2921572.

[2] Hervé Jégou et al. “Searching in One Billion Vectors: Re-rank with Source Coding”. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2011, pp. 861–864. DOI: 10.1109/ICASSP.2011.5946540.

[3] Kevin R. Moon et al. “Visualizing Structure and Transitions in High-Dimensional Biological Data”. In: Nature Biotechnology 37.12 (2019), <sub>pp</sub>. 1482–1492. DOI: 10.1038/s41587-019-0336-3.

[4] Wouter Saelens et al. “A Comparison of Single-Cell Trajectory Inference Methods”. In: Nature Biotechnology 37.5 (2019), pp<sup>.</sup> <sup>547–554.</sup> <sup>DOI:</sup> <sup>10.1038/s41587-019-0071-9.</sup>

[5] Tara Chari and Lior Pachter. “The Specious Art of Single-Cell Genomics”. In: PLOS Computational Biology 19.8 (2023), e1011288. DOI: 10.1371<sup>/</sup>journa<sup>l</sup>.pc<sup>bi</sup>.1011288.

[6] Per-Gunnar Martinsson and Joel A. Tropp. “Randomized numerical linear algebra: Foundations and algorithms”. In: Acta Numerica 29 (2020), pp. 403–572. DOI: 10.1017/S0962492920000021.

[7] Nathan Halko<sub>,</sub> Per-Gunnar Martinsson<sub>,</sub> and Joel A. Tro<sub>pp</sub>. “Findin<sub>g</sub> structure with randomness: Probabilistic al<sub>g</sub>orithms for constructing approximate matrix decompositions”. In: SIAM Review 53.2 (2011), pp. 217–288.

[8] Piotr Ind<sub>y</sub>k and Rajeev Motwani. “A<sub>pp</sub>roximate nearest nei<sub>g</sub>hbors: Towards removin<sub>g</sub> the curse of dimensionalit<sub>y</sub>”. In: Proceedings of the 30th Annual ACM Symposium on Theory of Computing (STOC). 1998, pp. 604–613.

[9] William B. Johnson and Joram Lindenstrauss. “Extensions of Lipschitz mappings into a Hilbert space”. In: Conference on Modern Analysis and Probability. Vol. 26. Contemporary Mathematics. American Mathematical Society, 1984, pp. 189–206.

[10] Noga Alon. “Problems and results in extremal combinatorics—I”. In: Discrete Mathematics 273.1–3 (2003), pp. 31–53. DOI: 10.1016/S0012-365X(03)00227-9.

[11] Kasper Green Larsen and Jelani Nelson. “Optimality of the Johnson–Lindenstrauss lemma”. In: 58th Annual Symposium on Foundations of Computer Science (FOCS). 2017, pp. 633–638. DOI: 10.1109/FOCS.2017.64. arXiv: 1609.02094 [cs.IT].

[12] Kevin Beyer et al. “When is “nearest neighbor” meaningful?” In: International Conference on Database Theory (ICDT). 1999, pp<sup>.</sup> <sup>217–235.</sup>

[13] Robert J. Durrant and Ata Kabán. “When is ‘nearest nei<sub>g</sub>hbour’ meanin<sub>g</sub>ful: A converse theorem and im<sub>p</sub>lications”. In: Journal of Complexity 25.4 (2009), pp. 385–397.

[14] Sanjoy Dasgupta. “Learning mixtures of Gaussians”. In: 40th Annual Symposium on Foundations of Computer Science (FOCS). <sup>1999</sup>, pp<sup>.</sup> <sup>634–644.</sup>

[15] Sanjoy Dasgupta. “Experiments with random projection”. In: Proceedings of the 16th Conference on Uncertainty in Artificial Intelligence (UAI). 2000, pp. 143–151.

[16] Sanjoy Dasgupta, Daniel Hsu, and Nakul Verma. “A concentration theorem for projections”. In: Proceedings of the 22nd Conference on Uncertainty in Artificial Intelligence (UAI). 2006.

[17] Persi Diaconis and David Freedman. “Asymptotics of graphical projection pursuit”. In: The Annals of Statistics 12.3 (1984), pp<sup>.</sup> <sup>793–815.</sup>

[18] Ke Li and Jitendra Malik. “Fast k-nearest neighbour search via Dynamic Continuous Indexing”. In: International Conference on Machine Learning (ICML). 2016, pp. 671–679.

[19] Wassily Hoefding. “Probability inequalities for sums of bounded random variables”. In:Journal ofthe American Statistical Association 58.301 (1963), pp. 13–30. DOI: 10.1080/01621459.1963.10500830.

[20] Anru R. Zhang and Yuchen Zhou. “On the Non-Asymptotic and Sharp Lower Tail Bounds of Random Variables”. In: Stat 9.1 (2020), e314. DOI: 10.1002/sta4.314.

[21] Alfréd Rényi. “On measures of dependence”. In: Acta Mathematica Academiae Scientiarum Hungaricae 10.3–4 (1959), pp. 441– 451<sub>.</sub> DOI<sub>:</sub> 10<sub>.</sub>1007/BF02024507<sub>.</sub>

[22] Amir Dembo, Abram Kagan, and Lawrence A. Shepp. “Remarks on the maximum correlation coeficient”. In: Bernoulli 7.2 (2001), <sub>pp</sub>. 343–350.

[23] Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. 2nd. Wiley, 2006.

[24] Robert C. Grifiths. “The canonical correlation coeficients ofbivariate gamma distributions”. In: The Annals ofMathematical Statistics 40.4 (1969), pp. 1401–1408.

[25] H. S. Witsenhausen. “On sequences of pairs of dependent random variables”. In: SIAM Journal on Applied Mathematics 28.1 (1975), <sub>pp</sub>. 100–113. DOI: 10.1137/0128010.

[26] Maurice G. Kendall. “A new measure ofrank correlation”. In: Biometrika 30.1–2 (1938), pp. 81–93. DOI: 10.1093/biomet/30.1- 2<sub>.</sub>81<sub>.</sub>

[27] William F. She<sub>pp</sub>ard. “On the a<sub>pp</sub>lication of the theor<sub>y</sub> of error to cases of normal distribution and normal correlation”. In: Philosophical Transactions of the Royal Society of London, Series A 192 (1899), pp. 101–167.

[28] V. Bentkus. “On the dependence of the Berry–Esseen bound on dimension”. In: Journal of Statistical Planning and Inference 113.2 (2003), <sub>pp</sub>. 385–402. DOI: 10.1016/S0378-3758(02)00094-0.

[29] H. A. David<sub>,</sub> M. J. O’Connell<sub>,</sub> and S. S. Yan<sub>g</sub>. “Distribution and ex<sub>p</sub>ected value of the rank of a concomitant of an order statistic”. In: The Annals of Statistics 5.1 (1977), pp. 216–223. DOI: 10.1214/aos/1176343756.

[30] Alan M. Frieze and Mark Jerrum. “Im<sub>p</sub>roved a<sub>pp</sub>roximation al<sub>g</sub>orithms for MAX k-CUT and MAX BISECTION”. In: Algorithmica 18.1 (1997), pp. 67–81. DOI: 10.1007/BF02523688.

[31] A. W. van der Vaart. Asymptotic Statistics. Vol. 3. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge Uni<sub>vers</sub>it<sub>y</sub> P<sub>ress,</sub> 1998<sub>. DOI:</sub> 10<sub>.</sub>1017/CBO9780511802256<sub>.</sub>

[32] Moses Charikar. “Similarity estimation techniques from rounding algorithms”. In: Proceedings of the 34th Annual ACM Symposium on Theory of Computing (STOC). 2002, pp. 380–388.

[33] Hervé Jégou, Matthijs Douze, and Cordelia Schmid. “Product quantization for nearest neighbor search”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 33.1 (2011), pp. 117–128.

[34] Tiezheng Ge et al. “Optimized product quantization for approximate nearest neighbor search”. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2013, pp. 2946–2953.

[35] Aditya Kusupati et al. “Matryoshka representation learning”. In: Advances in Neural Information Processing Systems (NeurIPS). 2022<sub>.</sub>

[36] Timothy I. Cannings and Richard J. Samworth. “Random-projection ensemble classification”. In: Journal of the Royal Statistical Society: Series B 79.4 (2017), pp. 959–1035.

[37] Xiaoli Z. Fern and Carla E. Brodle<sub>y</sub>. “Random <sub>p</sub>rojection for hi<sub>g</sub>h dimensional data clusterin<sub>g</sub>: A cluster ensemble a<sub>pp</sub>roach”. In: International Conference on Machine Learning (ICML). 2003, pp. 186–193.

[38] Subhash Khot et al. “Optimal inapproximability results for MAX-CUT and other 2-variable CSPs?” In: SIAMJournal on Computing 37.1 (2007), pp. 319–357. DOI: 10.1137/S0097539705447372.

[39] E. de Klerk, D. V. Pasechnik, and J. P. Warners. “On a<sub>pp</sub>roximate <sub>g</sub>ra<sub>p</sub>h colourin<sub>g</sub> and MAX-k-CUT al<sub>g</sub>orithms based on the ϑ-function”. In:Journal ofCombinatorial Optimization 8.3 (2004), pp. 267–294. DOI: 10.1023/B:JOCO.0000038911.67280.3F.

[40] Jinho Baik<sub>,</sub> Gérard Ben Arous<sub>,</sub> and Sandrine Péché. “Phase transition of the lar<sub>g</sub>est ei<sub>g</sub>envalue for nonnull com<sub>p</sub>lex sam<sub>p</sub>le covariance matrices”. In: The Annals of Probability 33.5 (2005), pp. 1643–1697.

[41] Miloš Radovanović<sub>,</sub> Alexandros Nano<sub>p</sub>oulos<sub>,</sub> and Mirjana Ivanović. “Hubs in s<sub>p</sub>ace: Po<sub>p</sub>ular nearest nei<sub>g</sub>hbors in hi<sub>g</sub>hdimensional data”. In: Journal of Machine Learning Research 11 (2010), pp. 2487–2531.

[42] Boris Hanin and Mihai Nica. “Products of Man<sub>y</sub> Lar<sub>g</sub>e Random Matrices and Gradients in Dee<sub>p</sub> Neural Networks”. In: Communications in Mathematical Physics 376.1 (2020), pp. 287–322. DOI: 10.1007/s00220-019-03624-z.

[43] Vladimir A. Marchenko and Leonid A. Pastur. “Distribution of Ei<sub>g</sub>envalues for Some Sets of Random Matrices”. In: Mathematics of the USSR-Sbornik 1.4 (1967), pp. 457–483. DOI: 10.1070/SM1967v001n04ABEH001994.

[44] Alan Edelman. “Eigenvalues and Condition Numbers of Random Matrices”. In: SIAMJournal on Matrix Analysis and Applications 9.4 (1988), pp. 543–560. DOI: 10.1137/0609045.

[45] Jeesen Chen and Herman Rubin. “Bounds for the Diference Between Median and Mean of Gamma and Poisson Distributions”. In: Statistics & Probability Letters 4.6 (1986), pp. 281–283. DOI: 10.1016/0167-7152(86)90044-1.

[46] B. Laurent and P. Massart. “Adaptive estimation of a quadratic functional by model selection”. In: The Annals ofStatistics 28.5 (2000), <sub>pp</sub>. 1302–1338. DOI: 10.1214/aos/1015957395.

[47] Yonatan Bilu and Nati Linial. “Monotone maps, sphericity and bounded second eigenvalue”. In: Journal of Combinatorial Theory, Series B 95.2 (2005), pp. 283–299. DOI: 10.1016/j.jctb.2005.04.005.

[48] Hugh E. Warren. “Lower bounds for approximation by nonlinear manifolds”. In: Transactions of the American Mathematical Society 133.1 (1968), pp. 167–178.

[49] John Milnor. “On the Betti numbers of real varieties”. In: Proceedings of the American Mathematical Society 15.2 (1964), pp<sup>.</sup> <sup>275–280.</sup>

[50] Víctor Hu<sub>g</sub>o Almendra-Hernández and Leonardo Martínez-Sandoval. “On <sub>p</sub>rescribin<sub>g</sub> total orders and <sub>p</sub>reorders to pairwise distances of points in Euclidean space”. In: Computational Geometry 107 (2022). See arXiv:2111.08895v2 (February 2026) <sup>f</sup>or t<sup>h</sup>e correction to t<sup>h</sup>e s<sup>h</sup>arpness c<sup>l</sup>aim, p. 101898. DOI: 10.1016/j.com<sub>g</sub>eo.2022.101898. arXiv: 2111.08895 [math.CO].

[51] Gerardo L. Maldonado<sub>,</sub> Mi<sub>g</sub>uel Ra<sub>gg</sub>i Pérez<sub>,</sub> and Ed<sub>g</sub>ardo Roldán-Pensado. “Total orders realizable as the distances between two sets of points”. In: Discrete Applied Mathematics 378 (2026), pp. 755–761. DOI: 10.1016/j.dam.2025.10.002. arXiv: <sup>230</sup>4.<sup>05535</sup> [math.CO].