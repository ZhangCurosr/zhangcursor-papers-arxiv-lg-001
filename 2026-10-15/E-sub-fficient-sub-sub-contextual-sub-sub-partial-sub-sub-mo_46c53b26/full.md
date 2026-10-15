# E<sub>fficient</sub> <sub>contextual</sub> <sub>partial</sub> <sub>monitoring</sub> <sub>with</sub> <sub>gen-</sub> eral function approximation

Kapilan Balagopalan, Hao Qin   
D<sub>epa</sub>rtm<sub>e</sub>nt <sub>o</sub>f C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub>r S<sub>c</sub>i<sub>e</sub>n<sub>ce</sub>   
U<sub>n</sub>i<sub>vers</sub>it<sub>y o</sub>f A<sub>r</sub>i<sub>zona</sub>   
T<sub>ucson,</sub> AZ<sub>,</sub> USA   
{kapilanbgp, hqin}@arizona.edu   
Chicheng Zhang   
D<sub>epa</sub>rtm<sub>e</sub>nt <sub>o</sub>f C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub>r S<sub>c</sub>i<sub>e</sub>n<sub>ce</sub>   
U<sub>n</sub>i<sub>vers</sub>it<sub>y</sub> <sub>o</sub>f A<sub>r</sub>i<sub>zona</sub>   
T<sub>ucson,</sub> AZ<sub>,</sub> USA   
chichengz@arizona.edu   
Kwang-Sung Jun   
G<sub>ra</sub>d<sub>ua</sub>t<sub>e</sub> S<sub>c</sub>h<sub>oo</sub>l <sub>o</sub>f A<sub>r</sub>tifi<sub>c</sub>i<sub>a</sub>l I<sub>n</sub>t<sub>e</sub>lli<sub>gence</sub>   
POSTECH   
P<sub>o</sub>h<sub>a</sub>n<sub>g,</sub> S<sub>ou</sub>th K<sub>o</sub>r<sub>ea</sub>   
kwangsungjun@gmail.com

## A<sub>bstract</sub>

We stud<sub>y</sub> finite stochastic <sub>p</sub>artial monitorin<sub>g</sub> (PM) <sub>g</sub>ames with side information, a paradigm of sequential learning with partial feedback. On each round t ∈ [T], b<sub>o</sub>th <sub>p</sub>l<sub>ayer an</sub>d <sub>env</sub>i<sub>ronmen</sub>t <sub>o</sub>b<sub>serve a con</sub>t<sub>ex</sub>t <sub>an</sub>d <sub>p</sub>l<sub>ayer c</sub>h<sub>ooses an ac</sub>ti<sub>on an</sub>d <sub>env</sub>i<sub>ronmen</sub>t <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y c</sub>h<sub>ooses an ou</sub>t<sub>come.</sub> Th<sub>e p</sub>l<sub>ayer ne</sub>ith<sub>er o</sub>b<sub>serves</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come nor</sub> th<sub>e</sub> l<sub>oss;</sub> i<sub>ns</sub>t<sub>ea</sub>d th<sub>ey on</sub>l<sub>y o</sub>b<sub>serve a</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>sym</sub>b<sub>o</sub>l <sub>w</sub>hi<sub>c</sub>h i<sub>s par</sub>ti<sub>a</sub>ll<sub>y</sub> informative about the outcome. We utilize the Estimation-to-Decisions (E2D) metaal<sub>g</sub>orithm of Foster et al. (2021) to desi<sub>g</sub>n E2D.PMSide, a com<sub>p</sub>utationall<sub>y</sub>-eficient f<sub>ramewor</sub>k f<sub>or con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng, w</sub>hi<sub>c</sub>h <sub>uses an on</sub>li<sub>ne regress</sub>i<sub>on orac</sub>l<sub>e</sub> to eficientl<sub>y</sub> estimate the outcome distributions based on historical (context, action, feedback s<sub>y</sub>mbol) tu<sub>p</sub>les. Since the ex<sub>p</sub>loration rule <sub>g</sub>iven b<sub>y</sub> the ori<sub>g</sub>inal E2D <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> i<sub>s</sub> i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> t<sub>o compu</sub>t<sub>e, we propose</sub> t<sub>wo compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y-e</sub>fi<sub>c</sub>i<sub>en</sub>t exploration rules that achieve optimal regret orders in T: in the globall<sub>y</sub> observable setting, our exploration rule is based on ϵ-greed<sub>y</sub>, while in the locall<sub>y</sub> observable settin<sub>g</sub>, our ex<sub>p</sub>loration rule uses inverse <sub>g</sub>a<sub>p</sub> wei<sub>g</sub>htin<sub>g</sub> (Foster & Rakhlin, 2020) <sub>w</sub>hi<sub>c</sub>h <sub>was prev</sub>i<sub>ous</sub>l<sub>y on</sub>l<sub>y</sub> k<sub>nown</sub> t<sub>o</sub> b<sub>e e</sub>f<sub>ec</sub>ti<sub>ve</sub> i<sub>n</sub> th<sub>e</sub> b<sub>an</sub>dit <sub>se</sub>tti<sub>ng.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>a var</sub>i<sub>an</sub>t <sub>o</sub>f E2D<sub>.</sub>PMS<sub>ide name</sub>d OE2D<sub>.</sub>PMS<sub>ide can</sub> b<sub>e com</sub>bi<sub>ne</sub>d <sub>w</sub>ith <sub>o</sub>fli<sub>ne</sub> <sub>regress</sub>i<sub>on orac</sub>l<sub>es</sub> t<sub>o g</sub>i<sub>ve an e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>m w</sub>ith <sub>op</sub>ti<sub>ma</sub>l <sub>regre</sub>t <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>s are</sub> d<sub>rawn</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> f<sub>rom a</sub> fi<sub>xe</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> W<sub>e eva</sub>l<sub>ua</sub>t<sub>e our</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m on a syn</sub>th<sub>e</sub>ti<sub>c</sub> l<sub>a</sub>b<sub>e</sub>l<sub>-e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on game an</sub>d <sub>an</sub> LLM <sub>rou</sub>ti<sub>ng</sub> t<sub>as</sub>k<sub>,</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>a</sub> l<sub>earner</sub> d<sub>ec</sub>id<sub>es</sub> f<sub>rom promp</sub>t<sub>, uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y an</sub>d i<sub>n</sub>t<sub>erna</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> f<sub>ea</sub>t<sub>ures</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> t<sub>o esca</sub>l<sub>a</sub>t<sub>e a query</sub> f<sub>rom a wea</sub>k <sub>mo</sub>d<sub>e</sub>l t<sub>o a s</sub>t<sub>rong one.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>our</sub> <sub>me</sub>th<sub>o</sub>d h<sub>as super</sub>i<sub>or per</sub>f<sub>ormance</sub> th<sub>an pr</sub>i<sub>or approac</sub>h<sub>es.</sub>

## 1 I<sub>ntroduction</sub>

The contextual stochastic artial monitorin roblem (Bartok & Sze esv´ ari, 2012) is an online´ l<sub>earn</sub>i<sub>ng pro</sub>bl<sub>em w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>earner repea</sub>t<sub>e</sub>dl<sub>y</sub> t<sub>a</sub>k<sub>es ac</sub>ti<sub>ons so as</sub> t<sub>o m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> it<sub>s</sub> l<sub>oss</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> <sub>env</sub>i<sub>ronmen</sub>t’<sub>s ar</sub>ti<sub>a</sub>l f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> At th<sub>e</sub> b<sub>e</sub> i<sub>nn</sub>i<sub>n o</sub>f th<sub>e ame,</sub> th<sub>e</sub> l<sub>earner an</sub>d th<sub>e env</sub>i<sub>ronmen</sub>t <sub>are</sub> <sub>g</sub><sup>i</sup>ven a <sub>p</sub>a<sup>i</sup>r $\mathcal { G } = ( L , \Phi )$ of k × d matrices, where k is the number of actions available to the learner and d is the number of outcomes available to the environment. The loss matrix $\pmb { L } \in [ 0 , 1 ] ^ { k \times d }$ <sub>recor</sub>d<sub>s</sub> th<sub>e</sub> l<sub>oss</sub> $L _ { a o }$ incurred when the learner pla<sub>y</sub>s action a and the environment’s outcome is o. Thefeedback <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $\bar { \Phi } \in { \Sigma ^ { k \times d } }$ records what the learner is shown instead: pla<sub>y</sub>ing a against outcome o reveals the f<sub>ee</sub>db<sub>ac</sub>k <sub>s m</sub>b<sub>o</sub>l $\Phi _ { a o }$ f<sub>rom a</sub> fi<sub>n</sub>it<sub>e a</sub>l<sub>p</sub>h<sub>a</sub>b<sub>e</sub>t $\Sigma = \{ 1 , \ldots , m \}$ <sub>.</sub> Th<sub>e</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> l<sub>earner an</sub>d th<sub>e env</sub>i<sub>ronmen</sub>t <sub>procee</sub>d<sub>s over</sub> $T$ rounds. At round t the learner observes a context $x _ { t }$ (which ma<sub>y</sub> be chosen adversariall<sub>y</sub>), sam<sub>p</sub>les an action $A _ { t } \sim p _ { t } \big ( \cdot \mid x _ { t } \big )$ <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> <sub>a</sub> <sub>po</sub>li<sub>cy</sub> $p _ { t } : \mathcal { X } \to \Delta ( k )$ <sub>, an</sub>d th<sub>e</sub> <sub>env</sub>i<sub>ronmen</sub>t d<sub>raws an ou</sub>t<sub>come</sub> $O _ { t }$ f<sub>rom some s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q ^ { * } ( x _ { t } )$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y o</sub>f th<sub>e</sub> l<sub>earner</sub>’<sub>s ran</sub>d<sub>om</sub>i<sub>za</sub>ti<sub>on.</sub> Th<sub>e</sub> l<sub>earner</sub> th<sub>en o</sub>b<sub>serves</sub> th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>sym</sub>b<sub>o</sub>l $\sigma _ { t } = \Phi _ { A _ { t } O _ { i } }$ <sub>an</sub>d i<sub>ncurs</sub> th<sub>e</sub> l<sub>oss</sub> $L _ { A _ { t } O _ { t } }$ <sub>, w</sub>hi<sub>c</sub>h <sub>rema</sub>i<sub>ns</sub> hidd<sub>en.</sub> Th<sub>e goa</sub>l <sub>o</sub>f th<sub>e</sub> l<sub>earner</sub> i<sub>s</sub> t<sub>o m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> it<sub>s regre</sub>t<sub>, w</sub>hi<sub>c</sub>h i<sub>s</sub> th<sub>e</sub> dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> it<sub>s</sub> <sub>expec</sub>t<sub>e</sub>d <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> l<sub>oss</sub> <sub>an</sub>d th<sub>a</sub>t <sub>o</sub>f th<sub>e</sub> b<sub>es</sub>t <sub>po</sub>li<sub>cy</sub> i<sub>n</sub> hi<sub>n</sub>d<sub>s</sub>i<sub>g</sub>ht<sub>:</sub>

$$
\mathrm { R e g } _ { T } : = \sum _ { t = 1 } ^ { T } ( p _ { t } ^ { \intercal } \boldsymbol { L } \boldsymbol { q } ^ { * } ( \boldsymbol { x } _ { t } ) - \underset { \boldsymbol { \lambda } \in \Delta ( k ) } { \operatorname* { m i n } } \boldsymbol { \lambda } ^ { \intercal } \boldsymbol { L } \boldsymbol { q } ^ { * } ( \boldsymbol { x } _ { t } ) ) .
$$

C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> h<sub>as</sub> <sub>many</sub> <sub>app</sub>li<sub>ca</sub>ti<sub>ons,</sub> <sub>suc</sub>h <sub>as</sub> l<sub>earn</sub>i<sub>ng</sub> f<sub>rom</sub> <sub>one-s</sub>id<sub>e</sub>d feedback (Helmbold et al., 2000; Sculle<sub>y</sub>, 2007), d<sub>y</sub>namic <sub>p</sub>ricin<sub>g</sub> (Kleinber<sub>g</sub> & Lei<sub>g</sub>hton, 2003), data <sub>p</sub>rocurement in crowdsourcin<sub>g</sub> (Sin<sub>g</sub>la et al., 2014), and label-eficient <sub>p</sub>rediction (Helmbold et al., 1997). Below we <sub>g</sub>ive two exam<sub>p</sub>les:

Example 1: LLM routing. Modern Large language models (LLM)-based question-answering service o<sub>p</sub>timizes its cost-eficienc b o<sub>p</sub>eratin<sub>g</sub> in a h brid mode (On<sub>g</sub> et al., 2025; Chen et al., 2023; An et al., 2026; Li et al., 2026) : it kee<sub>p</sub>s two LLMs – one weak but chea<sub>p</sub> and one stron<sub>g</sub> but expensive; and given a prompt, it judiciously decides which LLMs to route the prompt to, obtaining its response. At time step t, the service receives a new prompt represented b<sub>y</sub> context $x _ { t } ,$ <sub>, an</sub>d th<sub>e wea</sub>k <sub>mo</sub>d<sub>e</sub>l <sub>genera</sub>t<sub>es a response a</sub>t <sub>neg</sub>li<sub>g</sub>ibl<sub>e cos</sub>t<sub>, an</sub>d th<sub>e rou</sub>t<sub>er</sub> th<sub>en e</sub>ith<sub>er serves</sub> th<sub>e wea</sub>k <sub>response</sub> $( a _ { 1 } )$ <sub>or</sub> <sub>quer</sub>i<sub>es</sub> th<sub>e</sub> <sub>s</sub>t<sub>rong</sub> <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>serves</sub> it<sub>s</sub> <sub>response</sub> $\left( a _ { 2 } \right)$ . The environment’s (hidden) outcome is <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> <sub>wea</sub>k <sub>response</sub> <sub>agrees</sub> <sub>w</sub>ith th<sub>e</sub> <sub>s</sub>t<sub>rong</sub> <sub>one</sub> $\left( o _ { 1 } \right)$ or <sup>d</sup>oes not a<sub>g</sub>ree $\left( o _ { 2 } \right)$ <sub>.</sub> W<sub>e</sub> t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> <sub>s</sub>t<sub>rong</sub> model to be correct and to cost 1 per query, and we incur cost of $\ell _ { 0 1 } > 1$ <sub>w</sub>h<sub>en</sub> <sub>a</sub> <sub>serve</sub>d <sub>wea</sub>k <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> <sub>response</sub> i<sub>s</sub> <sub>wrong.</sub> P<sub>ar</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> <sub>en</sub>t<sub>ers</sub> th<sub>roug</sub>h th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>:</sub> <sub>serv</sub>i<sub>ng</sub> th<sub>e</sub> <sub>wea</sub>k <sub>response</sub> <sub>never</sub> <sub>revea</sub>l<sub>s w</sub>h<sub>e</sub>th<sub>er</sub> it <sub>was correc</sub>t<sub>, so</sub> b<sub>o</sub>th <sub>ou</sub>t<sub>comes y</sub>i<sub>e</sub>ld th<sub>e same sym</sub>b<sub>o</sub>l $\odot ,$ an<sup>d</sup> on<sup>l</sup><sub>y q</sub>uer<sub>y</sub><sup>i</sup>n<sub>g</sub> t<sup>h</sup>e <sub>s</sub>t<sub>rong mo</sub>d<sub>e</sub>l di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>es</sub> th<sub>em.</sub> Th<sub>e game</sub>’<sub>s</sub> l<sub>oss ma</sub>t<sub>r</sub>i<sub>x an</sub>d f<sub>ee</sub>db<sub>ac</sub>k <sub>ma</sub>t<sub>r</sub>i<sub>x are summar</sub>i<sub>ze</sub>d b<sub>e</sub>l<sub>ow:</sub>

$$
\begin{array} { r } { L = \mathbf { \Phi } _ { a _ { 2 } } ^ { a _ { 1 } } \left[ \begin{array} { l l } { \mathbf { \Phi } _ { 0 } ^ { o _ { 1 } } } & { \mathbf { \Phi } _ { \ell _ { 0 1 } } ^ { o _ { 2 } } } \\ { \mathbf { \Phi } _ { 1 } ^ { ( 0 } } & { \mathbf { \Phi } _ { 1 } ^ { \ell _ { 0 1 } } } \\ { \mathbf { \Phi } _ { 1 } ^ { ( 1 } } & { \mathbf { \Phi } _ { 1 } ^ { 1 } } \end{array} \right] , \qquad \Phi = \mathbf { \Phi } _ { a _ { 2 } } ^ { a _ { 1 } } \left[ \begin{array} { l l } { \mathbf { \Phi } _ { \odot } ^ { o _ { 1 } } } & { \mathbf { \Phi } _ { \odot } ^ { o _ { 2 } } } \\ { \mathbf { \Phi } _ { \perp } ^ { \odot } } & { \ddots } \end{array} \right] . } \end{array}
$$

This <sub>g</sub>ame was also referred to as the A<sub>pp</sub>le Tastin<sub>g</sub> <sub>g</sub>ame Helmbold et al. (2000) in earlier works.

Example 2: contextual label eficient prediction game (Lattimore & Szepesvari, 2020; Helmbold´ et al., 1997). Suppose we would like to learn a spam filter online, which predicts whether an email <sup>i</sup>s s<sub>p</sub>am $\left( o _ { 1 } \right)$ or not $\left( o _ { 2 } \right)$ b<sub>ase</sub>d <sub>on</sub> it<sub>s</sub> t<sub>ex</sub>t f<sub>ea</sub>t<sub>ures</sub> $\left( { { x } _ { t } } \right)$ <sub>an</sub>d di<sub>rec</sub>t<sub>s</sub> it t<sub>o</sub> th<sub>e</sub> i<sub>n</sub>b<sub>ox</sub> $\left( a _ { 2 } \right)$ <sub>or spam</sub> f<sub>o</sub>ld<sub>er</sub> $\left( a _ { 3 } \right)$ <sub>.</sub> Th<sub>ere</sub> i<sub>s a</sub>l<sub>so a</sub> thi<sub>r</sub>d <sub>op</sub>ti<sub>on</sub> $( a _ { 1 } )$ <sub>w</sub>h<sub>ere</sub> th<sub>e sys</sub>t<sub>em can as</sub>k th<sub>e rec</sub>i<sub>p</sub>i<sub>en</sub>t/<sub>exper</sub>t f<sub>or c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s accura</sub>t<sub>e</sub> b<sub>u</sub>t i<sub>ncurs a</sub> fi<sub>xe</sub>d <sub>cos</sub>t<sub>.</sub> O<sub>n</sub>l<sub>y w</sub>h<sub>en</sub> th<sub>e</sub> l<sub>earner c</sub>h<sub>ooses ac</sub>ti<sub>on</sub> $a _ { 1 }$ d<sub>oes</sub> it <sub>ge</sub>t t<sub>o see</sub> th<sub>e</sub> t<sub>rue</sub> l<sub>a</sub>b<sub>e</sub>l <sub>o</sub>f th<sub>e ema</sub>il<sub>.</sub> Th<sub>e game</sub>’<sub>s</sub> l<sub>oss ma</sub>t<sub>r</sub>i<sub>x an</sub>d f<sub>ee</sub>db<sub>ac</sub>k <sub>ma</sub>t<sub>r</sub>i<sub>x are summar</sub>i<sub>ze</sub>d b<sub>e</sub>l<sub>ow:</sub>

$$
\begin{array}{c} \begin{array} { r } { \pmb { L } = \ b { a } _ { 2 } } \\ { \pmb { a } _ { 3 } } \end{array} \left[ \begin{array} { r } { o _ { 1 } \pmb { \mathrm { ~ \iota ~ } } ^ { o _ { 2 } } } \\ { \pmb { c } \pmb { \mathrm { ~ \iota ~ } } ^ { c } } \\ { 1 \pmb { \mathrm { ~ \iota ~ } } ^ { 0 } } \\ { 0 \pmb { \mathrm { ~ \iota ~ } } 1 } \end{array} \right] , \qquad \pmb { \Phi } = \ b { a } _ { 2 }  \\ { \pmb { a } _ { 3 } } \end{array} \left[ \begin{array} { r } { o _ { 1 } \pmb { \mathrm { ~ \iota ~ } } ^ { o _ { 2 } } } \\ { \pmb { \mathrm { ~ \iota ~ } } \wedge \ \wedge } \\ { \odot \ \odot } \\ { \odot \ \odot } \end{array} \right] .
$$

D<sub>epen</sub>di<sub>ng on</sub> th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>s</sub>t<sub>ruc</sub>t<sub>ure,</sub> fi<sub>n</sub>it<sub>e s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c par</sub>ti<sub>a</sub>l<sub>-mon</sub>it<sub>or</sub>i<sub>ng games are c</sub>l<sub>ass</sub>ifi<sub>e</sub>d i<sub>n</sub>t<sub>o</sub> four cate<sub>g</sub>ories (classification theorem): trivial, eas<sub>y</sub>, hard, and ho<sub>p</sub>eless <sub>g</sub>ames (Bartok et al., 2011;´ Lattimore & Sze<sub>p</sub>esvari, 2020). In ho<sub>p</sub>eless <sub>g</sub>ames, the feedback is too uninformative to su<sub>pp</sub>ort ´ <sub>mean</sub>i<sub>ng</sub>f<sub>u</sub>l l<sub>earn</sub>i<sub>ng, an</sub>d th<sub>e regre</sub>t <sub>m</sub>i<sub>g</sub>ht <sub>grow</sub> li<sub>near</sub>l<sub>y w</sub>ith th<sub>e</sub> ti<sub>me</sub> h<sub>or</sub>i<sub>zon</sub> $( \Omega ( T ) )$ <sub>.</sub> I<sub>n con</sub>t<sub>ras</sub>t<sub>,</sub> i<sub>n</sub> t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>l <sub>games,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s an ac</sub>ti<sub>on w</sub>hi<sub>c</sub>h i<sub>s op</sub>ti<sub>ma</sub>l f<sub>or every ou</sub>t<sub>come; consequen</sub>tl<sub>y,</sub> th<sub>ey can</sub> b<sub>e</sub> <sub>so</sub>l<sub>ve</sub>d <sub>w</sub>ith <sub>zero</sub> <sub>regre</sub>t<sub>.</sub> Th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> <sub>cases</sub> <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> <sub>more</sub> <sub>re</sub>fi<sub>ne</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>are</sub> <sub>easy</sub> $( \Theta ( { \sqrt { T } } ) )$ ) and hard g<sup>ames</sup> $\left( \Theta ( T ^ { 2 / 3 } ) \right)$ ). Both classes satisf<sub>y g</sub>lobal observabilit<sub>y</sub>, which we define shortl<sub>y</sub>. In addition, <sub>easy games sa</sub>ti<sub>s</sub>f<sub>y</sub> l<sub>oca</sub>l <sub>o</sub>b<sub>serva</sub>bilit<sub>y, w</sub>h<sub>ereas</sub> h<sub>ar</sub>d <sub>games</sub> d<sub>o no</sub>t<sub>.</sub>

Related work: partial monitoring with side information. We adopt the contextual partial monitorin<sub>g</sub> settin<sub>g</sub> of Bartok et al. (2012a); Heuillet et al. (2024), and use the terms´ contextual <sub>p</sub>artial monitoring and partial monitoring with side information interchangeably for it. The classification of eas<sub>y</sub> and hard <sub>g</sub>ames established for finite stochastic non-contextual <sub>p</sub>artial monitorin<sub>g</sub> (Bartok et al.,´ 2011; Lattimore & Sze<sub>p</sub>esvari, 2020) extends to this settin<sub>g</sub>, at least in the minimax re<sub>g</sub>ret’s de<sub>p</sub>endence´ on $T .$ . First, the non-contextual lower bounds of Bartok et al. (2011) transfer immediatel<sub>y</sub>, since a´ <sub>non-con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>game</sub> <sub>can</sub> b<sub>e</sub> <sub>v</sub>i<sub>ewe</sub>d <sub>as</sub> <sub>a</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>game</sub> <sub>w</sub>ith <sub>a</sub> d<sub>ummy</sub> <sub>con</sub>t<sub>ex</sub>t<sub>.</sub> S<sub>econ</sub>d<sub>,</sub> f<sub>or</sub> th<sub>e</sub> <sub>upper</sub> bounds, Foster et al. (2022) shows the equivalence between the Decision-Estimation Coeficient Foster et al. (2021) and the ex<sub>p</sub>loration-b<sub>y</sub>-o<sub>p</sub>timization (ExO) objective of Lattimore & Sze<sub>p</sub>esvari (2020),´ which im<sub>p</sub>lies that the contextual version of the Estimation-to-Decision al<sub>g</sub>orithm (Foster et al., 2021, Section 8) achieve re<sub>g</sub>rets that match o<sub>p</sub>timal rates <sub>g</sub>iven b<sub>y</sub> ExO. These o<sub>p</sub>timal statistical <sub>g</sub>uarantees <sub>come a</sub>t <sub>a compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>cos</sub>t<sub>,</sub> h<sub>owever:</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m a</sub>tt<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>em se</sub>l<sub>ec</sub>t<sub>s eac</sub>h <sub>ac</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> b<sub>y</sub> solvin<sub>g</sub> a minimax <sub>p</sub>roblem at ever<sub>y</sub> round, which is not known to be tractable (see Section 2.3 for further discussion). Furthermore, our work is closel<sub>y</sub> related to Lattimore & Sze<sub>p</sub>esvari (2019;´ 2020); Lattimore (2022); Lattimore & G<sub>y</sub>or<sub>gy</sub> (2021) and Foster et al. (2022). A detailed discussion i<sub>s</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> S<sub>ec</sub>ti<sub>on</sub> $\mathbf { A } .$

M<sub>eanw</sub>hil<sub>e, compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> h<sub>ave</sub> b<sub>een propose</sub>d f<sub>or con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng,</sub> b<sub>y</sub> i<sub>mpos</sub>i<sub>ng s</sub>t<sub>rong assump</sub>ti<sub>ons on</sub> th<sub>e</sub> f<sub>unc</sub>ti<sub>on c</sub>l<sub>ass</sub> th<sub>a</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> th<sub>e</sub> t<sub>rue ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q ^ { * }$ . Utilizin<sub>g g</sub>eneralized linear models for modelin<sub>g</sub> (functions of) $q ^ { * }$ <sub>,</sub> B<sub>ar</sub>t<sub>o</sub>k <sub>e</sub>t <sub>a</sub>l<sub>.</sub>´ (2012a) <sub>p</sub>ro<sub>p</sub>osed CBPSide, a deterministic al<sub>g</sub>orithm based on confidence intervals that extends UCB<sub>-s</sub>t<sub>y</sub>l<sub>e me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>rom</sub> th<sub>e</sub> b<sub>an</sub>dit <sub>se</sub>tti<sub>ng;</sub> it <sub>app</sub>li<sub>es on</sub>l<sub>y</sub> t<sub>o easy games, w</sub>h<sub>ere</sub> it <sub>ac</sub>hi<sub>eves</sub> $O ( \sqrt { T } )$ re<sub>g</sub>ret. Heuillet et al. (2024) later introduced CBPSide and its randomized counter<sub>p</sub>art $\mathrm { R a N D C B P S I D E } ^ { * }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>cover</sub> b<sub>o</sub>th <sub>c</sub>l<sub>asses,</sub> <sub>a</sub>tt<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> $O ( \sqrt { T } )$ <sup>re</sup>g<sup>ret</sup> <sup>on</sup> <sup>eas</sup>y g<sup>ames</sup> <sup>and</sup> $O ( T ^ { 2 / 3 } )$ on hard ones. Kirschner et al. (2023) studies the linear <sub>p</sub>artial monitorin<sub>g</sub> and its contextual extension, <sub>w</sub>hi<sub>c</sub>h <sub>genera</sub>li<sub>zes</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l fi<sub>n</sub>it<sub>e</sub> <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng;</sub> h<sub>owever,</sub> th<sub>ey</sub> <sub>requ</sub>i<sub>re</sub> th<sub>a</sub>t th<sub>e</sub> <sub>mapp</sub>i<sub>ngs</sub> f<sub>rom</sub> (context, action) to the ex<sub>p</sub>ected reward and the si<sub>g</sub>nal feedback to be (context-inde<sub>p</sub>endent) linear f<sub>unc</sub>ti<sub>ons.</sub>

Thi<sub>s mo</sub>ti<sub>va</sub>t<sub>es</sub> th<sub>e ques</sub>ti<sub>on</sub> th<sub>a</sub>t d<sub>r</sub>i<sub>ves our wor</sub>k<sub>: are</sub> th<sub>ere compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> th<sub>a</sub>t <sub>suppor</sub>t <sub>genera</sub>l f<sub>unc</sub>ti<sub>on approx</sub>i<sub>ma</sub>ti<sub>on an</sub>d <sub>a</sub>tt<sub>a</sub>i<sub>n</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>ra</sub>t<sub>es</sub>? I<sub>n</sub> thi<sub>s paper, we answer</sub> thi<sub>s ques</sub>ti<sub>on a</sub>fi<sub>rma</sub>ti<sub>ve</sub>l<sub>y.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y, we ma</sub>k<sub>e</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons:</sub> Contributions. (1) we propose an algorithmic framework E2D.PMSide that reduces stochastic contextual <sub>p</sub>artial monitorin<sub>g</sub> to online re<sub>g</sub>ression. It is based on the E2D (Estimation to Decision) framework (Foster et al., 2021), where the al<sub>g</sub>orithm aims at findin<sub>g</sub> an ex<sub>p</sub>loration distribution that certifies a low Decision-Estimation Coeficient (See Section 2.3 for its definition); our ke<sub>y</sub> observation is that this can be done by optimizing a surrogate objective that trades of between greediness <sub>an</sub>d <sub>coverage, w</sub>hi<sub>c</sub>h <sub>paves</sub> th<sub>e way</sub> f<sub>or</sub> th<sub>e</sub> d<sub>es</sub>i<sub>gn o</sub>f <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l monitorin<sub>g</sub>. (2) Based on the above relaxed objective, we instantiate E2D.PMSide in the <sub>g</sub>loball<sub>y</sub> observable and locall<sub>y</sub> observable games to obtain ϵ-Greedy.PMSide and SquareCB.PMSide <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>ac</sub>hi<sub>ev</sub>i<sub>ng</sub> <sub>regre</sub>t <sub>o</sub>f <sub>op</sub>ti<sub>ma</sub>l <sub>or</sub>d<sub>er</sub> i<sub>n</sub> $T$ i<sub>n</sub> th<sub>e</sub>i<sub>r</sub> <sub>respec</sub>ti<sub>ve</sub> <sub>se</sub>tti<sub>ngs.</sub> Th<sub>ese,</sub> t<sub>o</sub> our knowled e, are the first com utationall eficient contextual artial monitorin al orithms with <sub>g</sub>eneralfunction a<sub>pp</sub>roximation. (3) When the contexts are iid, we <sub>p</sub>ro<sub>p</sub>ose a variant of E2D.PMSide <sub>name</sub>d OE2D<sub>.</sub>PMS<sub>ide, w</sub>hi<sub>c</sub>h i<sub>s an e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng a</sub>l<sub>gor</sub>ith<sub>m</sub> b<sub>ase</sub>d <sub>on o</sub>fli<sub>ne regress</sub>i<sub>ons, a compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>m</sub>iti<sub>ve</sub> th<sub>a</sub>t <sub>can a</sub>d<sub>m</sub>it <sub>s</sub>i<sub>mp</sub>l<sub>er</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>ons</sub> i<sub>n prac</sub>ti<sub>ce.</sub> (4) We evaluate SquareCB.PMSide in two contextual <sub>p</sub>artial monitorin<sub>g</sub> tasks: label-eficient <sub>pre</sub>di<sub>c</sub>ti<sub>on an</sub>d LLM <sub>rou</sub>ti<sub>ng, an</sub>d <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t it <sub>ou</sub>t<sub>per</sub>f<sub>orms severa</sub>l b<sub>ase</sub>li<sub>nes.</sub>

## 2 P<sub>reliminaries</sub>

Basic Notation. We write 0 and 1 for the all-zeros and all-ones column vectors, $e _ { 1 } , \ldots , e _ { k }$ f<sub>or</sub> th<sub>e</sub> standard basis vectors, and bold s<sub>y</sub>mbols for matrices. For a vector or matrix A, $\left\| A \right\| _ { \infty } = \operatorname* { m a x } _ { i j } \left| A _ { i j } \right|$ i<sub>s</sub> th<sub>e</sub> <sub>en</sub>t<sub>ryw</sub>i<sub>se</sub> <sub>max</sub>i<sub>mum</sub> <sub>norm,</sub> $M _ { i }$ denotes the vector generated b<sub>y</sub> i-th row, and $M _ { i j }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> element at i-th row and $j \mathrm { - t h }$ column. For two distributions u and $v , D _ { \mathrm { H } } ^ { 2 } \big ( u \big | v \big )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub>i<sub>r square</sub>d H<sub>e</sub>lli<sub>nger</sub> di<sub>s</sub>t<sub>ance.</sub> L<sub>e</sub>t $\mathrm { c o } (  { \boldsymbol { S } } )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e convex</sub> h<sub>u</sub>ll <sub>o</sub>f <sub>a se</sub>t $s$ <sub>.</sub> F<sub>or a su</sub>b<sub>se</sub>t $A \subseteq [ k ]$ <sub>we wr</sub>it<sub>e</sub> $\Delta ( \mathcal { A } ) \dot { : = } \left\{ \lambda \in \Delta ( k ) : \lambda _ { b } = 0 \right.$ f<sub>or a</sub>ll $b \notin A \}$ f<sub>or</sub> th<sub>e</sub> f<sub>ace o</sub>f $\Delta ( k )$ supported on A. Elements of $\Delta ( \mathcal { A } )$ remain k-dimensional vectors; the<sub>y</sub> simpl<sub>y</sub> place no mass outside $\bar { \mathcal { A } }$

Signal matrix. (Komiyama et al. (2015)) Given a feedback matrix $\Phi ,$ <sub>,</sub> it i<sub>s conven</sub>i<sub>en</sub>t t<sub>o</sub> d<sub>escr</sub>ib<sub>e</sub> the feedback of a single action as a linear map. The signal matrix of action a is the $m \times d$ <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $S _ { a }$ <sub>w</sub>ith $S _ { a } [ \sigma , o ] = 1 \ \{ \stackrel { \bf { \bar { \Phi } } } { \Phi _ { a o } } = \sigma \}$ <sub>.</sub> W<sub>r</sub>iti<sub>ng</sub> $\Delta ( n )$ for the probabilit<sub>y</sub> simplex on n elements, if the outcome i<sub>s</sub> d<sub>rawn</sub> f<sub>rom</sub> $q \in \Delta ( d )$ th<sub>en</sub> $A _ { a } q \in \Delta ( \dot { m } )$ i<sub>s prec</sub>i<sub>se</sub>l<sub>y</sub> th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f th<sub>e sym</sub>b<sub>o</sub>l <sub>o</sub>b<sub>serve</sub>d <sub>on</sub> pla<sub>y</sub>ing a. Two candidate outcome distributions $q _ { 1 } , q _ { 2 }$ are therefore distinguishable through action a <sub>exac</sub>tl<sub>y</sub> t<sub>o</sub> th<sub>e ex</sub>t<sub>en</sub>t th<sub>a</sub>t $S _ { a } q _ { 1 }$ <sub>an</sub>d $S _ { a } q _ { 2 }$ difer; and it is this quantit<sub>y</sub>, rather than the gap between q<sub>1</sub> <sub>an</sub>d $q _ { 2 }$ th<sub>emse</sub>l<sub>ves,</sub> th<sub>a</sub>t <sub>governs</sub> <sub>w</sub>h<sub>a</sub>t th<sub>e</sub> l<sub>earner</sub> <sub>can</sub> h<sub>ope</sub> t<sub>o</sub> id<sub>en</sub>tif<sub>y.</sub> R<sub>eca</sub>ll th<sub>a</sub>t i<sub>n</sub> <sub>our</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng pro</sub>bl<sub>em, eac</sub>h <sub>roun</sub>d i<sub>s prece</sub>d<sub>e</sub>d b<sub>y a con</sub>t<sub>ex</sub>t $x _ { t }$ drawn from a set X, <sub>an</sub>d th<sub>e ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> d<sub>epen</sub>d<sub>s on</sub> it<sub>: ou</sub>t<sub>comes</sub> f<sub>o</sub>ll<sub>ow</sub> $q ^ { * } ( \bar { x _ { t } } )$ f<sub>or an un</sub>k<sub>nown</sub> $q ^ { * } : \mathcal { X } \to \Delta ( d )$ Th<sub>e</sub> l<sub>earner</sub> i<sub>s g</sub>i<sub>ven a mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>ass</sub> $\mathcal { Q } \subseteq \{ q : \mathcal { X } \to \Delta ( d ) \}$ }, which we assume is well-specified.

Assumption 2.1 (Realizability). The model class $\mathcal { Q }$ <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> t<sub>rue ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q ^ { * }$

## 2<sub>.</sub>1 S<sub>tructure</sub> <sub>of</sub> <sub>partial</sub> <sub>monitoring</sub> <sub>games</sub>

W<sub>e</sub> <sub>now</sub> <sub>rev</sub>i<sub>ew</sub> <sub>some</sub> b<sub>as</sub>i<sub>c</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> <sub>o</sub>f <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> <sub>games</sub> <sub>revea</sub>l<sub>e</sub>d i<sub>n</sub> <sub>pr</sub>i<sub>or</sub> <sub>wor</sub>k<sub>s.</sub>

Definition 2.1 (Cell decomposition, Bartok et al. (2012b))´ . The cell $\scriptstyle { { \mathcal { C } } _ { a } }$ of action a is defined as the subset of outcome distributions where action a is optimal. Formall<sub>y</sub>, $\mathcal { C } _ { a } = \{ q \in \Delta ( d ) : ( e _ { a } - e _ { b } ) ^ { \top } L q \leq$ $0 , \forall b \in [ k ] \}$ . The collection $\{ { \mathcal { C } } _ { a } : a \in [ k ] \}$ i<sub>s ca</sub>ll<sub>e</sub>d th<sub>e ce</sub>ll d<sub>ecompos</sub>iti<sub>on.</sub>

Neighborhood graph (Lattimore & Szepesvari, 2020).´ Cells that meet in a face of dimension $d - 2$ corres ond to actions that can be o timal simultaneousl , and this adjacenc structure or anizes the analysis that follows. Call an action Pareto optimal when its cell is full dimensional, dim $( \bar { \mathcal { C } } _ { a } ) = d - 1$ taking the dimension of a polytope to be that of its afine hull, and write Π for the set of such actions t<sub>o e</sub>th<sub>er w</sub>ith $\Delta ( \Pi ) : = \{ \bar { \lambda } \in \bar { \Delta } ( \bar { k } ) : \lambda _ { b } = 0$ f<sub>or a</sub>ll $b \notin \Pi \}$ f<sub>or</sub> th<sub>e correspon</sub>di<sub>ng</sub> f<sub>ace o</sub>f $\Delta ( k )$ <sub>.</sub> Th<sub>e</sub> remaining actions are of two kinds: a is degenerate when $\mathcal { C } _ { a } \neq \emptyset$ but dim $\bar { ( \mathcal { C } _ { a } ) } \leq \bar { d } - 2$ <sub>, an</sub>d d<sub>om</sub>i<sub>na</sub>t<sub>e</sub>d <sub>w</sub>h<sub>en</sub> $\mathcal { C } _ { a } = \varnothing$ . Since dim $( \Delta ( d ) ) = d - \bar { 1 }$ <sub>,</sub> <sub>ne</sub>ith<sub>er</sub> ki<sub>n</sub>d i<sub>s</sub> <sub>ever</sub> <sub>un</sub>i<sub>que</sub>l<sub>y</sub> <sub>op</sub>ti<sub>ma</sub>l<sub>;</sub> <sub>eac</sub>h i<sub>s</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>on</sub>l<sub>y on a se</sub>t <sub>o</sub>f <sub>measure zero.</sub> A <sub>game w</sub>ith <sub>no</sub> d<sub>egenera</sub>t<sub>e ac</sub>ti<sub>ons</sub> i<sub>s ca</sub>ll<sub>e</sub>d <sub>non-</sub>d<sub>egenera</sub>t<sub>e, an</sub>d t<sub>wo ac</sub>ti<sub>ons are</sub> d<sub>up</sub>li<sub>ca</sub>t<sub>es w</sub>h<sub>en</sub> th<sub>e</sub>i<sub>r</sub> l<sub>oss rows co</sub>i<sub>nc</sub>id<sub>e,</sub> $( e _ { a } - e _ { b } ) ^ { \intercal } \mathbf { { L } } = \mathbf { { 0 } }$ T<sub>wo</sub> P<sub>a</sub>r<sub>e</sub>t<sub>o op</sub>tim<sub>a</sub>l <sub>ac</sub>ti<sub>ons are ne</sub>i<sub>g</sub>hb<sub>ors w</sub>h<sub>en</sub> th<sub>e</sub>i<sub>r ce</sub>ll<sub>s mee</sub>t i<sub>n a</sub> f<sub>ace o</sub>f th<sub>e</sub> l<sub>arges</sub>t di<sub>mens</sub>i<sub>on s</sub>h<sub>or</sub>t <sub>o</sub>f <sub>co</sub>i<sub>nc</sub>id<sub>ence,</sub> dim $( \mathcal { C } _ { a } \cap \mathcal { C } _ { b } ) \stackrel { \cdot } { = } d - 2$ <sub>;</sub> th<sub>e</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>on reac</sub>h<sub>es</sub> di<sub>mens</sub>i<sub>on</sub> $d - 1$ only for duplicates. Writing E for the resulting edge set, (Π, E) is the neighborhood graph of $\mathcal { G }$

## 2<sub>.</sub>2 O<sub>bservability</sub>

I<sub>n par</sub>ti<sub>a</sub>l<sub>-mon</sub>it<sub>or</sub>i<sub>ng games,</sub> th<sub>e</sub> l<sub>earner o</sub>b<sub>serves</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e</sub> l<sub>osses or rewar</sub>d<sub>s assoc</sub>i<sub>a</sub>t<sub>e</sub>d with its actions. Therefore, a central challen<sub>g</sub>e is to infer (diferences of) losses of actions from the <sub>o</sub>b<sub>serve</sub>d f<sub>ee</sub>db<sub>ac</sub>k <sub>sym</sub>b<sub>o</sub>l<sub>.</sub> Wh<sub>e</sub>th<sub>er</sub> thi<sub>s</sub> i<sub>s poss</sub>ibl<sub>e</sub> d<sub>epen</sub>d<sub>s on</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erp</sub>l<sub>ay</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> l<sub>oss ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>an</sub>d th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>ma</sub>t<sub>r</sub>i<sub>x.</sub>

Definition 2.2 (Global observability, adapted from Bartok et al. (2012b) and Lattimore & Szepesv ´ ari´ (2020)). A <sub>p</sub>artial monitorin<sub>g</sub> <sub>g</sub>ame is <sub>g</sub>loball<sub>y</sub> observable if, for ever<sub>y</sub> <sub>p</sub>air of actions $( a , b ) \in [ \dot { k } ] \times [ k ]$ there exists a witnessfunction $w _ { ( a , b ) } : [ k ] \times \Sigma  \mathbb { R }$ <sub>suc</sub>h th<sub>a</sub>t

$$
L _ { a o } - L _ { b o } = \sum _ { c = 1 } ^ { k } w _ { ( a , b ) } ( c , \Phi _ { c o } ) , \qquad { \mathrm { f o r ~ a l l } } o \in [ d ] .
$$

Definition 2.3 (Local observability, Lattimore & Szepesvari (2020))´ . A globally observable, nondegenerate partial monitoring game is locally observable if for every edge $e = ( a , b ) \in \mathcal { E }$ <sub>o</sub>f th<sub>e</sub> <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d <sub>grap</sub>h th<sub>e</sub> <sub>w</sub>it<sub>ness</sub> f<sub>unc</sub>ti<sub>on</sub> $w _ { e }$ <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>2 <sub>may</sub> b<sub>e</sub> <sub>c</sub>h<sub>osen</sub> t<sub>o</sub> b<sub>e</sub> <sub>suppor</sub>t<sub>e</sub>d <sub>on</sub>l<sub>y</sub> <sub>on</sub> th<sub>e en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>s,</sub> th<sub>a</sub>t i<sub>s</sub> $w _ { e } ( c , \cdot ) \equiv 0$ f<sub>or a</sub>ll $c \notin \{ a , b \}$ <sub>, so</sub> th<sub>a</sub>t

$$
L _ { a o } - L _ { b o } = \sum _ { c \in \{ a , b \} } w _ { e } ( c , \Phi _ { c o } ) , \qquad { \mathrm { f o r ~ a l l } } o \in [ d ] .
$$

B<sub>y</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on,</sub> <sub>every</sub> l<sub>oca</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>game</sub> i<sub>s</sub> <sub>a</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>game.</sub> U<sub>n</sub>d<sub>er</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>o</sub>b<sub>serva</sub>bilit<sub>y,</sub> th<sub>e</sub> l<sub>oss</sub> dif<sub>erence</sub> $L _ { a o } - L _ { b o }$ <sub>may</sub> d<sub>epen</sub>d <sub>on</sub> f<sub>ee</sub>db<sub>ac</sub>k f<sub>rom a</sub>ll <sub>ac</sub>ti<sub>ons</sub> $c \in [ k ]$ <sub>, w</sub>h<sub>ereas un</sub>d<sub>er</sub> l<sub>oca</sub>l <sub>o</sub>b<sub>serva</sub>bilit<sub>y,</sub> it <sub>can</sub> b<sub>e expresse</sub>d <sub>us</sub>i<sub>ng on</sub>l<sub>y</sub> th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k f<sub>rom</sub> th<sub>e sequence o</sub>f <sub>ac</sub>ti<sub>ons</sub> i<sub>n</sub> $\mathcal { E }$ connect<sup>i</sup>n<sub>g</sub> a and b. Consequentl<sub>y</sub>, locall<sub>y</sub> observable games permit more direct loss-diference estimation than <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e games</sub> i<sub>n genera</sub>l<sub>.</sub> Si<sub>nce</sub> t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>l <sub>games an</sub>d h<sub>ope</sub>l<sub>ess games are no</sub>t i<sub>n</sub>t<sub>eres</sub>ti<sub>ng</sub> f<sub>rom an a</sub>l<sub>gor</sub>ith<sub>m</sub> d<sub>es</sub>i<sub>gn perspec</sub>ti<sub>ve,</sub> f<sub>or</sub> th<sub>e rema</sub>i<sub>n</sub>d<sub>er o</sub>f th<sub>e paper we</sub> f<sub>ocus on g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e</sub> g<sup>ames</sup>.

Assumption 2.2 (Global observability and non-degeneracy; Lattimore & Szepesvari (2020))´ . We <sub>assume</sub> th<sub>a</sub>t th<sub>e par</sub>ti<sub>a</sub>l<sub>-mon</sub>it<sub>or</sub>i<sub>ng game</sub> $\mathcal { G } = ( L , \breve { \Phi } )$ i<sub>s g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e, non-</sub>d<sub>egenera</sub>t<sub>e, an</sub>d <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> <sub>no</sub> d<sub>up</sub>li<sub>ca</sub>t<sub>e</sub> <sub>ac</sub>ti<sub>ons.</sub>

F<sub>or s</sub>i<sub>mp</sub>li<sub>c</sub>it<sub>y, we ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub> thi<sub>s assump</sub>ti<sub>on</sub> th<sub>roug</sub>h<sub>ou</sub>t th<sub>e rema</sub>i<sub>n</sub>d<sub>er o</sub>f th<sub>e ana</sub>l<sub>ys</sub>i<sub>s,</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> Lattimore & Sze<sub>p</sub>esvari (2020). We discuss im<sub>p</sub>lications of non-de<sub>g</sub>enerac<sub>y</sub> and <sub>p</sub>ossible relaxations´ i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>4<sub>.</sub>

## 2<sub>.</sub>3 T<sub>he</sub> E2D <sub>meta-algorithm</sub> <sub>for</sub> <sub>interactive</sub> <sub>decision</sub> <sub>making</sub>

Foster et al. (2021) <sub>p</sub>ro<sub>p</sub>oses E2D (Estimation-to-Decision), a meta-al<sub>g</sub>orithm for interactive decision <sub>ma</sub>ki<sub>ng, w</sub>hi<sub>c</sub>h <sub>re</sub>d<sub>uces</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>ve</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on ma</sub>ki<sub>ng</sub> t<sub>o on</sub>li<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on.</sub> Th<sub>e</sub> E2D <sub>me</sub>t<sub>a-a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>y</sub>ielded man<sub>y p</sub>ractical al<sub>g</sub>orithms for decision makin<sub>g</sub>, es<sub>p</sub>eciall<sub>y</sub> for contextual bandit <sub>p</sub>roblems (e.<sub>g</sub>., Foster & Rakhlin, 2020; Zhu et al., 2022; Zhu & Mineiro, 2022; Zhan<sub>g</sub> et al., 2023). In a nutshell, at <sub>eac</sub>h <sub>roun</sub>d<sub>,</sub> it <sub>ca</sub>ll<sub>s an on</sub>li<sub>ne mo</sub>d<sub>e</sub>l <sub>es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n an es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l $\hat { q } ;$ b<sub>ase</sub>d <sub>on</sub> $\hat { q }$ <sub>an</sub>d context $x ,$ it <sub>so</sub>l<sub>ves a m</sub>i<sub>n</sub>i<sub>max o</sub> ti<sub>m</sub>i<sub>za</sub>ti<sub>on ro</sub>bl<sub>em</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n an ex</sub> l<sub>ora</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $p$ t<sub>o</sub> t<sub>a</sub>k<sub>e</sub> <sub>ac</sub>ti<sub>ons.</sub> S<sub>pec</sub>i<sub>a</sub>li<sub>ze</sub>d t<sub>o par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng,</sub> th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em can</sub> b<sub>e wr</sub>itt<sub>en as:</sub>

$$
\begin{array} { r l } & { \underset { p \in \Delta ( k ) } { \operatorname* { m i n } } \underset { q \in \mathcal { Q } _ { x } , \lambda \in \Delta ( k ) } { \operatorname* { s u p } } \biggl [ p ^ { \top } L q ( x ) - \lambda ^ { \top } L q ( x ) - \gamma \cdot \mathbb { E } _ { a \sim p } \left[ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { a } \hat { q } ( x ) \big | S _ { a } q ( x ) \big ) \right] \biggr ] , } \\ & { } \end{array}\tag{1}
$$

<sub>w</sub>h<sub>ere</sub> $\mathcal { Q } _ { x } : = \left\{ q ( x ) : q \in \mathcal { Q } \right\} \subseteq \Delta ( d )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>ass</sub> $\mathcal { Q }$ <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o</sub> $x .$ The objective explicitly t<sub>ra</sub>d<sub>es o</sub>f b<sub>e</sub>t<sub>ween regre</sub>t <sub>an</sub>d i<sub>n</sub>f<sub>orma</sub>ti<sub>on ga</sub>i<sub>n:</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>par</sub>t $p ^ { \intercal } L q ( x ) - \lambda ^ { \intercal } L q ( x )$ <sup>enco</sup>u<sup>ra</sup>g<sup>es</sup> $p$ to h<sub>ave a</sub> l<sub>ow regre</sub>t<sub>, w</sub>hil<sub>e</sub> th<sub>e secon</sub>d <sub>par</sub>t $- \gamma \cdot \mathbb { E } _ { a \sim p } \left\lceil D _ { \mathrm { H } } ^ { 2 } \big ( S _ { a } \hat { q } ( x ) \big | S _ { a } q ( x ) \big ) \right\rceil$ <sup>enco</sup>u<sup>ra</sup>g<sup>es</sup> $p$ to ex<sub>p</sub><sup>l</sup>ore di<sub>verse ac</sub>ti<sub>ons</sub> t<sub>o correc</sub>t $\hat { q }$ <sub>an</sub>d <sub>ga</sub>i<sub>n new</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on a</sub>b<sub>ou</sub>t $q ^ { * }$ . The minimax value of the objective is called the Decision-Estimation Coeficient (DEC) (Foster et al., 2021), denoted as dec ${ \bf \Phi } _ { \gamma } ( \tilde { q } ( x ) , \mathcal { Q } _ { x } )$ T<sub>o</sub> h<sub>e</sub>l th<sub>e</sub> l<sub>earner ma</sub>k<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons e</sub>fi<sub>c</sub>i<sub>en</sub>tl <sub>,</sub> E2D <sub>re u</sub>i<sub>res access</sub> t<sub>o an orac</sub>l<sub>e</sub> th<sub>a</sub>t <sub>can rov</sub>id<sub>e</sub> <sub>accura</sub>t<sub>e es</sub>ti<sub>ma</sub>ti<sub>on o</sub>f th<sub>e ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> f<sub>or a</sub>ll <sub>con</sub>t<sub>ex</sub>t<sub>s on</sub>li<sub>ne, w</sub>hi<sub>c</sub>h<sub>, w</sub>h<sub>en spec</sub>i<sub>a</sub>li<sub>ze</sub>d t<sub>o</sub> <sub>our s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng se</sub>tti<sub>ng,</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as:</sub>

Definition 2.4. An online outcome estimation oracle $\mathrm { A l g } _ { \mathrm { O n } }$ i<sub>s an a</sub>l<sub>gor</sub>ith<sub>m</sub> th<sub>a</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>es</sub> th<sub>e</sub> <sub>un</sub>k<sub>nown ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q ^ { * }$ f<sub>rom</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> i<sub>n a sequen</sub>ti<sub>a</sub>l f<sub>as</sub>hi<sub>on.</sub> Th<sub>a</sub>t i<sub>s, a</sub>t <sub>eac</sub>h <sub>roun</sub>d $t \in [ T ]$ <sub>g</sub>i<sub>ven</sub> th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>sym</sub>b<sub>o</sub>l<sub>s co</sub>ll<sub>ec</sub>t<sub>e</sub>d <sub>so</sub> f<sub>ar,</sub> th<sub>e orac</sub>l<sub>e pro</sub>d<sub>uces an es</sub>ti<sub>ma</sub>t<sub>e</sub>

$$
\begin{array} { r } { \hat { q } _ { t } = \mathrm { A l g } _ { \mathrm { O n } } ^ { ( t ) } \Big ( \big \{ ( A _ { s } , x _ { s } , \sigma _ { s } ) \big \} _ { s = 1 } ^ { t - 1 } \Big ) \in \mathrm { c o } ( \mathcal { Q } ) , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $\mathrm { A l g } _ { \mathrm { O n } } ^ { ( t ) }$ denotes the map applied at round t and $\left\{ ( A _ { s } , x _ { s } , \sigma _ { s } ) \right\} _ { s = 1 } ^ { t - 1 }$ i<sub>s</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory ava</sub>il<sub>a</sub>bl<sub>e a</sub>t th<sub>e</sub> beginning of round t (note that $\sigma _ { s }$ =is an observed feedback symbol, but not the incurred loss).

$\mathrm { A l g _ { O n } }$ i<sub>s</sub> <sub>sa</sub>id t<sub>o</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> <sub>a</sub> H<sub>e</sub>lli<sub>nger</sub> <sub>regre</sub>t b<sub>oun</sub>d <sub>o</sub>f $\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right)$ if it<sub>s</sub> <sub>on</sub>li<sub>ne</sub> H<sub>e</sub>lli<sub>nger</sub> <sub>regre</sub>t

$$
\mathrm { R e g } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T \right) : = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { a \sim p _ { t } ( \cdot | x _ { t } ) } \left[ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { a } \hat { q } _ { t } ( x _ { t } ) \big | S _ { a } q ^ { * } ( x _ { t } ) \big ) \right] ,\tag{2}
$$

i<sub>s a</sub>t <sub>mos</sub>t $\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right)$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $1 - \delta .$

W<sub>e no</sub>t<sub>e</sub> th<sub>a</sub>t i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> $\hat { q } _ { t } ( x _ { t } )$ t<sub>o</sub> b<sub>e po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se c</sub>l<sub>ose</sub> t<sub>o</sub> $q ^ { * } ( x _ { t } )$ <sub>on avera e</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>4 <sub>on</sub>l<sub>y requ</sub>i<sub>res</sub> th<sub>e</sub>i<sub>r assoc</sub>i<sub>a</sub>t<sub>e</sub>d f<sub>ee</sub>db<sub>ac</sub>k di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> t<sub>o</sub> b<sub>e c</sub>l<sub>ose on average.</sub> Thi<sub>s</sub> i<sub>s reasona</sub>bl<sub>e s</sub>i<sub>nce</sub> <sub>we</sub> <sub>can</sub> <sub>on</sub>l<sub>y</sub> <sub>use</sub> th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k $\sigma _ { s } { ' } \mathrm { s }$ t<sub>o</sub> <sub>gu</sub>id<sub>e</sub> <sub>our</sub> <sub>on</sub>li<sub>ne</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>o</sub>f ${ \hat { q } } _ { t } ;$ i<sub>n</sub> th<sub>e</sub> <sub>ex</sub>t<sub>reme</sub> <sub>case,</sub> it i<sub>s</sub> <sub>poss</sub>ibl<sub>e</sub> th<sub>a</sub>t dif<sub>eren</sub>t $q ^ { * } { \bf s }$ <sub>resu</sub>lt i<sub>n</sub> th<sub>e</sub> <sub>same</sub> f<sub>ee</sub>db<sub>ac</sub>k di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on,</sub> <sub>ma</sub>ki<sub>ng</sub> th<sub>e</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> $q$ <sub>un</sub>id<sub>en</sub>tifi<sub>a</sub>bl<sub>e.</sub> A<sub>ppen</sub>di<sub>x</sub> B <sub>g</sub>i<sub>ves</sub> t<sub>wo concre</sub>t<sub>e examp</sub>l<sub>es o</sub>f th<sub>e on</sub>li<sub>ne ou</sub>t<sub>come es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e.</sub> F<sub>or examp</sub>l<sub>e,</sub> F<sub>or</sub> fi<sub>n</sub>it<sub>e</sub> $\mathcal { Q } ,$ Vovk’s a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> al<sub>g</sub>orithm (Vovk, 1995) attains $\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta ) = O ( \log ( | \mathcal { Q } | / \delta ) )$ E<sub>qu</sub>i<sub>ppe</sub>d <sub>w</sub>ith <sub>an on</sub>li<sub>ne ou</sub>t<sub>come es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e,</sub> E2D i<sub>s</sub> k<sub>nown</sub> t<sub>o</sub> h<sub>ave a regre</sub>t b<sub>oun</sub>d <sub>o</sub>f

$$
\mathrm { R e g } _ { T } \leq O \left( T \operatorname* { m a x } _ { x \in \mathcal { X } } \deg _ { \gamma } ( \widehat { q } ( x ) , \mathcal { Q } _ { x } ) + \gamma \overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right) \right) ,
$$

<sub>w</sub>hi<sub>c</sub>h t<sub>rans</sub>l<sub>a</sub>t<sub>es</sub> t<sub>o</sub> $O ( T ^ { 2 / 3 } )$ <sub>an</sub>d $O ( \sqrt { T } )$ <sub>regre</sub>t<sub>s w</sub>ith <sub>op</sub>ti<sub>ma</sub>l t<sub>un</sub>i<sub>ng</sub> $\mathrm { o f } \ \gamma$ i<sub>n g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y an</sub>d l<sub>oca</sub>ll<sub>y</sub> observable settin<sub>g</sub>s, res<sub>p</sub>ectivel<sub>y</sub> (Foster et al., 2021). However, a ke<sub>y</sub> drawback of the naive E2D al<sub>g</sub>orithm is that directl<sub>y</sub> solvin<sub>g</sub> the minimax o<sub>p</sub>timization <sub>p</sub>roblem (Equation (1)) is com<sub>p</sub>utationall<sub>y</sub> intractable: although the objective is linear in $p ,$ it is not jointly concave in $( q , \lambda )$ <sub>.</sub> N<sub>ever</sub>th<sub>e</sub>l<sub>ess,</sub> it <sub>was</sub> shown in Foster et al. (2021) that as long as distribution p certifies a small value of dec<sub>γ</sub> $( \hat { q } ( x ) , \mathcal { Q } _ { x } )$ (Equation (1)), i.e., it achieves a small $\mathcal { V } _ { \gamma } ^ { \mathrm { d e c } } ( \hat { q } ( x ) , \mathcal { Q } _ { x } , p )$ <sub>va</sub>l<sub>ue,</sub> E2D <sub>w</sub>ill h<sub>ave sma</sub>ll <sub>regre</sub>t<sub>.</sub> A<sub>s we</sub> <sub>w</sub>ill <sub>see,</sub> <sub>our</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> d<sub>es</sub>i<sub>gn</sub> <sub>w</sub>ill <sub>u</sub>tili<sub>ze</sub> thi<sub>s</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> t<sub>o</sub> <sub>a</sub>dd<sub>ress</sub> th<sub>e</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>c</sub>h<sub>a</sub>ll<sub>enge.</sub>

## 3 E<sub>fficient</sub> <sub>algorithm</sub> <sub>for</sub> <sub>contextual</sub> <sub>partial</sub> <sub>monitoring</sub>

In this section we <sub>p</sub>resent E2D.PMSide (Estimation-to-Decision for Partial Monitorin<sub>g</sub> with Side information, Al orithm 1), which can be viewed as a s ecialization of the E2D meta-al orithm. At ever<sub>y</sub> round t, E2D.PMSide obtains an outcome distribution estimate $\hat { q } _ { t } ( \cdot )$ (line 4) from an online <sub>ou</sub>t<sub>come es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e w</sub>ith H<sub>e</sub>lli<sub>nger regre</sub>t <sub>guaran</sub>t<sub>ee.</sub> Th<sub>en</sub> it <sub>uses</sub> $\hat { q } _ { t } ( \cdot )$ <sub>an</sub>d <sub>o</sub>b<sub>serve</sub>d <sub>con</sub>t<sub>ex</sub>t $x _ { t }$ t<sub>o cons</sub>t<sub>ruc</sub>t <sub>an exp</sub>l<sub>ora</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $p _ { t }$ that approximately minimizes an objective function $\mathcal { V } _ { \gamma } ( \hat { q } _ { t } ( x _ { t } ) , p )$ , a relaxation of the DEC objective $\mathcal { \dot { V } } _ { \gamma } ^ { \mathrm { d e c } } ( \hat { q } ( x _ { t } ) , \dot { \mathcal { Q } } _ { x _ { t } } , p )$ (line 5). The choice of $p _ { t }$ balances two competing demands. The first term in the objective measures the loss of $p _ { t }$ <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o a</sub> <sup>com</sup>p<sup>arator</sup> p<sup>olic</sup>y $\lambda$ <sub>un</sub>d<sub>er</sub> th<sub>e es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\hat { q } _ { t } ( x _ { t } )$ <sub>, an</sub>d i<sub>s sma</sub>ll <sub>w</sub>h<sub>en</sub> $p _ { t }$ <sub>exp</sub>l<sub>o</sub>it<sub>s</sub> what has been learned. The second term is a covera<sub>g</sub>e term (which we will define shortl<sub>y</sub>) that measures the amenabilit<sub>y</sub> of using feedback from following p in evaluating $\lambda \mathbf { \dot { s } }$ l<sub>oss;</sub> th<sub>e sma</sub>ll<sub>er</sub> th<sub>e</sub> <sub>coverage va</sub>l<sub>ue</sub> th<sub>e</sub> b<sub>e</sub>tt<sub>er</sub> $p$ i<sub>s</sub> f<sub>or exp</sub>l<sub>ora</sub>ti<sub>on.</sub> Th<sub>e exp</sub>l<sub>ora</sub>ti<sub>on parame</sub>t<sub>er</sub> $\gamma$ fi<sub>xes</sub> th<sub>e ra</sub>t<sub>e o</sub>f <sub>exc</sub>h<sub>ange</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo.</sub> W<sub>e now</sub> d<sub>e</sub>fi<sub>ne coverage</sub> f<sub>orma</sub>ll<sub>y.</sub>

Algorithm 1 E2D.PMSide   
1: parameters:   
O<sub>n</sub>li<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> $\mathrm { A l g } _ { \mathrm { O n } }$   
E<sub>xp</sub>l<sub>o</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r $\gamma > 0$   
2: for $t \overset { \cdot } { = } 1 , 2 , \cdots , \overset { \cdot } { T }$ do   
3<sub>:</sub> R<sub>ece</sub>i<sub>ve con</sub>t<sub>ex</sub>t $x _ { t }$   
4<sub>:</sub> C<sub>o</sub>m<sub>pu</sub>t<sub>e es</sub>tim<sub>a</sub>t<sub>e</sub> $\hat { q } _ { t } = \operatorname { A l g } _ { \operatorname { O n } } ^ { ( t ) } \big ( \big \{ A _ { s } , x _ { s } , \sigma _ { s } \big \} _ { s = 1 } ^ { t - 1 } \big )$   
5<sub>:</sub> Fi<sub>n</sub>d $p _ { t }$ th<sub>a</sub>t <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub>   
$\mathcal { V } _ { \gamma } ( \hat { q } _ { t } ( x _ { t } ) , p ) : = \operatorname* { s u p } _ { \lambda \in \Delta ( \Pi ) } \underbrace { p ^ { \top } L \hat { q } _ { t } ( x _ { t } ) - \lambda ^ { \top } L \hat { q } _ { t } ( x _ { t } ) } _ { \mathrm { e x p l o i t } } + \frac { 1 } { \gamma } \cdot \underbrace { \mathrm { C o v e r a g e } ( p , \lambda ) } _ { \mathrm { e x p l o r e } } .$   
usin<sub>g</sub> ϵ-Greedy.PMSide (Al<sub>g</sub>. 2) or SquareCB.PMSide (Al<sub>g</sub>. 3) for <sub>g</sub>loball<sub>y</sub> and locall<sub>y</sub>   
<sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>se</sub>tti<sub>ngs,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub>   
$_ 6 \colon$ Ch<sub>oose ac</sub>ti<sub>on</sub> $A _ { t } \sim p _ { t }$ <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e w</sub>ith $( A _ { t } , x _ { t } , \sigma _ { t } )$   
7: end for

Definition 3.1 (Coverage). We define the coverage of action distribution $p \in \Delta ( k )$ f<sub>or ac</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\lambda \in \Delta ( k )$ f<sub>or</sub> <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> <sub>game</sub> $\bar { \mathcal { G } } = ( L , \Phi )$ as

$$
\mathrm { C o v e r a g e } ( p , \lambda ) : = \operatorname* { s u p } _ { q _ { 1 } , q _ { 2 } \in \Delta ( d ) } \frac { ( ( \lambda - p ) ^ { \top } L ( q _ { 1 } - q _ { 2 } ) ) ^ { 2 } } { \mathbb { E } _ { a \sim p } [ D _ { \mathrm { H } } ^ { 2 } ( S _ { a } q _ { 1 }  S _ { a } q _ { 2 } ) ] } \cdot \mathrm { ( w i t h ~ c o n v e n t i o n ~ \frac { \partial 0 } { \partial \eta } = 0 ) }
$$

T<sub>o ma</sub>k<sub>e sense o</sub>f thi<sub>s</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> it <sub>may</sub> b<sub>e</sub> h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l t<sub>o</sub> thi<sub>n</sub>k <sub>a</sub>b<sub>ou</sub>t th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> h<sub>ypo</sub>th<sub>e</sub>ti<sub>ca</sub>l <sub>o</sub>fli<sub>ne</sub> <sub>po</sub>li<sub>cy</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em</sub> i<sub>n</sub> <sub>non-con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng:</sub> <sub>suppose</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q ^ { * }$ i<sub>s un</sub>k<sub>nown; we use a</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or po</sub>li<sub>cy</sub> $p$ t<sub>o</sub> t<sub>a</sub>k<sub>e ac</sub>ti<sub>on an</sub>d <sub>co</sub>ll<sub>ec</sub>t f<sub>ee</sub>db<sub>ac</sub>k <sub>sym</sub>b<sub>o</sub>l<sub>s,</sub> so as to estimate the expected loss of λ (more precisel<sub>y</sub>, the expected loss diference between λ and $p ) ;$ h<sub>ow muc</sub>h <sub>u</sub>tilit<sub>y</sub> d<sub>oes eac</sub>h d<sub>a</sub>t<sub>a po</sub>i<sub>n</sub>t <sub>carry</sub> f<sub>or suc</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub>? L<sub>e</sub>t $q _ { 1 }$ b<sub>e</sub> $q ^ { * }$ <sub>an</sub>d $q _ { 2 }$ b<sub>e</sub> ${ \hat { q } } ,$ our <sub>es</sub>ti<sub>ma</sub>t<sub>e o</sub>f $q ^ { * }$ b<sub>ase</sub>d <sub>on</sub> d<sub>a</sub>t<sub>a.</sub> Th<sub>e numera</sub>t<sub>or</sub> th<sub>us measures</sub> th<sub>e square</sub>d <sub>po</sub>li<sub>cy eva</sub>l<sub>ua</sub>ti<sub>on error;</sub> th<sub>e</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>or measures</sub> th<sub>e</sub> i<sub>n-</sub>di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on es</sub>ti<sub>ma</sub>ti<sub>on error, w</sub>hi<sub>c</sub>h <sub>we expec</sub>t t<sub>o</sub> d<sub>ecrease w</sub>ith <sub>more</sub> <sub>samp</sub>l<sub>es co</sub>ll<sub>ec</sub>t<sub>e</sub>d<sub>.</sub> A <sub>sma</sub>ll $\mathrm { C o v e r a g e } ( p , \lambda )$ i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t f<sub>ee</sub>db<sub>ac</sub>k <sub>s</sub>i<sub>gna</sub>l<sub>s v</sub>i<sub>a</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> $p$ i<sub>s use</sub>f<sub>u</sub>l i<sub>n</sub> estimating p and λ’s loss diference.

O<sub>ur covera e</sub> i<sub>n</sub> th<sub>e a</sub>b<sub>ove</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> b<sub>ears some s</sub>i<sub>m</sub>il<sub>ar</sub>iti<sub>es w</sub>ith th<sub>e covera e</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> th<sub>e con</sub>t<sub>ex</sub>t<sub>ua</sub>l bandit settin<sub>g</sub> b<sub>y</sub> Qin & Zhan<sub>g</sub> (2026b), but is diferent in two im<sub>p</sub>ortant wa<sub>y</sub>s: (1) here, the numerator and the denominator are with res<sub>p</sub>ect to the loss matrix (evaluation error) and the si<sub>g</sub>nal matrix (information <sub>g</sub>ain), res<sub>p</sub>ectivel<sub>y</sub>, where in the bandit settin<sub>g</sub> the<sub>y</sub> are defined both usin<sub>g</sub> the loss function; (2) it is well-known that for some <sub>g</sub>loball<sub>y</sub> observable <sub>p</sub>artial monitorin<sub>g p</sub>roblems the absolute value of a loss of an action is not identifiable (Lattimore & Sze<sub>p</sub>esvari, 2020, Exam<sub>p</sub>le 37.9),´ <sub>an</sub>d <sub>on</sub>l<sub>y</sub> l<sub>oss</sub> dif<sub>erences</sub> b<sub>e</sub>t<sub>ween ac</sub>ti<sub>on pa</sub>i<sub>rs</sub> i<sub>s</sub> id<sub>en</sub>tifi<sub>a</sub>bl<sub>e;</sub> th<sub>us,</sub> th<sub>e numera</sub>t<sub>or</sub> i<sub>n</sub> th<sub>e coverage</sub> <sub>measures</sub> th<sub>e</sub> dif<sub>erence o</sub>f l<sub>osses o</sub>f t<sub>wo po</sub>li<sub>c</sub>i<sub>es.</sub> O<sub>ur coverage a</sub>l<sub>so s</sub>h<sub>ares some s</sub>i<sub>m</sub>il<sub>ar</sub>iti<sub>es w</sub>ith <sub>g</sub>lobal and local ali<sub>g</sub>nment constants of (Kirschner et al., 2023, Eqs. (15), (16)), whose numerator $\overline { { ( \langle \phi _ { a } - \phi _ { b } , v \rangle ) ^ { 2 } } }$ <sub>can</sub> b<sub>e</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>e</sub>d <sub>as some es</sub>ti<sub>ma</sub>ti<sub>on error o</sub>f th<sub>e</sub> l<sub>oss</sub> dif<sub>erences</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo ac</sub>ti<sub>ons</sub> $a , b ,$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>or</sub> $\| M _ { c } v \| ^ { 2 }$ can be interpreted as the information gain of taking action c in correcting the model estimation error v. Intead of measuring coverage between an action and a pair <sub>o</sub>f <sub>ac</sub>ti<sub>ons, our</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on measures</sub> th<sub>e coverage</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo ac</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons.</sub>

## W<sub>e</sub> n<sub>ow</sub> <sub>p</sub>r<sub>ese</sub>nt th<sub>e</sub> r<sub>eg</sub>r<sub>e</sub>t <sub>gua</sub>r<sub>a</sub>nt<sub>ee</sub> <sub>o</sub>f E2D<sub>.</sub>PMS<sub>ide:</sub>

Theorem 3.1 (Regret of E2D.PMSide). Suppose $\mathrm { A l g _ { O n } }$ h<sub>as a</sub> H<sub>e</sub>lli<sub>nger regre</sub>t b<sub>oun</sub>d $\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right)$ Fi<sub>x</sub> $\gamma > 0$ <sub>.</sub> Th<sub>en,</sub> f<sub>or any</sub> $\delta \in ( 0 , 1 )$ <sub>, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta$

$$
\mathrm { R e g } _ { T } \leq T \cdot \operatorname* { m a x } _ { t = 1 } ^ { T } \mathcal { V } _ { \gamma } \big ( \hat { q } _ { t } ( x _ { t } ) , p _ { t } \big ) + \gamma \cdot \overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right) .
$$

It<sub>s</sub> <sub>proo</sub>f i<sub>s</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>1<sub>.</sub> I<sub>n</sub> <sub>a</sub> <sub>nu</sub>t<sub>s</sub>h<sub>e</sub>ll<sub>,</sub> <sub>our</sub> th<sub>eorem</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> <sub>me</sub>t<sub>a-regre</sub>t theorem of Foster et al. (2021): as lon<sub>g</sub> as all action distributions $p _ { t }$ <sub>cer</sub>tif<sub>y</sub> th<sub>a</sub>t $\deg _ { \gamma } ( \hat { q } ( x ) , \mathcal { Q } _ { x } )$ h<sub>as</sub> <sub>an upper</sub> b<sub>oun</sub>d<sub>,</sub> E2D h<sub>as a sma</sub>ll <sub>regre</sub>t i<sub>n</sub> t<sub>erms o</sub>f th<sub>a</sub>t b<sub>oun</sub>d<sub>.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>, our</sub> $p _ { t }$ h<sub>ere cer</sub>tifi<sub>es</sub> th<sub>a</sub>t

Algorithm 2 ϵ-Greedy.PMSide for globally observable games   
1: parameters:   
2<sub>:</sub> Context x with predicted outcome distribution ${ \hat { q } } ( x )$   
3<sub>:</sub> E<sub>xp</sub>l<sub>o</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r $\gamma > 1 6 k ^ { 2 } \beta ^ { 2 }$   
4: Let <sup>ˆ</sup>b ∈ arg min $\boldsymbol { \mathrm { \Phi } } _ { a \in \Pi } e _ { a } ^ { \top } \pmb { L } \hat { q } ( \boldsymbol { x } )$   
5: return $\boldsymbol { p } = \big ( 1 - \epsilon \big ) \boldsymbol { e } _ { \hat { b } } + \epsilon \frac { 1 } { k } \mathbf { 1 }$ f<sub>or</sub> $\epsilon = k \beta / \sqrt { \gamma }$   
de $\therefore \textstyle ( { \hat { q } } ( x ) , \ Q _ { x } )$ i<sub>s u er</sub> b<sub>oun</sub>d<sub>e</sub>d b $\mathcal { V } _ { \gamma } ( \hat { q } _ { t } ( x _ { t } ) , p _ { t } )$ (Please see Remark $3 )$ <sub>; an</sub>d id<sub>ea</sub>ll <sub>, we s</sub>h<sub>ou</sub>ld   
<sub>c</sub>h<sub>oose exp</sub>l<sub>ora</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\mathbf { \Lambda } _ { p _ { t } } \mathbf { \dot { s } }$ th<sub>a</sub>t <sub>ma</sub>k<sub>e</sub> $\mathcal { V } _ { \gamma } ( \hat { q } _ { t } ( x _ { t } ) , p _ { t } ) ^ { , }$ <sub>s as sma</sub>ll <sub>as poss</sub>ibl<sub>e.</sub>   
U<sub>n</sub>f<sub>or</sub>t<sub>una</sub>t<sub>e</sub>l <sub>,</sub> fi<sub>n</sub>di<sub>n</sub> $p _ { t }$ th<sub>a</sub>t <sub>exac</sub>tl <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> $\mathcal { V } _ { \gamma } ( \hat { q } _ { t } ( x _ { t } ) , p )$ <sub>ma s</sub>till b<sub>e com u</sub>t<sub>a</sub>ti<sub>ona</sub>ll i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bl<sub>e;</sub>   
t<sub>o see</sub> thi<sub>s, no</sub>t<sub>e</sub> th<sub>a</sub>t th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f <sub>coverage</sub> i<sub>nvo</sub>l<sub>ves supremum over a pa</sub>i<sub>r o</sub>f <sub>ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub>   
$q _ { 1 } , q _ { 2 } .$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s no</sub>t <sub>a convex op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em.</sub> Thi<sub>s mo</sub>ti<sub>va</sub>t<sub>es</sub> th<sub>e ques</sub>ti<sub>on: can we</sub> fi<sub>n</sub>d   
<sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>exp</sub>l<sub>ora</sub>ti<sub>on ru</sub>l<sub>es</sub> $x \mapsto p ( x )$ that <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> $\mathcal { V } _ { \gamma } \big ( \hat { q } _ { t } ( x ) , p ( x ) \big )$   
f<sub>or a</sub>ll $x ?$ I<sub>n</sub>d<sub>ee</sub>d<sub>, we s</sub>h<sub>ow</sub> i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>n</sub> t<sub>wo su</sub>b<sub>sec</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>suc</sub>h <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>ex</sub> l<sub>ora</sub>ti<sub>on ru</sub>l<sub>e ex</sub>i<sub>s</sub>t   
<sub>an</sub>d i<sub>e</sub>ld <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>ons o</sub>f E2D<sub>.</sub>PMS<sub>ide.</sub>   
3<sub>.</sub>1 E<sub>fficient algorithm for globally observable games</sub>   
We show that in the loball observable settin ϵ-Greedy.PMSide $( \mathrm { A l g } . 2 ) ,$ <sub>,</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>a</sub> <sub>po</sub>li<sub>cy</sub> $p$ <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub>   
$\mathcal { V } _ { \gamma } ( \hat { q } ( x ) , p ( x ) ) \leq O ( \sqrt { 1 / \gamma } )$ <sub>, w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves an</sub> $O ( T ^ { 2 / 3 } )$ <sub>re re</sub>t b<sub>oun</sub>d <sub>w</sub>h<sub>en</sub> l<sub>u e</sub>d i<sub>n</sub>t<sub>o</sub> E2D<sub>.</sub>PMS<sub>ide.</sub>   
ϵ-Greedy.PMSide identifies the empirical best action $\hat { b }$ (Line 4) and assi<sub>g</sub>ns it <sub>p</sub>robabilit<sub>y</sub> $1 - \epsilon .$   
spreading the remaining ϵ uniforml<sub>y</sub> over all k actions (Line 5).   
Lemma 3.1 (Guarantee for Alg. 2 for globally observable games). Let $\mathcal { G } = \left( L , \Phi \right)$ b<sub>e a g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serv-</sub>   
<sub>a</sub>bl<sub>e par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng game</sub> i<sub>n</sub> th<sub>e sense o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>2<sub>, an</sub>d <sub>se</sub>t $\beta : = \operatorname* { m a x } _ { ( a , b ) \in [ k ] \times [ k ] } \| w _ { ( a , b ) } \| _ { \infty } .$   
W<sub>e</sub> h<sub>ave</sub> f<sub>or any</sub> $\hat { q } \in \Delta ( d )$   
$\mathcal { V } _ { \gamma } \big ( \widehat { q } , p \big ) \leq \operatorname* { m a x } _ { \lambda \in \Delta ( \Pi ) } p ^ { T } L \widehat { q } - \lambda ^ { T } L \widehat { q } + \frac { 1 6 \beta ^ { 2 } } { \gamma } \sum _ { c = 1 } ^ { k } \frac { 1 } { p _ { c } } .$   
Th<sub>us,</sub> f<sub>or</sub> <sub>every</sub> $\gamma > 1 6 k ^ { 2 } \beta ^ { 2 }$ , <sup>ever</sup>y $\hat { q } \in \Delta ( d )$ <sub>,</sub> Al<sub>g.</sub> 2 <sub>ou</sub>t<sub>pu</sub>t<sub>s</sub> $p$ <sub>suc</sub>h th<sub>a</sub>t   
$\mathcal { V } _ { \gamma } \left( \hat { q } , p \right) \leq O \left( k \beta \sqrt { \frac { 1 } { \gamma } } \right) .$   
P<sub>roo</sub>f i<sub>s</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> A <sub>en</sub>di<sub>x</sub> C<sub>.</sub>2<sub>.</sub> F<sub>ur</sub>th<sub>ermore,</sub> b <sub>com</sub>bi<sub>n</sub>i<sub>n</sub> Th<sub>eorem</sub> 3<sub>.</sub>1 <sub>an</sub>d L<sub>emma</sub> 3<sub>.</sub>1<sub>, our</sub>   
<sub>a</sub>l<sub>gor</sub>ith<sub>m ac</sub>hi<sub>eves</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng m</sub>i<sub>n</sub>i<sub>max ra</sub>t<sub>e</sub> f<sub>or g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e games.</sub>   
Corollary 1 (Re ret of E2D.PMSide for loball observable ames). Su ose $\mathrm { A l g } _ { \mathrm { O n } }$ h<sub>as a</sub>   
H<sub>e</sub>lli<sub>nger regre</sub>t b<sub>oun</sub>d $\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta )$ <sub>.</sub> F<sub>or a g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng game</sub>   
$( L , \Phi )$ , set $\bar { \beta } = \operatorname* { m a x } _ { ( a , b ) \in [ k ] \times [ k ] } \left\| \boldsymbol { w } _ { ( a , b ) } \right\| _ { \infty }$ . Then,for any $\delta \in ( 0 , 1 )$ , andfor any large enough $T$   
that satisfies $\gamma > 1 6 k ^ { 2 } \beta ^ { 2 }$ <sub>w</sub>ith $\gamma = ( T k \beta / \overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta ) ) ^ { 2 / 3 }$ , with probability ofat least $1 - \delta ,$   
E2D<sub>.</sub>PMS<sub>ide run w</sub>ith $A l g 2$ as the exploration rule satisfies,   
$\mathrm { R e g } _ { T } \leq O \bigg ( \sqrt [ 3 ] { T ^ { 2 } k ^ { 2 } \beta ^ { 2 } \overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathscr { Q } , T , \delta ) } \bigg )$   
I<sub>n</sub> th<sub>e se</sub>tti<sub>ng w</sub>h<sub>en</sub> $\mathcal { Q }$ i<sub>s</sub> fi<sub>n</sub>it<sub>e, se</sub>tti<sub>ng</sub> th<sub>e es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> t<sub>o</sub> b<sub>e</sub> V<sub>ov</sub>k’<sub>s aggrega</sub>ti<sub>ng a</sub>l<sub>gor</sub>ith<sub>m,</sub>   
th<sub>e a</sub>b<sub>ove regre</sub>t b<sub>oun</sub>d i<sub>s o</sub>f <sub>or</sub>d<sub>er</sub> $O ( \sqrt [ 3 ] { T ^ { 2 } k ^ { 2 } \beta ^ { 2 } \log | \mathcal { Q } | } )$ I<sub>n</sub> th<sub>e noncon</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>se</sub>tti<sub>ng, w</sub>h<sub>en</sub>   
$\mathcal { Q } = \Delta ( d _ { \mathcal { Q } } )$ <sub>,</sub> <sub>we</sub> <sub>can</sub> <sub>se</sub>t th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>orac</sub>l<sub>e</sub> t<sub>o</sub> b<sub>e</sub> <sub>any</sub> <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>exp-concave</sub> <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>a</sub>l<sub>go-</sub>   
rithm (Hazan et al., 2007), which has $\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta ) = O ( d _ { \mathcal { Q } } \ln T )$ <sub>,</sub> im<sub>p</sub>l<sub>y</sub>in<sub>g</sub> E2D<sub>.</sub>PMS<sub>ide</sub> h<sub>as</sub>   
a re<sub>g</sub>ret o<sup>f</sup> $\tilde { O } \big ( \sqrt [ 3 ] { T ^ { 2 } k ^ { 2 } \beta ^ { 2 } d _ { \mathcal { Q } } } \big )$ QThi<sub>s ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>o</sub>f <sub>non-con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c par</sub>ti<sub>a</sub>l   
Qmonitoring (Bartok et al., 2011) with a simple computationally eficient approach.´   
3<sub>.</sub>2 E<sub>fficient algorithm for locally observable games</sub>   
I<sub>n</sub> th<sub>e</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e se</sub>tti<sub>ng, we a</sub>i<sub>m</sub> t<sub>o g</sub>i<sub>ve an e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>exp</sub>l<sub>ora</sub>ti<sub>on ru</sub>l<sub>e</sub> th<sub>a</sub>t fi<sub>n</sub>d<sub>s</sub> $p$ <sub>suc</sub>h th<sub>a</sub>t   
<sub>w</sub>hi<sub>c</sub>h i<sub>e</sub>ld<sub>s a</sub> <sub>re re</sub>t h<sub>en</sub> l <sub>e</sub>d i<sub>n</sub>t<sub>o</sub> E2D PMS<sub>ide</sub> U<sub>n</sub>d<sub>er</sub>

$$
\mathcal { V } _ { \gamma } ( \widehat { q } _ { t } ( x ) , p ( x ) ) \leq O ( \frac { 1 } { \gamma } )
$$

$$
O ( \sqrt { T } )
$$

<sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>o</sub>b<sub>serva</sub>bilit<sub>y,</sub> t<sub>o es</sub>ti<sub>ma</sub>t<sub>e</sub> th<sub>e</sub> l<sub>oss</sub> dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo ac</sub>ti<sub>ons</sub> $( a , b )$ <sub>,</sub> it<sub>s assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>it<sub>ness</sub> f<sub>unc</sub>ti<sub>on</sub> $w _ { ( a , b ) }$ ma<sub>y</sub> be su<sub>pp</sub>orted on all actions (Definition 2.2), so recoverin<sub>g</sub> the diference between a ( )sin<sub>g</sub>le <sub>p</sub>air of actions can re<sub>q</sub>uire <sub>p</sub>la<sub>y</sub>in<sub>g</sub> all k of them<sub>;</sub> this is wh<sub>y</sub> the certif<sub>y</sub>in<sub>g p</sub>olic<sub>y</sub> of Al<sub>g</sub>orithm 2 mixes with the uniform distribution over [k]. In contrast, under local observability, the loss diference b<sub>e</sub>t<sub>ween</sub> t<sub>wo ne</sub>i<sub>g</sub>hb<sub>or</sub>i<sub>ng</sub> P<sub>are</sub>t<sub>o op</sub>ti<sub>ma</sub>l <sub>ac</sub>ti<sub>ons can</sub> b<sub>e expresse</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>o</sub>f th<sub>ose</sub> t<sub>wo</sub> actions alone (Definition 2.3), and the diference between an arbitrar<sub>y</sub> <sub>p</sub>air of Pareto o<sub>p</sub>timal actions is then recovered by chaining such local relations along a path joining them in the neighborhood <sub>grap</sub>h<sub>,</sub> <sub>so</sub> th<sub>a</sub>t <sub>on</sub>l<sub>y</sub> th<sub>e</sub> <sub>ac</sub>ti<sub>ons</sub> <sub>encoun</sub>t<sub>ere</sub>d <sub>a</sub>l<sub>ong</sub> th<sub>a</sub>t <sub>pa</sub>th <sub>nee</sub>d b<sub>e</sub> <sub>p</sub>l<sub>aye</sub>d<sub>.</sub> P<sub>er</sub>h<sub>aps</sub> <sub>surpr</sub>i<sub>s</sub>i<sub>ng</sub>l<sub>y,</sub> <sub>we</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t I<sub>nverse</sub> G<sub>ap</sub> W<sub>e</sub>i<sub>g</sub>hti<sub>ng, an exp</sub>l<sub>ora</sub>ti<sub>on ru</sub>l<sub>e</sub> th<sub>a</sub>t <sub>was prev</sub>i<sub>ous</sub>l<sub>y</sub> k<sub>nown</sub> t<sub>o</sub> b<sub>e e</sub>f<sub>ec</sub>ti<sub>ve</sub> f<sub>or</sub> ex<sub>p</sub>loration in contextual bandits (SquareCB.PMSide, Al<sub>g</sub>. 3), <sub>y</sub>ields a o<sub>p</sub>timal-re<sub>g</sub>ret al<sub>g</sub>orithm h<sub>ere w</sub>ith<sub>ou</sub>t <sub>mo</sub>difi<sub>ca</sub>ti<sub>on.</sub>

Lemma 3.2 (Guarantee for Alg. 3 for locally observable games). Let $\mathcal { G } = \left( L , \Phi \right)$ b<sub>e</sub> <sub>a</sub> l<sub>oca</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng game</sub> i<sub>n</sub> th<sub>e sense o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>3<sub>.</sub> F<sub>or every</sub> $\gamma > 0 .$ an<sup>d</sup> ever<sub>y</sub> $\hat { q } \in \Delta ( d )$ <sub>,</sub> Al<sub>g.</sub> 3 <sub>run w</sub>ith $\kappa = \frac { \gamma } { 8 k m ^ { 2 } }$ <sub>an</sub>d $\mu = k ,$ , outputs p such that

$$
\mathcal { V } _ { \gamma } \left( \hat { q } , p \right) \leq O \left( \frac { m ^ { 2 } k ^ { 2 } } { \gamma } \right) .
$$

P<sub>roo</sub>f i<sub>s</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>3<sub>.</sub> F<sub>ur</sub>th<sub>ermore,</sub> b<sub>y</sub> <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> Th<sub>eorem</sub> 3<sub>.</sub>1 <sub>an</sub>d L<sub>emma</sub> 3<sub>.</sub>2<sub>,</sub> <sub>we</sub> <sub>can</sub> <sub>ac</sub>hi<sub>eve</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng m</sub>i<sub>n</sub>i<sub>max ra</sub>t<sub>e</sub> f<sub>or</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e games.</sub>

Corollary 2 (Regret of E2D.PMSide for locally observable games). Suppose $\mathrm { A l g } _ { \mathrm { O n } }$ h<sub>as a</sub> H<sub>e</sub>lli<sub>nger regre</sub>t b<sub>oun</sub>d $\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta )$ <sub>.</sub> L<sub>e</sub>t A<sub>ssump</sub>ti<sub>on</sub> 2<sub>.</sub>2 h<sub>o</sub>ld<sub>, an</sub>d <sub>suppose</sub> i<sub>n a</sub>dditi<sub>on</sub> that the game is locally observable. Then, for any $\delta \in ( 0 , 1 )$ , with probability of at least $1 - \delta ,$ E2D.PMSide run with Alg. 3 as the exploration rule (with $\begin{array} { r } { \kappa = \frac { \gamma } { 8 k m ^ { 2 } } } \end{array}$ <sub>an</sub>d $\mu = k )$ <sub>, w</sub>ith $\gamma = ( T m ^ { 2 } k ^ { 2 } / \overrightarrow { \mathrm { R e g } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta ) ) ^ { 1 / 2 }$ satisfies

$$
\mathrm { R e g } _ { T } \leq O \left( \sqrt { T \cdot m ^ { 2 } k ^ { 2 } \cdot \overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta ) } \right) .
$$

F<sub>or</sub> fi<sub>n</sub>it<sub>e</sub> $\mathcal { Q } ,$ <sub>,</sub> V<sub>ov</sub>k’<sub>s aggrega</sub>ti<sub>ng a</sub>l<sub>gor</sub>ith<sub>m g</sub>i<sub>ves</sub> $O ( \sqrt { T m ^ { 2 } k ^ { 2 } \log | \mathcal { Q } | } )$ <sub>;</sub> i<sub>n</sub> th<sub>e</sub> <sub>non-con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>se</sub>tti<sub>ng</sub> <sub>w</sub>ith $\mathcal { Q } = \Delta ( d _ { \mathcal { Q } } )$ , an ex<sub>p</sub>-concave oracle (Hazan et al., 2007) <sub>g</sub>ives $\tilde { O } ( \sqrt { T m ^ { 2 } k ^ { 2 } d _ { \mathcal { Q } } } )$ <sub>.</sub> B<sub>o</sub>th <sub>ma</sub>t<sub>c</sub>h th<sub>e</sub> Qminimax rate in $T$ Qfor this class. We next explain the key idea underlying the proof of Lemma 3.2, which hinges on a sparse relaxation of the objective $\mathcal { V } _ { \gamma } \big ( \widehat { q } , p \big )$

## 3<sub>.</sub>2<sub>.</sub>1 S<sub>parse</sub> <sub>relaxation</sub> <sub>of</sub> $\mathcal { V } _ { \gamma } \big ( \widehat { q } , p \big )$

W<sub>e a</sub>i<sub>m</sub> t<sub>o</sub> fi<sub>n</sub>d t<sub>rac</sub>t<sub>a</sub>bl<sub>e re</sub>l<sub>axa</sub>ti<sub>ons o</sub>f $\mathcal { V } _ { \gamma }$ <sub>w</sub>hi<sub>c</sub>h <sub>s</sub>h<sub>e</sub>d<sub>s</sub> li<sub>g</sub>ht<sub>s on</sub> fi<sub>n</sub>di<sub>ng e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>exp</sub>l<sub>ora</sub>ti<sub>on s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> f<sub>or</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> $\mathcal { V } _ { \gamma }$ <sub>.</sub> I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> <sub>we</sub> <sub>a</sub>i<sub>m</sub> t<sub>o</sub> <sub>exp</sub>l<sub>o</sub>it th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d f<sub>ee</sub>db<sub>ac</sub>k <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> <sub>o</sub>f l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e games</sub> t<sub>o res</sub>t<sub>r</sub>i<sub>c</sub>t <sub>exp</sub>l<sub>ora</sub>t<sub>ory ac</sub>ti<sub>ons</sub> t<sub>o</sub> th<sub>ose</sub> l<sub>y</sub>i<sub>ng a</sub>l<sub>ong pa</sub>th<sub>s</sub> th<sub>a</sub>t <sub>connec</sub>t <sub>pa</sub>i<sub>rs</sub> <sub>o</sub>f <sub>ac</sub>ti<sub>ons</sub> <sub>w</sub>h<sub>ose</sub> l<sub>oss</sub> dif<sub>erences</sub> <sub>are</sub> <sub>o</sub>f i<sub>n</sub>t<sub>eres</sub>t<sub>.</sub> Th<sub>e</sub> <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d <sub>grap</sub>h $\bar { ( \Pi , \mathcal { E } ) }$ i<sub>s</sub> <sub>connec</sub>t<sub>e</sub>d<sub>.</sub> Assign an arbitrary action root as the root and choose a spanning in-tree $\bar { \boldsymbol { \tau } }$ (See Fi<sub>g</sub>ure 1) such Tthat, there is a unique directed path (a) from any action a ∈ Π to root ∈ Π. Now, we can relax the $\mathcal { V } _ { \gamma } \big ( \widehat { q } , p \big )$ <sub>as</sub> f<sub>o</sub>ll<sub>ows,</sub>

Lemma 3.3. Let $\mathcal { G } = \left( L , \Phi \right)$ b<sub>e a</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng game</sub> i<sub>n</sub> th<sub>e sense o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>3<sub>.</sub> L<sub>e</sub>t $\tau$ b<sub>e an</sub> i<sub>n-</sub>t<sub>ree o</sub>f th<sub>e ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d <sub>grap</sub>h<sub>.</sub> F<sub>or a</sub>ll $p \in \bar { \Delta ( \Pi ) }$ <sub>, we</sub> h<sub>ave</sub>

$$
\mathcal { V } _ { \gamma } \big ( \hat { q } ( x ) , p \big ) \leq \operatorname* { m a x } _ { \lambda \in \Delta ( \Pi ) } p ^ { T } L \hat { q } ( x ) - \lambda ^ { T } L \hat { q } ( x ) + \frac { 3 2 } { \gamma } \operatorname* { m a x } _ { e \in \mathcal { E } } \big \| w _ { e } \big \| _ { \infty } ^ { 2 } \sum _ { ( b \in \Pi , c \in \Pi ) : c \in \mathrm { p a t h } _ { \mathcal { T } } ( b ) } \left( \frac { \lambda _ { b } } { p _ { c } } + \frac { p _ { b } } { p _ { c } } \right)\tag{3}
$$

P<sub>roo</sub>f i<sub>s</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>3<sub>.</sub> Ab<sub>ove resu</sub>lt i<sub>s</sub> t<sub>rue</sub> f<sub>or any</sub> i<sub>n-</sub>t<sub>ree w</sub>ith <sub>an ar</sub>bit<sub>rary roo</sub>t<sub>.</sub> H<sub>owever,</sub> i<sub>n genera</sub>l<sub>, we</sub> d<sub>o no</sub>t <sub>wan</sub>t th<sub>e pa</sub>th<sub>s</sub> t<sub>o</sub> b<sub>e ar</sub>bit<sub>rary:</sub> f<sub>or</sub> $\dot { \mathrm { \mathbf { a } } } \ \cdots \mathrm { b a d } ^ { \mathrm { \mathbf { \prime } } }$ action c and $\mathrm { { a \dot { \tilde { \ g o o d } } \vec { \Omega } } }$ action b $( \mathrm { i } . \mathrm { e } . , \mathrm { } e _ { c } ^ { \top } L \hat { q }$ i<sub>s</sub> l<sub>arge w</sub>hil<sub>e</sub> $e _ { b } ^ { \top } L \hat { q }$ is small), we would like $p _ { c }$ t<sub>o</sub> b<sub>e sma</sub>ll <sub>an</sub>d $p _ { b }$ t<sub>o</sub> b<sub>e</sub> l<sub>arge</sub> t<sub>o ensure</sub> $p ^ { T } L \hat { q } ( x ) - \lambda ^ { T } L \hat { q } ( x )$ to be small; however, if c happens to lie in $\mathrm { p a t h } _ { \tau } ( b )$ <sub>,</sub> th<sub>a</sub>t <sub>can cause</sub> th<sub>e</sub> $\frac { \lambda _ { b } } { p _ { c } }$ t<sub>erm</sub> t<sub>o</sub> b<sub>e very</sub> l<sub>arge.</sub> H<sub>ow s</sub>h<sub>ou</sub>ld th<sub>e</sub> i<sub>n-</sub>t<sub>ree</sub> $\tau$ Tbe chosen so that the actions lying on each path are <sub>ones we are w</sub>illi<sub>ng</sub> t<sub>o p</sub>l<sub>ay</sub>? O<sub>ur answer</sub> i<sub>s</sub> t<sub>o</sub> l<sub>e</sub>t th<sub>e</sub> t<sub>ree</sub> f<sub>o</sub>ll<sub>ow</sub> th<sub>e es</sub>ti<sub>ma</sub>t<sub>e</sub>d l<sub>osses.</sub> W<sub>e</sub> t<sub>a</sub>k<sub>e</sub> $\tau$ t<sub>o</sub> b<sub>e</sub> $ { L } \hat { q } ( x )$ -decreasin<sub>g</sub> (that is for a directed ed<sub>g</sub>e $e = ( b , c ) , e _ { b } ^ { \top } L \hat { q } ( x ) \geq e _ { c } ^ { \top } L \hat { q } ( x )$ <sub>;</sub> S<sub>ee</sub> D<sub>e</sub>f C<sub>.</sub>1 f<sub>or</sub> f<sub>orma</sub>l definition), so that estimated loss decreases alon<sub>g</sub> ever<sub>y p</sub>ath toward the root and, in <sub>p</sub>articular, the <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l b<sub>es</sub>t <sub>ac</sub>ti<sub>on</sub> <sub>s</sub>it<sub>s</sub> <sub>a</sub>t th<sub>e</sub> <sub>roo</sub>t<sub>.</sub> With thi<sub>s</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f $\tau _ { \ast }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> th<sub>a</sub>t $p$ <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide</sub> i<sub>s</sub> <sub>mono</sub>t<sub>on</sub>i<sub>ca</sub>ll<sub>y</sub> d<sub>ecreas</sub>i<sub>ng</sub> <sub>o</sub>f i<sub>n</sub> th<sub>e</sub> l<sub>oss,</sub> th<sub>e</sub> <sub>coverage</sub> t<sub>erm</sub> i<sub>s</sub> <sub>upper</sub> b<sub>oun</sub>d<sub>e</sub>d <sup>b</sup><sub>y</sub> $\begin{array} { r } { k \sum _ { b } \bigl ( \frac { \lambda _ { b } } { p _ { b } } + 1 \bigr ) } \end{array}$ <sub>, w</sub>hi<sub>c</sub>h <sub>a</sub>ll<sub>ows us</sub> t<sub>o</sub> b<sub>oun</sub>d $\mathcal { V } _ { \gamma } ( q , p )$ su<sup>b</sup>se<sub>q</sub>uent<sup>l</sup><sub>y</sub>.

![](images/2de1c08778c578ed7f138c069a8210edf6f2d5838f5a2119126fc36b5cc64665.jpg)

Fi<sub>gure</sub> 1<sub>:</sub> A <sub>spann</sub>i<sub>ng</sub> i<sub>n-</sub>t<sub>ree</sub> $\tau$ of the neighborhood graph (Π, E), with root root $\curvearrowleft$ ever<sub>y</sub> e<sup>d</sup><sub>g</sub>e <sup>i</sup>s directed toward the root. The dashed edges form path $\boldsymbol { \tau } ^ { ( a ) }$ T, the directed path from action a to the Troot. The construction follows Lattimore & Szepesvari (2020).´  
Algorithm 3 SquareCB.PMSide for Locally observable games   
1: parameters:   
2<sub>:</sub> Context x with redicted outcome distribution ${ \hat { q } } ( x )$   
3<sub>:</sub> Ex<sub>p</sub>loration <sub>p</sub>arameter $\gamma > 0 , \mu = k$ <sub>an</sub>d $\begin{array} { r } { \kappa = \frac { \gamma } { 8 k m ^ { 2 } } } \end{array}$   
4: For each action a ∈ Π, compute $\hat { \ell } _ { a } = e _ { a } ^ { \top } \pmb { L } \hat { q } ( x )$ <sub>an</sub>d l<sub>e</sub>t $b = \arg \operatorname* { m i n } _ { a \in \Pi } \hat { \ell } _ { t , a }$   
5: For each a ∈ $\Pi \setminus \{ b \}$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> $\begin{array} { r } { p _ { a } = \frac { 1 } { \mu + \kappa ( \hat { \ell } _ { a } - \hat { \ell } _ { b } ) } ; \mathrm { l e t } p _ { b } = 1 - \sum _ { a \in \Pi \setminus \{ b \} } p _ { a } . } \end{array}$   
6: return $p$

Ofline oracle-eficient algorithms In Appendix D, we present a close variant of E2D.PMSide, namel<sub>y</sub> OE2D.PMSide (Ofline Estimation to Decisions for Partial Monitorin<sub>g</sub> with Side Information), that eficiently reduces contextual partial monitoring to ofline estimation. Our OE2D.PMSide <sub>g</sub>eneralizes FALCON (Simchi-Levi & Xu, 2022) and its linear version (Xu & Zeevi, 2020). Similar to E2D<sub>.</sub>PMS<sub>ide,</sub> OE2D<sub>.</sub>PMS<sub>ide</sub> h<sub>as</sub> t<sub>wo vers</sub>i<sub>ons, one</sub> th<sub>a</sub>t <sub>emp</sub>l<sub>oys</sub> Al<sub>g.</sub> 2 f<sub>or</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e</sub> setting, the other employs Alg. 3 for the locally observable setting, which enjoys $O ( T ^ { 2 / 3 } )$ <sub>an</sub>d $O ( { \sqrt { T } } )$ <sub>regre</sub>t i<sub>n</sub> th<sub>e respec</sub>ti<sub>ve se</sub>tti<sub>ngs.</sub> C<sub>ompare</sub>d <sub>w</sub>ith E2D<sub>.</sub>PMS<sub>ide</sub> th<sub>a</sub>t <sub>ma</sub>k<sub>es</sub> $O ( T )$ <sub>ca</sub>ll<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>on</sub>li<sub>ne</sub> <sub>ou</sub>t<sub>come</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>orac</sub>l<sub>e,</sub> OE2D<sub>.</sub>PMS<sub>ide</sub> <sub>on</sub>l<sub>y</sub> <sub>ma</sub>k<sub>es</sub> ${ \cal O } ( \log T )$ <sub>ca</sub>ll<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>o</sub>fli<sub>ne</sub> <sub>ou</sub>t<sub>come</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e, ma</sub>ki<sub>ng</sub> it <sub>su</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or app</sub>li<sub>ca</sub>ti<sub>ons w</sub>ith l<sub>ow compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>cos</sub>t <sub>requ</sub>i<sub>remen</sub>t<sub>s.</sub>

## 4 E<sub>mpirical studies</sub>

We evaluate SquareCB.PMSide in two settin<sub>g</sub>s we discussed in Section 1 (Exam<sub>p</sub>les 1 & 2): LLM <sub>rou</sub>ti<sub>ng</sub> <sub>an</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l l<sub>a</sub>b<sub>e</sub>l<sub>-e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>game.</sub>

LLM routing. We evaluate the routin roblem on a cached version of the BoolQ benchmark conta<sup>i</sup>n<sup>i</sup>ng 12,648 va<sup>lid</sup> responses <sup>f</sup>rom <sup>b</sup>ot<sup>h</sup> mo<sup>d</sup>e<sup>l</sup>s. <sup>W</sup>e use Qwen2.5-0.5B-Instruct as t<sup>h</sup>e wea<sup>k</sup> mo<sup>d</sup>e<sup>l</sup> an<sup>d</sup> Llama-2-13B-Chat as t<sup>h</sup>e strong mo<sup>d</sup>e<sup>l</sup>. <sup>Th</sup>e rout<sup>i</sup>ng outcome recor<sup>d</sup>s w<sup>h</sup>et<sup>h</sup>er t<sup>h</sup>e<sup>i</sup>r cached responses disagree. Each prompt is represented by a 138-dimensional context comprising 10 weak-model uncertainty statistics, 64 PCA components of its hidden states, and 64 PCA components <sub>o</sub>f <sub>a con</sub>t<sub>ex</sub>t<sub>-</sub>f<sub>ree promp</sub>t <sub>em</sub>b<sub>e</sub>ddi<sub>ng.</sub>

We com<sub>p</sub>are linear-lo<sub>g</sub>istic CBPSide (Bartok et al., 2012a; An et al., 2026),´ SquareCB.PMSide (ours), ETC (Ex<sub>p</sub>lore-Then-Commit, ada<sub>p</sub>ted for a<sub>pp</sub>le tastin<sub>g</sub>) (Garivier et al., 2016) with both linear lo<sub>g</sub>istic regression and a nonlinear 15-leaf histogram gradient-boosting (HGB) model, Ba<sub>y</sub>esian-logistic PG-TS (Grant & Leslie, 2021), and a random baseline. We consider nine values of $\ell _ { 0 1 }$ between 1.8 and 3.3 and evaluate each configuration over 20 paired random permutations. Figure 2a reports the <sub>mean rea</sub>li<sub>ze</sub>d t<sub>o</sub>t<sub>a</sub>l <sub>cos</sub>t<sub>.</sub>

F<sub>or eac</sub>h <sub>va</sub>l<sub>ue o</sub>f $\ell _ { 0 1 }$ <sub>we</sub> t<sub>une every a</sub>l<sub>gor</sub>ith<sub>m separa</sub>t<sub>e</sub>l<sub>y, se</sub>l<sub>ec</sub>ti<sub>ng</sub> th<sub>e con</sub>fi<sub>gura</sub>ti<sub>on w</sub>ith th<sub>e</sub> l<sub>owes</sub>t t<sub>o</sub>t<sub>a</sub>l <sub>cos</sub>t<sub>.</sub> F<sub>or a</sub>ll <sub>me</sub>th<sub>o</sub>d<sub>s excep</sub>t th<sub>e ran</sub>d<sub>om</sub> b<sub>ase</sub>li<sub>ne</sub> thi<sub>s amoun</sub>t<sub>s</sub> t<sub>o c</sub>h<sub>oos</sub>i<sub>ng a mu</sub>lti<sub>p</sub>li<sub>er</sub> $s \in \{ 0 . 1 , 0 . 3 , 1 , 3 , 1 0 \}$ <sub>, w</sub>hi<sub>c</sub>h <sub>sca</sub>l<sub>es</sub> th<sub>e con</sub>fid<sub>ence ra</sub>di<sub>us</sub> i<sub>n</sub> CBPSid<sub>e,</sub> th<sub>e exp</sub>l<sub>ora</sub>ti<sub>on parame</sub>t<sub>er</sub> s ⋅ κ in both SquareCB.PMSide variants, the initial exploration budget $s \cdot T ^ { 2 / 3 }$ i<sub>n</sub> ETC<sub>, an</sub>d th<sub>e pr</sub>i<sub>or</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> PG<sub>-</sub>TS<sub>.</sub> Th<sub>e ran</sub>d<sub>om</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>rou</sub>t<sub>es eac</sub>h <sub>query</sub> t<sub>o</sub> th<sub>e s</sub>t<sub>rong mo</sub>d<sub>e</sub>l <sub>w</sub>ith <sub>a</sub> fi<sub>xe</sub>d <sub>pro</sub>b<sub>a</sub>bilit<sub>y,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f <sub>con</sub>t<sub>ex</sub>t<sub>, swep</sub>t <sub>over</sub> $\{ 0 , 0 . 2 , 0 . 4 , 0 \mathrm { \bar { . } 6 , 0 . 8 , 1 } \}$ }. All methods, ours i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d<sub>, are</sub> th<sub>ere</sub>f<sub>ore</sub> t<sub>une</sub>d <sub>w</sub>ith hi<sub>n</sub>d<sub>s</sub>i<sub>g</sub>ht k<sub>now</sub>l<sub>e</sub>d<sub>ge o</sub>f th<sub>e</sub> l<sub>oss.</sub> F<sub>ur</sub>th<sub>er</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>an</sub>d <sub>resu</sub>lt<sub>s are prov</sub>id<sub>e</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> E<sub>.</sub>

![](images/56a1bb7611c113bee99aa71508d716e5547c84a11382bfc660f9b9ef8278f811.jpg)  
(a) LLM routin<sub>g</sub>: Total cost vs $\ell _ { 0 1 }$

![](images/1b3b5ec308319f4ffe4d1f66adcab0a7c952cc0b1cd8ff4d4c4186521f1c1531.jpg)  
(b) LE <sub>g</sub>ame: Total cost vs c  
Fi<sub>gure</sub> 2<sub>:</sub> C<sub>ompar</sub>i<sub>son</sub> <sub>o</sub>f <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or</sub> <sub>an</sub>d LLM <sub>rou</sub>ti<sub>ng</sub> <sub>an</sub>d L<sub>a</sub>b<sub>e</sub>l <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>game</sub>

A<sub>cross</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d l<sub>oss se</sub>tti<sub>ngs,</sub> b<sub>o</sub>th th<sub>e</sub> li<sub>near-</sub>l<sub>og</sub>i<sub>s</sub>ti<sub>c an</sub>d <sub>non</sub>li<sub>near</sub> HGB i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>ti<sub>ons o</sub>f S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide ac</sub>hi<sub>eve</sub> l<sub>ower mean rea</sub>li<sub>ze</sub>d t<sub>o</sub>t<sub>a</sub>l <sub>cos</sub>t th<sub>an</sub> CBPS<sub>ide, w</sub>hil<sub>e</sub> th<sub>e non</sub>li<sub>near</sub> HGB <sub>var</sub>i<sub>an</sub>t <sub>a</sub>l<sub>so ou</sub>t<sub>per</sub>f<sub>orms</sub> PG<sub>-</sub>TS<sub>.</sub> Th<sub>ese resu</sub>lt<sub>s</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e a prac</sub>ti<sub>ca</sub>l <sub>a</sub>d<sub>van</sub>t<sub>age o</sub>f th<sub>e orac</sub>l<sub>e-e</sub>fi<sub>c</sub>i<sub>en</sub>t S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on:</sub> it <sub>can</sub> i<sub>ncorpora</sub>t<sub>e</sub> fl<sub>ex</sub>ibl<sub>e non</sub>li<sub>near pre</sub>di<sub>c</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s w</sub>ith<sub>ou</sub>t <sub>mo</sub>dif<sub>y</sub>i<sub>ng</sub> th<sub>e surroun</sub>di<sub>ng rou</sub>ti<sub>ng po</sub>li<sub>cy.</sub> F<sub>ur</sub>th<sub>ermore, genera</sub>ll<sub>y</sub> ETC i<sub>s expec</sub>t<sub>e</sub>d t<sub>o per</sub>f<sub>orm worse</sub> d<sub>ue</sub> t<sub>o</sub> $O ( T ^ { 2 / 3 } )$ regret, however here due to limited test data size 12,648 compared to larger dimension 138, it performs competitively. See Figure 2a.

Contextual label-eficient prediction. We conduct an analogous study for the contextual label-<sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on game.</sub> L<sub>oca</sub>l <sub>o</sub>b<sub>serva</sub>bilit<sub>y requ</sub>i<sub>res</sub> th<sub>e exper</sub>t<sub>-query cos</sub>t t<sub>o sa</sub>ti<sub>s</sub>f<sub>y</sub> $c < 0 . 5 ;$ accordingl<sub>y</sub>, we evaluate five values of c in the range $0 . 1 \leq c < 0 . 5$ <sub>.</sub> B<sub>ecause</sub> PG<sub>-</sub>TS i<sub>s</sub> d<sub>es</sub>i<sub>g</sub>n<sub>e</sub>d <sub>spec</sub>ifi<sub>ca</sub>ll<sub>y</sub> f<sub>or app</sub>l<sub>e-</sub>t<sub>as</sub>ti<sub>ng games an</sub>d i<sub>s</sub> th<sub>us</sub> i<sub>napp</sub>li<sub>ca</sub>bl<sub>e</sub> t<sub>o</sub> thi<sub>s pro</sub>bl<sub>em, we rep</sub>l<sub>ace</sub> it <sub>w</sub>ith RandCBPSide (Heuillet et al., 2024). We <sub>g</sub>enerate the data s<sub>y</sub>ntheticall<sub>y</sub> from a linear lo<sub>g</sub>istic model <sub>w</sub>ith 10<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>con</sub>t<sub>ex</sub>t <sub>vec</sub>t<sub>ors.</sub> F<sub>ur</sub>th<sub>er,</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s are rov</sub>id<sub>e</sub>d i<sub>n</sub> A <sub>en</sub>di<sub>x</sub> E<sub>.</sub> O<sub>ur me</sub>th<sub>o</sub>d <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orms</sub> ETC <sub>an</sub>d CBPSid<sub>e</sub> <sub>across</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>se</sub>tti<sub>ngs</sub> <sub>an</sub>d <sub>rema</sub>i<sub>ns</sub> <sub>compe</sub>titi<sub>ve</sub> <sub>w</sub>ith R<sub>a</sub>ndCBPSid<sub>e, ou</sub>t<sub>pe</sub>rf<sub>o</sub>rmin<sub>g</sub> it f<sub>o</sub>r <sub>so</sub>m<sub>e o</sub>f th<sub>e expe</sub>rt<sub>-que</sub>r<sub>y cos</sub>t<sub>s.</sub> M<sub>o</sub>r<sub>eove</sub>r<sub>,</sub> R<sub>a</sub>ndCBPSid<sub>e</sub> r<sub>equ</sub>ir<sub>es</sub> <sub>compara</sub>ti<sub>ve</sub>l<sub>y expens</sub>i<sub>ve con</sub>fid<sub>ence-w</sub>idth <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons, w</sub>h<sub>ereas our me</sub>th<sub>o</sub>d i<sub>s compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y</sub> <sub>c</sub>h<sub>eaper.</sub> S<sub>ee</sub> Fi<sub>gure</sub> 2b<sub>.</sub>

## 5 C<sub>onclusion</sub> <sub>and</sub> <sub>future</sub> <sub>work</sub>

O<sub>ur wor</sub>k <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng w</sub>ith <sub>op</sub>ti<sub>ma</sub>l <sub>regre</sub>t <sub>w</sub>ith <sub>genera</sub>l f<sub>unc</sub>ti<sub>on approx</sub>i<sub>ma</sub>ti<sub>on.</sub> E<sub>xper</sub>i<sub>men</sub>t<sub>s on</sub> LLM <sub>rou</sub>ti<sub>ng an</sub>d <sub>a mo</sub>difi<sub>e</sub>d l<sub>a</sub>b<sub>e</sub>l<sub>-e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>ga</sub>m<sub>e suppo</sub>rt th<sub>ese gua</sub>r<sub>a</sub>nt<sub>ees.</sub> O<sub>u</sub>r <sub>a</sub>n<sub>a</sub>l<sub>ys</sub>i<sub>s</sub> i<sub>s</sub> r<sub>es</sub>tri<sub>c</sub>t<sub>e</sub>d t<sub>o</sub> n<sub>o</sub>n<sub>-</sub>d<sub>ege</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e ga</sub>m<sub>es.</sub> L<sub>e</sub>mm<sub>a</sub> G<sub>.</sub>3<sub>, w</sub>hi<sub>c</sub>h <sub>un</sub>d<sub>er</sub>li<sub>es</sub> th<sub>e</sub> $O ( 1 / \gamma )$ bound for locally observable games, relies on non-degeneracy. We conjecture th<sub>a</sub>t <sub>our</sub> S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide</sub> <sub>m</sub>i<sub>g</sub>ht <sub>wor</sub>k f<sub>or</sub> d<sub>egenera</sub>t<sub>e</sub> <sub>games</sub> <sub>as</sub> <sub>we</sub>ll <sub>w</sub>ith <sub>m</sub>i<sub>nor</sub> <sub>mo</sub>difi<sub>ca</sub>ti<sub>on.</sub> H<sub>owever,</sub> th<sub>e</sub>i<sub>r ana</sub>l<sub>ys</sub>i<sub>s nee</sub>d <sub>a care</sub>f<sub>u</sub>l <sub>approac</sub>h <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> l<sub>e</sub>ft t<sub>o</sub> f<sub>u</sub>t<sub>ure wor</sub>k<sub>.</sub> A<sub>no</sub>th<sub>er</sub> i<sub>n</sub>t<sub>eres</sub>ti<sub>ng</sub> <sub>ques</sub>ti<sub>on</sub> i<sub>s, can our a</sub>l<sub>gor</sub>ith<sub>m</sub> b<sub>e genera</sub>li<sub>ze</sub>d t<sub>o</sub> th<sub>e se</sub>tti<sub>ng w</sub>h<sub>en</sub> th<sub>e</sub> l<sub>oss ma</sub>t<sub>r</sub>i<sub>x an</sub>d th<sub>e</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> i<sub>s</sub> <sub>con</sub>t<sub>ex</sub>t<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t? Thi<sub>s</sub> <sub>seems</sub> <sub>prac</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>re</sub>l<sub>evan</sub>t f<sub>or</sub> <sub>e.g.</sub> LLM <sub>rou</sub>ti<sub>ng</sub> <sub>app</sub>li<sub>ca</sub>ti<sub>ons,</sub> <sub>s</sub>i<sub>nce</sub> th<sub>e</sub> <sub>cos</sub>t <sub>o</sub>f <sub>wrong</sub>l<sub>y</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> <sub>wea</sub>k <sub>mo</sub>d<sub>e</sub>l <sub>may</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e</sub> i<sub>mpor</sub>t<sub>ance</sub> <sub>o</sub>f th<sub>e</sub> <sub>promp</sub>t<sub>.</sub> Furthermore, extendin<sub>g</sub> our work to contextual linear <sub>p</sub>artial monitorin<sub>g</sub> (Kirschner et al., 2023) and contextual <sub>p</sub>artial monitorin<sub>g</sub> with infinite outcomes (Lattimore, 2022) are also interestin<sub>g</sub> future di<sub>rec</sub>ti<sub>ons.</sub>

## AI <sub>use statement</sub>

I<sub>n</sub> thi<sub>s</sub> <sub>wor</sub>k<sub>,</sub> <sub>we</sub> <sub>use</sub>d <sub>genera</sub>ti<sub>ve</sub> AI t<sub>oo</sub>l<sub>s</sub> f<sub>or</sub> <sub>co</sub>di<sub>ng</sub> f<sub>or</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s,</sub> <sub>wr</sub>iti<sub>ng</sub> th<sub>ose</sub> <sub>exper</sub>i<sub>men</sub>t d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>an</sub>d <sub>po</sub>li<sub>s</sub>hi<sub>ng</sub> th<sub>e</sub> <sub>wr</sub>iti<sub>ng.</sub> W<sub>e</sub> did <sub>rev</sub>i<sub>ewe</sub>d th<sub>ose</sub> <sub>co</sub>di<sub>ng</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> <sub>correc</sub>t<sub>ness.</sub> W<sub>e</sub> h<sub>ave</sub> <sub>no</sub>t <sub>use</sub>d <sub>genera</sub>ti<sub>ve</sub> AI t<sub>oo</sub>l<sub>s</sub> f<sub>or wr</sub>iti<sub>ng proo</sub>f <sub>or re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>wor</sub>k<sub>s, an</sub>d th<sub>e res</sub>t <sub>o</sub>f th<sub>e requ</sub>i<sub>re</sub>d di<sub>sc</sub>l<sub>osure</sub> t<sub>as</sub>k<sub>s are no</sub>t <sub>app</sub>li<sub>ca</sub>bl<sub>e</sub> t<sub>o</sub> thi<sub>s wor</sub>k<sub>.</sub> W<sub>e</sub> h<sub>ave rev</sub>i<sub>ewe</sub>d <sub>a</sub>ll AI<sub>-ass</sub>i<sub>s</sub>t<sub>e</sub>d <sub>wor</sub>k<sub>.</sub> W<sub>e</sub> t<sub>a</sub>k<sub>e respons</sub>ibilit<sub>y</sub> f<sub>or</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>con</sub>t<sub>en</sub>t <sub>o</sub>f thi<sub>s wor</sub>k<sub>,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> t<sub>ex</sub>t<sub>, c</sub>l<sub>a</sub>i<sub>ms or ar</sub>tif<sub>ac</sub>t<sub>s pro</sub>d<sub>uce</sub>d <sub>w</sub>ith th<sub>e a</sub>id <sub>o</sub>f <sub>genera</sub>ti<sub>ve</sub> AI<sub>.</sub>

## E<sub>thics statement</sub>

Thi<sub>s wor</sub>k d<sub>eve</sub>l<sub>ops</sub> l<sub>earn</sub>i<sub>ng a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or rou</sub>ti<sub>ng quer</sub>i<sub>es among</sub> l<sub>arge</sub> l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s.</sub> S<sub>uc</sub>h <sub>me</sub>th<sub>o</sub>d<sub>s may re</sub>d<sub>uce</sub> i<sub>n</sub>f<sub>erence cos</sub>t<sub>s an</sub>d <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>resource consump</sub>ti<sub>on w</sub>hil<sub>e</sub> i<sub>mprov</sub>i<sub>ng</sub> <sub>access</sub> t<sub>o</sub> <sub>capa</sub>bl<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> H<sub>owever,</sub> <sub>rou</sub>ti<sub>ng</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> <sub>may</sub> <sub>propaga</sub>t<sub>e</sub> <sub>or</sub> <sub>amp</sub>lif<sub>y</sub> bi<sub>ases</sub> <sub>presen</sub>t i<sub>n</sub> th<sub>e</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>an</sub>d d<sub>a</sub>t<sub>a,</sub> <sub>po</sub>t<sub>en</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>pro</sub>d<sub>uc</sub>i<sub>ng</sub> <sub>unequa</sub>l <sub>per</sub>f<sub>ormance</sub> <sub>across</sub> d<sub>emograp</sub>hi<sub>c</sub> <sub>groups,</sub> l<sub>anguages, or app</sub>li<sub>ca</sub>ti<sub>on</sub> d<sub>oma</sub>i<sub>ns.</sub> F<sub>ur</sub>th<sub>ermore, quer</sub>i<sub>es may con</sub>t<sub>a</sub>i<sub>n sens</sub>iti<sub>ve</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on,</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y</sub> <sub>w</sub>h<sub>en</sub> <sub>rou</sub>t<sub>e</sub>d t<sub>o</sub> <sub>ex</sub>t<sub>erna</sub>ll<sub>y</sub> h<sub>os</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> O<sub>ur</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>op</sub>ti<sub>m</sub>i<sub>ze</sub> <sub>a</sub> <sub>spec</sub>ifi<sub>e</sub>d <sub>per</sub>f<sub>ormance</sub> objective and do not independently guarantee fairness, privacy, reliability, or safety. Practical d<sub>ep</sub>l<sub>oymen</sub>t<sub>s</sub> <sub>s</sub>h<sub>ou</sub>ld th<sub>ere</sub>f<sub>ore</sub> i<sub>ncorpora</sub>t<sub>e</sub> <sub>appropr</sub>i<sub>a</sub>t<sub>e</sub> d<sub>a</sub>t<sub>a</sub> <sub>pro</sub>t<sub>ec</sub>ti<sub>on,</sub> <sub>su</sub>b<sub>group</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> h<sub>uman</sub> <sub>overs</sub>i<sub>g</sub>ht<sub>, an</sub>d f<sub>a</sub>llb<sub>ac</sub>k <sub>mec</sub>h<sub>an</sub>i<sub>sms.</sub>

## R<sub>eproducibility</sub> <sub>statement</sub>

W<sub>e</sub> <sub>prov</sub>id<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons,</sub> <sub>assump</sub>ti<sub>ons,</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>ons,</sub> <sub>an</sub>d <sub>regre</sub>t <sub>ana</sub>l<sub>yses</sub> i<sub>n</sub> th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> <sub>paper an</sub>d <sub>appen</sub>di<sub>ces.</sub> Th<sub>e exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>sec</sub>ti<sub>on repor</sub>t<sub>s</sub> th<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s, preprocess</sub>i<sub>ng proce</sub>d<sub>ures, mo</sub>d<sub>e</sub>l <sub>con</sub>fi<sub>gura</sub>ti<sub>ons,</sub> b<sub>ase</sub>li<sub>nes, eva</sub>l<sub>ua</sub>ti<sub>on me</sub>t<sub>r</sub>i<sub>cs, an</sub>d h<sub>yperparame</sub>t<sub>ers use</sub>d i<sub>n our s</sub>t<sub>u</sub>d<sub>y.</sub> W<sub>e w</sub>ill <sub>re</sub>l<sub>ease</sub> th<sub>e</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on, exper</sub>i<sub>men</sub>t <sub>con</sub>fi<sub>gura</sub>ti<sub>ons, an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on scr</sub>i<sub>p</sub>t<sub>s</sub> l<sub>a</sub>t<sub>er</sub> t<sub>o</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>e repro</sub>d<sub>uc</sub>ti<sub>on</sub> of our results, subject to the licenses and access conditions of the underlying datasets and language <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> W<sub>e a</sub>l<sub>so repor</sub>t <sub>resu</sub>lt<sub>s across mu</sub>lti<sub>p</sub>l<sub>e ran</sub>d<sub>om see</sub>d<sub>s</sub> t<sub>o accoun</sub>t f<sub>or exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>var</sub>i<sub>a</sub>bilit<sub>y.</sub>

## A<sub>cknowledgments</sub>

This work was <sub>p</sub>artl<sub>y</sub> su<sub>pp</sub>orted b<sub>y</sub> the National Research Foundation of Korea (NRF) <sub>g</sub>rant funded b<sub>y</sub> the Korean <sub>g</sub>overnment (MSIT) in 2026 (Grant No. RS-2026-25612350, Project Title: D<sub>eve</sub>l<sub>opmen</sub>t <sub>o</sub>f <sub>an</sub> E<sub>n</sub>d<sub>-</sub>t<sub>o-</sub>E<sub>n</sub>d D<sub>es</sub>i<sub>gn an</sub>d V<sub>a</sub>lid<sub>a</sub>ti<sub>on</sub> F<sub>ramewor</sub>k f<sub>or</sub> M<sub>a</sub>th<sub>ema</sub>ti<sub>ca</sub>ll<sub>y</sub> P<sub>r</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub>d Next-Generation AI Architectures and Learnin<sub>g</sub> Al<sub>g</sub>orithms, contribution 45%). This work was <sub>p</sub>artl<sub>y</sub> su<sub>pp</sub>orted b<sub>y</sub> Institute of Information & communications Technolo<sub>gy</sub> Plannin<sub>g</sub> & Evaluation (IITP) <sub>g</sub>rant funded b<sub>y</sub> the Korea <sub>g</sub>overnment(MSIT) (RS-2019-II191906, Artificial Intelli<sub>g</sub>ence Graduate School Pro<sub>g</sub>ram(POSTECH), contribution 5%; RS-2026-25616313, AI Star Fellowshi<sub>p</sub>(POSTECH, Interdisci<sub>p</sub>linar<sub>y</sub> Pro<sub>g</sub>ram), contribution 50%). This work was su<sub>pp</sub>orted in <sub>p</sub>art b<sub>y</sub> the Nationa Science Foundation under <sub>g</sub>rant CCF-2327013, IIS-2440266 (CAREER) and Meta Platforms, Inc.

## R<sub>eferences</sub>

Al<sub>e</sub>kh A<sub>garwa</sub>l<sub>,</sub> Sh<sub>am</sub> K<sub>a</sub>k<sub>a</sub>d<sub>e,</sub> Ak<sub>s</sub>h<sub>ay</sub> K<sub>r</sub>i<sub>s</sub>h<sub>namur</sub>th<sub>y,</sub> <sub>an</sub>d W<sub>en</sub> S<sub>un.</sub> Fl<sub>am</sub>b<sub>e:</sub> St<sub>ruc</sub>t<sub>ura</sub>l <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> and representation learning of low rank mdps. Advances in neural information processing systems, 33<sub>:</sub>20095<sub>–</sub>20107<sub>,</sub> 2020<sub>.</sub>

B<sub>y</sub>eon<sub>gw</sub>oo An<sub>,</sub> Ka<sub>p</sub>ilan Bala<sub>g</sub>o<sub>p</sub>alan<sub>,</sub> H<sub>y</sub>o<sub>w</sub>on Wi<sub>,</sub> Seh<sub>w</sub>a Jeon<sub>g,</sub> Ji<sub>u</sub>n Jeon<sub>g,</sub> K<sub>y</sub>o<sub>u</sub>n<sub>g</sub>seok Jan<sub>g,</sub> N<sub>oseo</sub>n<sub>g</sub> P<sub>a</sub>rk<sub>,</sub> Gi<sub>-</sub>S<sub>oo</sub> Kim<sub>, a</sub>nd K<sub>wa</sub>n<sub>g-</sub>S<sub>u</sub>n<sub>g</sub> J<sub>u</sub>n<sub>.</sub> Efi<sub>c</sub>i<sub>e</sub>nt <sub>a</sub>l<sub>go</sub>rithm<sub>s</sub> f<sub>o</sub>r <sub>co</sub>nt<sub>ex</sub>t<sub>ua</sub>l <sub>app</sub>l<sub>e</sub> t<sub>as</sub>tin<sub>g</sub> <sub>w</sub>ith l<sub>og-</sub>l<sub>oss.</sub> I<sub>n</sub> ICML 2026 DEMO <sub>wor</sub>k<sub>s</sub>h<sub>op,</sub> 2026<sub>.</sub>

P<sub>e</sub>t<sub>er</sub> A<sub>uer,</sub> Ni<sub>co</sub>l<sub>o</sub> C<sub>esa-</sub>Bi<sub>anc</sub>hi<sub>,</sub> Y<sub>oav</sub> F<sub>reun</sub>d<sub>, an</sub>d R<sub>o</sub>b<sub>er</sub>t E S<sub>c</sub>h<sub>ap</sub>i<sub>re.</sub> Th<sub>e</sub> N<sub>ons</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> M<sub>u</sub>lti<sub>arme</sub>d\` Bandit Problem. SIAM J. Com<sub>p</sub>ut., 32(1):48–77, jan 2003. ISSN 0097-5397. doi: 10.1137/ S0097539701398375<sub>.</sub>

G<sub>a</sub>b<sub>or</sub> B<sub>ar</sub>t´ <sub>o</sub>k <sub>an</sub>d C<sub>sa</sub>b<sub>a</sub> S<sub>zepesv</sub>´ <sub>ar</sub>i<sub>.</sub> P<sub>ar</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng w</sub>ith <sub>s</sub>id<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on.</sub> I<sub>n</sub>´ I<sub>n</sub>t<sub>erna</sub>ti<sub>ona</sub>l Conference on Algorithmic Learning Theory, pp. 305–319. Springer, 2012.

G<sub>a</sub>b<sub>or</sub> B<sub>ar</sub>t ´ <sub>o</sub>k <sub>e</sub>t <sub>a</sub>l<sub>.</sub> P<sub>ar</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> <sub>w</sub>ith <sub>s</sub>id<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on.</sub> I<sub>n</sub> ´ I<sub>n</sub> P<sub>roc.</sub> ALT<sub>,</sub> 2012<sub>a.</sub>

G<sub>a</sub>b<sub>or</sub> B<sub>ar</sub>t ´ <sub>o</sub>k <sub>e</sub>t <sub>a</sub>l<sub>.</sub> A<sub>n</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> f<sub>or</sub> fi<sub>n</sub>it<sub>e</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng.</sub> ´ I<sub>n</sub> P<sub>roc.</sub> ICML<sub>,</sub> 2012b<sub>.</sub>

G<sub>a</sub>b<sub>or</sub> B<sub>ar</sub>t ´ <sub>o</sub>k<sub>,</sub> D ´ <sub>av</sub>id P ´ <sub>a</sub>l<sub>,</sub> <sub>an</sub>d C<sub>sa</sub>b<sub>a</sub> S<sub>zepesv</sub> ´ <sub>ar</sub>i<sub>.</sub> Mi<sub>n</sub>i<sub>max</sub> <sub>regre</sub>t <sub>o</sub>f fi<sub>n</sub>it<sub>e</sub> <sub>par</sub>ti<sub>a</sub>l<sub>-mon</sub>it<sub>or</sub>i<sub>ng</sub> <sub>games</sub>´ in stochastic environments. In Sham M. Kakade and Ulrike von Luxbur<sub>g</sub> (eds.), Proceedin<sub>g</sub>s of the 24th Annual Conference on Learning Theory, volume 19 of Proceedings of Machine <sup>L</sup>earn<sup>i</sup>ng <sup>R</sup>esearc<sup>h</sup>, pp. <sup>133</sup>–<sup>154</sup>, <sup>B</sup>u<sup>d</sup>apest, <sup>H</sup>ungary, <sup>09</sup>–<sup>11</sup> <sup>J</sup>un <sup>2011</sup>. <sup>PMLR</sup>. <sup>URL</sup> https: //proceedings.mlr.press/v19/bartok11a.html.

Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use large language models while <sub>re</sub>d<sub>uc</sub>i<sub>ng cos</sub>t <sub>an</sub>d i<sub>mprov</sub>i<sub>ng per</sub>f<sub>ormance. ar</sub>Xi<sub>v prepr</sub>i<sub>n</sub>t <sub>ar</sub>Xi<sub>v:</sub>2305<sub>.</sub>05176<sub>,</sub> 2023<sub>.</sub>

D<sub>y</sub>l<sub>an</sub> J F<sub>os</sub>t<sub>er an</sub>d Al<sub>exan</sub>d<sub>er</sub> R<sub>a</sub>khli<sub>n.</sub> B<sub>eyon</sub>d UCB<sub>:</sub> O<sub>p</sub>ti<sub>ma</sub>l <sub>an</sub>d <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l b<sub>an</sub>dit<sub>s w</sub>ith regression oracles. Proceedings ofthe International Conference on Machine Learning (ICML), 2020<sub>.</sub>

D<sub>y</sub>lan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The Statistical Complexit<sub>y</sub> of I<sub>n</sub>t<sub>erac</sub>ti<sub>ve</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> M<sub>a</sub>ki<sub>ng.</sub> C<sub>o</sub>RR<sub>,</sub> <sub>a</sub>b<sub>s</sub>/2112<sub>.</sub>1<sub>,</sub> 2021<sub>.</sub>

D<sub>y</sub>l<sub>an</sub> J F<sub>os</sub>t<sub>er,</sub> Al<sub>exan</sub>d<sub>er</sub> R<sub>a</sub>khli<sub>n,</sub> A<sub>yus</sub>h S<sub>e</sub>kh<sub>ar</sub>i<sub>, an</sub>d K<sub>ar</sub>thik S<sub>r</sub>idh<sub>aran.</sub> O<sub>n</sub> th<sub>e comp</sub>l<sub>ex</sub>it<sub>y o</sub>f adversarial decision making. Advances in Neural Information Processing Systems, 35:35404–35417, 2022<sub>.</sub>

A<sub>ure</sub>li<sub>en</sub> G<sub>ar</sub>i<sub>v</sub>i<sub>er,</sub> T<sub>or</sub> L<sub>a</sub>tti<sub>more, an</sub>d E<sub>m</sub>ili<sub>e</sub> K<sub>au</sub>f<sub>mann.</sub> O<sub>n exp</sub>l<sub>ore-</sub>th<sub>en-comm</sub>it <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es.</sub> I<sub>n</sub>´ Advances in Neural Information Processing Systems, volume 29, pp. 784–792. Curran Associates, I<sub>nc.,</sub> 2016<sub>.</sub>

J<sub>ames</sub> A G<sub>ran</sub>t <sub>an</sub>d D<sub>av</sub>id S L<sub>es</sub>li<sub>e.</sub> A<sub>pp</sub>l<sub>e</sub> t<sub>as</sub>ti<sub>ng rev</sub>i<sub>s</sub>it<sub>e</sub>d<sub>:</sub> B<sub>ayes</sub>i<sub>an approac</sub>h<sub>es</sub> t<sub>o par</sub>ti<sub>a</sub>ll<sub>y mon</sub>it<sub>ore</sub>d <sub>on</sub>li<sub>ne</sub> bi<sub>nary c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on. ar</sub>Xi<sub>v prepr</sub>i<sub>n</sub>t <sub>ar</sub>Xi<sub>v:</sub>2109<sub>.</sub>14412<sub>,</sub> 2021<sub>.</sub>

El<sub>a</sub>d H<sub>azan,</sub> A<sub>m</sub>it A<sub>garwa</sub>l<sub>, an</sub>d S<sub>a</sub>t<sub>yen</sub> K<sub>a</sub>l<sub>e.</sub> L<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> R<sub>egre</sub>t Al<sub>gor</sub>ith<sub>ms</sub> f<sub>or</sub> O<sub>n</sub>li<sub>ne</sub> C<sub>onvex</sub> O<sub>p</sub>timization. Mach. Learn., 69(2-3):169–192, 2007.

D<sub>av</sub>id H<sub>e</sub>l<sub>m</sub>b<sub>o</sub>ld <sub>e</sub>t <sub>a</sub>l<sub>.</sub> S<sub>ome</sub> l<sub>a</sub>b<sub>e</sub>l <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t l<sub>earn</sub>i<sub>ng resu</sub>lt<sub>s.</sub> I<sub>n</sub> I<sub>n</sub> P<sub>roc.</sub> C<sub>o</sub>LT<sub>,</sub> 1997<sub>.</sub>

David P Helmbold, Nicholas Littlestone, and Philip M Long. Apple tasting. Information and Com<sub>p</sub>utation, 161(2):85–139, 2000.

M<sub>ax</sub>i<sub>me</sub> H<sub>eu</sub>ill<sub>e</sub>t<sub>,</sub> Ol<sub>a</sub> Ah<sub>ma</sub>d<sub>, an</sub>d A<sub>u</sub>d<sub>rey</sub> D<sub>uran</sub>d<sub>.</sub> R<sub>an</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>con</sub>fid<sub>ence</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng,</sub> 2024<sub>.</sub>

J<sub>o</sub>h<sub>annes</sub> Ki<sub>rsc</sub>h<sub>ner,</sub> T<sub>or</sub> L<sub>a</sub>tti<sub>more, an</sub>d A<sub>n</sub>d<sub>reas</sub> K<sub>rause.</sub> Li<sub>near par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> f<sub>or sequen</sub>ti<sub>a</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>ki<sub>ng:</sub> Al<sub>gor</sub>ith<sub>ms, regre</sub>t b<sub>oun</sub>d<sub>s an</sub>d <sub>app</sub>li<sub>ca</sub>ti<sub>ons.</sub> JMLR<sub>,</sub> 2023<sub>.</sub>

R<sub>o</sub>b<sub>er</sub>t Kl<sub>e</sub>i<sub>n</sub>b<sub>erg</sub> <sub>an</sub>d T<sub>om</sub> L<sub>e</sub>i<sub>g</sub>ht<sub>on.</sub> Th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f k<sub>now</sub>i<sub>ng</sub> <sub>a</sub> d<sub>eman</sub>d <sub>curve:</sub> B<sub>oun</sub>d<sub>s</sub> <sub>on</sub> <sub>regre</sub>t for online posted-price auctions. In 44th Annual IEEE Symposium on Foundations of Computer S<sub>c</sub>i<sub>ence,</sub> 2003<sub>.</sub> P<sub>rocee</sub>di<sub>ngs., pp.</sub> 594<sub>–</sub>605<sub>.</sub> IEEE<sub>,</sub> 2003<sub>.</sub>

J<sub>unpe</sub>i K<sub>om</sub>i<sub>yama,</sub> J<sub>unya</sub> H<sub>on</sub>d<sub>a,</sub> Hi<sub>sas</sub>hi K<sub>as</sub>hi<sub>ma, an</sub>d Hi<sub>ros</sub>hi N<sub>a</sub>k<sub>agawa.</sub> R<sub>egre</sub>t l<sub>ower</sub> b<sub>oun</sub>d <sub>an</sub>d optimal algorithm in dueling bandit problem. In Conference on learning theory, pp. 1141–1154. PMLR<sub>,</sub> 2015<sub>.</sub>

T<sub>or</sub> L<sub>a</sub>tti<sub>more.</sub> Mi<sub>n</sub>i<sub>max regre</sub>t f<sub>or par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng:</sub> I<sub>n</sub>fi<sub>n</sub>it<sub>e ou</sub>t<sub>comes an</sub>d <sub>rus</sub>ti<sub>c</sub>hi<sub>n</sub>i’<sub>s regre</sub>t<sub>.</sub> I<sub>n</sub> Conference on Learning Theory, pp. 1547–1575. PMLR, 2022.

Tor Lattimore and Andras Gyorgy. Mirror descent and the information ratio. In Conference on L<sub>earn</sub>i<sub>ng</sub> Th<sub>eory,</sub> <sub>pp.</sub> 2965<sub>–</sub>2992<sub>.</sub> PMLR<sub>,</sub> 2021<sub>.</sub>

T<sub>or</sub> L<sub>a</sub>tti<sub>more an</sub>d C<sub>sa</sub>b<sub>a</sub> S<sub>zepesvar</sub>i<sub>.</sub> A<sub>n</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on-</sub>th<sub>eore</sub>ti<sub>c approac</sub>h t<sub>o m</sub>i<sub>n</sub>i<sub>max regre</sub>t i<sub>n par</sub>ti<sub>a</sub>l´ monitoring. In Conference on Learning Theory, pp. 2111–2139. PMLR, 2019.

Tor Lattimore and Csaba Szepesvari. Exploration by optimisation in partial monitoring. In ´ Conference <sub>on</sub> L<sub>earn</sub>i<sub>ng</sub> Th<sub>eory,</sub> <sub>pp.</sub> 2488<sub>–</sub>2515<sub>.</sub> PMLR<sub>,</sub> 2020<sub>.</sub>

T<sub>or</sub> L<sub>a</sub>tti<sub>more an</sub>d C<sub>sa</sub>b<sub>a</sub> S<sub>zepesvar</sub>i<sub>.</sub>´ B<sub>an</sub>dit Al<sub>gor</sub>ith<sub>ms.</sub> C<sub>am</sub>b<sub>r</sub>id<sub>ge</sub> U<sub>n</sub>i<sub>vers</sub>it<sub>y</sub> P<sub>ress,</sub> 2020<sub>.</sub>

D<sub>av</sub>id A L<sub>ev</sub>i<sub>n an</sub>d Y<sub>uva</sub>l P<sub>eres.</sub> M<sub>ar</sub>k<sub>ov</sub> Ch<sub>a</sub>i<sub>ns an</sub>d Mi<sub>x</sub>i<sub>ng</sub> Ti<sub>mes.</sub> A<sub>mer</sub>i<sub>can</sub> M<sub>a</sub>th<sub>ema</sub>ti<sub>ca</sub>l S<sub>oc</sub>i<sub>e</sub>t<sub>y,</sub> P<sub>rov</sub>id<sub>ence,</sub> RI<sub>,</sub> 2 <sub>e</sub>diti<sub>on,</sub> 2017<sub>.</sub> ISBN 978<sub>-</sub>1<sub>-</sub>4704<sub>-</sub>2962<sub>-</sub>1<sub>.</sub>

Yubo Li, Yidi Miao, Ramayya Krishnan, and Rema Padman. Jev-as-a-judge: Accept when confident, <sub>esca</sub>l<sub>a</sub>t<sub>e w</sub>h<sub>en unsure. ar</sub>Xi<sub>v prepr</sub>i<sub>n</sub>t <sub>ar</sub>Xi<sub>v:</sub>2609<sub>.</sub>26550<sub>,</sub> 2026<sub>.</sub>

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, M K<sub>a</sub>d<sub>ous, an</sub>d I<sub>on</sub> St<sub>o</sub>i<sub>ca.</sub> R<sub>ou</sub>t<sub>e</sub>ll<sub>m:</sub> L<sub>earn</sub>i<sub>ng</sub> t<sub>o rou</sub>t<sub>e</sub> ll<sub>ms</sub> f<sub>rom pre</sub>f<sub>erence</sub> d<sub>a</sub>t<sub>a.</sub> I<sub>n</sub> Y<sub>.</sub> Y<sub>ue,</sub> A. Gar<sub>g</sub>, N. Pen<sub>g</sub>, F. Sha, and R. Yu (eds.), International Conference on Learnin<sub>g</sub> Re<sub>p</sub>resentations, vo<sup>l</sup>ume <sup>2025</sup>, pp. <sup>34433</sup>–<sup>34448</sup>, <sup>2025</sup>. <sup>URL</sup> https://proceedings.iclr.cc/paper\_files/ paper/2025/file/5503a7c69d48a2f86fc00b3dc09de686-Paper-Conference.pdf.

Hao Qin and Chichen<sub>g</sub> Zhan<sub>g</sub>. Optimal and eficient contextual combinatorial semi-bandits with genera<sup>l f</sup>unct<sup>i</sup>on approx<sup>i</sup>mat<sup>i</sup>on, <sup>2026</sup>a. U<sup>RL</sup> https://arxiv.org/abs/2607.13686.

Hao Qin and Chichen<sub>g</sub> Zhan<sub>g</sub>. Tamin<sub>g</sub> the monster ever<sub>y</sub> context: Complexit<sub>y</sub> measure and unified framework for ofline-oracle eficient contextual bandits, 2026b. URL https://arxiv.org/abs/ 2602.09456.

David Sculley. Practical learning from one-sided feedback. In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 609–618, 2007.

D<sub>av</sub>id Si<sub>mc</sub>hi<sub>-</sub>L<sub>ev</sub>i <sub>an</sub>d Y<sub>unzong</sub> X<sub>u.</sub> B<sub>ypass</sub>i<sub>ng</sub> th<sub>e</sub> <sub>mons</sub>t<sub>er:</sub> A f<sub>as</sub>t<sub>er</sub> <sub>an</sub>d <sub>s</sub>i<sub>mp</sub>l<sub>er</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> for contextual bandits under realizabilit<sub>y</sub>. Mathematics ofO<sub>p</sub>erations Research, 47(3):1904–1931, 2022<sub>.</sub>

Adi<sub>s</sub>h Si<sub>ng</sub>l<sub>a,</sub> I<sub>an</sub> Li<sub>ener</sub>t<sub>,</sub> G<sub>a</sub>b<sub>or</sub> B<sub>ar</sub>t´ <sub>o</sub>k<sub>, an</sub>d A<sub>n</sub>d<sub>reas</sub> K<sub>rause.</sub> C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>procuremen</sub>t i<sub>n on</sub>li<sub>ne</sub>´ crowdsourcing markets. In Proceedings of the AAAI Conference on Human Computation and C<sub>row</sub>d<sub>sourc</sub>i<sub>ng,</sub> <sub>vo</sub>l<sub>ume</sub> 2<sub>,</sub> <sub>pp.</sub> 58<sub>–</sub>59<sub>,</sub> 2014<sub>.</sub>

A<sub>.</sub> T<sub>sy</sub>b<sub>a</sub>k<sub>ov.</sub> Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n t<sub>o</sub> n<sub>o</sub>n<sub>pa</sub>r<sub>a</sub>m<sub>e</sub>tri<sub>c es</sub>tim<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> In S<sub>pr</sub>i<sub>nger</sub> S<sub>er</sub>i<sub>es</sub> i<sub>n</sub> St<sub>a</sub>ti<sub>s</sub>ti<sub>cs,</sub> 2008<sub>.</sub> URL https://api.semanticscholar.org/CorpusID:42933599.

Vladimir G Vovk. A game of prediction with expert advice. In Proceedings of the eighth annual conference on Computational learning theory, pp. 51–60, 1995.

Y<sub>un</sub>b<sub>e</sub>i X<sub>u</sub> <sub>an</sub>d A<sub>ssa</sub>f Z<sub>eev</sub>i<sub>.</sub> U<sub>pper</sub> <sub>coun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l <sub>con</sub>fid<sub>ence</sub> b<sub>oun</sub>d<sub>s:</sub> <sub>a</sub> <sub>new</sub> <sub>op</sub>ti<sub>m</sub>i<sub>sm</sub> <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> f<sub>or</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l b<sub>an</sub>dit<sub>s. ar</sub>Xi<sub>v prepr</sub>i<sub>n</sub>t <sub>ar</sub>Xi<sub>v:</sub>2007<sub>.</sub>07876<sub>,</sub> 2020<sub>.</sub>

M<sub>e</sub>n<sub>gx</sub>i<sub>ao</sub> Zh<sub>a</sub>n<sub>g,</sub> Y<sub>u</sub>h<sub>e</sub>n<sub>g</sub> Zh<sub>a</sub>n<sub>g,</sub> Ol<sub>ga</sub> Vr<sub>ousgou,</sub> H<sub>a</sub>i<sub>pe</sub>n<sub>g</sub> L<sub>uo, a</sub>nd P<sub>au</sub>l Min<sub>e</sub>ir<sub>o.</sub> Pr<sub>ac</sub>ti<sub>ca</sub>l contextual bandits with feedback graphs. Advances in Neural Information Processing Systems, 36: 30592<sub>–</sub>30617<sub>,</sub> 2023<sub>.</sub>

Yi<sub>ng</sub>l<sub>un</sub> Zh<sub>u an</sub>d P<sub>au</sub>l Mi<sub>ne</sub>i<sub>ro.</sub> C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l b<sub>an</sub>dit<sub>s w</sub>ith <sub>smoo</sub>th <sub>regre</sub>t<sub>:</sub> Efi<sub>c</sub>i<sub>en</sub>t l<sub>earn</sub>i<sub>ng</sub> i<sub>n con</sub>ti<sub>nuous</sub> action spaces. In International Conference on Machine Learning, pp. 27574–27590. PMLR, 2022.

Yi<sub>ng</sub>l<sub>un</sub> Zh<sub>u,</sub> D<sub>y</sub>l<sub>an</sub> J F<sub>os</sub>t<sub>er,</sub> J<sub>o</sub>h<sub>n</sub> L<sub>ang</sub>f<sub>or</sub>d<sub>, an</sub>d P<sub>au</sub>l Mi<sub>ne</sub>i<sub>ro.</sub> C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l b<sub>an</sub>dit<sub>s w</sub>ith l<sub>arge ac</sub>ti<sub>on</sub> spaces: Made practical. In Proceedings of the International Conference on Machine Learning (ICML), pp. 27428–27453, 2022.

## Appendix

A<sub>ppen</sub>di<sub>x</sub> A Additi<sub>ona</sub>l Di<sub>scuss</sub>i<sub>ons on</sub> I<sub>mpor</sub>t<sub>an</sub>t R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k<sub>s .</sub> 15   
A.1 Ex<sub>p</sub>loration-b<sub>y</sub>-o<sub>p</sub>timization (ExO) versus our a<sub>pp</sub>roach . . . 15   
A<sub>.</sub>2 R<sub>e</sub>l<sub>a</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-es</sub>ti<sub>ma</sub>ti<sub>on coe</sub>fi<sub>c</sub>i<sub>en</sub>t 16   
A<sub>ppen</sub>di<sub>x</sub> B O<sub>n</sub>li<sub>ne an</sub>d <sub>o</sub>fli<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>es</sub> 17   
B<sub>.</sub>1 O<sub>n</sub>li<sub>ne</sub> O<sub>rac</sub>l<sub>es</sub> 17   
B<sub>.</sub>2 Ofli<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> 19   
A<sub>ppen</sub>di<sub>x</sub> C A<sub>na</sub>l<sub>ys</sub>i<sub>s o</sub>f E2D<sub>.</sub>PMS<sub>ide</sub> 21   
C<sub>.</sub>1 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 3<sub>.</sub>1 21   
C.2 Proofs for Al<sub>g</sub>orithm 2 (ϵ-Greedy.PMSide) 23   
C.3 Proofs for Al<sub>g</sub>orithm 3 (SquareCB.PMSide) 25   
A<sub>ppen</sub>di<sub>x</sub> D C<sub>on</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng:</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> t<sub>o</sub> <sub>o</sub>fli<sub>ne</sub> <sub>regress</sub>i<sub>on</sub> 28   
D<sub>.</sub>1 OE2D<sub>.</sub>PMS<sub>ide</sub> 28   
D<sub>.</sub>2 R<sub>eg</sub>r<sub>e</sub>t <sub>a</sub>n<sub>a</sub>l<sub>ys</sub>i<sub>s</sub> 29   
A<sub>ppen</sub>di<sub>x</sub> E E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>u</sub>di<sub>es</sub> 35   
Appendix F Algorithm for Lq-decreasing tree construction 38   
A<sub>ppen</sub>di<sub>x</sub> G A<sub>ux</sub>ili<sub>ary</sub> L<sub>emma</sub>t<sub>a</sub> 38

## A A<sub>dditional</sub> D<sub>iscussions</sub> <sub>on</sub> I<sub>mportant</sub> R<sub>elated</sub> W<sub>orks</sub>

## A.1 Exploration-by-optimization (ExO) versus our approach.

Lattimore & Sze<sub>p</sub>esvari [2020] <sub>p</sub>ro<sub>p</sub>ose the ex<sub>p</sub>loration-b<sub>y</sub>-o<sub>p</sub>timization (ExO) al<sub>g</sub>orithm for adver-´ <sub>sar</sub>i<sub>a</sub>l <sub>non-con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng.</sub> Th<sub>ey</sub> fi<sub>rs</sub>t f<sub>orm</sub> <sub>an</sub> <sub>exponen</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d b<sub>enc</sub>h<sub>mar</sub>k <sub>po</sub>li<sub>cy</sub> $\lambda _ { t }$ <sub>over</sub> th<sub>e</sub> P<sub>are</sub>t<sub>o op</sub>ti<sub>ma</sub>l <sub>ac</sub>ti<sub>ons</sub> f<sub>rom un</sub>bi<sub>ase</sub>d l<sub>oss es</sub>ti<sub>ma</sub>t<sub>es,</sub>

$$
\lambda _ { t a } = \frac { \mathbb { 1 } _ { \Pi } ( a ) \exp \left( - \eta \sum _ { s = 1 } ^ { t - 1 } \hat { y } _ { s a } \right) } { \sum _ { b \in \Pi } \exp \left( - \eta \sum _ { s = 1 } ^ { t - 1 } \hat { y } _ { s b } \right) } ,
$$

<sub>w</sub>h<sub>ere</sub> $\mathbb { 1 } _ { \Pi }$ is the indicator of Π and $\hat { y } _ { t } = G _ { t } \big ( A _ { t } , \sigma _ { t } \big ) / p _ { t A _ { t } }$ i<sub>s an</sub> i<sub>mpor</sub>t<sub>ance-we</sub>i<sub>g</sub>ht<sub>e</sub>d l<sub>oss es</sub>ti<sub>ma</sub>t<sub>e,</sub> <sub>w</sub>ith $G _ { t }$ (Equation (10)) obtained b<sub>y</sub> solvin<sub>g</sub> Equation (4) at round t. This <sub>g</sub>eneralizes the EXP3 al<sub>g</sub>orithm [Auer et al., 2003] for adversarial bandits.

Th<sub>ey</sub> th<sub>en</sub> i<sub>mprove</sub> thi<sub>s</sub> b<sub>enc</sub>h<sub>mar</sub>k b<sub>y</sub> <sub>so</sub>l<sub>v</sub>i<sub>ng</sub>

$$
\operatorname* { m i n i m i s e } _ { G \in \mathcal { H } , p \in \mathcal { P } } \qquad \operatorname* { m a x } _ { o \in [ d ] } \Bigg [ ( p - \lambda ) ^ { \top } L e _ { o } + \mathrm { b i a s } _ { \lambda } ( G ; o ) + \frac { 1 } { \eta } \sum _ { a = 1 } ^ { k } p _ { a } \Psi _ { \lambda } \Bigg ( \frac { \eta G ( a , \Phi _ { a o } ) } { p _ { a } } \Bigg ) \Bigg ] ,\tag{4}
$$

<sub>w</sub>h<sub>ere</sub> $\eta$ <sub>p</sub>l<sub>ays</sub> th<sub>e ro</sub>l<sub>e o</sub>f <sub>our</sub> $\gamma , \Psi _ { q } ( z ) = \langle q , \exp ( - z ) + z - 1 \rangle$ <sub>,</sub> <sub>an</sub>d

$$
\mathrm { b i a s } _ { \lambda } ( G ; o ) = \left. \lambda , L e _ { o } - \sum _ { a = 1 } ^ { k } G ( a , \Phi _ { a o } ) \right. + \operatorname* { m a x } _ { c \in \Pi } \left( \sum _ { a = 1 } ^ { k } G ( a , \Phi _ { a o } ) _ { c } - L _ { c o } \right) .
$$

We ma<sub>y</sub> write this e<sub>q</sub>uivalentl<sub>y</sub>, followin<sub>g</sub> the notation of Lattimore & G<sub>y</sub>or<sub>gy</sub> [2021], as

$$
\operatorname* { m i n i m i s e } _ { G \in \mathcal { H } , p \in \mathcal { P } } \operatorname* { m a x } _ { \sigma \in [ d ] , \epsilon \in \Pi } \left[ ( p - \lambda ) ^ { \top } L e _ { o } + \left. e _ { c } - p , \sum _ { a = 1 } ^ { k } G ( a , \Phi _ { a o } ) - L e _ { o } \right. + \frac { 1 } { \eta } \sum _ { a = 1 } ^ { k } p _ { a } \Psi _ { \lambda } \left( \frac { \eta G ( a , \Phi _ { a o } ) } { p _ { a } } \right) \right] .
$$

Thi<sub>s s</sub>t<sub>ep rea</sub>ll<sub>oca</sub>t<sub>es pro</sub>b<sub>a</sub>bilit<sub>y away</sub> f<sub>rom</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>po</sub>li<sub>cy an</sub>d t<sub>owar</sub>d <sub>ac</sub>ti<sub>ons w</sub>h<sub>ose</sub> f<sub>ee</sub>db<sub>ac</sub>k i<sub>s</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve.</sub> W<sub>r</sub>iti<sub>ng</sub> $\mathrm { o p t } _ { \lambda } ( \eta )$ for the o<sub>p</sub>timal value of Equation (4), the analo<sub>g</sub>ue of our $\mathcal { V } _ { \gamma }$ Lattimore & Sze<sub>p</sub>esvari [2020] bound it for both locall<sub>y</sub> and <sub>g</sub>loball<sub>y</sub> observable <sub>g</sub>ames without´ <sub>ex</sub>hibiti<sub>ng an exp</sub>li<sub>c</sub>it <sub>or approx</sub>i<sub>ma</sub>t<sub>e so</sub>l<sub>u</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em.</sub>

The<sub>y</sub> first restrict the class of estimators G. Let

$$
\mathcal { H } _ { 0 } = \left\{ G \in \mathcal { H } : \left. e _ { b } - e _ { c } , \sum _ { a = 1 } ^ { k } G ( a , \Phi _ { a o } ) - L e _ { o } \right. = 0 \mathrm { f o r } \mathrm { a l l } b , c \right\}
$$

d<sub>eno</sub>t<sub>e</sub> th<sub>e un</sub>bi<sub>ase</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>ors,</sub> f<sub>or w</sub>hi<sub>c</sub>h $\begin{array} { r } { \operatorname { b i a s } _ { \lambda } ( G ; o ) = 0 } \end{array}$ for every o. The minimax objective is then b<sub>oun</sub>d<sub>e</sub>d <sub>a</sub>b<sub>ove</sub> b<sub>y</sub>

$$
\operatorname* { m i n } _ { G \in \mathcal { H } _ { 0 } , p \in \mathcal { P } } \operatorname* { m a x } _ { o \in [ d ] } \left[ ( p - \lambda ) ^ { \top } L e _ { o } + \frac { 1 } { \eta } \sum _ { a = 1 } ^ { k } p _ { a } \Psi _ { \lambda } \bigg ( \frac { \eta G ( a , \Phi _ { a o } ) } { p _ { a } } \bigg ) \right] ,
$$

<sub>an</sub>d<sub>, s</sub>i<sub>nce</sub> th<sub>e</sub> b<sub>rac</sub>k<sub>e</sub>t<sub>e</sub>d <sub>express</sub>i<sub>on</sub> i<sub>s</sub> li<sub>near</sub> i<sub>n</sub> $e _ { o } ,$ <sub>max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng over ou</sub>t<sub>comes</sub> i<sub>s equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> <sub>max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng over ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons:</sub>

$$
\operatorname* { m i n } _ { G \in \mathcal { H } _ { 0 } , p \in \mathcal { P } } \operatorname* { m a x } _ { q \in \mathcal { D } } \left[ ( p - \lambda ) ^ { \top } L q + \frac { 1 } { \eta } \sum _ { o = 1 } ^ { d } q _ { o } \sum _ { a = 1 } ^ { k } p _ { a } \Psi _ { \lambda } \Bigg ( \frac { \eta G ( a , \Phi _ { a o } ) } { p _ { a } } \Bigg ) \right] .\tag{5}
$$

Equation (5) is concave in $q$ <sub>an</sub>d <sub>convex</sub> i<sub>n</sub> $( G , p )$ , so to (nonconstructivel<sub>y</sub>) u<sub>pp</sub>er bound the value of Equation (5), Sion’s minimax theorem <sub>p</sub>ermits exchan<sub>g</sub>in<sub>g</sub> the two o<sub>p</sub>erations. Writin<sub>g</sub> $F ( G , p , q )$ f<sub>or</sub> th<sub>e</sub> b<sub>rac</sub>k<sub>e</sub>t<sub>e</sub>d <sub>express</sub>i<sub>on,</sub>

$$
\operatorname* { m i n } _ { G \in \mathcal { H } _ { 0 } , p \in \mathcal { P } } \operatorname* { m a x } _ { q \in \mathcal { D } } F ( G , p , q ) \ = \ \operatorname* { m a x } _ { q \in \mathcal { D } } \operatorname* { m i n } _ { G \in \mathcal { H } _ { 0 } , p \in \mathcal { P } } F ( G , p , q ) .
$$

T<sub>o</sub> <sub>upper</sub> b<sub>oun</sub>d thi<sub>s</sub> <sub>express</sub>i<sub>on,</sub> it <sub>now</sub> <sub>su</sub>fi<sub>ces</sub> t<sub>o</sub> <sub>c</sub>h<sub>oose</sub> $G$ <sub>an</sub>d $p$ <sub>as</sub> f<sub>unc</sub>ti<sub>ons</sub> <sub>o</sub>f $q$ <sub>an</sub>d $\lambda { \vdots }$ t<sup>h</sup>e<sub>y</sub> b<sub>u</sub>ild <sub>an</sub> $\pmb { L q }$ <sub>-</sub>d<sub>ecreas</sub>i<sub>ng</sub> t<sub>ree,</sub> t<sub>a</sub>k<sub>e</sub> $G$ t<sub>o</sub> b<sub>e</sub> th<sub>e</sub> l<sub>oss es</sub>ti<sub>ma</sub>t<sub>or</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> th<sub>a</sub>t t<sub>ree, an</sub>d <sub>se</sub>t $p = W _ { q } ( \lambda )$ <sub>w</sub>h<sub>ere</sub> $W _ { q }$ is the water-transfer o<sub>p</sub>erator (see Lemma G.5 for a reca<sub>p</sub>). The o<sub>p</sub>erator redistributes the benchmark polic<sub>y</sub> λ so that the resulting $p$ i<sub>ncurs</sub> <sub>no</sub> <sub>grea</sub>t<sub>er</sub> l<sub>oss</sub> <sub>un</sub>d<sub>er</sub> $q ,$ <sub>w</sub>hil<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>ng</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t i<sub>ncreases</sub> t<sub>owar</sub>d th<sub>e roo</sub>t <sub>o</sub>f th<sub>e</sub> t<sub>ree.</sub>

In contrast<sub>, g</sub>i<sub>ven an ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q ,$ our ex<sub>p</sub><sup>l</sup>orat<sup>i</sup>on strate<sub>gy</sub> a<sub>pp</sub>rox<sup>i</sup>mate<sup>l</sup><sub>y</sub> so<sup>l</sup>ves t<sup>h</sup>e f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em:</sub>

$$
\operatorname* { m i n } _ { p \in \mathcal { P } } \operatorname* { m a x } _ { \lambda \in \Delta ( \Pi ) } ( p - \lambda ) ^ { \top } L \hat { q } + \frac { 1 } { \gamma } \cdot \operatorname { C o v e r a g e } ( p , \lambda )
$$

Gi<sub>ven</sub> th<sub>a</sub>t f<sub>or</sub> <sub>any</sub> $G \in { \mathcal { H } } _ { 0 }$ , Lemma C.1 implies that Coverage $\begin{array} { r } { ( p , \lambda ) \leq \sum _ { a } \sum _ { b } \frac { \lambda _ { b } + p _ { b } } { p _ { a } } \| G _ { b } ( a , : ) \| _ { \infty } ^ { 2 } } \end{array}$ <sub>,</sub> <sub>an</sub>d thus it sufices to find exploration distribution p that solves,

$$
\operatorname* { m i n } _ { p \in \mathcal { P } } \operatorname* { m a x } _ { \lambda \in \Delta ( \Pi ) } ( p - \lambda ) ^ { \top } \pmb { L } \hat { q } + \frac { 1 } { \gamma } \cdot \left( \sum _ { a } \sum _ { b } \frac { \lambda _ { b } + p _ { b } } { p _ { a } } \| G _ { b } ( a , : ) \| _ { \infty } ^ { 2 } \right)\tag{6}
$$

Com<sub>p</sub>arin<sub>g</sub> Equation (5) with Equation (6), the two objectives share a sha<sub>p</sub>e: each consists of a <sub>gree</sub>di<sub>ness</sub> t<sub>erm o</sub>f th<sub>e</sub> f<sub>orm</sub> $( p - \bar { \lambda } ) ^ { \intercal } \mathbfcal { L } \hat { q }$ and a coverage term measuring how well p covers λ. The<sub>y</sub> dif<sub>er</sub> i<sub>n</sub> th<sub>ree respec</sub>t<sub>s:</sub>

• First<sub>,</sub> in the minimax <sub>p</sub>la<sub>y</sub>ers: we minimize over ex<sub>p</sub>loration <sub>p</sub>olicies a<sub>g</sub>ainst a maximizin<sub>g</sub> com<sub>p</sub>arator <sub>p</sub>olic<sub>y</sub> λ, whereas Equation (5) minimizes a<sub>g</sub>ainst a maximizin<sub>g</sub> adversarial outcome distribution q.

• Second<sub>,</sub> in the form of the covera<sub>g</sub>e term. The one in E<sub>q</sub>uation (5) is ex<sub>p</sub>onential<sub>,</sub> $\Psi _ { \lambda } \big ( \eta G ( a , \Phi _ { a o } ) / p _ { a } \big )$ , and Lattimore & Sze<sub>p</sub>esvari [2020] relax it to´ $\begin{array} { r } { \eta ^ { 2 } \sum _ { b } \lambda _ { b } \| G ( a , \bar { \cdot } ) _ { b } \| _ { \infty } ^ { 2 } / p _ { a } ^ { 2 } } \end{array}$ at the cost of requiring p to be bounded below entr<sub>y</sub>wise b<sub>y</sub> $\eta \operatorname* { m a x } _ { a , b } \| G ( a , \cdot ) _ { b } \| _ { \infty }$

∞• Third<sub>,</sub> in what the anal<sub>y</sub>sis delivers. ExO is tractable but <sub>g</sub>ives no ex<sub>p</sub>licit ex<sub>p</sub>loration <sub>p</sub>olic<sub>y</sub> p. For our objective we show that the inverse-gap-weighting rule of SquareCB.PMSide <sub>cer</sub>tifi<sub>es</sub> <sub>a</sub> <sub>sma</sub>ll <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f $\mathcal { V } _ { \gamma }$ di<sub>rec</sub>tl<sub>y:</sub> it <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> <sub>a</sub>ll<sub>oca</sub>t<sub>es</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> t<sub>owar</sub>d th<sub>e</sub> root of an Lqˆ-decreasing tree, so neither the optimization over G nor the water-transfer step i<sub>s</sub> <sub>nee</sub>d<sub>e</sub>d<sub>,</sub> <sub>an</sub>d th<sub>e</sub> i<sub>n-</sub>t<sub>ree</sub> <sub>on</sub>l<sub>y</sub> <sub>appears</sub> i<sub>n</sub> th<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s.</sub>

W<sub>e</sub> <sub>a</sub>l<sub>so</sub> <sub>re</sub>f<sub>er</sub> th<sub>e</sub> <sub>rea</sub>d<sub>er</sub> t<sub>o</sub> T<sub>a</sub>bl<sub>e</sub> 1 f<sub>or</sub> <sub>a</sub> <sub>summary.</sub>

Table 1: Comparison with exploration-by-optimization [Lattimore & Szepesvari, 2020]. ´ ↑ indicates a b<sub>e</sub>tt<sub>er</sub> <sub>resu</sub>lt<sub>.</sub> NA i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> <sub>no</sub>t <sub>app</sub>li<sub>ca</sub>bl<sub>e</sub>
<table><tr><td></td><td>ExO</td><td>Ours</td></tr><tr><td>Side information</td><td>No</td><td>Yes ↑</td></tr><tr><td>Outcome distribution</td><td>Adversarial ↑</td><td>Stochastic</td></tr><tr><td>Objective</td><td> $\mathrm { o p t } _ { q } ( \eta )$ </td><td> $\mathcal { V } _ { \gamma }$ </td></tr><tr><td>Minimax players</td><td>Policy vs. adversarial outcome x</td><td>Policy vs. comparator policy λ</td></tr><tr><td>Optimization variables</td><td>G and p jointly Exact solution of Equation (4)</td><td>p only ↑ IGW for locally observable;</td></tr><tr><td>Per-round computation</td><td></td><td>€-greedy for globally observable ↑</td></tr><tr><td>In-tree</td><td>Optimized at every round</td><td>Analysis only ↑</td></tr><tr><td>Water-transfer</td><td>Required</td><td>Not required ↑</td></tr><tr><td>Oracle-efficient</td><td>No</td><td>Efficient ↑</td></tr><tr><td>Offline oracle compatibility</td><td>NA</td><td>Compatible ↑</td></tr><tr><td>Locally observable regret</td><td></td><td> $\sqrt { k }$  smaller ↑</td></tr></table>

## A<sub>.</sub>2 R<sub>elation</sub> <sub>to</sub> <sub>the</sub> <sub>decision-estimation</sub> <sub>coefficient</sub>

The decision-estimation coeficient (DEC) was introduced b<sub>y</sub> Foster et al. [2021] as a com<sub>p</sub>lexit<sub>y</sub> measure for stochastic se<sub>q</sub>uential decision makin<sub>g</sub>. Foster et al. [2022] extend it to the adversarial <sub>se</sub>tti<sub>ng,</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t th<sub>e</sub> DEC <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> <sub>convex</sub> h<sub>u</sub>ll <sub>o</sub>f th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>ass</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>zes</sub> th<sub>e</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> <sub>o</sub>f th<sub>a</sub>t <sub>se</sub>tti<sub>ng</sub> <sub>as</sub> <sub>we</sub>ll<sub>.</sub> O<sub>n</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub>i<sub>c</sub> <sub>s</sub>id<sub>e</sub> th<sub>ey</sub> <sub>propose</sub> $\mathrm { E x O ^ { + } }$ <sub>a</sub> hi<sub>g</sub>h<sub>-pro</sub>b<sub>a</sub>bilit<sub>y var</sub>i<sub>an</sub>t <sub>o</sub>f th<sub>e exp</sub>l<sub>ora</sub>ti<sub>on-</sub>b<sub>y-op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on a</sub>l<sub>gor</sub>ith<sub>m o</sub>f L<sub>a</sub>tti<sub>more</sub> & S<sub>zepesvar</sub>i´ [2020]; Lattimore & G<sub>y</sub>or<sub>gy</sub> [2021]. Whereas the ori<sub>g</sub>inal ExO <sub>g</sub>ives in-ex<sub>p</sub>ectation <sub>g</sub>uarantees, $\mathrm { E x O ^ { + } }$ <sub>g</sub>i<sub>ves</sub> hi<sub>g</sub>h<sub>-pro</sub>b<sub>a</sub>bilit<sub>y guaran</sub>t<sub>ees</sub> th<sub>a</sub>t h<sub>o</sub>ld <sub>aga</sub>i<sub>ns</sub>t <sub>a</sub>d<sub>ap</sub>ti<sub>ve a</sub>d<sub>versar</sub>i<sub>es.</sub> Lik<sub>e</sub> E<sub>x</sub>O<sub>,</sub> th<sub>e</sub> <sub>exp</sub>l<sub>ora</sub>ti<sub>on ru</sub>l<sub>e an</sub>d th<sub>e</sub> l<sub>oss es</sub>ti<sub>ma</sub>t<sub>or are</sub> d<sub>e</sub>fi<sub>ne</sub>d th<sub>roug</sub>h <sub>an op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em ra</sub>th<sub>er</sub> th<sub>an a</sub> <sub>c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm po</sub>li<sub>cy.</sub>

Th<sub>e</sub>i<sub>r ana</sub>l<sub>ys</sub>i<sub>s</sub> d<sub>oes no</sub>t b<sub>oun</sub>d th<sub>e</sub> $\mathrm { E x O ^ { + } }$ or DEC objective directly for locally and globally observable <sub>games.</sub> I<sub>ns</sub>t<sub>ea</sub>d th<sub>ey es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>an equ</sub>i<sub>va</sub>l<sub>ence,</sub> f<sub>or genera</sub>l di<sub>vergences</sub> $D ,$ b<sub>e</sub>t<sub>ween</sub> th<sub>e convex</sub>ifi<sub>e</sub>d DEC, a <sub>p</sub>arameterized variant of the <sub>g</sub>eneralized information ratio of Lattimore & G<sub>y</sub>or<sub>gy</sub> [2021], and th<sub>e</sub> $\mathrm { E x O ^ { + } }$ objective. Since Lattimore & Sze<sub>p</sub>esvari [2019]; Lattimore & G<sub>y</sub>or<sub>gy</sub> [2021]; Lattimore´ [2022] bound the re<sub>g</sub>ret for locall and <sub>g</sub>loball observable <sub>g</sub>ames in terms of the <sub>g</sub>eneralized information ratio under the KL diver<sub>g</sub>ence, in a<sub>g</sub>reement with the classification theorem [Bartok et al.,´ 2011; Lattimore & Sze<sub>p</sub>esvari, 2020], those bounds transfer to´ $\mathrm { E x O ^ { + } }$ th<sub>roug</sub>h thi<sub>s equ</sub>i<sub>va</sub>l<sub>ence.</sub> Th<sub>e</sub> equivalence itself holds for general D, but the information-ratio bounds are established for the KL di<sub>vergence; we are no</sub>t <sub>aware o</sub>f <sub>correspon</sub>di<sub>ng</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or o</sub>th<sub>er</sub> di<sub>vergences,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e square</sub>d H<sub>e</sub>lli<sub>nger.</sub>

To summarize, Foster et al. [2022]’s result im<sub>p</sub>licitl<sub>y</sub> im<sub>p</sub>lies that the E2D al<sub>g</sub>orithm Foster et al. [2021] has o<sub>p</sub>timal order-in-T re<sub>g</sub>ret for stochastic contextual <sub>p</sub>artial monitorin<sub>g</sub>. However, as we di<sub>scusse</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 2<sub>.</sub>3<sub>,</sub> th<sub>e</sub> E2D <sub>a</sub>l<sub>gor</sub>ith<sub>m may</sub> b<sub>e compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y</sub> i<sub>ne</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>.</sub> O<sub>ur paper a</sub>dd<sub>resses</sub> th<sub>e</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>c</sub>h<sub>a</sub>ll<sub>enge.</sub> F<sub>or</sub> l<sub>oca</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>games</sub> <sub>we</sub> <sub>g</sub>i<sub>ve</sub> <sub>a</sub> <sub>c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm</sub> i<sub>nverse-gap-</sub> wei<sub>g</sub>htin<sub>g</sub> rule (SquareCB.PMSide), and for <sub>g</sub>loball<sub>y</sub> observable <sub>g</sub>ames a closed-form ε-<sub>g</sub>reed<sub>y</sub> rule (ϵ-Greedy.PMSide). Both extend to be com<sub>p</sub>atible with ofline estimation oracles, because the exploration term of our objective, the coverage, does not involve the current estimate.

## B O<sub>nline</sub> <sub>and</sub> <sub>offline</sub> <sub>estimation</sub> <sub>oracles</sub>

## B<sub>.</sub>1 O<sub>nline</sub> O<sub>racles</sub>

I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on, we prov</sub>id<sub>e examp</sub>l<sub>es</sub> f<sub>or</sub> th<sub>e on</sub>li<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>es</sub> $\mathrm { A l g _ { O n } }$ <sub>an</sub>d <sub>prove</sub> th<sub>e</sub>i<sub>r</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> b<sub>o</sub>th fi<sub>n</sub>it<sub>e an</sub>d i<sub>n</sub>fi<sub>n</sub>it<sub>e mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>asses.</sub>

Lemma B.1 (Finite-class online Hellinger guarantee via Vovk’s algorithm). Suppose that the candidate <sub>con</sub>diti<sub>ona</sub>l <sub>ou</sub>t<sub>come-mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>ass</sub>

$$
\mathcal { Q } \subseteq \{ q : \mathcal { X }  \Delta ( d ) \}
$$

i<sub>s</sub> fi<sub>n</sub>it<sub>e an</sub>d <sub>rea</sub>li<sub>za</sub>bl<sub>e, so</sub> th<sub>a</sub>t $q ^ { * } \in \mathcal { Q }$ <sub>an</sub>d th<sub>e orac</sub>l<sub>es are a</sub>ll<sub>owe</sub>d t<sub>o ou</sub>t<sub>pu</sub>t <sub>pre</sub>di<sub>c</sub>t<sub>ors</sub> i<sub>n</sub> $\mathrm { c o } ( \mathcal { Q } )$ <sub>.</sub> Th<sub>en</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>on</sub>li<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> $\mathrm { A l g } _ { \mathrm { O n } }$ <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub>

$$
\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta ) = 2 \log \left( \frac { | \mathcal { Q } | } { \delta } \right) .
$$

The contextual online estimation oracle can be instantiated b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Vovk’s [Vovk, 1995] <sub>aggrega</sub>ti<sub>ng a</sub>l<sub>gor</sub>ith<sub>m</sub> t<sub>o</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>ou</sub>t<sub>come mo</sub>d<sub>e</sub>l<sub>s</sub> i<sub>n</sub> $\mathcal { Q } .$

Proof. Define $\mathcal { G } \subseteq \Delta ( m )$ <sub>as a se</sub>t <sub>o</sub>f f<sub>unc</sub>ti<sub>ons</sub> $g _ { q } : { \mathcal { X } } \times [ k ] \to \Delta ( m )$ i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> $q ,$ f<sub>orma</sub>ll<sub>y</sub>

$$
\mathcal { G } = \{ g _ { q } : g _ { q } ( \sigma \mid x , a ) = S _ { a } q ( x ) [ \sigma ] \ \forall q \in \mathcal { Q } \}
$$

H<sub>ence,</sub>

$$
| { \mathcal { G } } | \leq | { \mathcal { Q } } |
$$

S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub>

$$
\begin{array} { r l } & { g _ { q ^ { * } } \bigl ( \sigma \bigr | x _ { t } , A _ { t } \bigr ) = \mathbb { P } \bigl ( \sigma _ { t } = \sigma \mid x _ { t } , A _ { t } \bigr ) } \\ & { \qquad = S _ { A _ { t } } q ^ { * } \bigl ( x _ { t } \bigr ) \bigl [ \sigma \bigr ] } \end{array}
$$

V<sub>ov</sub>k’<sub>s</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> $g _ { \hat { q } _ { t } }$ i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>way.</sub>

L<sub>e</sub>t<sub>,</sub>

$$
\begin{array} { r l } & { \ell _ { \log } ^ { ( t ) } ( g _ { q } ) : = \log \left( \frac { 1 } { g _ { q } ( \sigma _ { t } \mid x _ { t } , A _ { t } ) } \right) } \\ & { \qquad = \log \left( \frac { 1 } { \big ( S _ { A _ { t } } q ( x _ { t } ) \big ) \big [ \sigma _ { t } \big ] } \right) . } \end{array}
$$

Th<sub>en</sub> $\forall g _ { q } \in \mathcal { G }$ <sub>,</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>es</sub> <sub>a</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>s</sub>i<sub>mp</sub>l<sub>ex</sub> $b _ { t } \in \Delta ( | \mathcal { G } | )$ <sub>as</sub> f<sub>o</sub>ll<sub>ows.</sub>

$$
b _ { t } ( g _ { q } ) \propto \exp \left( - \sum _ { s = 1 } ^ { t - 1 } \ell _ { \log } ^ { ( s ) } ( g _ { q } ) \right)
$$

E<sub>s</sub>ti<sub>ma</sub>t<sub>e</sub> $g _ { \hat { q } _ { t } }$

$$
g _ { \hat { q } _ { t } } ( \cdot ) = \mathbb { E } _ { g \sim b _ { t } } \left[ g ( \cdot ) \right]
$$

L<sub>e</sub>t th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> $\hat { q } _ { t }$ <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub>

$$
\hat { q } _ { t } ( \cdot ) = \mathbb { E } _ { g \sim b _ { t } } \left[ q ( \cdot ) \right] .
$$

Th<sub>en</sub> V<sub>ov</sub>k’<sub>s</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>guaran</sub>t<sub>ees,</sub>

$$
\sum _ { t = 1 } ^ { T } \log \frac { 1 } { g _ { \hat { q } _ { t } } \big ( \sigma _ { t } \mid x _ { t } , a _ { t } \big ) } - \sum _ { t = 1 } ^ { T } \log \frac { 1 } { g _ { q ^ { * } } \big ( \sigma _ { t } \mid x _ { t } , a _ { t } \big ) } \leq \log \vert \mathcal { G } \vert
$$

Then, b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Lemma A.14 from Foster et al. [2021], the out<sub>p</sub>ut estimator of Vovk’s a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> $g _ { \hat { q } _ { t } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>a</sub>t l<sub>eas</sub>t $1 - \delta .$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { t - 1 } \left[ D _ { \mathrm { H } } ^ { 2 } \left( g _ { \hat { q } _ { t } } ( \cdot \mid x _ { t } , A _ { t } ) \mid g _ { q ^ { * } } ( \cdot \mid x _ { t } , A _ { t } ) \right) \right] \leq 2 \log \frac { | \mathcal { G } | } { \delta }\tag{7}
$$

t<sub>rans</sub>f<sub>orm</sub>i<sub>ng</sub> thi<sub>s</sub> t<sub>o our case, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta , { \hat { q } } _ { t }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es,</sub>

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { a \sim p _ { t } } \left[ D _ { \mathrm { H } } ^ { 2 } \left( S _ { a } \hat { q } _ { t } ( x _ { t } ) \mid S _ { a } q ^ { * } ( x _ { t } ) \right) \right] \leq 2 \log \frac { | \boldsymbol { Q } | } { \delta } .
$$

Remark 1. Define filtration $\mathcal { H }$ as,

$$
\mathcal { H } _ { t - 1 } = \sigma \left( x _ { 1 } , A _ { 1 } , \sigma _ { 1 } , \cdots , x _ { t - 1 } , A _ { t - 1 } , \sigma _ { t - 1 } \right)
$$

E<sub>qua</sub>ti<sub>on equa</sub>ti<sub>on</sub> 7 <sub>a</sub>l<sub>so</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t<sub>,</sub> if $x _ { 1 } , \cdots , x _ { T } \stackrel { \mathrm { i . i . d . } } { \sim } { \mathcal { D } } _ { x }$ <sub>an</sub>d $p _ { t } , \hat { q } _ { t }$ are $\mathcal { H } _ { t - 1 }$ <sub>measura</sub>bl<sub>e,</sub> th<sub>en</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>a</sub>t l<sub>eas</sub>t $1 - \delta ,$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { x \sim \mathcal { D } _ { x } , a \sim p _ { t } ( \cdot \mid x ) } \left[ D _ { \mathrm { H } } ^ { 2 } \left( S _ { a } \hat { q } _ { t } ( x ) \mid S _ { a } q ^ { * } ( x ) \right) \right] \leq 2 \log \frac { | \mathcal { Q } | } { \delta } .
$$

Lemma B.2 (Infinite-class online Hellinger guarantee via contextual smoothed aggregation). Suppose th<sub>a</sub>t th<sub>e can</sub>did<sub>a</sub>t<sub>e con</sub>diti<sub>ona</sub>l <sub>ou</sub>t<sub>come-mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>ass</sub> i<sub>s,</sub>

$$
\mathcal { Q } \subseteq \{ q : \mathcal { X }  \Delta ( d ) \} ,
$$

i<sub>s poss</sub>ibl<sub>y</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub> b<sub>u</sub>t h<sub>as</sub> fi<sub>n</sub>it<sub>e cover</sub>i<sub>ng num</sub>b<sub>ers un</sub>d<sub>er</sub> th<sub>e un</sub>if<sub>orm</sub> H<sub>e</sub>lli<sub>nger me</sub>t<sub>r</sub>i<sub>c an</sub>d th<sub>a</sub>t th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l i<sub>s rea</sub>li<sub>za</sub>bl<sub>e</sub> $q ^ { * } \in \mathcal { Q }$ <sub>an</sub>d th<sub>e orac</sub>l<sub>es are a</sub>ll<sub>owe</sub>d t<sub>o ou</sub>t<sub>pu</sub>t <sub>pre</sub>di<sub>c</sub>t<sub>ors</sub> i<sub>n</sub> $\mathrm { c o } ( \mathcal { Q } )$ .

D<sub>e</sub>fi<sub>ne</sub> $\mathcal { G } \subseteq \Delta ( m )$ <sub>as a se</sub>t <sub>o</sub>f f<sub>unc</sub>ti<sub>ons</sub> $g _ { q } : { \mathcal { X } } \times [ k ] \to \Delta ( m )$ i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> $q ,$ f<sub>orma</sub>ll<sub>y</sub>

$$
\mathcal { G } = \{ g _ { q } : g _ { q } ( \sigma \mid x , a ) = S _ { a } q ( x ) [ \sigma ] ~ q \in \mathcal { Q } \}
$$

Moreover, let us define an ϵ-cover ${ \mathcal { I } } ,$ <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub>

$$
\forall g _ { q } \in \mathcal { G } \quad \exists g _ { q } ^ { \prime } \in \mathcal { T } \mathrm { ~ s u c h ~ t h a t ~ } \operatorname* { s u p } _ { x \in \mathcal { X } , a \in [ k ] } D _ { \mathrm { H } } ^ { 2 } \left( g _ { q } ^ { \prime } ( \cdot \vert x , a ) \vert g _ { q } ( \cdot \vert x , a ) \right) \leq \epsilon ^ { 2 }
$$

F<sub>ur</sub>th<sub>ermore,</sub> l<sub>e</sub>t ${ \mathcal { N } } ( { \mathcal { G } } , \epsilon )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e s</sub>i<sub>ze o</sub>f th<sub>e sma</sub>ll<sub>es</sub>t <sub>suc</sub>h <sub>cover.</sub> D<sub>e</sub>fi<sub>ne,</sub>

$$
\operatorname { e s t } ( \mathcal { G } , T ) : = \operatorname* { i n f } _ { \epsilon \geq 0 } \{ \log \left( \mathcal { N } ( \mathcal { G } , \epsilon ) \right) + \epsilon ^ { 2 } T \}
$$

Th<sub>en</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>on</sub>li<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> $\mathrm { A l g } _ { \mathrm { O n } }$ <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub>

$$
\begin{array} { r } { \overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } ( \mathcal { Q } , T , \delta ) \leq O \left( \log ( 2 m ^ { 2 } T ) \cdot \mathrm { e s t } ( \mathcal { G } , T ) + \log ^ { 2 } ( 2 m ^ { 2 } T ) \log \frac { 1 } { \delta } \right) . } \end{array}
$$

Th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>on</sub>li<sub>ne</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>orac</sub>l<sub>e</sub> <sub>can</sub> b<sub>e</sub> i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>e</sub>d b<sub>y</sub> <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> <sub>an</sub> <sub>on</sub>li<sub>ne</sub> <sub>aggrega</sub>ti<sub>on</sub> al<sub>g</sub>orithm Vovk [1995]; Foster et al. [2021] to a suitabl<sub>y</sub> discretized and smoothed a<sub>pp</sub>roximation of the parameter space Q.

Proof. The re<sub>g</sub>ularit<sub>y</sub> assum<sub>p</sub>tions re<sub>q</sub>uired in Assum<sub>p</sub>tion A.2 of Foster et al. [2021] are alread<sub>y</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>e</sub>d b<sub>y</sub> <sub>se</sub>tti<sub>ng</sub> $\bar { B ^ { \prime } } = \operatorname* { m a x } \{ m , e \}$

1<sub>.</sub> N<sub>um</sub>b<sub>er o</sub>f <sub>poss</sub>ibl<sub>e ou</sub>t<sub>come</sub> f<sub>or a con</sub>t<sub>ex</sub>t i<sub>s upper</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> $m \le B$

2<sub>.</sub> $g _ { q } ( \sigma \vert x , a )$ i<sub>s</sub> <sub>a</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>s</sub>i<sub>mp</sub>l<sub>ex</sub> $( 1 \leq B )$

S<sub>moo</sub>th<sub>e</sub>d V<sub>ov</sub>k’<sub>s</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> $g _ { \hat { q } _ { t } }$ i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>way,</sub> l<sub>e</sub>t<sub>,</sub>

$$
h ( \sigma \mid x , a ) = ( 1 - \alpha ) g _ { q } ( \sigma \mid x , a ) + { \frac { \alpha } { m } }
$$

th<sub>en</sub> $\forall g _ { q } \in \mathcal { I }$ <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e a pro</sub>b<sub>a</sub>bilit<sub>y s</sub>i<sub>mp</sub>l<sub>ex</sub> $b _ { t } \in \Delta ( | \mathcal { I } | )$ <sub>as</sub> f<sub>o</sub>ll<sub>ows</sub>

$$
b _ { t } ( g _ { q } ) \propto \exp \left( - \sum _ { s = 1 } ^ { t - 1 } \ell _ { \log } ^ { ( s ) } ( h ) \right)
$$

E<sub>s</sub>ti<sub>ma</sub>t<sub>e</sub> $\hat { g } _ { t }$

$$
g _ { \hat { q } _ { t } } = \mathbb { E } _ { g _ { q } \sim b _ { t } } \left[ g _ { q } \right]
$$

H<sub>ence,</sub>

$$
\hat { q } _ { t } = \mathbb { E } _ { q \sim b _ { t } } \left[ q \right]
$$

Then b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Lemma A.16 from Foster et al. [2021], with <sub>p</sub>robabilit<sub>y</sub> of at least $1 - \delta ,$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { t - 1 } \left[ D _ { \mathbb { H } } ^ { 2 } \left( g _ { \hat { q } _ { t } } ( \cdot \vert x _ { t } , A _ { t } ) \vert g _ { q ^ { * } } ( \cdot \vert x _ { t } , A _ { t } ) \right) \right] \leq O \left( \log ( 2 m ^ { 2 } T ) \cdot \mathrm { e x t } ( \mathcal { G } , T ) + \log ^ { 2 } ( 2 m ^ { 2 } T ) \log \frac { 1 } { \delta } \right)\tag{8}
$$

L<sub>e</sub>t $\hat { q } _ { t } \in \mathcal { Q }$ b<sub>e</sub> th<sub>e correspon</sub>di<sub>ng so</sub>l<sub>u</sub>ti<sub>on, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>a</sub>t l<sub>eas</sub>t $1 - \delta , { \hat { q } } _ { t }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es,</sub>

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { a \sim p _ { t } } \left[ D _ { \mathrm { H } } ^ { 2 } \left( S _ { a } \hat { q } _ { t } ( x _ { t } ) \mid S _ { a } q ^ { * } ( x _ { t } ) \right) \right] \leq O \left( \log ( 2 m ^ { 2 } T ) \cdot \mathrm { e s t } ( \mathcal { G } , T ) + \log ^ { 2 } ( 2 m ^ { 2 } T ) \log \frac { 1 } { \delta } \right) .
$$

Remark 2. Define filtration H as,

$$
\mathcal { H } _ { t - 1 } = \sigma \left( x _ { 1 } , A _ { 1 } , \sigma _ { 1 } , \cdots , x _ { t - 1 } , A _ { t - 1 } , \sigma _ { t - 1 } \right)
$$

E<sub>qua</sub>ti<sub>on equa</sub>ti<sub>on</sub> 7 <sub>a</sub>l<sub>so</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t<sub>,</sub> if $x _ { 1 } , \cdots , x _ { T } \stackrel { \mathrm { i . i . d . } } { \sim } { \mathcal { D } } _ { x }$ <sub>an</sub>d $p _ { t } , \hat { q } _ { t }$ are $\mathcal { H } _ { t - 1 }$ <sub>measura</sub>bl<sub>e,</sub> th<sub>en</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>a</sub>t l<sub>eas</sub>t $1 - \delta$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { x \sim \mathcal { D } _ { x } , a \sim p _ { t } ( \cdot \vert x ) } \left[ D _ { \mathbb { H } } ^ { 2 } \left( S _ { a } \hat { q } _ { t } ( x ) \mid S _ { a } q ^ { * } ( x ) \right) \right] \leq O \left( \log ( 2 m ^ { 2 } T ) \cdot \mathrm { e s t } ( \mathcal { G } , T ) + \log ^ { 2 } ( 2 m ^ { 2 } T ) \log \frac { 1 } { \delta } \right)
$$

## B<sub>.</sub>2 O<sub>ffline</sub> <sub>estimation</sub> <sub>oracle</sub>

In this section, first we define ofline estimation oracle and we provide examples for the ofline <sub>es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>es</sub> $\operatorname { A l g } _ { \mathrm { O f f } }$ <sub>an</sub>d <sub>prove</sub> th<sub>e</sub>i<sub>r</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> b<sub>oun</sub>d<sub>s.</sub>

Definition B.1 (Ofline estimation oracle). An ofline estimation oracle $\mathrm { A l g _ { O f f } }$ i<sub>s an a</sub>l<sub>gor</sub>ith<sub>m</sub> th<sub>a</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>es</sub> th<sub>e un</sub>k<sub>nown parame</sub>t<sub>er</sub> $q ^ { * } \in \mathcal { Q }$ f<sub>rom</sub> th<sub>e o</sub>b<sub>serve</sub>d d<sub>a</sub>t<sub>a</sub> i<sub>n a</sub> b<sub>a</sub>t<sub>c</sub>h f<sub>as</sub>hi<sub>on.</sub> Th<sub>a</sub>t i<sub>s, a</sub>ft<sub>er</sub> observing t rounds, given the feedback s<sub>y</sub>mbols collected so far, the oracle produces an estimate

$$
\begin{array} { r } { \hat { q } = \mathrm { A l g } _ { \mathrm { O f f } } \left( \left\{ \left( A _ { s } , x _ { s } , \sigma _ { s } \right) \right\} _ { s = 1 } ^ { t } \right) . } \end{array}
$$

Assumption B.1 (Hellinger-closeness of the ofline estimation oracle). For any $t \in \mathbb { N } ,$ l<sub>e</sub>t $x _ { 1 } , \dotsc , x _ { t } \stackrel { \mathrm { i . i . d } } { \sim }$ $\mathcal { D } _ { x }$ <sub>an</sub>d l<sub>e</sub>t $\hat { q }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> <sub>re</sub>t<sub>urne</sub>d b<sub>y</sub> $\operatorname { A l g } _ { \mathrm { O f f } }$ <sub>on</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>samp</sub>l<sub>e.</sub> D<sub>e</sub>fi<sub>ne</sub> $\operatorname { R e g } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , t \right)$ <sub>suc</sub>h th<sub>a</sub>t

$$
\mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \boldsymbol { x } } , \boldsymbol { a } \sim p ( \cdot | \boldsymbol { x } ) } \left[ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { \boldsymbol { a } } \hat { q } ( \boldsymbol { x } ) \big | \big | S _ { \boldsymbol { a } } q ^ { * } ( \boldsymbol { x } ) \big ) \right] = \frac { \mathrm { R e g } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , t \right) } { t }
$$

<sub>w</sub>h<sub>ere</sub> $p$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l <sub>po</sub>li<sub>cy use</sub>d t<sub>o co</sub>ll<sub>ec</sub>t th<sub>e samp</sub>l<sub>e, an</sub>d th<sub>e expec</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> t<sub>a</sub>k<sub>en over a</sub> f<sub>res</sub>h d<sub>raw</sub> $( x , a )$ <sub>, con</sub>diti<sub>ona</sub>ll <sub>on</sub> ${ \hat { q } } .$ W<sub>e assume</sub> th<sub>a</sub>t th<sub>e o</sub>fli<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> $\mathrm { A l g _ { O f f } }$ gu<sup>arantees</sup>, <sup>f</sup>or an<sub>y</sub> $\delta \in ( 0 , 1 )$ <sub>,</sub> th<sub>a</sub>t <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta .$

$$
\begin{array} { r } { \mathrm { R e g } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , t \right) \leq \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , t , \delta \right) , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $\overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , t , \delta \right)$ i<sub>s a</sub> k<sub>nown,</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c upper</sub> b<sub>oun</sub>d<sub>.</sub>

H<sub>av</sub>i<sub>ng es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>1 th<sub>e ex</sub>i<sub>s</sub>t<sub>ence o</sub>f <sub>on</sub>li<sub>ne orac</sub>l<sub>es w</sub>ith f<sub>avora</sub>bl<sub>e guaran</sub>t<sub>ees</sub> f<sub>or</sub> b<sub>o</sub>th fi<sub>n</sub>it<sub>e an</sub>d i<sub>n</sub>fi<sub>n</sub>it<sub>e mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>asses, we can a</sub>l<sub>ways o</sub>bt<sub>a</sub>i<sub>n o</sub>fli<sub>ne orac</sub>l<sub>es</sub> f<sub>or</sub> b<sub>o</sub>th b<sub>y on</sub>li<sub>ne-</sub>t<sub>o-</sub>b<sub>a</sub>t<sub>c</sub>h <sub>convers</sub>i<sub>on.</sub> B<sub>eyon</sub>d thi<sub>s gener</sub>i<sub>c cons</sub>t<sub>ruc</sub>ti<sub>on, we a</sub>l<sub>so s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e con</sub>diti<sub>ona</sub>l <sub>max</sub>i<sub>mum-</sub>lik<sub>e</sub>lih<sub>oo</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>or sa</sub>ti<sub>s</sub>fi<sub>es</sub> A<sub>ssump</sub>ti<sub>on</sub> B<sub>.</sub>1 <sub>w</sub>ith <sub>a</sub> f<sub>avora</sub>bl<sub>e</sub> b<sub>oun</sub>d $\overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } ( \mathcal { Q } , t , \delta )$

Lemma B.3 (Existence of a contextual ofline estimation oracle for a finite class). Suppose that the <sub>can</sub>did<sub>a</sub>t<sub>e con</sub>diti<sub>ona</sub>l <sub>ou</sub>t<sub>come-mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>ass</sub>

$$
\mathcal { Q } \subseteq \{ q : \mathcal { X }  \Delta ( d ) \}
$$

i<sub>s</sub> fi<sub>n</sub>it<sub>e an</sub>d <sub>rea</sub>li<sub>za</sub>bl<sub>e, so</sub> th<sub>a</sub>t $q ^ { * } \in \mathcal { Q }$ <sub>.</sub> Th<sub>en</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>o</sub>fli<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> $\mathrm { A l g _ { O f f } }$ <sub>suc</sub>h th<sub>a</sub>t A<sub>ssump</sub>ti<sub>on</sub> B<sub>.</sub>1 i<sub>s sa</sub>ti<sub>s</sub>fi<sub>e</sub>d <sub>w</sub>ith

$$
\overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , t , \delta \right) = 2 \log \left( \frac { | \mathcal { Q } | } { \delta } \right) .
$$

Th<sub>e con</sub>t<sub>ex</sub>t<sub>ua</sub>l <sub>o</sub>fli<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e can</sub> b<sub>e</sub> i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>max</sub>i<sub>mum-</sub>lik<sub>e</sub>lih<sub>oo</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub>

$$
\hat { q } _ { t } \in \underset { q \in \mathcal { Q } } { \arg \operatorname* { m a x } } \sum _ { s = 1 } ^ { t } \log \left( \left( S _ { A _ { s } } q ( x _ { s } ) \right) \left[ \sigma _ { s } \right] \right) .
$$

Proof. Let

$$
\begin{array} { l } { { z _ { s } = \left( x _ { s } , A _ { s } \right) } } \\ { { y _ { s } = \sigma _ { s } } } \end{array}
$$

$$
( { \mathrm { W h e r e ~ } } x _ { s } \sim { \mathcal { X } } ( { \mathrm { i . i . d } } ) { \mathrm { ~ a n d ~ } } A _ { s } \sim p _ { s } \left( z _ { 1 : s - 1 } , y _ { 1 : s - 1 } , x _ { s } \right) )
$$

H<sub>ence,</sub>

$$
z _ { s } \sim \mathcal { D } _ { s } \left( z _ { 1 : s - 1 } , y _ { 1 : s - 1 } \right)
$$

(for some distribution $\mathcal { D } _ { s } )$

D<sub>e</sub>fi<sub>ne</sub>

$$
\begin{array} { r l } & { f ^ { * } \big ( z _ { s } , y _ { s } \big ) = \mathbb { P } ( y _ { s } \mid z _ { s } ) } \\ & { \phantom { f ^ { * } \big ( z _ { s } , y _ { s } \big ) = } = \mathbb { P } ( \sigma _ { s } \mid A _ { s } , x _ { s } ) } \\ & { \phantom { f ^ { * } \big ( z _ { s } , y _ { s } \big ) = } = \mathbb { P } ( \sigma _ { s } \mid S _ { A _ { s } } q ^ { * } ( x _ { s } ) ) } \\ & { \phantom { f ^ { * } \big ( z _ { s } , y _ { s } \big ) = } = S _ { A _ { s } } q ^ { * } ( x _ { s } ) [ \sigma ] } \end{array}
$$

F<sub>ur</sub>th<sub>ermore,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\mathcal { F } = \left\{ f : f ( z , y ) = S _ { a } q ( x ) [ \sigma ] \ \forall q \in \mathcal { Q } \right\}
$$

H<sub>ence,</sub>

$$
\vert \mathcal F \vert \leq \vert \mathcal Q \vert
$$

Followin<sub>g</sub> the work of A<sub>g</sub>arwal et al. [2020] (section E)

$$
\begin{array} { r l } & { \hat { f } : = \underset { f \in \mathcal { F } } { \arg \operatorname* { m a x } } \sum _ { s = 1 } ^ { t } \log f ( z _ { s } , y _ { s } ) } \\ & { \quad \quad = \underset { q } { \arg \operatorname* { m a x } } \sum _ { s = 1 } ^ { t } \mathbb { P } ( \sigma _ { s } \mid S _ { A _ { s } } q ( x _ { s } ) ) } \end{array}
$$

Th<sub>ere</sub>f<sub>ore,</sub> l<sub>e</sub>t

$$
\begin{array} { r l } {  { \hat { q } _ { t } = \underset { q } { \arg \operatorname* { m a x } } \sum _ { s = 1 } ^ { t } \log \Big ( \mathbb { P } \big ( \sigma _ { s } | x _ { s } , A _ { s } , q \big ) \Big ) } } \\ & { = \underset { q } { \arg \operatorname* { m a x } } \sum _ { s = 1 } ^ { t } \log \big ( S _ { A _ { s } } \cdot q ( x _ { s } ) [ \sigma _ { s } ] \big ) } \end{array}
$$

th<sub>en,</sub>

$$
\hat { f } ( z , y ) = S _ { a } \hat { q } _ { t } ( x )
$$

B<sub>y</sub> Theorem 21 of A<sub>g</sub>arwal et al. [2020], this maximum likelihood solution, satisfies the followin<sub>g</sub>,

$$
\sum _ { s = 1 } ^ { t } \mathbb { E } _ { z \sim D _ { s } } \left\| \hat { f } ( z , \cdot ) - f ^ { * } ( z , \cdot ) \right\| _ { \mathrm { T V } } ^ { 2 } \leq 2 \log \left( \frac { | \mathcal { F } | } { \delta } \right)
$$

Alth<sub>oug</sub>h Th<sub>eorem</sub> 21 i<sub>s</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d i<sub>n</sub> t<sub>erms</sub> <sub>o</sub>f <sub>square</sub>d t<sub>o</sub>t<sub>a</sub>l <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>ance,</sub> R<sub>emar</sub>k 23 <sub>o</sub>b<sub>serves</sub> th<sub>a</sub>t it<sub>s proo</sub>f di<sub>rec</sub>tl<sub>y es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>e correspon</sub>di<sub>ng square</sub>d<sub>-</sub>H<sub>e</sub>lli<sub>nger</sub> b<sub>oun</sub>d<sub>.</sub>

$$
\sum _ { s = 1 } ^ { t } \mathbb { E } _ { z \sim D _ { s } } \left[ D _ { \mathrm { H } } ^ { 2 } \left( \hat { f } ( z , \cdot ) \big | f ^ { * } ( z , \cdot ) \right) \right] \leq 2 \log \left( \frac { | \mathcal { F } | } { \delta } \right)
$$

Th<sub>ere</sub>f<sub>ore,</sub> i<sub>n our case</sub>

$$
\sum _ { s = 1 } ^ { t } \mathbb { E } _ { x \sim \mathcal { X } , a \sim p _ { s } } \left[ D _ { \mathrm { H } } ^ { 2 } \left( \boldsymbol { S } _ { a } \cdot \boldsymbol { \hat { q } } _ { t } ( x ) \big | \boldsymbol { S } _ { a } \cdot \boldsymbol { q } ^ { * } ( x ) \right) \right] \leq 2 \log \left( \frac { | \boldsymbol { Q } | } { \delta } \right)
$$

## C A<sub>nalysis of</sub> E2D<sub>.</sub>PMS<sub>ide</sub>

## C<sub>.</sub>1 P<sub>roof of</sub> T<sub>heorem</sub> 3<sub>.</sub>1

Theorem 3.1 (Regret of E2D.PMSide). Suppose $\mathrm { A l g _ { O n } }$ h<sub>as a</sub> H<sub>e</sub>lli<sub>nger regre</sub>t b<sub>oun</sub>d $\overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right)$ Fi<sub>x</sub> $\gamma > 0$ <sub>.</sub> Th<sub>en,</sub> f<sub>or any</sub> $\delta \in ( 0 , 1 )$ <sub>,</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta .$

$$
\mathrm { R e g } _ { T } \leq T \cdot \operatorname* { m a x } _ { t = 1 } ^ { T } \mathcal { V } _ { \gamma } \big ( \hat { q } _ { t } ( x _ { t } ) , p _ { t } \big ) + \gamma \cdot \overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right) .
$$

Proof. Let

$$
\lambda _ { x _ { t } } ^ { * } = \arg \operatorname* { m i n } _ { p ^ { \prime } \in \Delta ( \Pi ) } ( p ^ { \prime } ) ^ { \top } L q ^ { * } ( x _ { t } )
$$

(The best <sub>p</sub>olic<sub>y</sub> for context x<sub>t</sub>)

$$
\begin{array} { l } { { \displaystyle \mathrm { R e g } _ { T } = \sum _ { t = 1 } ^ { T } p _ { t } ^ { \top } { \cal L } q ^ { * } ( x _ { t } ) - ( \lambda _ { x _ { t } } ^ { * } ) ^ { \top } { \cal L } q ^ { * } ( x _ { t } ) } \ ~ } \\ { { \displaystyle ~ = \sum _ { t = 1 } ^ { T } p _ { t } ^ { \top } { \cal L } \hat { q } _ { t } ( x _ { t } ) - ( \lambda _ { x _ { t } } ^ { * } ) ^ { \top } { \cal L } \hat { q } _ { t } ( x _ { t } ) + ( \lambda _ { x _ { t } } ^ { * } - p _ { t } ) ^ { \top } { \cal L } ( \hat { q } _ { t } ( x _ { t } ) - q ^ { * } ( x _ { t } ) ) } \ ~ } \end{array}
$$

$$
\begin{array} { r l } {  { \big ( ( \lambda _ { x _ { t } } ^ { * } - p _ { t } ) ^ { \top } L \big ( \hat { q } _ { t } ( x _ { t } ) - q ^ { * } ( x _ { t } ) \big ) \big ) ^ { 2 } = \frac { \big ( \big ( \lambda _ { x _ { t } } ^ { * } - p _ { t } \big ) ^ { \top } L \big ( \hat { q } _ { t } ( x _ { t } ) - q ^ { * } ( x _ { t } ) \big ) \big ) ^ { 2 } } { \mathbb { E } _ { \alpha \sim p _ { t } } [ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { \alpha } \hat { q } _ { t } ( x _ { t } ) \big | S _ { \alpha } q ^ { * } ( x _ { t } ) \big ) ] } \cdot \bigg ( \mathbb { E } _ { \alpha \sim p _ { t } } [ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { \alpha } \hat { q } _ { t } ( x _ { t } ) \big | S _ { \alpha } q ^ { * } ( x _ { t } ) \big ) ] \bigg ) } \quad } & \\ { \leq \quad } & { \operatorname* { m a x } _ { \boldsymbol { q } _ { 1 } , \boldsymbol { q } \geq \boldsymbol { \hat { \alpha } } ( \boldsymbol { \hat { \alpha } } ) } \frac { \big ( \big ( \lambda _ { x _ { t } } ^ { * } - p _ { t } \big ) ^ { \top } L \big ( q _ { 1 } - q _ { 2 } \big ) \big ) ^ { 2 } } { \mathbb { E } _ { \alpha \sim p _ { t } } [ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { \alpha } q _ { 1 } \big | S _ { \alpha } q _ { 2 } \big ) ] } \cdot \bigg ( \mathbb { E } _ { \alpha \sim p _ { t } } [ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { \alpha } \hat { q } _ { t } ( x _ { t } ) \big | S _ { \alpha } q ^ { * } ( x _ { t } ) \big ) ] \bigg ) . } \end{array}
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\begin{array} { r l } & { \left( ( \lambda _ { x _ { t } } ^ { * } - p _ { t } ) ^ { \top } L ( \hat { q } _ { t } ( x _ { t } ) - q ^ { * } ( x _ { t } ) ) \right) ^ { 2 } \leq \operatorname { C o v e r a g e } ( p _ { t } , \lambda _ { x _ { t } } ^ { * } ) \cdot \left( \mathbb { E } _ { a \sim p _ { t } } \left[ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { a } \hat { q } _ { t } ( x _ { t } ) \big | S _ { a } q ^ { * } ( x _ { t } ) \big ) \right] \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \leq \sqrt { \operatorname { C o v e r a g e } ( p _ { t } , \lambda _ { x _ { t } } ^ { * } ) \cdot \left( \mathbb { E } _ { a \sim p _ { t } } \left[ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { a } \hat { q } _ { t } ( x _ { t } ) \big | S _ { a } q ^ { * } ( x _ { t } ) \big ) \right] \right) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \leq \frac { 1 } { \gamma } \operatorname { C o v e r a g e } ( p _ { t } , \lambda _ { x _ { t } } ^ { * } ) + \gamma \left( \mathbb { E } _ { a \sim p _ { t } } \left[ D _ { \mathrm { H } } ^ { 2 } \big ( S _ { a } \hat { q } _ { t } ( x _ { t } ) \big | S _ { a } q ^ { * } ( x _ { t } ) \big ) \right] \right) . } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { R e g } _ { T } \leq \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { s u p } _ { x \in \Delta \left( \Pi \right) } \left[ p _ { t } ^ { \top } L \hat { q } _ { t } ( x _ { t } ) - \lambda ^ { \top } L \hat { q } _ { t } ( x _ { t } ) + \frac { 1 } { \gamma } \mathrm { C o v e r a g e } ( p _ { t } , \lambda ) \right] + \displaystyle \sum _ { t = 1 } ^ { T } \gamma \left( \mathbb { E } _ { a \sim p _ { t } } \left[ D _ { \mathrm { H } } ^ { 2 } \Big ( S _ { a } \hat { q } _ { t } ( x _ { t } ) \big | S _ { a } q ^ { * } ( x _ { t } ) \Big ) \right] \right) } \\ & { \qquad \leq \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { s u p } _ { x \in \Delta ( \Pi ) } \left[ p _ { t } ^ { \top } L \hat { q } _ { t } ( x _ { t } ) - \lambda ^ { \top } L \hat { q } _ { t } ( x _ { t } ) + \frac { 1 } { \gamma } \cdot \mathrm { C o v e r a g e } ( p _ { t } , \lambda ) \right] + \gamma \left( \mathrm { R e g } _ { \mathrm { O } } ^ { \mathrm { H } } \left( \mathscr { Q } , T \right) \right) . } \end{array}
$$

Th<sub>ere</sub>f<sub>ore</sub> th<sub>e</sub> <sub>regre</sub>t <sub>can</sub> b<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>as,</sub>

$$
\begin{array} { r l } & { \log _ { T } \leq \displaystyle \sum _ { t = 1 } ^ { T } \mathcal { V } _ { \gamma } \big ( \hat { q } _ { t } ( x _ { t } ) , p _ { t } \big ) + \gamma \left( \mathrm { R e g } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T \right) \right) } \\ & { \qquad \leq T \cdot \displaystyle \operatorname* { m a x } _ { t = 1 } ^ { T } \mathcal { V } _ { \gamma } \big ( \hat { q } _ { t } ( x _ { t } ) , p _ { t } \big ) + \gamma \left( \mathrm { R e g } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T \right) \right) . } \end{array}
$$

H<sub>ence, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>a</sub>t l<sub>eas</sub>t $1 - \delta ,$

$$
\mathrm { R e g } _ { T } \leq T \cdot \operatorname* { m a x } _ { t = 1 } ^ { T } \mathcal { V } _ { \gamma } \big ( \hat { q } _ { t } ( x _ { t } ) , p _ { t } \big ) + \gamma \cdot \overline { { \mathrm { R e g } } } _ { \mathrm { O n } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right) .
$$

M<sub>oreover,</sub>

$$
\begin{array} { r l } & { \mathrm { R e m a r k } 3 ( \mathrm { D E C ~ b o u n d } ) . \mathrm { ~ F o r ~ a l l } x \in \mathcal { K } , \hat { q } \in Q , p \in \Delta ( d ) , } \\ & { \qquad \mathcal { V } _ { \gamma } ^ { \mathrm { d e c } } ( \hat { q } ( x ) , Q _ { x } , p ) \le \mathcal { V } _ { \gamma } ( \hat { q } ( x ) , p ) . } \end{array}
$$

Proof.

$$
\begin{array} { r l } { \mathcal { V } _ { \gamma } ^ { \mathrm { d e c } } ( \hat { q } ( x ) , Q _ { x } , p ) = \underset { q \in \mathcal { G } _ { x , x } \leq \Delta ( \Lambda ) } { \operatorname* { s u p } } \left[ p ^ { \gamma } L q ( x ) - \lambda ^ { \gamma } L q ( x ) - \gamma \cdot \mathbb { E } _ { a \sim p } \left[ D _ { \mathbb { H } } ^ { 2 } \big ( S _ { a } \hat { q } ( x ) \big | S _ { a } q ( x ) \big ) \right] \right] } & \\ { = \underset { q \in \mathcal { G } _ { x , x } \leq \Delta ( \Lambda ) } { \operatorname* { s u p } } \left[ p ^ { \gamma } L q ( x ) - \lambda ^ { \gamma } L q ( x ) - \gamma \cdot \mathbb { E } _ { a \sim p } \left[ D _ { \mathbb { H } } ^ { 2 } \big ( S _ { a } \hat { q } ( x ) \big | S _ { a } q ( x ) \big ) \right] \right] } & \\ { = \underset { q \leq p _ { \lambda } , \ldots \leq \Delta ( \Lambda ) } { \operatorname* { s u p } } \left[ p ^ { \gamma } L \hat { q } ( x ) - \lambda ^ { \gamma } L \hat { q } ( x ) + ( \lambda - p ) ^ { \gamma } L ( \hat { q } ( x ) - q ( x ) ) - \gamma \cdot \mathbb { E } _ { a \sim p } \left[ D _ { \mathbb { H } } ^ { 2 } \big ( S _ { a } \hat { q } ( x ) \big | S _ { a } q ( x ) \big ) \right] \right] } & \\ { = \underset { q \leq p _ { \lambda } , \ldots \leq \Delta ( \Lambda ) } { \operatorname* { s u p } } \left[ p ^ { \gamma } L \hat { q } ( x ) - \lambda ^ { \gamma } L \hat { q } ( x ) + \frac { 1 } { \gamma } \mathrm { C o v e r n } { \operatorname* { c a g } } ( p , \lambda ) \right] } & \\  \leq \underset { q \leq q _ { \lambda } , \ldots \leq \Delta ( \Lambda ) } { \operatorname* { s u p } } \left[ p ^ { \gamma } L \hat { q } ( x ) - \lambda ^ { \gamma } L \hat { q } ( x ) + \frac { 1 } { \gamma } \mathrm { C o v e r n } { \operatorname* { c a g } } ( p , \lambda ) \right]  \end{array}
$$

## C.2 Proofs for Algorithm 2 (ϵ-Greedy.PMSide)

R<sub>eca</sub>ll th<sub>a</sub>t i<sub>n par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>oss</sub> dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo ac</sub>ti<sub>ons can</sub> b<sub>e expresse</sub>d i<sub>n</sub> t<sub>erms o</sub>f the feedback from other actions (Definitions 2.2 and 2.3), thou<sub>g</sub>h the two observabilit<sub>y</sub> conditions <sub>supp</sub>l<sub>y</sub> th<sub>a</sub>t <sub>express</sub>i<sub>on</sub> dif<sub>eren</sub>tl<sub>y.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> l<sub>emma</sub> <sub>a</sub>b<sub>s</sub>t<sub>rac</sub>t<sub>s</sub> b<sub>o</sub>th <sub>cons</sub>t<sub>ruc</sub>ti<sub>ons</sub> i<sub>n</sub>t<sub>o</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> reference witness G and bounds the coverage in terms of it.

Lemma C.1. Suppose there exists a reference witnessfunction $G : { \mathcal { A } } \times \Sigma  \mathbb { R } ^ { A }$ <sub>,</sub> i<sub>n</sub> th<sub>e sense</sub> th<sub>a</sub>t f<sub>or</sub> <sub>a</sub>ll $b , c \in { \mathcal { A } } \subseteq [ k ]$

$$
\sum _ { a \in \mathcal { A } } G _ { b } ( a , \Phi _ { a o } ) - \sum _ { a \in \mathcal { A } } G _ { c } ( a , \Phi _ { a o } ) = L _ { b o } - L _ { c o } \quad \forall o \in [ d ] .\tag{9}
$$

<sub>w</sub>h<sub>ere,</sub> $G _ { b }$ denotes b-th index of G. Then $\forall p , \lambda \in \Delta ( { \mathcal { A } } )$

$$
\mathrm { C o v e r a g e } ( p , \lambda ) \le 4 \sum _ { a \in \mathcal { A } } \frac { \left( \sum _ { b \in \mathcal { A } } ( p _ { b } - \lambda _ { b } ) \| G _ { b } ( a , : ) \| _ { \infty } \right) ^ { 2 } } { p _ { a } } \le 8 \sum _ { a \in \mathcal { A } } \sum _ { b \in \mathcal { A } } \frac { p _ { b } + \lambda _ { b } } { p _ { a } } \| G _ { b } ( a , : ) \| _ { \infty } ^ { 2 }
$$

Proof. First, since $D _ { \mathrm { H } } ^ { 2 } { \big ( } u { \big | } v { \big ) } \geq { \frac { 1 } { 4 } } \| u - v \| _ { 1 } ^ { 2 }$ (Lemma G.2), we have,

$$
\mathrm { C o v e r a g e } ( p , \lambda ) = \operatorname* { s u p } _ { q _ { 1 } , q _ { 2 } \in \Delta ( d ) } \frac { ( ( \lambda - p ) ^ { \top } L ( q _ { 1 } - q _ { 2 } ) ) ^ { 2 } } { \mathbb { E } _ { a \sim p } [ D _ { \mathrm { H } } ^ { 2 } ( S _ { a } q _ { 1 }  S _ { a } q _ { 2 } ) ] } \leq 4 \operatorname* { s u p } _ { q _ { 1 } , q _ { 2 } \in \Delta ( d ) } \frac { ( ( \lambda - p ) ^ { \top } L ( q _ { 1 } - q _ { 2 } ) ) ^ { 2 } } { \sum _ { a \in A } \| S _ { a } ( q _ { 1 } - q _ { 2 } ) \| _ { 1 } ^ { 2 } }
$$

Takin<sub>g</sub> a wei<sub>g</sub>hted summation of Equation (9) over $b ,$ c with weight $p _ { b } \lambda _ { c }$ <sub>g</sub><sup>i</sup>ves,

$$
( p - \lambda ) ^ { T } ( \sum _ { a } G ( a , \Phi _ { a o } ) - L _ { : , o } ) = 0 , \forall o
$$

A<sub>n</sub>d th<sub>us,</sub>

$$
\begin{array} { r l } { ( p - \lambda ) ^ { T } L ( q _ { 1 } - q _ { 2 } ) = \displaystyle \sum _ { o } ( p - \lambda ) ^ { T } \sum _ { a \in \mathcal { A } } G ( a , \Phi _ { a o } ) ( q _ { 1 } - q _ { 2 } ) _ { o } } & { } \\ { \displaystyle } & { = \sum _ { o } ( p - \lambda ) ^ { T } \sum _ { a \in \mathcal { A } } \sum _ { \sigma } G ( a , \sigma ) ( S _ { a } ) _ { \sigma , o } ( q _ { 1 } - q _ { 2 } ) _ { o } } \\ { \displaystyle } & { = ( p - \lambda ) ^ { T } \sum _ { \sigma } \sum _ { a \in \mathcal { A } } G ( a , \sigma ) ( S _ { a } ( q _ { 1 } - q _ { 2 } ) ) _ { \sigma } } \\ { \displaystyle } & { \leq \sum _ { b \in \mathcal { A } } \sum _ { \sigma } \sum _ { a \in \mathcal { A } } ( p _ { b } - \lambda _ { b } ) G _ { b } ( a , \sigma ) ( S _ { a } ( q _ { 1 } - q _ { 2 } ) ) _ { \sigma } } \end{array}
$$

A AA<sub>n</sub>d th<sub>ere</sub>f<sub>ore, exc</sub>h<sub>ang</sub>i<sub>ng</sub> th<sub>e summa</sub>ti<sub>on</sub> b<sub>y mov</sub>i<sub>ng</sub> $\scriptstyle \sum _ { o }$ t<sub>o</sub> th<sub>e</sub> i<sub>nnermos</sub>t<sub>, an</sub>d b<sub>y</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y,</sub> $\begin{array} { r } { | \sum _ { \sigma } G _ { b } ( a , \sigma ) S _ { a } ( q _ { 1 } - \overline { { q _ { 2 } } } ) _ { \sigma } | \leq \| G _ { b } ( a , : ) \| _ { \infty } \| S _ { a } ( q _ { 1 } - \overline { { q _ { 2 } } } ) \| _ { 1 } } \end{array}$ <sub>;</sub> thi<sub>s</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t

$$
\Big | ( p - \lambda ) ^ { T } L ( q _ { 1 } - q _ { 2 } ) \Big | \leq \sum _ { a \in \mathcal { A } } \sum _ { b \in \mathcal { A } } | p _ { b } - \lambda _ { b } | \| G _ { b } ( a , : ) \| _ { \infty } \| S _ { a } ( q _ { 1 } - q _ { 2 } ) \| _ { 1 } .
$$

A ATh<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> <sub>o</sub>f th<sub>e</sub> l<sub>emma</sub> <sub>now</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f <sub>coverage</sub> <sub>an</sub>d C<sub>auc</sub>h<sub>y-</sub>S<sub>c</sub>h<sub>warz</sub> i<sub>nequa</sub>lit<sub>y.</sub>

Th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>a</sub>t

$$
\begin{array} { r l r } {  { \big ( \sum _ { b \in \mathcal { A } } \big ( p _ { b } - \lambda _ { b } \big ) \| G _ { b } ( a , : ) \| _ { \infty } \big ) ^ { 2 } \le 2 \big ( \sum _ { b \in \mathcal { A } } p _ { b } \| G _ { b } ( a , : ) \| _ { \infty } \big ) ^ { 2 } + 2 \big ( \sum _ { b \in \mathcal { A } } \lambda _ { b } \| G _ { b } ( a , : ) \| _ { \infty } \big ) ^ { 2 } } } \\ & { } & { \le 2 \sum _ { b \in \mathcal { A } } p _ { b } \| G _ { b } ( a , : ) \| _ { \infty } ^ { 2 } + 2 \sum _ { b \in \mathcal { A } } \lambda _ { b } \| G _ { b } ( a , : ) \| _ { \infty } ^ { 2 } } \end{array}
$$

In Lemma C.1 the indices b and c range over a set A, which we take to be [k] for globally observable games and Π for locally observable games. In the globally observable case the behavioral policy <sub>p</sub>l<sub>aces mass on every ac</sub>ti<sub>on, so</sub> th<sub>e sums mus</sub>t <sub>run over a</sub>ll <sub>o</sub>f $[ k ] ;$ i<sub>n</sub> th<sub>e</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e case</sub> b<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es</sub> li<sub>e</sub> i<sub>n</sub> $\Delta ( \Pi )$ and the witness vanishes outside Π, so the restriction is without loss.

Lemma 3.1 (Guarantee for Alg. 2 for globally observable games). Let $\mathcal { G } = \left( L , \Phi \right)$ b<sub>e a g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serv-</sub> <sub>a</sub>bl<sub>e par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng game</sub> i<sub>n</sub> th<sub>e sense o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>2<sub>, an</sub>d <sub>se</sub>t $\beta : = \operatorname* { m a x } _ { ( a , b ) \in [ k ] \times [ k ] } \| w _ { ( a , b ) } \| _ { \infty } .$

W<sub>e</sub> h<sub>ave</sub> f<sub>or</sub> <sub>any</sub> $\hat { q } \in \Delta ( d )$

$$
\mathcal { V } _ { \gamma } \big ( \widehat { q } , p \big ) \leq \operatorname* { m a x } _ { \lambda \in \Delta ( \Pi ) } p ^ { T } L \widehat { q } - \lambda ^ { T } L \widehat { q } + \frac { 1 6 \beta ^ { 2 } } { \gamma } \sum _ { c = 1 } ^ { k } \frac { 1 } { p _ { c } } .
$$

Th<sub>us,</sub> f<sub>or</sub> <sub>every</sub> $\gamma > 1 6 k ^ { 2 } \beta ^ { 2 }$ , an<sup>d</sup> ever<sub>y</sub> $\hat { q } \in \Delta ( d )$ <sub>,</sub> Al<sub>g.</sub> 2 <sub>ou</sub>t<sub>pu</sub>t<sub>s</sub> $p$ <sub>suc</sub>h th<sub>a</sub>t

$$
\mathcal { V } _ { \gamma } \left( \hat { q } , p \right) \leq O \left( k \beta \sqrt { \frac { 1 } { \gamma } } \right) .
$$

Proof. For globally observable games, we can construct reference witness function by simply setting, f<sub>or a</sub>ll $b \in [ \breve { k } ]$

$$
G _ { b } ( \cdot , \cdot ) = w _ { b , a _ { 0 } } ( \cdot , \cdot )
$$

(for some reference action $a _ { 0 } )$

H<sub>ence,</sub> b<sub>y</sub> L<sub>emma</sub> C<sub>.</sub>1<sub>,</sub>

$$
\begin{array} { l l } { { \mathrm { C o v e r a g e } ( p , \lambda ) \le 1 6 \displaystyle \operatorname* { m a x } _ { ( a , b ) \in [ k ] \times [ k ] } \| w _ { ( a , b ) } \| _ { \infty } ^ { 2 } \sum _ { c = 1 } ^ { k } \frac { 1 } { p _ { c } } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } = 1 6 \beta ^ { 2 } \displaystyle \sum _ { c = 1 } ^ { k } \frac { 1 } { p _ { c } } . } } \end{array}
$$

Thi<sub>s</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f th<sub>e</sub> fi<sub>rs</sub>t <sub>par</sub>t <sub>o</sub>f th<sub>e</sub> l<sub>emma.</sub>

Let aˆ be the empirical best arm based on our current estimate ${ \hat { q } } .$ Th<sub>a</sub>t i<sub>s,</sub>

$$
\hat { a } : = \arg \operatorname* { m i n } _ { a \in \Pi } e _ { a } ^ { \intercal } { \cal L } \hat { q } ( x ) .
$$

S<sub>e</sub>t<sub>,</sub>

$$
p = ( 1 - \epsilon ) e _ { \hat { a } } + \epsilon \frac { { \bf 1 } } { k } ,
$$

<sub>so</sub> th<sub>a</sub>t $p _ { c } \ge \frac { \epsilon } { k }$ <sup>f</sup>or ever<sub>y</sub> $c \in [ k ]$ <sub>,</sub> h<sub>ence</sub>

$$
\sum _ { c = 1 } ^ { k } \frac { 1 } { p _ { c } } \leq \frac { k ^ { 2 } } { \epsilon } .
$$

$$
\begin{array} { r l } { \mathcal { V } _ { \gamma } \big ( \hat { q } ( x ) , p \big ) = } & { \underset { \mathcal { A } \ = \Delta ( 1 ) } { \operatorname* { s u p } } \left[ p ^ { \top } \hat { Z } \hat { q } ( x ) - \lambda ^ { \top } \hat { Z } \hat { q } ( x ) + \frac { 1 } { \gamma } \cdot \operatorname { C o v e r a g e } ( p , \lambda ) \right] } \\ & { \leq \rho ^ { \top } L \hat { q } ( x ) - e _ { \mathrm { s u } } ^ { \top } L \hat { q } ( x ) + \frac { 1 } { \gamma } \mathrm { l o } \theta ^ { \top } \displaystyle \sum _ { \ell = 1 } ^ { k } \frac { 1 } { p _ { \ell } } } \\ & { \leq \bigg ( ( 1 - \epsilon ) \epsilon _ { 0 } + \epsilon \frac { 1 } { k } \bigg ) ^ { \top } L \hat { q } ( x ) - e _ { \mathrm { s } } ^ { \top } L \hat { q } ( x ) + \frac { 1 6 k ^ { 2 } \beta ^ { 2 } } { \gamma \epsilon } } \\ & { \leq \epsilon \frac { 1 } { k } ^ { 1 1 } \tau \hat { L } \hat { q } ( x ) + \frac { 1 6 k ^ { 2 } \beta ^ { 2 } } { \gamma \epsilon } } \\ & { \leq \alpha + \frac { 1 6 k ^ { 2 } \beta ^ { 2 } } { \gamma \epsilon } } \\ & { = \sigma \bigg ( k \beta \sqrt { \frac { 1 } { \gamma } } \bigg ) . } \end{array}
$$

Thi<sub>s</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f th<sub>e</sub> <sub>secon</sub>d <sub>par</sub>t <sub>o</sub>f th<sub>e</sub> l<sub>emma.</sub>

## C.3 Proofs for Algorithm 3 (SquareCB.PMSide)

We can choose a spanning in-tree over the neighborhood graph (Π, E). That is, assign an arbitrary <sub>ac</sub>ti<sub>on</sub> $\mathrm { r o o t } _ { \mathcal { T } }$ as the root and choose an in-tree T such that, there is a unique directed $\mathrm { p a t h } _ { \tau } ( a )$ T Tfrom any action a ∈ Π to root ∈ Π (See Figure 1). Now, define witness reference function $G \tau : [ k ] \times \Sigma  \mathbb { R } ^ { \Pi }$ T∀a ∈ Π such that,

$$
G _ { a } ^ { \mathcal { T } } ( c , \sigma ) : = \sum _ { e \in \mathrm { p a t h } _ { \mathcal { T } } ( a ) } w _ { e } ( c , \sigma ) , \qquad a \in \Pi ,\tag{10}
$$

<sub>w</sub>h<sub>ere</sub> $w _ { e }$ denotes the witness function associated with the ed<sub>g</sub>e e b<sub>y</sub> local observabilit<sub>y</sub> (Definition 2.3). H<sub>ere</sub> it i<sub>s no</sub>t<sub>ewor</sub>th<sub>y</sub> th<sub>a</sub>t<sub>,</sub> l<sub>e</sub>t $\boldsymbol { e } = \left( e _ { 1 } , e _ { 2 } \right)$ th<sub>en</sub> $w _ { e } ( c , \sigma ) = \bar { 0 } ~ \forall c \dot { \notin } ~ \{ e _ { 1 } , e _ { 2 } \}$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
G _ { a } ^ { \mathcal { T } } ( c , \sigma ) = 0 \mathrm { i f } c \notin \mathrm { p a t h } _ { \mathcal { T } } ( a ) .\tag{11}
$$

Th<sub>ere</sub>f<sub>ore,</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>o</sub>b<sub>serva</sub>bilit<sub>y con</sub>diti<sub>on</sub> i<sub>n</sub>d<sub>uces a sparse</sub> $G _ { a } ^ { \mathcal { T } } ( \cdot , \sigma )$ f<sub>unc</sub>ti<sub>on, w</sub>hi<sub>c</sub>h <sub>w</sub>ill b<sub>ecome</sub> h<sub>an</sub>d<sub>y</sub> f<sub>or</sub> <sub>us</sub> t<sub>o</sub> <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>on</sub>l<sub>y</sub> f<sub>or</sub> th<sub>e</sub> <sub>ac</sub>ti<sub>ons</sub> <sub>popu</sub>l<sub>a</sub>ti<sub>ng</sub> it<sub>.</sub> F<sub>ur</sub>th<sub>ermore,</sub> <sub>we</sub> <sub>can</sub> <sub>a</sub>l<sub>so</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t<sub>,</sub> $\forall a \in \Pi$

$$
\sum _ { c \in \Pi } G _ { a } ^ { T } ( c , \Phi _ { c o } ) = \sum _ { c \in \Pi } \sum _ { e \in \mathrm { p a t h } _ { T } ( a ) } w _ { e } ( c , \Phi _ { c o } ) = \sum _ { e \in \mathrm { p a t h } _ { T } ( a ) } \sum _ { c \in \Pi } w _ { e } ( c , \Phi _ { c o } ) = L _ { a o } - L _ { \mathrm { r o o t } \tau o }\tag{12}
$$

Hence, witness referencefunction $G _ { T }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e con</sub>diti<sub>ons o</sub>f L<sub>emma</sub> C<sub>.</sub>1<sub>.</sub> H<sub>ence,</sub>

N<sub>ow we re</sub>l<sub>ax</sub> $\mathcal { V } _ { \gamma } \big ( \widehat { q } , p \big )$ <sub>w</sub>ith <sub>sparse</sub> $G _ { a } ^ { \mathcal { T } } ( \cdot , \sigma )$

Lemma 3.3. Let $\mathcal { G } = \left( L , \Phi \right)$ b<sub>e a</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng game</sub> i<sub>n</sub> th<sub>e sense o</sub>f Definition 2.3. Let T be an in-tree of the neighborhood graph and $G ^ { \mathcal { T } }$ b<sub>e</sub> it<sub>s</sub> <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>it<sub>ness</sub> referencefunction. For all $p \in \Delta ( \Pi )$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } & { \mathcal { V } _ { \gamma } \big ( \hat { q } ( x ) , p \big ) \leq \underset { \lambda \in \Delta ( \Pi ) } { \operatorname* { m a x } } p ^ { T } L \hat { q } ( x ) - \lambda ^ { T } L \hat { q } ( x ) + \frac { 8 } { \gamma } \displaystyle \sum _ { b \in \Pi } \displaystyle \sum _ { c \in \Pi } \left( \frac { \lambda _ { b } } { p _ { c } } + \frac { p _ { b } } { p _ { c } } \right) \| G ^ { T } ( c , \cdot ) _ { b } \| _ { \infty } ^ { 2 } } \\ & { \qquad \leq \underset { \lambda \in \Delta ( \Pi ) } { \operatorname* { m a x } } p ^ { T } L \hat { q } ( x ) - \lambda ^ { T } L \hat { q } ( x ) + \frac { 3 2 } { \gamma } \displaystyle \operatorname* { m a x } _ { e \in \mathcal { E } } \| w _ { e } \| _ { \infty } ^ { 2 } \big ( } { _ { ( b \in \Pi , c \in \Pi ) : c \in \mathrm { p a t h } _ { T } ( b ) } } \left( \frac { \lambda _ { b } } { p _ { c } } + \frac { p _ { b } } { p _ { c } } \right) .  \end{array}
$$

Proof. By Lemma C.1,

$$
\mathrm { C o v e r a g e } ( p , \lambda ) \leq 8 \sum _ { b \in \Pi } \sum _ { c \in \Pi } \left( \frac { \lambda _ { b } } { p _ { c } } + \frac { p _ { b } } { p _ { c } } \right) \Vert G _ { b } ^ { \mathcal { T } } ( c , \cdot ) \Vert _ { \infty } ^ { 2 } .
$$

Thi<sub>s</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> <sub>o</sub>f th<sub>e</sub> l<sub>emma.</sub>

F<sub>ur</sub>th<sub>ermore, we</sub> h<sub>ave</sub>

$$
G _ { b } ^ { \mathcal { T } } ( c , \cdot ) = \sum _ { e \in \mathrm { p a t h } _ { \mathcal { T } } ( b ) } w _ { e } ( c , \cdot ) ,
$$

L<sub>e</sub>t $\boldsymbol { e } = \left( e _ { 1 } , e _ { 2 } \right)$ <sub>,</sub> th<sub>en</sub> b<sub>y</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>3<sub>,</sub> $w _ { e } ( c , \cdot ) = 0 \forall c \neq e _ { 1 } , e _ { 2 }$ <sub>.</sub> H<sub>ence,</sub>

$$
\begin{array} { r l } { \mathrm { C o v e r a g e } ( p , \lambda ) \le \displaystyle 8 \sum _ { \substack { i \in \mathrm { I I } \subsetneq \mathrm { I I } } } ( \frac { \lambda _ { \mathrm { P } } } { p _ { \mathrm { c } } } + \frac { p _ { i } } { p _ { \mathrm { c } } } ) | G _ { \mathrm { b } } ^ { T } ( \tau _ { \mathrm { c } } ; \tau ) | _ { \infty } ^ { 2 } } & { } \\ { = \displaystyle 8 \sum _ { \substack { i \in \mathrm { I I } \backslash i \neq 1 } } ( \frac { \lambda _ { \mathrm { P } } } { p _ { \mathrm { c } } } + \frac { p _ { i } } { p _ { \mathrm { c } } } ) \displaystyle 1 \sum _ { \substack { \mathrm { c e a n h } \tau _ { \mathrm { P } } ( k ) } } w _ { \mathrm { \lambda { \mathrm { e } } } } ( c , \cdot ) \textbf { 1 } \{ c \in \{ \varepsilon _ { 1 } , e _ { 2 } \} \} | _ { \infty } ^ { 2 } } & { } \\ { = \displaystyle 8 \operatorname* { m a x } _ { \mathrm { c e } } t w _ { \mathrm { i } } | w _ { \mathrm { i } } | ^ { 2 } \displaystyle \sum _ { \mathrm { b y c h } \tau } ( \frac { \lambda _ { \mathrm { b } } } { p _ { \mathrm { c } } } + \frac { p _ { i } } { p _ { \mathrm { c } } } ) \displaystyle 1 \sum _ { \substack { \mathrm { c e a n h } \tau _ { \mathrm { P } } ( k ) } } \displaystyle 1 \sum _ { \substack { \mathrm { c y c h } \tau ( k ) } } \{ c \in \{ \varepsilon _ { 1 } , e _ { 2 } \} \} | _ { \infty } ^ { 2 } } & { } \\  = \displaystyle 3 2 \operatorname* { m a x } _ { \mathrm { c e } } t w _ { \mathrm { i } } | w _ { \mathrm { i } } | ^ { 2 } \displaystyle \sum _ { \mathrm { c } , \tau \in \mathrm { I I I } \backslash i \neq 1 } \sum _ { \substack { \mathrm { f i r } \in \mathrm { I I } \backslash i \neq 1 } } ( \frac { \lambda _ { \mathrm { b } } }  p  \end{array}
$$

Thi<sub>s</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> <sub>o</sub>f th<sub>e</sub> l<sub>emma.</sub>

Ab<sub>ove resu</sub>lt i<sub>s</sub> t<sub>rue</sub> f<sub>or any</sub> i<sub>n-</sub>t<sub>ree w</sub>ith <sub>an ar</sub>bit<sub>rary roo</sub>t<sub>.</sub> H<sub>owever,</sub> i<sub>n genera</sub>l<sub>, we</sub> d<sub>o no</sub>t <sub>wan</sub>t th<sub>e</sub> paths to be arbitrar<sub>y</sub>: for a “bad” action c and a “good” action $b ( \mathrm { i } . \mathrm { e } . , \bar { e } _ { c } ^ { \top } L \hat { q }$ i<sub>s</sub> l<sub>arge w</sub>hil<sub>e</sub> $e _ { b } ^ { \top } L \hat { q }$ i<sub>s</sub> small), we would like $p _ { c }$ t<sub>o</sub> b<sub>e sma</sub>ll <sub>an</sub>d $p _ { b }$ t<sub>o</sub> b<sub>e</sub> l<sub>arge</sub> t<sub>o ensure</sub> $p ^ { T } L \hat { q } ( x ) - \lambda ^ { T } L \hat { q } ( x )$ t<sub>o</sub> b<sub>e sma</sub>ll<sub>;</sub> however, if c happens to lie in $\mathrm { p a t h } _ { \tau } ( b )$ <sub>,</sub> th<sub>a</sub>t <sub>can</sub> <sub>cause</sub> th<sub>e</sub> $\frac { \lambda _ { b } } { p _ { c } }$ t<sub>e</sub>rm t<sub>o</sub> b<sub>e</sub> <sub>ve</sub>r<sub>y</sub> l<sub>a</sub>r<sub>ge.</sub> H<sub>e</sub>n<sub>ce,</sub>

1<sub>.</sub> W<sub>e nee</sub>d t<sub>o c</sub>h<sub>oose an</sub> i<sub>n-</sub>t<sub>ree</sub> $\tau _ { \ast }$ , such that the path to an<sub>y</sub> action a have actions with smaller <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d l<sub>osses, suc</sub>h th<sub>a</sub>t <sub>ass</sub>i<sub>gn</sub>i<sub>ng</sub> hi<sub>g</sub>h<sub>er pro</sub>b<sub>a</sub>bilit<sub>y</sub> f<sub>or</sub> th<sub>em</sub> i<sub>s</sub> b<sub>o</sub>th i<sub>n</sub>f<sub>orma</sub>ti<sub>ve an</sub>d <sub>rewar</sub>di<sub>ng.</sub>

2<sub>.</sub> W<sub>e</sub> <sub>nee</sub>d t<sub>o</sub> fi<sub>n</sub>d <sub>a</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> t<sub>o</sub> <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>l<sub>ong</sub> thi<sub>s</sub> i<sub>n-</sub>t<sub>ree</sub> $\tau$ res<sub>p</sub>ectin<sub>g</sub> 1), such th<sub>a</sub>t b<sub>o</sub>th $p ^ { T } L \hat { q } ( x ) - \lambda ^ { T } L \hat { q } ( x )$ <sub>an</sub>d $\begin{array} { r } { \frac { 3 2 } { \gamma } \operatorname* { m a x } _ { e \in \mathcal { E } } \| w _ { e } \| _ { \infty } ^ { 2 } \sum _ { ( b \in \Pi , c \in \Pi ) : c \in \mathrm { p a t h } _ { \mathcal { T } } ( b ) } \left( \frac { \lambda _ { b } } { p _ { c } } + \frac { p _ { b } } { p _ { c } } \right) } \end{array}$ are <sub>con</sub>t<sub>ro</sub>ll<sub>e</sub>d<sub>.</sub>

The best solution for 1) is an $\pmb { L } \hat { q } \cdot$ <sub>-</sub>d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n-</sub>t<sub>ree.</sub> W<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub> $\tau .$ <sub>-</sub>d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n-</sub>t<sub>ree</sub> <sub>as</sub> f<sub>o</sub>ll<sub>ows,</sub>

Definition C.1 (T-decreasing in-tree [Lattimore & Szepesvari, 2020])´ . Let $\tau$ be an in-tree over [Π]. A <sub>vec</sub>t<sub>or</sub> $y \in \mathbb { R } ^ { \Pi }$ is called T-decreasing if $y _ { a } \geq y _ { b }$ f<sub>or a</sub>ll di<sub>rec</sub>t<sub>e</sub>d <sub>e</sub>d<sub>ge</sub> $( a , b ) \in \mathcal { T }$ <sub>.</sub> Th<sub>a</sub>t i<sub>s,</sub> $a \mapsto y _ { a }$ i<sub>s</sub> d<sub>ecreas</sub>i<sub>ng</sub> t<sub>owar</sub>d<sub>s</sub> th<sub>e</sub> <sub>roo</sub>t <sub>o</sub>f $\tau$

Basicall<sub>y</sub>, an Lqˆ-decreasing in-tree $\tau$ will have the empirical best action (arg min $\boldsymbol { \mathbf { \rho } } _ { a \in \Pi } e _ { a } ^ { \top } \pmb { L } \hat { q } ( \boldsymbol { x } ) )$ as th<sub>e roo</sub>t <sub>an</sub>d f<sub>or every</sub> di<sub>rec</sub>t<sub>e</sub>d <sub>e</sub>d<sub>ge</sub> $e = { \bigl ( } a , b { \bigr ) } , e _ { a } ^ { \intercal } L { \hat { q } } ( x ) { \overset { \cdot } { \geq } } e _ { b } ^ { \intercal } L { \hat { q } } ( x )$ <sub>.</sub> S<sub>uc</sub>h <sub>a</sub> $\bar { \tau }$ ∈is guaranteed to exist b<sub>y</sub> Lattimore & Sze<sub>p</sub>esvari [2020] (Lemma 19).´

For 2), we observe that, inverse-<sub>g</sub>a<sub>p</sub>-wei<sub>g</sub>htin<sub>g</sub> (IGW), b<sub>y</sub> construction, assi<sub>g</sub>ns <sub>p</sub>robabilit<sub>y</sub> in an i<sub>ncreas</sub>i<sub>ng</sub> f<sub>as</sub>hi<sub>on</sub> t<sub>owar</sub>d th<sub>e</sub> <sub>roo</sub>t<sub>.</sub> N<sub>ow</sub> <sub>we</sub> <sub>prov</sub>id<sub>e</sub> th<sub>e</sub> <sub>proo</sub>f f<sub>or</sub> L<sub>emma</sub> 3<sub>.</sub>2<sub>.</sub>

Lemma 3.2 (Guarantee for Alg. 3 for locally observable games). Let $\mathcal { G } = \left( L , \Phi \right)$ b<sub>e a</sub> l<sub>oca</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> <sub>game</sub> i<sub>n</sub> th<sub>e</sub> <sub>sense</sub> <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>3<sub>.</sub> F<sub>or</sub> <sub>every</sub> $\gamma > 0$ , an<sup>d</sup> ever<sub>y</sub> $\hat { q } \in \Delta ( d )$ <sub>,</sub> Al<sub>g.</sub> 2 <sub>run w</sub>ith $\kappa = \frac { \gamma } { 8 k m ^ { 2 } }$ <sub>an</sub>d $\mu = k ,$ , outputs p such that

$$
\mathcal { V } _ { \gamma } \left( \hat { q } , p \right) \leq O \left( \frac { m ^ { 2 } k ^ { 2 } } { \gamma } \right) .
$$

Proof. Recall that we define in $\mathrm { A l g } . 3$ th<sub>a</sub>t<sub>,</sub> $\hat { b } = \mathrm { a r g m i n } _ { a \in \Pi } e _ { a } ^ { \top } { \cal L } \hat { q } ( x )$ <sub>an</sub>d $\forall a \in \Pi \setminus \{ \hat { b } \} , p ( a ) =$ $\frac { 1 } { \mu + \kappa \big ( e _ { a } - e _ { \hat { b } } \big ) ^ { \top } L \hat { q } ( x ) }$ <sub>an</sub>d $\begin{array} { r } { p ( \hat { b } ) = 1 - \sum _ { a \in \Pi \setminus \{ \hat { b } \} } p ( a ) } \end{array}$

W<sub>e c</sub>h<sub>oose</sub> $\tau$ t<sub>o</sub> b<sub>e</sub> $\pmb { L } \hat { q } ( x )$ <sub>-</sub>d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n-</sub>t<sub>ree</sub> i<sub>n</sub> th<sub>e ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d <sub>grap</sub>h<sub>; we</sub> h<sub>ave</sub> th<sub>a</sub>t f<sub>or any</sub> $( b , c )$ <sub>suc</sub>h th<sub>a</sub>t $c \in \mathrm { p a t h } _ { T } ( b ) , p _ { c } \geq p _ { b }$ <sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> t<sub>rue</sub> b<sub>ecause</sub> f<sub>or any</sub> $c \in$ path(b) of $\pmb { L } \hat { q } ( x )$ <sub>-</sub>d<sub>ecreas</sub>i<sub>ng</sub> Tin-tree, estimated loss of c is lower than estimated loss of b. Hence, by construction of IGW based <sub>p</sub>o<sup>li</sup>c<sub>y</sub> $p , p _ { c } \ge p _ { b }$

B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> 3<sub>.</sub>3 <sub>we</sub> h<sub>ave,</sub>

$$
\mathrm { C o v e r a g e } ( p , \lambda ) \leq 3 2 \operatorname* { m a x } _ { e \in \mathcal { E } } \left\| w _ { e } \right\| _ { \infty } ^ { 2 } \sum _ { ( b , c ) : c \in \mathrm { p a t h } _ { \mathcal { T } } ( b ) } \left( \frac { \lambda _ { b } } { p _ { c } } + \frac { p _ { b } } { p _ { c } } \right) .
$$

Since for every c in pa $\mathrm { h } _ { \mathcal { T } } ( b ) , p _ { c } \geq p _ { b }$

$$
\begin{array} { l } { \displaystyle \mathrm { C o v e r a g e } ( p , \lambda ) \le 3 2 \displaystyle \operatorname* { m a x } _ { e \in \mathcal { E } } \| w _ { e } \| _ { \infty } ^ { 2 } \sum _ { b \in \Pi } \left( \frac { \lambda _ { b } } { p _ { b } } + \frac { p _ { b } } { p _ { b } } \right) } \\ { \displaystyle \qquad \le 3 2 \displaystyle \operatorname* { m a x } _ { e \in \mathcal { E } } \| w _ { e } \| _ { \infty } ^ { 2 } k ^ { 2 } + 3 2 \displaystyle \operatorname* { m a x } _ { e \in \mathcal { E } } \| w _ { e } \| _ { \infty } ^ { 2 } k \sum _ { b \in \Pi } \frac { \lambda _ { b } } { p _ { b } } } \end{array}
$$

M<sub>oreover,</sub>

$$
\operatorname* { s u p } _ { \lambda \in \Delta ( \Pi ) } \left\{ p ^ { \top } L \hat { q } ( x ) - \lambda ^ { \top } L \hat { q } ( x ) + \frac { 1 } { \gamma } \cdot 3 2 \operatorname* { m a x } _ { e \in \mathcal { E } } \| w _ { e } \| _ { \infty } ^ { 2 } k \sum _ { b \in \Pi } \frac { \lambda _ { b } } { p b } \right\}
$$

$$
\leq \operatorname* { s u p } _ { \lambda \in \Delta \left( \Pi \right) } \left[ \sum _ { b \in \Pi } \left( p _ { b } - \lambda _ { b } \right) L _ { b , : } \hat { q } ( x ) + \frac { 1 } { \gamma } \cdot 3 2 \operatorname* { m a x } _ { e \in \mathcal { E } } \left. w _ { e } \right. _ { \infty } ^ { 2 } k \sum _ { b \in \Pi } \lambda _ { b } \left( \mu + \kappa \left( e _ { b } - e _ { \hat { b } } \right) ^ { \top } L \hat { q } ( x ) \right) \right]
$$

I<sub>n</sub> th<sub>e a</sub>b<sub>ove</sub> i<sub>nequa</sub>lit<sub>y, we use</sub> th<sub>e o</sub>b<sub>serva</sub>ti<sub>on</sub> th<sub>a</sub>t $p _ { \hat { b } }$ i<sub>s</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> $\frac { 1 } { k }$ <sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> t<sub>rue, s</sub>i<sub>nce</sub> $\mu = k$

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { \lambda \in \Delta ( \Pi ) } \{ p ^ { \top } L \hat { q } ( \boldsymbol { x } ) - \lambda ^ { \top } L \hat { q } ( \boldsymbol { x } ) + \frac { 1 } { \gamma } \cdot 3 2 \operatorname* { m a x } _ { e \in \mathcal { E } } \lVert \boldsymbol { w } _ { e } \rVert _ { \infty } ^ { 2 } k \sum _ { b \in \Pi } \frac { \lambda _ { b } } { p _ { b } } \} } } \\ & { \leq } & { \operatorname* { s u p } _ { \lambda \in \Delta ( \Pi ) } [ \sum _ { b \in \Pi } ( p _ { b } - \lambda _ { b } ) L _ { b , : } \hat { q } ( \boldsymbol { x } ) + \frac { 1 } { \gamma } \cdot 3 2 \operatorname* { m a x } _ { e \in \mathcal { E } } \lVert \boldsymbol { w } _ { e } \rVert _ { \infty } ^ { 2 } k ^ { 2 } + \sum _ { b \in \Pi } \lambda _ { b } ( e _ { b } - e _ { \hat { b } } ) ^ { \top } L \hat { q } ( \boldsymbol { x } ) ] } \\ & { } & { ( \mu = k , \kappa = \frac { \gamma } { 3 2 \operatorname* { m a x } _ { e \in \mathcal { E } } \lVert \boldsymbol { w } _ { \infty } \rVert _ { \infty } ^ { 2 } k } ) } \end{array}
$$

F<sub>ur</sub>th<sub>ermore,</sub>

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { k \in \mathrm { I I } \setminus \{ b \} } \left[ \displaystyle \sum _ { b \in \mathbb { R } } \left( p _ { b } - \lambda _ { b } \right) L _ { b \cdot } \hat { q } ( x ) + \displaystyle \sum _ { b \in \mathbb { I } } \lambda _ { b } \left( e _ { b } - e _ { b } \right) ^ { \top } L \hat { q } ( x ) \right] } \\ { \displaystyle \quad \times \mathbb { E } = \sum _ { b \not = \mathbb { I } } p _ { b } \cdot e _ { b } ^ { \top } L \hat { q } ( x ) - e _ { b } ^ { \top } L \hat { q } ( x ) } \\ { \displaystyle \quad = \sum _ { b \in \mathrm { I I } \setminus \{ b \} } p _ { b } \cdot e _ { b } ^ { \top } L \hat { q } ( x ) - e _ { b } ^ { \top } L \hat { q } ( x ) } \\ { \displaystyle \quad = \sum _ { b \in \mathrm { I I } \setminus \{ b \} } \frac { 1 } { \mu + \kappa \left( e _ { a } - e _ { b } \right) ^ { \top } L \hat { q } ( x ) } \cdot e _ { b } ^ { \top } L \hat { q } ( x ) - e _ { b } ^ { \top } L \hat { q } ( x ) } \\ { \displaystyle \quad = \sum _ { b \in \mathrm { I I } \setminus \{ b \} } \frac { 1 } { \mu + \kappa \left( e _ { a } - e _ { b } \right) ^ { \top } L \hat { q } ( x ) } \cdot e _ { b } ^ { \top } L \hat { q } ( x ) - e _ { b } ^ { \top } L \hat { q } ( x ) } \\ { \displaystyle \leq \frac { k } { \kappa } - \frac { 1 } { \gamma } \cdot 3 2 \frac { \log \mathrm { a x } } { e _ { a } \varepsilon } \| w _ { c } \| _ { \infty } ^ { 2 } k ^ { 2 } } \end{array}
$$

H<sub>ence,</sub> $\forall \widehat { q } \in \Delta ( d )$

$$
\mathcal { V } _ { \gamma } \big ( \widehat { q } , p \big ) \leq \frac { 1 } { \gamma } \cdot 9 6 \operatorname* { m a x } _ { e \in \mathcal { E } } \lVert w _ { e } \rVert _ { \infty } ^ { 2 } k ^ { 2 } .
$$

F<sub>ur</sub>th<sub>ermore,</sub> <sub>rev</sub>i<sub>s</sub>iti<sub>ng</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>3<sub>,</sub>

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { a o } - { \cal L } _ { b o } = \sum _ { c \epsilon \left\{ a , b \right\} } w _ { e } ( c , \Phi _ { c o } ) } , \qquad \mathrm { f o r ~ a l l } ~ o \in [ d ] . }  \\ { { \mathrm { } } } \\ { { \mathrm { } = w _ { e } \left( a , \Phi _ { c o } \right) + w _ { e } ( b , \Phi _ { c o } ) . } } \end{array}
$$

B<sub>y</sub> L<sub>emma</sub> $\mathrm { { G . 3 } } ,$ <sub>one</sub> <sub>can</sub> <sub>a</sub>l<sub>ways</sub> fi<sub>n</sub>d<sub>,</sub> <sub>a</sub> <sub>w</sub>it<sub>ness</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> $\| w _ { e } \| _ { \infty } \leq \frac { m } { 2 }$ <sub>,</sub> H<sub>ence,</sub> $\forall \widehat { q } \in \Delta ( d )$

$$
\mathcal { V } _ { \gamma } \left( \hat { q } , p \right) = O \left( \frac { m ^ { 2 } k ^ { 2 } } { \gamma } \right) .
$$

Thi<sub>s</sub> <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub>

## C<sub>.</sub>4 N<sub>on-degeneracy assumption</sub>

D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>3 i<sub>s app</sub>li<sub>ca</sub>bl<sub>e on</sub>l<sub>y w</sub>h<sub>en</sub> th<sub>e game</sub> i<sub>s non-</sub>d<sub>egenera</sub>t<sub>e.</sub> Wh<sub>en</sub> th<sub>ere</sub> i<sub>s</sub> d<sub>egenera</sub>t<sub>e ac</sub>ti<sub>on</sub> in [k], the definition for local observability becomes,

Definition C.2 (Local observability, general form; Bartok et al. [2012b]) ´ . A globally observable <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng game</sub> i<sub>s</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e</sub> if f<sub>or every e</sub>d<sub>ge</sub> $e = ( a , b ) \in \mathcal { E }$ <sub>o</sub>f th<sub>e ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d

<sub>grap</sub>h th<sub>e</sub> <sub>w</sub>it<sub>ness</sub> f<sub>unc</sub>ti<sub>on</sub> $w _ { e }$ <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>.</sub>2 <sub>may</sub> b<sub>e</sub> <sub>c</sub>h<sub>osen</sub> <sub>suppor</sub>t<sub>e</sub>d <sub>on</sub>

$$
N _ { a , b } ^ { + } : = \{ c \in [ k ] : \mathcal { C } _ { a } \cap \mathcal { C } _ { b } \subseteq \mathcal { C } _ { c } \} ,
$$

th<sub>a</sub>t i<sub>s,</sub> $w _ { e } ( c , \cdot ) \equiv 0$ for all c ∉ $N _ { a , b } ^ { + } ,$ <sub>so</sub> th<sub>a</sub>t

$$
L _ { a o } - L _ { b o } = \sum _ { c \in N _ { a , b } ^ { + } } w _ { e } ( c , \Phi _ { c o } ) , \qquad { \mathrm { f o r ~ a l l } } o \in [ d ] .
$$

Th<sub>e se</sub>t $N _ { a , b } ^ { + }$ <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> t<sub>wo en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>s,</sub> b<sub>u</sub>t <sub>may a</sub>l<sub>so con</sub>t<sub>a</sub>i<sub>n</sub> d<sub>egenera</sub>t<sub>e ac</sub>ti<sub>ons w</sub>h<sub>ose ce</sub>ll<sub>s</sub> li<sub>e</sub> i<sub>ns</sub>id<sub>e</sub> th<sub>e s</sub>h<sub>are</sub>d f<sub>ace</sub> $\mathcal { C } _ { a } \cap \mathcal { C } _ { b }$ <sub>.</sub> F<sub>or</sub> $d = 3 ,$ , for instance, a degenerate action with dim $\left( \mathcal { C } _ { c } \right) = 1$ i<sub>s</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>on</sub>l<sub>y a</sub>l<sub>ong a</sub> li<sub>ne segmen</sub>t<sub>, w</sub>h<sub>ereas a</sub> P<sub>are</sub>t<sub>o op</sub>ti<sub>ma</sub>l <sub>ac</sub>ti<sub>on</sub> i<sub>s op</sub>ti<sub>ma</sub>l <sub>on a</sub> t<sub>wo-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>reg</sub>i<sub>on,</sub> <sub>ye</sub>t <sub>suc</sub>h <sub>an ac</sub>ti<sub>on can s</sub>till b<sub>e a necessary source o</sub>f i<sub>n</sub>f<sub>orma</sub>ti<sub>on.</sub>

T<sub>wo par</sub>t<sub>s o</sub>f <sub>our</sub> d<sub>eve</sub>l<sub>opmen</sub>t b<sub>rea</sub>k <sub>un</sub>d<sub>er</sub> thi<sub>s more genera</sub>l <sub>con</sub>diti<sub>on.</sub> Fi<sub>rs</sub>t<sub>,</sub> th<sub>e exp</sub>l<sub>ora</sub>ti<sub>on</sub> <sub>ru</sub>l<sub>e mus</sub>t <sub>ass</sub>i<sub>gn pro</sub>b<sub>a</sub>bilit<sub>y</sub> t<sub>o</sub> th<sub>e</sub> d<sub>egenera</sub>t<sub>e ac</sub>ti<sub>ons as we</sub>ll<sub>, w</sub>hi<sub>c</sub>h <sub>requ</sub>i<sub>res a mo</sub>difi<sub>ca</sub>ti<sub>on o</sub>f S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide.</sub> S<sub>eco</sub>nd<sub>,</sub> L<sub>e</sub>mm<sub>a</sub> G<sub>.</sub>3 n<sub>o</sub> l<sub>o</sub>n<sub>ge</sub>r <sub>app</sub>li<sub>es a</sub>nd m<sub>us</sub>t b<sub>e</sub> r<sub>ep</sub>l<sub>ace</sub>d b<sub>y</sub> Pr<sub>opos</sub>iti<sub>o</sub>n 21 <sub>o</sub>f Lattimore & Sze<sub>p</sub>esvari [2020] (see Lemma´ $\mathrm { G } . 4$ for restatement), which introduces a ex<sub>p</sub>onential factor in d. The modification to the algorithm is straightforward, and we conjecture that SquareCB.PMSide cont<sup>i</sup>nues to <sub>g</sub>uarantee $O ( \sqrt { T } )$ <sub>regre</sub>t<sub>,</sub> b<sub>u</sub>t th<sub>e ana</sub>l<sub>ys</sub>i<sub>s requ</sub>i<sub>res care.</sub> W<sub>e</sub> l<sub>eave</sub> thi<sub>s ex</sub>t<sub>ens</sub>i<sub>on</sub> t<sub>o</sub> f<sub>u</sub>t<sub>ure</sub> <sub>wor</sub>k<sub>.</sub>

## D C<sub>ontextual</sub> <sub>partial</sub> <sub>monitoring:</sub> <sub>reduction</sub> <sub>to</sub> <sub>offline</sub> <sub>regression</sub>

F<sub>or</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons an</sub>d <sub>examp</sub>l<sub>es o</sub>f <sub>o</sub>fli<sub>ne ou</sub>t<sub>come es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>es,</sub> Pl<sub>ease see</sub> S<sub>ec</sub>ti<sub>on</sub> B<sub>.</sub>2<sub>.</sub>

## D<sub>.</sub>1 OE2D<sub>.</sub>PMS<sub>ide</sub>

Th<sub>e</sub> b<sub>oun</sub>d<sub>s on</sub> $\mathcal { V } _ { \gamma } ( \hat { q } ( x ) , p ( x ) )$ established for <sub>g</sub>loball<sub>y</sub> observable <sub>g</sub>ames (Lemma 3.1) and for locall<sub>y</sub> observable <sub>g</sub>ames (Lemma 3.2), under the ex<sub>p</sub>loration strate<sub>g</sub>ies ϵ-Greedy.PMSide and S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>rema</sub>i<sub>n</sub> <sub>va</sub>lid i<sub>n</sub> th<sub>e</sub> <sub>o</sub>fli<sub>ne</sub> <sub>se</sub>tti<sub>ng,</sub> <sub>s</sub>i<sub>nce</sub> b<sub>o</sub>th h<sub>o</sub>ld f<sub>or</sub> <sub>every</sub> $\hat { q } \in \Delta ( d )$ <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> f<sub>or</sub> <sub>a</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e.</sub>

Wh<sub>a</sub>t d<sub>oes no</sub>t <sub>carry over</sub> i<sub>s</sub> th<sub>e regre</sub>t d<sub>ecompos</sub>iti<sub>on o</sub>f Th<sub>eorem</sub> 3<sub>.</sub>1<sub>.</sub> I<sub>nspec</sub>ti<sub>ng</sub> th<sub>e guaran</sub>t<sub>ee o</sub>f A<sub>ssump</sub>ti<sub>on</sub> B<sub>.</sub>1<sub>,</sub> <sub>an</sub> <sub>o</sub>fli<sub>ne</sub> <sub>orac</sub>l<sub>e</sub>’<sub>s</sub> H<sub>e</sub>lli<sub>nger</sub> b<sub>oun</sub>d <sub>ma</sub>k<sub>es</sub> t<sub>wo</sub> d<sub>eman</sub>d<sub>s</sub> <sub>o</sub>f th<sub>e</sub> <sub>samp</sub>li<sub>ng</sub> <sub>po</sub>li<sub>cy:</sub> it <sub>mus</sub>t b<sub>e</sub> h<sub>e</sub>ld fi<sub>xe</sub>d <sub>over</sub> th<sub>e</sub> <sub>roun</sub>d<sub>s</sub> t<sub>o</sub> <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> b<sub>oun</sub>d i<sub>s</sub> <sub>app</sub>li<sub>e</sub>d<sub>,</sub> <sub>an</sub>d it <sub>mus</sub>t b<sub>e</sub> th<sub>e</sub> <sub>po</sub>li<sub>cy</sub> th<sub>a</sub>t collected the data from which qˆ was estimated.

The first is obtained b<sub>y</sub> freezin<sub>g</sub> the <sub>p</sub>olic<sub>y</sub> within each e<sub>p</sub>och (Line 8). The second is where covera<sub>g</sub>e d<sub>oes</sub> th<sub>e wor</sub>k<sub>.</sub> B<sub>ecause</sub> $\mathrm { C o v e r a g e } ( \bar { p } , \lambda )$ i<sub>s a supremum over pa</sub>i<sub>rs o</sub>f <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> i<sub>n</sub> th<sub>e c</sub>l<sub>ass an</sub>d <sub>never</sub> <sub>re</sub>f<sub>erences</sub> ${ \hat { q } } ,$ <sub>a</sub> <sub>po</sub>li<sub>cy</sub> <sub>w</sub>ith <sub>sma</sub>ll <sub>coverage</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>es</sub> <sub>any</sub> t<sub>wo</sub> <sub>mem</sub>b<sub>ers</sub> <sub>o</sub>f $\mathcal { Q } ,$ in particular qˆ and $q ^ { * }$ <sub>, w</sub>hi<sub>c</sub>h<sub>ever es</sub>ti<sub>ma</sub>t<sub>e</sub> th<sub>e orac</sub>l<sub>e</sub> h<sub>appens</sub> t<sub>o re</sub>t<sub>urn.</sub> L<sub>emma</sub> D<sub>.</sub>1 <sub>ma</sub>k<sub>es</sub> thi<sub>s prec</sub>i<sub>se.</sub>

The DEC of Foster et al. [2021] lacks this <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub>: its ex<sub>p</sub>loration term $- \gamma \cdot \mathbb { E } _ { a \sim p } \left\lceil D _ { \mathrm { H } } ^ { 2 } \big ( S _ { a } \hat { q } ( x ) \big | S _ { a } q \big ) \right\rceil$ i<sub>s anc</sub>h<sub>ore</sub>d <sub>a</sub>t ${ \hat { q } } .$ In an epoch-based algorithm (Lines 7–14), the estimation error for a given qˆ therefore <sub>canno</sub>t b<sub>e</sub> <sub>con</sub>t<sub>ro</sub>ll<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> <sub>a</sub> <sub>po</sub>li<sub>cy</sub> f<sub>rom</sub> <sub>a</sub> dif<sub>eren</sub>t <sub>epoc</sub>h<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>p</sub>l<sub>aye</sub>d <sub>no</sub> <sub>par</sub>t i<sub>n</sub> <sub>pro</sub>d<sub>uc</sub>i<sub>ng</sub> it<sub>.</sub>

N<sub>ow</sub> <sub>we</sub> <sub>p</sub>r<sub>ese</sub>nt <sub>ou</sub>r <sub>a</sub>l<sub>go</sub>rithm OE2D<sub>.</sub>PMS<sub>ide.</sub>

F<sub>ur</sub>th<sub>ermore,</sub> <sub>we</sub> <sub>can</sub> <sub>emp</sub>l<sub>oy</sub> Al<sub>gor</sub>ith<sub>m</sub> 2 f<sub>or</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>games,</sub> <sub>suc</sub>h th<sub>a</sub>t f<sub>or</sub> <sub>every</sub> $\gamma _ { \ell } > 1 6 k ^ { 2 } \beta ^ { 2 }$ , an<sup>d</sup> ever<sub>y</sub> $\hat { q } \in \Delta ( d )$ , Alg. 2 outputs p such that

$$
\mathcal { V } _ { \gamma _ { \ell } } \big ( \widehat { q } , p \big ) \leq O \left( k \beta \sqrt { \frac { 1 } { \gamma _ { \ell } } } \right) .\tag{Lemma 3.1}
$$

<sub>an</sub>d f<sub>or</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e games, we can emp</sub>l<sub>oy</sub> Al<sub>gor</sub>ith<sub>m</sub> 3<sub>, suc</sub>h th<sub>a</sub>t F<sub>or every</sub> $\gamma _ { \ell } > 0$ , an<sup>d</sup> ever<sub>y</sub> $\hat { q } \in \Delta ( d ) , \Lambda \dot { \mathrm { l g } } . 2$ <sub>run w</sub>ith $\begin{array} { r } { \bar { \kappa } = \frac { \gamma } { 8 k ^ { 2 } m ^ { 2 } } } \end{array}$ <sub>an</sub>d $\mu = k$ , outputs p such that

$$
\mathcal { V } _ { \gamma _ { \ell } } \left( \hat { q } , p \right) \leq O \left( \frac { m ^ { 2 } k ^ { 3 } } { \gamma _ { \ell } } \right) .\tag{Lemma 3.2}
$$

Algorithm 4 OE2D.PMSide   
1: parameters:   
Ofli<sub>ne es</sub>ti<sub>ma</sub>ti<sub>on orac</sub>l<sub>e</sub> $\mathrm { A l g _ { O f f } }$   
E<sub>xp</sub>l<sub>o</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r<sub>s</sub> $\{ \gamma _ { \ell } \} _ { \ell = 1 } ^ { L }$   
E<sub>rror ra</sub>t<sub>es</sub> $\{ \delta _ { \ell } \} _ { \ell = 1 } ^ { L }$   
E<sub>poc</sub>h <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e</sub> $0 = \tau _ { 0 } < \tau _ { 1 } < \cdots < \tau _ { L }$   
2: for $t = 1$ to $\tau _ { 1 }$ do   
3<sub>:</sub> Ob<sub>serve</sub> th<sub>e con</sub>t<sub>ex</sub>t $x _ { t }$   
4<sub>:</sub> Fi<sub>n</sub>d $p _ { t }$ th<sub>a</sub>t <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes,</sub>   
sup Coverage $( p , \lambda )$   
λ ∆ Π   
5<sub>:</sub> Ch<sub>oose ac</sub>ti<sub>on</sub> $A _ { t } \sim p _ { t }$ <sub>an</sub>d <sub>o</sub>b<sub>serve</sub> f<sub>ee</sub>db<sub>ac</sub>k $\sigma _ { t }$   
6: end for   
7: for $\ell = 2 , 3 , \cdots , L$ do   
8<sub>:</sub> Com<sub>p</sub>ute estimate $\hat { q } _ { \ell } = \mathop { \mathrm { A l g } _ { \mathrm { O f f } } } \left( \left\{ A _ { s } , x _ { s } , \sigma _ { s } \right\} _ { s = \tau _ { \ell - 2 } + 1 } ^ { \tau _ { \ell - 1 } } \right)$   
9: for $t = \tau _ { \ell - 1 } + 1$ to $\tau _ { \ell }$ do   
10<sub>:</sub> −Receive context x   
11<sub>:</sub> Fi<sub>n</sub>d $p _ { t }$ th<sub>a</sub>t <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub>   
$\mathcal { V } _ { \gamma _ { \ell } } ( \widehat { q } _ { \ell } ( x _ { t } ) , p ) : = \operatorname* { s u p } _ { \lambda \in \Delta ( \Pi ) } \underbrace { p ^ { \top } L \widehat { q } _ { \ell } ( x _ { t } ) - \lambda ^ { \top } L \widehat { q } _ { \ell } ( x _ { t } ) } _ { \mathrm { e x p l o i t } } + \frac { 1 } { \gamma _ { \ell } } \cdot \underbrace { \mathrm { C o v e r a g e } ( p , \lambda ) } _ { \mathrm { e x p l o r e } }$   
12<sub>:</sub> Ch<sub>oose ac</sub>ti<sub>on</sub> $A _ { t } \sim p _ { t }$ <sub>an</sub>d <sub>o</sub>b<sub>serve</sub> f<sub>ee</sub>db<sub>ac</sub>k $\sigma _ { t }$   
13: end for   
14: end for

So in both cases, we have algorithms to find policies that establishes an upper bound for our objective f<sub>unc</sub>ti<sub>on.</sub> L<sub>e</sub>t <sub>us</sub> d<sub>eno</sub>t<sub>e</sub> it b<sub>y</sub> $V _ { \ell } . \mathrm { \sf ~ S o }$

$$
V _ { \ell } = O \left( k \beta \sqrt { \frac { 1 } { \gamma _ { \ell } } } \right)
$$

$$
V _ { \ell } = O \left( \frac { m ^ { 2 } k ^ { 3 } } { \gamma _ { \ell } } \right)
$$

Furthermore, within a single epoch ℓ for all round $t \in \left[ \tau _ { \ell - 1 } { + } 1 , \tau _ { \ell } \right]$ <sub>,</sub> th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub> d<sub>oes</sub> <sub>no</sub>t <sub>c</sub>h<sub>ange</sub> $( \hat { q } _ { \ell } ( \cdot ) )$ ), <sub>on</sub>l<sub>y</sub> th<sub>e con</sub>t<sub>ex</sub>t $x _ { t }$ <sub>c</sub>h<sub>anges.</sub> S<sub>o, w</sub>ithi<sub>n an epoc</sub>h $\ell ,$

$$
x _ { t _ { 1 } } = x _ { t _ { 2 } } \implies \hat { q } _ { \ell } \big ( x _ { t _ { 1 } } \big ) = \hat { q } _ { \ell } \big ( x _ { t _ { 2 } } \big ) \implies p _ { t _ { 1 } } = p _ { t _ { 2 } } \quad \forall t _ { 1 } , t _ { 2 } \in \left[ \tau _ { \ell - 1 } + 1 , \tau _ { \ell } \right]
$$

S<sub>o, we can a</sub>l<sub>ways</sub> thi<sub>n</sub>k th<sub>a</sub>t <sub>w</sub>ithi<sub>n an epoc</sub>h<sub>, every</sub> $p _ { t }$ i<sub>s samp</sub>l<sub>e</sub>d f<sub>rom a</sub> fi<sub>xe</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $p _ { \ell } ( \cdot \vert x _ { t } )$ th<sub>a</sub>t i<sub>s, we can</sub> d<sub>eno</sub>t<sub>e,</sub>

$$
p _ { t }  p _ { \ell } ( \cdot \vert x _ { t } ) \quad \forall t \in \big [ \tau _ { \ell - 1 } + 1 , \tau _ { \ell } \big ] .
$$

## D<sub>.</sub>2 R<sub>egret</sub> <sub>analysis</sub>

N<sub>ex</sub>t<sub>,</sub> f<sub>or</sub> <sub>rea</sub>d<sub>a</sub>bilit<sub>y</sub> <sub>an</sub>d <sub>re</sub>d<sub>uce</sub> th<sub>e</sub> <sub>no</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>c</sub>l<sub>u</sub>tt<sub>er</sub> <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng,</sub>

Definition D.1. Define:

$\mathcal { L } ^ { \star } ( \lambda \mid x ) = \lambda ^ { T } L q ^ { \star } ( x )$ : The loss incurred b<sub>y</sub> polic<sub>y</sub> λ under true outcome distribution $q ^ { * }$ for a context x.

$\widehat { \mathcal { L } } _ { \ell } ( \lambda | x ) = \lambda ^ { T } L \widehat { q } _ { \ell } ( x )$ : The loss incurred b<sub>y</sub> polic<sub>y</sub> λ under estimated distribution $\hat { q } _ { \ell }$ f<sub>or</sub> <sub>a</sub> context x.

$\begin{array} { r } { \lambda ^ { * } = \arg \operatorname* { m i n } _ { \lambda \in \Delta ( \Pi ) } \mathcal { L } ^ { \star } \big ( \lambda \mid x \big ) \ ; } \end{array}$ : The true best polic<sub>y</sub> for context x.

$\begin{array} { r } { \hat { \lambda } _ { \ell } = \arg \operatorname* { m i n } _ { \lambda \in \Delta ( \Pi ) } \widehat { \mathcal { L } } _ { \ell } \big ( \lambda \big | x \big ) } \end{array}$ : The estimated best polic<sub>y</sub> for context x.

• R $\deg ( \lambda \mid x ) = { \mathcal { L } } ^ { \star } ( \lambda \mid x ) - { \mathcal { L } } ^ { \star } ( \lambda ^ { \star } \mid x )$ : The regret incurred b<sub>y</sub> polic<sub>y</sub> λ under true outcome di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q ^ { * }$ for a context x.

${ \widehat { \mathrm { R e g } } } _ { \ell } ( \lambda \mid x ) = { \widehat { \mathcal { L } } } _ { \ell } ( \lambda \mid x ) - { \widehat { \mathcal { L } } } _ { \ell } ( { \hat { \lambda } } \mid x )$ : The regret incurred b<sub>y</sub> polic<sub>y</sub> λ under estimated di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\hat { q } _ { \ell }$ for a context x.

$$
\begin{array} { r l } & { \bullet \ C ( p _ { \ell } , \lambda ) = \operatorname { C o v e r a g e } ( p _ { \ell } , \lambda , ) } \\ & { \bullet \ E _ { \ell } = \mathbb { E } _ { a \sim p _ { \ell - 1 } } \left[ D _ { \mathrm { H } } ^ { 2 } \left( S _ { a } \widehat { q } _ { \ell } ( x ) \vert S _ { a } { q } ^ { * } ( x ) \right) \right] } \end{array}
$$

I<sub>n</sub> <sub>a</sub>ll th<sub>e</sub> <sub>a</sub>b<sub>ove</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons,</sub> <sub>we</sub> <sub>m</sub>i<sub>g</sub>ht d<sub>rop</sub> $( \cdot | x )$ <sub>,</sub> <sub>w</sub>h<sub>en</sub> it i<sub>s</sub> <sub>ev</sub>id<sub>en</sub>t f<sub>rom</sub> th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>.</sub> F<sub>or</sub> <sub>examp</sub>l<sub>e,</sub> <sub>we m</sub>i<sub>g</sub>ht d<sub>eno</sub>t<sub>e</sub> $\mathcal { L } ^ { \star } ( \lambda \vert x )  \mathcal { L } ^ { \star } ( \mathbf { \bar { \lambda } } )$ .

N<sub>ex</sub>t <sub>we</sub> <sub>presen</sub>t t<sub>wo</sub> i<sub>mpor</sub>t<sub>an</sub>t <sub>proper</sub>ti<sub>es</sub> <sub>o</sub>f <sub>our</sub> <sub>po</sub>li<sub>c</sub>i<sub>es</sub> <sub>an</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>ors.</sub>

## Certification:

This property is a direct consequence of our algorithms finding the policies which bounds our objective f<sub>unc</sub>ti<sub>on</sub> b<sub>y</sub> $V _ { \ell }$

∀x ∈ X and $\forall \ell \geq 1$

$$
\mathcal { V } _ { \gamma _ { \ell } } \big ( \hat { q } _ { \ell } ( x ) , p _ { \ell } \big ) = \operatorname* { s u p } _ { \lambda \in \Delta ( \Pi ) } p _ { \ell } ^ { \intercal } L \hat { q } _ { \ell } ( x ) - \lambda ^ { \intercal } L \hat { q } _ { \ell } ( x ) + \frac { 1 } { \gamma _ { \ell } } \cdot \mathrm { C o v e r a g e } \big ( p _ { \ell } , \lambda \big ) \leq V _ { \ell }
$$

H<sub>ence,</sub>

∀x ∈ X , $\forall \ell \geq 1$ <sub>an</sub>d<sub>,</sub> $\forall \lambda \in \Pi$

$$
\begin{array} { r l } & { p _ { \ell } ^ { \top } \mathbf { L } \widehat { q } _ { \ell } ( x ) - \lambda ^ { \top } \mathbf { L } \widehat { q } _ { \ell } ( x ) + \displaystyle \frac { 1 } { \gamma _ { \ell } } \cdot \mathrm { C o v e r a g e } ( p _ { \ell } , \lambda ) \leq V _ { \ell } } \\ & { \implies \widehat { \mathcal { L } } _ { \ell } ( p _ { \ell } | x ) - \widehat { \mathcal { L } } _ { \ell } ( \lambda | x ) + \displaystyle \frac { 1 } { \gamma _ { \ell } } \cdot C ( p _ { \ell } , \lambda ) \leq V _ { \ell } } \end{array}
$$

Si<sub>nce,</sub> thi<sub>s</sub> i<sub>s</sub> t<sub>rue</sub> f<sub>or a</sub>ll $x \in { \mathcal { X } } .$ <sub>,</sub> <sub>we</sub> t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> lib<sub>er</sub>t<sub>y</sub> t<sub>o</sub> d<sub>rop</sub> it<sub>,</sub>

$$
\widehat { \mathcal { L } } _ { \ell } ( p _ { \ell } ) - \widehat { \mathcal { L } } _ { \ell } ( \lambda ) + \frac { 1 } { \gamma _ { \ell } } \cdot C ( p _ { \ell } , \lambda ) \leq V _ { \ell } .
$$

(certification)

## Of-policy evaluation (OPE):

Lemma D.1 (Of-policy evaluation lemma). With all the definitions in Definition $\mathrm { D . 1 } , \forall \lambda , \forall \ell \geq 2$ <sub>an</sub>d $\forall x$

$$
\left| \widehat { \mathcal { L } } _ { \ell } \left( \lambda - p _ { \ell - 1 } \middle | x \right) - \mathcal { L } ^ { \star } \left( \lambda - p _ { \ell - 1 } \middle | x \right) \right| \leq \frac { 1 } { 6 \gamma _ { \ell } } \cdot C ( p _ { \ell - 1 } , \lambda ) + \frac { 3 \gamma _ { \ell } } { 2 } E _ { \ell } .
$$

(Of-<sub>p</sub>olic<sub>y</sub> evaluation)

Proof.

$$
\begin{array} { r l } & { ( \widehat { C } _ { \xi } ( \lambda - \rho _ { \varepsilon - 1 }  x  ) - C ^ { * } ( \lambda - p _ { \varepsilon - 1 }  x  ) ) ^ { 2 } } \\ & { \quad = ( ( \lambda - p _ { \varepsilon + 1 } ) ^ { \mathrm { T } } L _ { \widehat { G } } ( \widehat { x } ) - ( \lambda - p _ { \varepsilon - 1 } ) ^ { \mathrm { T } } L _ { \widehat { G } } ( \widehat { x } ) ) ^ { 2 } } \\ & { \quad = \frac { ( ( \lambda - p _ { \varepsilon - 1 } ) ^ { \mathrm { T } } L _ { \widehat { G } } ( \widehat { y } _ { \varepsilon } ( x ) ) - q ^ { * } ( x ) ) ) ^ { 2 } } { \mathrm { E } _ { \lambda - p _ { \varepsilon - 1 } } [ D _ { \widehat { G } } ^ { 2 } ( S _ { \widehat { \sigma } } ( \widehat { x } )  S _ { \widehat { \sigma } } \epsilon ^ { * } ( x ) ) ] } \cdot \frac { \| \widehat { x } _ { 0 } - \widehat { y } _ { \varepsilon - 1 } ( \widehat { x } ) \| ^ { 2 } } { \mathrm { E } _ { \lambda - p _ { \varepsilon - 1 } } [ D _ { \widehat { G } } ^ { 2 } ( S _ { \widehat { \sigma } } \widehat { y } _ { \varepsilon } ( x ) ) ] S _ { \widehat { \sigma } } \epsilon ^ { * } ( x ) ) ] } } \\ & { \quad \leq C \mathrm { O e r c a t } g ( \rho _ { \varepsilon - 1 } , \lambda ) \cdot \mathbf { E } _ { \mathbf { g } - p _ { \varepsilon - 1 } } [ D _ { \widehat { G } } ^ { 2 } ( S _ { \widehat { \sigma } } \widehat { y } _ { \varepsilon } ( x ) )  S _ { \widehat { \sigma } } \widehat { u }   } \\ & {   \widehat { C } _ { \widehat { G } } ( \lambda - p _ { \varepsilon - 1 }  x  ) - \widehat { C } ^ { * } ( \lambda - p _ { \varepsilon - 1 }  x  ) ] } \\ &  \quad \leq \end{array}\tag{AM-GM}
$$

$$
= \frac { 1 } { 6 \gamma _ { \ell } } \cdot C ( p _ { \ell - 1 } , \lambda ) + \frac { 3 \gamma _ { \ell } } { 2 } E _ { \ell } .
$$

Th<sub>e secon</sub>d t<sub>erm</sub> i<sub>s</sub> th<sub>e genera</sub>li<sub>za</sub>ti<sub>on error o</sub>f th<sub>e o</sub>fli<sub>ne orac</sub>l<sub>e:</sub> $\hat { q } _ { \ell }$ i<sub>s es</sub>ti<sub>ma</sub>t<sub>e</sub>d f<sub>rom</sub> d<sub>a</sub>t<sub>a co</sub>ll<sub>ec</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>prev</sub>i<sub>ous</sub> <sub>epoc</sub>h <sub>un</sub>d<sub>er</sub> $p _ { \ell - 1 }$ , so Assum<sub>p</sub>tion B.1 (Hellin<sub>g</sub>er boundedness) a<sub>pp</sub>lies to it directl<sub>y</sub>. −This coupling between the estimate and the policy that produced it is what makes the ofline analysis <sub>more</sub> d<sub>e</sub>li<sub>ca</sub>t<sub>e</sub> th<sub>an</sub> th<sub>e on</sub>li<sub>ne one, w</sub>h<sub>ere</sub> th<sub>e es</sub>ti<sub>ma</sub>t<sub>or</sub> i<sub>s up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>as</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a arr</sub>i<sub>ve.</sub>

Th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm</sub> i<sub>nvo</sub>l<sub>ves no es</sub>ti<sub>ma</sub>t<sub>e a</sub>t <sub>a</sub>ll<sub>:</sub> $\mathrm { C o v e r a g e } ( p _ { \ell - 1 } , \lambda )$ depends onl<sub>y</sub> on the model class, on λ <sub>an</sub>d <sub>on</sub> $p _ { \ell - 1 } . ~ \mathsf { A }$ − policy that makes it small therefore controls the evaluation error for every $\hat { q } \in \mathrm { c o } ( \mathcal { Q } _ { x } )$ −at once, in particular for whichever estimate the oracle returns in epoch ℓ.

N<sub>ow</sub> <sub>we</sub> <sub>presen</sub>t <sub>an</sub> <sub>a</sub>b<sub>s</sub>t<sub>rac</sub>t <sub>regre</sub>t th<sub>eorem</sub> <sub>a</sub>b<sub>ou</sub>t <sub>epoc</sub>h<sub>-</sub>b<sub>ase</sub>d <sub>o</sub>fli<sub>ne</sub> <sub>orac</sub>l<sub>e-e</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>n</sub>t<sub>erac</sub>ti<sub>ve</sub> learnin<sub>g</sub> al<sub>g</sub>orithm from Qin & Zhan<sub>g</sub> [2026a].

Theorem D.1 (Generalization of regret concentration theorem from Qin & Zhang [2026a]). Let Λ be <sub>a</sub> <sub>se</sub>t <sub>o</sub>f <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> $\mathcal { L } ( \cdot | \boldsymbol { x } ) : \Lambda \to [ 0 , 1 ]$ <sub>a va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on an</sub>d $\mathcal { L } ^ { \star } ( \cdot \vert x )$ <sub>an</sub>d $\widehat { \mathcal { L } } _ { \ell } ( \cdot \vert x ) , \ell \in [ L ]$ <sub>are va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>ons measure</sub>d <sub>un</sub>d<sub>er</sub> th<sub>e</sub> t<sub>rue mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s respec</sub>ti<sub>ve</sub>l<sub>y.</sub> $\lambda ^ { \star } \in \Lambda$ <sub>an</sub>d $\hat { \lambda } _ { \ell } \in \Lambda$ are <sub>respec</sub>ti<sub>ve</sub>l<sub>y</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>po</sub>li<sub>c</sub>i<sub>es un</sub>d<sub>er respec</sub>ti<sub>ve</sub> t<sub>rue an</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> L<sub>e</sub>t $C _ { \ell } : \Lambda \times \Lambda \to \mathbb { R }$ b<sub>e</sub> <sup>co</sup>v<sup>era</sup>g<sup>e ma</sup>p<sup>s</sup>, $p _ { \ell } \in \Lambda$ <sub>are</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons,</sub> $0 < \gamma _ { 1 } \leq \gamma _ { 2 } \leq \cdots \leq \gamma _ { L }$ l<sub>earn</sub>i<sub>ng ra</sub>t<sub>es, an</sub>d $V _ { \ell } , E _ { \ell } \ge 0$ +numbers <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub>

(C1) (certificate) for ever<sub>y</sub> $\ell \geq 1$ an<sup>d</sup> ever<sub>y</sub> $\lambda \in \Lambda$

$$
\widehat { \mathscr { L } } _ { \ell } ( p _ { \ell } \vert x ) - \widehat { \mathscr { L } } _ { \ell } ( \lambda \vert x ) + \frac { 1 } { \gamma _ { \ell } } C _ { \ell } ( p _ { \ell } , \lambda ) \leq V _ { \ell } .
$$

(C2) (of-<sub>p</sub>olic<sub>y</sub> evaluation) for ever<sub>y</sub> $\ell \geq 2$ an<sup>d</sup> ever<sub>y</sub> $\lambda \in \Lambda$

$$
\left| \widehat { \mathcal { L } } _ { \ell } ( \lambda - p _ { \ell - 1 } \mid x ) - \mathcal { L } ^ { \star } ( \lambda - p _ { \ell - 1 } \mid x ) \right| \leq \frac { 1 } { 6 \gamma _ { \ell } } C _ { \ell - 1 } ( p _ { \ell - 1 } , \lambda ) + \frac { 3 \gamma _ { \ell } } { 2 } E _ { \ell } .
$$

D<sub>e</sub>fi<sub>ne</sub> $G _ { 1 } : = 1$ <sub>an</sub>d<sub>,</sub> f<sub>or</sub> $\ell \geq 2 .$

$$
\gamma _ { \ell } G _ { \ell } : = \gamma _ { \ell - 1 } G _ { \ell - 1 } + \gamma _ { \ell - 1 } V _ { \ell - 1 } + 9 \gamma _ { \ell } ^ { 2 } E _ { \ell } , \qquad \mathrm { i . e . } \qquad G _ { \ell } = \frac { 1 } { \gamma _ { \ell } } \left[ \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } E _ { i } ) \right] .
$$

Th<sub>en,</sub> f<sub>or</sub> <sub>every</sub> $\ell \in [ L ]$ an<sup>d</sup> ever<sub>y</sub> $\lambda \in \Lambda$

$$
\operatorname { R e g } ( \lambda \mid x ) \le 2 \widehat { \operatorname { R e g } } _ { \ell } ( \lambda \mid x ) + G _ { \ell } \qquad \mathrm { a n d } \qquad \widehat { \operatorname { R e g } } _ { \ell } ( \lambda \mid x ) \le 2 \operatorname { R e g } ( \lambda \mid x ) + G _ { \ell } .
$$

Consequently Reg $\left( p _ { \ell } \mid x \right) \leq 2 V _ { \ell } + G _ { \ell }$ for ever<sub>y</sub> ℓ, and

$$
\sum _ { \ell = 1 } ^ { L } ( \tau _ { \ell } - \tau _ { \ell - 1 } ) \operatorname { R e g } ( p _ { \ell } \mid x ) \leq \tau _ { 1 } + \sum _ { \ell = 2 } ^ { L } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left[ 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } E _ { i } ) \right] .
$$

P<sub>roo</sub>f i<sub>s</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> A<sub>ppen</sub>di<sub>x</sub> G<sub>.</sub> W<sub>e</sub> <sub>can</sub> <sub>app</sub>l<sub>y</sub> thi<sub>s</sub> th<sub>eorem</sub> di<sub>rec</sub>tl<sub>y</sub> <sub>an</sub>d <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> <sub>a</sub> <sub>per-con</sub>t<sub>ex</sub>t <sub>regre</sub>t b<sub>oun</sub>d<sub>.</sub> N<sub>ow we s</sub>t<sub>a</sub>t<sub>e</sub> th<sub>e ma</sub>i<sub>n</sub> th<sub>eorem</sub> th<sub>a</sub>t b<sub>oun</sub>d<sub>s</sub> th<sub>e regre</sub>t <sub>w</sub>ith hi<sub>g</sub>h <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> f<sub>or any epoc</sub>h <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e.</sub>

Theorem D.2 (Regret of OE2D.PMSide). Suppose $\mathrm { A l g _ { O f f } }$ h<sub>as a</sub> H<sub>e</sub>lli<sub>nger regre</sub>t b<sub>oun</sub>d $\overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right)$ <sub>.</sub> F<sub>or</sub> <sub>any</sub> <sub>non-</sub>d<sub>egenera</sub>t<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng</sub> <sub>game</sub> <sub>w</sub>ith<sub>ou</sub>t d<sub>up</sub>li<sub>ca</sub>t<sub>e ac</sub>ti<sub>ons, assume</sub> th<sub>a</sub>t ${ x _ { t } } \stackrel { i . i . d } { \sim } { \mathcal { D } } _ { x }$ <sub>,</sub> th<sub>en</sub> f<sub>or</sub> <sub>any</sub> $\delta \in ( 0 , 1 )$ <sub>,</sub> OE2D<sub>.</sub>PMS<sub>ide</sub> <sub>run</sub> <sub>w</sub>ith

1. ex<sub>p</sub>loration rule Al<sub>g</sub>orithm 2 for <sub>g</sub>loball<sub>y</sub> observable <sub>g</sub>ame (with $\gamma _ { \ell } > 1 6 k ^ { 2 } \beta ^ { 2 } \forall \ell \in [ L ] )$

2. ex<sub>p</sub>loration rule Al<sub>g</sub>orithm 3 for locall<sub>y</sub> observable <sub>g</sub>ame (with $\begin{array} { r } { \kappa = \frac { \gamma _ { \ell } } { 8 k ^ { 2 } m ^ { 2 } } \forall \ell \in [ L ] } \end{array}$ <sub>an</sub>d $\mu = k )$

3<sub>.</sub> <sub>exp</sub>l<sub>ora</sub>ti<sub>on</sub> <sub>parame</sub>t<sub>ers</sub> <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> $0 < \gamma _ { 1 } \leq \gamma _ { 2 } \leq \cdots \leq \gamma _ { L }$

<sub>sa</sub>ti<sub>s</sub>fi<sub>es,</sub>

$$
\widetilde { \mathrm { R e g } } _ { T } \leq L \cdot \tau _ { 1 } + 2 L \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left( 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } \left( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } ( \mathcal { Q } , \tau _ { i } - \tau _ { i - 1 } , \delta _ { i } ) \right) \right) .
$$

<sub>w</sub>h<sub>ere,</sub>

$$
\bullet \widetilde { \mathrm { R e g } } _ { T } = \mathbb { E } _ { \boldsymbol { x } _ { 1 } , \ldots , \boldsymbol { x } _ { T } ^ { i . i . d } \mathcal { D } _ { \boldsymbol { x } } } \left[ \mathrm { R e g } _ { T } \right]
$$

$\begin{array} { r } { V _ { \ell } = O \bigg ( k \beta \sqrt { \frac { 1 } { \gamma _ { \ell } } } \bigg ) } \end{array}$ f<sub>or</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>games</sub>

$\textstyle V _ { \ell } = O \left( { \frac { m ^ { 2 } k ^ { 3 } } { \gamma _ { \ell } } } \right)$ f<sub>or</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e games</sub>

Proof. From Theorem D.1, the regret per-context is bounded by,

$$
\begin{array} { r l } & { \displaystyle \sum _ { \ell = 1 } ^ { L } ( \tau _ { \ell } - \tau _ { \ell - 1 } ) \operatorname { R e g } ( p _ { \ell } \mid x ) \leq \tau _ { 1 } + \sum _ { \ell = 2 } ^ { L } ( \tau _ { \ell } - \tau _ { \ell - 1 } ) ( 2 V _ { \ell } + G _ { \ell } ) \medskip } \\ & { \quad \quad \quad = \tau _ { 1 } + \displaystyle \sum _ { \ell = 2 } ^ { L } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left[ 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \displaystyle \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } E _ { i } ) \right] . } \end{array}
$$

L<sub>e</sub>t <sub>us</sub> d<sub>eno</sub>t<sub>e,</sub>

$$
\widetilde { \mathrm { R e g } } _ { T } = \mathbb { E } _ { \boldsymbol { x } ^ { i . } \sim \mathcal { D } _ { \boldsymbol { x } } } \left[ \mathrm { R e g } _ { T } \right]
$$

H<sub>ere, no</sub>t<sub>e</sub> th<sub>a</sub>t<sub>, even</sub> th<sub>oug</sub>h th<sub>e ran</sub>d<sub>omness w.r.</sub>t t<sub>o con</sub>t<sub>ex</sub>t<sub>s are remove</sub>d<sub>,</sub> $\widetilde { \mathrm { R e g } } _ { T }$ <sub>,</sub> i<sub>s s</sub>till <sub>a ran</sub>d<sub>om</sub> <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> i<sub>n</sub> t<sub>erms o</sub>f <sub>po</sub>li<sub>cy.</sub> H<sub>ence,</sub>

$$
\begin{array} { r l } & { \widetilde { \mathrm { R e g } } _ { T } = \mathbb { E } _ { x ^ { i \cdot \underline { { i } } \cdot d } \mathcal { D } _ { x } } \left[ \mathrm { R e g } _ { T } \right] = \mathbb { E } \left[ \displaystyle \sum _ { \ell = 1 } ^ { L } \displaystyle \sum _ { t = \tau _ { \ell - 1 } } ^ { \tau _ { \ell } } \mathrm { R e g } ( p _ { \ell } \mid x _ { t } ) \right] } \\ & { \qquad \leq \mathbb { E } \left[ \displaystyle \sum _ { \ell = 1 } ^ { L } \tau _ { 1 } + \displaystyle \sum _ { \ell = 2 } ^ { L } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left( 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \displaystyle \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } E _ { i } ) \right) \right] } \\ & { \qquad \leq L \cdot \tau _ { 1 } + 2 L \displaystyle \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left( 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \displaystyle \sum _ { i = 2 } ^ { \ell } \left( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } \mathbb { E } _ { x ^ { i \cdot \underline { { i } } \cdot d } \mathcal { D } _ { x } } \left[ E _ { i } \right] \right) \right) . } \end{array}
$$

If th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>s</sub> <sub>are</sub> i<sub>.</sub>i<sub>.</sub>d <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> ${ x _ { t } } \stackrel { i . i . d } { \sim } { \mathcal { D } } _ { x }$ <sub>,</sub> th<sub>en,</sub>

$$
\begin{array} { r l } & { \widetilde { \mathrm { R e g } } _ { T } \le L \cdot \tau _ { 1 } } \\ & { \quad + 2 L \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } ( 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \displaystyle \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } \mathbb { E } _ { x ^ { i , i , a } } \mathcal { I } _ { D _ { x } } \mathbb { E } _ { a \sim p _ { \ell - 1 } } [ D _ { H } ^ { 2 } ( S _ { a } \hat { q } _ { \ell } ( x ) \mid S _ { a } q ^ { * } ( x ) ) ] ) ) } \\ & { \quad \le L \cdot \tau _ { 1 } + 2 L \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } ( 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \displaystyle \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } D _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } \frac { \mathrm { R e g } _ { 0 } ^ { \mathrm { H } } \mathrm { ( } Q , \tau _ { i } - \tau _ { i - 1 } ) } { \tau _ { i } - \tau _ { i - 1 } } ) ) . } \end{array}
$$

W<sub>e</sub> k<sub>now</sub> th<sub>a</sub>t<sub>, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>a</sub>t l<sub>eas</sub>t $1 - \delta ^ { \prime } .$

$$
\mathrm { R e g } _ { \mathrm { O f f } } ^ { \mathrm { H } } ( \mathcal { Q } , \tau _ { i } - \tau _ { i - 1 } ) \le \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } ( \mathcal { Q } , \tau _ { i } - \tau _ { i - 1 } , \delta )
$$

S<sub>e</sub>t $\begin{array} { r } { \delta _ { \ell } = \frac { \delta } { L } } \end{array}$ <sub>,</sub> <sub>w</sub>ith <sub>a</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta ,$

$$
\widetilde { \mathrm { R e g } } _ { T } \leq L \cdot \tau _ { 1 } + 2 L \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left( 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } \left( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } \frac { \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } ( Q , \tau _ { i } - \tau _ { i - 1 } , \delta _ { i } ) } { \tau _ { i } - \tau _ { i - 1 } } \right) \right) .
$$

Th<sub>e</sub> fi<sub>na</sub>l <sub>regre</sub>t d<sub>epen</sub>d<sub>s on epoc</sub>h <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e.</sub> W<sub>e c</sub>h<sub>oose a</sub> d<sub>ou</sub>bli<sub>ng epoc</sub>h <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e.</sub>

Corollary 3. Suppose $\mathrm { A l g _ { O f f } }$ h<sub>as a</sub> H<sub>e</sub>lli<sub>nger regre</sub>t b<sub>oun</sub>d $\overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , T , \delta \right)$ . <sup>F</sup>or an<sub>y</sub> non  
d<sub>egenera</sub>t<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e par</sub>ti<sub>a</sub>l <sub>mon</sub>it<sub>or</sub>i<sub>ng game w</sub>ith<sub>ou</sub>t d<sub>up</sub>li<sub>ca</sub>t<sub>e ac</sub>ti<sub>ons, assume</sub> th<sub>a</sub>t   
$x _ { 1 } , \dotsc , x _ { T } \stackrel { i . i . d } { \sim } { \cal D } _ { x } ,$ thenfor any $\delta \in ( 0 , 1 )$ <sub>,</sub> th<sub>e a</sub>l <sub>or</sub>ith<sub>m</sub> OE2D<sub>.</sub>PMS<sub>ide run w</sub>ith<sub>,</sub>   
1. exploration rule Algorithm 2for globally observable game (with $\gamma _ { \ell } > 1 6 k ^ { 2 } \beta ^ { 2 } \forall \ell \in [ L ] )$   
2. exploration rule Algorithm 3for locally observable game (with $\begin{array} { r } { \kappa = { \frac { \gamma _ { \ell } } { 8 k ^ { 2 } m ^ { 2 } } } ~ \forall \ell \in [ L ] } \end{array}$ <sub>an</sub>d   
$\mu = k )$   
3<sub>.</sub> d<sub>ou</sub>bli<sub>ng sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e, suc</sub>h th<sub>a</sub>t $\tau _ { \ell } = 2 ^ { \ell }$   
4<sub>.</sub> <sub>exp</sub>l<sub>ora</sub>ti<sub>on</sub> <sub>parame</sub>t<sub>ers</sub> <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> $0 < \gamma _ { 1 } \leq \gamma _ { 2 } \leq \cdots \leq \gamma _ { L }$   
5<sub>. w</sub>ith <sub>assump</sub>ti<sub>on</sub> $\overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } ( \mathcal { Q } , t , \delta )$ does not depend on t. Generally this is the case.   
6<sub>.</sub> <sub>w</sub>ith $\begin{array} { r } { \delta _ { \ell } = \frac { \delta } { L } } \end{array}$   
satisfies thefollowing guarantees,   
1<sub>.</sub> F<sub>or g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e games, w</sub>ith $\gamma _ { \ell } = \left( \frac { C \cdot \tau _ { \ell } \cdot k \beta } { \overrightarrow { \mathrm { R e g } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , \tau _ { \ell } / 2 , \delta / L \right) } \right) ^ { \frac { 2 } { 3 } }$ for some constant C,   
<sup>Q</sup>Reg ̃<sub>T</sub> ≤ O (T <sup>2</sup>3 log<sup>2</sup>(T)(kβ) <sup>2</sup>3 (Reg<sup>H</sup><sub>Of</sub>(Q, T, δ/ log(T))) <sup>1</sup>3 )   
2<sub>.</sub> F<sub>or</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e games, w</sub>ith $\begin{array} { r } { \gamma _ { \ell } = \sqrt { \frac { D \cdot \tau _ { \ell } \cdot k ^ { 3 } m ^ { 2 } } { \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , \tau _ { \ell } / 2 , \delta _ { \ell } \right) } } f o r } \end{array}$ some constant $D ,$   
Reg ̃T = O (log2(T)<sup>√</sup>T ⋅ k3m2Reg<sup>H</sup>Of(Q, T/2, δ/ log(T)))   
<sub>w</sub>h<sub>ere,</sub> $\begin{array} { r } { \widetilde { \mathrm { R e g } } _ { T } = \mathbb { E } _ { \boldsymbol { x } ^ { i . } \sim \mathcal { D } _ { \boldsymbol { x } } } \left[ \mathrm { R e g } _ { T } \right] . } \end{array}$

Proof. First of all, for doubling epoch schedule,

$$
\begin{array} { c } { { \displaystyle \tau _ { L } = { \cal O } ( T ) } } \\ { { \displaystyle \tau _ { \ell } - \tau _ { \ell - 1 } = \frac { \tau _ { \ell } } { 2 } } } \\ { { \displaystyle L = { \cal O } ( \log T ) } } \end{array}
$$

$$
\widetilde { \mathrm { R e g } } _ { T } \leq L \cdot \tau _ { 1 } + 2 L \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left( 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } \left( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } \frac { \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } ( Q , \tau _ { i } - \tau _ { i - 1 } , \delta _ { i } ) } { \tau _ { i } - \tau _ { i - 1 } } \right) \right) .
$$

$$
= L \cdot \tau _ { 1 } + 2 L \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } } { 2 \gamma _ { \ell } } \left( 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } \left( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } \frac { \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , \tau _ { i } / 2 , \delta _ { i } \right) } { \tau _ { i } / 2 } \right) \right) .
$$

$$
\leq L \cdot \tau _ { 1 } + 2 L \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } } { 2 \gamma _ { \ell } } \left( \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } \gamma _ { i } \left( 2 V _ { i } + 1 8 \gamma _ { i } \frac { \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , \tau _ { i } / 2 , \delta _ { i } \right) } { \tau _ { i } } \right) \right) .
$$

$$
\leq L \cdot \tau _ { 1 } + 2 L ^ { 2 } \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } } { 2 \gamma _ { \ell } } \left( \gamma _ { 1 } + \operatorname* { m a x } _ { i \in [ \ell ] } \gamma _ { i } \left( 2 V _ { i } + 1 8 \gamma _ { i } \frac { \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , \tau _ { i } / 2 , \delta _ { i } \right) } { \tau _ { i } } \right) \right) .
$$

$$
= L \cdot \tau _ { 1 } + 2 L ^ { 2 } \operatorname* { m a x } _ { \ell \in [ L ] } \frac { \tau _ { \ell } } { 2 \gamma _ { \ell } } \left( \gamma _ { 1 } + \operatorname* { m a x } _ { i \in [ \ell ] } \frac { \gamma _ { i } } { \tau _ { i } } \left( 2 \tau _ { i } V _ { i } + 1 8 \gamma _ { i } \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } ( \mathcal { Q } , \tau _ { i } / 2 , \delta _ { i } ) \right) \right) .
$$

F<sub>or</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>games,</sub>

$$
\begin{array} { r l } & { \quad _ { 1 } , \quad _ { 2 } , \quad _ { 3 } , \quad _ { 4 } , \quad _ { 5 } , \quad _ { 6 } , \quad _ { 7 } , } \\ &  \quad _ { 4 } , \quad _ { 5 } , \quad _ { 7 } , \quad _ { 6 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } \quad _ { 7 } , \quad _ { 7 } , \quad _ { 7 } , \quad _  7  \end{array}
$$

A<sub>n</sub>d f<sub>or</sub> l<sub>oca</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>games,</sub>

$$
\begin{array} { c } { { { \displaystyle V _ { \xi } = D , { \frac { \partial \xi ^ { 2 } \delta ^ { 3 } } { \partial t } } } } } \\    \displaystyle \begin{array} { c }  { \begin{array} { c } { { \begin{array} { r l } { { V _ { \xi } = D , \cdots \neq D , \cdots \neq D , \cdots \neq D , \delta ^ { 2 } } \\ { { \sqrt { 6 } } } \end{array} } \left( \Im \cdot \mathrm { ~ n u m s i n e ~ f l u e s ~ i n } ^ { 3 } \right) } , } } \\   \mathrm { \begin{array} { c } { { \begin{array} { r l } { { \mathrm { ~ } } } \end{array} } \\ { { \begin{array} { c } { { \mathrm { ~ } } } \\ { { \mathrm { ~ } } \\ { { \mathrm { ~ } } } \\ { \mathrm { { ~ } } \\ { \mathrm { { ~ } } \mathrm { \sqrt { 6 4 } } \mathrm { } y _ { \xi } \leq \delta \mathrm { ~ n u m s i n e ~ f l u e s ~ i n } ^ { 5 } \mathrm { ~ f l u e s ~ i n } ^ { 6 } \mathrm { ~ } \left( \Im \cdot \mathrm { ~ n u m s i n e ~ } ^ { \frac { \pi } { 2 } } \right) ^ { 2 } } \mathrm { ~ } } } } \end{array} } } } \\   \mathrm  \begin{array} { c } { { \mathrm { ~ } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } } } \end{array}  \begin{ c { { ~ } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } \mathrm { { ~ } } \mathrm { { ~ } } \mathrm { { ~ } } \mathrm { ~ } } } \\  { \mathrm { ~ } \mathrm { { ~ } } \mathrm { { ~ } } \mathrm { { ~ } } \mathrm { { ~ } } } \\ { \mathrm { { ~ } \mathrm { ~ } } \mathrm { { ~ } } \mathrm { { ~ } } } \\  { \mathrm { ~ } \mathrm { { ~ } } \mathrm { { ~ } } \mathrm { { ~ } } \mathrm { { ~ } } \mathrm { { ~ } } } \\   \mathrm { ~ } \mathrm { { ~ } } \mathrm  \end{array} \end{array} \end{array} \end{array}
$$

$$
\begin{array} { r l } & { = O \left( L ^ { 2 } \sqrt { T \cdot k ^ { 3 } m ^ { 2 } \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , T / 2 , \delta / L \right) } \right) } \\ & { = O \left( \log ^ { 2 } ( T ) \sqrt { T \cdot k ^ { 3 } m ^ { 2 } \overline { { \mathrm { R e g } } } _ { \mathrm { O f f } } ^ { \mathrm { H } } \left( \mathcal { Q } , T / 2 , \delta / \log ( T ) \right) } \right) } \end{array}
$$

## E E<sub>mpirical studies</sub>

Additional LLM-routing details. The cached BoolQ dataset contains 12,697 prompts, of which 12,648 <sup>h</sup>ave parsea<sup>bl</sup>e responses <sup>f</sup>rom <sup>b</sup>ot<sup>h</sup> Qwen2.5-0.5B-Instruct, use<sup>d</sup> as t<sup>h</sup>e wea<sup>k</sup> mo<sup>d</sup>e<sup>l</sup>, an<sup>d</sup> Llama-2-13B-Chat, use<sup>d</sup> as t<sup>h</sup>e strong re<sup>f</sup>erence mo<sup>d</sup>e<sup>l</sup>. <sup>Th</sup>e <sup>bi</sup>nary rout<sup>i</sup>ng outcome <sup>i</sup>n<sup>di</sup>cates whether their cached responses disa<sub>g</sub>ree; BoolQ <sub>g</sub>old labels are not used as routin<sub>g</sub> outcomes. Each prompt is represented by a 138-dimensional context consisting of 10 weak-model uncertainty and probability statistics, 64 whitened principal components of the weak model’s hidden states, and 64 whitened principal components of a context-free prompt embedding. All 12,648 eligible examples <sub>are</sub> <sub>presen</sub>t<sub>e</sub>d <sub>on</sub>li<sub>ne,</sub> <sub>an</sub>d <sub>ou</sub>t<sub>comes</sub> <sub>are</sub> <sub>revea</sub>l<sub>e</sub>d <sub>on</sub>l<sub>y</sub> <sub>a</sub>ft<sub>er</sub> <sub>query</sub>i<sub>ng</sub> th<sub>e</sub> <sub>s</sub>t<sub>rong</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> E<sub>ac</sub>h <sub>po</sub>li<sub>cy</sub> makes a routing decision at every round, whereas its estimator is updated only once in 32 rounds, usin all feedback revealed before that round. CBPSide [Bartok et al., 2012a; An et al., 2026] uses a´ li<sub>near</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> di<sub>sagreemen</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d th<sub>e con</sub>fid<sub>ence ra</sub>di<sub>us</sub>

$$
\beta _ { t } ( x _ { t } ) = \operatorname* { m i n } \biggl \{ 0 . 5 s \sqrt { x _ { t } ^ { \top } V _ { t - 1 } ^ { - 1 } x _ { t } } , 0 . 5 \biggr \} ,
$$

where s is the tuned multiplier. The two SquareCB.PMSide variants use $\mu = 2 , \kappa = s \sqrt { T }$ <sub>, an</sub>d i<sub>nverse-</sub> propensity weights capped at 10. Their only intended diference is the disagreement-probability <sub>es</sub>ti<sub>ma</sub>t<sub>or.</sub> Th<sub>e</sub> li<sub>near var</sub>i<sub>an</sub>t fit<sub>s an</sub> $L _ { 2 }$ <sub>-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d l<sub>og</sub>i<sub>s</sub>ti<sub>c mo</sub>d<sub>e</sub>l <sub>w</sub>ith $C = 1$ , t<sup>h</sup>e lbfgs so<sup>l</sup>ver, an<sup>d</sup> <sub>a</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>di<sub>za</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ep.</sub> Th<sub>e</sub> <sub>non</sub>li<sub>near</sub> <sub>var</sub>i<sub>an</sub>t <sub>uses</sub> <sub>sc</sub>ikit<sub>-</sub>l<sub>earn</sub>’<sub>s</sub> hi<sub>s</sub>t<sub>ogram</sub> <sub>gra</sub>di<sub>en</sub>t<sub>-</sub>b<sub>oos</sub>ti<sub>ng</sub> (HGB) classifier with binary log loss, 50 boosting iterations, learning rate 0.05, at most 15 leaves per tree, a minimum of 20 samples per leaf, $L _ { 2 }$ regularization 1, and no early stopping. At each eligible <sub>up</sub>d<sub>a</sub>t<sub>e,</sub> b<sub>o</sub>th <sub>es</sub>ti<sub>ma</sub>t<sub>ors</sub> <sub>are</sub> <sub>re</sub>fitt<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>accumu</sub>l<sub>a</sub>t<sub>e</sub>d i<sub>nverse-propens</sub>it<sub>y-we</sub>i<sub>g</sub>ht<sub>e</sub>d feedback. PG-TS [Grant & Leslie, 2021] uses a Ba<sub>y</sub>esian lo<sub>g</sub>istic disa<sub>g</sub>reement model with

$$
\theta \sim \mathcal { N } \big ( 0 , ( s \sigma _ { 0 } ) ^ { 2 } I \big ) , \qquad \sigma _ { 0 } = 1 ,
$$

and performs 15 Polya–Gamma Gibbs transitions whenever its posterior is updated.´

Both ETC baselines [Garivier et al., 2016] use an initial ex<sub>p</sub>loration bud<sub>g</sub>et of $s \cdot T ^ { 2 / 3 }$ <sub>roun</sub>d<sub>s</sub> t<sub>o</sub> fit <sub>a</sub> linear-lo<sub>g</sub>istic model and a histo<sub>g</sub>ram <sub>g</sub>radient boostin<sub>g</sub> (HGB) model res<sub>p</sub>ectivel<sub>y</sub>. The fitted models <sub>are</sub> th<sub>en</sub> f<sub>rozen,</sub> <sub>an</sub>d th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>quer</sub>i<sub>es</sub> <sub>are</sub> <sub>rou</sub>t<sub>e</sub>d <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>em.</sub> Th<sub>e</sub> <sub>ran</sub>d<sub>om</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>rou</sub>t<sub>es eac</sub>h <sub>query</sub> t<sub>o</sub> th<sub>e s</sub>t<sub>rong mo</sub>d<sub>e</sub>l <sub>w</sub>ith <sub>a</sub> fi<sub>xe</sub>d <sub>pro</sub>b<sub>a</sub>bilit<sub>y,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f <sub>con</sub>t<sub>ex</sub>t<sub>, swep</sub>t <sub>over</sub> {0, 0.2, 0.4, 0.6, 0.8, 1}.

Table 2 shows how s is used to tune diferent h<sub>y</sub>per-parameters for diferent algorithms. For each <sub>po</sub>li<sub>cy</sub> <sub>an</sub>d <sub>eac</sub>h $\ell _ { 0 1 }$ , we select s ∈ {0.1, 0.3, 1, 3, 10} by minimizing mean total cost over 20 paired <sub>ran</sub>d<sub>om</sub> <sub>permu</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>o</sub>f th<sub>e</sub> <sub>on</sub>li<sub>ne</sub> <sub>arr</sub>i<sub>va</sub>l <sub>or</sub>d<sub>er.</sub> R<sub>an</sub>d<sub>om</sub> i<sub>s</sub> <sub>compu</sub>t<sub>e</sub>d <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>ll<sub>y</sub> b<sub>y</sub> <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> th<sub>e</sub> <sub>rou</sub>ti<sub>ng ra</sub>t<sub>e o</sub>f th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>non</sub>li<sub>near</sub> S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide po</sub>li<sub>cy.</sub> Fi<sub>gure</sub> 2<sub>a repor</sub>t<sub>s mean rea</sub>li<sub>ze</sub>d total cost with pointwise 95% Student-t confidence intervals. The learning curves in Figure 4 report <sub>cumu</sub>l<sub>a</sub>ti<sub>ve excess cos</sub>t <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o a</sub> fi<sub>xe</sub>d fi<sub>ve-</sub>f<sub>o</sub>ld <sub>cross-</sub>fitt<sub>e</sub>d HGB <sub>re</sub>f<sub>erence po</sub>li<sub>cy.</sub> Th<sub>ese ex</sub>t<sub>en</sub>d th<sub>e</sub> <sub>resu</sub>lt<sub>s</sub> di<sub>scusse</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4<sub>.</sub> E<sub>ac</sub>h <sub>re</sub>f<sub>erence pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s pro</sub>d<sub>uce</sub>d b<sub>y an</sub> HGB <sub>mo</sub>d<sub>e</sub>l th<sub>a</sub>t <sub>exc</sub>l<sub>u</sub>d<sub>es</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>examp</sub>l<sub>e an</sub>d i<sub>s use</sub>d <sub>on</sub>l<sub>y</sub> f<sub>or eva</sub>l<sub>ua</sub>ti<sub>on, never as on</sub>li<sub>ne</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> B<sub>ecause mu</sub>lti<sub>p</sub>li<sub>er</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on use</sub> th<sub>e same permu</sub>t<sub>a</sub>ti<sub>ons, a</sub>ll <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d<sub>-po</sub>li<sub>cy resu</sub>lt<sub>s are</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>e</sub>d <sub>as</sub> ex<sub>p</sub><sup>l</sup>orator<sub>y</sub> tune<sup>d</sup> com<sub>p</sub>ar<sup>i</sup>sons.

Observations From Figure 2a, it is evident that, both the linear-logistic and nonlinear HGB i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>ti<sub>ons o</sub>f S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide ac</sub>hi<sub>eve</sub> l<sub>ower mean rea</sub>li<sub>ze</sub>d t<sub>o</sub>t<sub>a</sub>l <sub>cos</sub>t th<sub>an</sub> CBPS<sub>ide, w</sub>hil<sub>e</sub> th<sub>e non</sub>li<sub>near</sub> HGB <sub>var</sub>i<sub>an</sub>t <sub>a</sub>l<sub>so ou</sub>t<sub>per</sub>f<sub>orms</sub> PG<sub>-</sub>TS<sub>.</sub> Th<sub>ese resu</sub>lt<sub>s</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e a prac</sub>ti<sub>ca</sub>l <sub>a</sub>d<sub>van</sub>t<sub>age o</sub>f th<sub>e orac</sub>l<sub>e-e</sub>fi<sub>c</sub>i<sub>en</sub>t S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on:</sub> it <sub>can</sub> i<sub>ncorpora</sub>t<sub>e</sub> fl<sub>ex</sub>ibl<sub>e non</sub>li<sub>near pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s w</sub>ith<sub>ou</sub>t <sub>mo</sub>dif<sub>y</sub>i<sub>ng</sub> th<sub>e surroun</sub>di<sub>ng rou</sub>ti<sub>ng po</sub>li<sub>cy.</sub> F<sub>ur</sub>th<sub>ermore, genera</sub>ll<sub>y</sub> ETC i<sub>s expec</sub>t<sub>e</sub>d t<sub>o</sub> <sub>per</sub>f<sub>orm worse</sub> d<sub>ue</sub> t<sub>o</sub> $O ( T ^ { 2 / 3 } )$ regret, however here due to limited test data size 12,648 compared to larger dimension 138, it performs competitively. It is evident in the learning curves as well (Figure 4).

![](images/60ae4687e09e211266e4787581413bba10fa3aa88bd3fc57dbafdd900d51126f.jpg)  
Fi<sub>g</sub>ure 3: Routin<sub>g</sub>–accurac<sub>y</sub> trade-of on BoolQ. The horizontal axis reports the fraction of prompts <sub>rou</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> <sub>s</sub>t<sub>rong</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>ver</sub>ti<sub>ca</sub>l <sub>ax</sub>i<sub>s</sub> <sub>repor</sub>t<sub>s</sub> <sub>agreemen</sub>t <sub>w</sub>ith th<sub>e</sub> <sub>cac</sub>h<sub>e</sub>d <sub>s</sub>t<sub>rong-mo</sub>d<sub>e</sub>l <sub>response.</sub> Th<sub>us, po</sub>i<sub>n</sub>t<sub>s c</sub>l<sub>oser</sub> t<sub>o</sub> th<sub>e upper-</sub>l<sub>e</sub>ft <sub>corner are pre</sub>f<sub>era</sub>bl<sub>e, as</sub> th<sub>ey ac</sub>hi<sub>eve</sub> hi<sub>g</sub>h<sub>er agreemen</sub>t <sub>w</sub>hil<sub>e</sub> <sub>ma</sub>ki<sub>ng</sub> f<sub>ewer</sub> <sub>s</sub>t<sub>rong-mo</sub>d<sub>e</sub>l <sub>quer</sub>i<sub>es.</sub> E<sub>ac</sub>h <sub>mar</sub>k<sub>er</sub> i<sub>s</sub> th<sub>e</sub> <sub>mean</sub> <sub>over</sub> 20 <sub>s</sub>h<sub>u</sub>fl<sub>e</sub>d <sub>on</sub>li<sub>ne</sub> <sub>or</sub>d<sub>ers,</sub> <sub>w</sub>ith <sub>con</sub>fid<sub>ence</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s om</sub>itt<sub>e</sub>d f<sub>or v</sub>i<sub>sua</sub>l <sub>c</sub>l<sub>ar</sub>it<sub>y.</sub> Al<sub>ong eac</sub>h <sub>curve,</sub> th<sub>e connec</sub>t<sub>e</sub>d <sub>mar</sub>k<sub>ers</sub> f<sub>o</sub>ll<sub>ow</sub> i<sub>ncreas</sub>i<sub>ng</sub> <sub>va</sub>l<sub>ues</sub> $\ell _ { 0 1 } \in \{ 1 . 8 , 2 . 0 , 2 . 2 , 2 . 4 , 2 . 6 , 2 . 8 , 3 . 0 , \bar { 3 } . 2 , 3 . 3 \}$ <sub>.</sub> In<sub>c</sub>r<sub>eas</sub>in<sub>g</sub> $\ell _ { 0 1 }$ ass<sup>i</sup><sub>g</sub>ns a <sup>l</sup>ar<sub>g</sub>er <sub>pena</sub>lt<sub>y</sub> t<sub>o an</sub> i<sub>ncorrec</sub>t <sub>wea</sub>k<sub>-mo</sub>d<sub>e</sub>l <sub>response an</sub>d th<sub>ere</sub>f<sub>ore genera</sub>ll<sub>y moves</sub> th<sub>e po</sub>li<sub>cy</sub> t<sub>owar</sub>d <sub>more</sub> <sub>s</sub>t<sub>rong-mo</sub>d<sub>e</sub>l <sub>rou</sub>ti<sub>ng an</sub>d hi<sub>g</sub>h<sub>er agreemen</sub>t<sub>.</sub>

M<sub>oreover,</sub> Fi<sub>gure</sub> 3 <sub>s</sub>h<sub>ows</sub> th<sub>e</sub> t<sub>ra</sub>d<sub>e-o</sub>f b<sub>e</sub>t<sub>ween</sub> <sub>rou</sub>ti<sub>ng</sub> <sub>accuracy</sub> <sub>an</sub>d th<sub>e</sub> <sub>ra</sub>t<sub>e</sub> <sub>a</sub>t <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>s</sub>t<sub>rong</sub> <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> i<sub>nvo</sub>k<sub>e</sub>d<sub>; a</sub>l<sub>gor</sub>ith<sub>ms ac</sub>hi<sub>ev</sub>i<sub>ng</sub> hi<sub>g</sub>h<sub>er accuracy a</sub>t <sub>a</sub> l<sub>ower</sub> i<sub>nvoca</sub>ti<sub>on ra</sub>t<sub>e are pre</sub>f<sub>era</sub>bl<sub>e.</sub> S<sub>quare</sub>CB<sub>.</sub>PMS<sub>ide w</sub>ith th<sub>e</sub> HGB <sub>mo</sub>d<sub>e</sub>l <sub>a</sub>tt<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> b<sub>es</sub>t t<sub>ra</sub>d<sub>e-o</sub>f <sub>among</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>ms compare</sub>d<sub>.</sub>

Table 2: Meaning of the multiplier s for each algorithm.
<table><tr><td>Algorithm</td><td>Scaled quantity</td><td>Effect of larger s</td></tr><tr><td>CBPSIDE</td><td>confidence width  $s w _ { i }$ </td><td>more exploration</td></tr><tr><td>RANDCBPSIDE</td><td>sampled confidence width  $s w _ { i }$ </td><td>more exploration</td></tr><tr><td>SQUARECB.PMSIDE</td><td>IGW rate  $\kappa = s \sqrt { T }$ </td><td>less exploration</td></tr><tr><td>ETC</td><td>exploration length  $s T ^ { 2 / 3 }$ </td><td>more exploration</td></tr><tr><td>PG-TS</td><td>prior variance  $s \sigma _ { 0 }$ </td><td>more exploration</td></tr></table>

Additional contextual label-eficient prediction experimental details. Contexts are drawn i.i.d. as $x _ { t } \sim \mathrm { U n i f } \left( [ 0 , 1 ] ^ { 1 0 } \right)$ , and a single weight vector w is sampled once from $\mathrm { U n i f } ( 0 , 0 . 1 ) ^ { 1 0 }$ <sub>an</sub>d <sub>norma</sub>li<sub>ze</sub>d <sub>so</sub> th<sub>a</sub>t $w _ { i } \geq 0$ <sub>an</sub>d $\Sigma _ { i } w _ { i } = 1 { : }$ ; <sup>conse</sup>qu<sup>entl</sup>y $w ^ { \top } x$ i<sub>s a convex com</sub>bi<sub>na</sub>ti<sub>on o</sub>f th<sub>e</sub> coordinates of x and lies in $[ 0 , 1 ]$ b<sub>y cons</sub>t<sub>ruc</sub>ti<sub>on.</sub> Th<sub>e</sub> bi<sub>nary ou</sub>t<sub>come</sub> i<sub>s</sub> th<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> th<sub>res</sub>h<sub>o</sub>ld $y _ { t } = \mathbb { 1 } \left\{ w ^ { \top } x _ { t } \geq 0 . 5 \right\}$ <sub>.</sub> W<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub> $c \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 4 5 \}$ <sub>; we s</sub>t<sub>op s</sub>h<sub>or</sub>t <sub>o</sub>f $c = 0 . 5$ b<sub>ecause a</sub>t th<sub>a</sub>t <sub>po</sub>i<sub>n</sub>t th<sub>e ce</sub>ll <sub>o</sub>f th<sub>e exper</sub>t <sub>ac</sub>ti<sub>on co</sub>ll<sub>apses</sub> t<sub>o a s</sub>i<sub>ng</sub>l<sub>e po</sub>i<sub>n</sub>t<sub>:</sub> th<sub>e ac</sub>ti<sub>on</sub> b<sub>ecomes</sub> d<sub>egenera</sub>t<sub>e,</sub> th<sub>e</sub> t<sub>wo pre</sub>di<sub>c</sub>ti<sub>on ac</sub>ti<sub>ons</sub> b<sub>ecome ne</sub>i<sub>g</sub>hb<sub>ors, an</sub>d th<sub>e game</sub> f<sub>a</sub>ll<sub>s ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e non-</sub>d<sub>egenera</sub>t<sub>e c</sub>l<sub>ass</sub> th<sub>a</sub>t <sub>our ana</sub>l<sub>ys</sub>i<sub>s covers.</sub> E<sub>ac</sub>h <sub>exper</sub>i<sub>men</sub>t <sub>uses</sub> h<sub>or</sub>i<sub>zon</sub> $T = 3 0 0 0$ and is repeated over 20 independent seeds, and for each policy and each c we select s ∈ {0.1, 0.3, 1, 3, 10} by minimizing mean total cost over those 20 paired seeds. Each method has exactly one exploration parameter. To place them on a common footing we scale each by a shared multiplier s, sweeping s ∈ {0.1, 0.3, 1, 3, 10} as <sub>summar</sub>i<sub>se</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> F<sub>ur</sub>th<sub>ermore,</sub> Th<sub>e ran</sub>d<sub>om</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>rou</sub>t<sub>es eac</sub>h <sub>query</sub> t<sub>o</sub> th<sub>e s</sub>t<sub>rong</sub> model with a fixed probability, independent of context, swept over {0, 0.2, 0.4, 0.6, 0.8, 1}.

![](images/1f6b99ab40a819ac70dc2a16b184c67a59fa9f001c968b2a4f0c629678276222.jpg)  
(a) $\ell _ { 0 1 } = 1 . 8$

![](images/64e0af27ec6d0054f30d4d57a660c5cc8b0398b1bfe6b83906f944b6cadbab5e.jpg)  
(b) $\ell _ { 0 1 } = 2 . 0$

![](images/02e96f53c25c6555ba06421c8dfdba457233d583d344132e89da505eddaeb2ea.jpg)  
(c) $\ell _ { 0 1 } = 2 . 2$

![](images/dfa0136791810bb1d09862e96302d74c5a07186ba88c4db42638ef2a6eba9251.jpg)

![](images/60e17d80951d4a30fc77426ae5c5eec304b0be4306feaeaf05559494e695fa39.jpg)  
(e) $\ell _ { 0 1 } = 2 . 6$

(d) $\ell _ { 0 1 } = 2 . 4$  
![](images/bee7bd66785ad948c74fbc78d09db96e5bf69973221d23b0a21a9bd647e3239a.jpg)

![](images/f6a09b92ec27f3ba315b0ec57bafc078e8b8697b8c90bbf49b15b4b1d9a5348c.jpg)  
(<sub>g</sub>) $\ell _ { 0 1 } = 3 . 0$

(f) $\ell _ { 0 1 } = 2 . 8$  
![](images/3efa9bf3921926b68fd92d6dae1c8dbdb5012d035ec1041dcd794ebfd1fe866d.jpg)

(h) $\ell _ { 0 1 } = 3 . 2$  
![](images/c05e634f5a1d3ce0ee80a7a7dc5d3fb0fd4987fc640d702643677f82ab5cfa17.jpg)  
(i) $\ell _ { 0 1 } = 3 . 3$  
Fi<sub>gure</sub> 4<sub>:</sub> C<sub>umu</sub>l<sub>a</sub>ti<sub>ve emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>regre</sub>t<sub>, measure</sub>d <sub>as excess rea</sub>li<sub>ze</sub>d <sub>cos</sub>t <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> fi<sub>xe</sub>d fi<sub>ve-</sub>f<sub>o</sub>ld <sub>cross-</sub>fitt<sub>e</sub>d HGB <sub>re</sub>f<sub>erence po</sub>li<sub>cy,</sub> f<sub>or eac</sub>h <sub>va</sub>l<sub>ue o</sub>f $\ell _ { 0 1 }$ . Curves show means over 20 paired random arrival orders, and shaded regions are pointwise 95% Student-t confidence intervals. Lower values <sub>are</sub> b<sub>e</sub>tt<sub>er; nega</sub>ti<sub>ve regre</sub>t i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> th<sub>a</sub>t th<sub>e on</sub>li<sub>ne po</sub>li<sub>cy</sub> i<sub>ncurs</sub> l<sub>ower rea</sub>li<sub>ze</sub>d <sub>cos</sub>t th<sub>an</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>re</sub>f<sub>erence po</sub>li<sub>cy.</sub>

![](images/960dd9666d07de506476aa031ffc5d73fa02d5eabca8faa28075261cd7ea81b9.jpg)  
(a) $c = 0 . 1$

![](images/39d399825c9ce6d1534915d128bae18b30255e048a3330364c0534a116d24b7b.jpg)  
(b) $c = 0 . 2$

![](images/63c30c519025fde51d164a48bc2a94cd5839cd7ddea23acb5bf9c9f48111ee35.jpg)  
(c) $c = 0 . 3$

1200 Random   
CBPside   
RandCBPside   
1000 SquareCB.PMSide   
ETC   
Cumulative regret 800   
600   
400   
200   
500 1000 1500 2000 2500 3000   
Round t  
(d) $c = 0 . 4$

1400   
Random   
CRPside   
1200 RandCBPside   
SquareCB.PMSide   
ETC   
1000   
Cumulative regret 800   
600   
400   
200   
500 1000 1500 2000 2500 3000   
Round t  
(e) $c = 0 . 4 5$  
Figure 5: Cumulative regret against round t for each value of the information cost $c ,$ <sub>w</sub>ith <sub>every</sub> algorithm at its best multiplier. Shaded bands are 95% Student-t confidence intervals for the mean over 20 seeds.

Observations Figure 2b shows total cost as a function of $c ,$ <sub>an</sub>d Fi<sub>gure</sub> 5 th<sub>e</sub> l<sub>earn</sub>i<sub>ng curves</sub> f<sub>or a</sub>ll <sub>compare</sub>d <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>across</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>sweep.</sub> Th<sub>ese</sub> <sub>ex</sub>t<sub>en</sub>d th<sub>e</sub> <sub>resu</sub>lt<sub>s</sub> di<sub>scusse</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4<sub>.</sub>

## F Algorithm for Lq-decreasing tree construction $\pmb { L q }$

Gi<sub>ven a</sub> l<sub>oca</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e game</sub> $\mathcal { G } = \left( L , \Phi \right)$ <sub>an</sub>d <sub>an ou</sub>t<sub>come</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q \in \Delta ( d )$ <sub>, we</sub> d<sub>eno</sub>t<sub>e</sub> $\ell _ { a } : = e _ { a } ^ { \intercal } \mathbf { { \cal L } } q$ for the loss of action a under the outcome distribution q. Since a partial monitoring game can be represented as a neighborhood graph (Π, E), we can construct a spanning in-tree $\mathcal T \subseteq \mathcal E$ rooted at the action with the smallest loss aˆ ∈ arg mi $\mathfrak { l } _ { a \in \Pi } \ell _ { a }$ <sub>.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng a</sub>l<sub>gor</sub>ith<sub>m cons</sub>t<sub>ruc</sub>t<sub>s an</sub> $\scriptstyle { L q } .$ ∈-decreasing in-tree by selecting each action’s best neighbor to be its parent, then constructing the in-tree T with the selected neighbors as edges.

Algorithm 5 Constructing an $L q \mathrm { . }$ <sub>-</sub>d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n-</sub>t<sub>ree</sub> b<sub>y se</sub>l<sub>ec</sub>ti<sub>ng eac</sub>h <sub>ac</sub>ti<sub>on</sub>’<sub>s paren</sub>t   
1: Input: A locally observable game $\mathcal { G } = \left( L , \Phi \right)$ <sub>,</sub> it<sub>s ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d <sub>grap</sub>h $( \Pi , { \mathcal { E } } )$ <sub>,</sub> <sub>an</sub>d $q \in \Delta ( d )$   
2: Output: A spanning in-tree $\bar { \tau } \subseteq \mathcal { E }$ with root root $\boldsymbol { \tau } = \boldsymbol { \hat { a } }$ <sub>suc</sub>h th<sub>a</sub>t $\pmb { L q }$ is T-decreasing.   
3<sub>:</sub> F<sub>or eac</sub>h $a \in { \dot { \Pi } } ,$ <sup>com</sup>pu<sup>te</sup> $\ell _ { a } = e _ { a } ^ { \top } \pmb { { \cal L } } q$ T, and choose the root aˆ ∈ arg min $\mathfrak { l } _ { a \in \Pi } \ell _ { a }$   
4<sub>:</sub> Ch<sub>oose</sub> $q ^ { \circ } \in \mathrm { r i } ( \mathcal { C } _ { \hat { a } } )$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s</sub> th<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or o</sub>f th<sub>e ce</sub>ll $\mathcal { C } _ { \hat { a } }$   
5<sub>:</sub> I<sub>n</sub>iti<sub>a</sub>li<sub>ze</sub> $\mathcal { T }  \emptyset .$   
6: for each $a \in \Pi \setminus \{ \hat { a } \}$ do   
7<sub>:</sub> Fi<sub>n</sub>d th<sub>e</sub> b<sub>es</sub>t <sub>ne</sub>i<sub>g</sub>hb<sub>ors</sub> $B _ { a } \gets \mathrm { a r g }$ min $\mathfrak { l } _ { b : ( a , b ) \in \mathcal { E } } \ell _ { b } .$   
8<sub>:</sub> Choose par (a) ∈ arg m $\mathrm { i n } _ { b \in B _ { a } } e _ { b } ^ { \top } \pmb { { \cal L } } q ^ { \circ }$ ∶ E, breaking any remaining ties arbitrarily.   
9: $\mathcal { T }  \mathcal { T } \overset { \cdot } { \cup } \{ ( a , \overset { \cdot } { \operatorname { p a r } } _ { \mathcal { T } } ( \overset { \cdot } { a } ) ) \}$   
10: end for   
11: return the in-tree T with roo $\gamma = { \hat { a } } .$

The proof of such an Lq-decreasing in-tree construction is provided in Lattimore & Szepesvari [2020,´ Lemma 19].

## G A<sub>uxiliary</sub> L<sub>emmata</sub>

Lemma G.1 (Pareto optimal comparators sufice). For every $q \in \Delta ( d )$

$$
\operatorname* { m i n } _ { \lambda \in \Delta ( \Pi ) } \lambda ^ { \top } L q = \operatorname* { m i n } _ { a \in \Pi } e _ { a } ^ { \top } L q = \operatorname* { m i n } _ { a \in [ k ] } e _ { a } ^ { \top } L q = \operatorname* { m i n } _ { \lambda \in \Delta ( k ) } \lambda ^ { \top } L q .
$$

Proof. A linear function on a simplex attains its minimum at a vertex, which gives the outer two <sub>equa</sub>liti<sub>es, so</sub> it <sub>su</sub>fi<sub>ces</sub> t<sub>o s</sub>h<sub>ow</sub> $\bar { \bigcup } _ { a \in \Pi } \mathcal { C } _ { a } = \Delta ( d )$ Th<sub>e ce</sub>ll<sub>s cover</sub> $\Delta ( d )$ b<sub>y</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> $2 . 1$ <sub>,</sub> <sub>an</sub>d $\cup _ { a \notin \Pi } \mathcal { C } _ { a }$ ∈is a finite union of <sub>p</sub>ol<sub>y</sub>to<sub>p</sub>es of dimension at most $d - 2$ <sub>,</sub> h<sub>ence</sub> h<sub>as emp</sub>t<sub>y</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> $( d - 1 )$ <sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>se</sub>t $\bar { \Delta ( d ) }$ <sub>.</sub> S<sub>o</sub> $\textstyle \bigcup _ { a \in \Pi } { \mathcal { C } } _ { a }$ i<sub>s</sub> d<sub>ense</sub> i<sub>n</sub> $\Delta ( d )$ <sub>;</sub> b<sub>e</sub>i<sub>ng a</sub> fi<sub>n</sub>it<sub>e un</sub>i<sub>on o</sub>f <sub>c</sub>l<sub>ose</sub>d <sub>se</sub>t<sub>s</sub> it i<sub>s c</sub>l<sub>ose</sub>d<sub>, an</sub>d th<sub>ere</sub>f<sub>ore equa</sub>l<sub>s</sub> $\Delta ( d )$ . □

Theorem D.1 (Regret concentration generalized from Qin & Zhang [2026a]). Let $\Lambda$ b<sub>e a se</sub>t <sub>o</sub>f <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> $\mathcal { L } ( \cdot \mid x ) : \Lambda \to \left[ 0 , 1 \right]$ <sub>a va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on an</sub>d $\mathcal { L } ^ { \star } ( \cdot \vert x )$ <sub>an</sub>d $\widehat { \mathcal { L } } _ { \ell } ( \cdot | \overset { - } { x } ) , \ell \in [ L ]$ <sub>are va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>ons</sub> <sub>measure</sub>d <sub>un</sub>d<sub>er</sub> th<sub>e</sub> t<sub>rue</sub> <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> $\lambda ^ { \star } \in \Lambda$ <sub>an</sub>d $\hat { \lambda } _ { \ell } \in \Lambda$ are res<sub>p</sub>ect<sup>i</sup>ve<sup>l</sup><sub>y</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>po</sub>li<sub>c</sub>i<sub>es</sub> <sub>un</sub>d<sub>er</sub> <sub>respec</sub>ti<sub>ve</sub> t<sub>rue</sub> <sub>an</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> L<sub>e</sub>t $C _ { \ell } : \Lambda \times \Lambda \to \mathbb { R }$ <sup>b</sup>e covera<sub>g</sub>e <sup>ma</sup>p<sup>s</sup>, $p _ { \ell } \in \Lambda$ <sub>are</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons,</sub> $0 < \gamma _ { 1 } \leq \gamma _ { 2 } \leq \cdots \leq \gamma _ { L }$ l<sub>earn</sub>i<sub>ng ra</sub>t<sub>es, an</sub>d $V _ { \ell } , E _ { \ell } \ge 0$ +numbers such that,

(C1) (certificate) for ever<sub>y</sub> $\ell \geq 1$ an<sup>d</sup> ever<sub>y</sub> $\lambda \in \Lambda$

$$
\widehat { \mathscr { L } } _ { \ell } ( p _ { \ell } \vert x ) - \widehat { \mathscr { L } } _ { \ell } ( \lambda \vert x ) + \frac { 1 } { \gamma _ { \ell } } C _ { \ell } ( p _ { \ell } , \lambda ) \leq V _ { \ell } .
$$

(C2) (of-<sub>p</sub>olic<sub>y</sub> evaluation) for ever<sub>y</sub> $\ell \geq 2$ an<sup>d</sup> ever<sub>y</sub> $\lambda \in \Lambda$

$$
\left| \widehat { \mathcal { L } } _ { \ell } ( \lambda - p _ { \ell - 1 } \mid x ) - \mathcal { L } ^ { \star } ( \lambda - p _ { \ell - 1 } \mid x ) \right| \leq \frac { 1 } { 6 \gamma _ { \ell } } C _ { \ell - 1 } ( p _ { \ell - 1 } , \lambda ) + \frac { 3 \gamma _ { \ell } } { 2 } E _ { \ell } .
$$

D<sub>e</sub>fi<sub>ne</sub> $G _ { 1 } : = 1$ <sub>an</sub>d<sub>,</sub> f<sub>or</sub> $\ell \geq 2 .$

$$
\gamma _ { \ell } G _ { \ell } : = \gamma _ { \ell - 1 } G _ { \ell - 1 } + \gamma _ { \ell - 1 } V _ { \ell - 1 } + 9 \gamma _ { \ell } ^ { 2 } E _ { \ell } , \qquad \mathrm { i . e . } \qquad G _ { \ell } = \frac { 1 } { \gamma _ { \ell } } \left[ \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } E _ { i } ) \right] .
$$

Th<sub>en,</sub> f<sub>or</sub> <sub>every</sub> $\ell \in [ L ]$ an<sup>d</sup> ever<sub>y</sub> $\lambda \in \Lambda$

$$
\operatorname { R e g } ( \lambda \mid x ) \leq 2 { \widehat { \operatorname { R e g } } } _ { \ell } ( \lambda \mid x ) + G _ { \ell } \qquad { \mathrm { a n d } } \qquad { \widehat { \operatorname { R e g } } } _ { \ell } ( \lambda \mid x ) \leq 2 \operatorname { R e g } ( \lambda \mid x ) + G _ { \ell } .\tag{13}
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub> $\operatorname { R e g } ( p _ { \ell } \mid x ) \leq 2 V _ { \ell } + G _ { \ell }$ <sup>f</sup>or ever<sub>y</sub> $\ell ,$ <sub>an</sub>d

$$
\sum _ { \ell = 1 } ^ { L } ( \tau _ { \ell } - \tau _ { \ell - 1 } ) \operatorname { R e g } ( p _ { \ell } \mid x ) \leq \tau _ { 1 } + \sum _ { \ell = 2 } ^ { L } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left[ 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } E _ { i } ) \right] .\tag{14}
$$

Proof. We prove $\operatorname { E q } .$ <sub>. equa</sub>ti<sub>on</sub> 13 b<sub>y</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on on</sub> $\ell .$ F<sub>or</sub> $\ell = 1$ b<sub>o</sub>th i<sub>nequa</sub>liti<sub>es</sub> h<sub>o</sub>ld <sub>w</sub>ith $G _ { 1 } = 1$ since Reg and $\widehat { \mathrm { R e g } } _ { 1 }$ take values in [0, 1]. Fix $\ell \geq 2$ <sub>an</sub>d <sub>assume</sub> E<sub>q. equa</sub>ti<sub>on</sub> 13 t<sub>o</sub> h<sub>o</sub>ld <sub>a</sub>t $\ell - 1$

Fi<sub>rs</sub>t<sub>, we es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>a</sub>t<sub>,</sub> f<sub>or every</sub> $\lambda ,$ our <sub>p</sub>olic<sub>y</sub> and estimator <sub>g</sub>uarantees Low Re<sub>g</sub>ret (LR) and Good Covera<sub>g</sub>e(GC).

That is, from (C1) for e<sub>p</sub>och $\ell - 1$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r } { \widehat { \mathcal { L } } _ { \ell - 1 } ( p _ { \ell - 1 } \mid x ) - \widehat { \mathcal { L } } _ { \ell - 1 } ( \lambda \mid x ) + \frac { 1 } { \gamma _ { \ell - 1 } } C _ { \ell - 1 } ( p _ { \ell - 1 } , \lambda ) \leq V _ { \ell - 1 } } \\ { \widehat { \mathcal { L } } _ { \ell - 1 } ( p _ { \ell - 1 } \mid x ) - \widehat { \mathcal { L } } _ { \ell - 1 } ( \widehat { \lambda } _ { \ell - 1 } \mid x ) - ( \widehat { \mathcal { L } } _ { \ell - 1 } ( \lambda \mid x ) - \widehat { \mathcal { L } } _ { \ell - 1 } ( \widehat { \lambda } _ { \ell - 1 } \mid x ) ) + \frac { 1 } { \gamma _ { \ell - 1 } } C _ { \ell - 1 } ( p _ { \ell - 1 } , \lambda ) \leq V _ { \ell - 1 } } \\ { \widehat { \mathrm { R e g } } _ { \ell - 1 } ( p _ { \ell - 1 } \mid x ) + \left( \frac { 1 } { \gamma _ { \ell - 1 } } C _ { \ell - 1 } ( p _ { \ell - 1 } , \lambda ) - \widehat { \mathrm { R e g } } _ { \ell - 1 } ( \lambda \mid x ) \right) \leq V _ { \ell - 1 } } \end{array}
$$

Ab<sub>ove s</sub>t<sub>a</sub>t<sub>emen</sub>t i<sub>s</sub> t<sub>rue</sub> f<sub>or a</sub>ll $\lambda ,$ i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> th<sub>e case w</sub>h<sub>ere</sub> ${ \widehat { \mathrm { R e g } } } _ { \ell - 1 } ( \lambda \mid x ) = 0 { \mathrm { a n d } } C _ { \ell } ( p _ { \ell - 1 } , \lambda )$ i<sub>s</sub> $\mathrm { \ a l w a y s \geq 0 }$ <sub>.</sub> H<sub>ence,</sub>

$$
\widehat { \mathrm { R e g } } _ { \ell - 1 } ( p _ { \ell - 1 } | x ) \leq V _ { \ell - 1 } .
$$

(Low Re<sub>g</sub>ret (LR))

Al<sub>so,</sub>

$$
\left( { \frac { 1 } { \gamma _ { \ell - 1 } } } C _ { \ell - 1 } ( p _ { \ell - 1 } , \lambda ) - \widehat { \mathrm { R e g } } _ { \ell - 1 } ( \lambda | x ) \right) \leq V _ { \ell - 1 } .
$$

H<sub>ence,</sub>

$$
C _ { \ell - 1 } ( p _ { \ell - 1 } , \lambda ) \leq \gamma _ { \ell - 1 } \left[ V _ { \ell - 1 } + \widehat { \mathrm { R e g } } _ { \ell - 1 } ( \lambda \mid x ) \right] .\tag{Good Covera<sub>g</sub>e (GC)}
$$

M<sub>oreover,</sub>

$$
\begin{array}{c} \begin{array} { r l } & { \mathrm { ~ t o t s i n g ~ } ( \begin{array} { l l l l } { 1 } & { \rho _ { 1 } ( \rho _ { 1 } , \rho _ { 1 } ) } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} ) } \\ & { = \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad - \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad - \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad - \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad - \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad - \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad \quad + \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \end{array} \{array ( \begin{array} { l l l l } { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} )  \\ & { \quad - \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad - \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad \quad + \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad \quad - \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \end{array} )  \\ & { = \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ & { \quad \sigma _ { 1 } ( \rho _ { 1 } , \rho _ { 1 } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \sigma _ { 1 } ( \rho _ { 1 } , \rho _ { 1 } ) + \sigma _ { 1 } ( \rho _ { 1 } , 0 ) } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad
$$

H<sub>ence</sub>

$$
\begin{array} { c } { { { \displaystyle \mathrm { R e g } ( \lambda \vert x ) - \widetilde { \mathrm { R e g } } _ { \ell } ( \lambda \vert x ) \le \frac { 1 } { 3 } \mathrm { R e g } ( \lambda \vert x ) + \frac { 1 } { 3 } G _ { \ell } } } } \\ { { { \displaystyle \frac { 2 } { 3 } \mathrm { R e g } ( \lambda \vert x ) \le \widetilde { \mathrm { R e g } } _ { \ell } ( \lambda \vert x ) + \frac { 1 } { 3 } G _ { \ell } } } } \\ { { { \displaystyle \mathrm { R e g } ( \lambda \vert x ) \le \frac { 3 } { 2 } \widetilde { \mathrm { R e g } } _ { \ell } ( \lambda \vert x ) + \frac { 1 } { 2 } G _ { \ell } } } } \\ { { { \le 2 \widetilde { \mathrm { R e g } } _ { \ell } ( \lambda \vert x ) + G _ { \ell } } } } \end{array}\tag{Ste<sub>p</sub> **}
$$

F<sub>ur</sub>th<sub>ermore,</sub>

$$
\begin{array} { r l } & { \mathrm { R e g } _ { \ell } ( \lambda \mid x ) - \mathrm { R e g } ( \lambda \mid x ) } \\ & { \quad = \widehat { \mathcal { L } } _ { \ell } ( \lambda \mid x ) - \widehat { \mathcal { L } } _ { \ell } ( \widehat { \lambda } _ { \ell } \mid x ) - \left( \mathcal { L } ^ { * } ( \lambda \mid x ) - \mathcal { L } ^ { * } ( \lambda ^ { * } \mid x ) \right) } \\ & { \quad = \widehat { \mathcal { L } } _ { \ell } ( \lambda - p _ { \ell - 1 } \mid x ) - \widehat { \mathcal { L } } _ { \ell } ( \widehat { \lambda } _ { \ell } - p _ { \ell - 1 } \mid x ) - \left( \mathcal { L } ^ { * } ( \lambda - p _ { \ell - 1 } \mid x ) - \mathcal { L } ^ { * } ( \lambda ^ { * } - p _ { \ell - 1 } \mid x ) \right) } \\ & { \quad = \widehat { \mathcal { L } } _ { \ell } ( \lambda - p _ { \ell - 1 } \mid x ) - \mathcal { L } ^ { * } ( \lambda - p _ { \ell - 1 } \mid x ) - \left( \widehat { \mathcal { L } } _ { \ell } ( \widehat { \lambda } _ { \ell } - p _ { \ell - 1 } \mid x ) - \mathcal { L } ^ { * } ( \lambda ^ { * } - p _ { \ell - 1 } \mid x ) \right) } \\ & { \quad \le \widehat { \mathcal { L } } _ { \ell } ( \lambda - p _ { \ell - 1 } \mid x ) - \mathcal { L } ^ { * } ( \lambda - p _ { \ell - 1 } \mid x ) - \left( \widehat { \mathcal { L } } _ { \ell } ( \widehat { \lambda } _ { \ell } - p _ { \ell - 1 } \mid x ) - \mathcal { L } ^ { * } ( \widehat { \lambda } _ { \ell } - p _ { \ell - 1 } \mid x ) \right) } \\ &  \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ( \mathcal { L } ^ { * } ( \widehat { \lambda } _ { \ell } \mid x ) \ge \mathcal { L } ^ { * } ( \lambda ^ { * } \mid x ) ) \end{array}
$$

$$
\begin{array} { r l } & { \quad \le \frac { 1 } { \delta \gamma _ { \sigma } } C _ { \sigma } ( \gamma _ { \sigma } \Lambda _ { 1 } ) + \frac { 3 \gamma _ { \sigma } } { 2 } E _ { \sigma } + \frac { 1 } { \delta \gamma _ { \sigma } } C _ { \sigma } ( - \frac { 1 } { \delta \gamma _ { \sigma } } \Lambda _ { 1 } ) + \frac { 3 \alpha _ { \sigma } } { 2 } E _ { \sigma } } \\ & { \quad \le \frac { 1 } { \delta \gamma _ { \sigma } } C _ { \sigma } ( \gamma _ { \sigma } \Lambda _ { 1 } ) - \frac { 3 } { 2 } \gamma _ { \sigma } ( - \frac { 1 } { \delta \gamma _ { \sigma } } \Lambda _ { 1 } ) + \frac { 3 } { 2 } \gamma _ { \sigma } ( \gamma _ { \sigma } \Lambda _ { 1 } ) - [ \gamma _ { \sigma } \Lambda _ { 1 } ] - \frac { 3 } { 2 } \gamma _ { \sigma } ( \gamma _ { \sigma } \Lambda _ { 1 } ) [ \gamma _ { \sigma } ] - \frac { 3 } { 2 } \gamma _ { \sigma } ( E _ { \sigma } ) } \\ & { \quad \le \frac { 1 } { \delta \gamma _ { \sigma } } C _ { \sigma } ( - [ 1 + \frac { 3 } { 2 } \delta \gamma _ { \sigma } + \Lambda _ { 1 } ] ) + \frac { 3 } { 2 } \gamma _ { \sigma } ( \gamma _ { \sigma } \Lambda _ { 1 } ) + \frac { 3 } { 2 } \gamma _ { \sigma } ( \gamma _ { \sigma } \Lambda _ { 1 } ) } \\ &  \quad \le \frac { 1 } { \delta \gamma _ { \sigma } } \gamma _ { \sigma } [ ( E _ { \sigma } ) + [ 1 + \frac { 3 } { 2 } \delta \gamma _ { \sigma } + \Lambda _ { 1 } ] + \frac { 3 } { 2 } \gamma _ { \sigma } ( \gamma _ { \sigma } ) + \frac { 1 } { \delta \gamma _ { \sigma } } \gamma _ { \sigma } ( \gamma _ { \sigma } ) + \frac { 3 } { 4 \gamma _ { \sigma } } (  \end{array}
$$

H<sub>ence,</sub>

$$
\begin{array} { r l r } & { } & { \widehat { \mathrm { R e g } } _ { \ell } ( \lambda \mid x ) - \mathrm { R e g } ( \lambda \mid x ) \le \displaystyle \frac { 1 } { 3 } \mathrm { R e g } ( \lambda \mid x ) + \displaystyle \frac { 2 } { 3 } G _ { \ell } } \\ & { } & { \widehat { \mathrm { R e g } } _ { \ell } ( \lambda \mid x ) \le \displaystyle \frac { 4 } { 3 } \mathrm { R e g } ( \lambda \mid x ) + \displaystyle \frac { 2 } { 3 } G _ { \ell } } \\ & { } & { \mathrm { R e g } ( \lambda \mid x ) \le \displaystyle \frac { 3 } { 2 } \widehat { \mathrm { R e g } } _ { \ell } ( \lambda \mid x ) + \displaystyle \frac { 1 } { 2 } G _ { \ell } } \\ & { } & { \le 2 \widehat { \mathrm { R e g } } _ { \ell } ( \lambda \mid x ) + G _ { \ell } } \end{array}
$$

Finall<sub>y</sub>, we <sub>p</sub>rove the consequences. A<sub>pp</sub>l<sub>y</sub> (C1) at $\begin{array} { r } { \lambda = \hat { \lambda } _ { \ell } , } \end{array}$ <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } & { \widehat { \mathrm { R e g } } _ { \ell } ( p _ { \ell }  { | \begin{array} { l } { x } \end{array}  } ) \leq V _ { \ell } } \\ & { \Rightarrow \mathrm { R e g } ( p _ { \ell }  { | \begin{array} { l } { x } \end{array}  ) } \leq 2 \widehat { \mathrm { R e g } } _ { \ell } ( p _ { \ell }  { | \begin{array} { l } { x } \end{array}  } ) + G _ { \ell } \leq 2 V _ { \ell } + G _ { \ell } . } \end{array}
$$

(equation Low Re<sub>g</sub>ret (LR))

(B<sub>y</sub> equation 13)

S<sub>umm</sub>i<sub>ng over epoc</sub>h<sub>s, we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } & { \displaystyle \sum _ { \ell = 1 } ^ { L } ( \tau _ { \ell } - \tau _ { \ell - 1 } ) \operatorname { R e g } ( p _ { \ell } \mid x ) \leq \tau _ { 1 } + \sum _ { \ell = 2 } ^ { L } ( \tau _ { \ell } - \tau _ { \ell - 1 } ) ( 2 V _ { \ell } + G _ { \ell } ) \medskip } \\ & { \quad \quad \quad = \tau _ { 1 } + \displaystyle \sum _ { \ell = 2 } ^ { L } \frac { \tau _ { \ell } - \tau _ { \ell - 1 } } { \gamma _ { \ell } } \left[ 2 \gamma _ { \ell } V _ { \ell } + \gamma _ { 1 } + \displaystyle \sum _ { i = 2 } ^ { \ell } ( \gamma _ { i - 1 } V _ { i - 1 } + 9 \gamma _ { i } ^ { 2 } E _ { i } ) \right] , } \end{array}
$$

(B<sub>y</sub> the closed form of $G _ { \ell } )$

<sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> E<sub>q.</sub> <sub>equa</sub>ti<sub>on</sub> 14<sub>.</sub>

Lemma G.2. Let $P , Q$ <sub>are</sub> t<sub>wo pro</sub>b<sub>a</sub>bilit<sub>y s</sub>i<sub>mp</sub>li<sub>ces.</sub> $D _ { \mathrm { H } } ( P \left| Q \right. ) , \ j _ { \mathrm { K L } } ( P \left| Q ) , \ j _ { \mathrm { l } } P - Q \right| \ l _ { \mathrm { T V } } , \ j _ { P } - Q \ j _ { \mathrm { l } } .$ b<sub>e</sub> th<sub>e</sub> H<sub>e</sub>lli<sub>nger</sub> di<sub>s</sub>t<sub>ance,</sub> K<sub>u</sub>llb<sub>ac</sub>k di<sub>vergence,</sub> t<sub>o</sub>t<sub>a</sub>l <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>ance an</sub>d l<sub>eve</sub>l<sub>-</sub>1 <sub>norm</sub> b<sub>e</sub>t<sub>ween</sub> $P _ { \mathrm { : } }$ $Q$ <sub>respec</sub>ti<sub>ve</sub>l<sub>y,</sub> th<sub>en</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> h<sub>o</sub>ld<sub>s,</sub>

$$
{ \frac { \| P - Q \| _ { 1 } } { 2 } } = \| P - Q \| _ { \mathrm { T V } } \leq D _ { \mathrm { H } } ( P | Q )
$$

<sub>an</sub>d<sub>,</sub>

$$
{ \frac { \| P - Q \| _ { 1 } } { 2 } } = \| P - Q \| _ { \mathrm { T V } } \leq { \sqrt { D _ { \mathrm { K L } } ( P | Q ) } }
$$

Proof.

$$
\| P - Q \| _ { \mathrm { T V } } \leq D _ { \mathrm { H } } ( P | Q )
$$

( Ts<sub>y</sub>bakov [2008], Lemma 2.3)

<sub>an</sub>d<sub>,</sub>

$$
\| P - Q \| _ { \mathrm { T V } } \leq { \sqrt { D _ { \mathrm { K L } } ( P | Q ) } }\tag{Ts<sub>y</sub>bakov [2008], Lemma 2.5}
$$

F<sub>ur</sub>th<sub>ermore,</sub>

$$
{ \frac { \| P - Q \| _ { 1 } } { 2 } } = \| P - Q \| _ { \mathrm { T V } }
$$

(Levin & Peres [2017], Pro<sub>p</sub>osition 4.2)

Lemma G.3 (Local observability, Lattimore & Szepesvari [2020])´ . If ${ \mathcal { G } } = ( \Phi , L )$ i<sub>s</sub> l<sub>oca</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e an</sub>d <sub>non-</sub>d<sub>egenera</sub>t<sub>e an</sub>d <sub>ac</sub>ti<sub>ons</sub> $e = ( a , b ) \in \mathcal { E }$ <sub>are ne</sub>i<sub>g</sub>hb<sub>ours,</sub> th<sub>en</sub> th<sub>ere ex</sub>i<sub>s</sub>t f<sub>unc</sub>ti<sub>ons</sub> $w _ { e } ( a , \cdot ) , w _ { e } ( b , \cdot ) : \Sigma $ R <sub>suc</sub>h th<sub>a</sub>t $\| w _ { e } \| _ { \infty } \leq m / 2$ <sub>an</sub>d

$$
\begin{array} { r } { \pmb { L _ { a o } } - \pmb { L _ { b o } } = w _ { e } \big ( { a } , \Phi _ { a o } \big ) + w _ { e } \big ( { b } , \Phi _ { b o } \big ) \mathrm { f o r ~ a l l } o \in \big [ d ] . } \end{array}
$$

I<sub>n</sub> L<sub>emma</sub> $\mathrm { { G . 3 } } ,$ <sub>,</sub> th<sub>e assump</sub>ti<sub>on o</sub>f <sub>non-</sub>d<sub>egeneracy</sub> i<sub>s</sub> i<sub>mpor</sub>t<sub>an</sub>t h<sub>ere</sub> $\mathrm { f o r } \| w _ { e } \| _ { \infty } \leq m / 2$ t<sub>o</sub> h<sub>o</sub>ld<sub>.</sub> Th<sub>e</sub> P<sub>ropos</sub>iti<sub>on</sub> G<sub>.</sub>4 b<sub>oun</sub>d<sub>s</sub> $\| \boldsymbol { w } _ { e } \| _ { \infty }$ f<sub>or</sub> <sub>more</sub> <sub>genera</sub>l <sub>case.</sub>

∞<sub>Lemma G.4 (</sub>L<sub>att</sub>i<sub>more</sub> & S<sub>zepesvar</sub>i <sub>[</sub>2020<sub>])</sub>´ <sub>.</sub> S<sub>uppose t</sub>h<sub>at</sub> $( L , \Phi )$ is globall<sub>y</sub> observable and a and b are neighbours. Then there exists a function $w _ { e } : [ k ] \times \Sigma  \mathbb { R }$ <sub>suc</sub>h th<sub>a</sub>t f<sub>or a</sub>ll $x \in [ d ]$

$$
\sum _ { c = 1 } ^ { k } w _ { e } ( c , \Phi _ { c o } ) = L _ { a o } - L _ { b o } .
$$

F<sub>ur</sub>th<sub>ermore,</sub> $w _ { e }$ <sub>can</sub> b<sub>e c</sub>h<sub>osen so</sub> th<sub>a</sub>t $\| w _ { e } \| _ { \infty } \leq d ^ { 1 / 2 } k ^ { d / 2 }$

∞<sub>Lemma G.5 (</sub>W<sub>ater-trans</sub>f<sub>er operator a</sub>d<sub>apte</sub>d f<sub>rom</sub> L<sub>att</sub>i<sub>more</sub> & S<sub>zepesvar</sub>i <sub>[</sub>2020<sub>])</sub>´ <sub>.</sub> S<sub>uppose t</sub>h<sub>at</sub> G is non-degenerate and locally observable and $q \in \Delta ( d )$ A<sub>ssume</sub> th<sub>a</sub>t <sub>we</sub> h<sub>ave an</sub> i<sub>n-</sub>t<sub>ree</sub> $\dot { \tau }$ th<sub>a</sub>t i<sub>s</sub> Lq-decreasing. Then there exists a function $W _ { q } ^ { \mathcal { T } } : \Delta ( \Pi )  \Delta ( \Pi )$ <sub>suc</sub>h th<sub>a</sub>t th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> h<sub>o</sub>ld f<sub>or a</sub>ll $p \in \Delta ( \Pi )$

(a) The ex<sub>p</sub>ected loss does not increase: $\left( W _ { q } ( p ) - p \right) ^ { \intercal } L q \leq 0$

(b) Action <sub>p</sub>robabilities are not too small: $W _ { q } ( p ) [ a ] \geq p [ a ] / k$ f<sub>or a</sub>ll $a \in [ k ]$

(c) Probabilities increase towards the root of the in-tree T. That is, $W _ { q } ( p ) [ a ] \leq W _ { q } ( p ) [ b ]$ f<sub>or</sub> <sub>a</sub>ll $e = ( a , b ) \in \mathcal { T }$