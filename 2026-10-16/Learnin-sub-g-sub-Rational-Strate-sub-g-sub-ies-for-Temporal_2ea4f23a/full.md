# Learnin<sub>g</sub> Rational Strate<sub>g</sub>ies for Temporal Objectives in Multi-Agent Systems

O<sub>mar</sub> Ad<sub>a</sub>l<sub>a</sub>t Im<sub>p</sub>erial Colle<sub>g</sub>e London L<sub>o</sub>nd<sub>o</sub>n<sub>,</sub> Unit<sub>e</sub>d Kin<sub>g</sub>d<sub>o</sub>m <sub>o.a</sub>d<sub>a</sub>l<sub>a</sub>t24<sub>@</sub>i<sub>mper</sub>i<sub>a</sub>l<sub>.ac.u</sub>k

Ed<sub>w</sub>in H<sub>a</sub>m<sub>e</sub>l-D<sub>e</sub> l<sub>e</sub> C<sub>ou</sub>rt Uni<sub>ve</sub>r<sub>s</sub>it<sub>y o</sub>f M<sub>a</sub>n<sub>c</sub>h<sub>es</sub>t<sub>e</sub>r   
M<sub>a</sub>n<sub>c</sub>h<sub>es</sub>t<sub>e</sub>r<sub>,</sub> Unit<sub>e</sub>d Kin<sub>g</sub>d<sub>o</sub>m <sub>e</sub>d<sub>w</sub>i<sub>n.</sub>h<sub>ame</sub>l<sub>-</sub> d<sub>e</sub>l<sub>ecour</sub>t<sub>@manc</sub>h<sub>es</sub>t<sub>er.ac.u</sub>k

F<sub>rancesco</sub> B<sub>e</sub>l<sub>ar</sub>di<sub>ne</sub>lli Im<sub>p</sub>erial Colle<sub>g</sub>e London L<sub>o</sub>nd<sub>o</sub>n<sub>,</sub> Unit<sub>e</sub>d Kin<sub>g</sub>d<sub>o</sub>m f<sub>rancesco.</sub>b<sub>e</sub>l<sub>ar</sub>di<sub>ne</sub>lli<sub>@</sub>i<sub>mper</sub>i<sub>a</sub>l<sub>.ac.u</sub>k

## Ab<sub>s</sub>t<sub>rac</sub>t

We stud<sub>y</sub> rational s<sub>y</sub>nthesis for stochastic multi-a<sub>g</sub>ent s<sub>y</sub>stems with unknown d<sub>y</sub>namics<sub>,</sub> where each a<sub>g</sub>ent is e<sub>q</sub>ui<sub>pp</sub>ed with its own pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic tempora<sup>l</sup> o<sup>b</sup>jective speci<sup>fi</sup>e<sup>d</sup> using Pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic Linear Tem<sub>p</sub>oral Lo<sub>g</sub>ic on Finite Traces (pltl<sub>�</sub>). We im<sub>p</sub>ose solution conce<sub>p</sub>ts (e.<sub>g</sub>., Nash e<sub>q</sub>uilibrium, sub<sub>g</sub>ame-<sub>p</sub>erfect e<sub>q</sub>uilibrium) as constraints on t<sup>h</sup>e joint strategy pro<sup>fil</sup>e, using payo<sup>f</sup>s <sup>d</sup>erive<sup>d</sup> <sup>f</sup>rom eac<sup>h</sup> agent<sup>’</sup>s pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity o<sup>f</sup> satis<sup>f</sup>ying its tempora<sup>l</sup> o<sup>b</sup>jective. In particu<sup>l</sup>ar, we formulate <sub>p</sub>er-a<sub>g</sub>ent rational s<sub>y</sub>nthesis as findin<sub>g</sub> a strate<sub>gy p</sub>rofile – if one exists – such that (i) the desi<sub>g</sub>nated a<sub>g</sub>ent’s <sub>p</sub>robabilistic tem<sub>p</sub>oral constraint is satisfied, and (ii) the <sub>p</sub>rofile re<sub>p</sub>resents a stabl<sub>e so</sub>l<sub>u</sub>ti<sub>on un</sub>d<sub>er a c</sub>h<sub>osen so</sub>l<sub>u</sub>ti<sub>on concep</sub>t<sub>.</sub> W<sub>e g</sub>i<sub>ve con</sub>diti<sub>ona</sub>l convergence guarantees for tabular Nash-Q learning, certify sati<sub>s</sub>f<sub>ac</sub>ti<sub>on w</sub>ith <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub> fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e con</sub>fid<sub>ence</sub> b<sub>oun</sub>d<sub>s, an</sub>d assess strate<sub>g</sub>ic stabilit<sub>y</sub> with a<sub>pp</sub>roximate best-res<sub>p</sub>onse checkin<sub>g</sub>. Our approach operates in a stochastic Multi-Agent Reinforcement Learnin<sub>g</sub> (MARL) environment, where model-free learnin<sub>g</sub> al<sub>g</sub>orithms attem<sub>p</sub>t to attain extremal satisfaction <sub>p</sub>robabilities for tempora<sup>l</sup> o<sup>b</sup>jectives automatica<sup>ll</sup>y compi<sup>l</sup>e<sup>d</sup> into rewar<sup>d</sup> automata. Our a<sub>pp</sub>roach is em<sub>p</sub>iricall<sub>y</sub> validated usin<sub>g</sub> 8 <sub>g</sub>eneral-sum concurrent stochastic <sub>g</sub>ames.

## Ke<sub>y</sub>words

M<sub>u</sub>lti-a<sub>g</sub>ent Reinforcement Learnin<sub>g,</sub> General-S<sub>u</sub>m Games<sub>,</sub> Rational Learning, Pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic Tempora<sup>l</sup> O<sup>b</sup>jectives

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Strate<sub>g</sub>ic reasonin<sub>g</sub> in multi-a<sub>g</sub>ent s<sub>y</sub>stems (MAS) hin<sub>g</sub>es on determinin<sub>g</sub> what outcomes <sub>g</sub>rou<sub>p</sub>s of a<sub>g</sub>ents (termed coalitions) can brin<sub>g</sub> about throu<sub>g</sub>h their interactions. In man<sub>y</sub> multi-a<sub>g</sub>ent settin<sub>g</sub>s<sub>,</sub> es<sub>p</sub>eciall<sub>y</sub> safet<sub>y</sub>-critical s<sub>y</sub>stems<sub>, p</sub>ro<sub>p</sub>erties of interest <sub>go</sub> b<sub>eyon</sub>d <sub>p</sub>l<sub>a</sub>i<sub>n</sub> <sub>rewar</sub>d <sub>s</sub>i<sub>gna</sub>l<sub>s</sub> <sub>an</sub>d i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> t<sub>empora</sub>ll<sub>y</sub> <sub>ex</sub>t<sub>en</sub>d<sub>e</sub>d <sub>g</sub>oals such as reachabilit<sub>y</sub>/liveness (e.<sub>g</sub>., eventuall<sub>y</sub> reachin<sub>g</sub> a tar-<sub>g</sub>et re<sub>g</sub>ion or com<sub>p</sub>letin<sub>g</sub> a mission), or safet<sub>y</sub> invariants (e.<sub>g</sub>., alwa<sub>y</sub>s avoid unsafe re<sub>g</sub>ions, or never violate constraints) [4, 23]. Similar objectives are naturall<sub>y</sub> ca<sub>p</sub>tured b<sub>y</sub> tem<sub>p</sub>oral lo<sub>g</sub>ics [38] which <sub>p</sub>rovide <sub>p</sub>recise lan<sub>g</sub>ua<sub>g</sub>es and semantics to s<sub>p</sub>ecif<sub>y</sub> and verif<sub>y</sub> such <sub>p</sub>ro<sub>p</sub>erties. Lan<sub>g</sub>ua<sub>g</sub>es relevant to s<sub>p</sub>ecification of strate-<sub>g</sub>ic <sub>p</sub>ro<sub>p</sub>erties include Alternatin<sub>g</sub>-Time Tem<sub>p</sub>oral Lo<sub>g</sub>ic (atl [3]), D<sub>y</sub>namic Coalition Lo<sub>g</sub>ic [22], Probabilistic Alternatin<sub>g</sub>-� Calculus [47], Game Descri<sub>p</sub>tion Lo<sub>g</sub>ic (gdl [16]), and See-To-It-That (stit [7]). However, such lo<sub>g</sub>ics t<sub>yp</sub>icall<sub>y</sub> resolve non-determinism b<sub>y</sub> restrictin<sub>g</sub> to coalitional modalities which aim to universall<sub>y</sub> and unilaterall<sub>y</sub> (inevitabl<sub>y</sub>) force a <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> to be true.

On the other hand, rational s<sub>y</sub>nthesis [20, 52] aims to find strate-<sub>g</sub>i<sub>es</sub> th<sub>a</sub>t <sub>are</sub> <sub>se</sub>lf<sub>-en</sub>f<sub>orc</sub>i<sub>ng</sub> th<sub>roug</sub>h <sub>s</sub>t<sub>a</sub>bl<sub>e</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>a,</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>q</sub>uantif<sub>y</sub>in<sub>g</sub> a<sub>g</sub>ainst all <sub>p</sub>ossible behaviour (includin<sub>g</sub> irrational behaviour) by other agents who have their own objectives. Therefore, rational s<sub>y</sub>nthesis is t<sub>yp</sub>icall<sub>y</sub> less conservative than adversarial s<sub>y</sub>nthesis.

Settin<sub>g</sub>s we consider can be re<sub>p</sub>resented formall<sub>y</sub> as (<sub>g</sub>eneralsum) stochastic <sub>g</sub>ames, which model multi<sub>p</sub>le decision-makin<sub>g</sub> <sub>age</sub>nt<sub>s</sub> th<sub>a</sub>t int<sub>e</sub>r<sub>ac</sub>t <sub>ove</sub>r tim<sub>e</sub> in <sub>a</sub> d<sub>y</sub>n<sub>a</sub>mi<sub>c e</sub>n<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt <sub>w</sub>ith <sub>p</sub>robabilistic behaviours. Each a<sub>g</sub>ent re<sub>p</sub>eatedl<sub>y</sub> selects an action<sub>,</sub> <sub>an</sub>d th<sub>e com</sub>bi<sub>na</sub>ti<sub>on o</sub>f <sub>agen</sub>t<sub>s</sub>’ <sub>non-</sub>d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c c</sub>h<sub>o</sub>i<sub>ces an</sub>d th<sub>e</sub> en<sub>v</sub>ironment’s <sub>p</sub>robabilistic transitions determines the e<sub>v</sub>ol<sub>u</sub>tion of the <sub>g</sub>ame. Stochastic <sub>g</sub>ames have been a<sub>pp</sub>lied in domains as diverse as c<sub>y</sub>bersecurit<sub>y</sub> [60], epidemics [29], auctions [39], and routing in stochastic trafic networks [59].

We use an ex<sub>p</sub>ressive s<sub>p</sub>ecification st<sub>y</sub>le in which each a<sub>g</sub>ent has its own pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic tempora<sup>l</sup> o<sup>b</sup>jective as a <sup>l</sup>ogica<sup>l f</sup>ormu<sup>l</sup>a inter-<sub>pre</sub>t<sub>e</sub>d <sub>on</sub> fi<sub>n</sub>it<sub>e</sub> t<sub>races.</sub> Fi<sub>n</sub>it<sub>e</sub> t<sub>races</sub> h<sub>ave a na</sub>t<sub>ura</sub>l <sub>correspon</sub>d<sub>ence</sub> with reinforcement learnin<sub>g</sub> (RL) as tasks are often trained and executed e<sub>p</sub>isodicall<sub>y</sub>. Rationalit<sub>y</sub> is ca<sub>p</sub>tured b<sub>y</sub> im<sub>p</sub>osin<sub>g</sub> that the <sup>l</sup>earne<sup>d</sup> joint strategy pro<sup>fil</sup>e satis<sup>fi</sup>es a speci<sup>fi</sup>c so<sup>l</sup>ution concept (e.<sub>g</sub>., (�-)Nash e<sub>q</sub>uilibrium, sub<sub>g</sub>ame-<sub>p</sub>erfect e<sub>q</sub>uilibrium), where <sub>eac</sub>h <sub>agen</sub>t’<sub>s</sub> <sub>payo</sub>f <sub>re</sub>fl<sub>ec</sub>t<sub>s</sub> th<sub>e</sub> <sub>pre</sub>f<sub>erre</sub>d di<sub>rec</sub>ti<sub>on</sub> <sub>o</sub>f it<sub>s</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> <sub>p</sub>robabilit<sub>y</sub>. Our a<sub>pp</sub>roach em<sub>p</sub>hasises learnin<sub>g</sub> a sin<sub>g</sub>le committed strate<sub>gy</sub> <sub>p</sub>rofile<sub>,</sub> allowin<sub>g</sub> tractabilit<sub>y</sub> b<sub>y</sub> not re<sub>q</sub>uirin<sub>g</sub> an ex<sub>p</sub>licit t<sub>rans</sub>iti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l <sub>o</sub>f th<sub>e</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>game</sub> <sub>w</sub>hi<sub>c</sub>h <sub>gener-</sub> <sub>a</sub>ll<sub>y</sub> <sub>grows</sub> <sub>com</sub>bi<sub>na</sub>t<sub>or</sub>i<sub>a</sub>ll<sub>y</sub> <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>agen</sub>t<sub>s</sub> i<sub>nvo</sub>l<sub>ve</sub>d<sub>.</sub>

Contributions. Our contributions are as follows: (i) We formulate a <sup>l</sup>earning-<sup>b</sup>ase<sup>d</sup> rationa<sup>l</sup>-synt<sup>h</sup>esis o<sup>b</sup>jective <sup>f</sup>or <sup>fi</sup>nite-<sup>h</sup>orizon stoc<sup>h</sup>astic games in w<sup>h</sup>ic<sup>h</sup> eac<sup>h</sup> agent <sup>h</sup>as a pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic ltl<sub>�</sub> o<sup>b</sup>jective <sub>an</sub>d <sub>u</sub>tiliti<sub>es enco</sub>d<sub>e</sub> th<sub>e pre</sub>f<sub>erre</sub>d di<sub>rec</sub>ti<sub>on o</sub>f <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on pro</sub>b<sub>a</sub>bil<sub>-</sub> ities; (ii) We com<sub>p</sub>ile <sub>p</sub>er-a<sub>g</sub>ent ltl<sub>�</sub> objectives into deterministic <sub>rewar</sub>d <sub>au</sub>t<sub>oma</sub>t<sub>a an</sub>d <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t t<sub>erm</sub>i<sub>na</sub>l <sub>au</sub>t<sub>oma</sub>t<sub>on rewar</sub>d<sub>s</sub> h<sub>ave</sub> ex<sub>p</sub>ectation e<sub>q</sub>ual to satisfaction <sub>p</sub>robabilit<sub>y</sub>; (iii) We <sub>p</sub>rovide theoretical anal<sub>y</sub>sis for the reward-automaton encodin<sub>g,</sub> discountin<sub>g</sub> <sub>e</sub>f<sub>ec</sub>t<sub>s,</sub> fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> <sub>cer</sub>tifi<sub>ca</sub>ti<sub>on,</sub> <sub>an</sub>d fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon</sub> e<sub>q</sub>uilibrium existence and conver<sub>g</sub>ence; (iv) We evaluate satisfaction and strate<sub>g</sub>ic stabilit<sub>y</sub> on 8 benchmark environment instances.

F<sub>o</sub>r r<sub>easo</sub>n<sub>s o</sub>f <sub>space, p</sub>r<sub>oo</sub>f<sub>s a</sub>r<sub>e</sub> d<sub>e</sub>f<sub>e</sub>rr<sub>e</sub>d t<sub>o</sub> A<sub>ppe</sub>ndix D<sub>.</sub> A hi<sub>g</sub>hlevel overview for our a<sub>pp</sub>roach is <sub>p</sub>resented in A<sub>pp</sub>endix B.

## 2 P<sub>re</sub>li<sub>m</sub>i<sub>nar</sub>i<sub>es</sub>

Multi-agent Reinforcement Learning. We model multi-agent RL <sub>p</sub>roblems as an �-a<sub>g</sub>ent concurrent stochastic <sub>g</sub>ame (also known as a Markov <sub>g</sub>ame [36]) $\mathcal { M } = ( S , s _ { 0 } , \mathcal { A } ^ { 1 } , \ldots , \mathcal { A } ^ { n } , d ^ { 1 } , \ldots , d ^ { n }$ , $\mathcal { P } , \mathcal { R } ^ { 1 } , \ldots , \mathcal { R } ^ { n } , \gamma )$ , where � is the finite state s<sub>p</sub>ace and $s _ { 0 } \in \mathcal { S }$ i<sub>s</sub> the initial state; �<sup>�</sup> is the finite action s<sub>p</sub>ace of a<sub>g</sub>ent $i \in [ 1 , \ldots , n ] ;$

$d ^ { i } : S  2 ^ { { \mathcal { A } } ^ { i } } \setminus \{ \emptyset \}$ is the protocol (action-availabilit<sub>y</sub>) function of a<sub>g</sub>ent �, where $d ^ { i } ( s )$ denotes the set of le<sub>g</sub>al actions for a<sub>g</sub>ent � in state �; the set of le<sub>g</sub>al joint actions at � is $\begin{array} { r } { \bar { d } ( s ) = \prod _ { i = 1 } ^ { n } { d ^ { i } } \bar { ( s ) } \subseteq \mathcal { A } ; } \end{array}$ $\mathcal { P } : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$ is the (<sub>p</sub>artial) transition function (defined onl<sub>y</sub> for joint actions � $\in d ( s ) )$ <sub>,</sub> <sub>w</sub>ith $\Delta ( S )$ b<sub>e</sub>i<sub>ng</sub> th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s ove</sub>r $S ; \mathcal { R } ^ { i } : S \times \mathcal { A } $ R is the (Markovian) reward function of agent �; and � is the discount factor.

A (stochastic) polic<sub>y</sub> $\pi ^ { i }$ (also known as a strate<sub>gy</sub>) of a<sub>g</sub>ent � is a f<sub>unc</sub>ti<sub>on</sub> $\pi ^ { i } : \bar { \cal S }  \bar { \Delta } ( { \mathcal { A } } ^ { i } )$ with supp $\pi ^ { i } ( \cdot \mid s ) \subseteq { \overline { { d } } } ^ { i } ( s )$ for ever<sub>y</sub> �. T<sup>h</sup>en, t<sup>h</sup>e joint action space o<sup>f</sup> agents is $\textstyle A = \prod _ { i = 1 } ^ { n } A ^ { i }$ , an<sup>d</sup> t<sup>h</sup>e joint <sub>p</sub>olic<sub>y</sub> of other a<sub>g</sub>ents is $\begin{array} { r } { \pi ^ { - i } ( a ^ { - i } | s ) = \prod _ { i \neq i } \bar { \pi } ^ { j } \bar { ( } a ^ { j } | s ) } \end{array}$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $a ^ { - i }$ is the joint action exce<sub>p</sub>t a<sub>g</sub>ent �’s action. Each a<sub>g</sub>ent � has their own <sub>goa</sub>l t<sub>o max</sub>i<sub>m</sub>i<sub>se</sub> th<sub>e</sub>i<sub>r expec</sub>t<sub>e</sub>d <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>scoun</sub>t<sub>e</sub>d <sub>rewar</sub>d<sub>s,</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as:</sub>

$$
\begin{array} { r l } { \mathbb { E } } & { { } [ \displaystyle \sum _ { \substack { s t + 1 \sim \mathcal { P } ( \cdot | s _ { t } , a _ { t } ^ { i } , a _ { t } ^ { - i } ) , } } ^ { \infty } [ \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathcal { R } ^ { i } \big ( s _ { t } , a _ { t } ^ { i } , a _ { t } ^ { - i } \big ) ] } \\ { a _ { t } ^ { i } \sim \pi ^ { i } ( \cdot | s _ { t } ) , a _ { t } ^ { - i } \sim \pi ^ { - i } ( \cdot | s _ { t } ) } \end{array}\tag{1}
$$

Th<sub>e</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>se</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>game,</sub> f<sub>rom</sub> th<sub>e</sub> <sub>perspec</sub>ti<sub>ve</sub> <sub>o</sub>f <sub>eac</sub>h a<sub>g</sub>ent, is non-stationary, as t<sup>h</sup>e ot<sup>h</sup>er a<sub>g</sub>ents concurrent<sup>l</sup><sub>y</sub> u<sub>p</sub><sup>d</sup>ate their <sub>p</sub>olicies. General-sum stochastic <sub>g</sub>ames have com<sub>p</sub>utational solution conce<sub>p</sub>ts which assess stabilit<sub>y</sub> under e<sub>q</sub>uilibria<sub>,</sub> since the<sub>y</sub> account for the actions of the remainin<sub>g</sub> a<sub>g</sub>ents.

A l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>game</sub> <sub>s</sub>i<sub>mp</sub>l<sub>y</sub> <sub>ex</sub>t<sub>en</sub>d<sub>s</sub> <sub>a</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>game</sub> <sub>w</sub>ith <sub>a</sub> l<sub>a</sub>b<sub>e</sub>lli<sub>ng</sub> f<sub>unc</sub>ti<sub>on</sub> $\mathcal { L } : \mathcal { S }  2 ^ { \mathsf { A } \bar { \mathsf { p } } }$ which ma<sub>p</sub>s each state � to a set $\mathcal { L } ( s ) \subseteq \mathsf { A p }$ of atoms, which denotes which atoms are true at �. We <sup>d</sup>enote a joint set o<sup>f</sup> strategies <sup>f</sup>or a coa<sup>l</sup>ition $C \subseteq [ 1 , \ldots , n ]$ as $\sigma _ { C ; }$ <sub>an</sub>d f<sub>or</sub> <sub>a</sub>ll <sub>agen</sub>t<sub>s</sub> i<sub>n</sub> th<sub>e</sub> <sub>concurren</sub>t <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>game,</sub> <sub>we</sub> d<sub>eno</sub>t<sub>e</sub> the strategy profile ${ \vec { \sigma } } \ { \stackrel { \mathrm { d e f } } { = } } \ \left( \sigma _ { 1 } , \ldots \ldots , \sigma _ { n } \right)$ <sub>as</sub> th<sub>e se</sub>t <sub>compose</sub>d <sub>o</sub>f <sub>eac</sub>h <sup>a</sup>g<sup>ent’s</sup> <sup>strate</sup>gy<sup>.</sup>

We denote a finite path as $\pi = s _ { 0 } a _ { 0 } s _ { 1 } \cdot \cdot \cdot a _ { T - 1 } s _ { T }$ <sub>, w</sub>h<sub>ere</sub> $a _ { t } \in$ $d ( s _ { t } )$ <sub>an</sub>d $\mathcal { P } ( s _ { t + 1 } \mid s _ { t } , a _ { t } ) > 0$ <sup>f</sup>or ever<sub>y</sub> $0 \leq t < T$ <sub>.</sub> Th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub> � bounds the number of transitions, so $T \leq H$ <sub>an</sub>d th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d l<sub>a</sub>b<sub>e</sub>l t<sub>race</sub> h<sub>as</sub> $T + 1$ <sub>pos</sub>iti<sub>o</sub>n<sub>s.</sub> L<sub>e</sub>t $\mathsf { F P a t h } _ { \mathcal { M } , s }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e se</sub>t <sub>o</sub>ffi<sub>n</sub>it<sub>e</sub> <sub>p</sub>aths from �. For a strate<sub>gy</sub> <sub>p</sub>rofile ${ \vec { \sigma } } ,$ l<sub>e</sub>t $\mathrm { P r } _ { { \mathcal M } , s } ^ { \vec { \sigma } }$ b<sub>e</sub> th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>measure</sub> <sub>on</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> fi<sub>n</sub>it<sub>e</sub> <sub>pa</sub>th<sub>s</sub> $\mathsf { F P a t h } _ { \mathcal { M } , s } ^ { \mathrm { c o m p } }$ <sub>.</sub> W<sub>e re</sub>f<sub>er</sub> t<sub>o</sub> [6] for the standard construction of the <sub>p</sub>robabilit<sub>y</sub> measure $\mathrm { P r } _ { { \mathcal M } , s } ^ { \vec { \sigma } }$ i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> <sub>a</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>pro</sub>fil<sub>e.</sub> W<sub>e</sub> l<sub>earn</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>pro</sub>fil<sub>e</sub> using IPPO [18, 44], IQL [49], Nash-Q Learning (NQL) [28], an<sup>d</sup> PR2-IQL/PR2-IPPO variants [57] (further details in A<sub>pp</sub>endix D.2).

Probabilistic Linear Tem<sub>p</sub>oral Lo<sub>g</sub>ic on Finite Traces. We use Prob-<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> $\mathbf { L T L } _ { f } \ \left( \mathbf { P L T L } _ { f } \right) \ \left[ 3 7 \right]$ to s<sub>p</sub>ecif<sub>y</sub> each a<sub>g</sub>ent’s <sub>p</sub>robabilistic tempora<sup>l</sup> o<sup>b</sup>jective on <sup>fi</sup>nite traces, matc<sup>h</sup>ing t<sup>h</sup>e episo<sup>d</sup>ic structure of RL tasks. Each com<sub>p</sub>leted e<sub>p</sub>isode su<sub>pp</sub>lies a finite se<sub>q</sub>uence of <sub>s</sub>t<sub>a</sub>t<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s on w</sub>hi<sub>c</sub>h <sub>a</sub> t<sub>empora</sub>l <sub>goa</sub>l i<sub>s eva</sub>l<sub>ua</sub>t<sub>e</sub>d<sub>:</sub> f<sub>or examp</sub>l<sub>e,</sub> <sub>reac</sub>hi<sub>ng</sub> <sub>a</sub> t<sub>arge</sub>t b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> <sub>ep</sub>i<sub>so</sub>d<sub>e</sub> <sub>en</sub>d<sub>s</sub> <sub>w</sub>hil<sub>e</sub> <sub>avo</sub>idi<sub>ng</sub> <sub>co</sub>lli<sub>s</sub>i<sub>ons</sub> throu<sub>g</sub>hout. The <sub>p</sub>robabilistic re<sub>q</sub>uirement then constrains how <sub>o</sub>ft<sub>en</sub> th<sub>a</sub>t <sub>goa</sub>l i<sub>s</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>e</sub>d <sub>un</sub>d<sub>er</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>pro</sub>fil<sub>e.</sub> E<sub>qu</sub>ilib<sub>-</sub> rium constraints ca<sub>p</sub>ture rationalit<sub>y</sub> throu<sub>g</sub>h utilities reflectin<sub>g</sub> each <sub>agen</sub>t’<sub>s pre</sub>f<sub>erre</sub>d di<sub>rec</sub>ti<sub>on o</sub>f <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on.</sub> Th<sub>e grammar o</sub>f P<sub>ro</sub>b<sub>a-</sub> bili<sub>s</sub>ti<sub>c</sub> Lin<sub>ea</sub>r T<sub>e</sub>m<sub>po</sub>r<sub>a</sub>l L<sub>og</sub>i<sub>c o</sub>n Finit<sub>e</sub> Tr<sub>aces</sub> $\left( \mathbf { P I T L } _ { f } \right)$ i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

Definition 1 (pltl<sub>�</sub> Formula). $P L T L f$ formulas are constructed inductively from the set of atoms $\mathsf { A p }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $p \in \mathsf { A p }$ , ⊤ denotes true and ⊥ denotes false, $d \in \mathsf { [ 0 , 1 ] }$ i<sub>s</sub> <sub>a</sub> <sub>ra</sub>ti<sub>ona</sub>l <sub>cons</sub>t<sub>an</sub>t<sub>,</sub> <sub>an</sub>d $\bowtie \in \{ < , \leq , > , \geq \}$ <sub>.</sub> F<sub>orma</sub>ll<sub>y,</sub> th<sub>e</sub> <sub>grammar</sub> i<sub>s:</sub>

$$
\begin{array} { r l } & { \varphi : : = \top \mid p \mid \neg \varphi \mid \varphi \land \varphi \mid \textsf { X } \varphi \mid \varphi _ { 1 } \textsf { U } \varphi _ { 2 } } \\ & { \psi : : = \mathsf { P } _ { \ J \ J } [ \varphi ] } \end{array}
$$

X denotes tem<sub>p</sub>oral Next and U denotes tem<sub>p</sub>oral Until. The remainin<sub>g</sub> Boolean and tem<sub>p</sub>oral o<sub>p</sub>erators are s<sub>y</sub>ntactic su<sub>g</sub>ar defined as usual: Eventuall<sub>y</sub> F as $\mathsf { F } \varphi \equiv \mathsf { T } \mathsf { U } \varphi$ and Alwa<sub>y</sub>s G as ${ \sf G } \varphi \equiv \neg { \sf F } \neg \varphi .$ In multi-a<sub>g</sub>ent s<sub>y</sub>stems<sub>,</sub> a<sub>g</sub>ents with no s<sub>p</sub>ecific o<sup>b</sup>jectives can <sup>b</sup>e inc<sup>l</sup>u<sup>d</sup>e<sup>d</sup> in a resi<sup>d</sup>ua<sup>l</sup> coa<sup>l</sup>ition wit<sup>h</sup> a trivia<sup>l</sup> o<sup>b</sup>jective a<sup>l</sup>ways ac<sup>h</sup>ieva<sup>bl</sup>e wit<sup>h</sup> pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity o<sup>f</sup> 1 $( \mathrm { e . g . , \top } )$ <sub>.</sub> Th<sub>e</sub> f<sub>u</sub>ll finite-trace semantics are <sub>g</sub>iven in A<sub>pp</sub>endix A. We <sub>p</sub>rovide the <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> <sub>c</sub>l<sub>ause</sub> <sub>o</sub>f $\mathbf { P L T L } _ { f }$ as:

$$
\begin{array} { c } { { \mathcal { M } , s , \vec { \sigma } \equiv \mathsf { P } _ { \bowtie d } [ \varphi ] \quad \Longleftrightarrow } } \\ { { \vec { \sigma } } _ { \tiny { \mathscr { M } , s } } \left( \{ \pi \in { \mathsf { F P a t h } } _ { \mathscr { M } , s } ^ { \mathrm { c o m p } } \mid \mathcal { M } , \pi \mid = \varphi \} \right) \bowtie d . }  \end{array}
$$

Rational S<sub>y</sub>nthesis. Classical lo<sub>g</sub>ics with strate<sub>g</sub>ic coalition modalities (such as atl [3] and patl [13]) assert tem<sub>p</sub>oral <sub>g</sub>oals a<sub>g</sub>ainst <sub>a</sub>ll b<sub>e</sub>h<sub>av</sub>i<sub>ours o</sub>f th<sub>e o</sub>th<sub>er agen</sub>t<sub>s.</sub> Thi<sub>s</sub> t<sub>rea</sub>t<sub>s o</sub>th<sub>er coa</sub>liti<sub>ons as</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> <sub>a</sub>d<sub>versar</sub>i<sub>a</sub>l<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>can</sub> l<sub>ea</sub>d t<sub>o</sub> <sub>over</sub>l<sub>y</sub> <sub>conserva</sub>ti<sub>ve</sub> <sub>so</sub>l<sub>u-</sub> tions. In settings w<sup>h</sup>ere ot<sup>h</sup>er agents <sup>h</sup>ave t<sup>h</sup>eir own o<sup>b</sup>jectives, it i<sub>s</sub> <sub>more</sub> <sub>na</sub>t<sub>ura</sub>l t<sub>o</sub> <sub>assume</sub> th<sub>ey</sub> <sub>ac</sub>t <sub>ra</sub>ti<sub>ona</sub>ll<sub>y</sub> <sub>an</sub>d <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t th<sub>e</sub>i<sub>r</sub> behaviour to strate<sub>gy</sub> <sub>p</sub>rofiles satisf<sub>y</sub>in<sub>g</sub> a solution conce<sub>p</sub>t (e.<sub>g</sub>., (stron<sub>g</sub>) Nash e<sub>q</sub>uilibrium, sub<sub>g</sub>ame-<sub>p</sub>erfect e<sub>q</sub>uilibrium) [14, 20]. We <sub>w</sub>rite $\vec { \sigma } \in \mathsf { S o l } _ { \zeta } ( s ; \vec { u } )$ t<sub>o</sub> d<sub>eno</sub>t<sub>e</sub> th<sub>a</sub>t $\vec { \sigma }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> <sub>concep</sub>t � at state � for the <sub>p</sub>a<sub>y</sub>of vector ⃗�, where the utilities encode each <sub>agen</sub>t’<sub>s pre</sub>f<sub>erre</sub>d di<sub>rec</sub>ti<sub>on o</sub>f <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on.</sub>

Definition 2 (Satisfaction Probabilities and Utilities). For a<sub>g</sub>ent � wit<sup>h</sup> re<sub>q</sub>uirement $\psi _ { i } = \mathsf { P } _ { \bowtie i d _ { i } } [ \varphi _ { i } ]$ , define $p _ { i } ( s , { \vec { \sigma } } ) \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { P r } _ { { \mathcal { M } } , s } ^ { \vec { \sigma } } ( \varphi _ { i } )$ <sub>an</sub>d $f _ { i } : \bar { [ 0 , 1 ] }  [ 0 , 1 ]$ <sup>b</sup><sub>y</sub>

$$
f _ { i } ( p ) \stackrel { \mathrm { d e f } } { = } \left\{ \begin{array} { l l } { p , } & { \bowtie _ { i } \in \{ > , \geq \} , } \\ { 1 - p , } & { \bowtie _ { i } \in \{ < , \leq \} . } \end{array} \right.
$$

Th<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>s</sub> $u _ { i } ( s , \vec { \sigma } ) \stackrel { \mathrm { d e f } } { = } f _ { i } ( p _ { i } ( s , \vec { \sigma } ) )$ . Write $\vec { u } ( s , \vec { \sigma } ) \stackrel { \mathrm { d e f } } { = }$ $\big ( u _ { i } ( s , \vec { \sigma } ) \big ) _ { i = 1 } ^ { n } .$

D<sub>e</sub>finiti<sub>o</sub>n 2 m<sub>a</sub>k<sub>es u</sub>tilit<sub>y</sub> m<sub>a</sub>ximi<sub>sa</sub>ti<sub>o</sub>n f<sub>avou</sub>r l<sub>a</sub>r<sub>ge</sub>r <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>o</sub>n <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> f<sub>or</sub> l<sub>ower-</sub>b<sub>oun</sub>d <sub>requ</sub>i<sub>remen</sub>t<sub>s</sub> <sub>an</sub>d <sub>sma</sub>ll<sub>er</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>-</sub> ties for u<sub>pp</sub>er-bound re<sub>q</sub>uirements. We tar<sub>g</sub>et the followin<sub>g</sub> notion of �-Nash e<sub>q</sub>uilibrium:

Definition 3 (�-Nash E<sub>q</sub>uilibrium). A strate<sub>gy</sub> profile $\vec { \sigma } ~ = ~ \left( \sigma _ { 1 } , \ldots , \sigma _ { n } \right)$ is an �-Nash e<sub>q</sub>uilibrium $( \varepsilon - N E )$ at state � if, for every agent � and every alternative strategy $\sigma _ { i } ^ { \prime } ,$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $u _ { i } ( s , ( \sigma _ { i } ^ { \prime } , \sigma _ { - i } ) ) \le u _ { i } ( s , \vec { \sigma } ) + \varepsilon$ , where $\sigma _ { - i }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> ofall agents except �. For $\varepsilon = 0 , \vec { \sigma }$ is a Nash e<sub>q</sub>uilibrium (NE).

The lar<sub>g</sub>est <sub>g</sub>ain an<sub>y</sub> sin<sub>g</sub>le a<sub>g</sub>ent can obtain b<sub>y</sub> deviatin<sub>g</sub> unilat-<sub>era</sub>ll<sub>y</sub> <sub>measures</sub> h<sub>ow</sub> f<sub>ar</sub> <sub>a</sub> <sub>pro</sub>fil<sub>e</sub> i<sub>s</sub> f<sub>rom</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>um:</sub>

Definition 4 (Exploitability). The exploitability of a strategy profile ⃗� at state � is Exploit $( s , { \vec { \sigma } } ) \ { \stackrel { \mathrm { d e f } } { = } }$ max<sub>�</sub> $\mathrm { s u p } _ { \sigma _ { i } ^ { \prime } } \left[ u _ { i } \left( s , ( \sigma _ { i } ^ { \prime } , \sigma _ { - i } ) \right) - \right.$ $u _ { i } ( s , \vec { \sigma } ) ]$

B<sub>y</sub> Definitions 3 and 4, ⃗� is an $\varepsilon { - } \mathsf { N E }$ at � if and onl<sub>y</sub> if Exploit $( s , { \vec { \sigma } } ) \leq \varepsilon$

P<sub>ro</sub>bl<sub>em</sub> St<sub>a</sub>t<sub>emen</sub>t<sub>.</sub> W<sub>e cons</sub>id<sub>er a</sub> fi<sub>n</sub>it<sub>e</sub> l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>concurren</sub>t <sub>s</sub>t<sub>o-</sub> chastic <sub>g</sub>ame $\mathcal { M }$ <sub>w</sub>ith initi<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { 0 } ,$ <sub>,</sub> fi<sub>n</sub>it<sub>e</sub> h<sub>or</sub>i<sub>zon</sub> $H ,$ <sub>an</sub>d <sub>a so-</sub> lution conce<sub>p</sub>t $\zeta ,$ <sub>suc</sub>h <sub>as</sub> $\varepsilon { - } \mathsf { N E }$ <sub>.</sub> Th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n <sub>p</sub>r<sub>o</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>a</sub>r<sub>e</sub> unknown<sub>,</sub> and strate<sub>g</sub>ies are learned throu<sub>g</sub>h e<sub>p</sub>isodic interaction with the environment. Each a<sub>g</sub>ent $i \in \{ 1 , \ldots , n \}$ h<sub>as</sub> <sub>an</sub> <sub>ltl� pa</sub>th o<sup>b</sup>jective $\varphi _ { i }$ and a <sub>p</sub>robabilistic re<sub>q</sub>uirement $\psi _ { i } = \mathsf { P } _ { \bowtie _ { i } d _ { i } } \big [ \varphi _ { i } \big ]$ . Let $\Sigma _ { i }$ <sup>d</sup>enote a<sub>g</sub>ent $i \ ' s$ admissible strate<sub>gy</sub> s<sub>p</sub>ace and $\begin{array} { r } { \Sigma \ { \stackrel { \mathrm { d e f } } { = } } \ \prod _ { i = 1 } ^ { n } \Sigma _ { i } } \end{array}$ Each a<sub>g</sub>ent seeks an extremal satisfaction <sub>p</sub>robabilit<sub>y</sub> at e<sub>q</sub>uilibrium: <sub>a max</sub>i<sub>mum</sub> f<sub>or a</sub> l<sub>ower-</sub>b<sub>oun</sub>d <sub>requ</sub>i<sub>remen</sub>t <sub>an</sub>d <sub>a m</sub>i<sub>n</sub>i<sub>mum</sub> f<sub>or an</sub> <sub>upper-</sub>b<sub>oun</sub>d <sub>requ</sub>i<sub>remen</sub>t<sub>, w</sub>hil<sub>e mee</sub>ti<sub>ng</sub> it<sub>s</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub>

Definition 5 (Equilibrium Profiles). The set of admissible profiles satisfying solution concept � for the utilities in Definition 2 is

$$
\mathcal { E } _ { \zeta } \overset { \mathrm { d e f } } { = } \left\{ \vec { \sigma } \in \Sigma \ | \ \vec { \sigma } \in { \sf S o l } _ { \zeta } ( s _ { 0 } ; \vec { u } ) \right\} .
$$

Definition 5 ca<sub>p</sub>tures stabilit<sub>y</sub> under the a<sub>g</sub>ents’ individual <sub>p</sub>a<sub>y</sub>of maximisin<sub>g</sub> <sub>p</sub>references. For a<sub>g</sub>ent �, the acce<sub>p</sub>table e<sub>q</sub>uilibrium <sub>pro</sub>fil<sub>es</sub> <sub>are</sub> $\mathcal { F } _ { \zeta } ^ { i } \stackrel { \mathrm { d e f } } { = } \{ \vec { \sigma } \in \mathcal { E } _ { \zeta } \mid p _ { i } ( s _ { 0 } , \vec { \sigma } ) \boxtimes _ { i } d _ { i } \}$

Definition 6 (Rational Strate<sub>gy</sub> S<sub>y</sub>nthesis). Given $( \mathcal { M } , s _ { 0 } , H ,$ $( \psi _ { j } ) _ { j = 1 } ^ { n } , \zeta )$ , a<sup>d</sup>missi<sup>bl</sup>e strate<sub>gy</sub> s<sub>p</sub>aces $\textstyle { \bigl ( } \sum _ { j } { \bigr ) } _ { j = 1 } ^ { n }$ , and an a<sub>g</sub>ent �, the rational strategy synthesis problem for agent � is to find a strategy profile $\vec { \sigma } ^ { \star } = ( { \sigma } _ { i } ^ { \star } , { \sigma } _ { - i } ^ { \star } ) \in \mathcal { F } _ { \zeta } ^ { i } ,$ , ifsuch a profile exists.

Definition 6 re<sub>q</sub>uires a<sub>g</sub>ent �’s threshold to hold at an e<sub>q</sub>uilibrium pro<sup>fil</sup>e. T<sup>h</sup>e ot<sup>h</sup>er agents<sup>’</sup> o<sup>b</sup>jectives <sup>d</sup>etermine t<sup>h</sup>eir strategic <sub>pre</sub>f<sub>erences,</sub> b<sub>u</sub>t th<sub>e</sub>i<sub>r</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s</sub> d<sub>o</sub> <sub>no</sub>t <sub>necessar</sub>il<sub>y</sub> <sub>nee</sub>d t<sub>o</sub> h<sub>o</sub>ld<sub>.</sub> Corres<sub>p</sub>ondin<sub>g</sub>l<sub>y,</sub> the feasibilit<sub>y</sub> <sub>q</sub>uestion is whether $\mathcal { F } _ { \zeta } ^ { i } \neq \emptyset _ { \vdots }$ , since th<sub>e</sub> <sub>ex</sub>i<sub>s</sub>t<sub>ence</sub> <sub>o</sub>f <sub>equ</sub>ilib<sub>r</sub>i<sub>a</sub> <sub>a</sub>l<sub>one</sub> i<sub>s</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>t f<sub>or</sub> th<sub>e</sub> <sub>ex</sub>i<sub>s</sub>t<sub>ence</sub> <sub>o</sub>f threshold-constrained Nash e<sub>q</sub>uilibria. Provin<sub>g</sub> infeasibilit<sub>y</sub> ma<sub>y</sub> re <sub>qu</sub>i<sub>re</sub> <sub>ru</sub>li<sub>ng</sub> <sub>ou</sub>t <sub>a</sub>ll <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> <sub>pro</sub>fil<sub>es,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> <sub>scope</sub> <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l<sub>-</sub>f<sub>ree</sub> <sub>ra</sub>ti<sub>ona</sub>l l<sub>earn</sub>i<sub>ng.</sub>

An equilibrium profile is additionally said to be jointly satisfying if it b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $\begin{array} { r } { \mathcal { F } _ { \zeta } \overset { \mathrm { d e f } } { = } \bigcap _ { i = 1 } ^ { n } \mathcal { F } _ { \zeta } ^ { i } } \end{array}$ <sub>,</sub> so that ever<sub>y</sub> a<sub>g</sub>ent’s <sub>p</sub>robabilistic th<sub>res</sub>h<sub>o</sub>ld h<sub>o</sub>ld<sub>s</sub> <sub>a</sub>t th<sub>e</sub> <sub>same</sub> <sub>pro</sub>fil<sub>e.</sub> E<sub>ac</sub>h <sub>agen</sub>t <sub>uses</sub> <sub>a</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>-</sub> constrained Nash e<sub>q</sub>uilibrium as its <sub>p</sub>rimar<sub>y</sub> solution conce<sub>p</sub>t<sub>,</sub> as oppose<sup>d</sup> to a joint<sup>l</sup>y satis<sup>f</sup>ying Nas<sup>h</sup> equi<sup>l</sup>i<sup>b</sup>rium.

We investi<sub>g</sub>ate model-free MARL a<sub>pp</sub>roaches in which a<sub>g</sub>ents concurrentl<sub>y</sub> learn strate<sub>g</sub>ies towards their res<sub>p</sub>ective s<sub>y</sub>nthesis o<sup>b</sup>jectives. For eac<sup>h</sup> <sup>l</sup>earne<sup>d</sup> pro<sup>fil</sup>e, we assess eac<sup>h</sup> agent<sup>’</sup>s pro<sup>b</sup>abili<sub>s</sub>ti<sub>c requ</sub>i<sub>remen</sub>t <sub>us</sub>i<sub>ng</sub> fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e con</sub>fid<sub>ence</sub> b<sub>oun</sub>d<sub>s an</sub>d th<sub>e</sub> <sub>p</sub>rofile’s strate<sub>g</sub>ic stabilit<sub>y</sub> usin<sub>g</sub> finite-bud<sub>g</sub>et a<sub>pp</sub>roximate bestres<sub>p</sub>onse searc<sup>h</sup>.

## 3 Encoding Quantitative Objectives

We out<sup>l</sup>ine <sup>h</sup>ow to trans<sup>l</sup>ate pltl<sub>�</sub> o<sup>b</sup>jectives into Mar<sup>k</sup>ovian reward si<sub>g</sub>nals for a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> standard (model-free) multi-a<sub>g</sub>ent RL algorithms, while preserving the single strategy profile witness re-<sub>q</sub>u<sup>i</sup>rement o<sup>f</sup> strate<sub>gy</sub> s<sub>y</sub>nt<sup>h</sup>es<sup>i</sup>s.

We compi<sup>l</sup>e eac<sup>h</sup> ltl<sub>�</sub> pat<sup>h</sup> o<sup>b</sup>jective into a <sup>d</sup>eterministic rewar<sup>d</sup> <sub>au</sub>t<sub>oma</sub>t<sub>on,</sub> th<sub>en</sub> t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> <sub>sync</sub>h<sub>ronous</sub> <sub>pro</sub>d<sub>uc</sub>t <sub>o</sub>f th<sub>ese</sub> <sub>au</sub>t<sub>oma</sub>t<sub>a</sub> with the stochastic <sub>g</sub>ame. Terminal acce<sub>p</sub>tance rewards recover satisfaction <sub>p</sub>robabilities (Theorem 1), and the <sub>p</sub>roduct makes the objectives Markovian while preserving payofs and equilibria (The orem 2). Discountin<sub>g</sub> results <sub>q</sub>uantif<sub>y</sub> how terminal rewards rel<sub>a</sub>t<sub>e</sub> t<sub>o</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>un</sub>d<sub>er</sub> fi<sub>xe</sub>d <sub>an</sub>d <sub>ear</sub>l<sub>y-</sub>t<sub>erm</sub>i<sub>na</sub>ti<sub>ng</sub> h<sub>or</sub>i<sub>zons.</sub> S<sub>amp</sub>li<sub>ng resu</sub>lt<sub>s prov</sub>id<sub>e</sub> fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e cer</sub>tifi<sub>ca</sub>t<sub>es</sub> f<sub>or</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s.</sub> Al<sub>gor</sub>ith<sub>m</sub> 1 <sub>summar</sub>i<sub>ses</sub> th<sub>e</sub> <sub>proce</sub>d<sub>ure.</sub>

Al<sub>gor</sub>ith<sub>m</sub> 1<sub>:</sub> L<sub>earn</sub>i<sub>ng-</sub>b<sub>ase</sub>d <sub>pltl� ra</sub>ti<sub>ona</sub>l <sub>syn</sub>th<sub>es</sub>i<sub>s un-</sub>   
d<sub>er un</sub>k<sub>nown</sub> d<sub>ynam</sub>i<sub>cs</sub>   
I<sub>npu</sub>t<sub>:</sub> L<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c game</sub> $\mathcal { M } ;$ o<sup>b</sup>jective pro<sup>fil</sup>e   
$\vec { \psi } = ( \mathsf { P } _ { \ J \ J _ { i } d _ { i } } [ \varphi _ { i } ] ) _ { i = 1 } ^ { n } ;$ t<sub>o</sub>l<sub>erance; a</sub>dditi<sub>ona</sub>l b<sub>u</sub>d<sub>ge</sub>t $B ,$   
solution conce<sub>p</sub>t �   
O<sub>u</sub>t<sub>pu</sub>t<sub>:</sub> l<sub>earne</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>egy pro</sub>fil<sub>e</sub> $\widehat { \vec { \sigma } }$ <sup>and</sup> p<sup>er-a</sup>g<sup>ent</sup>   
<sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on cer</sub>tifi<sub>ca</sub>t<sub>es</sub>   
<sub>1</sub> f<sub>or eac</sub>h <sub>agen</sub>t $i \in \{ 1 , \ldots , n \}$ d<sub>o</sub>   
<sub>2</sub> Com<sub>p</sub>ile $\varphi _ { i }$ int<sub>o</sub> <sub>a</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> r<sub>ewa</sub>rd <sub>au</sub>t<sub>o</sub>m<sub>a</sub>t<sub>o</sub>n $\mathcal { A } _ { \varphi _ { i } }$   
<sub>3 en</sub>d   
<sub>4</sub> Constr<sub>u</sub>ct the com<sub>p</sub>osite reward a<sub>u</sub>tomaton $\mathcal { A } _ { \vec { \varphi } }$   
<sub>5</sub> T<sub>ra</sub>i<sub>n w</sub>ith t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d<sub>s</sub> $\mathscr { R } _ { T } ^ { i } = f _ { i } ( { \bf 1 } [ { \pmb m } _ { T } ] = F _ { i } ] )$ <sub>an</sub>d   
counter<sup>f</sup>actua<sup>l</sup> experience trajectories   
<sub>6</sub> U<sub>s</sub>i<sub>ng</sub> fi<sub>xe</sub>d <sub>pro</sub>fil<sub>e</sub> $\widehat { \vec { \sigma } } ,$ estimate <sub>u</sub>nilateral-im<sub>p</sub>ro<sub>v</sub>ement score   
$\widehat { \delta } _ { \zeta }$ b<sub>y</sub> <sub>ro</sub>ll<sub>ou</sub>t<sub>s</sub>   
<sub>7</sub> if $\widehat { \delta } _ { \zeta } > \varepsilon _ { \zeta }$ th<sub>en</sub>   
<sub>8</sub> Train once more with bud<sub>g</sub>et $B ,$ th<sub>en re-es</sub>ti<sub>ma</sub>t<sub>e</sub> $\widehat { \delta } _ { \zeta }$   
<sub>9 en</sub>d   
<sub>10</sub> E<sub>s</sub>tim<sub>a</sub>t<sub>e s</sub>im<sub>u</sub>lt<sub>a</sub>n<sub>eous</sub> int<sub>e</sub>r<sub>va</sub>l<sub>s</sub> $[ L _ { i } , U _ { i } ]$ <sup>f</sup>or a<sup>ll</sup> o<sup>b</sup>jectives   
f<sub>rom</sub> f<sub>res</sub>h <sub>ro</sub>ll<sub>ou</sub>t<sub>s</sub> <sub>o</sub>f th<sub>e</sub> fi<sub>na</sub>l fi<sub>xe</sub>d <sub>pro</sub>fil<sub>e</sub>   
11 return ${ \widehat { \vec { \sigma } } } ,$ and for each agent �, whether its threshold   
condition is certified

![](images/c191700be417902ad15d2a8f6659e836536fc93f941181faa4ecaf27d1b0adb8.jpg)  
Figure 1: DFA for F(intruder\_breach) $\land \mathsf { G } ( \lnot \mathsf { c o l l i s i o n } )$ in Stochastic Pursuit Evasion. State 0 is initial and state 2 is acce<sub>p</sub>tin<sub>g</sub>. Acce<sub>p</sub>tance is evaluated at e<sub>p</sub>isode end.

Reward Automata. F<sub>o</sub>r <sub>eac</sub>h <sub>age</sub>nt<sub>, a</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> finit<sub>e au</sub>t<sub>o</sub>m<sub>a</sub>- ton (DFA) tracks whether the labelled trace of an e<sub>p</sub>isode satisfies the ltl<sub>� p</sub>ath formula inside its <sub>p</sub>robabilistic constraint. Its state <sub>summar</sub>i<sub>ses</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory re</sub>l<sub>evan</sub>t t<sub>o</sub> th<sub>a</sub>t f<sub>ormu</sub>l<sub>a an</sub>d i<sub>s up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>as</sub> <sub>eac</sub>h <sub>new s</sub>t<sub>a</sub>t<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l i<sub>s rea</sub>d<sub>.</sub> F<sub>or examp</sub>l<sub>e,</sub> th<sub>e au</sub>t<sub>oma</sub>t<sub>on</sub> i<sub>n</sub> Fi<sub>gure</sub> 1 <sub>remem</sub>b<sub>ers w</sub>h<sub>e</sub>th<sub>er a</sub> b<sub>reac</sub>h h<sub>as occurre</sub>d <sub>an</sub>d <sub>w</sub>h<sub>e</sub>th<sub>er a co</sub>lli<sub>s</sub>i<sub>on</sub> <sup>h</sup>as ma<sup>d</sup>e t<sup>h</sup>e o<sup>b</sup>jective impossi<sup>bl</sup>e. At episo<sup>d</sup>e en<sup>d</sup>, its accepting states identif<sub>y</sub> successful traces<sub>,</sub> and avera<sub>g</sub>in<sub>g</sub> these binar<sub>y</sub> outcomes estimates satisfaction <sub>p</sub>robabilit<sub>y</sub>. Constr<sub>u</sub>ctin<sub>g</sub> this a<sub>u</sub>tomaton takes 2EXPTIME in the worst case [17]. In <sub>p</sub>ractice, tem<sub>p</sub>oral o<sup>b</sup>jectives are o<sup>f</sup>ten simp<sup>l</sup>e an<sup>d</sup> avoi<sup>d</sup> <sup>d</sup>eep tempora<sup>l</sup> nesting, a<sup>ll</sup>owin<sub>g</sub> existin<sub>g</sub> tools to construct their automata in milliseconds. Thus<sub>,</sub> the worst-case com<sub>p</sub>lexit<sub>y</sub> need not make com<sub>p</sub>ilation a <sub>p</sub>ractical b<sub>o</sub>ttl<sub>enec</sub>k<sub>.</sub> Fi<sub>gure</sub> 1 ill<sub>us</sub>t<sub>ra</sub>t<sub>es a s</sub>i<sub>mp</sub>l<sub>e examp</sub>l<sub>e.</sub>

Definition 7 (Deterministic Reward Automaton). Given a ltl<sub>�</sub> formula, the corresponding reward automaton is a deterministic finite <sub>au</sub>t<sub>oma</sub>t<sub>on over</sub>th<sub>e a</sub>l<sub>p</sub>h<sub>a</sub>b<sub>e</sub>t $\Gamma \ { \stackrel { \mathrm { d e f } } { = } } \ 2 ^ { \mathsf { A p } }$ <sub>, represen</sub>t<sub>e</sub>d <sub>as</sub> th<sub>e</sub> t<sub>up</sub>l<sub>e</sub> $A _ { \varphi } =$ $( M , m _ { 0 } , \Gamma , \delta , F )$ , where � is the set ofstates; $m _ { 0 } \in M$ i<sub>s</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l state<sub>;</sub> $F \subseteq M$ is the set ofaccepting states; and $\delta \colon M \times \Gamma \to M$ is the total transition function.

For ever<sub>y</sub> finite word � $\in \Gamma ^ { + }$ <sub>, we</sub> h<sub>ave:</sub> $\tau , 0 \left| = \varphi \mathrm { i f f } \delta ^ { * } ( m _ { 0 } , \tau ) \in \right.$ $F ,$ <sub>w</sub>h<sub>ere</sub> $\delta ^ { * }$ is the usual extension of � to words. The automaton records acce<sub>p</sub>tance at the terminal timeste<sub>p</sub>. A<sub>g</sub>ent �’s learnin<sub>g</sub> reward a<sub>pp</sub>lies $f _ { i }$ to t<sup>h</sup>is in<sup>d</sup>icator. For eac<sup>h</sup> o<sup>b</sup>jective $\varphi$ <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> an e<sub>p</sub>isodic (terminal) reward functional on com<sub>p</sub>lete <sub>p</sub>aths:

$$
\begin{array} { r } { \mathcal { R } _ { \varphi } ( \pi ) \stackrel { \mathrm { d e f } } { = } \mathbf { 1 } \big [ \delta ^ { * } ( m _ { 0 } , \tau ( \pi ) ) \in F \big ] \ = \ \mathbf { 1 } \big [ \mathcal { M } , \pi \mid = \varphi \big ] . } \end{array}
$$

Thi<sub>s</sub> i<sub>s</sub> <sub>a</sub> B<sub>ernou</sub>lli <sub>ran</sub>d<sub>om</sub> <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>measure</sub> $\mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } }$ <sub>.</sub> Th<sub>eorem</sub> 1 <sub>s</sub>t<sub>a</sub>t<sub>es co</sub>h<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e expec</sub>t<sub>e</sub>d <sub>re</sub>t<sub>urn</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> <sub>a</sub> <sub>rewar</sub>d <sub>au</sub>t<sub>oma</sub>t<sub>on</sub> <sub>an</sub>d th<sub>e</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y.</sub>

Theorem 1 (Coherence). For any strate<sub>g</sub>y profile ⃗� and state �: $\mathbb { E } _ { \pi \sim \mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } } } [ \mathcal { R } _ { \varphi } ( \pi ) ] = \mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } } ( \varphi )$

Proof in A<sub>pp</sub>endix D.4. For multi<sub>p</sub>le a<sub>g</sub>ents<sub>,</sub> the individual reward automata are run s<sub>y</sub>nc<sup>h</sup>ronous<sup>l</sup><sub>y</sub> in a com<sub>p</sub>osite rewar<sup>d</sup> automaton, <sub>w</sub>h<sub>ose s</sub>t<sub>a</sub>t<sub>e recor</sub>d<sub>s eac</sub>h <sub>agen</sub>t’<sub>s au</sub>t<sub>oma</sub>t<sub>on s</sub>t<sub>a</sub>t<sub>e an</sub>d <sub>w</sub>h<sub>ose</sub> t<sub>erm</sub>i<sub>-</sub> <sub>na</sub>l <sub>rewar</sub>d <sub>vec</sub>t<sub>or</sub> <sub>recor</sub>d<sub>s</sub> <sub>componen</sub>t<sub>-w</sub>i<sub>se</sub> <sub>accep</sub>t<sub>ance.</sub>

Definition 8 (Com<sub>p</sub>osite Reward Automaton). Let $\varphi _ { 1 } , \ldots , \varphi _ { n } \in$ $L T L _ { f } ( \mathsf { A p } )$ , and for each $i \in \{ 1 , . . . , n \} l e t { \cal A } _ { \varphi _ { i } } = ( M _ { i } , m _ { 0 } ^ { i } , \Gamma , \delta _ { i } , F _ { i } )$ b<sub>e a</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c rewar</sub>d <sub>au</sub>t<sub>oma</sub>t<sub>on.</sub> Th<sub>e compos</sub>it<sub>e rewar</sub>d <sub>au</sub>t<sub>oma-</sub> t<sub>on</sub> i<sub>s</sub> th<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c mon</sub>it<sub>or</sub> $\begin{array} { r l r } { \mathcal { A } _ { \vec { \varphi } } } & { { } \stackrel { \mathrm { d e f } } { = } } & { \left( M _ { c } , m _ { 0 } ^ { c } , \Gamma , \delta _ { c } \right) } \end{array}$ <sub>w</sub>h<sub>ere</sub> $\begin{array} { r l r l r l } { M _ { c } } & { { } \stackrel { \mathrm { \tiny ~ { ~ d e f } ~ } } { = } } & { \prod _ { i = 1 } ^ { n } M _ { i } , } & { m _ { 0 } ^ { c } } & { { } \stackrel { \mathrm { \tiny ~ { ~ d e f } ~ } } { = } } & { \left( m _ { 0 } ^ { 1 } , \ldots , m _ { 0 } ^ { n } \right) } \end{array}$ <sub>,</sub> <sub>an</sub>d $\begin{array} { r l r } { \delta _ { c } \bigl ( ( m _ { 1 } , \ldots , m _ { n } ) , \ell \bigr ) } & { { } \stackrel { \mathrm { \tiny ~ d e f } } { = } } & { \big ( \delta _ { 1 } ( m _ { 1 } , \ell ) , \ldots , \delta _ { n } ( m _ { n } , \ell ) \big ) } \end{array}$ . Accept<sub>ance</sub> i<sub>s</sub> t<sub>rac</sub>k<sub>e</sub>d <sub>componen</sub>t<sub>-w</sub>i<sub>se v</sub>i<sub>a</sub> th<sub>e se</sub>t<sub>s</sub> $F _ { i } \colon f o r \ : .$ a compos<sup>it</sup>e s<sup>t</sup>a<sup>t</sup>e $\pmb { m } = ( m _ { 1 } , \ldots , m _ { n } )$ we write � ${ \bf \Phi } \cdot \in F _ { i } \mathrm { \Delta } i f f m _ { i } \in F _ { i }$

Definition 8 s<sub>y</sub>nchronises the individual monitors. Alon<sub>g</sub> an e<sub>p</sub>isode $\pi = s _ { 0 } \pmb { a } _ { 0 } s _ { 1 } \cdot \cdot \cdot s _ { T }$ <sub>,</sub> initi<sub>a</sub>li<sub>se</sub> ${ \pmb m } _ { 0 } = \delta _ { c } ( m _ { 0 } ^ { c } , \mathcal { L } ( s _ { 0 } ) )$ <sub>an</sub>d u<sub>p</sub><sup>d</sup>ate $\pmb { m } _ { t + 1 } = \delta _ { c } ( \pmb { m } _ { t } , \pmb { \mathcal { L } } ( s _ { t + 1 } ) )$ , so �<sub>�</sub> records the label <sub>p</sub>refix th<sub>roug</sub>h $s _ { t }$

Definition 9 (Time-Augmented Product Game). Let ℳ be a finite l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c game,</sub> $\mathcal { A } _ { \vec { \varphi } }$ its composite reward automaton, and � th<sub>e</sub> h<sub>or</sub>i<sub>zon.</sub> L<sub>e</sub>t $\kappa : \mathcal S \times \{ 0 , \ldots , H \} \to \{ 0 , 1 \}$ <sub>enco</sub>d<sub>e</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l s<sup>t</sup>opp<sup>i</sup>ng ru<sup>l</sup>e, w<sup>ith</sup> $\kappa ( s , H ) = 1$ for every �. The stopping time is $T \stackrel { \mathrm { d e f } } { = } \operatorname* { m i n } \{ 0 \leq t \leq H : \kappa ( s _ { t } , t ) = 1 \}$ (any information needed by this rule is included in �). The time-au<sub>g</sub>mented s<sub>y</sub>nchronous <sub>p</sub>roduct $\widehat { \mathcal { M } } _ { H }$ <sup>h</sup>as state s<sub>p</sub>ace

$$
{ \widehat { S } } _ { H } \ { \stackrel { \mathrm { d e f } } { = } } \ S \times M _ { c } \times \{ 0 , \ldots , H \} ,
$$

<sub>w</sub>h<sub>ere</sub> $\begin{array} { r l r } { x } & { { } = } & { ( s , m , t ) } \end{array}$ is formed of the environment state, <sub>compos</sub>it<sub>e mon</sub>it<sub>or s</sub>t<sub>a</sub>t<sub>e, an</sub>d <sub>e</sub>l<sub>apse</sub>d ti<sub>me, w</sub>ith i<sub>n</sub>iti<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $x _ { 0 } = ( s _ { 0 } , \delta _ { c } ( m _ { 0 } ^ { c } , \mathcal { L } ( s _ { 0 } ) ) , 0 )$ . A state � is terminal exactl<sub>y</sub> when $\kappa ( s , t ) = 1$ . No further action or transition occurs there. At each nonterminal �, a<sub>g</sub>ent � retains its ori<sub>g</sub>inal le<sub>g</sub>al actions $d _ { H } ^ { i } ( x ) \stackrel { \mathrm { d e f } } { = } d ^ { i } ( s )$ <sub>an</sub>d <sub>we wr</sub>it<sub>e</sub> $\begin{array} { r } { d _ { H } ( x ) \stackrel { \mathrm { \tiny ~ { d e f } ~ } } { = } \prod _ { i } d _ { H } ^ { i } ( x ) } \end{array}$ . For $\boldsymbol { x } ^ { \prime } ~ = ~ ( s ^ { \prime } , m ^ { \prime } , t ^ { \prime } )$ <sub>an</sub>d $\pmb { a } \in d _ { H } ( x )$ <sub>,</sub> th<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> k<sub>erne</sub>l i<sub>s</sub>

$$
\tilde { \mathcal { P } } ( x ^ { \prime } \mid x , \pmb { a } ) \overset { \mathrm { d e f } } { = } \left\{ \begin{array} { l c } { \mathcal { P } ( s ^ { \prime } \mid s , \pmb { a } ) , } & { \mathrm { i f } t ^ { \prime } = t + 1 a n d } \\ { } & { m ^ { \prime } = \delta _ { c } ( m , \mathcal { L } ( s ^ { \prime } ) ) ; } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

The game has a terminal payof of $R ( s , m , t ) \ { \stackrel { \mathrm { d e f } } { = } } \ ( f _ { i } ( 1 [ m \ | =$ $F _ { i } { \big ] } ) ) _ { i = 1 } ^ { n } .$

De<sup>fi</sup>nition 9 ma<sup>k</sup>es <sup>b</sup>ot<sup>h</sup> t<sup>h</sup>e tempora<sup>l</sup> o<sup>b</sup>jectives an<sup>d</sup> t<sup>h</sup>e <sup>h</sup>orizon Markovian. The e<sub>p</sub>isode ends at � with the ex<sub>p</sub>ected terminal <sub>p</sub>a<sub>y</sub>of bein<sub>g</sub> �<sub>�</sub> b<sub>y</sub> Theorem 1. Retainin<sub>g</sub> � allows strate<sub>g</sub>ies to de<sub>p</sub>end on the remainin<sub>g</sub> horizon. We take each $\Sigma _ { i }$ t<sub>o cons</sub>i<sub>s</sub>t <sub>o</sub>f <sub>ran</sub>d<sub>om</sub>i<sub>se</sub>d strate<sub>g</sub>ies that are memor<sub>y</sub>less on $( s _ { t } , m _ { t } , t )$ <sub>, w</sub>ith i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t randomisation across a<sub>g</sub>ents.

Theorem 2 (Markov Sufficiency in the Composite Product). Forfinite-horizon $L T L f$ o<sup>b</sup>jectives $\varphi _ { 1 } , \ldots , \varphi _ { n }$ <sup>and</sup> <sup>ever</sup>y <sup>strate</sup>gy p<sup>ro-</sup> $\begin{array} { r } { f l e \vec { \sigma } \in \Sigma , } \end{array}$ , the satisfaction probabilities $\left( p _ { i } \left( s _ { 0 } , \vec { \sigma } \right) \right) _ { i = 1 } ^ { n }$ <sub>an</sub>d <sub>u</sub>tiliti<sub>es</sub> $\vec { u } ( s _ { 0 } , \vec { \sigma } )$ can be realised by a memoryless strategy profile on $\widehat { \mathcal { M } } _ { H }$ and every such memoryless profile induces a profile in Σ with the same payofs.

Th<sub>eorem</sub> 2 id<sub>en</sub>tifi<sub>es</sub> th<sub>e</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> <sub>w</sub>ith <sub>memory</sub>l<sub>ess</sub> pro<sup>d</sup>uct po<sup>l</sup>icies w<sup>h</sup>i<sup>l</sup>e preserving t<sup>h</sup>eir o<sup>b</sup>jective va<sup>l</sup>ues. Proo<sup>f</sup> is <sub>p</sub>rovided in A<sub>pp</sub>endix D.5.

## 3<sub>.</sub>1 Di<sub>scoun</sub>t F<sub>ac</sub>t<sub>or an</sub>d Ef<sub>ec</sub>ti<sub>ve</sub> H<sub>or</sub>i<sub>zon</sub>

The <sub>p</sub>rod<sub>u</sub>ct above re<sub>p</sub>resents <sub>u</sub>ndisco<sub>u</sub>nted <sub>u</sub>tilities. O<sub>u</sub>r ex<sub>p</sub>eri<sub>men</sub>t<sub>s use</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d<sub>s w</sub>ith $\gamma \in \{ 0 . 9 9 , 1 \}$ at the ori<sub>g</sub>inal sto<sub>pp</sub>in<sub>g</sub> time �. The relationshi<sub>p</sub> of discounted rewards to satisf<sub>ac</sub>ti<sub>on pro</sub>b<sub>a</sub>bilit<sub>y</sub> d<sub>epen</sub>d<sub>s on w</sub>h<sub>e</sub>th<sub>er ep</sub>i<sub>so</sub>d<sub>e</sub> l<sub>eng</sub>th i<sub>s</sub> fi<sub>xe</sub>d<sub>.</sub>

Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>resu</sub>lt<sub>s</sub> <sub>app</sub>l<sub>y</sub> t<sub>o</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>accep</sub>t<sub>ance</sub> <sub>rewar</sub>d<sub>s</sub> f<sub>or</sub> $\varphi _ { i }$ <sub>un</sub>d<sub>er</sub> l<sub>ower-</sub>b<sub>oun</sub>d <sub>requ</sub>i<sub>remen</sub>t<sub>s</sub> <sub>an</sub>d f<sub>or</sub> $\lnot \varphi _ { i }$ <sup>under</sup> <sup>u</sup>pp<sup>er-</sup> b<sub>oun</sub>d <sub>requ</sub>i<sub>remen</sub>t<sub>s.</sub> W<sub>e</sub> fi<sub>rs</sub>t <sub>cons</sub>id<sub>er</sub> fi<sub>xe</sub>d<sub>-</sub>l<sub>eng</sub>th <sub>ep</sub>i<sub>so</sub>d<sub>es:</sub>

Theorem 3 (Fixed-Horizon Terminal Reward Is Invariant to Discounting). Consider episodes offixed time indices $t = 0 , \ldots , H$ and a terminal reward ofthe form $r _ { t } = 0 f o r t < H$ <sub>an</sub>d $r _ { H } =$ $\mathbf { 1 } [ \varphi ] \in \{ 0 , 1 \}$ . Then for any strategy profile $\begin{array} { r l } { \vec { \sigma } , \mathbb { E } _ { \vec { \sigma } } \Big [ \sum _ { t = 0 } ^ { H } \gamma ^ { t } r _ { t } \Big ] = } & { { } } \end{array}$ $\gamma ^ { H } \operatorname { P r } _ { \vec { \sigma } } ( \varphi )$ . In particular, forfixed � and $\gamma \in ( 0 , 1 )$ , <sup>an</sup>y <sup>strate</sup>gy profile that maximises the discounted return also maximises $\mathrm { P r } _ { \vec { \sigma } } ( \varphi )$ E<sub>xac</sub>t N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>a are unc</sub>h<sub>ange</sub>d<sub>,</sub> b<sub>u</sub>t <sub>exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y</sub> i<sub>n</sub> di<sub>scoun</sub>t<sub>e</sub>d<sub>-</sub> return units is $\gamma ^ { H }$ Exploit.

Proof in A<sub>pp</sub>endix D.6. Commonl<sub>y</sub> in RL<sub>,</sub> when the <sub>g</sub>oal is <sub>ac</sub>hi<sub>eve</sub>d <sub>ear</sub>l<sub>y or some</sub> i<sub>rrecovera</sub>bl<sub>e</sub> f<sub>a</sub>il<sub>ure s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s reac</sub>h<sub>e</sub>d<sub>,</sub> th<sub>e</sub> <sub>ep</sub>i<sub>so</sub>d<sub>e</sub> t<sub>e</sub>rmin<sub>a</sub>t<sub>es</sub> <sub>a</sub>nd r<sub>ese</sub>t<sub>s.</sub> With <sub>ea</sub>rl<sub>y</sub> t<sub>e</sub>rmin<sub>a</sub>ti<sub>o</sub>n<sub>,</sub> m<sub>a</sub>ximi<sub>s</sub>in<sub>g</sub> discounted return is not exactl<sub>y</sub> maximisin<sub>g</sub> satisfaction <sub>p</sub>robabilit<sub>y,</sub> <sub>as</sub> it f<sub>avou</sub>r<sub>s ea</sub>rli<sub>e</sub>r <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>o</sub>n<sub>.</sub>

Pro<sub>p</sub>osition 1 (Discounted Return under Earl<sub>y</sub> Termination). Let $T \leq H$ be the (random) termination time of an episode and let � be the acceptance event. If the reward is terminal, $i . e . , r _ { t } = 0$ $f o r t < T$ <sub>an</sub>d $r _ { T } = \mathbf { 1 } [ A ]$ <sub>,</sub> th<sub>en</sub> th<sub>e</sub> di<sub>scoun</sub>t<sub>e</sub>d <sub>re</sub>t<sub>urn</sub> i<sub>s</sub> $\begin{array} { r l } { G _ { \gamma } } & { { } = } \end{array}$ $\textstyle \sum _ { t = 0 } ^ { T } \gamma ^ { t } r _ { t } \ = \ \gamma ^ { T } \mathbf { 1 } [ A ]$ . Hence for any strategy profile ⃗�, we have $\mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] \ = \ \mathbb { E } _ { \vec { \sigma } } [ \gamma ^ { T } \mathbf { 1 } [ A ] ]$

Proof in A<sub>pp</sub>endix D.7. However<sub>,</sub> b<sub>y</sub> Corollar<sub>y</sub> 1<sub>,</sub> even with earl<sub>y</sub> stopping t<sup>h</sup>e <sup>d</sup>iscounte<sup>d</sup> termina<sup>l</sup>-rewar<sup>d</sup> o<sup>b</sup>jective is a <sup>l</sup>ower-<sup>b</sup>iase<sup>d</sup> surro<sub>g</sub>ate <sup>f</sup>or $\mathrm { P r } _ { \vec { \sigma } } ( A )$ th<sub>a</sub>t <sub>a</sub>dditi<sub>ona</sub>ll<sub>y</sub> <sub>enco</sub>d<sub>es</sub> <sub>a</sub> <sub>pre</sub>f<sub>erence</sub> f<sub>or</sub> <sub>ea</sub>rli<sub>e</sub>r <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>o</sub>n:

Corollary 1 (Discounted Objective Bounds Satisfaction Probability). Under the assumptions of Prop. 1 and $T \leq H$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $\gamma ^ { H } \operatorname { P r } _ { \vec { \sigma } } ( A ) \leq \mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] \ \leq \ \operatorname { P r } _ { \vec { \sigma } } ( A )$

Proof in A<sub>pp</sub>endix D.8. Thus<sub>,</sub> when e<sub>p</sub>isodes terminate earl<sub>y</sub> upon success or <sup>f</sup>ai<sup>l</sup>ure, t<sup>h</sup>e <sup>d</sup>iscounte<sup>d</sup> o<sup>b</sup>jective ran<sup>k</sup>s po<sup>l</sup>icies b<sub>y</sub> a discount-wei<sub>g</sub>hted satisfaction <sub>p</sub>robabilit<sub>y,</sub> i.e.<sub>,</sub> it ma<sub>y</sub> <sub>p</sub>refer earlier satisfaction even if the raw <sub>p</sub>robabilit<sub>y</sub> is unchan<sub>g</sub>ed. If one wishes to o<sub>p</sub>timise exactl<sub>y</sub> $\mathrm { P r } _ { \vec { \sigma } } ( A )$ under earl<sub>y</sub> termination<sub>,</sub> one can set $\gamma = 1$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>s</sub>till <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e</sub> <sub>re</sub>t<sub>urn</sub> i<sub>n</sub> <sub>ep</sub>i<sub>so</sub>di<sub>c</sub> settings. Otherwise, the bias introduced can also be bounded by � <sub>p</sub>er Corollar<sub>y</sub> 2:

Corollar<sub>y</sub> 2 (�-Bounded Discount Bias under Earl<sub>y</sub> Termination). Under the assumptions ofCorollary 1, to bound the discount-induced bi<sub>as</sub> b<sub>y a</sub>t <sub>mos</sub>t $\xi \in [ 0 , 1 ]$ uniformly over strategy profiles, it sufices t<sub>o</sub> <sub>c</sub>h<sub>oose</sub> $\gamma \ge ( 1 - \xi ) ^ { 1 / H }$ . That is, for every strategy profile ${ \vec { \sigma } } ,$ $0 \leq \operatorname* { P r } _ { \vec { \sigma } } ( A ) - \mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] \leq \xi$

Pr<sub>oo</sub>f i<sub>s</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> in A<sub>ppe</sub>ndix D<sub>.</sub>10<sub>.</sub>

## 3.2 PAC-St<sub>y</sub>le Guarantees b<sub>y</sub> Sam<sub>p</sub>lin<sub>g</sub>

We <sub>g</sub>ive finite-sam<sub>p</sub>le certificates for <sub>p</sub>robabilistic constraints $\mathsf { P } _ { \mathsf { P } \boxtimes q } [ \varphi ]$ under a fixed strategy profile ⃗� in a finite-horizon setting. Since the d<sub>y</sub>namics are unknown, we estimate � from rollouts of th<sub>e</sub> fi<sub>xe</sub>d <sub>pro</sub>fil<sub>e an</sub>d b<sub>oun</sub>d th<sub>e es</sub>ti<sub>ma</sub>ti<sub>on error.</sub> S<sub>uc</sub>h <sub>cer</sub>tifi<sub>ca</sub>t<sub>es</sub> <sub>are nee</sub>d<sub>e</sub>d b<sub>ecause convergence</sub> t<sub>o an equ</sub>ilib<sub>r</sub>i<sub>um</sub> d<sub>oes no</sub>t <sub>ensure</sub> th<sub>a</sub>t <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s are me</sub>t<sub>.</sub>

Fix ℳ, a start state �, a <sub>p</sub>ath formula �, and a strate<sub>gy</sub> <sub>p</sub>rofile ⃗�. Let $p \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } } ( \varphi ) \ \in [ 0 , 1 ]$ . Consider � i.i.d. rollouts (e<sub>p</sub>isodes): $\pi _ { 1 } , . . . , \pi _ { n } \sim \mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } }$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> bi<sub>nary sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>ors</sub> $X _ { j } \stackrel { \mathrm { \tiny ~ { d e f } } } { = } { \bf 1 } [ { \cal M } , \pi _ { j } { \bf \pi } _ { j } \stackrel { } { = } \varphi ] \in \{ 0 , 1 \}$ , where $j = 1 , \dotsc , n$ <sub>so</sub> th<sub>a</sub>t $\begin{array} { r } { \mathbb { E } [ X _ { j } ] = p . \operatorname { L e t } \hat { p } \stackrel { \mathrm { d e f } } { = } \frac { 1 } { n } \sum _ { j = 1 } ^ { n } X _ { j } } \end{array}$

Theorem 4 (Hoeffding Confidence Interval for Satisfaction Probability). For any confidence parameter � $\in ( 0 , 1 )$ , define the half-width $\rho _ { n } ( \delta ) \ { \stackrel { \mathrm { d e f } } { = } } \ { \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n } } }$ , and the clipped bounds

$$
L _ { n } \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { m a x } \{ 0 , { \hat { p } } - \rho _ { n } ( \delta ) \} , \qquad U _ { n } \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { m i n } \{ 1 , { \hat { p } } + \rho _ { n } ( \delta ) \} .
$$

Th<sub>en</sub> $\begin{array} { r } { \operatorname* { P r } \big ( p \in [ L _ { n } , U _ { n } ] \big ) ~ \ge ~ 1 - \delta , } \end{array}$ <sub>w</sub>h<sub>ere</sub> th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s over</sub> th<sub>e</sub> randomness of the � sampled rollouts.

Pr<sub>oo</sub>f i<sub>s ava</sub>il<sub>a</sub>bl<sub>e</sub> in A<sub>ppe</sub>ndix D<sub>.</sub>9<sub>.</sub> Sin<sub>ce</sub> $\hat { p }$ i<sub>s</sub> th<sub>e</sub> f<sub>rac</sub>ti<sub>on o</sub>f th<sub>e</sub> � rollouts that satisf<sub>y</sub> $\varphi ,$ Th<sub>eorem</sub> 4 <sub>s</sub>t<sub>a</sub>t<sub>es</sub> th<sub>a</sub>t $| \hat { p } - p | \leq \rho _ { n } ( \delta )$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta$ <sub>, w</sub>ith<sub>ou</sub>t k<sub>now</sub>l<sub>e</sub>d<sub>ge o</sub>f $\mathcal { P } _ { : }$ <sub>, w</sub>h<sub>ere</sub> ${ \rho } _ { n } ( \delta ) = O ( n ^ { - \mathrm { { 1 } } / 2 } )$

Corollar<sub>y</sub> 3 (One-Sided Certification of Threshold Constraints). Fi<sub>x a</sub> th<sub>res</sub>h<sub>o</sub>ld $q \in [ 0 , 1 ]$ . Under the assumptions of Theorem $^ { 4 , }$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta$ over the validation rollouts, all four i<sub>mp</sub>li<sub>ca</sub>ti<sub>ons</sub> h<sub>o</sub>ld<sub>:</sub>

$$
\begin{array} { r } { L _ { n } > q \implies p > q , \quad U _ { n } < q \implies p < q , } \\ { L _ { n } \geq q \implies p \geq q , \quad U _ { n } \leq q \implies p \leq q . } \end{array}
$$

Thus the probability of issuing any false threshold certificate is at most �.

Proof is a<sub>v</sub>ailable in A<sub>pp</sub>endix D.11. Corollar<sub>y</sub> 3 certifies $\mathsf { P } { \geq } q \left[ \varphi \right]$ <sub>on</sub>l<sub>y</sub> <sub>w</sub>h<sub>en</sub> $L _ { n } \geq q$ <sub>,</sub> <sub>an</sub>d <sub>symme</sub>t<sub>r</sub>i<sub>ca</sub>ll<sub>y</sub> f<sub>or</sub> <sub>upper</sub> b<sub>oun</sub>d<sub>s.</sub> If $L _ { n } <$ $q \leq U _ { n }$ <sub>,</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> i<sub>s</sub> i<sub>nconc</sub>l<sub>us</sub>i<sub>ve</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>ev</sub>id<sub>ence</sub> th<sub>a</sub>t $p < q$

Corollar<sub>y</sub> 4 (Joint Certification via Bonferroni Adjustment). Fix an objective profile $\vec { \psi } = ( \psi _ { 1 } , \ldots , \psi _ { m } )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\psi _ { i } = \mathsf { P } _ { \bowtie _ { i } q _ { i } } \left[ \varphi _ { i } \right]$ <sub>,</sub> <sub>an</sub>d a fixed strategy profile ⃗�. Before validation, fix sample sizes $n _ { i } \geq 1$ and failure budgets $\delta _ { i } \in ( 0 , 1 )$ <sub>suc</sub>h th<sub>a</sub>t $\textstyle \sum _ { i = 1 } ^ { m } \delta _ { i } \leq \delta$ for a joint b<sub>u</sub>d<sub>ge</sub>t $\delta \in ( 0 , 1 )$ <sub>.</sub> F<sub>or eac</sub>h $i ,$ l<sub>e</sub>t $p _ { i } = \mathrm { P r } _ { { \mathcal M } , s } ^ { \vec { \sigma } } ( \varphi _ { i } )$ <sub>an</sub>d <sub>cons</sub>t<sub>ruc</sub>t <sub>a</sub> Hoefding interval $[ L _ { i } , U _ { i } ]$ <sub>as</sub> i<sub>n</sub> Th<sub>eorem</sub> $4 , f r o m n _ { i }$ i<sub>.</sub>i<sub>.</sub>d<sub>. ro</sub>ll<sub>ou</sub>t<sub>s an</sub>d with marginal failure probability $\delta _ { i }$ <sub>.</sub> Th<sub>en</sub>

$$
\operatorname* { P r } \left( \bigcap _ { i = 1 } ^ { m } \{ p _ { i } \in [ L _ { i } , U _ { i } ] \} \right) \geq 1 - \delta .
$$

C<sub>onsequen</sub>tl<sub>y,</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta ,$ <sub>a</sub>ll <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>-</sub>b<sub>ase</sub>d certificates from Corollary 3 are simultaneously sound. In particular, if, for every �, the relevant condition

$$
\begin{array} { r l } { L _ { i } > q _ { i } ( \bowtie _ { i } = > ) , \qquad } & { U _ { i } < q _ { i } ( \bowtie _ { i } = < ) , } \\ { L _ { i } \geq q _ { i } ( \bowtie _ { i } = \geq ) , \qquad } & { U _ { i } \leq q _ { i } ( \bowtie _ { i } = \leq ) } \end{array}
$$

h<sub>o</sub>ld<sub>s,</sub> th<sub>en</sub> $\mathcal { M } , s , \vec { \sigma } \vdash \vec { \psi }$ is jointly certified with confidence at least $1 - \delta .$ . The uniform Bonferroni choice $\delta _ { i } = \delta / m$ gives half-width $\rho _ { n _ { i } } ( \delta / m ) \ : = \ : \sqrt { \ln ( 2 m / \delta ) / ( 2 n _ { i } ) }$ . The rollouts used for diferent o<sup>b</sup>jectives nee<sup>d</sup> not <sup>b</sup>e in<sup>d</sup>epen<sup>d</sup>ent. In particu<sup>l</sup>ar, t<sup>h</sup>e same ro<sup>ll</sup>outs may be evaluated against every �<sub>�</sub>.

Pr<sub>oo</sub>f i<sub>s ava</sub>il<sub>a</sub>bl<sub>e</sub> in A<sub>ppe</sub>ndix D<sub>.</sub>12<sub>.</sub> C<sub>o</sub>r<sub>o</sub>ll<sub>a</sub>r<sub>y</sub> 4 <sub>ce</sub>rtifi<sub>es a</sub>ll <sub>agen</sub>t<sub>s</sub>’ <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> f<sub>or</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>pro</sub>fil<sub>e,</sub> <sub>regar</sub>dl<sub>ess</sub> <sub>o</sub>f h<sub>ow</sub> it <sub>was</sub> l<sub>earne</sub>d <sub>w</sub>h<sub>en va</sub>lid<sub>a</sub>ti<sub>on uses</sub> f<sub>res</sub>h i<sub>.</sub>i<sub>.</sub>d <sub>ro</sub>ll<sub>ou</sub>t<sub>s.</sub>

B<sub>oun</sub>d<sub>ary</sub> P<sub>ro</sub>b<sub>a</sub>biliti<sub>es.</sub> At $p = q ,$ <sub>s</sub>tri<sub>c</sub>t r<sub>equ</sub>ir<sub>e</sub>m<sub>e</sub>nt<sub>s</sub> $( p > q$ or $p < q )$ <sub>are</sub> f<sub>a</sub>l<sub>se, w</sub>hil<sub>e non-s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>requ</sub>i<sub>remen</sub>t<sub>s</sub> $( p \geq q \mathrm { o r } p \leq q )$ h<sub>o</sub>ld b<sub>u</sub>t l<sub>eave</sub> <sub>no</sub> <sub>marg</sub>i<sub>n</sub> f<sub>or</sub> <sub>separa</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>con</sub>fid<sub>ence</sub> b<sub>oun</sub>d f<sub>rom</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>us</sub>i<sub>ng</sub> fi<sub>n</sub>it<sub>e</sub>l<sub>y many samp</sub>l<sub>es.</sub> A <sub>use</sub>f<sub>u</sub>l <sub>a</sub>dditi<sub>ona</sub>l feasibilit<sub>y</sub> condition, analo<sub>g</sub>ous to [41], is satisfaction with <sub>p</sub>ositive <sub>s</sub>l<sub>ac</sub>k<sub>:</sub>

Definition 10 (Threshold Stren<sub>g</sub>thenin<sub>g</sub>). For a positive rational �, l<sub>e</sub>t $T ( \psi , \eta )$ re<sub>p</sub><sup>l</sup>ace $\mathsf { P } { \geq } d \ b y \mathsf { P } { \geq } d { + } \eta$ <sub>an</sub>d $\mathsf { P } { \leq } _ { d } b y \mathsf { P } { \leq } _ { d - \eta }$ (and similarl<sub>y</sub> $f o r >$ <sub>an</sub>d $< ) ,$ provided the strengthened threshold remains in [0, 1].

D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 10 <sub>moves eac</sub>h th<sub>res</sub>h<sub>o</sub>ld i<sub>n</sub> th<sub>e more</sub> d<sub>eman</sub>di<sub>ng</sub> di<sub>-</sub> rection.

Assum<sub>p</sub>tion A1 (Generalised Slater Condition). There exist a positive rational � and a strategy profile $\vec { \sigma } ^ { \star } \in \Sigma$ such that, for every <sup>a</sup>g<sup>ent</sup> $i , T ( \psi _ { i } , \eta )$ is defined and $\mathcal { M } , s _ { 0 } , \vec { \sigma } ^ { \star } \vdash T ( \psi _ { i } , \eta )$

Assumption A1 expresses joint <sup>f</sup>easi<sup>b</sup>i<sup>l</sup>ity wit<sup>h</sup> positive s<sup>l</sup>ac<sup>k</sup>. It neither <sub>p</sub>laces the witness at e<sub>q</sub>uilibrium nor ensures that learnin<sub>g</sub> fi<sub>n</sub>d<sub>s</sub> it<sub>.</sub> R<sub>a</sub>th<sub>er,</sub> it <sub>ensures</sub> th<sub>e ex</sub>i<sub>s</sub>t<sub>ence o</sub>f<sub>a pro</sub>fil<sub>e w</sub>h<sub>ose</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s</sub> hold with a <sub>p</sub>ositive certification mar<sub>g</sub>in.

## 3<sub>.</sub>3 E<sub>x</sub>i<sub>s</sub>t<sub>ence</sub> <sub>o</sub>f E<sub>qu</sub>ilib<sub>r</sub>i<sub>a</sub>

For infinite-horizon conc<sub>u</sub>rrent stochastic <sub>g</sub>ames <sub>w</sub>ith <sub>qu</sub>alitati<sub>v</sub>e o<sup>b</sup>jectives, equi<sup>l</sup>i<sup>b</sup>rium existence is on<sup>l</sup>y <sup>k</sup>nown <sup>f</sup>or certain <sup>l</sup>ow levels of the Borel hierarch<sub>y</sub> (e.<sub>g</sub>., safet<sub>y</sub> objectives). For exam<sub>p</sub>le, <sub>ex</sub>i<sub>s</sub>t<sub>ence o</sub>f $\varepsilon { - } \mathsf { N E }$ for <sub>g</sub>eneral �-<sub>p</sub>la<sub>y</sub>er concurrent reachabilit<sub>y</sub> <sub>g</sub>ames remains an o<sub>p</sub>en <sub>q</sub>uestion [10]. In the finite-horizon case, however<sub>,</sub> existence is classical: ever<sub>y</sub> finite extensive-form <sub>g</sub>ame with <sub>p</sub>erfect recall has a sub<sub>g</sub>ame-<sub>p</sub>erfect e<sub>q</sub>uilibrium [45], and a M<sub>ar</sub>k<sub>ov one</sub> i<sub>s o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc</sub>ti<sub>on, so</sub>l<sub>v</sub>i<sub>ng a</sub>t <sub>eac</sub>h state and remainin<sub>g</sub> time the one-shot sta<sub>g</sub>e <sub>g</sub>ame whose <sub>p</sub>a<sub>y</sub>ofs <sub>are</sub> th<sub>e</sub> <sub>con</sub>ti<sub>nua</sub>ti<sub>on</sub> <sub>va</sub>l<sub>ues,</sub> <sub>w</sub>hi<sub>c</sub>h h<sub>as</sub> <sub>a</sub> <sub>m</sub>i<sub>xe</sub>d <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> b<sub>y</sub> Nash’s theorem [40] (see e.<sub>g</sub>., [21]). We restate this standard result f<sub>o</sub>r <sub>ou</sub>r <sub>se</sub>ttin<sub>g</sub> in Pr<sub>opos</sub>iti<sub>o</sub>n 2:

Pro<sub>p</sub>osition 2 (Existence of SPNE in Finite Horizon [45]). Every fully observed concurrent stochastic game with finite state and <sub>ac</sub>ti<sub>on</sub> <sub>spaces,</sub> M<sub>ar</sub>k<sub>ov</sub>i<sub>an</sub> <sub>s</sub>t<sub>age</sub> <sub>an</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d<sub>s,</sub> <sub>an</sub>d h<sub>or</sub>i<sub>zon</sub> bounded b<sub>y</sub> � admits a sub<sub>g</sub>ame-<sub>p</sub>erfect Nash e<sub>q</sub>uilibrium (SPNE) i<sub>n</sub> <sub>ran</sub>d<sub>om</sub>i<sub>se</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> th<sub>a</sub>t i<sub>s</sub> M<sub>ar</sub>k<sub>ov,</sub> i<sub>.e.</sub> <sub>eac</sub>h <sub>agen</sub>t’<sub>s</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub>d th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> ti<sub>me</sub> $H - t$ . In particular, this holds for the fully observed product game $\widehat { \mathcal { M } } _ { H }$ <sub>w</sub>ith utilities from Definition 2.

Pr<sub>oo</sub>f in A<sub>ppe</sub>ndix D<sub>.</sub>13<sub>.</sub> Pr<sub>opos</sub>iti<sub>o</sub>n 2 <sub>e</sub>n<sub>su</sub>r<sub>es</sub> <sub>a</sub>n <sub>equ</sub>ilibri<sub>u</sub>m in th<sub>e unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d <sub>c</sub>l<sub>ass o</sub>f <sub>ran</sub>d<sub>om</sub>i<sub>se</sub>d <sub>po</sub>li<sub>c</sub>i<sub>es over</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>y o</sub>b<sub>serve</sub>d <sub>p</sub>roduct state. However<sub>,</sub> existence does not <sub>g</sub>uarantee feasibilit<sub>y</sub> for a<sub>g</sub>ent � in Definition 6: $\mathcal { E } _ { \zeta }$ ma<sub>y</sub> be nonem<sub>p</sub>t<sub>y</sub> while $\mathcal { F } _ { \zeta } ^ { i }$ <sup>is</sup> <sup>em</sup>p<sup>t</sup>y<sup>.</sup>

## 3.4 Con<sub>v</sub>er<sub>g</sub>ence Guarantees

Man<sub>y</sub> of the methods considered in our ex<sub>p</sub>eriments<sub>,</sub> includin<sub>g</sub> IPPO, IQL, an<sup>d</sup> PR2, <sup>h</sup>ave no genera<sup>l</sup> guarantee o<sup>f</sup> convergence to Nash e<sub>q</sub>uilibria under non-linear function a<sub>pp</sub>roximation and concurrent <sub>p</sub>olic<sub>y</sub> u<sub>p</sub>dates [51]. We instead anal<sub>y</sub>se tabular Nash-Q [28], w<sup>h</sup>ic<sup>h</sup> <sup>b</sup>ac<sup>k</sup>s up equi<sup>l</sup>i<sup>b</sup>rium payo<sup>f</sup>s o<sup>f</sup> successor-state g<sup>ames.</sup>

I<sub>n</sub> fi<sub>n</sub>it<sub>e</sub> h<sub>or</sub>i<sub>zon,</sub> <sub>convergence</sub> <sub>procee</sub>d<sub>s</sub> b<sub>ac</sub>k<sub>war</sub>d<sub>s</sub> f<sub>rom</sub> t<sub>erm</sub>i<sub>na</sub>l o<sub>u</sub>tcomes as contin<sub>u</sub>ation val<sub>u</sub>es stabilise. The s<sub>u</sub>btlet<sub>y</sub> is e<sub>qu</sub>ilibri<sub>u</sub>m selection: <sub>w</sub>hen se<sub>v</sub>eral Nash e<sub>qu</sub>ilibria exist<sub>,</sub> small estimation errors may c<sup>h</sup>ange w<sup>h</sup>ic<sup>h</sup> one Nash-Q c<sup>h</sup>ooses. We t<sup>h</sup>ere<sup>f</sup>ore re-<sub>q</sub>uire vanishin<sub>g</sub> errors in the sta<sub>g</sub>e-<sub>g</sub>ame <sub>p</sub>a<sub>y</sub>ofs to <sub>y</sub>ield vanishin<sub>g</sub> <sub>errors</sub> i<sub>n</sub> th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>con</sub>ti<sub>nua</sub>ti<sub>on va</sub>l<sub>ues.</sub>

W<sub>e use</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>y o</sub>b<sub>serve</sub>d <sub>pro</sub>d<sub>uc</sub>t $\hat { \mathcal { M } } _ { H }$ <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 9<sub>.</sub> A <sub>s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s</sub> reac<sup>h</sup>a<sup>bl</sup>e i<sup>f</sup> it <sup>h</sup>as positive pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity un<sup>d</sup>er some <sup>l</sup>ega<sup>l</sup> joint action se<sub>q</sub>uence <sup>f</sup>rom $x _ { 0 }$ . At each non-terminal �, a sta<sub>g</sub>e <sub>g</sub>ame � assi<sub>g</sub>ns a <sub>p</sub>a<sub>y</sub>o<sup>f</sup> vector <sup>i</sup>n $[ 0 , 1 ] ^ { n }$ t<sub>o eac</sub>h $\pmb { a } \in d _ { H } ( \boldsymbol { x } )$ . In <sub>p</sub>artic<sub>u</sub>lar<sub>,</sub> $Q ( x )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e game</sub> $\mathbf { \em a } \mapsto Q ( x , \mathbf { \em a } )$

Definition 11 (Selector and Selector Profile). A selector Sel maps each non-terminal state � and sta<sub>g</sub>e <sub>g</sub>ame � at � to a Nash e<sub>q</sub>uilibrium $\mathsf { S e l } _ { x } ( g )$ of �, with value $\mathsf { V a l } _ { x } ( g ) \overset { \mathrm { d e f } } { = } \mathbb { E } _ { \pmb { a } \sim \mathsf { S e l } _ { x } ( g ) } [ g ( \pmb { a } ) ] \in [ 0 , 1 ] ^ { n }$ . Its <sub>se</sub>l<sub>ec</sub>t<sub>or pro</sub>fil<sub>e</sub> $\vec { \sigma } _ { \mathsf { S e l } }$ i<sub>s o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc</sub>ti<sub>on:</sub> $V ^ { \mathsf { S e l } } ( x ) \ { \stackrel { \mathrm { d e f } } { = } }$ $R ( x )$ for terminal �, and, at each non-terminal � in time la<sub>y</sub>ers $t = H - 1 , \ldots , 0 ,$

$$
\begin{array} { r l r } & {  { Q ^ { 5 \mathrm { e l } } ( x , a ) \stackrel { \mathrm { d e f } } { = } \sum _ { x ^ { \prime } } \tilde { \mathcal { P } } ( x ^ { \prime } \mid x , a ) V ^ { 5 \mathrm { e l } } ( x ^ { \prime } ) , } } \\ & { } & \\ & { \vec { \sigma } _ { 5 \mathrm { e l } } ( x ) \stackrel { \mathrm { d e f } } { = } \mathsf { S e l } _ { x } \big ( Q ^ { 5 \mathrm { e l } } ( x ) \big ) , \qquad V ^ { 5 \mathrm { e l } } ( x ) \stackrel { \mathrm { d e f } } { = } \mathsf { V a l } _ { x } \big ( Q ^ { 5 \mathrm { e l } } ( x ) \big ) . } \end{array}
$$

D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 11 <sub>s</sub>t<sub>a</sub>t<sub>es</sub> th<sub>e equ</sub>ilib<sub>r</sub>i<sub>um c</sub>h<sub>o</sub>i<sub>ces use</sub>d th<sub>roug</sub>h<sub>ou</sub>t b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc</sub>ti<sub>on.</sub> I<sub>n</sub>t<sub>u</sub>iti<sub>ve</sub>l<sub>y,</sub> th<sub>e se</sub>l<sub>ec</sub>t<sub>or</sub> fi<sub>xes one</sub> N<sub>as</sub>h<sub>-</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>um con</sub>ti<sub>nua</sub>ti<sub>on a</sub>t <sub>eac</sub>h <sub>s</sub>t<sub>a</sub>t<sub>e w</sub>h<sub>en severa</sub>l <sub>are ava</sub>il<sub>a</sub>bl<sub>e,</sub> and backward induction <sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ates the corres<sub>p</sub>ondin<sub>g</sub> <sub>p</sub>a<sub>y</sub>of <sub>v</sub>ector to earlier states. In o<sub>u</sub>r ex<sub>p</sub>eriments<sub>, w</sub>e <sub>u</sub>se a selector th<sub>a</sub>t <sub>ran</sub>k<sub>s</sub> th<sub>e</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>a</sub> <sub>re</sub>t<sub>urne</sub>d b<sub>y</sub> th<sub>e</sub> <sub>s</sub>t<sub>age-game</sub> <sub>so</sub>l<sub>ver</sub> b<sub>y</sub> their total expected Q-value. Ties favour the equilibrium with the hi<sub>g</sub>hest minimum individual value.

Nash-Q with selector Sel maintains $Q ^ { k } ( x , \pmb { a } ) ~ \in ~ \mathbb { R } ^ { n }$ <sub>an</sub>d<sub>,</sub> after observin<sub>g</sub> a transition $( x , a , x ^ { \prime } )$ at iteration �, u<sub>p</sub>dates $Q ^ { k + 1 } ( x , \pmb { a } ) = \overset { \smile } { \left( 1 - \alpha _ { k } ( x , \pmb { a } ) \right) } \acute { Q } ^ { k } ( x , \pmb { a } ) + \alpha _ { k } ( x , \pmb { a } ) V ^ { k } ( x ^ { \prime } )$ , where $V ^ { k } ( x ^ { \prime } ) = R ( x ^ { \prime } )$ if $x ^ { \prime }$ i<sub>s</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>an</sub>d $V ^ { k } ( x ^ { \prime } ) = \mathsf { V a l } _ { x ^ { \prime } } ( Q ^ { k } ( x ^ { \prime } ) )$ <sub>o</sub>th<sub>erw</sub>i<sub>se.</sub> All <sub>o</sub>th<sub>er en</sub>t<sub>r</sub>i<sub>es are unc</sub>h<sub>ange</sub>d $( \alpha _ { k } \ = \ 0 )$ <sub>.</sub> At <sub>non-</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es,</sub> th<sub>e ex</sub>t<sub>rac</sub>t<sub>e</sub>d <sub>pro</sub>fil<sub>e</sub> i<sub>s</sub> $\vec { \sigma } _ { k } ( x ) \stackrel { \mathrm { d e f } } { = } \mathsf { S e l } _ { x } ( Q ^ { k } ( x ) )$ .

Theorem 5 (Convergence of Nash-Q Values and Exploitability). Let Sel be a selector, and suppose that $( i ) Q ^ { 0 } ( x , \pmb { a } ) \in [ 0 , 1 ] ^ { n }$ for all $( x , \pmb { a } ) ; ( i i ) .$ for every reachable non-terminal � and $\pmb { a } \in d _ { H } ( x )$ the step sizes satisfy $\begin{array} { r } { \alpha _ { k } ( x , \pmb { a } ) \in [ 0 , 1 ] , \sum _ { k } \alpha _ { k } ( x , \pmb { a } ) = \infty } \end{array}$ <sub>an</sub>d $\begin{array} { r } { \sum _ { k } \alpha _ { k } ( x , \pmb { a } ) ^ { 2 } < } \end{array}$ ∞ almost surel<sub>y</sub>; (iii) each $\mathsf { S e l } _ { \mathsf { i } }$ <sub>�</sub> i<sub>s</sub> B<sub>ore</sub>l <sub>measur-</sub> able, the updated pair and step sizes are measurable functions ofthe past, and, conditional on that past, the successor is sampled from $\tilde { \mathcal { P } } ( \cdot \mid x , a ) ,$ ; and (iv) for ever<sub>y</sub> reachable non-terminal �, $\mathsf { V a l } _ { x }$ is continuous at $\log ^ { \mathsf { S e l } } ( x )$ <sub>, w</sub>hi<sub>c</sub>h h<sub>o</sub>ld<sub>s</sub> i<sub>n par</sub>ti<sub>cu</sub>l<sub>ar</sub> $i f S e l _ { x }$ is continuous at $Q ^ { \mathsf { S e l } } ( x )$ or if all Nash equilibria $f \bar { Q } ^ { \mathsf { S e l } } ( x )$ h<sub>ave</sub> th<sub>e same va</sub>l<sub>ue vec</sub>t<sub>or.</sub> Th<sub>en</sub> $\vec { \sigma } _ { \mathsf { S e l } }$ is a Markov SPNE $o f \widehat { \mathcal { M } } _ { H }$ <sub>an</sub>d<sub>, a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y,</sub> $Q ^ { k } \to Q ^ { \mathsf { S e l } }$ <sub>on a</sub>ll <sub>reac</sub>h<sub>a</sub>bl<sub>e non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es,</sub> $u _ { i } ( s _ { 0 } , \vec { \sigma } _ { k } )  u _ { i } ( s _ { 0 } , \vec { \sigma } _ { 5 \mathrm { e l } } )$ for ever<sub>y</sub> a<sub>g</sub>ent �, and $\mathsf { E x p l o i t } ( s _ { 0 } , \vec { \sigma } _ { k } ) \to 0$ (Definition $4 ) .$ In <sub>p</sub>articu<sup>l</sup>ar, for $e \nu e r y \varepsilon > 0 , { \vec { \sigma } } _ { k }$ is an �-NE for all suficiently large �. Ifmoreover $\mathsf { S e l } _ { x }$ is continuous at $Q ^ { \mathsf { S e l } } ( x )$ for every reachable non-terminal �, th<sub>en a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y</sub> $\vec { \sigma } _ { k }  \vec { \sigma } _ { \mathsf { S e l } }$ <sub>on a</sub>ll <sub>reac</sub>h<sub>a</sub>bl<sub>e non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es.</sub>

Pr<sub>oo</sub>f in A<sub>ppe</sub>ndix D<sub>.</sub>14<sub>.</sub> Th<sub>eo</sub>r<sub>e</sub>m 5 <sub>assu</sub>m<sub>es</sub> th<sub>a</sub>t th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>p</sub>a<sub>y</sub>of ma<sub>p</sub> Val<sub>�</sub> is continuous at $Q ^ { \mathsf { S e l } } ( x )$ . To<sub>g</sub>ether with (i)–(iii), this yields convergence of Q-values and payofs and vanishing ex-<sub>p</sub>loitabilit<sub>y</sub>. The stron<sub>g</sub>er assum<sub>p</sub>tion that $\mathsf { S e l } _ { x }$ i<sub>s</sub> <sub>co</sub>ntin<sub>uous</sub> th<sub>e</sub>r<sub>e</sub> also <sub>y</sub>ields conver<sub>g</sub>ence of action <sub>p</sub>robabilities. Pa<sub>y</sub>of continuit<sub>y</sub> <sub>a</sub>l<sub>one</sub> <sub>a</sub>ll<sub>ows</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> b<sub>e</sub>t<sub>ween</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>a</sub> <sub>w</sub>ith th<sub>e</sub> <sub>same</sub> <sub>payo</sub>f vector.

Since ties are broken onl<sub>y</sub> after maximisin<sub>g</sub> the total ex<sub>p</sub>ected Q-value, the resulting backup may satisfy fewer thresholds than <sub>ano</sub>th<sub>er</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> if hi<sub>g</sub>h <sub>va</sub>l<sub>ues</sub> f<sub>or</sub> <sub>some</sub> <sub>agen</sub>t<sub>s</sub> <sub>com-</sub> <sub>pensa</sub>t<sub>e</sub> f<sub>or</sub> l<sub>ow</sub> <sub>va</sub>l<sub>ues</sub> f<sub>or</sub> <sub>o</sub>th<sub>ers.</sub> H<sub>ence,</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> <sub>convergence</sub> d<sub>oes no</sub>t <sub>ensure sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on o</sub>f th<sub>e pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s, w</sub>hi<sub>c</sub>h must be checked using Corollary 3 (or Corollary 4 for joint certification).

## 4 Ex<sub>pe</sub>rim<sub>e</sub>nt<sub>a</sub>l An<sub>a</sub>l<sub>ys</sub>i<sub>s</sub>

We evaluate Al<sub>g</sub>orithm 1<sub>,</sub> tar<sub>g</sub>etin<sub>g</sub> $\zeta ~ = ~ \varepsilon { \mathrm { - } } \mathsf { N } \mathsf { E }$ <sub>w</sub>ith $\varepsilon \ = \ 0 . 0 5$ (Definition 3) as a <sub>p</sub>ractical tolerance for unilateral im<sub>p</sub>rovements. I<sub>mprovemen</sub>t<sub>s</sub> b<sub>e</sub>l<sub>ow</sub> thi<sub>s</sub> t<sub>o</sub>l<sub>erance</sub> <sub>are</sub> t<sub>rea</sub>t<sub>e</sub>d <sub>as</sub> <sub>prac</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>neg-</sub> li<sub>g</sub>ible relative to finite-sam<sub>p</sub>le <sub>p</sub>a<sub>y</sub>of estimation<sub>,</sub> a<sub>pp</sub>roximate best-res<sub>p</sub>onse trainin<sub>g,</sub> and stochastic learnin<sub>g</sub> d<sub>y</sub>namics. A<sub>pp</sub>endi<sub>x</sub> G <sub>prov</sub>id<sub>es</sub> f<sub>u</sub>ll h<sub>yperparame</sub>t<sub>ers,</sub> <sub>an</sub>d <sub>our</sub> <sub>co</sub>d<sub>e</sub> i<sub>s</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> <sub>a</sub>t htt<sub>p</sub>s://<sub>g</sub>ithub.com/ni<sub>g</sub>htl<sub>y</sub>/learnin<sub>g</sub>-rational-strate<sub>g</sub>ies for re<sub>p</sub>roducibilit<sub>y</sub>.

Counterfactual Experience Replay. Terminal rewards are sparse <sub>an</sub>d l<sub>ess</sub> <sub>su</sub>it<sub>e</sub>d t<sub>o</sub> l<sub>ong-</sub>h<sub>or</sub>i<sub>zon</sub> t<sub>as</sub>k<sub>s,</sub> <sub>so</sub> <sub>we</sub> <sub>use</sub> <sub>ltl� coun</sub>t<sub>er</sub>f<sub>ac-</sub> tual ex<sub>p</sub>erience re<sub>p</sub>la<sub>y</sub> (CER), ins<sub>p</sub>ired b<sub>y</sub> Voloshin et al. [54]. For <sub>eac</sub>h <sub>agen</sub>t<sub>, we rep</sub>l<sub>ay a recor</sub>d<sub>e</sub>d <sub>ro</sub>ll<sub>ou</sub>t <sub>c</sub>h<sub>un</sub>k f<sub>rom a</sub>lt<sub>erna</sub>ti<sub>ve</sub> <sub>rewar</sub>d<sub>-au</sub>t<sub>oma</sub>t<sub>on</sub> <sub>s</sub>t<sub>a</sub>t<sub>es,</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>propos</sub>iti<sub>on</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> t<sub>o</sub> r<sub>eco</sub>m<sub>pu</sub>t<sub>e</sub> m<sub>o</sub>nit<sub>o</sub>r <sub>s</sub>t<sub>a</sub>t<sub>es a</sub>nd t<sub>e</sub>rmin<sub>a</sub>l r<sub>ewa</sub>rd<sub>s.</sub> En<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt <sub>o</sub>b<sub>serva</sub>ti<sub>ons, ac</sub>ti<sub>ons, an</sub>d <sub>ep</sub>i<sub>so</sub>d<sub>e en</sub>di<sub>ngs s</sub>t<sub>ay</sub> fi<sub>xe</sub>d<sub>:</sub> th<sub>e same</sub> <sub>ro</sub>ll<sub>ou</sub>t <sub>prov</sub>id<sub>es</sub> <sub>exper</sub>i<sub>ence</sub> f<sub>or</sub> <sub>severa</sub>l <sub>s</sub>t<sub>ages</sub> <sub>o</sub>f <sub>progress</sub> t<sub>owar</sub>d<sub>s</sub> t<sup>h</sup>e o<sup>b</sup>jective. IQL variants store re<sup>l</sup>a<sup>b</sup>e<sup>ll</sup>e<sup>d</sup> experience in rep<sup>l</sup>ay, Nash-Q up<sup>d</sup>ates on<sup>l</sup>ine, an<sup>d</sup> PPO variants com<sup>b</sup>ine origina<sup>l</sup> an<sup>d</sup> <sub>re</sub>l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>c</sub>h<sub>un</sub>k<sub>s</sub> <sub>an</sub>d <sub>recompu</sub>t<sub>e</sub> <sub>ro</sub>ll<sub>ou</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs.</sub> A<sub>ppen</sub>di<sub>x</sub> D<sub>.</sub>1 <sub>g</sub>i<sub>ves comp</sub>l<sub>e</sub>t<sub>e pseu</sub>d<sub>oco</sub>d<sub>e an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s.</sub>

E<sub>nv</sub>i<sub>ronmen</sub>t<sub>s.</sub> W<sub>e cons</sub>id<sub>er</sub> 8 t<sub>as</sub>k i<sub>ns</sub>t<sub>ances</sub> f<sub>rom</sub> fi<sub>ve env</sub>i<sub>ron-</sub> ment families<sub>,</sub> f<sub>u</sub>ll<sub>y</sub> s<sub>p</sub>ecified in A<sub>pp</sub>endix C and A<sub>pp</sub>endix E. The suite contains two- and three-<sub>p</sub>la<sub>y</sub>er classic and ca<sub>p</sub>ture variants of

$$
\begin{array} { r } { \begin{array} { r l } { \underset { \mathrm { \tiny { - \infty ~ \mathbb { P R } 2 - 1 0 L ~ -- ~ \mathbb { P R } 2 \cdot 1 \mathbb { Q } 1 . - \mathbb { C } \mathbb { R } } } } { \mathrm { \tiny { - \infty ~ \mathbb { P R } 2 . 1 \mathbb { P } 0 . ~ -- ~ \mathbb { P r } 2 . 1 \mathbb { Q } 1 . - \mathbb { C } \mathbb { R } } } } ~ \underset { \mathrm { \tiny { N a s h - Q ~ L e a r n i n g ~ -- ~ \mathbb { P r } 2 . 1 \mathbb { P r } 0 . ~ \mathbb { Q } - \mathbb { C } \mathbb { R } } } } { \mathrm { \tiny { \mathbb { P R } 2 . 1 \mathbb { P } \mathbb { P } 0 . ~ - \mathbb { C } \mathbb { R } } } } ~ \underset { \mathrm { \tiny { - \infty ~ \mathbb { P r } 2 . 1 \mathbb { P r } \mathbb { Q } 1 . \mathbb { C } \mathbb { R } } } } { \mathrm { \tiny { - \mathbb { Q } . 1 . - \mathbb { P r } \mathbb { Q } 1 . - \mathbb { Q } . 1 . \mathbb { Q } 1 . - \mathbb { Q } . 1 . \mathbb { Q } 1 . - \mathbb { P r } \mathbb { R } } } } } & { \mathrm { \textnormal { \tiny { 1 . - \mathbb { Q } L . C \mathbb { R } } } } } \end{array} } \end{array}
$$

![](images/d7148a62543b0361649099436be7c0fdc81ea2b71f00bdded18bde83a0bd7b81.jpg)  
(a) Pursuit Classic 2

![](images/9071d33ae06ccc7e12688b06dcfde32847bd1a2b79b2b713c5d93d3cc122ef5f.jpg)  
(b) Pursuit Capture 2

![](images/02655bda581e34b1f213a8dc97eaefe25d9bc06c47430a00c2197e493102c2bf.jpg)  
(c) Pursuit Classic 3

![](images/2c49beffadef3adbd71832fdc51a1c4b3cff8fad8cb54cf03a1f555cf8f16382.jpg)  
(d) Pursuit Capture 3

![](images/44d14ecd3d785cb52d145e837987519968e877362a6cded35221e6e21de804e8.jpg)  
(e) Coins

![](images/50ac2d374fd23bc050cb7b188ca2347864150ca30a2c803e31b153fb5b67d512.jpg)  
(f) Matrix Stag Hunt

![](images/031dc5f176138f9fe2b01e237128ad021ae11c8a09b04f7db388d06cc4f520b2.jpg)  
(g) Commons Harvest

![](images/bb6578b46d817501e80cc244b7318a2a514806c2d1117fe11cf9295ba1c102c0.jpg)  
(h) Territory

Fi ure 2: Best-a ent satisfaction of each famil ’s stron er variant with or without CER (mean and 95% CIs over five runs). Di<sub>amon</sub>d<sub>s</sub> <sub>mar</sub>k th<sub>e</sub> <sub>s</sub>t<sub>ronges</sub>t h<sub>an</sub>d<sub>cra</sub>ft<sub>e</sub>d<sub>-rewar</sub>d b<sub>ase</sub>li<sub>ne</sub>’<sub>s</sub> <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d fi<sub>na</sub>l <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> <sub>a</sub>t it<sub>s</sub> fi<sub>na</sub>l t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>s</sub>t<sub>ep.</sub>
<table><tr><td>Benchmark</td><td>Learner</td><td>i</td><td>Agent&#x27;s objective</td><td>pi</td><td>95% CI</td><td> $\mathsf { N a s h C o n v }$ </td></tr><tr><td>Pursuit Classic 2</td><td>PR2-IPP0</td><td>0</td><td> $\mathsf { P } _ { \geq 0 . 1 } [ \mathsf { G }$  ¬goal_reached]</td><td>0.979</td><td>[0.936, 0.993]</td><td>0.000</td></tr><tr><td>Pursuit Capture 2</td><td>PR2-IPPO</td><td>1</td><td> $\mathsf { P } { \scriptstyle \geq } 0 . 5$  [G¬guard_intruder_collision ∧G¬collision_ever]</td><td>1.000</td><td>[0.957, 1.000]</td><td>0.000</td></tr><tr><td>Pursuit Classic 3</td><td>PR2-IQL-CER</td><td>2</td><td> $\mathsf { P { \geq } 0 . 1 } \mathopen { } \mathclose \bgroup \left[ \mathsf { F g o a l } \aftergroup \egroup \right]$  reached ∧G¬collision_ever]</td><td>0.975</td><td>[0.932, 0.993]</td><td>0.008</td></tr><tr><td>Pursuit Capture 3</td><td>IQL-CER</td><td>2</td><td> $\mathsf { P } _ { \geq 0 . 8 } [ \mathsf { G } \cdot$  guard_intruder_collision∧G¬collision_ever]</td><td>0.969</td><td>[0.926, 0.998]</td><td>0.002</td></tr><tr><td>Coins Own Coin First</td><td>IPPO-CER</td><td>1</td><td> $\mathsf { P } _ { \geq 0 . 4 } [ \neg w _ { 1 } \cup m _ { 1 } ]$ </td><td>0.503</td><td>[0.460, 0.546]</td><td>0.044</td></tr><tr><td>Matrix Sustained Stag</td><td>Nash-Q-CER</td><td>1</td><td> $\mathsf { P } _ { \geq 0 . 9 }$  [F(both_stag∧X(both_stag∧G¬unmirrored))]</td><td>1.000</td><td>[0.957, 1.000]</td><td>0.000</td></tr><tr><td>Commons Safe Harvest</td><td>IQL-CER</td><td>0</td><td> $\mathsf { P { \geq } 0 . 8 } \mathsf { [ G \mathrm { \ - } p 1 a y e r \_ 0 } _ { \mathsf { - } }$  zapped ∧ Fplayer_0_apple_collected]</td><td>0.817</td><td>[0.774, 0.860]</td><td>0.206</td></tr><tr><td>Territory Role Claim</td><td>IQL</td><td>0</td><td> $\mathsf { P } _ { \geq 0 . 2 5 } [ \neg \mathsf { p } _ { 1 }$  _claimed U(po_claimed ∧¬p₁_claimed)]</td><td>0.923</td><td>[0.880, 0.944]</td><td>0.031</td></tr></table>

Table 1: Selected learners and agents (player\_�); objectives omit the outer X. Satisfaction estimates and marginal 95% Hoefding interval end<sub>p</sub>oints are five-seed means from 1000 new rollouts <sub>p</sub>er seed, without selection adjustment. A<sub>pp</sub>endix E gives <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s, a</sub>t<sub>om</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons an</sub>d <sub>a</sub>ll <sub>agen</sub>t<sub>s</sub>’ <sub>resu</sub>lt<sub>s.</sub>

Stochastic Pursuit Evasion [56], the re<sub>p</sub>eated matrix-<sub>g</sub>ame Stag H<sub>unt, an</sub>d th<sub>ree spa</sub>ti<sub>a</sub>l <sub>su</sub>b<sub>s</sub>t<sub>ra</sub>t<sub>es a</sub>d<sub>ap</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e</sub> M<sub>elting</sub>P<sub>ot</sub> suite [1]: Coins, Commons Harvest, and Territory. Each a<sub>g</sub>ent � has objective $\psi _ { i } = \mathsf { P } _ { \geq q _ { i } } [ \mathsf { X } \varphi _ { i } ]$ <sub>w</sub>ith <sub>a ro</sub>l<sub>e-spec</sub>ifi<sub>c</sub> th<sub>res</sub>h<sub>o</sub>ld $q _ { i }$ (A<sub>pp</sub>endix E).

Al<sub>g</sub>orithms. O<sub>u</sub>r set<sub>up</sub> tar<sub>g</sub>ets fi<sub>v</sub>e al<sub>g</sub>orithm families: <sub>v</sub>ariants of IPPO, PR2-IPPO, IQL, PR2-IQL, an<sup>d</sup> Nash-Q, wit<sup>h</sup> an<sup>d</sup> wit<sup>h</sup>out counterfactual ex<sub>p</sub>erience re<sub>p</sub>la<sub>y</sub> (CER, e.<sub>g</sub>., IQL and IQL-CER), and one handcrafted-reward (hc-) counter<sub>p</sub>art for each learner famil<sub>y</sub>. Probabilistic recursive reasonin<sub>g</sub> (PR2) evaluates actions a<sub>g</sub>ainst modelled o<sub>pp</sub>onent res<sub>p</sub>onses<sub>,</sub> <sub>g</sub>uidin<sub>g</sub> action selection and <sub>p</sub>olic<sub>y</sub> u<sub>p</sub><sup>d</sup>ates.

However, Nash-Q comp<sup>l</sup>ete<sup>d</sup> on<sup>l</sup>y <sup>f</sup>or Stag Hunt <sup>d</sup>ue to it <sup>b</sup>ei<sub>ng a</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> l<sub>ow-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>ma</sub>t<sub>r</sub>i<sub>x game, w</sub>h<sub>ereas</sub> th<sub>e o</sub>th<sub>er</sub> <sub>env</sub>i<sub>ronmen</sub>t<sub>s</sub> ti<sub>me</sub>d <sub>ou</sub>t <sub>a</sub>ft<sub>er</sub> 96 h<sub>ours.</sub> Th<sub>e</sub> h<sub>an</sub>d<sub>cra</sub>ft<sub>e</sub>d<sub>-rewar</sub>d <sup>b</sup>ase<sup>l</sup>ine tries to capture t<sup>h</sup>e same o<sup>b</sup>jective <sup>f</sup>ait<sup>hf</sup>u<sup>ll</sup>y wit<sup>h</sup> a Pyt<sup>h</sup>on <sub>rewar</sub>d f<sub>unc</sub>ti<sub>on,</sub> b<sub>u</sub>t h<sub>as</sub> <sub>a</sub> dif<sub>eren</sub>t <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> <sub>se</sub>t i<sub>n</sub> <sub>prac</sub>ti<sub>ce,</sub> <sub>an</sub>d thus serves <sub>p</sub>rimaril<sub>y</sub> as an auxiliar<sub>y</sub> baseline.

Evaluation Metrics. Each fixed <sub>p</sub>rofile is evaluated on 1000 episo<sup>d</sup>es per see<sup>d</sup>, using <sup>fi</sup>ve training see<sup>d</sup>s. We se<sup>l</sup>ect t<sup>h</sup>e o<sup>b</sup>jectivet<sub>ra</sub>i<sub>ne</sub>d l<sub>earner an</sub>d <sub>agen</sub>t <sub>w</sub>ith th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t <sub>mean sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> i<sub>n</sub> thi<sub>s</sub> initial evaluation. Table 1 re<sub>p</sub>orts that a<sub>g</sub>ent’s satisfaction estimate f<sub>rom</sub> <sub>an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>,</sub> <sub>marg</sub>i<sub>na</sub>l H<sub>oe</sub>fdi<sub>ng</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>s,</sub> <sub>an</sub>d th<sub>e</sub> <sub>pro</sub>fil<sub>e</sub>’<sub>s</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> N<sub>as</sub>hC<sub>onv.</sub>

A<sub>pp</sub>endix E <sub>g</sub>ives the <sub>p</sub>er-a<sub>g</sub>ent results<sub>,</sub> learnin<sub>g</sub> curves<sub>,</sub> and h<sub>an</sub>d<sub>cra</sub>ft<sub>e</sub>d <sub>compar</sub>i<sub>sons.</sub>

W<sub>e mo</sub>d<sub>e</sub>l th<sub>e un</sub>il<sub>a</sub>t<sub>era</sub>l i<sub>mprovemen</sub>t<sub>s</sub> f<sub>oun</sub>d b<sub>y</sub> fi<sub>n</sub>it<sub>e-</sub>b<sub>u</sub>d<sub>ge</sub>t res<sub>p</sub>onse searches usin<sub>g</sub> a<sub>pp</sub>roximate NashConv [50]. For each a<sub>g</sub>ent �, we freeze its o<sub>pp</sub>onents and continue trainin<sub>g</sub> from the <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>pro</sub>fil<sub>e</sub> <sub>w</sub>ith th<sub>e</sub> <sub>same</sub> l<sub>earner</sub> <sub>an</sub>d <sub>a</sub> fi<sub>xe</sub>d <sub>a</sub>dditi<sub>ona</sub>l b<sub>u</sub>d <sub>g</sub>et. In <sub>p</sub>ractice<sub>,</sub> this can instead be estimated usin<sub>g</sub> best-res<sub>p</sub>onse <sub>p</sub>redictors.

Definition 12 (A<sub>pp</sub>roximate NashConv). Let $\vec { \sigma } ^ { ( i ) }$ denote the profile obtained by the unilateral search for agent �. For objective-reward profiles, with payof $\mathrm { \dot { \alpha } } _ { u _ { i } } ( \vec { \sigma } ) = \mathrm { P r } _ { \mathcal { M } , s _ { 0 } } ^ { \vec { \sigma } } ( \mathsf { X } _ { \varphi _ { i } } )$ <sub>an</sub>d <sub>ro</sub>ll<sub>ou</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> $\widehat { u } _ { i }$ th<sub>e score</sub> i<sub>s</sub>

$$
{ \sf N a s h C o n v \widetilde { \sigma } } ^ { \mathrm { q e f } } ( \vec { \sigma } ) \overset { n } { = } \sum _ { i = 1 } ^ { n } \operatorname* { m a x } \biggr \{ 0 , \widehat { u } _ { i } \bigl ( \vec { \sigma } ^ { ( i ) } \bigr ) - \widehat { u } _ { i } ( \vec { \sigma } ) \biggr \} .
$$

Th<sub>e sco</sub>r<sub>e</sub> in D<sub>e</sub>finiti<sub>o</sub>n 12 m<sub>easu</sub>r<sub>es</sub> th<sub>e u</sub>nil<sub>a</sub>t<sub>e</sub>r<sub>a</sub>l im<sub>p</sub>r<sub>ove</sub>m<sub>e</sub>nt<sub>s</sub> found b<sub>y</sub> the search. We call it a<sub>pp</sub>roximate NashConv because <sub>eac</sub>h <sub>un</sub>il<sub>a</sub>t<sub>era</sub>l <sub>response</sub> i<sub>s o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> fi<sub>n</sub>it<sub>e-</sub>b<sub>u</sub>d<sub>ge</sub>t t<sub>ra</sub>i<sub>n</sub>i<sub>ng ra</sub>th<sub>er</sub> than exact best-res<sub>p</sub>onse com<sub>p</sub>utation<sub>,</sub> and the resultin<sub>g</sub> utilit<sub>y</sub> <sub>ga</sub>i<sub>ns</sub> <sub>are</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d f<sub>rom</sub> fi<sub>n</sub>it<sub>e</sub> <sub>ro</sub>ll<sub>ou</sub>t<sub>s.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> fi<sub>n</sub>it<sub>e</sub> <sub>searc</sub>h and sam<sub>p</sub>lin<sub>g</sub> noise <sub>p</sub>reclude an �-NE certificate. As a heuristic, if $\mathsf { N a s h C o n v } ^ { \approx } > 0 . 0 5$ , we train the ori<sub>g</sub>inal <sub>p</sub>rofile for u<sub>p</sub> to 25% additional ste<sub>p</sub>s. Unlike ex<sub>p</sub>loitabilit<sub>y</sub> (Definition 4), which takes the maximum unilateral <sub>g</sub>ain over a<sub>g</sub>ents, NashConv<sup>≈</sup> sums the <sub>g</sub>ains found across a<sub>g</sub>ents.

R<sub>esu</sub>lt<sub>s.</sub> Fi<sub>gure</sub> 2 <sub>s</sub>h<sub>ows</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>curves</sub> <sub>across</sub> <sub>a</sub>ll 8 b<sub>enc</sub>h<sub>mar</sub>k<sub>s,</sub> while <sub>p</sub>er-a<sub>g</sub>ent curves and com<sub>p</sub>lete learner com<sub>p</sub>arisons a<sub>pp</sub>ear in A<sub>ppe</sub>ndix E<sub>.</sub> Th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d l<sub>ea</sub>rn<sub>e</sub>r<sub>s</sub> in T<sub>a</sub>bl<sub>e</sub> 1 <sub>ac</sub>hi<sub>eve</sub> ind<sub>epe</sub>nd<sub>e</sub>nt satis<sup>f</sup>action estimates a<sup>b</sup>ove 0.9 on six <sup>b</sup>enc<sup>h</sup>mar<sup>k</sup>s: PR2-IPPO on two-p<sup>l</sup>ayer Pursuit, PR2-IQL-CER on C<sup>l</sup>assic 3, IQL-CER on Capture 3, IQL on Territory, an<sup>d</sup> Nash-Q-CER on Stag Hunt. Harvest se<sup>l</sup>ects IQL-CER wit<sup>h</sup> mean 0.817, an<sup>d</sup> Coins se<sup>l</sup>ects IPPO-CER with mean 0.503.

Threshold feasibility. Capture 2 illustrates why rational synthesis is conditional on feasibilit<sub>y</sub> (Definition 6). Its d<sub>y</sub>namics <sub>y</sub>ield a numericall<sub>y</sub> com<sub>p</sub>uted minimax <sub>g</sub>uard satisfaction <sub>p</sub>robabilit<sub>y</sub> � ≈ 0.130. At an<sub>y</sub> unrestricted 0.05-NE, the <sub>g</sub>uard’s satisfaction <sub>p</sub>robabilit<sub>y</sub> is at most � + 0.05 ≈ 0.180, below its re<sub>q</sub>uired 0.3. The intruder’s threshold 0.5 remains feasible. Thus, the <sub>g</sub>uard’s acce<sub>p</sub>table-e<sub>q</sub>uilibrium set $\mathcal { F } _ { 0 . 0 5 \mathrm { - N E } } ^ { \mathrm { g } }$ i<sub>s</sub> <sub>e</sub>m<sub>p</sub>t<sub>y.</sub> Additi<sub>o</sub>n<sub>a</sub>l tr<sub>a</sub>inin<sub>g</sub> i<sub>s</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>t t<sub>o reso</sub>l<sub>ve</sub> thi<sub>s, as</sub> th<sub>e o</sub>b<sub>s</sub>t<sub>ruc</sub>ti<sub>on</sub> li<sub>es</sub> i<sub>n</sub> th<sub>e env</sub>i<sub>ron-</sub> <sub>men</sub>t d<sub>ynam</sub>i<sub>cs coup</sub>l<sub>e</sub>d <sub>w</sub>ith th<sub>e spec</sub>ifi<sub>e</sub>d <sub>requ</sub>i<sub>remen</sub>t<sub>.</sub>

## 5 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

W<sub>e p</sub>r<sub>ov</sub>id<sub>e a</sub>n <sub>e</sub>xt<sub>e</sub>nd<sub>e</sub>d r<sub>ev</sub>i<sub>ew o</sub>f r<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d <sub>wo</sub>rk in A<sub>ppe</sub>ndix F<sub>,</sub> b<sub>u</sub>t <sub>summar</sub>i<sub>se</sub> th<sub>e mos</sub>t <sub>re</sub>l<sub>evan</sub>t <sub>wor</sub>k<sub>s</sub> h<sub>ere.</sub> H<sub>a</sub>h<sub>n e</sub>t <sub>a</sub>l<sub>. cons</sub>id<sub>er ver</sub>if<sub>y-</sub> ing minimax o<sup>b</sup>jectives using t<sup>h</sup>e <sup>l</sup>ogic patl, a<sup>l</sup>t<sup>h</sup>oug<sup>h</sup> t<sup>h</sup>ey consi<sup>d</sup>er <sub>a mo</sub>d<sub>e</sub>l <sub>c</sub>h<sub>ec</sub>ki<sub>ng pro</sub>bl<sub>em ra</sub>th<sub>er</sub> th<sub>an a syn</sub>th<sub>es</sub>i<sub>s pro</sub>bl<sub>em as</sub> th<sub>ey</sub> a<sup>ll</sup>ow re<sup>l</sup>a<sup>b</sup>e<sup>ll</sup>ing, an<sup>d</sup> on<sup>l</sup>y a<sup>ll</sup>ow an o<sup>b</sup>jective consi<sup>d</sup>ere<sup>d b</sup>etween a <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>coa</sub>liti<sub>on</sub> <sub>an</sub>d it<sub>s</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>.</sub> A <sub>c</sub>l<sub>ose</sub>l<sub>y</sub> <sub>re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>-</sub>b<sub>ase</sub>d tools is PRISM-Games [33], which su<sub>pp</sub>orts verification and strategy synt<sup>h</sup>esis <sup>f</sup>or rpatl-sty<sup>l</sup>e pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic an<sup>d</sup> rewar<sup>d</sup> o<sup>b</sup>jectives in stoc<sup>h</sup>astic games, inc<sup>l</sup>u<sup>d</sup>ing mu<sup>l</sup>ti-o<sup>b</sup>jective an<sup>d</sup> equi<sup>l</sup>i<sup>b</sup>rium-<sup>b</sup>ase<sup>d</sup> extensions [31, 32, 34]. These methods assume ex<sub>p</sub>licit transition <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> V<sub>arr</sub>i<sub>cc</sub>hi<sub>one e</sub>t <sub>a</sub>l<sub>. cons</sub>t<sub>ruc</sub>t <sub>a rewar</sub>d <sub>mac</sub>hi<sub>ne</sub> f<sub>rom an</sub> atl s<sub>p</sub>ecification<sub>,</sub> but their a<sub>pp</sub>roach assumes a full<sub>y</sub> coo<sub>p</sub>erative <sub>gran</sub>d <sub>coa</sub>liti<sub>on, w</sub>hi<sub>c</sub>h i<sub>n</sub> t<sub>urn a</sub>ll<sub>ows</sub> d<sub>e</sub>l<sub>ega</sub>ti<sub>ng</sub> t<sub>o rewar</sub>d <sub>ma-</sub> <sub>c</sub>hi<sub>nes</sub> <sub>genera</sub>t<sub>e</sub>d f<sub>or</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>agen</sub>t<sub>s.</sub> H<sub>u</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> <sub>cons</sub>id<sub>er</sub> <sub>rewar</sub>d machines for stochastic <sub>g</sub>ames<sub>,</sub> but are limited to exact Nash e<sub>q</sub>uilibria<sub>,</sub> non-<sub>q</sub>uantitative s<sub>p</sub>ecifications<sub>,</sub> and two-<sub>p</sub>la<sub>y</sub>er <sub>g</sub>eneral-sum <sub>g</sub>ames. Jothimuru<sub>g</sub>an et al. <sub>p</sub>rovide a bes<sub>p</sub>oke enumerate-and-verif<sub>y</sub> <sub>p</sub>rocedure for a non-<sub>p</sub>robabilistic task s<sub>p</sub>ecification lan<sub>g</sub>ua<sub>g</sub>e that is d<sub>es</sub>i<sub>gne</sub>d <sub>exc</sub>l<sub>us</sub>i<sub>ve</sub>l<sub>y</sub> f<sub>or equ</sub>ilib<sub>r</sub>i<sub>a w</sub>ith hi<sub>g</sub>h <sub>soc</sub>i<sub>a</sub>l <sub>we</sub>lf<sub>are.</sub> H<sub>am-</sub> <sub>mon</sub>d <sub>e</sub>t <sub>a</sub>l<sub>.</sub> <sub>s</sub>t<sub>u</sub>di<sub>e</sub>d t<sub>empora</sub>l<sub>-</sub>l<sub>og</sub>i<sub>c</sub> <sub>rewar</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>sa</sub>ti<sub>on</sub> <sub>o</sub>f <sub>a</sub> <sub>sum</sub> of <sub>p</sub>robabilities<sub>,</sub> b<sub>u</sub>t restricted their anal<sub>y</sub>sis to coo<sub>p</sub>erati<sub>v</sub>e MARL. Our use of e<sub>q</sub>uilibrium constraints is ins<sub>p</sub>ired b<sub>y</sub> classical rational s<sub>y</sub>nthesis [14, 20, 52]. Subramanian et al. stud<sub>y</sub> reward-machine based learnin<sub>g</sub> in <sub>g</sub>eneral-sum stochastic <sub>g</sub>ames<sub>,</sub> <sub>p</sub>ro<sub>p</sub>osin<sub>g</sub> CROM and QROM, which combine counterfactual experience updates over <sub>rewar</sub>d<sub>-mac</sub>hi<sub>ne</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>w</sub>ith <sub>opponen</sub>t <sub>mo</sub>d<sub>e</sub>lli<sub>ng</sub> <sub>an</sub>d <sub>prov</sub>id<sub>e</sub> t<sub>a</sub>b<sub>-</sub> ular convergence and finite-time guarantees for Nash Q-values under additional assum<sub>p</sub>tions. However<sub>,</sub> our <sub>p</sub>a<sub>p</sub>er deals with <sub>p</sub>rob-<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s ra</sub>th<sub>er</sub> th<sub>an</sub> di<sub>scoun</sub>t<sub>e</sub>d <sub>rewar</sub>d<sub>s.</sub> W<sub>e prov</sub>id<sub>e</sub> <sub>samp</sub>li<sub>ng-</sub>b<sub>ase</sub>d <sub>cer</sub>tifi<sub>ca</sub>ti<sub>on guaran</sub>t<sub>ees an</sub>d b<sub>oun</sub>d<sub>s on</sub> di<sub>scoun</sub>ti<sub>ng</sub> efects<sub>,</sub> as well as e<sub>q</sub>uilibrium stabilit<sub>y</sub> and conver<sub>g</sub>ence anal<sub>y</sub>sis.

## 6 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n<sub>s a</sub>nd F<sub>u</sub>t<sub>u</sub>r<sub>e</sub> W<sub>o</sub>rk

Our a<sub>pp</sub>roach tar<sub>g</sub>ets multi<sub>p</sub>le strate<sub>g</sub>ic a<sub>g</sub>ents<sub>,</sub> each e<sub>q</sub>ui<sub>pp</sub>ed with its own pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic tempora<sup>l</sup>-<sup>l</sup>ogic o<sup>b</sup>jective in pltl<sub>�</sub> over <sup>fi</sup>nite traces. Each a<sub>g</sub>ent seeks a strate<sub>gy</sub> satisf<sub>y</sub>in<sub>g</sub> its own <sub>p</sub>robabilistic th<sub>res</sub>h<sub>o</sub>ld <sub>w</sub>ithi<sub>n</sub> <sub>a</sub> <sub>pro</sub>fil<sub>e</sub> th<sub>a</sub>t i<sub>s</sub> <sub>s</sub>t<sub>a</sub>bl<sub>e</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub> <sub>c</sub>h<sub>osen</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> notion (solution conce<sub>p</sub>t), inter<sub>p</sub>reted over the utilities induced b<sub>y</sub> the <sub>p</sub>robabilistic re<sub>q</sub>uirements.

Com<sub>p</sub>ared to existin<sub>g</sub> a<sub>pp</sub>roaches<sub>,</sub> our work focuses on modelfree strate<sub>gy</sub> s<sub>y</sub>nthesis in a learnin<sub>g</sub> settin<sub>g,</sub> without re<sub>q</sub>uirin<sub>g</sub> <sup>k</sup>nown transition <sup>d</sup>ynamics. T<sup>h</sup>is a<sup>ll</sup>ows consi<sup>d</sup>ering o<sup>b</sup>jectives b<sub>eyon</sub>d th<sub>ose</sub> th<sub>a</sub>t <sub>a</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>c</sub> <sub>coa</sub>liti<sub>on</sub> <sub>can</sub> <sub>un</sub>il<sub>a</sub>t<sub>era</sub>ll<sub>y</sub> <sub>en</sub>f<sub>orce,</sub> <sub>as</sub> <sub>requ</sub>i<sub>re</sub>d b<sub>y</sub> l<sub>og</sub>i<sub>cs</sub> <sub>suc</sub>h <sub>as</sub> atl <sub>an</sub>d patl<sub>.</sub> W<sub>e</sub> <sub>ana</sub>l<sub>yse</sub> h<sub>ow</sub> di<sub>s-</sub> counting a<sup>f</sup>ects t<sup>h</sup>e <sup>l</sup>earning o<sup>f</sup> pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic o<sup>b</sup>jectives, esta<sup>bl</sup>is<sup>h</sup> conditional convergence guarantees for tabular Nash-Q learning, an<sup>d</sup> provi<sup>d</sup>e a<sup>l</sup>gorit<sup>h</sup>m-agnostic <sup>fi</sup>nite-samp<sup>l</sup>e certi<sup>fi</sup>cates o<sup>f</sup> o<sup>b</sup>jecti<sub>ve sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>o</sub>n<sub>.</sub>

Future work could tar<sub>g</sub>et nested <sub>p</sub>robabilistic <sub>p</sub>ro<sub>p</sub>erties and the undiscounted infinite-horizon case for �-re<sub>g</sub>ular <sub>p</sub>ro<sub>p</sub>erties, which would re<sub>q</sub>uire learnin<sub>g</sub> maximal end com<sub>p</sub>onents [11, 24], althou<sub>g</sub>h this is non-trivial as end com<sub>p</sub>onents for multi-a<sub>g</sub>ent s<sub>y</sub>st<sub>ems</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e</sub> d<sub>ynam</sub>i<sub>cs o</sub>f <sub>o</sub>th<sub>er agen</sub>t<sub>s.</sub> A<sub>no</sub>th<sub>er ex</sub>t<sub>ens</sub>i<sub>on</sub> <sub>o</sub>f i<sub>n</sub>t<sub>eres</sub>t <sub>cou</sub>ld b<sub>e eva</sub>l<sub>ua</sub>ti<sub>ng</sub> dif<sub>eren</sub>t <sub>so</sub>l<sub>u</sub>ti<sub>on concep</sub>t<sub>s, suc</sub>h <sub>as a</sub> jointly satisficing objective [43]. It may also be worth investigating con<sup>fid</sup>ence-aware equi<sup>l</sup>i<sup>b</sup>rium se<sup>l</sup>ection wit<sup>h</sup> Nash-Q; <sup>h</sup>owever t<sup>h</sup>is i<sub>n</sub>t<sub>ro</sub>d<sub>uces s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>no</sub>i<sub>se</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e se</sub>l<sub>ec</sub>t<sub>or an</sub>d <sub>non-</sub>l<sub>oca</sub>l d<sub>epen-</sub> d<sub>ence across s</sub>t<sub>a</sub>t<sub>es.</sub>

## A<sub>c</sub>k<sub>now</sub>l<sub>e</sub>d<sub>gmen</sub>t<sub>s</sub>

The research described in this <sub>p</sub>a<sub>p</sub>er was s<sub>upp</sub>orted b<sub>y</sub> the EPSRC (<sub>g</sub>rant number EP/X015823/1) and the UKRI Centre for Doctoral Trainin<sub>g</sub> in Safe and Trusted AI (<sub>g</sub>rant number EP/S0233356/1).

## R<sub>e</sub>f<sub>erences</sub>

[1] John P A<sub>g</sub>a<sub>p</sub>iou, Alexander Sasha Vezhnevets, Ed<sub>g</sub>ar A Duéñez-Guzmán, Ja<sub>y</sub>d M<sub>a</sub>t<sub>yas,</sub> Yir<sub>a</sub>n M<sub>ao,</sub> P<sub>e</sub>t<sub>e</sub>r S<sub>u</sub>n<sub>e</sub>h<sub>ag,</sub> R<sub>ap</sub>h<sub>ae</sub>l Kö<sub>s</sub>t<sub>e</sub>r<sub>,</sub> Ud<sub>a</sub>ri M<sub>a</sub>dh<sub>us</sub>h<sub>a</sub>ni<sub>,</sub> K<sub>avya</sub> <sup>K</sup>opparapu, <sup>R</sup>amona <sup>C</sup>omanescu, e<sup>t</sup> a<sup>l</sup>. <sup>2022</sup>. <sup>M</sup>e<sup>lti</sup>ng <sup>P</sup>o<sup>t</sup> <sup>2</sup>.<sup>0</sup>. ar<sup>Xi</sup>v prepr<sup>i</sup>n<sup>t</sup> arXiv:2211.13746 (2022).

[2] Thomas <sup>Å</sup><sub>g</sub>otnes, Valentin Goranko, and Wojciech Jamro<sub>g</sub>a. 2007. Alternatin<sub>g</sub>- time temporal logics with irrevocable strategies. In Proceedings ofthe 11th conference on Theoretical aspects ofrationality and knowledge. 15–24.

[3] Rajeev Alur, Thomas A Henzin<sub>g</sub>er, and Orna Ku<sub>p</sub>ferman. 2002. Alternatin<sub>g</sub>-time tem<sub>p</sub>oral lo<sub>g</sub>ic. Journal ofthe ACM (JACM) 49, 5 (2002), 672–713.

[4] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. 2016. Concrete <sub>p</sub>roblems in AI safet<sub>y</sub>. arXiv <sub>p</sub>re<sub>p</sub>rint arXiv:1606.06565 (2016).

[5] Fahiem Bacchus, Crai<sub>g</sub> Boutilier, and Adam Grove. 1996. Rewardin<sub>g</sub> behaviors. In Proceedings ofthe National Conference on Artificial Intelligence. 1160–1167.

[6] Christel Baier and Joost-Pieter Katoen. 2008. Principles ofmodel checkin<sub>g</sub>. MIT p<sup>ress.</sup>

[7] Nuel Belna<sub>p</sub> and Michael Perlof. 1988. Seein<sub>g</sub> to it that: a canonical form for a entives. Theoria 54, 3 (1988), 175–199.

[8] Al<sub>p</sub>er Kamil Bozkurt, Yu Wan<sub>g</sub>, Michael M Zavlanos, and Miroslav Pajic. 2021. M<sub>o</sub>d<sub>e</sub>l<sub>-</sub>f<sub>ree re</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng</sub> f<sub>or s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c games w</sub>ith li<sub>near</sub> t<sub>empora</sub>l logic objectives. In 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 10649–10655.

[9] Ronen Brafman, Giuse<sub>pp</sub>e De Giacomo, and Fabio Patrizi. 2018. LTLf/LDLf nonmarkovian rewards. In Proceedings of the AAAI conference on artificial intelligence, V<sub>o</sub>l<sub>.</sub> 32<sub>.</sub>

[10] Krishnendu Chatterjee. 2007. Errata: Flaw in Concurrent Reachabilit<sub>y</sub> Games: Com<sub>p</sub>lexit<sub>y</sub> and Existence. Erratum note (PDF). htt<sub>p</sub>s://<sub>p</sub>ub.ista.ac.at/ \~<sup>k</sup>c<sup>h</sup>atterjee/errata-cs<sup>l</sup>04.p<sup>df</sup> Erratum note <sup>f</sup>or: C<sup>h</sup>atterjee, Majum<sup>d</sup>ar, an<sup>d</sup> Jur<sup>d</sup>z iński (CSL 2004) “On Nash E<sub>q</sub>uilibria in Stochastic Games”. Accessed 2026-02-02..

[11] Krishnendu Chatterjee and Monika Henzin<sub>g</sub>er. 2011. Faster and d<sub>y</sub>namic al<sub>g</sub>o <sub>r</sub>ith<sub>ms</sub> f<sub>or</sub> <sub>max</sub>i<sub>ma</sub>l <sub>en</sub>d<sub>-componen</sub>t d<sub>ecompos</sub>iti<sub>on</sub> <sub>an</sub>d <sub>re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>grap</sub>h <sub>pro</sub>bl<sub>ems</sub> i<sub>n</sub> probabilistic verification. In Proceedings of the twenty-second annual ACM-SIAM s<sub>y</sub>m<sub>p</sub>osium on Discrete Al<sub>g</sub>orithms. SIAM<sub>,</sub> 1318–1336.

[12] Taolue Chen, Vojtěch Forejt, Marta Kwiatkowska, Aistis Simaitis, and Clemens Wi<sup>l</sup>tsc<sup>h</sup>e. 2013. On stoc<sup>h</sup>astic games wit<sup>h</sup> mu<sup>l</sup>tip<sup>l</sup>e o<sup>b</sup>jectives. In Internationa<sup>l</sup> Symposium on Mathematical Foundations ofComputer Science. Springer, 266–277.

[13] Taolue Chen and Jian Lu. 2007. Probabilistic alternatin<sub>g</sub>-time tem<sub>p</sub>oral lo<sub>g</sub>ic and model checking algorithm. In Fourth International Conference on Fuzzy Systems and Knowled<sub>g</sub>e Discover<sub>y</sub> (FSKD 2007), Vol. 2. IEEE, 35–39.

[14] Rodica Condurache, Emmanuel Filiot, Rafaella Gentilini, Jean Fran Raskin, et al. 2016. The com<sub>p</sub>lexit<sub>y</sub> of rational s<sub>y</sub>nthesis. In 43rd International Collo<sub>q</sub>uium on Automata, Lan<sub>g</sub>ua<sub>g</sub>es, and Pro<sub>g</sub>rammin<sub>g</sub> (ICALP 2016), Vol. 121. Ioannis Chatzi-<sub>g</sub>i<sub>a</sub>nn<sub>a</sub>ki<sub>s,</sub> Mi<sub>c</sub>h<sub>ae</sub>l Mitz<sub>e</sub>nm<sub>ac</sub>h<sub>e</sub>r<sub>,</sub> Y<sub>uva</sub>l R<sub>a</sub>b<sub>a</sub>ni<sub>, a</sub>nd D<sub>av</sub>id<sub>e . . . ,</sub> 121–1<sub>.</sub>

[15] Arnaud Da Costa, François Laroussinie, and Nicolas Marke<sub>y</sub>. 2010. ATL with strategy contexts: Expressiveness and Model Checking.. In 30th {C} onference on {F} oundations of{S} oftware {T} echnology and {T} heoretical {C} omputer {S} cience ({FSTTCS}’10), Vol. 8. 120–132.

[16] Giuse<sub>pp</sub>e De Giacomo, Yves Les<sub>p</sub>érance, and Adrian R Pearce. 2010. Situation Calc<sub>u</sub>l<sub>u</sub>s-Based Pro<sub>g</sub>rams for Re<sub>p</sub>resentin<sub>g</sub> and Reasonin<sub>g</sub> abo<sub>u</sub>tGame Str<sub>u</sub>ct<sub>u</sub>res. Proc. ofKR (2010), 445–455.

[17] Giuse<sub>pp</sub>e De Giacomo, Moshe Y Vardi, et al. 2015. S<sub>y</sub>nthesis for LTL and LDL on finite traces. In Proceedin s ofthe Twent -Fourth International Joint Conference on Artificial Intelligence, IJCAI 2015. AAAI Press, 1558–1564

[18] Christian Schroeder De Witt, Tarun Gu<sub>p</sub>ta, Den<sub>y</sub>s Makoviichuk, Viktor Makovi<sub>y</sub> <sub>c</sub>h<sub>u</sub>k<sub>,</sub> Phili<sub>p</sub> HS T<sub>o</sub>rr<sub>,</sub> Min<sub>g</sub>f<sub>e</sub>i S<sub>u</sub>n<sub>, a</sub>nd Shim<sub>o</sub>n Whit<sub>eso</sub>n<sub>.</sub> 2020<sub>.</sub> I<sub>s</sub> ind<sub>epe</sub>nd<sub>e</sub>nt l<sub>earn</sub>i<sub>ng</sub> <sub>a</sub>ll <sub>you</sub> <sub>nee</sub>d i<sub>n</sub> th<sub>e</sub> <sub>s</sub>t<sub>arcra</sub>ft <sub>mu</sub>lti<sub>-agen</sub>t <sub>c</sub>h<sub>a</sub>ll<sub>enge</sub>? <sub>ar</sub>Xi<sub>v</sub> <sub>prepr</sub>i<sub>n</sub>t arXiv:2011.09533 (2020).

[19] Lasse Es<sub>p</sub>eholt, Hubert So<sub>y</sub>er, Remi Munos, Karen Simon<sub>y</sub>an, Vlad Mnih, Tom W<sub>a</sub>rd<sub>,</sub> Y<sub>o</sub>t<sub>a</sub>m D<sub>o</sub>r<sub>o</sub>n<sub>,</sub> Vl<sub>a</sub>d Fir<sub>o</sub>i<sub>u,</sub> Tim H<sub>a</sub>rl<sub>ey,</sub> I<sub>a</sub>in D<sub>u</sub>nnin<sub>g,</sub> Sh<sub>a</sub>n<sub>e</sub> L<sub>egg, a</sub>nd K<sub>o</sub>r<sub>ay</sub> K<sub>avu</sub>k<sub>cuog</sub>l<sub>u.</sub> 2018<sub>.</sub> IMPALA: S<sub>ca</sub>l<sub>a</sub>bl<sub>e</sub> Di<sub>s</sub>trib<sub>u</sub>t<sub>e</sub>d D<sub>eep</sub>-RL <sub>w</sub>ith Im<sub>po</sub>r tance Weighted Actor-Learner Architectures. In Proceedings ofthe 35th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 80). PMLR, 1407–1416. htt<sub>p</sub>s://<sub>p</sub>roceedin<sub>g</sub>s.mlr.<sub>p</sub>ress/v80/es<sub>p</sub>eholt18a.htm

[20] Dana Fisman, Orna Ku<sub>p</sub>ferman, and Yoad Lusti<sub>g</sub>. 2010. Rational s<sub>y</sub>nthesis. In International Conference on Tools and Algorithmsfor the Construction and Analysis ofSystems. Springer, 190–204.

[21] Drew Fudenber<sub>g</sub> and Jean Tirole. 1991. Game Theor<sub>y</sub>. MIT Press, Cambrid<sub>g</sub>e, MA<sub>.</sub>

[22] Rustam Galimullin and Thomas <sup>Å</sup><sub>g</sub>otnes. 2021. D<sub>y</sub>namic coalition lo<sub>g</sub>ic: <sub>g</sub>rantin<sub>g</sub> <sub>an</sub>d <sub>revo</sub>ki<sub>ng</sub> di<sub>c</sub>t<sub>a</sub>t<sub>or</sub>i<sub>a</sub>l <sub>powers.</sub> I<sub>n</sub> I<sub>n</sub>t<sub>erna</sub>ti<sub>ona</sub>l W<sub>or</sub>k<sub>s</sub>h<sub>op</sub> <sub>on</sub> L<sub>og</sub>i<sub>c,</sub> R<sub>a</sub>ti<sub>ona</sub>lit<sub>y</sub> and Interaction. S<sub>p</sub>rin<sub>g</sub>er<sub>,</sub> 88–101.

[23] Javier Garcıa and Fernando Fernández. 2015. A com<sub>p</sub>rehensive surve<sub>y</sub> on safe reinforcement learning. Journal ofMachine Learning Research 16, 1 (2015), 1437– 1480.

[24] Ernst Moritz Hahn, Mateo Perez, Sven Schewe, Fabio Somenzi, Ashutosh Trivedi, an<sup>d</sup> Domini<sup>k</sup> Wojtcza<sup>k</sup>. 2019. Omega-regu<sup>l</sup>ar o<sup>b</sup>jectives in mo<sup>d</sup>e<sup>l</sup>-<sup>f</sup>ree rein forcement learning. In International conference on tools and algorithms for the construction and analysis of systems. Springer, 395–412.

[25] Ernst Moritz Hahn, Mateo Perez, Sven Schewe, Fabio Somenzi, Ashutosh Trive<sup>d</sup>i, an<sup>d</sup> Domini<sup>k</sup> Wojtcza<sup>k</sup>. 2024. Mu<sup>l</sup>ti-Agent Rein<sup>f</sup>orcement Learning for Alternating-Time Logic. In 27th European Conference on Artificial Intelligence, ECAI 2024. IOS<sub>,</sub> 1680–1687.

[26] Lewis Hammond, Alessandro Abate, Julian Gutierrez, and Michael Wooldrid<sub>g</sub>e. 2021. Multi-a<sub>g</sub>ent reinforcement learnin<sub>g</sub> with tem<sub>p</sub>oral lo<sub>g</sub>ic s<sub>p</sub>ecifications. arXiv preprint arXiv:2102.00582 (2021).

[27] Juemin<sub>g</sub> Hu, Jean-Ra<sub>p</sub>haël Ga<sub>g</sub>lione, Yanze Wan<sub>g</sub>, Zhe Xu, Ufuk To<sub>p</sub>cu, and Y<sub>o</sub>n<sub>g</sub>min<sub>g</sub> Li<sub>u.</sub> 2023<sub>.</sub> R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt l<sub>ea</sub>rnin<sub>g w</sub>ith r<sub>ewa</sub>rd m<sub>ac</sub>hin<sub>es</sub> in <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> <sub>g</sub>ames. arXiv preprint arXiv:2305.17372 (2023).

[28] Junlin<sub>g</sub> Hu and Michael P Wellman. 2003. Nash Q-learnin<sub>g</sub> for <sub>g</sub>eneral-sum stochastic games. Journal ofmachine learning research 4, Nov (2003), 1039–1069.

[29] Yunhan Huan<sub>g</sub> and Quan<sub>y</sub>an Zhu. 2022. Game-theoretic frameworks for e<sub>p</sub>idemic s<sub>p</sub>readin<sub>g</sub> and human decision-makin<sub>g</sub>: A review. D<sub>y</sub>namic Games and Applications 12, 1 (2022), 7–48.

[30] Kishor Jothimuru<sub>g</sub>an, Su<sub>g</sub>uman Bansal, Osbert Bastani, and Rajeev Alur. 2022. S<sub>p</sub>ecification-<sub>g</sub>uided learnin<sub>g</sub> of nash e<sub>q</sub>uilibria with hi<sub>g</sub>h social welfare. In International Conference on Computer Aided Verification. Springer, 343–363.

[31] Marta Kwiatkowska. 2016. Model checkin<sub>g</sub> and strate<sub>gy</sub> s<sub>y</sub>nthesis for stochastic <sub>ga</sub>m<sub>es</sub>: Fr<sub>o</sub>m th<sub>eo</sub>r<sub>y</sub> t<sub>o p</sub>r<sub>ac</sub>ti<sub>ce.</sub> In ICALP2016: 43rd International Collo<sub>q</sub>uium on Automata<sub>,</sub> Lan<sub>g</sub>ua<sub>g</sub>es and Pro<sub>g</sub>rammin<sub>g</sub>. Schloss Da<sub>g</sub>stuhl.

[32] Marta Kwiatkowska, Gethin Norman, David Parker, and Gabriel Santos. 2020. Multi-<sub>p</sub>la<sub>y</sub>er e<sub>q</sub>uilibria verification for concurrent stochastic <sub>g</sub>ames. In International Conference on Quantitative Evaluation ofSystems. Springer, 74–95.

[33] Marta Kwiatkowska, Gethin Norman, David Parker, and Gabriel Santos. 2020. PRISM-<sub>g</sub>ames 3.0: Stochastic <sub>g</sub>ame <sub>v</sub>erification <sub>w</sub>ith conc<sub>u</sub>rrenc<sub>y,</sub> e<sub>qu</sub>ilibria and time. In International Conference on Computer Aided Verification. Springer, 475–487.

[34] Marta Kwiatkowska and David Parker. 2013. Automated verification and strate<sub>gy</sub> synthesis for probabilistic systems. In Automated Technology for Verification and A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub>: 11th I<sub>n</sub>t<sub>erna</sub>ti<sub>ona</sub>l S<sub>ympos</sub>i<sub>um,</sub> ATVA 2013<sub>,</sub> H<sub>ano</sub>i<sub>,</sub> Vi<sub>e</sub>t<sub>nam,</sub> O<sub>c</sub>t<sub>o</sub>b<sub>er</sub> 15-18<sub>,</sub> 2013. Proceedin<sub>g</sub>s. S<sub>p</sub>rin<sub>g</sub>er<sub>,</sub> 5–22.

[35] Marc Lanctot, Vinicius Zambaldi, Audrunas Grusl<sub>y</sub>s, An<sub>g</sub>eliki Lazaridou, Karl Tu<sub>y</sub>ls, Julien Pérolat, David Silver, and Thore Grae<sub>p</sub>el. 2017. A unified <sub>g</sub>ame theoretic a<sub>pp</sub>roach to multia<sub>g</sub>ent reinforcement learnin<sub>g</sub>. Advances in neural information processin<sub>g</sub> s<sub>y</sub>stems 30 (2017).

[36] Michael L Littman. 1994. Markov <sub>g</sub>ames as a framework for multi-a<sub>g</sub>ent rein f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng.</sub> I<sub>n</sub> M<sub>ac</sub>hi<sub>ne</sub> l<sub>earn</sub>i<sub>ng procee</sub>di<sub>ngs</sub> 1994<sub>.</sub> El<sub>sev</sub>i<sub>er,</sub> 157<sub>–</sub>163<sub>.</sub>

[37] Fabrizio M Ma<sub>gg</sub>i, Marco Montali, and Rafael Peñaloza. 2020. Tem<sub>p</sub>oral lo<sub>g</sub>ics over finite traces with uncertainty. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 34. 10218–10225.

[38] Zohar Manna and Amir Pnueli. 1990. A hierarch<sub>y</sub> of tem<sub>p</sub>oral <sub>p</sub>ro<sub>p</sub>erties (invited paper, 1989). In Proceedings of the ninth annual ACM symposium on Principles of di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d <sub>compu</sub>ti<sub>ng.</sub> 377<sub>–</sub>410<sub>.</sub>

[39] Bharat Manvi and Easwar Subramanian. 2023. A Nash E<sub>q</sub>uilibrium Solution for Periodic Double Auctions. In 2023 62nd IEEE Conference on Decision and Control (CDC). IEEE, 209–214.

[40] John Nash. 1950. E<sub>q</sub>uilibrium <sub>p</sub>oints in n-<sub>p</sub>erson <sub>g</sub>ames. Proceedin<sub>g</sub>s of the National Academ<sub>y</sub> ofSciences 36, 1 (1950), 48–49.

[41] Gas<sub>p</sub>ard Ohlmann, Edwin Hamel-De le Court, and Francesco Belardinelli. 2025. S<sub>y</sub>nthesis of Safet<sub>y</sub> S<sub>p</sub>ecifications for Probabilistic S<sub>y</sub>stems. arXiv <sub>p</sub>re<sub>p</sub>rint arXiv:2511.16579 (2025)

[42] Martin L Puterman. 2014. Markov decision processes: discrete stochastic d<sub>y</sub>namic pro<sub>g</sub>rammin<sub>g</sub>. John Wile<sub>y</sub> & Sons.

[43] Senthil Rajasekaran, Su<sub>g</sub>uman Bansal, and Moshe Y. Vardi. 2023. Multi-A<sub>g</sub>ent S stems with Quantitative Satisficin Goals. (8 2023), 280–288. doi:10.24963/ ijcai.2023/32 Main Trac<sup>k</sup>.

[44] John Schulman, Fili<sub>p</sub> Wolski, Prafulla Dhariwal, Alec Radford, and Ole<sub>g</sub> Klimov. <sup>2017</sup>. <sup>P</sup>rox<sup>i</sup>ma<sup>l</sup> po<sup>li</sup>cy op<sup>ti</sup>m<sup>i</sup>za<sup>ti</sup>on a<sup>l</sup>gor<sup>ith</sup>ms. ar<sup>Xi</sup>v prepr<sup>i</sup>n<sup>t</sup> ar<sup>Xi</sup>v:<sup>1707</sup>.<sup>06347</sup> (2017).

[45] Reinhard Selten. 1975. Reexamination of the <sub>p</sub>erfectness conce<sub>p</sub>t for e<sub>q</sub>uilibrium points in extensive games. International Journal of Game Theory 4, 1 (1975), 25–55.

[46] Satinder Sin<sub>g</sub>h, Tommi Jaakkola, Michael L Littman, and Csaba Sze<sub>p</sub>esvári. 2000. Conver<sub>g</sub>ence results for sin<sub>g</sub>le-ste<sub>p</sub> on-<sub>p</sub>olic<sub>y</sub> reinforcement-learnin<sub>g</sub> al<sub>g</sub>orithms. Machine learnin<sub>g</sub> 38, 3 (2000), 287–308.

[47] Fu Son<sub>g</sub>, Yedi Zhan<sub>g</sub>, Taolue Chen, Yu Tan<sub>g</sub>, and Zhiwu Xu. 2019. Probabilistic alternating-time �-calculus. In Proceedings ofthe AAAI Conference on Artificial I<sub>n</sub>t<sub>e</sub>lli<sub>gence,</sub> V<sub>o</sub>l<sub>.</sub> 33<sub>.</sub> 6179<sub>–</sub>6186<sub>.</sub>

[48] Sriram Gana<sub>p</sub>athi Subramanian, Tor<sub>y</sub>n Q. Klassen, and Sheila A. McIlraith. 2026<sub>.</sub> M<sub>u</sub>lti-A<sub>ge</sub>nt R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt L<sub>ea</sub>rnin<sub>g w</sub>ith R<sub>ewa</sub>rd M<sub>ac</sub>hin<sub>es</sub> f<sub>o</sub>r Mix<sub>e</sub>d Cooperative-Competitive Environments. Reinforcement Learning Journal 7 (2026).

[49] Min<sub>g</sub> Tan. 1993. Multi-A<sub>g</sub>ent Reinforcement Learnin<sub>g</sub>: Inde<sub>p</sub>endent versus Cooperative Agents. In Proceedings of the Tenth International Conference on Machine Learnin<sub>g</sub>. Mor<sub>g</sub>an Kaufmann<sub>,</sub> 330–337.

[50] Finbarr Timbers, Nolan Bard, Edward Lockhart, Marc Lanctot, Martin Schmid, Neil Burch, Julian Schrittwieser, Thomas Hubert, and Michael Bowlin<sub>g</sub>. 2020. A<sub>pp</sub>roximate ex<sub>p</sub>loitabilit<sub>y</sub>: Learnin<sub>g</sub> a best res<sub>p</sub>onse in lar<sub>g</sub>e <sub>g</sub>ames. arXiv preprint arXiv:2004.09677 (2020).

[51] John Tsitsiklis and Benjamin Van Ro<sub>y</sub>. 1996. Anal<sub>y</sub>sis of tem<sub>p</sub>oral-difference learning with function approximation. Advances in neural information processing s<sub>y</sub>stems 9 (1996).

[52] Michael Ummels. 2006. Rational behaviour and strate<sub>gy</sub> construction in infi nite multiplayer games. In International Conference on Foundations ofSoftware Technolo<sub>gy</sub> and Theoretical Com<sub>p</sub>uter Science. S<sub>p</sub>rin<sub>g</sub>er<sub>,</sub> 212–223.

[53] Giovanni Varricchione, Natasha Alechina, Mehdi Dastani, and Brian Lo<sub>g</sub>an. 2023. S<sub>y</sub>nthesisin<sub>g</sub> reward machines for coo<sub>p</sub>erative multi-a<sub>g</sub>ent reinforcement learning. In European Conference on Multi-Agent Systems. Springer, 328–344.

[54] Cameron Voloshin, Abhinav Verma, and Yison<sub>g</sub> Yue. 2023. Eventual discountin<sub>g</sub> temporal logic counterfactual experience replay. In International Conference on Machine Learnin <sub>.</sub> PMLR<sub>,</sub> 35137–35150<sub>.</sub>

[55] Dirk Walther, Wiebe van der Hoek, and Michael Wooldrid<sub>g</sub>e. 2007. Alternatin<sub>g</sub>- time temporal logic with explicit strategies. In Proceedings ofthe 11th conference on Theoretical aspects ofrationality and knowledge. 269–278.

[56] Isaac E Weintraub, Meir Pachter, and Elo<sub>y</sub> Garcia. 2020. An introduction to <sub>p</sub>ursuit-evasion diferential <sub>g</sub>ames. In 2020 American Control Conference (ACC). IEEE<sub>,</sub> 1049–1066.

[57] Yin<sub>g</sub> Wen, Yaodon<sub>g</sub> Yan<sub>g</sub>, Rui Luo, Jun Wan<sub>g</sub>, and Wei Pan. 2019. Probabilistic R<sub>ecu</sub>r<sub>s</sub>i<sub>ve</sub> R<sub>easo</sub>nin<sub>g</sub> f<sub>o</sub>r M<sub>u</sub>lti-A<sub>ge</sub>nt R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt L<sub>ea</sub>rnin<sub>g.</sub> In International Conference on Learning Representations.

[58] Michael Wooldrid<sub>g</sub>e, Julian Gutierrez, Paul Harrenstein, Enrico Marchioni, Gi<sub>use e</sub> P<sub>ere</sub>lli<sub>, an</sub>d Al<sub>ex</sub>i<sub>s</sub> T<sub>oum</sub>i<sub>.</sub> 2016<sub>.</sub> R<sub>a</sub>ti<sub>ona</sub>l <sub>ver</sub>ifi<sub>ca</sub>ti<sub>on:</sub> F<sub>rom mo</sub>d<sub>e</sub>l checking to equilibrium checking. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 30.

[59] Shan Yan<sub>g</sub> and Yan<sub>g</sub> Liu. 2024. Markov <sub>g</sub>ame for CV joint ada<sub>p</sub>tive routin<sub>g</sub> in <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> t<sub>ra</sub>fi<sub>c</sub> <sub>ne</sub>t<sub>wor</sub>k<sub>s:</sub> A <sub>sca</sub>l<sub>a</sub>bl<sub>e</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>approac</sub>h<sub>.</sub> T<sub>ranspor</sub>t<sub>a</sub>ti<sub>on</sub> R<sub>esearc</sub>h Part B: Methodolo<sub>g</sub>ical 189 (2024), 102997.

[60] Yunfei Zhao, Linan Huan<sub>g</sub>, Carol Smidts, and Quan<sub>y</sub>an Zhu. 2020. Finite-horizon semi-Markov <sub>g</sub>ame for time-sensitive attack res<sub>p</sub>onse and <sub>p</sub>robabilistic risk assessment in nuclear power plants. Reliability Engineering & System Safety 201 (2020), 106878.

## A<sub>pp</sub>endix

The a<sub>pp</sub>endix is or<sub>g</sub>anised as follows:   
A<sub>ppe</sub>ndix A: Th<sub>e</sub> finit<sub>e</sub>-tr<sub>ace</sub> <sub>se</sub>m<sub>a</sub>nti<sub>cs</sub> <sub>o</sub>f pltl<sub>�</sub>   
A<sub>pp</sub>endix B: Overview of our a<sub>pp</sub>roach   
A<sub>ppe</sub>ndix C: En<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt d<sub>esc</sub>ri<sub>p</sub>ti<sub>o</sub>n<sub>s</sub> <sub>a</sub>nd h<sub>a</sub>nd<sub>c</sub>r<sub>a</sub>ft<sub>e</sub>d r<sub>ewa</sub>rd<sub>s</sub>   
A<sub>ppe</sub>ndix D: Al<sub>go</sub>rithm d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>a</sub>nd <sub>p</sub>r<sub>oo</sub>f<sub>s</sub>   
A<sub>pp</sub>endix E: Ex<sub>p</sub>erimental results   
A<sub>pp</sub>endix F: Extended literat<sub>u</sub>re re<sub>v</sub>ie<sub>w</sub>   
<sup>A</sup><sub>pp</sub>en<sup>di</sup>x <sup>G</sup>: <sup>H</sup><sub>yp</sub>er<sub>p</sub>arameters

## A pltl<sub>�</sub> S<sub>e</sub>m<sub>a</sub>nti<sub>cs</sub>

Definition 13 (pltl<sub>�</sub> Semantics). Fix a labelled stochastic <sub>g</sub>ame ℳ and a horizon of� transitions (label traces have at most $H + 1$ positions). Let $\mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } }$ b<sub>e</sub> th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>measure</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> profile ⃗� on complete finite paths $\mathsf { F P a t h } _ { \mathcal { M } , s } ^ { \mathrm { c o m p } }$ . For any finite trajectory $\pi = s _ { 0 } a _ { 0 } s _ { 1 } \cdot \cdot \cdot s _ { \ell } \in { \sf F P a t h } _ { \mathcal { M } , s } ^ { \mathrm { c o m p } } , l e t \tau ( \pi ) = \mathcal { L } ( s _ { 0 } ) \cdot \cdot \cdot \mathcal { L } ( s _ { \ell } ) \in$ Γ<sup>ℓ+1</sup>.

ltl satisfaction (finite label traces � ∈ Γ<sup>+</sup>):   
�, � |= ⊤   
�, � |= � ↔� ∈ �[�]   
�, � |= ¬� ↔�, � ̸|= �   
�, � |= �<sub>1</sub> ∧ �<sub>2</sub> $ ( \tau , i \Vdash ^ { \prime } \varphi _ { 1 } ) \land ( \tau , i \Vdash \varphi _ { 2 } )$   
�, � |= X� $ i + 1 < | \tau | \wedge \tau , i + 1 \lceil  = \varphi$   
�, � |= �<sub>1</sub>U�<sub>2</sub> $ \exists j \in [ i , \dot { | \tau | } - 1 ] . ~ ( \tau , j \dot { | = { \dot { \varphi } } _ { 2 } } \wedge$   
$\forall k \in [ i , j - 1 ] , \tau , k \in \varphi _ { 1 } \big )$   
P<sub>a</sub>th <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on:</sub>   
ℳ, � |= � ↔�(�), 0 |= �   
P<sub>ro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c mo</sub>d<sub>a</sub>lit<sub>y:</sub>   
$\begin{array} { r l } { | \mathcal { M } , s , \vec { \sigma } | = \mathsf { P } _ { \ J \ J \ J } [ \varphi ] } & { { }  \mathsf { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } } ( \{ \pi \in \mathsf { F P a t h } _ { \mathcal { M } , s } ^ { \mathrm { c o m p } } \ |  } \end{array}$   
$\mathcal { M } , \pi \models \varphi \models \ v C$

## B O<sub>ve</sub>r<sub>v</sub>i<sub>ew</sub>

Figure 3 summarises our approac<sup>h</sup>. Given eac<sup>h</sup> agent<sup>’</sup>s o<sup>b</sup>jective an<sup>d</sup> <sub>a</sub> t<sub>arge</sub>t <sub>so</sub>l<sub>u</sub>ti<sub>on concep</sub>t<sub>, we comp</sub>il<sub>e</sub> th<sub>e pa</sub>th f<sub>ormu</sub>l<sub>as</sub> i<sub>n</sub>t<sub>o rewar</sub>d automata and form the <sub>p</sub>roduct stochastic <sub>g</sub>ame. We train a sin<sub>g</sub>le joint pro<sup>fil</sup>e using mo<sup>d</sup>e<sup>l</sup>-<sup>f</sup>ree MARL, wit<sup>h</sup> optiona<sup>l</sup> counter<sup>f</sup>actua<sup>l</sup> re<sub>p</sub>la<sub>y</sub>, and evaluate its satisfaction <sub>p</sub>robabilities and NashConv<sup>≈</sup>.

## C En<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt<sub>s</sub>

W<sub>e cons</sub>id<sub>er</sub> 8 b<sub>enc</sub>h<sub>mar</sub>k i<sub>ns</sub>t<sub>ances</sub> d<sub>rawn</sub> f<sub>rom</sub> fi<sub>ve env</sub>i<sub>ronmen</sub>t f<sub>am</sub>ili<sub>es.</sub> T<sub>a</sub>bl<sub>e</sub> 2 <sub>summar</sub>i<sub>ses</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>su</sub>it<sub>e.</sub>

Fi<sub>g</sub>ure 4 shows the s<sub>p</sub>atial environments.

Ob<sub>serva</sub>ti<sub>ons.</sub> A<sub>gen</sub>t<sub>s</sub> <sub>o</sub>b<sub>serve</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>sym</sub>b<sub>o</sub>li<sub>c</sub> <sub>gr</sub>id<sub>s:</sub> <sub>eac</sub>h <sub>gr</sub>id <sub>recor</sub>d<sub>s,</sub> <sub>ce</sub>ll b<sub>y</sub> <sub>ce</sub>ll<sub>,</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> <sub>o</sub>f <sub>one</sub> l<sub>ayer</sub> <sub>o</sub>f th<sub>e</sub> <sub>scene</sub> (terrain, avatars and their orientations, coins, a<sub>pp</sub>les, claims, beams, etc), alon<sub>g</sub>side scalar side channels. The <sub>g</sub>rids are flattened and scaled to [0, 1], and the index of the agent’s reward-automaton state is a<sub>pp</sub>ended alon<sub>g</sub>side s<sub>p</sub>atial coordinates.

## C<sub>.</sub>1 St<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> P<sub>u</sub>r<sub>su</sub>it-E<sub>vas</sub>i<sub>o</sub>n

W<sub>e p</sub>r<sub>ov</sub>id<sub>e a s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c va</sub>ri<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> Pursuit Evasion environment in the literature [56], which is also referred to as Guard Gridworld. The sli<sub>pp</sub>er<sub>y g</sub>ridworld is instantiated with one <sub>g</sub>uar<sup>d</sup>, $\mathsf { p l a y e r } _ { 0 } ,$ <sub>, an</sub>d <sub>e</sub>ith<sub>er one or</sub> t<sub>wo</sub> i<sub>n</sub>t<sub>ru</sub>d<sub>ers.</sub> I<sub>n</sub> th<sub>e c</sub>l<sub>ass</sub>i<sub>c</sub> task, the <sub>g</sub>uard satisfies G¬goal\_reached, whereas ever<sub>y</sub> intruder satisfies Fgoal\_reached ∧ G¬collision\_ever. In the ca<sub>p</sub>ture tas<sup>k</sup>, t<sup>h</sup>e guar<sup>d</sup> sat<sup>i</sup>s<sup>fi</sup>es Fguard\_intruder\_collision, w<sup>h</sup>ereas each intruder satisfies G¬guard\_intruder\_collision ∧ G¬collision\_ever. Each move is subject to a role-s<sub>p</sub>ecific sli<sub>p</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y, so</sub> th<sub>e env</sub>i<sub>ronmen</sub>t <sub>rema</sub>i<sub>ns s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c un</sub>d<sub>er a</sub> fi<sub>xe</sub>d joint po<sup>l</sup>icy.

Handcrafted rewards. Let $D _ { t }$ d<sub>eno</sub>t<sub>e ep</sub>i<sub>so</sub>d<sub>e comp</sub>l<sub>e</sub>ti<sub>on a</sub>ft<sub>er</sub> ste<sub>p</sub> �, and let $G _ { t } , C _ { t } , K _ { t } ,$ <sub>, an</sub>d $E _ { t }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> B<sub>oo</sub>l<sub>ean</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> goal\_reached, collision, guard\_intruder\_collision, an<sup>d</sup> collision\_ever, respect<sup>i</sup>ve<sup>l</sup>y. <sup>F</sup>or t<sup>h</sup>e c<sup>l</sup>ass<sup>i</sup>c <sup>b</sup>enc<sup>h</sup>mar<sup>k</sup>, t<sup>h</sup>e <sub>guar</sub>d <sub>rewar</sub>d i<sub>s</sub>

$$
\begin{array} { r } { \tilde { r } _ { t , \mathrm { c l a s s i c } } ^ { 0 } = \left\{ \begin{array} { l l } { - 1 , } & { G _ { t } , } \\ { 1 , } & { D _ { t } \wedge \neg G _ { t } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}
$$

<sub>an</sub>d <sub>eac</sub>h i<sub>n</sub>t<sub>ru</sub>d<sub>er</sub> $i > 0$ r<sub>ece</sub>i<sub>ves</sub>

$$
\tilde { r } _ { t , \mathrm { c l a s s i c } } ^ { i } = { \bf 1 } [ G _ { t } ] - { \bf 1 } [ C _ { t } ] .
$$

F<sub>or</sub> th<sub>e cap</sub>t<sub>ure</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>,</sub> th<sub>e guar</sub>d <sub>rewar</sub>d i<sub>s</sub>

$$
\tilde { r } _ { t , \mathrm { c a p t u r e } } ^ { 0 } = \left\{ \begin{array} { l l } { 1 , } & { K _ { t } , } \\ { - 0 . 5 , } & { D _ { t } \wedge \neg K _ { t } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

<sub>an</sub>d <sub>eac</sub>h i<sub>n</sub>t<sub>ru</sub>d<sub>er</sub> $i > 0$ r<sub>ece</sub>i<sub>ves</sub>

$$
\tilde { r } _ { t , \mathrm { c a p t u r e } } ^ { i } = \left\{ \begin{array} { l l } { - 1 , } & { K _ { t } \vee E _ { t } , } \\ { 1 , } & { D _ { t } \wedge \neg K _ { t } \wedge \neg E _ { t } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

## C.2 Coins

Thi<sub>s</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>uses a</sub> t<sub>wo-p</sub>l<sub>ayer a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on o</sub>f th<sub>e</sub> Me<sub>l</sub>t<sub>in</sub>gPot coins substrate wit<sup>h</sup> a 256-step <sup>h</sup>orizon and coin regrowt<sup>h</sup> rate 0.005. Each <sub>p</sub>la<sub>y</sub>er owns one coin colour. Writin<sub>g</sub> �<sub>�</sub> and $w _ { i }$ f<sub>or</sub> the indexed <sub>p</sub>ro<sub>p</sub>ositions that <sub>p</sub>la<sub>y</sub>er � collected a matchin<sub>g</sub> or mismatc<sup>h</sup>ing coin, respective<sup>l</sup>y, its Own Coin First o<sup>b</sup>jective is

$$
\varphi _ { i } = \lnot w _ { i } \lor m _ { i } , \qquad i \in \{ 0 , 1 \} .
$$

T<sup>h</sup>e strong-unti<sup>l</sup> o<sup>b</sup>jective requires a matc<sup>h</sup>ing co<sup>ll</sup>ection to occur <sub>an</sub>d f<sub>or</sub>bid<sub>s</sub> <sub>a</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>hi<sub>ng</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> b<sub>e</sub>f<sub>ore</sub>h<sub>an</sub>d<sub>.</sub> C<sub>o</sub>i<sub>n</sub> <sub>consump-</sub> tion and re<sub>g</sub>rowth create strate<sub>g</sub>ic externalities even thou<sub>g</sub>h the f<sub>ormu</sub>l<sub>ae</sub> <sub>are</sub> <sub>symme</sub>t<sub>r</sub>i<sub>c.</sub>

Handcrafted reward. On each coin collection, the collector re-<sub>ce</sub>i<sub>ves</sub>

$$
\tilde { r } _ { t } ^ { i } = \mathbf { 1 } [ m _ { i } ] - \mathbf { 1 } [ w _ { i } ] .
$$

Th<sub>e o</sub>th<sub>er agen</sub>t <sub>rece</sub>i<sub>ves no rewar</sub>d f<sub>or</sub> th<sub>a</sub>t <sub>co</sub>ll<sub>ec</sub>ti<sub>on.</sub>

## C.3 Matrix Sta<sub>g</sub> Hunt

This benchmark uses a re<sub>p</sub>eated two-<sub>p</sub>la<sub>y</sub>er matrix-<sub>g</sub>ame version of the classic Stag Hunt. At each of 20 sta<sub>g</sub>es, both a<sub>g</sub>ents choose <sup>b</sup>etween Hare an<sup>d</sup> Stag. <sup>Th</sup>e pu<sup>bli</sup>c state recor<sup>d</sup>s t<sup>h</sup>e prev<sup>i</sup>ous joint action an<sup>d</sup> w<sup>h</sup>et<sup>h</sup>er t<sup>h</sup>e agents c<sup>h</sup>ose <sup>d</sup>i<sup>f</sup>erent actions at t<sup>h</sup>at <sub>s</sub>t<sub>age.</sub> Th<sub>e</sub> b<sub>ase env</sub>i<sub>ronmen</sub>t t<sub>runca</sub>t<sub>es on</sub>l<sub>y a</sub>t th<sub>e</sub> h<sub>or</sub>i<sub>zon, an</sub>d t<sup>h</sup>e <sup>d</sup>e<sup>f</sup>au<sup>l</sup>t stage payo<sup>f</sup>s are s<sup>h</sup>own <sup>i</sup>n <sup>Fi</sup>gure 5: mutua<sup>l</sup> Stag <sup>i</sup>s payo<sup>f</sup>-<sup>d</sup>om<sup>i</sup>nant, mutua<sup>l</sup> Hare <sup>i</sup>s t<sup>h</sup>e sa<sup>f</sup>er coor<sup>di</sup>nate<sup>d</sup> outcome, an<sup>d</sup> un<sup>il</sup>atera<sup>l</sup> Stag rece<sup>i</sup>ves t<sup>h</sup>e <sup>f</sup>a<sup>il</sup>e<sup>d</sup>-<sup>h</sup>unt payo<sup>f</sup>.

![](images/78ebaef612c45050568d6069d2a8483335f55a1a75e021c31c85e94889da91d6.jpg)

Fi<sub>g</sub>ure 3: Overview of our a<sub>pp</sub>roach to automata-based rational s<sub>y</sub>nthesis.
<table><tr><td>Environment</td><td>Selected benchmark</td><td>Agents</td><td>Roles</td><td>High-level objective</td></tr><tr><td>STOCHASTIC PURSUIT-EVASION</td><td>Classic (2-player)</td><td>2</td><td>Guard / intruder</td><td>Guard prevents a breach; the intruder reaches the goal without collision.</td></tr><tr><td>STOCHASTIC PURSUIT-EVASION</td><td>Capture (2-player)</td><td>2</td><td>Guard / intruder</td><td>Guard seeks contact; the intruder avoids guard contact and all collisions.</td></tr><tr><td>STOCHASTIC PURSUIT-EVASION</td><td>Classic (3-player)</td><td>3</td><td>Guard / two intruders</td><td>The opposed classic objectives are instantiated for both intruders.</td></tr><tr><td>STOCHASTIC PURSUIT-EVASION</td><td>Capture (3-player)</td><td>3</td><td>Guard / two intruders</td><td>The opposed capture/stealth objectives are instantiated for both intruders.</td></tr><tr><td>CoOINS</td><td>Own Coin First (2-player)</td><td>2</td><td>Two colour owners</td><td>Each player must collect its own colour before collecting the other colour.</td></tr><tr><td>MATRIX STAG HUNT</td><td>Sustained Stag (2-player)</td><td>2</td><td>Two symmetric players</td><td>Coordinate on a sustained mutual Stag convention.</td></tr><tr><td>COMMONS HARVEST</td><td>Safe Harvest (4-player)</td><td>4</td><td>Four indexed harvesters</td><td>Each player collects an apple while avoiding being zapped.</td></tr><tr><td>TERRITORY</td><td>Role Claim (2-player)</td><td>2</td><td>Two competing claimants</td><td>Each player tries to make the first sole resource claim.</td></tr></table>

Table 2: Our eight environments and their high-level objectives and roles. Objectives that share a schema remain agent-indexed unless the<sub>y</sub> are s<sub>y</sub>mmetric.

Bot<sup>h</sup> agents <sup>h</sup>ave t<sup>h</sup>e Sustaine<sup>d</sup> Stag o<sup>b</sup>jective an<sup>d</sup> receive t<sup>h</sup>e same terminal Boolean reward for satisf<sub>y</sub>in<sub>g</sub>

$$
\begin{array} { c } { \varphi _ { i } = \mathsf { F } \big ( \mathtt { b o t h \_ s t a g } \wedge \mathsf { X } ( \mathtt { b o t h \_ s t a g } \wedge \mathsf { G \_ u n m i r r o r e d } ) \big ) , } \\ { i \in \{ 0 , 1 \} . } \end{array}
$$

<sup>Th</sup>e atoms both\_stag an<sup>d</sup> unmirrored are <sup>d</sup>er<sup>i</sup>ve<sup>d</sup> <sup>f</sup>rom t<sup>h</sup>e pre vious stage outcome. T<sup>h</sup>us, t<sup>h</sup>e o<sup>b</sup>jective a<sup>ll</sup>ows <sup>f</sup>ai<sup>l</sup>e<sup>d</sup> coor<sup>d</sup>ination <sub>a</sub>tt<sub>emp</sub>t<sub>s</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> <sub>conven</sub>ti<sub>on</sub> i<sub>s</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d<sub>,</sub> b<sub>u</sub>t <sub>requ</sub>i<sub>res</sub> t<sub>wo</sub> consecut<sup>i</sup>ve Stag-Stag outcomes a<sup>f</sup>ter w<sup>hi</sup>c<sup>h</sup> t<sup>h</sup>e agents c<sup>h</sup>oose the same action at ever<sub>y</sub> remainin<sub>g</sub> sta<sub>g</sub>e. Earlier <sub>p</sub>airs followed <sup>b</sup>y a m<sup>i</sup>smatc<sup>h</sup> are perm<sup>i</sup>tte<sup>d</sup>, as are <sup>l</sup>ater coor<sup>di</sup>nate<sup>d</sup> Hare-Hare outcomes.

Handcrafted reward. We use the native stage payof as the hand-<sub>cra</sub>ft<sub>e</sub>d <sub>rewar</sub>d f<sub>or</sub> S<sub>t</sub>a<sub>g</sub> H<sub>unt:</sub>

$$
\tilde { r } _ { t } ^ { i } = r _ { \mathrm { e n v } , t } ^ { i } = \left\{ \begin{array} { l l } { 3 , } & { a _ { t } ^ { 0 } = a _ { t } ^ { 1 } = \mathrm { H a r e } , } \\ { 4 , } & { a _ { t } ^ { 0 } = a _ { t } ^ { 1 } = \mathbf { S t a g } , } \\ { 0 , } & { a _ { t } ^ { i } = \mathbf { S t a g } \wedge a _ { t } ^ { - i } = \mathrm { H a r e } , } \\ { 3 , } & { a _ { t } ^ { i } = \mathrm { H a r e } \wedge a _ { t } ^ { - i } = \mathbf { S t a g } . } \end{array} \right.
$$

![](images/b33f05f4491cf69631ff3c30ed38f174081f90d8cff07af14d40db1d4bd11946.jpg)  
(a) Stochastic Pursuit-Evasion

![](images/07bca765ad97ef1538ec680f8965032c3d04927242472e8b7a847008b1cba566.jpg)  
(b) Coins

![](images/0830b4414726c061b97a4f551ba9fd4feca914903bccf012b830a4e06686eef9.jpg)  
(c) Territory

![](images/614e4990b25fecd091011bd232da675081e9e0b128222f0664c506c6a476f63c.jpg)  
(d) Commons Harvest  
Fi<sub>g</sub>ure 4: The four s<sub>p</sub>atial environment families used in our ex<sub>p</sub>eriments.

Pla<sub>y</sub>er 2
<table><tr><td colspan="2">Hare</td><td>Stag</td></tr><tr><td>Hare PIyye1</td><td>3,3</td><td>3,0</td></tr><tr><td>Stag</td><td>0,3</td><td>4,4</td></tr></table>

Fi<sub>g</sub>ure 5: Sta<sub>g</sub>e <sub>p</sub>a<sub>y</sub>of matrix for Matrix Stag Hunt. The <sub>or</sub>d<sub>ere</sub>d <sub>pa</sub>i<sub>r</sub> i<sub>n</sub> <sub>eac</sub>h <sub>ce</sub>ll <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>payo</sub>f t<sub>o</sub> Pl<sub>ayer</sub> 1 <sub>an</sub>d Pla<sub>y</sub>er $^ { 2 , }$ res<sub>p</sub>ect<sup>i</sup>ve<sup>l</sup><sub>y</sub>.

## C.4 Commons Harvest

T<sup>h</sup>is <sup>b</sup>enc<sup>h</sup>mar<sup>k</sup> uses t<sup>h</sup>e Me<sup>l</sup>ting Pot su<sup>b</sup>strate commons\_ harvest\_\_partnership, w<sup>h</sup>ere <sup>f</sup>our <sup>h</sup>arvesters compete <sup>f</sup>or <sub>app</sub>l<sub>es</sub> i<sub>n</sub> <sub>a</sub> <sub>s</sub>h<sub>are</sub>d fi<sub>e</sub>ld <sub>an</sub>d <sub>can</sub> <sub>zap</sub> <sub>one</sub> <sub>ano</sub>th<sub>er.</sub> E<sub>ac</sub>h <sub>p</sub>l<sub>ayer</sub> h<sub>as</sub> t<sup>h</sup>e Safe Harvest o<sup>b</sup>jective:

$$
\begin{array} { r } { \varphi _ { i } = \mathsf { G } \mathrm { - p } \mathsf { 1 a y e r \_ i \_ z a p p e d } \wedge \mathsf { F p } \mathsf { 1 a y e r \_ i \_ a p p l e \_ c o l l e } _ { - } \mathsf { c o l l e c t e d } , } \\ { i \in \{ 0 , 1 , 2 , 3 \} . } \end{array}
$$

Handcrafted reward. Each a<sub>g</sub>ent receives +1 for its first a<sub>pp</sub>le collected before bein<sub>g</sub> za<sub>pp</sub>ed. The first za<sub>p</sub> <sub>g</sub>ives −1 and <sub>p</sub>revents f<sub>ur</sub>th<sub>er</sub> <sub>app</sub>l<sub>e</sub> <sub>rewar</sub>d<sub>s,</sub> <sub>cance</sub>lli<sub>ng</sub> <sub>any</sub> <sub>ear</sub>li<sub>er</sub> <sub>app</sub>l<sub>e</sub> <sub>rewar</sub>d<sub>.</sub> A<sub>no</sub>th<sub>er</sub> <sub>p</sub>l<sub>ayer</sub>’<sub>s</sub> <sub>app</sub>l<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> <sub>or</sub> <sub>zap</sub> d<sub>oes</sub> <sub>no</sub>t <sub>a</sub>f<sub>ec</sub>t thi<sub>s</sub> <sub>rewar</sub>d<sub>.</sub> W<sub>e</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> <sub>app</sub>l<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> f<sub>rom</sub> th<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>t<sub>or</sub>’<sub>s</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>na</sub>ti<sub>ve</sub> <sub>payo</sub>f <sub>an</sub>d <sub>zap</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> f<sub>rom</sub> th<sub>e</sub> <sub>su</sub>b<sub>s</sub>t<sub>ra</sub>t<sub>e</sub>’<sub>s</sub> <sub>per-</sub>t<sub>arge</sub>t <sub>zap</sub> <sub>even</sub>t<sub>s.</sub>

## C.5 Territor<sub>y</sub>

T<sup>h</sup>is <sup>b</sup>enc<sup>h</sup>mar<sup>k</sup> uses t<sup>h</sup>e territory\_\_rooms su<sup>b</sup>strate <sup>f</sup>rom Meltin<sub>g</sub> Pot. In the two-<sub>p</sub>la<sub>y</sub>er Role Claim task<sub>, p</sub>la<sub>y</sub>ers race to <sub>c</sub>l<sub>a</sub>i<sub>m</sub> <sub>a</sub> <sub>resource</sub> fi<sub>rs</sub>t<sub>.</sub> L<sub>e</sub>t $c _ { i }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>owners</sub>hi<sub>p-c</sub>h<sub>ange</sub> <sub>even</sub>t player\_i\_resource\_claimed. <sup>Th</sup>e o<sup>b</sup>ject<sup>i</sup>ves are

$$
\varphi _ { 0 } = \neg c _ { 1 } \mathsf { U } \big ( c _ { 0 } \wedge \neg c _ { 1 } \big ) , \qquad \varphi _ { 1 } = \neg c _ { 0 } \mathsf { U } \big ( c _ { 1 } \wedge \neg c _ { 0 } \big ) .
$$

Th<sub>us</sub> <sub>eac</sub>h <sub>p</sub>l<sub>ayer</sub> t<sub>r</sub>i<sub>es</sub> t<sub>o</sub> <sub>ma</sub>k<sub>e</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>so</sub>l<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> Si<sub>mu</sub>lt<sub>aneous</sub> fi<sub>rs</sub>t <sub>c</sub>l<sub>a</sub>i<sub>ms are</sub> ti<sub>es, an</sub>d l<sub>a</sub>t<sub>er c</sub>l<sub>a</sub>i<sub>ms canno</sub>t <sub>reverse</sub> th<sub>e</sub> t<sub>empora</sub>l outcome.

Handcrafted reward. At the first step on which a claim occurs, the sole claimant receives +1 and the o<sub>pp</sub>onent −1. A simultaneous first claim <sub>g</sub>ives both <sub>p</sub>la<sub>y</sub>ers 0. Later claims <sub>g</sub>ive no reward.

## C<sub>.</sub>6 H<sub>an</sub>d<sub>cra</sub>ft<sub>e</sub>d B<sub>ase</sub>li<sub>nes</sub>

We train each handcrafted baseline (hc-IPPO, hc-PR2-IPPO, hc-IQL, hc-PR2-IQL and hc-Nash-Q) on the rewards above, usin<sub>g</sub> the corres<sub>p</sub>ondin<sub>g</sub> learner’s h<sub>yp</sub>er<sub>p</sub>arameters (A<sub>pp</sub>endix G) <sub>an</sub>d fi<sub>ve see</sub>d<sub>s</sub> $\{ 0 , \ldots , 4 \}$ <sub>, w</sub>ith<sub>ou</sub>t <sub>coun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l <sub>rep</sub>l<sub>ay.</sub> Th<sub>e</sub> plt $\mathbf { L } _ { f }$ monitors recor<sup>d</sup> o<sup>b</sup>jective satis<sup>f</sup>action t<sup>h</sup>roug<sup>h</sup>out training, <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> l<sub>earner rece</sub>i<sub>ves</sub> th<sub>e</sub> h<sub>an</sub>d<sub>cra</sub>ft<sub>e</sub>d <sub>rewar</sub>d<sub>.</sub>

We evaluate each fixed <sub>p</sub>rofile on 1000 new e<sub>p</sub>isodes <sub>p</sub>er seed, reporting o<sup>b</sup>jective satis<sup>f</sup>action an<sup>d</sup> Hoe<sup>fd</sup>ing interva<sup>l</sup>s at overa<sup>ll</sup> $\delta = 0 . 0 5$ , as <sup>f</sup>or t<sup>h</sup>e o<sup>b</sup>jective-traine<sup>d</sup> pro<sup>fil</sup>es. Eac<sup>h l</sup>earning-curve <sub>pane</sub>l <sub>mar</sub>k<sub>s</sub> th<sub>e</sub> <sub>s</sub>t<sub>ronges</sub>t h<sub>an</sub>d<sub>cra</sub>ft<sub>e</sub>d b<sub>ase</sub>li<sub>ne</sub>’<sub>s</sub> fi<sub>na</sub>l <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> with a diamond<sub>,</sub> usin<sub>g</sub> the same a<sub>gg</sub>re<sub>g</sub>ation over a<sub>g</sub>ents as the curves. We <sub>p</sub>lace the diamond at the last trainin<sub>g</sub> e<sub>p</sub>isode across th<sub>a</sub>t b<sub>ase</sub>li<sub>ne</sub>’<sub>s see</sub>d<sub>s, cappe</sub>d <sub>a</sub>t th<sub>e en</sub>d <sub>o</sub>f th<sub>e p</sub>l<sub>o</sub>tt<sub>e</sub>d <sub>curves.</sub> W<sub>e</sub> also com<sub>p</sub>ute its NashConv<sup>≈</sup> from im<sub>p</sub>rovements in tem<sub>p</sub>oral satisfaction (A<sub>pp</sub>endix D.3).

## D T<sub>ec</sub>h<sub>n</sub>i<sub>ca</sub>l D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

## D.1 Counterfactual Ex<sub>p</sub>erience Re<sub>p</sub>la<sub>y</sub>

We a<sub>pp</sub>l<sub>y</sub> CER se<sub>p</sub>aratel<sub>y</sub> to each a<sub>g</sub>ent’s deterministic reward <sub>au</sub>t<sub>oma</sub>t<sub>on.</sub> W<sub>e</sub> <sub>appen</sub>d th<sub>e</sub> <sub>au</sub>t<sub>oma</sub>t<sub>on-s</sub>t<sub>a</sub>t<sub>e</sub> id<sub>en</sub>tifi<sub>er</sub> t<sub>o</sub> th<sub>e</sub> <sub>agen</sub>t’<sub>s</sub> observation and store the successor-state <sub>p</sub>ro<sub>p</sub>ositions at each ste<sub>p</sub>. R<sub>e</sub>l<sub>a</sub>b<sub>e</sub>lli<sub>ng</sub> k<sub>eeps</sub> th<sub>e env</sub>i<sub>ronmen</sub>t <sub>ro</sub>ll<sub>ou</sub>t fi<sub>xe</sub>d <sub>an</sub>d <sub>recompu</sub>t<sub>es</sub> th<sub>e</sub> <sub>au</sub>t<sub>oma</sub>t<sub>on</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> f<sub>rom</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ve</sub> <sub>s</sub>t<sub>ar</sub>t <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> W<sub>e</sub> k<sub>eep</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>samp</sub>l<sub>e</sub> <sub>an</sub>d di<sub>scar</sub>d d<sub>up</sub>li<sub>ca</sub>t<sub>e</sub> <sub>re</sub>l<sub>a</sub>b<sub>e</sub>lli<sub>ngs.</sub> O<sub>ur</sub> <sub>proce</sub>d<sub>ure</sub> ada<sub>p</sub>ts Voloshin et al. [54] to finite-trace $\mathbf { L T L } _ { f }$ . In <sub>p</sub>artic<sub>u</sub>lar<sub>,</sub> o<sub>u</sub>r deterministic finite automata require no �-jump transitions used <sub>w</sub>ith Bü<sub>c</sub>hi <sub>au</sub>t<sub>oma</sub>t<sub>a</sub> i<sub>n</sub> ltl<sub>, an</sub>d <sub>rewar</sub>d<sub>s</sub> d<sub>epen</sub>d <sub>on accep</sub>t<sub>ance a</sub>t e<sub>p</sub>isode termination instead of re<sub>p</sub>eated <sub>v</sub>isits to Büchi acce<sub>p</sub>tin<sub>g</sub> states.

Al<sub>gor</sub>ith<sub>m</sub> 2<sub>:</sub> R<sub>e</sub>l<sub>a</sub>b<sub>e</sub>lli<sub>ng a ro</sub>ll<sub>ou</sub>t <sub>c</sub>h<sub>un</sub>k f<sub>or one agen</sub>t   
Input: a<sub>g</sub>ent rewar<sup>d</sup> automaton $A _ { i } = ( M _ { i } , m _ { i } ^ { 0 } , \Gamma , \delta _ { i } , F _ { i } ) ,$   
<sub>ro</sub>ll<sub>ou</sub>t <sub>c</sub>h<sub>un</sub>k $( x _ { 0 } , a _ { 0 } , r _ { 0 } , \ell _ { 0 } , d _ { 0 } ,$   
$\dots , x _ { T - 1 } , a _ { T - 1 } , r _ { T - 1 } , \ell _ { T - 1 } , d _ { T - 1 } , x _ { T } ) ,$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub>   
${ \boldsymbol x } _ { t } = \left( { \boldsymbol o } _ { t } , m _ { t } \right)$ <sub>an</sub>d $\ell _ { t } = \mathcal { L } ( s _ { t + 1 } )$   
<sup>O</sup>ut<sub>p</sub>ut: set $\widetilde { c }$ <sub>o</sub>f <sub>re</sub>l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>c</sub>h<sub>un</sub>k<sub>s</sub>   
<sub>1</sub> Initi<sub>a</sub>li<sub>se</sub> $\widetilde { c }$ <sub>w</sub>ith th<sub>e</sub> f<sub>ac</sub>t<sub>ua</sub>l <sub>c</sub>h<sub>un</sub>k   
2 for each counterfactual start state �˜ $\in M _ { i }$ d<sub>o</sub>   
3 S<sub>e</sub>t $m \gets \tilde { m } _ { 0 }$   
4 f<sub>or</sub> $t = 0 , \ldots , T - 1$ d<sub>o</sub>   
5 Set relabelled observation �˜<sub>�</sub> $ ( o _ { t } , m )$   
6 <sup>Com</sup>p<sup>ute</sup> p<sup>re-reset successor</sup> $m ^ { + } \gets \delta _ { i } ( m , \ell _ { t } )$   
7 S<sub>e</sub>t t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d $\tilde { r } _ { t } \gets \mathbf { 1 } [ d _ { t } ] f _ { i } ( \mathbf { 1 } [ m ^ { + } \in F _ { i } ] )$   
8 Set � $\gets \tilde { m } _ { 0 }$ if $d _ { t } ,$ <sub>e</sub>l<sub>se</sub> $m ^ { + }$   
9 S<sub>e</sub>t <sub>s</sub>t<sub>ore</sub>d <sub>nex</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> $\tilde { x } _ { t + 1 } \gets ( o _ { t + 1 } , m )$   
10 <sub>en</sub>d   
11 Add $( \tilde { x } _ { 0 } , a _ { 0 } , \tilde { r } _ { 0 } , d _ { 0 } , . . . , \tilde { x } _ { T } )$ to $\tilde { c }$ if <sub>non-</sub>d<sub>up</sub>li<sub>ca</sub>t<sub>e</sub>   
<sub>12 en</sub>d   
1<sub>3</sub> return $\widetilde { c }$

Al<sub>g</sub>orithm 3: CER inte<sub>g</sub>ration for re<sub>p</sub>la<sub>y</sub>-based learners   
In<sub>p</sub>ut: re<sub>p</sub>la<sub>y</sub> bufer $\mathcal { D } ,$ factual rollout chunk �   
1 <sup>C</sup>om<sub>p</sub>ute $\tilde { c } \gets$ RelabelChunk(�)   
<sub>2</sub> f<sub>or eac</sub>h $\tilde { c } \in \tilde { { \mathcal { C } } }$ d<sub>o</sub>   
3 $\mid$ Insert �˜ into $\mathcal { D }$   
<sub>4</sub> <sub>en</sub>d   
5 return $\mathcal { D }$

Write $\boldsymbol { x } _ { t } ~ = ~ ( o _ { t } , m _ { t } )$ f<sub>or</sub> th<sub>e s</sub>t<sub>ore</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on, w</sub>ith $o _ { t }$ th<sub>e</sub> environment features and �<sub>�</sub> the a<sub>pp</sub>ended monitor identifier. $\scriptstyle \mathrm { R e ^ { - } }$ l<sub>a</sub>b<sub>e</sub>lli<sub>ng</sub> <sub>rep</sub>l<sub>aces</sub> th<sub>a</sub>t id<sub>en</sub>tifi<sub>er,</sub> <sub>g</sub>i<sub>v</sub>i<sub>ng</sub> $\tilde { \boldsymbol { x } } _ { t } = \left( o _ { t } , \tilde { m } _ { t } \right)$ <sub>.</sub> In Al<sub>go</sub>- <sub>r</sub>ith<sub>m</sub> $2 , d _ { t }$ d<sub>eno</sub>t<sub>es ep</sub>i<sub>so</sub>d<sub>e comp</sub>l<sub>e</sub>ti<sub>on un</sub>d<sub>er</sub> th<sub>e env</sub>i<sub>ronmen</sub>t termination/truncation rule or wra<sub>pp</sub>er horizon<sub>,</sub> inde<sub>p</sub>endentl<sub>y</sub> of th<sub>e re</sub>l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>mon</sub>it<sub>or s</sub>t<sub>a</sub>t<sub>e.</sub> A <sub>ro</sub>ll<sub>ou</sub>t<sub>-c</sub>h<sub>un</sub>k b<sub>oun</sub>d<sub>ary a</sub>l<sub>one</sub> d<sub>oes</sub> not set $d _ { t } = 1$ . Acce<sub>p</sub>tance is evaluated at the <sub>p</sub>ost-transition monitor state $m ^ { + }$ b<sub>e</sub>f<sub>ore rese</sub>t<sub>, w</sub>hil<sub>e</sub> th<sub>e s</sub>t<sub>ore</sub>d <sub>nex</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on a</sub>t <sub>an</sub> <sub>ep</sub>i<sub>so</sub>d<sub>e</sub> b<sub>oun</sub>d<sub>ary</sub> <sub>uses</sub> th<sub>e</sub> <sub>rese</sub>t <sub>mon</sub>it<sub>or</sub> <sub>s</sub>t<sub>a</sub>t<sub>e.</sub>

Actor-critic integration. IPPO-CER an<sup>d</sup> PR2-IPPO-CER com<sup>b</sup>ine f<sub>ac</sub>t<sub>ua</sub>l <sub>an</sub>d <sub>re</sub>l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>c</sub>h<sub>un</sub>k<sub>s,</sub> <sub>recompu</sub>ti<sub>ng</sub> <sub>ro</sub>ll<sub>ou</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs</sub> <sub>w</sub>ith fi<sub>xe</sub>d <sub>pre-up</sub>d<sub>a</sub>t<sub>e parame</sub>t<sub>ers.</sub> Th<sub>e</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on re</sub>t<sub>a</sub>i<sub>ns</sub> f<sub>ac</sub>t<sub>ua</sub>l behaviour lo<sub>g</sub>-<sub>p</sub>robabilities for conditional im<sub>p</sub>ortance correction. Write $\rho _ { t } = \bar { \pi } _ { i } ^ { \mathrm { { \bar { o l d } } } } ( a _ { t } ^ { i } \mid \tilde { x } _ { t } ) / \mu _ { i } ( a _ { t } ^ { i } \mid x _ { t } )$ <sub>, w</sub>h<sub>ere</sub> $\mu _ { i }$ <sub>genera</sub>t<sub>e</sub>d th<sub>e</sub> <sup>f</sup>actua<sup>l</sup> action. In corrected mo<sup>d</sup>e, $\operatorname* { m i n } ( 1 , \rho _ { t } )$ wei<sub>g</sub>hts the cli<sub>pp</sub>ed PPO <sub>su</sub>rr<sub>oga</sub>t<sub>e</sub> <sub>sepa</sub>r<sub>a</sub>t<sub>e</sub>l<sub>y</sub> fr<sub>o</sub>m PPO’<sub>s</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub>-r<sub>a</sub>ti<sub>o</sub> <sub>c</sub>li<sub>pp</sub>in<sub>g.</sub> IPPO also uses �-V-trace-st<sub>y</sub>le tar<sub>g</sub>ets [19], with $\lambda = 0 . 9 5$ <sub>an</sub>d <sub>ac</sub>ti<sub>on an</sub>d tr<sub>ace</sub> <sub>caps</sub> <sub>o</sub>n<sub>e.</sub> PR2-IPPO r<sub>e</sub>t<sub>a</sub>in<sub>s</sub> it<sub>s</sub> r<sub>espo</sub>n<sub>se</sub>-<sub>awa</sub>r<sub>e</sub> <sub>a</sub>d<sub>va</sub>nt<sub>age</sub> <sub>a</sub>nd one-step joint-action critic. In critic\_only mo<sup>d</sup>e, actor up<sup>d</sup>ates, <sub>en</sub>t<sub>ropy</sub> <sub>an</sub>d <sub>a</sub>d<sub>van</sub>t<sub>age</sub> <sub>norma</sub>li<sub>sa</sub>ti<sub>on</sub> <sub>use</sub> f<sub>ac</sub>t<sub>ua</sub>l <sub>samp</sub>l<sub>es,</sub> <sub>w</sub>hil<sub>e</sub> <sub>re</sub>l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>samp</sub>l<sub>es</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>cr</sub>iti<sub>c.</sub> E<sub>ac</sub>h <sub>run</sub> <sub>recor</sub>d<sub>s</sub> it<sub>s</sub> <sub>rep</sub>l<sub>ay</sub> mode and im<sub>p</sub>ortance-ca<sub>p</sub> settin<sub>g</sub>s. Evaluation alwa<sub>y</sub>s uses the origina<sup>l</sup> tempora<sup>l</sup> o<sup>b</sup>jectives.

Al<sub>g</sub>orithm 4: CER inte<sub>g</sub>ration for actor-critic learners   
In<sub>p</sub>ut: factual rollout batch ℬ, fixed <sub>p</sub>re-u<sub>p</sub>date <sub>p</sub>arameters   
1 for each a ent � do   
2 for each factual rollout chunk � ofagent � do   
3 <sup>C</sup>om<sub>p</sub>ute $\tilde { c } \gets$ RelabelChunk(�)   
4 Add <sub>a</sub>ll <sub>c</sub>h<sub>un</sub>k<sub>s</sub> i<sub>n</sub> $\widetilde { c }$ to a<sub>g</sub>ent �’s u<sub>p</sub>date batch   
5 <sub>en</sub>d   
6 R<sub>ecompu</sub>t<sub>e</sub> th<sub>e requ</sub>i<sub>re</sub>d <sub>ro</sub>ll<sub>ou</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs us</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>xe</sub>d   
p<sup>re-u</sup>p<sup>date</sup> p<sup>arameters</sup>   
<sub>7 en</sub>d   
8 return u<sub>p</sub>dated batch ℬ

Nas<sup>h</sup>-Q integration. For NQL, we app<sup>l</sup>y CER on<sup>l</sup>ine: we re<sup>l</sup>a<sup>b</sup>e<sup>l</sup> each observed joint transition and perform a tabular Nash-Q backup <sub>on</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l t<sub>rans</sub>iti<sub>on</sub> <sub>an</sub>d <sub>eac</sub>h <sub>re</sub>l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>var</sub>i<sub>an</sub>t b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> next env<sup>i</sup>ronment ste<sub>p</sub>.

## D.2 PR2

O<sub>u</sub>r di<sub>sc</sub>r<sub>e</sub>t<sub>e</sub>-<sub>ac</sub>ti<sub>o</sub>n <sub>va</sub>ri<sub>a</sub>nt<sub>s</sub> <sub>o</sub>f <sub>p</sub>r<sub>o</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> r<sub>ecu</sub>r<sub>s</sub>i<sub>ve</sub> r<sub>easo</sub>nin<sub>g</sub> (PR2) are ins<sub>p</sub>ired b<sub>y</sub> Wen et al. [57] and use the IQL and IPPO trainin<sub>g</sub> interfaces. For each a<sub>g</sub>ent �, we encode the o<sub>pp</sub>onents’ actions as a sing<sup>l</sup>e joint action $\begin{array} { r } { \bar { a } ^ { - i } \in \mathcal { A } ^ { - i } = \prod _ { j \neq i } \mathcal { A } ^ { j } } \end{array}$ . The a<sub>g</sub>ent <sup>h</sup>as a joint-action critic

$$
Q _ { i } ( x _ { i } ) \in \mathbb { R } ^ { | { \cal { A } } ^ { i } | \times | { \cal { A } } ^ { - i } | } ,
$$

<sub>w</sub>h<sub>ere</sub> $x _ { i }$ i<sub>s</sub> th<sub>e agen</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e rewar</sub>d<sub>-au</sub>t<sub>oma</sub>t<sub>on</sub> state when monitorin<sub>g</sub> is enabled. The o<sub>pp</sub>onent-res<sub>p</sub>onse distribu tion is ind<sub>u</sub>ced directl<sub>y</sub> from this critic<sub>,</sub>

$$
\rho _ { i } ( a ^ { - i } \mid x _ { i } , a ^ { i } ) = \mathrm { s o f t m a x } _ { a ^ { - i } } Q _ { i } ( x _ { i } , a ^ { i } , a ^ { - i } ) ,
$$

<sub>an</sub>d th<sub>e va</sub>l<sub>ue o</sub>f th<sub>e agen</sub>t’<sub>s ac</sub>ti<sub>on un</sub>d<sub>er</sub> thi<sub>s response mo</sub>d<sub>e</sub>l i<sub>s</sub>

$$
\bar { Q } _ { i } ( x _ { i } , a ^ { i } ) = \sum _ { a ^ { - i } \in A ^ { - i } } \rho _ { i } ( a ^ { - i } \mid x _ { i } , a ^ { i } ) Q _ { i } ( x _ { i } , a ^ { i } , a ^ { - i } ) .
$$

We use unit tem<sub>p</sub>erature for $\rho _ { i } .$ <sub>.</sub> Thi<sub>s</sub> <sub>response</sub> <sub>mo</sub>d<sub>e</sub>l <sub>comes</sub> f<sub>rom</sub> <sup>a</sup>g<sup>ent</sup> $\romannumeral 5$ <sub>own</sub> <sub>cr</sub>iti<sub>c</sub> <sub>an</sub>d <sub>nee</sub>d <sub>no</sub>t <sub>ma</sub>t<sub>c</sub>h th<sub>e</sub> <sub>opponen</sub>t<sub>s</sub>’ <sub>con</sub>diti<sub>ona</sub>l <sub>ac</sub>ti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>.</sub> Th<sub>e</sub> PR2-IPPO <sub>a</sub>d<sub>va</sub>nt<sub>age</sub> b<sub>e</sub>l<sub>ow uses</sub> thi<sub>s</sub> m<sub>o</sub>d<sub>e</sub>l<sub>.</sub>

PR2-IQL.. Action se<sup>l</sup>ection is �-gree<sup>d</sup>y wit<sup>h</sup> respect to $\bar { Q } _ { i }$ . Re-<sub>p</sub>l<sub>ay s</sub>t<sub>ores</sub> th<sub>e agen</sub>t’<sub>s own ac</sub>ti<sub>on,</sub> th<sub>e o</sub>b<sub>serve</sub>d <sub>enco</sub>d<sub>e</sub>d <sub>opponen</sub>t joint action, an<sup>d</sup> t<sup>h</sup>e termina<sup>l</sup> automaton rewar<sup>d</sup>. T<sup>h</sup>e up<sup>d</sup>ate regresses on<sup>l</sup>y t<sup>h</sup>e o<sup>b</sup>serve<sup>d</sup> joint-action entry:

$$
\begin{array} { c l } { \mathscr { L } _ { i } = \Big ( Q _ { i } ( x _ { i } , a ^ { i } , a _ { \mathrm { o b s } } ^ { - i } ) - r _ { i } } \\ { - \gamma ( 1 - d ) \underset { b ^ { i } \in \mathcal { A } ^ { i } } { \operatorname* { m a x } } \bar { Q } _ { i } ^ { \mathrm { t a r g } } ( x _ { i } ^ { \prime } , b ^ { i } ) \Big ) ^ { 2 } . } \end{array}
$$

PR2-IPPO.. T<sup>h</sup>e actor outputs <sup>l</sup>ogits <sup>f</sup>or $\pi _ { i } ( \cdot \mid x _ { i } )$ <sub>, w</sub>hil<sub>e</sub> th<sub>e</sub> critic <sup>h</sup>as t<sup>h</sup>e same joint-action s<sup>h</sup>ape as a<sup>b</sup>ove. T<sup>h</sup>e state va<sup>l</sup>ue use<sup>d</sup> b<sub>y</sub> the <sub>p</sub>olic<sub>y</sub> u<sub>p</sub>date is

$$
V _ { i } ( x _ { i } ) = \sum _ { a ^ { i } \in \mathcal { A } ^ { i } } \pi _ { i } ( a ^ { i } \mid x _ { i } ) \bar { Q } _ { i } ( x _ { i } , a ^ { i } ) ,
$$

<sub>an</sub>d th<sub>e ro</sub>ll<sub>ou</sub>t <sub>a</sub>d<sub>van</sub>t<sub>age,</sub> fi<sub>xe</sub>d b<sub>e</sub>f<sub>ore op</sub>ti<sub>m</sub>i<sub>sa</sub>ti<sub>on,</sub> i<sub>s</sub>

$$
\begin{array} { l } { { \displaystyle \widehat { A } _ { i } ^ { \rho } ( \boldsymbol { x } _ { i } , \boldsymbol { a } ^ { i } ) = \mathrm { s t o p g r a d } \Big [ \bar { Q } _ { i } ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { i } , \boldsymbol { a } ^ { i } ) } } \\ { { \displaystyle ~ - \sum _ { b } \pi _ { i } ^ { \mathrm { o l d } } ( b \mid \boldsymbol { x } _ { i } ) \bar { Q } _ { i } ^ { \mathrm { o l d } } ( \boldsymbol { x } _ { i } , b ) \Big ] . } } \end{array}
$$

H<sub>e</sub>r<sub>e</sub> $\mathrm { { \tilde { \Delta } o l d \vec { \Delta } } } ^ { m }$ denotes the <sub>p</sub>re-<sub>up</sub>date <sub>p</sub>arameters. The cli<sub>pp</sub>ed PPO objective uses t<sup>h</sup>is a<sup>d</sup>vantage a<sup>f</sup>ter norma<sup>l</sup>isation over eac<sup>h</sup> mini<sup>b</sup>atc<sup>h</sup>, while the critic re<sub>g</sub>resses

$$
Q _ { i } ( x _ { i } , a ^ { i } , a _ { \mathrm { o b s } } ^ { - i } ) \quad \mathrm { t o w a r d s } \quad r _ { i } + \gamma ( 1 - d ) V _ { i } ^ { \mathrm { o l d } } ( x _ { i } ^ { \prime } ) .
$$

T<sup>h</sup>e -CER variants use t<sup>h</sup>e re<sup>l</sup>a<sup>b</sup>e<sup>ll</sup>ing proce<sup>d</sup>ure in Appen<sup>d</sup>ix D.1. For PR2-IPPO, t<sup>h</sup>e <sup>l</sup>og-pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ities an<sup>d</sup> a<sup>d</sup>vantages are recom-<sub>pu</sub>t<sub>e</sub>d <sub>a</sub>ft<sub>er</sub> <sub>re</sub>l<sub>a</sub>b<sub>e</sub>lli<sub>ng.</sub>

## D.3 Unilateral Im<sub>p</sub>ro<sub>v</sub>ement Search

For each a<sub>g</sub>ent, we initialise its corres<sub>p</sub>ondin<sub>g</sub> learner (IPPO, PR2-IPPO, IQL, PR2-IQL, or Nash-Q) from the evaluated profile <sub>an</sub>d t<sub>ra</sub>i<sub>n</sub> th<sub>a</sub>t <sub>agen</sub>t <sub>w</sub>ith it<sub>s opponen</sub>t<sub>s</sub> fi<sub>xe</sub>d<sub>.</sub> Th<sub>e a</sub>dditi<sub>ona</sub>l b<sub>u</sub>d<sub>ge</sub>t i<sub>s</sub> th<sub>e pro</sub>d<sub>uc</sub>t <sub>o</sub>f th<sub>e</sub> BR it<sub>era</sub>ti<sub>on an</sub>d b<sub>a</sub>t<sub>c</sub>h<sub>-s</sub>t<sub>ep coun</sub>t<sub>s</sub> i<sub>n</sub> Table 13. Res<sub>p</sub>onse searches <sub>p</sub>reserve the learner’s CER settin<sub>g</sub>s. For hc- pro<sup>fil</sup>es, t<sup>h</sup>e <sup>d</sup>eviating agent instea<sup>d</sup> trains on its tempora<sup>l</sup> objective with CER (A<sub>pp</sub>endix C.6).

We evaluate the candidate and ori<sub>g</sub>inal <sub>p</sub>olicies with <sub>p</sub>aired <sub>ep</sub>i<sub>so</sub>d<sub>e</sub> <sub>see</sub>d<sub>s</sub> <sub>an</sub>d <sub>measure</sub> th<sub>e</sub>i<sub>r</sub> t<sub>empora</sub>l <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>on</sub> $u _ { i } .$ <sub>,</sub> includin<sub>g</sub> f<sub>or</sub> h<sub>an</sub>d<sub>cra</sub>ft<sub>e</sub>d <sub>pro</sub>fil<sub>es.</sub> Th<sub>e score sums</sub> th<sub>e un</sub>il<sub>a</sub>t<sub>era</sub>l i<sub>mprove-</sub> <sub>men</sub>t<sub>s</sub> f<sub>oun</sub>d <sub>w</sub>ithi<sub>n</sub> th<sub>e g</sub>i<sub>ven</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>an</sub>d l<sub>earner</sub> f<sub>am</sub>il<sub>y.</sub>

Frozen-opponent operator. The Nash-Q trainer has a dedicated sin<sub>g</sub>le-a<sub>g</sub>ent trainin<sub>g</sub> mode. It obtains o<sub>pp</sub>onents’ <sub>p</sub>olicies from a frozen copy of the original joint Q-tables and replaces the Nash continuation value b<sub>y</sub> max<sub>�</sub> E<sub>�</sub> $\dots { \sim } \sigma _ { - i } Q _ { i } ( x ^ { \prime } , a _ { i } , a _ { - i } )$ <sub>.</sub> C<sub>an</sub>did<sub>a</sub>t<sub>e</sub> evaluation likewise sam<sub>p</sub>les o<sub>pp</sub>onents from that frozen co<sub>py</sub>. With f<sub>u</sub>ll <sub>pro</sub>d<sub>uc</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub>d th<sub>e</sub> t<sub>empora</sub>l <sub>u</sub>tilit<sub>y,</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>un</sub>di<sub>s-</sub> <sub>coun</sub>t<sub>e</sub>d <sub>recurs</sub>i<sub>on</sub> <sub>wou</sub>ld b<sub>e</sub>

$$
B _ { i } ( x ) = \operatorname* { m a x } _ { a _ { i } } \sum _ { a _ { - i } } \sigma _ { - i } ( a _ { - i } \mid x ) \sum _ { x ^ { \prime } } \tilde { \mathcal { P } } ( x ^ { \prime } \mid x , a _ { i } , a _ { - i } ) B _ { i } ( x ^ { \prime } ) ,
$$

<sub>w</sub>ith $B _ { i } ( x ) = R _ { i } ( x )$ <sub>a</sub>t t<sub>e</sub>rmin<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd <sub>su</sub>m<sub>s</sub> <sub>a</sub>nd m<sub>a</sub>xim<sub>a</sub> <sub>ove</sub>r l<sub>ega</sub>l <sub>ac</sub>ti<sub>ons.</sub> I<sub>n our exper</sub>i<sub>men</sub>t<sub>s,</sub> th<sub>e searc</sub>h <sub>uses</sub> th<sub>e</sub> l<sub>earner</sub>’<sub>s</sub> di<sub>scoun</sub>t f<sub>ac</sub>t<sub>or,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>score</sub> <sub>measures</sub> i<sub>mprovemen</sub>t<sub>s</sub> i<sub>n</sub> th<sub>e</sub> undiscounted tem<sub>p</sub>oral utilit<sub>y</sub> $u _ { i } .$

<sup>A</sup>gg<sup>re</sup>g<sup>ation andsam</sup>p<sup>lin</sup>g<sup>. Writin</sup>g $\begin{array} { r } { \Delta _ { i } = \operatorname* { s u p } _ { \sigma _ { i } ^ { \prime } } u _ { i } ( s _ { 0 } , ( \sigma _ { i } ^ { \prime } , \sigma _ { - i } ) ) - } \end{array}$ $u _ { i } ( s _ { 0 } , \vec { \sigma } )$ <sub>,</sub> <sub>e</sub>x<sub>ac</sub>t N<sub>as</sub>hC<sub>o</sub>n<sub>v</sub> i<sub>s</sub> $\textstyle \sum _ { i } \Delta _ { i }$ <sub>�,</sub> whereas ex<sub>p</sub>loitabilit<sub>y</sub> is $\operatorname* { m a x } _ { i } \Delta _ { i }$ . Retainin<sub>g</sub> the ori<sub>g</sub>inal strate<sub>gy</sub> is admissible<sub>,</sub> so ever<sub>y</sub> $\Delta _ { i } \ \geq \ 0$ and hence Exploit $\begin{array} { r } { \leq \sum _ { i } \Delta _ { i } \ \leq } \end{array}$ �Exploit. Two <sub>g</sub>ains of 0.03 therefore <sub>g</sub>ive NashConv 0.06 while satisf<sub>y</sub>in<sub>g</sub> 0.05-NE. Finit<sub>e sea</sub>r<sub>c</sub>h <sub>ca</sub>n mi<sub>ss</sub> d<sub>ev</sub>i<sub>a</sub>ti<sub>o</sub>n<sub>s, w</sub>hil<sub>e sa</sub>m<sub>p</sub>lin<sub>g</sub> n<sub>o</sub>i<sub>se a</sub>nd cli<sub>pp</sub>in<sub>g</sub> can inflate either a<sub>gg</sub>re<sub>g</sub>ation<sub>,</sub> so the estimated score is <sub>no</sub>t <sub>a s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l l<sub>ower</sub> b<sub>oun</sub>d <sub>on exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y.</sub>

For paired success diferences in [−1, 1], a simultaneous twosided Hoefdin<sub>g</sub> half-width over � a<sub>g</sub>ents is $\sqrt { 2 \ln ( 2 m / \delta ) / N }$ <sub>.</sub> At $N = 3 2$ <sub>an</sub>d $\delta = 0 . 0 5$ , it is a<sub>pp</sub>roximatel<sub>y</sub> 0.52 for two a<sub>g</sub>ents and 0.56 for four, substantiall<sub>y</sub> exceedin<sub>g</sub> $\varepsilon = 0 . 0 5$ <sub>.</sub> S<sub>ma</sub>ll <sub>scores</sub> th<sub>ere</sub>f<sub>ore</sub> d<sub>o no</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>c s</sub>t<sub>a</sub>bilit<sub>y.</sub>

## D<sub>.</sub>4 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 1

Theorem 1 (Coherence). For any strate<sub>g</sub>y profile ⃗� and state �: $\mathbb { E } _ { \pi \sim \mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } } } [ \mathcal { R } _ { \varphi } ( \pi ) ] = \mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } } ( \varphi )$

Proof<sub>.</sub> B<sub>y co</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n<sub>,</sub> $\mathcal { R } _ { \varphi } ( \pi ) = 1 \operatorname { i f f } \mathcal { M } , \pi \vdash \varphi ,$ <sub>an</sub>d <sub>o</sub>th<sub>er-</sub> <sub>w</sub>i<sub>se</sub> $\mathcal { R } _ { \varphi } ( \pi ) = 0$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
\mathcal { R } _ { \varphi } ( \pi ) = \mathbf { 1 } \big [ \mathcal { M } , \pi \big | = \varphi \big ] .
$$

Takin<sub>g</sub> ex<sub>p</sub>ectation under $\mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } }$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\begin{array} { r l } & { \mathbb { E } _ { \pi \sim \mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } } } [ \mathcal { R } _ { \varphi } ( \pi ) ] = \mathbb { E } _ { \pi \sim \mathrm { P r } _ { \mathcal { M } , s } ^ { \vec { \sigma } } } \big [ \mathbf { 1 } \big [ \mathcal { M } , \pi \big \vert = \varphi \big ] \big ] } \\ & { \qquad = \underset { \mathcal { M } , s } { \overset { \vec { \sigma } } { \mathrm { P r } } } \big ( \{ \pi \mid \mathcal { M } , \pi \mid = \varphi \} \big ) = \underset { \mathcal { M } , s } { \overset { \vec { \sigma } } { \mathrm { P r } } } ( \varphi ) . } \end{array}
$$

## D<sub>.</sub>5 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 2

Theorem 2 (Markov Sufficiency in the Composite Product). For finite-horizon $L T L f$ o<sup>b</sup>jectives $\varphi _ { 1 } , \ldots , \varphi _ { n }$ <sup>and</sup> <sup>ever</sup>y <sup>strate</sup>gy p<sup>ro-</sup> $\hbar l e \vec { \sigma } \in \Sigma$ , the satisfaction probabilities $\left( p _ { i } ( s _ { 0 } , \vec { \sigma } ) \right) _ { i = 1 } ^ { n }$ <sub>an</sub>d <sub>u</sub>tiliti<sub>es</sub> $\vec { u } ( s _ { 0 } , \vec { \sigma } )$ can be realised by a memoryless strategy profile on $\widehat { \mathcal { M } } _ { H }$ and every such memoryless profile induces a profile in Σ with the same payofs.

We ada<sub>p</sub>t the e<sub>q</sub>uivalence ar<sub>g</sub>ument of [5, 9] to stochastic <sub>g</sub>ames.

Definition 14 (Non-Markovian Reward Stochastic Game). A non-Markovian reward stochastic <sub>g</sub>ame (NMRSG) � inherits the states, initial state, le<sub>g</sub>al actions, transition kernel, horizon �, and stoppin<sub>g</sub> rule � $o f \mathcal { M }$ . Rewards are zero before termination, with undiscounted terminal payofs $\mathcal { R } ^ { i } ( h _ { T } )$ d<sub>epen</sub>di<sub>ng</sub> <sub>on</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> hi<sub>s</sub>t<sub>ory</sub> $h _ { T } =$ $s _ { 0 } { \pmb a } _ { 0 } \cdots s _ { T }$

Definition 15 (E<sub>q</sub>uivalence of Reward Games). An NMRSG � is <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o an ex</sub>t<sub>en</sub>d<sub>e</sub>d M<sub>ar</sub>k<sub>ov</sub>i<sub>an game</sub> $\widehat { \mathcal G }$ <sub>w</sub>ith <sub>zero non-</sub>t<sub>erm</sub>i<sub>na</sub>l rewards and undiscounted terminal payofs $\widehat { R } _ { i }$ if there are maps $\rho : { \widehat { S } }  S$ <sub>an</sub>d $\iota : { \mathcal { S } }  { \widehat { \mathcal { S } } }$ <sub>suc</sub>h th<sub>a</sub>t<sub>:</sub>

(1) $\rho ( \iota ( s ) ) = s$ for ever<sub>y</sub> �, and a<sub>g</sub>ent �’s le<sub>g</sub>al actions at each non-terminal � are �<sup>�</sup> $( \rho ( x ) )$ .

(2) For ever<sub>y</sub> non-terminal � and le<sub>g</sub>al joint action ${ \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } \mathbf { } } _ { \mathbf { } } { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf } { \mathbf } _ { \mathbf { } \mathbf { } } _ { \mathbf } _ { \mathbf { } \mathbf { } \mathbf } _ { \mathbf { } \mathbf } _ { \mathbf } { \mathbf } _ { \mathbf }  _ { \mathbf } _ { \mathbf { } \mathbf } _ { \mathbf } _ { \mathbf } \mathbf { } _ \mathbf { } _ { \mathbf } \mathbf \mathbf { } _ \mathbf { } \mathbf \mathbf { } _ \mathbf \Sigma  _ { \mathbf \mathbf } _ { \mathbf \Sigma } _ \mathbf { } \mathbf \Sigma _ \Sigma \mathbf \Sigma \Sigma _ \Sigma \Sigma \Sigma _ \Sigma \Sigma \Sigma  _ \Sigma \mathbf \Sigma \Sigma \Sigma _ \Sigma \Sigma \Sigma \Sigma _ \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma _ \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma $ maps positive-pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity exten<sup>d</sup>e<sup>d</sup> successors <sup>b</sup>ijective<sup>l</sup>y onto origi<sub>na</sub>l <sub>successors,</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es:</sub>

$$
{ \widehat { \mathcal { P } } } ( x ^ { \prime } \mid x , \mathbf { a } ) = { \mathcal { P } } ( \rho ( x ^ { \prime } ) \mid \rho ( x ) , \mathbf { a } ) .
$$

(3) For correspondin<sub>g</sub> feasible histories $h _ { t } = s _ { 0 } { \pmb a } _ { 0 } \cdot \cdot \cdot { \pmb s } _ { t }$ <sub>an</sub>d $\hat { h } _ { t } = x _ { 0 } \pmb { a } _ { 0 } \cdot \cdot \cdot \boldsymbol { x } _ { t }$ <sub>, w</sub>ith $x _ { 0 } = \iota ( s _ { 0 } )$ <sub>an</sub>d $\rho ( x _ { k } ) = s _ { k } , x _ { t }$ is t<sub>erm</sub>i<sub>na</sub>l $i f f \kappa ( s _ { t } , t ) = 1$ <sub>, an</sub>d $\widehat { R } _ { i } ( x _ { T } ) = \mathcal { R } ^ { i } ( h _ { T } )$ for every a<sub>g</sub>ent � at termination.

In Definition 15, � initialises the additional state and � projects back to the environment<sub>,</sub> <sub>p</sub>reservin<sub>g</sub> d<sub>y</sub>namics<sub>,</sub> termination<sub>,</sub> and p<sup>a</sup>y<sup>ofs.</sup>

Lemma 1 (Strate<sub>gy</sub>-Profile E<sub>q</sub>uivalence). For e<sub>q</sub>uivalent <sub>g</sub>ames, feasible histories from �<sub>0</sub> lift uniquely from $\iota ( s _ { 0 } )$ . Memor<sub>y</sub>less exten<sup>d</sup>e<sup>d</sup> strate<sub>g</sub>ies corres<sub>p</sub>on<sup>d</sup> to ori<sub>g</sub>ina<sup>l</sup> strate<sub>g</sub>ies <sup>d</sup>e<sub>p</sub>en<sup>d</sup>in<sub>g</sub> on<sup>l</sup><sub>y</sub> on the final lifted state, up to choices at unreachable states. With independent randomisation across agents, corresponding profiles preserve complete-history probabilities, expected terminal payofs, and <sub>un</sub>il<sub>a</sub>t<sub>era</sub>l d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> <sub>ga</sub>i<sub>ns</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>ese</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>c</sub>l<sub>asses.</sub>

Proof. Condition 2 constr<sub>u</sub>cts the lift rec<sub>u</sub>rsi<sub>v</sub>el<sub>y, w</sub>ith <sub>p</sub>rojection as its inverse, w<sup>h</sup>i<sup>l</sup>e con<sup>d</sup>itions 1 an<sup>d</sup> 3 preserve <sup>l</sup>ega<sup>l</sup> actions and sto<sub>pp</sub>in<sub>g</sub>. For corres<sub>p</sub>ondin<sub>g</sub> histories endin<sub>g</sub> at $x _ { t } ,$ set $\sigma _ { i } ( h _ { t } ) = \hat { \sigma } _ { i } ( x _ { t } )$ <sub>.</sub> Thi<sub>s</sub> d<sub>e</sub>fi<sub>nes</sub> b<sub>o</sub>th di<sub>rec</sub>ti<sub>ons o</sub>f th<sub>e correspon-</sub> dence<sub>,</sub> since ori<sub>g</sub>inal strate<sub>g</sub>ies de<sub>p</sub>end onl<sub>y</sub> on $x _ { t }$ . Choose arbitrar<sub>y</sub> l<sub>ega</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> <sub>a</sub>t <sub>unreac</sub>h<sub>a</sub>bl<sub>e</sub> <sub>non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> I<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t ran<sup>d</sup>omisation gives equa<sup>l</sup> joint-action pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ities, so in<sup>d</sup>uction usin<sub>g</sub> condition 2 <sub>g</sub>ives e<sub>q</sub>ual histor<sub>y</sub> <sub>p</sub>robabilities. Condition 3 then <sub>g</sub>ives e<sub>q</sub>ual ex<sub>p</sub>ected <sub>p</sub>a<sub>y</sub>ofs. Since the corres<sub>p</sub>ondence acts a<sub>g</sub>ent b<sub>y</sub> a<sub>g</sub>ent<sub>,</sub> it also <sub>p</sub>reserves unilateral deviations and their <sub>g</sub>ains. □

Proof of Theorem 2<sub>.</sub> C<sub>o</sub>n<sub>s</sub>id<sub>e</sub>r th<sub>e</sub> NMRSG <sub>o</sub>f D<sub>e</sub>finiti<sub>o</sub>n 14 with terminal <sub>p</sub>a<sub>y</sub>ofs

$$
\mathscr { R } ^ { i } ( h _ { T } ) \stackrel { \mathrm { d e f } } { = } f _ { i } \big ( { \bf 1 } [ \mathscr { M } , h _ { T } \ : \in \ : \varphi _ { i } ] \big ) .
$$

Their ex<sub>p</sub>ectations e<sub>q</sub>ual $u _ { i } ( s _ { 0 } , \vec { \sigma } )$ b<sub>ecause</sub> $f _ { i }$ i<sub>s</sub> <sub>a</sub>fi<sub>ne.</sub> T<sub>o</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h e<sub>q</sub>uivalence with $\hat { \mathcal { M } } _ { H }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\begin{array} { c } { \iota ( s ) \ \stackrel { \mathrm { d e f } } { = } \ \big ( s , \delta _ { c } ( m _ { 0 } ^ { c } , \mathcal { L } ( s ) ) , 0 \big ) , } \\ { \rho ( s , \pmb { m } , t ) \ \stackrel { \mathrm { d e f } } { = } \ s . } \end{array}
$$

C<sub>o</sub>nditi<sub>o</sub>n 1 i<sub>s</sub> imm<sub>e</sub>di<sub>a</sub>t<sub>e.</sub> B<sub>y</sub> D<sub>e</sub>finiti<sub>o</sub>n 9<sub>,</sub> <sub>eac</sub>h <sub>o</sub>ri<sub>g</sub>in<sub>a</sub>l <sub>successo</sub>r $s ^ { \prime }$ <sub>g</sub>ives the uni<sub>q</sub>ue <sub>p</sub>roduct successor

$$
( s ^ { \prime } , \delta _ { c } ( m , \mathcal { L } ( s ^ { \prime } ) ) , t + 1 )
$$

with <sub>p</sub>robabilit<sub>y</sub> $\mathcal { P } ( s ^ { \prime } \mid s , \mathbf { a } )$ <sub>,</sub> establishin<sub>g</sub> condition 2. Both <sub>g</sub>ames t<sub>erm</sub>i<sub>na</sub>t<sub>e a</sub>t th<sub>e same</sub> $T$ <sub>un</sub>d<sub>er</sub> $\kappa .$ Th<sub>e mon</sub>it<sub>or rea</sub>d<sub>s exac</sub>tl<sub>y</sub> th<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> th<sub>roug</sub>h $s _ { T }$ , so

$$
\begin{array} { r l } & { R _ { i } ( s _ { T } , { m _ { T } } , T ) = f _ { i } \big ( \mathbf { 1 } [ { m _ { T } } \mid = F _ { i } ] \big ) } \\ & { \qquad = f _ { i } \big ( \mathbf { 1 } [ { \mathcal { M } } , h _ { T } \mid = \varphi _ { i } ] \big ) } \\ & { \qquad = \mathcal { R } ^ { i } ( h _ { T } ) , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> <sub>con</sub>diti<sub>on</sub> 3<sub>.</sub>

Th<sub>e</sub> fi<sub>na</sub>l lift<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s</sub> $( s _ { t } , m _ { t } , t )$ <sub>,</sub> so the ori<sub>g</sub>inal strate<sub>gy</sub> class in Lemma 1 is exactl<sub>y</sub> Σ. The lemma therefore <sub>g</sub>ives both directions of the memor<sub>y</sub>less corres<sub>p</sub>ondence and <sub>p</sub>reser<sub>v</sub>es <sub>u</sub>tilities. Acce<sub>p</sub>- tance indicators also a<sub>g</sub>ree on corres<sub>p</sub>ondin<sub>g</sub> com<sub>p</sub>lete histories<sub>,</sub> so e<sub>q</sub>ualit<sub>y</sub> of their <sub>p</sub>robabilities <sub>g</sub>ives e<sub>q</sub>ualit<sub>y</sub> of ever<sub>y</sub> satisfaction <sub>p</sub>robabilit<sub>y</sub> $p _ { i }$ □

## D<sub>.</sub>6 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 3

Theorem 3 (Fixed-Horizon Terminal Reward Is Invariant to Discounting). Consider episodes offixed time indices $t = 0 , \ldots , H$ and a terminal reward ofthe form $r _ { t } = 0 f o r t < H$ <sub>an</sub>d $r _ { H } =$ $\mathbf { 1 } [ \varphi ] \in \{ 0 , 1 \}$ . Then for any strategy profile $\begin{array} { r l } { \vec { \sigma } , \mathbb { E } _ { \vec { \sigma } } \Big [ \sum _ { t = 0 } ^ { H } \gamma ^ { t } r _ { t } \Big ] = } & { { } } \end{array}$ $\gamma ^ { H } \operatorname { P r } _ { \vec { \sigma } } ( \varphi )$ . In particular, forfixed � and $\gamma \in ( 0 , 1 )$ , <sup>an</sup>y <sup>strate</sup>gy profile that maximises the discounted return also maximises $\mathrm { P r } _ { \vec { \sigma } } ( \varphi )$ E<sub>xac</sub>t N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>a are unc</sub>h<sub>ange</sub>d<sub>,</sub> b<sub>u</sub>t <sub>exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y</sub> i<sub>n</sub> di<sub>scoun</sub>t<sub>e</sub>d<sub>-</sub> return units $i s \gamma ^ { H }$ Exploit.

Proof<sub>.</sub> Si<sub>nce</sub> $r _ { t } = 0$ f<sub>or a</sub>ll $t < H$ <sub>an</sub>d $r _ { H } = { \bf 1 } [ \varphi ]$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\sum _ { t = 0 } ^ { H } \gamma ^ { t } r _ { t } = \gamma ^ { H } r _ { H } = \gamma ^ { H } \mathbf { 1 } [ \varphi ] .
$$

Takin<sub>g</sub> ex<sub>p</sub>ectation under ⃗� <sub>g</sub>ives

$$
\mathbb { E } _ { \vec { \sigma } } \left[ \sum _ { t = 0 } ^ { H } \gamma ^ { t } r _ { t } \right] = \gamma ^ { H } \mathbb { E } _ { \vec { \sigma } } [ \mathbf { 1 } [ \varphi ] ] = \gamma ^ { H } \operatorname* { P r } _ { \vec { \sigma } } ( \varphi ) .
$$

B<sub>ecause</sub> $\gamma ^ { H } > 0 \mathrm { f o r } \gamma \in ( 0 , 1 )$ <sub>an</sub>d fi<sub>xe</sub>d $H ,$ multi<sub>p</sub>l<sub>y</sub>in<sub>g</sub> b<sub>y</sub> the constant $\gamma ^ { H }$ does not chan<sub>g</sub>e the set of maximisers. Hence<sub>,</sub> an<sub>y</sub> strate<sub>gy</sub> <sub>p</sub>rofile maximisin<sub>g</sub> discounted return also maximises $\mathrm { P r } _ { \vec { \sigma } } ( \varphi )$ <sub>.</sub> Th<sub>e</sub> same <sub>p</sub>ositive scalin<sub>g</sub> a<sub>pp</sub>lies to ever<sub>y</sub> unilateral <sub>g</sub>ain<sub>,</sub> and commutes with their su<sub>p</sub>remum and maximum<sub>, g</sub>ivin<sub>g</sub> the stated e<sub>q</sub>uilibrium <sup>conse</sup>q<sup>uences.</sup> □

## D<sub>.</sub>7 Pr<sub>oo</sub>f <sub>o</sub>f Pr<sub>opos</sub>iti<sub>o</sub>n 1

Pro<sub>p</sub>osition 1 (Discounted Return under Earl<sub>y</sub> Termination). Let $T \leq H$ be the (random) termination time of an episode and let � be the acceptance event. If the reward is terminal, $i . e . , r _ { t } = 0$ for $t < T$ <sub>an</sub>d $r _ { T } = \mathbf { 1 } [ A ]$ <sub>,</sub> th<sub>en</sub> th<sub>e</sub> di<sub>scoun</sub>t<sub>e</sub>d <sub>re</sub>t<sub>urn</sub> i<sub>s</sub> $\begin{array} { r l } { G _ { \gamma } } & { { } = } \end{array}$ $\textstyle \sum _ { t = 0 } ^ { T } \gamma ^ { t } r _ { t } \ = \ \gamma ^ { T } \mathbf { 1 } [ A ]$ . Hence for any strategy profile ${ \vec { \sigma } } ,$ <sub>we</sub> h<sub>ave</sub> $\mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] \ = \ \mathbb { E } _ { \vec { \sigma } } [ \gamma ^ { T } \mathbf { 1 } [ A ] ]$

Proof<sub>.</sub> B<sub>y</sub> <sub>assu</sub>m<sub>p</sub>ti<sub>o</sub>n<sub>,</sub> $r _ { t } = 0$ f<sub>or</sub> <sub>a</sub>ll $t < T$ <sub>an</sub>d $r _ { T } = \mathbf { 1 } [ A ]$ Th<sub>ere</sub>f<sub>ore</sub>

$$
G _ { \gamma } = \sum _ { t = 0 } ^ { T } \gamma ^ { t } r _ { t } = \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } \cdot 0 + \gamma ^ { T } r _ { T } = \gamma ^ { T } \mathbf { 1 } [ A ] .
$$

Ta<sup>k</sup>ing expectation un<sup>d</sup>er t<sup>h</sup>e trajectory <sup>d</sup>istri<sup>b</sup>ution in<sup>d</sup>uce<sup>d</sup> <sup>b</sup>y $\vec { \sigma }$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] = \mathbb { E } _ { \vec { \sigma } } \big [ \gamma ^ { T } \mathbf { 1 } [ A ] \big ] .
$$

## D<sub>.</sub>8 P<sub>roo</sub>f <sub>o</sub>f C<sub>oro</sub>ll<sub>ary</sub> 1

Corollary 1 (Discounted Objective Bounds Satisfaction Probability). Under the assumptions of Prop. 1 and $T \ \leq \ H ,$ <sub>we</sub> h<sub>ave</sub> $\gamma ^ { H } \mathring { \operatorname* { P r } } _ { \vec { \sigma } } ( A ) \ \leq \ \mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] \ \overset { \cdot } { \ \leq } \ \operatorname* { P r } _ { \vec { \sigma } } ( A )$

Proof. Under the assum<sub>p</sub>tions of Pro<sub>p</sub>osition 1, on the event � <sub>we</sub> h<sub>ave</sub> $T \leq H$ <sub>, an</sub>d th<sub>ere</sub>f<sub>ore</sub> $\gamma ^ { T } \in [ \gamma ^ { \dot { H } }$ , 1]. On the complement event $\neg A$ , the return is 0. Hence, <sub>p</sub>ointwise,

$$
\gamma ^ { H } \mathbf { 1 } [ A ] \leq \gamma ^ { T } \mathbf { 1 } [ A ] \leq \mathbf { 1 } [ A ] .
$$

Takin<sub>g</sub> ex<sub>p</sub>ectation under $\vec { \sigma }$ <sub>a</sub>nd <sub>us</sub>in<sub>g</sub> Pr<sub>opos</sub>iti<sub>o</sub>n 1 <sub>g</sub>i<sub>ves</sub>

$$
\gamma ^ { H } \operatorname* { P r } ( A ) \leq \mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] \leq \operatorname* { P r } _ { \vec { \sigma } } ( A ) .
$$

## D<sub>.</sub>9 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 4

Theorem 4 (Hoeffding Confidence Interval for Satisfaction Probability). For any confidence parameter $\delta \in ( 0 , 1 )$ , define the half-width $\rho _ { n } ( \delta ) \ { \stackrel { \mathrm { d e f } } { = } } \ { \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n } } }$ , and the clipped bounds

$$
L _ { n } \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { m a x } \{ 0 , { \hat { p } } - \rho _ { n } ( \delta ) \} , \qquad U _ { n } \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { m i n } \{ 1 , { \hat { p } } + \rho _ { n } ( \delta ) \} .
$$

Th<sub>en</sub> $\begin{array} { r } { \operatorname* { P r } \big ( p \in [ L _ { n } , U _ { n } ] \big ) ~ \ge ~ 1 - \delta , } \end{array}$ <sub>w</sub>h<sub>ere</sub> th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s over</sub> th<sub>e</sub> randomness of the � sampled rollouts.

P<sub>roof.</sub> Th<sub>e var</sub>i<sub>a</sub>bl<sub>es</sub> $X _ { 1 } , \ldots , X _ { n }$ <sub>are</sub> i<sub>.</sub>i<sub>.</sub>d<sub>. an</sub>d t<sub>a</sub>k<sub>e va</sub>l<sub>ues</sub> i<sub>n</sub> [0, 1] with mean $p .$ B<sub>y</sub> Hoefdin<sub>g</sub>’s ine<sub>q</sub>ualit<sub>y,</sub> for an<sub>y</sub> $t > 0$

$$
\operatorname* { P r } ( { \hat { p } } - p \geq t ) \leq e ^ { - 2 n t ^ { 2 } } \qquad { \mathrm { a n d ~ } } \qquad \operatorname* { P r } ( p - { \hat { p } } \geq t ) \leq e ^ { - 2 n t ^ { 2 } } .
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> a union bound <sub>y</sub>ields

$$
\mathrm { P r } ( | \hat { p } - p | \geq t ) \leq 2 e ^ { - 2 n t ^ { 2 } } .
$$

Now set

$$
t = \rho _ { n } ( \delta ) = \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n } } .
$$

Th<sub>en</sub> $2 e ^ { - 2 n t ^ { 2 } } = \delta ,$ so

$$
\Pr \left( | { \hat { p } } - p | \leq \rho _ { n } ( \delta ) \right) \geq 1 - \delta .
$$

Wh<sub>enever</sub> $| \hat { p } - p | \leq \rho _ { n } ( \delta )$ h<sub>o</sub>ld<sub>s,</sub> <sub>we</sub> <sub>a</sub>l<sub>so</sub> h<sub>ave</sub>

$$
p \in [ \hat { p } - \rho _ { n } ( \delta ) , \hat { p } + \rho _ { n } ( \delta ) ] .
$$

Sin<sub>ce</sub> $p \in \ [ 0 , 1 ]$ , clipping the endpoints to [0, 1] preserves this containment<sub>,</sub> so

$$
\operatorname* { P r } \left( p \in [ L _ { n } , U _ { n } ] \right) \geq 1 - \delta .
$$

## D<sub>.</sub>10 P<sub>roo</sub>f <sub>o</sub>f C<sub>oro</sub>ll<sub>ary</sub> 2

Corollar<sub>y</sub> 2 (�-Bounded Discount Bias under Earl<sub>y</sub> Termination). Under the assumptions ofCorollary 1, to bound the discount-induced bi<sub>as</sub> b<sub>y</sub> <sub>a</sub>t <sub>mos</sub>t $\xi \in [ 0 , 1 ]$ uniformly over strategy profiles, it sufices t<sub>o c</sub>h<sub>oose</sub> $\gamma \ge ( 1 - \xi ) ^ { 1 / H }$ . That is, for every strategy profile ${ \vec { \sigma } } ,$ $0 \leq \operatorname* { P r } _ { \vec { \sigma } } ( A ) - \mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] \leq \xi .$

Proof. B<sub>y</sub> Corollar<sub>y</sub> 1<sub>,</sub>

$$
0 \leq \operatorname* { P r } _ { \vec { \sigma } } ( A ) - \mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] \leq ( 1 - \gamma ^ { H } ) \operatorname* { P r } _ { \vec { \sigma } } ( A ) \leq ( 1 - \gamma ^ { H } ) ,
$$

<sub>s</sub>in<sub>ce</sub> $\operatorname* { P r } _ { \vec { \sigma } } ( A ) \leq 1 . \operatorname { I f } \gamma \geq ( 1 - \xi ) ^ { 1 / H }$ <sub>,</sub> th<sub>en</sub> ${ \gamma } ^ { H } \geq 1 - { \xi } ,$ <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> $1 - \gamma ^ { H } \leq \xi .$ Combinin<sub>g</sub> the ine<sub>q</sub>ualities <sub>g</sub>ives the result. □

The bound is ti<sub>g</sub>ht in that if � occurs with <sub>p</sub>robabilit<sub>y</sub> 1 and $T = H$ <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y,</sub> th<sub>en</sub> $\operatorname* { P r } _ { \vec { \sigma } } ( A ) - \mathbb { E } _ { \vec { \sigma } } [ G _ { \gamma } ] = \hat { 1 } - \gamma ^ { H }$ . Hence no smaller � can guarantee bias at most $\xi$ uniforml<sub>y</sub> (over all <sub>p</sub>ossible acce<sub>p</sub>tance-time distributions).

## D<sub>.</sub>11 P<sub>roo</sub>f <sub>o</sub>f C<sub>oro</sub>ll<sub>ary</sub> 3

Corollar<sub>y</sub> 3 (One-Sided Certification of Threshold Constraints). Fi<sub>x a</sub> th<sub>res</sub>h<sub>o</sub>ld $q \in [ 0 , 1 ]$ . Under the assumptions of Theorem 4, <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \delta$ over the validation rollouts, all four i<sub>mp</sub>li<sub>ca</sub>ti<sub>ons</sub> h<sub>o</sub>ld<sub>:</sub>

$$
\begin{array} { r } { L _ { n } > q \implies p > q , \quad U _ { n } < q \implies p < q , } \\ { L _ { n } \geq q \implies p \geq q , \quad U _ { n } \leq q \implies p \leq q . } \end{array}
$$

Thus the probability of issuing any false threshold certificate is at most �.

Proof<sub>.</sub> L<sub>e</sub>t $E \ = \ \{ p \in \ [ L _ { n } , U _ { n } ] \}$ <sub>.</sub> B<sub>y</sub> Th<sub>eo</sub>r<sub>e</sub>m 4<sub>, we</sub> h<sub>ave</sub> $\operatorname* { P r } ( E ) \geq 1 - \delta$ <sub>.</sub> O<sub>n</sub> th<sub>e</sub> <sub>even</sub>t $E _ { \mathrm { { i } } }$ <sub>,</sub> <sub>eac</sub>h <sub>c</sub>l<sub>a</sub>i<sub>m</sub> f<sub>o</sub>ll<sub>ows</sub> di<sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> t<sup>h</sup>e a<sub>pp</sub>ro<sub>p</sub>r<sup>i</sup>ate en<sup>d</sup><sub>p</sub>o<sup>i</sup>nt com<sub>p</sub>ar<sup>i</sup>son:

$$
\begin{array} { r } { L _ { n } > q \Rightarrow p \geq L _ { n } > q , } \\ { U _ { n } < q \Rightarrow p \leq U _ { n } < q , } \end{array}
$$

<sub>an</sub>d

$$
L _ { n } \geq q \Rightarrow p \geq L _ { n } \geq q ,
$$

$$
U _ { n } \leq q \Rightarrow p \leq U _ { n } \leq q .
$$

Th<sub>ese</sub> <sub>are</sub> <sub>exac</sub>tl<sub>y</sub> th<sub>e</sub> <sub>seman</sub>ti<sub>c</sub> <sub>con</sub>diti<sub>ons</sub> f<sub>or</sub> $\mathsf { P } _ { > q } [ \varphi ] , \mathsf { P } _ { < q } [ \varphi ] ,$ $\mathsf { P } { \geq } q \left[ \varphi \right]$ <sub>,</sub> <sub>an</sub>d $\mathsf { P } { \scriptstyle \leq _ { q } [ \varphi ] }$ , res<sub>p</sub>ect<sup>i</sup>ve<sup>l</sup><sub>y</sub>. □

## D<sub>.</sub>12 P<sub>roo</sub>f <sub>o</sub>f C<sub>oro</sub>ll<sub>ary</sub> 4

Corollar<sub>y</sub> 4 (Joint Certification via Bonferroni Adjustment). Fix an objective profile $\vec { \psi } = ( \psi _ { 1 } , \ldots , \psi _ { m } )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\psi _ { i } = \mathsf { P } _ { \bowtie _ { i } q _ { i } } \left[ \varphi _ { i } \right]$ <sub>,</sub> <sub>an</sub>d a fixed strategy profile ⃗�. Before validation, fix sample sizes $n _ { i } \geq 1$ and failure budgets $\delta _ { i } \in ( 0 , 1 )$ <sub>suc</sub>h th<sub>a</sub>t $\bar { \sum } _ { i = 1 } ^ { m } \delta _ { i } \stackrel { - } { \le } \delta$ for a joint b<sub>u</sub>d<sub>ge</sub>t $\delta \in ( 0 , 1 )$ . For each �, let $p _ { i } = \mathrm { P r } _ { { \mathcal M } , s } ^ { \vec { \sigma } } ( \varphi _ { i } )$ <sub>an</sub>d <sub>cons</sub>t<sub>ruc</sub>t <sub>a</sub> Hoefding interval $[ L _ { i } , U _ { i } ]$ as in Theorem 4, from �<sub>�</sub> i.i.d. rollouts and with marginal failure probability �<sub>�</sub>. Then

$$
\operatorname* { P r } \left( \bigcap _ { i = 1 } ^ { m } \{ p _ { i } \in [ L _ { i } , U _ { i } ] \} \right) \geq 1 - \delta .
$$

C<sub>onsequen</sub>tl<sub>y, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \delta ,$ <sub>a</sub>ll <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>-</sub>b<sub>ase</sub>d certificates from Corollary 3 are simultaneously sound. In particular, if, for every �, the relevant condition

$$
\begin{array} { r l } { L _ { i } > q _ { i } ( \bowtie _ { i } = > ) , \qquad } & { U _ { i } < q _ { i } ( \bowtie _ { i } = < ) , } \\ { L _ { i } \geq q _ { i } ( \bowtie _ { i } = \geq ) , \qquad } & { U _ { i } \leq q _ { i } ( \bowtie _ { i } = \leq ) } \end{array}
$$

h<sub>o</sub>ld<sub>s,</sub> th<sub>en</sub> $\mathcal { M } , s , \vec { \sigma } \vdash \vec { \psi }$ is jointly certified with confidence at least $1 - \delta$ . The uniform Bonferroni choice $\delta _ { i } = \delta / m$ gives half-width $\rho _ { n _ { i } } ( \delta / m ) \ : = \ : \sqrt { \ln ( 2 m / \delta ) / ( 2 n _ { i } ) }$ . The rollouts used for diferent o<sup>b</sup>jectives nee<sup>d</sup> not <sup>b</sup>e in<sup>d</sup>epen<sup>d</sup>ent. In particu<sup>l</sup>ar, t<sup>h</sup>e same ro<sup>ll</sup>outs ma<sub>y</sub> <sup>b</sup>e eva<sup>l</sup>uate<sup>d</sup> a<sub>g</sub>ainst ever<sub>y</sub> $\varphi _ { i }$

Proof. For each �, let $E _ { i } = \{ p _ { i } \in [ L _ { i } , U _ { i } ] \}$ }. By Theorem 4, $\Pr ( E _ { i } ^ { c } ) \leq \delta _ { i }$ . Hence the Bonferroni ine<sub>q</sub>ualit<sub>y</sub> (union bound) <sub>g</sub>ives

$$
\begin{array} { r l } {  { \operatorname* { P r } ( \bigcap _ { i = 1 } ^ { m } E _ { i } ) = 1 - \operatorname* { P r } ( \bigcup _ { i = 1 } ^ { m } E _ { i } ^ { c } ) } \ ~ } & { } \\ & { \geq 1 - \displaystyle \sum _ { i = 1 } ^ { m } \operatorname* { P r } ( E _ { i } ^ { c } ) } \\ & { \geq 1 - \displaystyle \sum _ { i = 1 } ^ { m } \delta _ { i } \geq 1 - \delta . } \end{array}
$$

No inde<sub>p</sub>endence amon<sub>g</sub> the events $E _ { i }$ i<sub>s</sub> r<sub>equ</sub>ir<sub>e</sub>d<sub>.</sub> On th<sub>e</sub>ir int<sub>e</sub>r<sub>sec</sub>- tion<sub>,</sub> each end<sub>p</sub>oint com<sub>p</sub>arison im<sub>p</sub>lies the corres<sub>p</sub>ondin<sub>g</sub> thresh-<sub>o</sub>ld <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t b<sub>y</sub> C<sub>oro</sub>ll<sub>ary</sub> 3<sub>,</sub> <sub>so</sub> <sub>a</sub>ll <sub>suc</sub>h i<sub>mp</sub>li<sub>ca</sub>ti<sub>ons</sub> h<sub>o</sub>ld <sub>s</sub>i<sub>mu</sub>lt<sub>a-</sub> neous<sup>l</sup>y. I<sup>f</sup> every o<sup>b</sup>jective is certi<sup>fi</sup>e<sup>d</sup>, t<sup>h</sup>is is precise<sup>l</sup>y $\mathcal { M } , s , \vec { \sigma } \vdash \vec { \psi } .$ Finall<sub>y,</sub> substitutin<sub>g</sub> $\delta _ { i } = \delta / m$ into $\rho _ { n _ { i } } ( \delta _ { i } )$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>un</sub>i<sub>-</sub> f<sub>orm</sub> h<sub>a</sub>lf<sub>-w</sub>idth<sub>.</sub> □

## D<sub>.</sub>13 Pr<sub>oo</sub>f <sub>o</sub>f Pr<sub>opos</sub>iti<sub>o</sub>n 2

Pro<sub>p</sub>osition 2 (Existence of SPNE in Finite Horizon [45]). Every fully observed concurrent stochastic game with finite state and <sub>ac</sub>ti<sub>on spaces,</sub> M<sub>ar</sub>k<sub>ov</sub>i<sub>an s</sub>t<sub>age an</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d<sub>s, an</sub>d h<sub>or</sub>i<sub>zon</sub> bounded b<sub>y</sub> � admits a sub<sub>g</sub>ame-<sub>p</sub>erfect Nash e<sub>q</sub>uilibrium (SPNE) i<sub>n</sub> <sub>ran</sub>d<sub>om</sub>i<sub>se</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> th<sub>a</sub>t i<sub>s</sub> M<sub>ar</sub>k<sub>ov,</sub> i<sub>.e.</sub> <sub>eac</sub>h <sub>agen</sub>t’<sub>s</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub>d th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> ti<sub>me</sub> $H - t .$ In particular, this holds for the fully observed product game $\widehat { \mathcal { M } } _ { H }$ <sub>w</sub>ith utilities from Definition 2.

Proof. This is the standard backward-induction ar<sub>g</sub>ument [21], <sub>recor</sub>d<sub>e</sub>d f<sub>or comp</sub>l<sub>e</sub>t<sub>eness.</sub> L<sub>e</sub>t $r _ { i }$ be a<sub>g</sub>ent �’s sta<sub>g</sub>e reward and set $V _ { i } ^ { t } ( s )$ to its terminal <sub>p</sub>a<sub>y</sub>of whenever $( s , t )$ is terminal<sub>,</sub> incl<sub>u</sub>din<sub>g</sub> at $t = H$ <sub>.</sub> In $\widehat { \mathcal { M } } _ { H } , r _ { i } \equiv 0 , \gamma = 1$ <sub>, an</sub>d th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>payo</sub>f i<sub>s</sub> $R _ { i } ( x )$ F<sub>o</sub>r ${ t = H - 1 , \ldots , 0 }$ <sub>an</sub>d <sub>every non-</sub>t<sub>erm</sub>i<sub>na</sub>l $( s , t )$ <sub>,</sub> l<sub>e</sub>t $G _ { t } ( s )$ b<sub>e</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e norma</sub>l<sub>-</sub>f<sub>orm game w</sub>ith <sub>ac</sub>ti<sub>on se</sub>t<sub>s</sub> $d ^ { 1 } ( s ) , \ldots , d ^ { n } ( s )$ <sub>an</sub>d p<sup>a</sup>y<sup>ofs</sup>

$$
g _ { i } ^ { t } ( s , \pmb { a } ) \ \stackrel { \mathrm { d e f } } { = } \ r _ { i } ( s , \pmb { a } ) + \gamma \sum _ { s ^ { \prime } } \mathscr { P } ( s ^ { \prime } \mid s , \pmb { a } ) V _ { i } ^ { t + 1 } ( s ^ { \prime } ) .
$$

B<sub>y</sub> N<sub>as</sub>h’<sub>s</sub> th<sub>eorem</sub> $[ 4 0 ] , G _ { t } ( s )$ has a mixed e<sub>q</sub>uilibrium $\alpha ^ { t } ( s )$ <sub>.</sub> S<sub>e</sub>t $\sigma _ { i } ( s , t ) \ { \stackrel { \mathrm { d e f } } { = } } \ \alpha _ { i } ^ { t } ( s )$ <sub>an</sub>d $V _ { i } ^ { t } ( s ) \stackrel { \mathrm { d e f } } { = } \mathbb { E } _ { \pmb { a } \sim \alpha ^ { t } ( s ) } \left[ g _ { i } ^ { t } ( s , \pmb { a } ) \right]$ <sub>.</sub> B<sub>y</sub> ind<sub>uc</sub>ti<sub>o</sub>n on $H - t , V _ { i } ^ { t } ( s )$ is a<sub>g</sub>ent �’s ex<sub>p</sub>ected <sub>p</sub>a<sub>y</sub>of under ⃗� from state � at time $t ,$ irres<sub>p</sub>ective of the <sub>p</sub>recedin<sub>g</sub> histor<sub>y</sub>. Hence<sub>,</sub> as $\alpha ^ { t } ( s )$ i<sub>s</sub> <sub>a</sub>n e<sub>q</sub>uilibrium of $G _ { t } ( s )$ <sub>,</sub> no a<sub>g</sub>ent <sub>g</sub>ains from a one-sta<sub>g</sub>e deviation at an<sub>y</sub> histor<sub>y</sub>. B<sub>y</sub> the one-sta<sub>g</sub>e deviation <sub>p</sub>rinci<sub>p</sub>le for finite-horizon <sub>g</sub>ames [21], no deviation is <sub>p</sub>rofitable in an<sub>y</sub> sub<sub>g</sub>ame, so $\vec { \sigma }$ i<sub>s</sub> <sub>a</sub>n SPNE, and it is Markov b<sub>y</sub> construction. □

## D<sub>.</sub>14 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 5

Theorem 5 (Convergence of Nash-Q Values and Exploitability). Let Sel be a selector, and suppose that $( i ) Q ^ { 0 } ( x , \pmb { a } ) \in [ 0 , 1 ] ^ { n }$ for all $( x , \pmb { a } ) ; ( i i )$ for every reachable non-terminal � and $\pmb { a } \in d _ { H } ( \boldsymbol { x } )$ the step sizes satisfy $\alpha _ { k } ( x , \pmb { a } ) \in [ 0 , 1 ] , \sum _ { k } \alpha _ { k } ( x , \pmb { a } ) = \infty$ <sub>an</sub>d $\begin{array} { r } { \sum _ { k } \alpha _ { k } ( x , \pmb { a } ) ^ { 2 } < \infty } \end{array}$ almost surel<sub>y</sub>; (iii) each $\mathsf { S e l } _ { x }$ i<sub>s</sub> B<sub>ore</sub>l <sub>measur-</sub> able, the updated pair and step sizes are measurable functions of the past, and, conditional on that past, the successor is sampled from $\tilde { \mathcal { P } } ( \cdot \mid x , a )$ ; and (iv) for ever<sub>y</sub> reachable non-terminal $x , \mathsf { V a l } _ { x }$ is continuous at $Q ^ { \mathsf { S e l } } ( x )$ <sub>, w</sub>hi<sub>c</sub>h h<sub>o</sub>ld<sub>s</sub> i<sub>n par</sub>ti<sub>cu</sub>l<sub>ar</sub> $i f S e l _ { x }$ is continuous at $Q ^ { \mathsf { S e l } } ( x )$ or if all Nash equilibria o $f \bar { Q } ^ { \mathsf { S e l } } ( x )$ h<sub>ave</sub> th<sub>e same va</sub>l<sub>ue vec</sub>t<sub>or.</sub> Th<sub>en</sub> $\vec { \sigma } _ { \mathsf { S e l } }$ is a Markov SPNE $o f \widehat { \mathcal { M } } _ { H }$ <sub>an</sub>d<sub>, a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y,</sub> $Q ^ { k } \to Q ^ { \mathsf { S e l } }$ on all reachable non-terminal states, � $( s _ { 0 } , \vec { \sigma } _ { k } )  u _ { i } ( s _ { 0 } , \vec { \sigma } _ { 5 \mathrm { e l } } )$ for ever<sub>y</sub> a<sub>g</sub>ent �, and Exploit $( s _ { 0 } , \vec { \sigma } _ { k } ) \to 0$ (Definition 4). In particular, $f o r e \nu e r y \varepsilon > 0 , \vec { \sigma } _ { k }$ is an �-NE for all suficiently large �. If moreover $\mathsf { S e l } _ { x }$ is continuous at $Q ^ { \mathsf { S e l } } ( x )$ for every reachable non-terminal $x ,$ th<sub>en</sub> <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y</sub> $\vec { \sigma } _ { k }  \vec { \sigma } _ { \mathsf { S e l } }$ <sub>on a</sub>ll <sub>reac</sub>h<sub>a</sub>bl<sub>e non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es.</sub>

P<sub>r</sub>oof<sub>.</sub> E<sub>qu</sub>ilib<sub>r</sub>i<sub>um.</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 11 <sub>app</sub>li<sub>es</sub> th<sub>e</sub> b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc-</sub> ti<sub>o</sub>n <sub>o</sub>f Pr<sub>opos</sub>iti<sub>o</sub>n 2 t<sub>o</sub> $\widehat { \mathcal { M } } _ { H }$ , choosin<sub>g</sub> Sel<sub>�</sub> $( Q ^ { \mathsf { S e l } } ( x ) )$ <sub>a</sub>t <sub>eac</sub>h <sub>non-</sub> t<sub>e</sub>rmin<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e.</sub> H<sub>e</sub>n<sub>ce</sub> $\vec { \sigma } _ { \mathsf { S e l } }$ is a Markov SPNE with continuation p<sup>a</sup>y<sup>ofs</sup> $V ^ { \mathsf { S e l } }$

Convergence ofQ-values. By (i), convex updates and equilibrium averaging keep all Q-values and targets in $[ 0 , 1 ] ^ { n }$ . <sup>W</sup>e <sub>p</sub>rove conver-<sub>g</sub>ence o<sup>f</sup> $Q ^ { k }$ <sub>an</sub>d $V ^ { \bar { k } }$ <sub>on</sub> <sub>reac</sub>h<sub>a</sub>bl<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> b<sub>y</sub> b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> <sub>on</sub> tim<sub>e.</sub> At <sub>eve</sub>r<sub>y</sub> t<sub>e</sub>rmin<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> $V ^ { k } = R = { \dot { V } } ^ { \mathsf { S e l } }$ <sub>, cover</sub>i<sub>ng a</sub>ll <sub>s</sub>t<sub>a</sub>t<sub>es a</sub>t $t = H$ <sup>and</sup> <sup>an</sup>y <sup>earlier</sup> <sup>sto</sup>pp<sup>in</sup>g <sup>states.</sup> <sup>Su</sup>pp<sup>ose</sup> ${ \cal V } ^ { k } ( y )  { \cal V } ^ { 5 { \mathsf { e l } } } ( y )$ almost surely at every reachable state � at time $t + 1 .$ Let $\mathcal { G } _ { k }$ cont<sub>a</sub>i<sub>n</sub> th<sub>e</sub> <sub>pas</sub>t <sub>an</sub>d th<sub>e</sub> <sub>curren</sub>t <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f <sub>up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>pa</sub>i<sub>r</sub> <sub>an</sub>d <sub>s</sub>t<sub>ep</sub> <sub>s</sub>i<sub>zes,</sub> before sam<sub>p</sub>lin<sub>g</sub> its successor $x ^ { \prime }$ . For an u<sub>p</sub>dated <sub>p</sub>air $( x , a )$ at a non-terminal state in time la<sub>y</sub>er �, condition (iii) and the definition <sub>o</sub>f $Q ^ { \mathsf { S e l } }$ <sub>g</sub><sup>i</sup>ve

$$
\begin{array} { r } { \left\| \mathbb { E } [ V ^ { k } ( x ^ { \prime } ) \mid { \mathcal G } _ { k } ] - Q ^ { 5 \mathrm { e l } } ( x , \pmb { a } ) \right\| _ { \infty } \le c _ { k } , } \end{array}
$$

$$
c _ { k } \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { m a x } _ { y } \left\| V ^ { k } ( y ) - V ^ { 5 \mathrm { e l } } ( y ) \right\| _ { \infty } \longrightarrow 0 ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e max</sub>i<sub>mum</sub> i<sub>s over reac</sub>h<sub>a</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>es a</sub>t ti<sub>me</sub> $t + 1$ <sub>.</sub> Th<sub>e</sub> <sup>tar</sup>g<sup>et errors</sup> $V _ { i } ^ { k } ( x ^ { \prime } ) - Q _ { i } ^ { 5 { \mathsf { e l } } } ( x , \mathbf { a } )$ <sub>are</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y one.</sub> S<sub>u</sub>bt<sub>rac</sub>ti<sub>ng</sub> $Q _ { i } ^ { \mathsf { S e l } } ( x , \pmb { a } )$ f<sub>rom</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e g</sub>i<sub>ves</sub> th<sub>e sca</sub>l<sub>ar recurs</sub>i<sub>on o</sub>f L<sub>emma</sub> 1 <sub>o</sub>f $[ 4 6 ]$ <sub>coo</sub>rdin<sub>a</sub>t<sub>ew</sub>i<sub>se, w</sub>ith $\alpha _ { k } = 0$ <sub>w</sub>h<sub>en</sub> $( x , a )$ i<sub>s</sub> <sub>no</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e</sub>d<sub>.</sub> Thus (ii) and that lemma, with $\kappa = 0$ and vanishin<sub>g</sub> bias $c _ { k }$ , <sup>i</sup>m<sub>p</sub><sup>l</sup><sub>y</sub> $Q ^ { k } ( x ) \stackrel { \textstyle } { \to } Q ^ { 5 { \mathsf { e l } } } ( x )$ <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y</sub> f<sub>or every reac</sub>h<sub>a</sub>bl<sub>e non-</sub>t<sub>erm</sub>i<sub>na</sub>l � at time �. Condition (iv) then <sub>g</sub>ives $V ^ { k } ( x ) = \mathsf { V a l } _ { x } ( Q ^ { k } ( x ) ) \to$ $V ^ { \mathsf { S e l } } ( x )$ <sub>,</sub> com<sub>p</sub>letin<sub>g</sub> the induction. Finiteness makes these conclu-<sub>s</sub>i<sub>ons s</sub>i<sub>mu</sub>lt<sub>aneous on a</sub>ll <sub>reac</sub>h<sub>a</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>es.</sub>

Suficient conditions $f o r ( i \nu )$ . Continuit<sub>y</sub> of Sel<sub>�</sub> at $g ^ { \star } = Q ^ { \mathsf { S e l } } ( x )$ im<sub>p</sub>lies contin<sub>u</sub>it<sub>y</sub> of $\mathsf { V a l } _ { x } ( g ) = \mathbb E _ { \pmb { a } \sim \mathsf { S e l } _ { x } ( g ) } [ g ( \pmb { a } ) ]$ th<sub>ere.</sub> Alt<sub>erna-</sub> tivel<sub>y,</sub> su<sub>pp</sub>ose all e<sub>q</sub>uilibria of $\boldsymbol { \cdot } \boldsymbol { g } ^ { \star }$ h<sub>ave</sub> <sub>va</sub>l<sub>ue</sub> $v ^ { \star }$ . <sup>F</sup>or an<sub>y</sub> $g _ { j }  g ^ { \star }$ com<sub>p</sub>actness and the e<sub>q</sub>uilibrium ine<sub>q</sub>ualities ensure that ever<sub>y</sub> limit <sub>p</sub>oint of $\mathsf { S e l } _ { x } ( g _ { j } )$ is an e<sub>qu</sub>ilibri<sub>u</sub>m of $g ^ { \star }$ . <sup>E</sup>ver<sub>y</sub> corres<sub>p</sub>on<sup>d</sup>- in<sub>g</sub> value limit is therefore $v ^ { \star } = \mathsf { V a l } _ { x } ( g ^ { \star } )$ , <sub>p</sub>rov<sup>i</sup>n<sub>g</sub> cont<sup>i</sup>nu<sup>i</sup>t<sub>y</sub>.

Continuity of payofs and exploitability. On Markov profiles, $u _ { i } ( s _ { 0 } , \vec { \sigma } )$ is a <sub>p</sub>ol<sub>y</sub>nomial in the action <sub>p</sub>robabilities at reachable states. A<sub>g</sub>ainst fixed Markov o<sub>pp</sub>onents, a<sub>g</sub>ent � faces a finitehorizon MDP<sub>,</sub> so its best-res<sub>p</sub>onse <sub>p</sub>a<sub>y</sub>of over all deviations is attained b<sub>y</sub> a deterministic Markov <sub>p</sub>olic<sub>y</sub> [42]. It is therefore the <sub>max</sub>i<sub>mum o</sub>f fi<sub>n</sub>it<sub>e</sub>l<sub>y many suc</sub>h <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>s.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> b<sub>o</sub>th $u _ { i } ( s _ { 0 } , \cdot )$ <sub>an</sub>d $\mathsf { E x p l o i t } ( s _ { 0 } , \cdot )$ <sub>are</sub> <sub>con</sub>ti<sub>nuous</sub> <sub>on</sub> M<sub>ar</sub>k<sub>ov</sub> <sub>pro</sub>fil<sub>es.</sub>

Limiting profiles. Fix an outcome on which $Q ^ { k } \to Q ^ { \hat { \mathsf { S } } \mathsf { e l } }$ . <sup>B</sup><sub>y</sub> comp<sup>actness</sup>, <sup>ever</sup>y <sup>subse</sup>q<sup>uence of</sup> $\left( \vec { \sigma } _ { k } \right)$ h<sub>as a</sub> f<sub>ur</sub>th<sub>er su</sub>b<sub>sequence</sub> <sub>converg</sub>i<sub>ng</sub> <sub>on</sub> <sub>reac</sub>h<sub>a</sub>bl<sub>e</sub> <sub>non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es</sub> t<sub>o</sub> <sub>some</sub> $\bar { \vec { \sigma } } .$ Writ<sub>e</sub> it<sub>s</sub> indi<sub>ces as</sub> $k _ { \ell } .$ Passin<sub>g</sub> to the limit in the sta<sub>g</sub>e-e<sub>qu</sub>ilibri<sub>u</sub>m ine<sub>qu</sub>aliti<sub>es s</sub>h<sub>ows</sub> th<sub>a</sub>t $\bar { \vec { \sigma } } ( x )$ is an e<sub>qu</sub>ilibri<sub>u</sub>m of $\breve { Q ^ { \sf S e l } } ( x )$ . Moreover<sub>,</sub> alon<sub>g</sub> thi<sub>s</sub> <sub>su</sub>b<sub>sequence,</sub>

$$
\begin{array} { l } { \mathbb { E } _ { { \pmb { a } } \sim \bar { \vec { \sigma } } ( x ) } [ Q ^ { 5 \mathrm { e l } } ( x , { \pmb { a } } ) ] = \underset { \ell  \infty } { \operatorname* { l i m } } \mathsf { V a l } _ { x } ( Q ^ { k _ { \ell } } ( x ) ) } \\ { = V ^ { 5 \mathrm { e l } } ( x ) , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom convergence o</sub>f b<sub>o</sub>th th<sub>e pro-</sub> files and Q-values, and the second from (iv). Backward induction then <sub>g</sub>ives continuation <sub>p</sub>a<sub>y</sub>ofs $V ^ { \mathsf { S e l } }$ and e<sub>q</sub>uilibrium in ever<sub>y</sub> reachable s<sub>u</sub>b<sub>g</sub>ame<sub>,</sub> as in Pro<sub>p</sub>osition 2. Th<sub>u</sub>s $u _ { i } ( s _ { 0 } , \bar { \vec { \sigma } } ) = u _ { i } ( s _ { 0 } , \vec { \sigma } _ { \mathsf { S e l } } )$ and Exploit $( s _ { 0 } , \bar { \vec { \sigma } } ) = 0$ . Continuit<sub>y y</sub>ields the claimed <sub>p</sub>a<sub>y</sub>of and ex<sub>p</sub>loitabilit<sub>y</sub> conver<sub>g</sub>ence, and hence eventual �-NE for each � $> 0$ Finall<sub>y,</sub> continuit<sub>y</sub> of each $\mathsf { S e l } _ { x }$ at $Q ^ { \mathsf { S e l } } ( x )$ ) gives $\vec { \sigma } _ { k } ( x )  \vec { \sigma } _ { 5 \mathrm { e l } } ( x )$ di<sub>rec</sub>tl<sub>y.</sub> □

## E Ex<sub>p</sub>eriments

## E.1 Ex<sub>p</sub>erimental Setu<sub>p</sub>

F<sub>or eac</sub>h t<sub>as</sub>k<sub>, we se</sub>l<sub>ec</sub>t th<sub>e</sub> l<sub>earner,</sub> di<sub>scoun</sub>t <sub>an</sub>d <sub>agen</sub>t <sub>w</sub>ith th<sub>e</sub> hi<sub>g</sub>hest mean satisfaction in the initial evaluation<sub>,</sub> usin<sub>g</sub> five seeds and the trainin<sub>g</sub> bud<sub>g</sub>ets in Table 9. A<sub>pp</sub>endix G summarises trainin<sub>g</sub> discounts b<sub>y</sub> environment and lists the remainin<sub>g</sub> h<sub>yp</sub>er<sub>p</sub>arameters.

Learnin<sub>g</sub> curves. Each row shows one a<sub>g</sub>ent<sub>,</sub> with the best variant per <sup>l</sup>earner <sup>f</sup>ami<sup>l</sup>y on t<sup>h</sup>e <sup>l</sup>e<sup>f</sup>t an<sup>d</sup> a<sup>ll</sup> o<sup>b</sup>jective-traine<sup>d</sup> variants <sub>on</sub> th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht<sub>.</sub> V<sub>ar</sub>i<sub>an</sub>t<sub>s</sub> <sub>are</sub> <sub>c</sub>h<sub>osen</sub> b<sub>y</sub> fi<sub>na</sub>l <sub>smoo</sub>th<sub>e</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>sa</sub>ti<sub>s-</sub> f<sub>ac</sub>ti<sub>on</sub> <sub>average</sub>d <sub>over</sub> fi<sub>ve</sub> <sub>see</sub>d<sub>s,</sub> <sub>a</sub>l<sub>so</sub> i<sub>n</sub> Fi<sub>gure</sub> 2<sub>.</sub> S<sub>o</sub>lid <sub>an</sub>d d<sub>as</sub>h<sub>e</sub>d lin<sub>es</sub> di<sub>s</sub>tin<sub>gu</sub>i<sub>s</sub>h <sub>va</sub>ri<sub>a</sub>nt<sub>s</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub>nd <sub>w</sub>ith CER<sub>;</sub> di<sub>a</sub>m<sub>o</sub>nd<sub>s</sub> m<sub>a</sub>rk the stron<sub>g</sub>est handcrafted baseline. Means and 95% confidence b<sub>an</sub>d<sub>s</sub> <sub>use</sub> <sub>exponen</sub>ti<sub>a</sub>l <sub>smoo</sub>thi<sub>ng</sub> $( \alpha = 0 . 0 0 2 )$ <sub>,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> <sub>a</sub> <sub>cen-</sub> tred window coverin<sub>g</sub> 3% of the e<sub>p</sub>isode ran<sub>g</sub>e. Bands ma<sub>y</sub> extend b<sub>eyon</sub>d th<sub>e ver</sub>ti<sub>ca</sub>l <sub>ax</sub>i<sub>s</sub> li<sub>m</sub>it<sub>s.</sub>

Computationa<sup>l</sup> resources. Eac<sup>h</sup> a<sup>l</sup>gorit<sup>h</sup>m–tas<sup>k</sup> jo<sup>b</sup> uses see<sup>d</sup>s $\{ 0 , 1 , 2 , 3 , 4 \}$ , one <sub>p</sub>h<sub>y</sub>sical CPU core (two threads) on an AMD EPYC 7552<sub>,</sub> 7<sub>.</sub>81 GiB RAM <sub>a</sub>nd <sub>a</sub> 96-h<sub>ou</sub>r limit<sub>, w</sub>ith<sub>ou</sub>t <sub>a</sub> GPU<sub>.</sub> Nash-Q variants complete only on Matrix Stag Hunt, using one Inte<sup>l</sup> Core U<sup>l</sup>tra 7 155H t<sup>h</sup>rea<sup>d</sup> per jo<sup>b</sup> an<sup>d</sup> t<sup>h</sup>e same training an<sup>d</sup> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>s.</sub>

<table><tr><td>Benchmark</td><td>n</td><td>Selected agent</td><td>Selected agent&#x27;s objective</td><td> $\widehat { p } _ { i }$ </td><td>95% CI</td><td>m</td><td> $\mathsf { N a s h C o n v }$ </td></tr><tr><td>Pursuit Classic 2</td><td>2</td><td> $\mathsf { p l a y e r } _ { - } \Theta$ </td><td> $\mathsf { P } _ { \geq 0 . 1 } [ \mathsf { G } \neg \mathsf { g o a l } _ { . }$  reached]</td><td>0.979</td><td>[0.936,0.993]</td><td>-0.744</td><td>0.000</td></tr><tr><td>Pursuit Capture 2</td><td>2</td><td> $\mathsf { p l a y e r } _ { - } 1$ </td><td> $\mathsf { P } _ { \geq 0 . 5 } [ \mathsf { G } \neg \mathsf { g u a r d } _ { - }$  intruder_collision∧  $\mathsf { G } \mathsf { \neg c o l l i s i o n \_ e v e r }$ </td><td>1.000</td><td>[0.957, 1.000]</td><td>-0.300</td><td>0.000</td></tr><tr><td>Pursuit Classic 3</td><td>3</td><td> $\mathsf { p l a y e r } _ { - } 2$ </td><td> $\mathsf { P } _ { \geq 0 . 1 } [ \mathsf { F g o a l } _ { }$  reached ∧G¬collision_ever]</td><td>0.975</td><td>[0.932,0.993]</td><td>-0.500</td><td>0.008</td></tr><tr><td>Pursuit Capture 3</td><td>3</td><td> $\mathsf { p l a y e r } _ { - } 2$ </td><td> $\mathsf { P } _ { \geq 0 . 8 } [ \mathsf { G } \neg \mathsf { g u a r d }$  _intruder_collision∧  $\mathsf { G } \mathsf { \neg c o l l i s i o n \_ e v e r }$ </td><td>0.969</td><td>[0.926, 0.998]</td><td>-0.049</td><td>0.002</td></tr><tr><td>Coins Own Coin First</td><td>2</td><td> $\mathsf { p l a y e r } _ { - } 1$ </td><td> $\mathsf { P } _ { \geq 0 . 4 } [ \neg w _ { 1 } \cup m _ { 1 } ]$ </td><td>0.503</td><td>[0.460, 0.546]</td><td>+0.046</td><td>0.044</td></tr><tr><td>Matrix Sustained Stag</td><td>2</td><td> $\mathsf { p l a y e r } _ { - } 1$ </td><td> $\mathsf { P } _ { \geq 0 . 9 } [ \mathsf { F } ( \mathtt { b o t h \_ s t a g } \wedge \mathsf { X } ( \mathtt { b o t h \_ s t a g } \wedge \mathsf { G } \cdot \mathsf { G } ^ { \prime } )$  unmirrored))]</td><td>1.000</td><td>[0.957, 1.000]</td><td>+0.053</td><td>0.000</td></tr><tr><td>Commons Safe Harvest</td><td>4</td><td> $\mathsf { p l a y e r } _ { - } \Theta$ </td><td> $\mathsf { P { \mathop { \ge 0 . 8 } } } \mathsf { [ G \mathrm { \ - p 1 a y e r \_ 0 \_ z a p p e d \ } } \wedge$  Fplayer_0_apple_collected]</td><td>0.817</td><td>[0.774,0.860]</td><td>-0.383</td><td>0.206</td></tr><tr><td>Territory Role Claim</td><td>2</td><td></td><td>player_0 P≥0.25[¬p1_claimedU(po_claimed  $\land \neg \mathtt { p 1 \_ c l a i m e d } ]$ </td><td>0.923</td><td>[0.880,0.944]</td><td>-0.156</td><td>0.031</td></tr></table>

Table 3: Selected-agent objectives, marginal intervals and joint margins for the learners in Table 1. Satisfaction estimates and marginal 95% Hoefding interval end<sub>p</sub>oints are five-seed means from 1000 inde<sub>p</sub>endent confirmator<sub>y</sub> rollouts <sub>p</sub>er seed. The repor<sup>t</sup>e<sup>d</sup> marg<sup>i</sup>n averages ${ \underline { { m } } } ( { \vec { \sigma } } ) \ { \stackrel { \mathrm { d e f } } { = } } \ \operatorname* { m i n } _ { i } ( L _ { i } - q _ { i } )$ <sub>over see</sub>d<sub>s, w</sub>h<sub>ere</sub> $L _ { i }$ is the joint Hoefding lower bound with $\delta _ { i } = 0 . 0 5 / n$ f<sub>or</sub> � agents. A non-negative <sub>p</sub>er-seed margin certifies ever<sub>y</sub> agent’s threshold for that <sub>p</sub>rofile. $\mathsf { N a s h C o n v } ^ { \approx }$ i<sub>s average</sub>d <sub>over see</sub>d<sub>s.</sub> Objectives omit the outer X.

<table><tr><td>Benchmark</td><td>Agent</td><td>Objective φi</td><td>qi</td><td> $\widehat { p } _ { i }$ </td><td>Marginal 95% CI</td><td>Joint 95% CI</td><td>Certified seeds</td><td> $\mathsf { N a s h C o n v } ^ { \approx }$ </td></tr><tr><td>Classic 2</td><td> $\mathsf { p l a y e r } _ { - } \Theta$ </td><td>G¬goal_reached</td><td>0.1</td><td>0.979</td><td>[0.936, 0.993]</td><td>[0.932, 0.994]</td><td>5/5</td><td>0.000</td></tr><tr><td>Classic 2</td><td> $\mathsf { p l a y e r } _ { - } 1$ </td><td>Fgoal_reached ∧G¬collision_ever</td><td>0.75</td><td>0.020</td><td>[0.007, 0.063]</td><td>[0.006, 0.067]</td><td>0/5</td><td>0.000</td></tr><tr><td>Capture 2</td><td> $\mathsf { p l a y e r } _ { - } \Theta$ </td><td>Fguard_intruder_collision</td><td>0.3</td><td>0.000</td><td>[0.000, 0.043]</td><td>[0.000, 0.047]</td><td>0/5</td><td>0.000</td></tr><tr><td>Capture 2</td><td>player_1</td><td>G¬guard_intruder_collision∧G¬collision_ever</td><td>0.5</td><td>1.000</td><td>[0.957, 1.000]</td><td>[0.953, 1.000]</td><td>5/5</td><td>0.000</td></tr><tr><td>Classic 3</td><td>player_0</td><td>G¬goal_reached</td><td>0.5</td><td>0.011</td><td>[0.000, 0.054]</td><td>[0.000, 0.060]</td><td>0/5</td><td>0.008</td></tr><tr><td>Classic 3</td><td>player_1</td><td>Fgoal_reached ∧ G¬collision_ever</td><td>0.1</td><td>0.975</td><td>[0.932, 0.993]</td><td>[0.926,0.994]</td><td>5/5</td><td>0.008</td></tr><tr><td>Classic 3</td><td> $\mathsf { p l a y e r } _ { - } 2$ </td><td>Fgoal_reached ∧ G¬collision_ever</td><td>0.1</td><td>0.975</td><td>[0.932, 0.993]</td><td>[0.926,0.994]</td><td>5/5</td><td>0.008</td></tr><tr><td>Capture 3</td><td> $\mathsf { p l a y e r } _ { - } \Theta$ </td><td>Fguard_intruder_collision</td><td>0.05</td><td>0.030</td><td>[0.002, 0.073]</td><td>[0.001,0.079]</td><td>0/5</td><td>0.002</td></tr><tr><td>Capture 3</td><td> $\mathsf { p l a y e r } _ { - } 1$ </td><td>G¬guard_intruder_collision∧G¬collision_ever</td><td>0.8</td><td>0.969</td><td>[0.926, 0.998]</td><td>[0.920, 0.999]</td><td>5/5</td><td>0.002</td></tr><tr><td>Capture 3</td><td> $\mathsf { p l a y e r } _ { - } 2$ </td><td>G¬guard_intruder_collision∧G¬collision_ever</td><td>0.8</td><td>0.969</td><td>[0.926, 0.998]</td><td>[0.920, 0.999]</td><td>5/5</td><td>0.002</td></tr></table>

Tab<sup>l</sup>e 4: Per-agent Pursuit resu<sup>l</sup>ts: PR2-IPPO <sup>f</sup>or C<sup>l</sup>assic 2 and Capture 2, PR2-IQL-CER <sup>f</sup>or C<sup>l</sup>assic 3, and IQL-CER <sup>f</sup>or Capture 3. Satisfaction estimates and Hoefding interval end<sub>p</sub>oints are five-seed means from 1000 inde<sub>p</sub>endent confirmator<sub>y</sub> rollouts <sub>per see</sub>d<sub>, s</sub>h<sub>are</sub>d <sub>across agen</sub>t<sub>s.</sub> M<sub>arg</sub>i<sub>na</sub>l i<sub>n</sub>t<sub>erva</sub>l<sub>s use</sub> $\delta = 0 . 0 5 ;$ joint intervals use $\delta _ { i } = 0 . 0 5 / m$ for � agents. Certified seeds counts runs whose joint lower bound $L _ { i }$ <sub>mee</sub>t<sub>s</sub> th<sub>e row</sub>’<sub>s</sub> th<sub>res</sub>h<sub>o</sub>ld $q _ { i }$ $\mathsf { N a s h C o n v } ^ { \approx }$ i<sub>s average</sub>d <sub>over see</sub>d<sub>s an</sub>d <sub>s</sub>h<sub>are</sub>d <sub>across eac</sub>h <sub>pro</sub>fil<sub>e</sub>’<sub>s</sub> <sub>agen</sub>t<sub>s.</sub>

<table><tr><td>Agent</td><td>Objective φi</td><td>qi</td><td> $\widehat { p } _ { i }$ </td><td>Marginal 95% CI</td><td>Joint 95% CI</td><td>Certified seeds</td><td> $\mathsf { N a s h C o n v }$ </td></tr><tr><td> $\mathsf { p l a y e r } _ { - } \Theta$ </td><td> $\neg w _ { 0 } \cup m _ { 0 }$ </td><td>0.4</td><td>0.507</td><td> $[ 0 . 4 6 4 , 0 . 5 5 0 ]$ </td><td>[0.460, 0.554]</td><td>5/5</td><td>0.044</td></tr><tr><td>player_1</td><td> $\neg w _ { 1 } \lor m _ { 1 }$ </td><td>0.4</td><td>0.503</td><td> $[ 0 . 4 6 0 , 0 . 5 4 6 ]$ </td><td>[0.456, 0.550]</td><td>5/5</td><td>0.044</td></tr></table>

Tab<sup>l</sup>e 5: Per-agent Coins resu<sup>l</sup>ts <sup>f</sup>or IPPO-CER; $m _ { i } , w _ { i }$ denote matchin<sub>g</sub>- and mismatchin<sub>g</sub>-coin collections. Satisfaction estimates and Hoefding interval end<sub>p</sub>oints are five-seed means from 1000 inde<sub>p</sub>endent confirmator<sub>y</sub> rollouts <sub>p</sub>er seed, shared across a<sub>g</sub>ents. Mar<sub>g</sub>inal intervals use $\delta = 0 . 0 5 ;$ joint intervals use $\delta _ { i } = 0 . 0 2 5$ for the two agents. Certified seeds counts runs whose joint l<sub>ower</sub> b<sub>oun</sub>d $L _ { i }$ <sub>mee</sub>t<sub>s</sub> th<sub>e row</sub>’<sub>s</sub> th<sub>res</sub>h<sub>o</sub>ld $q _ { i }$ . NashConv<sup>≈</sup> is averaged over seeds and shared across agents.

![](images/6a0c909ea99f702199794527fef44fbb1b6e4731c6da7567fe44f1805f80b8da.jpg)  
(a) Guard (player\_0), best algorithms

![](images/619516dedb93387209f687e728a9ee63acecd492d3d75d52026bb952a5f5434d.jpg)  
(b) Guard (player\_0), all algorithms

![](images/63b6bf7c8183a1f07a9b97b71c635af1882ac7b62fde95b6d4fc2c4000a74acc.jpg)  
(c) Intruder $( \mathsf { p l a y e r \_ 1 } ) ,$ b<sub>es</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub>

![](images/1d80541a41b090a1cbff952a078a00beabd1b8c1212479c2a6e6a7cb9dd07f5d.jpg)  
(d) Intruder $( \mathsf { p l a y e r \_ 1 } ) ,$ <sub>a</sub>ll <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub>

Figure 6: Two-player Pursuit Classic: per-agent satisfaction (mean and 95% CIs over five seeds). Left: best variant per learner family; right: all objective-trained variants.
<table><tr><td>Agent</td><td>Objective  $\varphi _ { i }$ </td><td>qi</td><td> $\widehat { p } _ { i }$ </td><td>Marginal 95% CI</td><td>Joint 95% CI</td><td>Certified seeds</td><td> $\mathsf { N a s h C o n v }$ </td></tr><tr><td>player_0</td><td>F(both_stag ∧X(both_stag∧ G¬unmirrored))</td><td>0.9</td><td>1.000</td><td>[0.957,1.000]</td><td>[0.953, 1.000]</td><td>5/5</td><td>0.000</td></tr><tr><td>player_1</td><td>F(both_stag∧X(both_stag ∧G¬unmirrored))</td><td>0.9</td><td>1.000</td><td>[0.957, 1.000]</td><td>[0.953, 1.000]</td><td>5/5</td><td>0.000</td></tr></table>

Tab<sup>l</sup>e 6: Per-agent Matrix Stag Hunt resu<sup>l</sup>ts <sup>f</sup>or Nash-Q-CER. Satis<sup>f</sup>action estimates and Hoe<sup>f</sup>ding interva<sup>l</sup> endpoints are five-seed means from 1000 inde<sub>p</sub>endent confirmator<sub>y</sub> rollouts <sub>p</sub>er seed, shared across agents. Marginal intervals use $\delta = 0 . 0 5 ;$ joint intervals use $\delta _ { i } = 0 . 0 2 5$ for the two agents. Certified seeds counts runs whose joint lower bound $L _ { i }$ <sub>mee</sub>t<sub>s</sub> th<sub>e row</sub>’<sub>s</sub> th<sub>res</sub>h<sub>o</sub>ld $q _ { i }$ . Nash $\mathsf { C o n v } ^ { \approx }$ i<sub>s average</sub>d <sub>over see</sub>d<sub>s an</sub>d <sub>s</sub>h<sub>are</sub>d <sub>across agen</sub>t<sub>s.</sub>

<table><tr><td>Agent</td><td>Objective φi</td><td>qi</td><td> $\widehat { p } _ { i }$ </td><td>Marginal 95% CI</td><td>Joint 95% CI</td><td>Certified seeds</td><td>NashConv≈</td></tr><tr><td>player_0</td><td> ${ \sf G \to p l a y e r \_ 0 } _ { - }$  zapped ∧ Fplayer_0_apple_collected</td><td>0.8</td><td>0.817</td><td>[0.774,0.860]</td><td>[0.767, 0.867]</td><td>3/5</td><td>0.206</td></tr><tr><td>player_1</td><td>G¬player_1_zapped ∧ Fplayer_1_apple_collected</td><td>0.7</td><td>0.608</td><td>[0.565, 0.651]</td><td>[0.558, 0.658]</td><td>3/5</td><td>0.206</td></tr><tr><td>player_2</td><td>G¬player_2_zapped ∧ Fplayer_2_apple_collected</td><td>0.3</td><td>0.001</td><td>[0.000, 0.044]</td><td>[0.000, 0.051]</td><td>0/5</td><td>0.206</td></tr><tr><td>player_3</td><td>G¬player_3_zapped ∧ Fplayer_3_apple_collected</td><td>0.2</td><td>0.005</td><td>[0.000, 0.048]</td><td>[0.000, 0.056]</td><td>0/5</td><td>0.206</td></tr></table>

Tab<sup>l</sup>e 7: Per-agent Commons Harvest resu<sup>l</sup>ts <sup>f</sup>or IQL-CER. Satis<sup>f</sup>action estimates and Hoe<sup>f</sup>ding interva<sup>l</sup> endpoints are <sup>fi</sup>ve-seed means from 1000 inde<sub>p</sub>endent confirmator<sub>y</sub> rollouts <sub>p</sub>er seed, shared across agents. Marginal intervals use $\delta = 0 . 0 5 ;$ joint i<sub>n</sub>t<sub>erva</sub>l<sub>s use</sub> $\delta _ { i } = 0 . 0 1 2 5$ for the four agents. Certified seeds counts runs whose joint lower bound $L _ { i }$ <sub>mee</sub>t<sub>s</sub> th<sub>e row</sub>’<sub>s</sub> th<sub>res</sub>h<sub>o</sub>ld $q _ { i } . \mathsf { N a s h C o n v } ^ { \approx }$ i<sub>s avera e</sub>d <sub>over see</sub>d<sub>s an</sub>d <sub>s</sub>h<sub>are</sub>d <sub>across a en</sub>t<sub>s.</sub>

$$
\_ { \mathrm { P R 2 - I O L ~ - I O L - C R ~ \_ { C E R } ~ \_ { N a s h - Q ~ L e a r n i n g } ~ -- ~ \_ { N a s h - Q ~ L e a r n i n g ~ ( C E R ) } ~ * ~ \_ { B e s t } ~ H a n d c r a f t e d } }
$$

![](images/fd3d912fe4935fab8be3b8ce3c33fc0f96e403a17ced4c4d4a4b9a61da285e83.jpg)  
(a) Guard (player\_0), best algorithms

![](images/2737e5c4575d050374a1381334b42dff102efe1dbd5f9b536aa62d20d05c2330.jpg)  
(b) Guard (player\_0), all algorithms

![](images/7eba548068406f33eb717b8cf031dc35838cabdd5e09013b675691734a7755e6.jpg)  
(c) Intruder (player\_1), best algorithms

![](images/6a811361d853399b5cd855e5d412c29c65789164171d9c83b6495859004b75ab.jpg)  
(d) Intruder (player\_1), all algorithms

Figure 7: Two-player Pursuit Capture: per-agent satisfaction (mean and 95% CIs over five seeds). Left: best variant per learner family; right: all objective-trained variants.
<table><tr><td>Agent</td><td>Objective  $\varphi _ { i }$ </td><td>qi</td><td> $\widehat { p } _ { i }$ </td><td>Marginal 95% CI</td><td>Joint 95% CI</td><td>Certified seeds</td><td>NashConv≈</td></tr><tr><td>player_0</td><td> $\neg \mathtt { p } _ { 1 _ { - } } \mathtt { c l a i m e d } \cup \mathtt { ( p _ { 0 _ - } c l a i m e d \wedge \neg { p } _ { 1 _ - } c l a i m e d ) }$ </td><td>0.25</td><td>0.923</td><td>[0.880,0.944]</td><td>[0.876, 0.945]</td><td>5/5</td><td>0.031</td></tr><tr><td>player_1</td><td> $\neg \mathtt { p } _ { 0 _ { - } }$  claimed U (p₁_claimed ∧ ¬Po_claimed)</td><td>0.2</td><td>0.054</td><td>[0.045, 0.097]</td><td>[0.044, 0.101]</td><td>1/5</td><td>0.031</td></tr></table>

Table 8: Per-a<sub>g</sub>ent Territor<sub>y</sub> results for $\mathtt { I Q L } ; \mathtt { p _ { i _ { - } } c } .$ laimed a<sup>bb</sup>rev<sup>i</sup>ates player\_i\_resource\_claimed. <sup>S</sup>at<sup>i</sup>s<sup>f</sup>act<sup>i</sup>on est<sup>i</sup>mates an<sup>d</sup> Hoefding interval end<sub>p</sub>oints are five-seed means from 1000 inde<sub>p</sub>endent confirmator<sub>y</sub> rollouts <sub>p</sub>er seed, shared across agents. Mar<sub>g</sub>inal intervals use $\delta = 0 . 0 5 ;$ joint intervals use $\delta _ { i } = 0 . 0 2 5$ for the two agents. Certified seeds counts runs whose joint lower b<sub>oun</sub>d $L _ { i }$ <sub>mee</sub>t<sub>s</sub> th<sub>e</sub> <sub>row</sub>’<sub>s</sub> th<sub>res</sub>h<sub>o</sub>ld $q _ { i }$ . NashConv<sup>≈</sup> is averaged over seeds and shared across agents.

![](images/449850fad2d5dcf8e338e1aa3044e6546b81cde011ee89de4ecbbb8def67cd7c.jpg)  
(a) Guard (player\_0), best algorithms

![](images/7cd80f0b4ea2da462ea5d1965a54ddfa8f8db3b8e1da6f3127b39ea04cbcbdf5.jpg)  
(b) Guard (player\_0), all algorithms

![](images/f84eeeee726af4fe2d28ed0df84fd99b6a17fdbde1b00252cf6ec7f15e82d15f.jpg)  
(c) Intruder 1 (player\_1), best algorithm

![](images/c9186cf1455f72ac56c600edac56eb29c42023bc3fb49d72023650ed48a03577.jpg)  
(d) Intruder 1 (player\_1), all algorithms

![](images/0f11151b1b428f47845c8dc929663bf09897b4d8a2ae01cf422fe014c4dbd198.jpg)  
(e) Intruder 2 (player\_2), best algorithm

![](images/18854c541f196142809bf4513a0d432fc320a437a75888cbf041fee6b45298fd.jpg)  
(f) Intruder 2 (player\_2), all algorithms  
Figure 8: Three-player Pursuit Classic: per-agent satisfaction (mean and 95% CIs over five seeds). Left: best variant per learner family; right: all objective-trained variants.

![](images/b7dae87f3de1b27bf43ac8aa564e9c2253c110cdb28729d04f2a10675aa10dc4.jpg)  
(a) Guard (player\_0), best algorithms

![](images/b421d2817d8ff25870164c077201c151064c8d2286726afd4c652b6f1c55062d.jpg)  
(b) Guard (player\_0), all algorithms

![](images/53ad3361f78f671fb53d7fbf8b38d38a70f1d0fc9f9e92fc7749412bd6f0f3c6.jpg)  
(c) Intruder 1 (player\_1), best algorithm

![](images/f85f8444c955fe863bf2c5184bcb328a5b17f94ca12cae72905c61411ddf3628.jpg)  
(d) Intruder 1 (player\_1), all algorithms

![](images/cef2d63d3289f970c7e991557bf56e173a11e9c51f03e1d2de300dd1f64ecfff.jpg)  
(e) Intruder 2 (player\_2), best algorithms

![](images/2ededeebcafe22b7c18ea03a703546ad2ba1046a1de6bef167c3315fe02d9a5b.jpg)  
(f) Intruder 2 (player\_2), all algorithms  
Figure 9: Three-player Pursuit Capture: per-agent satisfaction (mean and 95% CIs over five seeds). Left: best variant per learner family; right: all objective-trained variants.

IPPO - - IPPO-CER  PR2-IPPO - - PR2-IPPO-CER  IQL - - IQL-CER PR2-IQL   PR2-IQL-CER  Nash-Q Learning  Nash-Q Learning (CER)  Best Handcrafted  
![](images/dd99f4261fc7cd173e6941d7c341f5b011ab650fe3146403775b55171c03d78e.jpg)  
(a) player\_0, best algorithms

![](images/bc8935f2f77b55e0eabded040c56bad3d40d477df840a7e19ab01a3b38240aea.jpg)  
(b) player\_0, all algorithms

![](images/233fd0a53e3ce8cff660396739d5f7a077d315fb57fc55395449a247456f492e.jpg)  
(c) player\_1, best algorithms

![](images/3935417e4ed0d03db666c95d40b15dfb198e070fc3542298ba8235afb6dd177a.jpg)  
(d) player\_1, all algorithms  
Figure 10: Coins: per-agent satisfaction (mean and 95% CIs over five seeds). Left: best variant per learner family; right: all objective-trained variants.

![](images/f44208069b990416abd8ce56168070363e7d495b193eb3adda404d9bf85a9dfe.jpg)  
(a) player\_0, best algorithms

![](images/a740c54061d31c8fde6e32afc9e04440c76c3b199464db23b21252abc4465219.jpg)  
(b) player\_0, all algorithms

![](images/2042db9ecf118e558935b5249e09bcbbddbd14f4acc3b4e98d540e08ccbf4373.jpg)  
(c) player\_1, best algorithms

![](images/7f9d0d16e91fdcabd050f550ce2877b929c3d9f2d5ec1018118548a8efe41a43.jpg)  
(d) player\_1, all algorithms  
Figure 11: Matrix Stag Hunt: per-agent satisfaction (mean and 95% CIs over five seeds). Left: best variant per learner family; right: all objective-trained variants.

![](images/93675cc0e98a451bd881c51442189fb93ffaa83d1c3ad62f981fe83699e8e89c.jpg)  
(a) player\_0, best algorithms

![](images/1d9376ec45c650aebbb69e2ca654dabbbe6e558ba35c490479fff863e0210480.jpg)  
(b) player\_0, all algorithms

![](images/15d0d131c8efa3b88c5f18a05df3c307d5d8fc616d24beae9bb15833127d47da.jpg)  
(c) player\_1, best algorithms

![](images/53823dfe6a927b04ce4823e6a3041fa0a4969f86a7e5aaede14853e65faef243.jpg)  
(d) player\_1, all algorithms

![](images/f6f598b928be5843eee714357ddbc8649b171158b6fe91b5253af6eaab646a5a.jpg)  
(e) player\_2, best algorithms

![](images/5cb12a0afbf6747cc7e11c014ef4f2bf6375eb046f8f3e1f31664ad85539ecb8.jpg)  
(f) player\_2, all algorithms

![](images/cd202c964896f7459a187acd5f444a889726ace02c46448c4695b62bc167db25.jpg)  
(g) player\_3, best algorithms

![](images/53934069e07389d3e7f3f5d255889d0470ff57959d88d796ebe35e699d432254.jpg)  
(h) player\_3, all algorithms  
Figure 12: Commons Harvest: per-agent satisfaction (mean and 95% CIs over five seeds). Left: best variant per learner family; right: all objective-trained variants.

![](images/5eda2f46472302a4eadbd6edd165f08adc6e1491188c2b9f25aeb7d128289782.jpg)  
(a) Competitor (player\_0), best algorithms

![](images/54b9d4ed01b621c1f32248df240235c91094a781ad93f8d236485cb812ea2efd.jpg)  
(b) Competitor (player\_0), all algorithms

![](images/b0b64ead38301ff76b91f3264ad85be7afad8998eb3b266852b3fb0c24cadcc1.jpg)  
(c) Competitor (player\_1), best algorithms

![](images/9c28254f1fe332cd2b61e277172f98bf47a48a414f67455c4a69a77a27ad4393.jpg)  
(d) Competitor (player\_1), all algorithms  
Figure 13: Territory: per-agent satisfaction (mean and 95% CIs over five seeds). Left: best variant per learner family; right: all objective-trained variants.

## F Ext<sub>e</sub>nd<sub>e</sub>d Lit<sub>e</sub>r<sub>a</sub>t<sub>u</sub>r<sub>e</sub> R<sub>ev</sub>i<sub>ew</sub>

Probabilistic Strate<sub>g</sub>ic Lo<sub>g</sub>ics and S<sub>y</sub>nthesis Tools. Hahn et al. [25] trans<sup>l</sup>ate reac<sup>h</sup>a<sup>b</sup>i<sup>l</sup>ity-norma<sup>l</sup> patl o<sup>b</sup>jectives into rewar<sup>d</sup>s <sup>f</sup>or Minimax Q-Learning. Their model-checking procedure recursively rel<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> f<sub>or</sub> <sub>nes</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>a</sub>liti<sub>es,</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> fi<sub>n</sub>di<sub>ng</sub> <sub>one</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> witnessin<sub>g</sub> all subformulae. It considers a sin<sub>g</sub>le coalition in turnbased <sub>g</sub>ames. Bozkurt et al. [8] similarl<sub>y</sub> stud<sub>y</sub> model-free learnin<sub>g</sub> <sup>f</sup>or ltl o<sup>b</sup>jectives in two-p<sup>l</sup>ayer turn-<sup>b</sup>ase<sup>d</sup> zero-sum games, using d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> R<sub>a</sub>bin <sub>au</sub>t<sub>o</sub>m<sub>a</sub>t<sub>a.</sub>

The zero-sum fra<sub>g</sub>ments of PRISM-Games [31] s<sub>y</sub>nthesise strategies <sup>f</sup>or pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>istic an<sup>d</sup> rewar<sup>d</sup>-<sup>b</sup>ase<sup>d</sup> rpatl o<sup>b</sup>jectives <sup>f</sup>rom <sup>k</sup>nown transition mo<sup>d</sup>e<sup>l</sup>s. Sing<sup>l</sup>e- an<sup>d</sup> mu<sup>l</sup>ti-o<sup>b</sup>jective properties concern one coalition a<sub>g</sub>ainst its com<sub>p</sub>lement<sub>;</sub> where o<sub>p</sub>timal strate<sub>g</sub>ies need not exist [12], the tool com<sub>p</sub>utes �-a<sub>pp</sub>roximations of Pareto-o<sub>p</sub>timal strate<sub>g</sub>ies.

E<sub>q</sub>uilibrium-based extensions s<sub>y</sub>nthesise sub<sub>g</sub>ame-<sub>p</sub>erfect<sub>,</sub> social-welfare-o<sub>p</sub>timal Nash e<sub>q</sub>uilibria in concurrent stochastic <sub>g</sub>ames [32–34], also assumin<sub>g</sub> known models.

Varricchione et al. [53] construct reward machines from atl s<sub>p</sub>ecifications for a full<sub>y</sub> coo<sub>p</sub>erative <sub>g</sub>rand coalition.

Rational S<sub>y</sub>nthesis. Rational s<sub>y</sub>nthesis [20, 52] considers a<sub>g</sub>ents <sub>p</sub>ursuin<sub>g</sub> their own tem<sub>p</sub>oral <sub>g</sub>oals. Fisman et al. [20] reduce ltl rational s<sub>y</sub>nthesis to satisfiabilit<sub>y</sub> in Extended Strate<sub>gy</sub> Lo<sub>g</sub>ic<sub>,</sub> obtainin<sub>g</sub> 2EXPTIME-com<sub>p</sub>leteness for dominant-strate<sub>gy</sub>, Nash and sub<sub>g</sub>ame-<sub>p</sub>erfect solution conce<sub>p</sub>ts.

Ummels [52] stud<sub>y</sub> <sub>p</sub>ure strate<sub>g</sub>ies in infinite multi<sub>p</sub>la<sub>y</sub>er turn-based games with �-regular objectives, establishing exis t<sub>ence,</sub> <sub>payo</sub>f<sub>-cons</sub>t<sub>ra</sub>i<sub>ne</sub>d d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>proce</sub>d<sub>ures</sub> <sub>an</sub>d fi<sub>n</sub>it<sub>e-s</sub>t<sub>a</sub>t<sub>e</sub> im<sub>p</sub>lementations for sub<sub>g</sub>ame-<sub>p</sub>erfect Nash e<sub>q</sub>uilibria. Condurache et al. [14] stud<sub>y</sub> com<sub>p</sub>lexit<sub>y</sub> refinements and coo<sub>p</sub>erative and n<sub>o</sub>n-<sub>coope</sub>r<sub>a</sub>ti<sub>ve</sub> <sub>va</sub>ri<sub>a</sub>nt<sub>s.</sub>

Our settin<sub>g</sub> combines <sub>p</sub>er-a<sub>g</sub>ent <sub>p</sub>robabilistic $\mathrm { L T L } _ { f }$ re<sub>q</sub>u<sup>i</sup>rements <sub>w</sub>ith l<sub>earn</sub>i<sub>ng</sub> f<sub>rom</sub> <sub>un</sub>k<sub>nown</sub> d<sub>ynam</sub>i<sub>cs</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>comm</sub>itt<sub>e</sub>d <sub>p</sub>rofile. Polic<sub>y</sub> S<sub>p</sub>ace Res<sub>p</sub>onse Oracles [35] instead train a<sub>pp</sub>roxi-<sup>mate</sup> <sup>best</sup> <sup>res</sup>p<sup>onses</sup> <sup>a</sup>g<sup>ainst</sup> <sup>a</sup> <sup>meta-strate</sup>gy <sup>over</sup> <sup>a</sup> p<sup>olic</sup>y p<sup>o</sup>p<sup>ula-</sup> tion.

Rational Verification. Rational verification [58] asks whether a t<sub>empora</sub>l <sub>proper</sub>t<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> <sub>on</sub> <sub>some</sub> <sub>or</sub> <sub>a</sub>ll <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> b<sub>e</sub>h<sub>av</sub>i<sub>ours.</sub> E<sub>q</sub>uilibrium Lo<sub>g</sub>ic ex<sub>p</sub>resses such <sub>p</sub>ro<sub>p</sub>erties throu<sub>g</sub>h modalities [NE] and ⟨NE⟩ over Nash-equilibrium computations. Model checkin<sub>g</sub> is 2EXPTIME-hard in <sub>g</sub>eneral, with matchin<sub>g</sub> u<sub>pp</sub>er bounds f<sub>or res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d f<sub>ragmen</sub>t<sub>s.</sub>

Commitment-aware Lo<sub>g</sub>ics. Strate<sub>gy</sub> Lo<sub>g</sub>ic (sl) ex<sub>p</sub>licitl<sub>y</sub> <sub>q</sub>uantifi<sub>es an</sub>d bi<sub>n</sub>d<sub>s s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es, su</sub>b<sub>sumes</sub> a<sub>tl</sub><sup>⋆</sup><sub>, an</sub>d <sub>can express equ</sub>ilib<sub>-</sub> <sub>r</sub>i<sub>um</sub> th<sub>roug</sub>h <sub>s</sub>t<sub>a</sub>bilit<sub>y aga</sub>i<sub>ns</sub>t d<sub>ev</sub>i<sub>a</sub>ti<sub>ons.</sub> M<sub>o</sub>d<sub>e</sub>l <sub>c</sub>h<sub>ec</sub>ki<sub>ng</sub> f<sub>or</sub> f<sub>u</sub>ll <sub>sl</sub> i<sub>s</sub> d<sub>ec</sub>id<sub>a</sub>bl<sub>e</sub> b<sub>u</sub>t <sub>non-e</sub>l<sub>emen</sub>t<sub>ary.</sub>

E<sub>x</sub>t<sub>ens</sub>i<sub>ons o</sub>f a<sub>tl eva</sub>l<sub>ua</sub>t<sub>e nes</sub>t<sub>e</sub>d f<sub>ormu</sub>l<sub>as un</sub>d<sub>er prev</sub>i<sub>ous</sub>l<sub>y</sub> selected strate<sub>g</sub>ies, includin<sub>g</sub> strate<sub>gy</sub> contexts (atl<sub>sc</sub>) [15], ex<sub>p</sub>licit strate<sub>g</sub>ies (atl<sub>es</sub>) [55], and irrevocable semantics [2].

## <sup>G</sup> <sup>H</sup>yp<sup>er</sup>p<sup>arameters</sup>

## G.1 Trainin<sub>g</sub> Bud<sub>g</sub>ets

T<sub>a</sub>bl<sub>es</sub> 9<sub>–</sub>13 <sub>g</sub>i<sub>ve</sub> th<sub>e se</sub>tti<sub>ngs</sub> f<sub>or a</sub>ll t<sub>as</sub>k<sub>s an</sub>d l<sub>earners, eac</sub>h <sub>us</sub>i<sub>ng</sub> seeds {0, 1, 2, 3, 4}.

Trainin<sub>g</sub> discounts. The en<sub>v</sub>ironment-le<sub>v</sub>el disco<sub>u</sub>nt s<sub>u</sub>mmar<sub>y</sub> i<sub>s</sub> $\gamma = 0$ .99 for all Pursuit-Evasion variants and Matrix Stag H<sub>un</sub>t<sub>, an</sub>d $\gamma = 1$ f<sub>o</sub>r Coins<sub>,</sub> Commons Harvest<sub>, a</sub>nd Territory<sub>.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>o</sub>n <sub>uses u</sub>ndi<sub>scou</sub>nt<sub>e</sub>d <sub>sa</sub>ti<sub>s</sub>f<sub>ac</sub>ti<sub>o</sub>n <sub>p</sub>r<sub>o</sub>b<sub>a</sub>biliti<sub>es.</sub>

<table><tr><td>Environment</td><td>Agents</td><td>Episode horizon</td><td>Initial budget</td></tr><tr><td>PuRsUIT-EvAsION Classic 2</td><td>2</td><td>100</td><td>1,000,192</td></tr><tr><td>PURSUIT-EvASION Capture 2</td><td>2</td><td>100</td><td>1,000,192</td></tr><tr><td>PURsUIT-EvAsION Classic 3</td><td>3</td><td>100</td><td>1,500,160</td></tr><tr><td>PURSUIT-EvAsION Capture 3</td><td>3</td><td>100</td><td>1,500,160</td></tr><tr><td>Coıns Own Coin First</td><td>2</td><td>256</td><td>200,192</td></tr><tr><td>MATRIX STAG HUNT</td><td>2</td><td>20</td><td>500,224</td></tr><tr><td>COMMONS HARVEST Safe Harvest</td><td>4</td><td>1000</td><td>200,192</td></tr><tr><td>TERRITORY Role Claim</td><td>2</td><td>256</td><td>200,192</td></tr></table>

Table 9: E<sub>p</sub>isode horizons and initial trainin<sub>g</sub> bud<sub>g</sub>ets in environment ste<sub>p</sub>s, before the o<sub>p</sub>tional 25% extension.

## G.2 Learner Settin<sub>g</sub>s

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Applicable algorithms</td><td>IPPO, PR2-IPPO, and their CER and handcrafted variants.</td></tr><tr><td>Rollout batch</td><td>256 transitions; environment counts and chunk lengths are in Table 12.</td></tr><tr><td>Policy/value networks</td><td>Two-layer MLPs, hidden widths [64, 64], tanh activations. PR2-IPPO uses a joint-action critic.</td></tr><tr><td>Optimizer</td><td>Adam, learning rate  $3 \times 1 0 ^ { - 4 } ( 1 0 ^ { - 4 }$  for Coins); annealing only for Safe Harvest; gradient-norm clipping 0.5.</td></tr><tr><td>PPO objective</td><td>GAE parameter λ = 1 for Safe Harvest and 0.95 otherwise; clipping parameter 0.2, value-loss coefficient 0.5, entropy coefficient 0.01.</td></tr><tr><td>Update schedule</td><td>4 epochs and 4 minibatches per rollout; 64 transitions per minibatch before CER expansion.</td></tr><tr><td>Coins CER replay</td><td>uncorrected mode.</td></tr></table>

Table 10: Polic<sub>y</sub>-<sub>g</sub>radient h<sub>yp</sub>er<sub>p</sub>arameters for IPPO-famil<sub>y</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>ms.</sub>

Nash-Q convergence. The $1 / N$ <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 11<sub>,</sub> <sub>coun</sub>ti<sub>ng</sub> CER u<sub>p</sub>dates, satisfies Theorem 5(ii) under infinitel<sub>y</sub> man<sub>y</sub> u<sub>p</sub>dates. The theorem also re<sub>q</sub>uires its sam<sub>p</sub>lin<sub>g</sub> and selector assum<sub>p</sub>tions <sub>an</sub>d <sub>concerns</sub> t<sub>empora</sub>l <sub>u</sub>tiliti<sub>es;</sub> th<sub>e</sub> h<sub>an</sub>d<sub>cra</sub>ft<sub>e</sub>d b<sub>ase</sub>li<sub>ne uses na</sub>ti<sub>ve</sub> sta<sub>g</sub>e rewar<sup>d</sup>s.

<table><tr><td>Setting</td><td>IQL /PR2-IQL</td><td>Nash-Q</td></tr><tr><td>Applicable variants</td><td>IQL, PR2-IQL, and their CER and handcrafted variants.</td><td>Nash-Q, Nash-Q-CER, hc-Nash-Q (Matrix Stag Hunt).</td></tr><tr><td>Function class</td><td>Recurrent Q-network, hidden size 64, tanh activation; joint-action critic for PR2-IQL.</td><td>Tabular joint-action Q-values indexed by joint environment and monitor observations and elapsed time.</td></tr><tr><td>Optimizer / step size</td><td>Adam, learning rate  $3 \times 1 0 ^ { - 4 }$  , gradient-norm clipping 0.5; 4 Visit-count step size epochs per replay batch.</td><td> $\alpha ( x , \pmb { a } ) = 1 / N ( x , \pmb { a } )$ </td></tr><tr><td>Target network</td><td>Update interval 25 trainer updates; target mixing τ = 1.0.</td><td></td></tr><tr><td>Replay and exploration</td><td>Buffer capacity 25,000 chunks; learning starts after 1024 steps; linear ε-greedy decay from 1 to 0.05 over the first half of training.</td><td>Linear ε-greedy decay from 1 to 0.05 over the first half of training.</td></tr></table>

Table 11: Value-learning hyperparameters. Nash-Q settings apply to Matrix Stag Hunt.

<table><tr><td>Benchmark</td><td>Parallel envs.</td><td>Chunk length</td><td>IQL replay batch</td></tr><tr><td>STOCHASTIC</td><td>4</td><td>64</td><td>16 chunks</td></tr><tr><td>PURSUIT-EVASION</td><td></td><td></td><td></td></tr><tr><td>MATRIX STAG HUNT</td><td>8</td><td>32</td><td>32 chunks</td></tr><tr><td>COINS COMMONS HARVEST</td><td>1 1</td><td>256 256</td><td>4 chunks</td></tr><tr><td>TERRITORY</td><td>1</td><td>256</td><td>4 chunks</td></tr><tr><td></td><td></td><td></td><td>4 chunks</td></tr></table>

Table 12: Rollout settings for all learners and IQL replay batches (1024 transitions each).

## G<sub>.</sub>3 E<sub>va</sub>l<sub>ua</sub>ti<sub>o</sub>n S<sub>e</sub>ttin<sub>gs</sub>

<table><tr><td>Benchmark</td><td>NashConv episodes</td><td>BR train iters</td><td>BR batch steps</td></tr><tr><td>PURSUIT-EVASION (2-player)</td><td>64</td><td>6</td><td>1024</td></tr><tr><td>PURSUIT-EVASION (3-player)</td><td>96</td><td>8</td><td>1024</td></tr><tr><td>MATRIX STAG HUNT</td><td>50</td><td>4</td><td>512</td></tr><tr><td>CoINs</td><td>32</td><td>3</td><td>512</td></tr><tr><td>COMMONS HARVEST</td><td>32</td><td>3</td><td>512</td></tr><tr><td>TERRITORY</td><td>32</td><td>3</td><td>512</td></tr></table>

Table 13: Unilateral res<sub>p</sub>onse-search budgets for NashConv<sup>≈</sup>. Scores above 0.05 trigger u<sub>p</sub> to 25% additional <sub>p</sub>rofile training.