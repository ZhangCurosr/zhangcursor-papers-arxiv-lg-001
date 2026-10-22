# MHeatmap: Two-Walk Laplacian Reordering and Mosaic Layout for Rectangular Matrix Visualization

Juntang Wang Harvard University Cambridge, USA juntangwang@g.harvard.edu

Yihan Wang Duke University Durham, USA yihan.wang@dukekunshan.edu.cn

Man Zhang   
Westlake University   
Hangzhou, China   
man.zhang@westlake.edu.cn

Sheng’en Li Duke Kunshan University Kunshan, China

Siyuan Cao Duke Kunshan University Kunshan, China siyuan.cao@dukekunshan.edu.cn

Shixin Xu   
Duke Kunshan University   
Kunshan, China   
shixin.xu@duke.edu

Liming Zhu Duke Kunshan University Kunshan, China liming.zhu@dukekunshan.edu.cn

Hao Wu   
Duke Kunshan University   
Kunshan, China   
hw412@duke.edu   
Dongmian Zou<sup>∗</sup>   
Duke Kunshan University   
Kunshan, China   
dongmian.zou@duke.edu

## Ab<sub>s</sub>t<sub>rac</sub>t

Weighted rectangular matrices record relations between two types of objects, such as occupations and industries, documents and terms, origins and destinations, or annotated classes and computed clusters. Their heatmaps become dificult to read when arbitrary row and column orders scatter related objects, while equal size cells hide large diferences in row and column totals. We present MHeatmap (Mosaic Heatmap), which addresses these problems in two stages. Two-Walk spectral seriation first represents the matrix as a weighted bipartite graph. Ordinary bipartite ordering uses only observed row to column links. Two-Walk also uses lengthtwo walks, so rows that share columns and columns that share rows receive explicit similarity links. Sorting a Fiedler vector of the resulting graph Laplacian gives coordinated row and column orders. The Mosaic layout then partitions a unit square so that row height is proportional to the row total and column width is proportional to the column total, while color represents the observed entry. We prove how the Two-Walk energy combines direct alignment with shared-neighbor smoothness, derive an exact matrix vector product that permits sparse computation, and formalize the tiling and area semantics of Mosaic. Across 300 synthetic matrices and seven real data sets, Two-Walk obtains the lowest normalized 2-SUM in all 15 synthetic regimes and the highest Band@10% in 14. The largest gains occur when shared neighborhoods express the latent relational structure.

## Ke<sub>y</sub>words

matrix visualization, matrix reordering, spectral reordering, mosaic heatmap, visual pattern recovery, bipartite graphs, marginal distribution

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

A weighted rectangular matrix $\pmb { { B } } \in \mathbb { R } _ { > 0 } ^ { m \times n }$ records relations between two diferent entity types. Rows represent one type, columns represent the other, and $B _ { i j }$ gives the nonnegative strength or frequency of their relation. Examples include education by occupation crosswalks, author by topic associations, document by term data, home by work commuting flows, and cell type by cluster contingency tables. Analysts inspect these matrices to identify correspondences, overlapping groups, split and merge relations, and mass that leaks outside expected blocks.

The input order is often alphabetical, administrative, or otherwise unrelated to relational structure. We use the term rectangular matrix seriation for the task of permuting rows and columns so that related entities occupy nearby positions while both axes remain separately interpretable. A useful seriation can turn a scattered view into blocks or bands that are easier to inspect. Figure 1 illustrates this efect on an OpenAlex author by topic matrix.

A natural representation of � is a weighted bipartite graph, which is a graph with one vertex set for rows, one vertex set for columns, and weighted edges only between the two sets. A standard spectral method places both sets in one symmetric adjacency matrix, forms the graph Laplacian � = � − �, and sorts a Fiedler vector. The graph Laplacian is the degree matrix minus the adjacency matrix. A Fiedler vector is an eigenvector associated with its smallest positive eigenvalue, and its entries provide a one dimensional coordinate for ordering. We call this direct construction One-Walk because its adjacency explicitly contains only direct row to column edges.

Direct edges do not explicitly state that two rows may connect to similar columns or that two columns may receive mass from similar rows. A length-two walk follows two consecutive edges. In a bipartite graph, a length-two walk starts and ends on the same side, either row to column to row or column to row to column. These walks make shared neighborhoods explicit. We introduce Two-Walk spectral seriation, which combines direct relations with the row similarity ��<sup>⊤</sup> and the column similarity �<sup>⊤</sup>� in one Laplacian. The resulting Fiedler vector gives coordinated row and column coordinates.

The second stage is the Mosaic layout, a proportional heatmap that partitions a unit square into adjacent rectangles. A row marginal is a row sum, and a column marginal is a column sum. Mosaic assigns row height in proportion to the row marginal and column width in proportion to the column marginal, while color encodes the observed entry. The rectangles cover the display as a complete tiling, which motivates the name Mosaic. For a contingency table, the area of a cell is the product of its row and column proportions. This is also the expected cell proportion when the two variables are independent. Geometry therefore shows the marginal baseline, while color shows the observed relation. This construction belongs to the historical family ofmosaic and sieve displays, which Section 2 discusses in detail.

![](images/a36da76c902982912be1adfcb0b09d0eebe76fb5a8ea6161a21c5374e148834c.jpg)  
Fi ure 1: The MHeatmap workflow on an O enAlex author b to ic matrix. Panel (a) uses the in ut order and e ual size cells. Panel (b) applies Two-Walk seriation and keeps equal size cells, which isolates the ordering efect. Panel (c) applies the Mosaic layout to the input order, which isolates the geometry efect. Panel (d) combines Two-Walk seriation and Mosaic rendering.

The main contribution of this paper is the seriation method. The Mosaic stage is part of the complete MHeatmap workflow and is defined precisely so that its visual semantics are reproducible. Our contributions are:

• We define Two-Walk spectral seriation for nonnegative rectangular matrices. It combines direct cross-partition relations with same-side afinities induced by shared neighbors.

• We provide a Laplacian energy decomposition, a scale invariance result, a limit relation to One-Walk, and an exact sparse matrix vector product that avoids explicit Gram matrices.

• We evaluate the method on 300 controlled synthetic matrices and seven real relation matrices. The One-Walk comparison isolates the contribution of the length-two terms.

• We define the Mosaic layout used by MHeatmap and prove its tiling and area semantics. We also connect it to established mosaic and sieve displays.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2<sub>.</sub>1 M<sub>a</sub>t<sub>r</sub>i<sub>x</sub> S<sub>er</sub>i<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d R<sub>eor</sub>d<sub>er</sub>i<sub>ng</sub>

Seriation arranges objects along an axis so that structurally related objects become nearby [15]. Spectral seriation sorts a Fiedler vector and can be viewed as a relaxation of quadratic arrangement objectives [1, 3, 9, 10]. Other approaches include hierarchical clustering with optimal leaf ordering [2], bandwidth and profile reduction for sparse matrices [21], and alternating procedures motivated by weighted bipartite layouts [6]. Recent visualization work evaluates whether an ordering reveals blocks, bands, and related motifs rather than judging only one numerical objective [4, 27, 29]. Most classical formulations assume a square similarity matrix or order the two axes independently. We study coordinated row and column orders for a rectangular relation matrix.

## 2.2 Bi<sub>p</sub>artite S<sub>p</sub>ectral Methods and Shared N<sub>e</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d<sub>s</sub>

Spectral co-clustering represents a rectangular matrix as a bipartite graph and uses coupled row and column coordinates to obtain discrete groups [8]. Correspondence analysis and singular value methods also provide paired coordinates [12]. Our task is seriation, which returns continuous orders rather than only clusters. The plain bipartite adjacency contains direct cross-partition edges but no explicit same-side edges.

A shared-neighbor relation is a similarity induced by common adjacent vertices. Such relations are standard in graph similarity and link analysis [5, 14]. For a bipartite adjacency �, the square $M ^ { 2 }$ records length-two walks and is block diagonal. It therefore captures row similarity and column similarity but separates the two vertex sets. Two-Walk uses $M ^ { 2 } + \alpha M$ so that shared neighborhoods and direct relations contribute to one spectral coordinate. The contribution is this joint seriation formulation and its analysis, not the general observation that graph walks encode proximity.

## 2<sub>.</sub>3 M<sub>osa</sub>i<sub>c,</sub> Si<sub>eve, an</sub>d P<sub>ro</sub>d<sub>uc</sub>t Pl<sub>o</sub>t<sub>s</sub>

The term mosaic has an established history in statistical graphics. Hartigan and Kleiner used it for a collection of rectangles whose areas represent contingency table counts [13]. Friendly developed classical mosaic displays for multiway tables [11]. A related parquet or sieve diagram sets row heights and column widths from the corresponding marginals, so each rectangle has area proportional to the expected count under independence [17, 22]. Product plots provide a general framework in which area is the product of width and height and include mosaic, treemap, and equal area displays [28].

This lineage explains the name Mosaic in MHeatmap. Its geometry follows the marginal partition of a sieve diagram, while its color channel follows a heatmap by encoding the observed entry. The result is related to, but not identical with, a classical mosaic display. The general idea of a proportional rectangular tiling is established prior work. Our method-specific contribution is the precise combination of marginal geometry and heatmap color inside the MHeatmap workflow. Marginal bars provide another direct way to show row and column totals. Equal area heatmaps, marginal bars, and Mosaic support diferent reading tasks. Since area judgments are less precise than aligned position and length judgments [7], our claims for Mosaic concern its encoding semantics and qualitative use, not universal perceptual superiority.

## 3 M<sub>e</sub>th<sub>o</sub>d

## 3<sub>.</sub>1 P<sub>re</sub>li<sub>m</sub>i<sub>nar</sub>i<sub>es an</sub>d O<sub>ne-</sub>W<sub>a</sub>lk S<sub>er</sub>i<sub>a</sub>ti<sub>on</sub> Let $\pmb { { B } } \in \mathbb { R } _ { \geq 0 } ^ { m \times n }$ be nonzero.

Definition 1 (Active matrix). The active matrix of � is obtained by deleting every row and column whose marginal is zero. These deleted indices have no incident graph edge and therefore no informative spectral coordinate.

We state the spectral results for an active matrix whose bipartite support graph is connected. The support graph keeps an edge exactly where the corresponding matrix entry is positive. Section 3.3 gives a deterministic extension to disconnected support.

For any symmetric nonnegative adjacency �, let $D = \mathrm { d i a g } ( A 1 )$ and $L = D - A$ . When the graph is connected, a unit Fiedler vector $f$ satisfies the variational characterization

$$
\boldsymbol { f } \in \arg \operatorname* { m i n } _ { \boldsymbol { z } ^ { \top } 1 = 0 , \ \| \boldsymbol { z } \| _ { 2 } = 1 } \boldsymbol { z } ^ { \top } \boldsymbol { L } \boldsymbol { z } .\tag{1}
$$

Thus strongly connected vertices are encouraged to receive nearby coordinates.

Definition 2 (Rectangular matrix seriation). A rectangular seriation of � is a pair of permutations $( \pi _ { R } , \pi _ { C } )$ of its row and column indices. The permutations are coordinated when both are derived from one model of the row and column relations.

Our implementation normalizes the active matrix by its largest entry. We therefore define

$$
\bar { B } = \frac { B } { s ( B ) } , \qquad s ( B ) = \operatorname* { m a x } _ { i , j } B _ { i j } > 0 .\tag{2}
$$

This normalization is necessary because the two-walk term is quadratic in � and the direct term is linear.

Definition 3 (Bipartite lift and One-Walk seriation). The bipartite lift of �<sup>¯</sup> is

$$
M = \left[ \begin{array} { l l l } { { { \bf 0 } } } & { { { \bar { \cal B } } } } \\ { { { \bar { \cal B } } ^ { \top } } } & { { { \bf 0 } } } \end{array} \right] .\tag{3}
$$

Let $\scriptstyle L _ { \mathrm { o w } }$ be the Laplacian of �. Split a Fiedler vector as $[ \pmb { x } ^ { \top } , \pmb { y } ^ { \top } ] ^ { \top }$ where $\pmb { x } \in \mathbb { R } ^ { m }$ and $\pmb { y } \in \mathbb { R } ^ { n }$ . Sorting � and � separately defines One-Walk row and column orders.

Proposition 1 (Scale invariance). For any scalar $\gamma > 0 ,$ rep<sup>l</sup>ac<sup>i</sup>ng � by �� leaves �<sup>¯</sup>, the One-Walk operator, and every Two-Walk <sub>opera</sub>t<sub>or</sub> <sub>w</sub>ith th<sub>e</sub> <sub>same</sub> <sub>�</sub> <sub>unc</sub>h<sub>ange</sub>d<sub>.</sub>

Proof. The maximum entry scales as $s ( \gamma B ) = \gamma s ( B )$ . Hence $( \gamma B ) / s ( \gamma B ) = B / s ( B )$ . □

## 3<sub>.</sub>2 T<sub>wo</sub>-W<sub>a</sub>lk S<sub>pec</sub>tr<sub>a</sub>l S<sub>e</sub>ri<sub>a</sub>ti<sub>o</sub>n

A length-two walk in the bipartite lift returns to its starting partition. Squaring Eq. (3) gives

$$
M ^ { 2 } = \left[ \begin{array} { c c } { { \bar { B } \bar { B } ^ { \top } } } & { { 0 } } \\ { { { \bf 0 } } } & { { \bar { B } ^ { \top } \bar { B } } } \end{array} \right] .\tag{4}
$$

The first diagonal block measures row similarity through shared columns. The second measures column similarity through shared rows. Since $M ^ { 2 }$ alone disconnects the two partitions, direct edges are retained.

Definition 4 (Two-Walk spectral seriation). For $\alpha > 0 _ { : }$ , define

$$
A _ { \alpha } = M ^ { 2 } + \alpha M = \left[ \bar { B } \bar { B } ^ { \top } ~ \alpha \bar { B } ~ \right] .\tag{5}
$$

Let $L _ { \alpha }$ be its graph Laplacian. Split a Fiedler vector of $L _ { \alpha }$ into row coordinates � and column coordinates �. Sorting the two coordinate vectors separately defines the Two-Walk seriation.

Proposition 2 (Connectivity). If the bipartite support graph of the <sub>ac</sub>ti<sub>ve</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> i<sub>s</sub> <sub>connec</sub>t<sub>e</sub>d <sub>an</sub>d $\alpha > 0$ , t<sup>h</sup>en t<sup>h</sup>e grap<sup>h</sup> wit<sup>h</sup> a<sup>d</sup>jacency $A _ { \alpha }$ is connected. Consequently, the nullspace of $\mathbf { \nabla } \cdot \mathbf { L } _ { \alpha }$ is span{1}, and it<sub>s</sub> Fi<sub>e</sub>dl<sub>er e</sub>i<sub>genva</sub>l<sub>ue</sub> i<sub>s</sub> th<sub>e sma</sub>ll<sub>es</sub>t <sub>pos</sub>iti<sub>ve e</sub>i<sub>genva</sub>l<sub>ue.</sub>

Proof. Every positive edge of the bipartite lift � remains a positive edge of $A _ { \alpha }$ with weight multiplied by �. Hence the graph of $A _ { \alpha }$ contains the connected support graph of � as a spanning subgraph. The Laplacian of a connected undirected graph has a one dimensional nullspace spanned by 1. □

The diagonal entries of the two Gram matrices are self loops. They cancel in the combinatorial Laplacian and therefore do not alter the energy below.

Theorem 1 (Two-Walk energy decomposition). For $\boldsymbol { z } = [ \mathbf { \boldsymbol { x } } ^ { \top } , \mathbf { \boldsymbol { y } } ^ { \top } ] ^ { \top }$

$$
\begin{array} { l } { \displaystyle z ^ { \top } L _ { \alpha } z = \frac { 1 } { 2 } \sum _ { i , k } ( \bar { B } \bar { B } ^ { \top } ) _ { i k } ( x _ { i } - x _ { k } ) ^ { 2 } } \\ { \displaystyle ~ + \frac { 1 } { 2 } \sum _ { j , \ell } ( \bar { B } ^ { \top } \bar { B } ) _ { j \ell } ( y _ { j } - y _ { \ell } ) ^ { 2 } } \\ { \displaystyle ~ + \alpha \sum _ { i , j } \bar { B } _ { i j } ( x _ { i } - y _ { j } ) ^ { 2 } . } \end{array}\tag{6}
$$

Proof. For a symmetric adjacency �, the identity $z ^ { \top } ( D - A ) z =$ $\begin{array} { r } { { \frac { 1 } { 2 } } \sum _ { p , q } A _ { p q } ( z _ { \mathstrut p } - z _ { q } ) ^ { 2 } } \end{array}$ follows by expansion. Applying it to the four blocks of Eq. (5) gives the two same-side sums. The two symmetric cross blocks together give the final sum. □

The first term makes rows with similar column neighborhoods smooth in the spectral coordinate. The second does the same for columns. The third aligns directly related rows and columns. Thus � controls the relative emphasis on direct alignment and sharedneighbor smoothness.

Proposition 3 (Relation to One-Walk). Let $L ( M )$ <sub>an</sub>d $L ( M ^ { 2 } )$ d<sub>eno</sub>t<sub>e</sub> the Laplacians of� and $M ^ { 2 }$ <sub>.</sub> Th<sub>en</sub>

$$
L _ { \alpha } = L ( M ^ { 2 } ) + \alpha L ( M ) , \qquad \alpha ^ { - 1 } L _ { \alpha } \longrightarrow L ( M )\tag{7}
$$

as $\alpha \to \infty$ . If the Fiedler eigenvalue of �(�) is simple, the normalized Two-Walk Fiedler vector conver<sub>g</sub>es to the One-Walk Fiedler vector up to sign. Ifits row and column coordinates are distinct, the induced permutations also agree for suficiently large �.

Proof. Laplacian construction is linear in the adjacency, which gives the first identity. Dividing by � yields $L ( M ) + \alpha ^ { - 1 } L ( M ^ { 2 } )$ The second term converges to zero in operator norm. Standard perturbation theory for a simple eigenvalue gives convergence of the eigenvector up to sign. Distinct coordinates preserve their sorted order under suficiently small perturbations. □

## 3.3 Exact Matrix Vector Product

Explicitly forming $\bar { B } \bar { B } ^ { \top }$ and $\bar { B } ^ { \top } \bar { B }$ can destroy sparsity. The following identity permits a sparse eigensolver without constructing either matrix.

Lemma 1 (Matrix-free Two-Walk product). For $\boldsymbol { z } = [ \boldsymbol { x } ^ { \top } , \boldsymbol { y } ^ { \top } ] ^ { \top }$ define $\pmb { u } = \bar { \pmb { B } }$ <sup>⊤</sup>� and $\pmb { \upsilon } = \bar { \mathbfcal B } \pmb { y }$ <sub>.</sub> Th<sub>en</sub>

$$
\begin{array} { r } { A _ { \alpha } z = \left[ \begin{array} { c } { \bar { B } u + \alpha v } \\ { \bar { B } ^ { \top } v + \alpha u } \end{array} \right] . } \end{array}\tag{8}
$$

$I f r = { \bar { B } } 1$ <sub>an</sub>d $\begin{array} { r } { \pmb { c } = \bar { \pmb { B } } ^ { \top } \pmb { 1 } } \end{array}$ <sub>,</sub> th<sub>e</sub> d<sub>egree</sub> <sub>vec</sub>t<sub>or</sub> i<sub>s</sub>

$$
\begin{array} { r } { \pmb { d } _ { \alpha } = \left[ \frac { \bar { B } c + \alpha r } { \bar { B } ^ { \top } r + \alpha c } \right] , \qquad \pmb { L } _ { \alpha } z = \pmb { d } _ { \alpha } \odot z - \pmb { A } _ { \alpha } z . } \end{array}\tag{9}
$$

Proof. Multiply the block matrix in Eq. (5) by $[ \pmb { x } ^ { \top } , \pmb { y } ^ { \top } ] ^ { \top }$ and associate the products as $\bar { B } ( \bar { B } ^ { \top } x )$ and $\bar { B } ^ { \top } ( \bar { B } y )$ . Setting $z = 1$ gives the degree vector. □

After precomputing the degree vector, one application of $L _ { \alpha }$ uses four sparse products with �<sup>¯</sup> or $\bar { B } ^ { \top }$ . Its arithmetic cost is $O ( \mathrm { n n z } ( B ) )$ ) and its storage is $O ( { \mathrm { n n z } } ( B ) + m + n )$ apart from eigensolver workspace. A direct implementation forms the Gram matrices explicitly; Eq. (9) specifies an exact sparse replacement rather than an approximation.

A Fiedler vector is defined only up to sign, so reversing both axis orders is equivalent. For deterministic output, we orient the vector so that its first coordinate of maximum absolute value is positive, then apply a stable sort. If the active support has several connected components, we order each component independently, arrange components by decreasing total mass, and append zeromarginal rows and columns in their original order. A repeated Fiedler eigenvalue can still make the order nonunique, which is a limitation of spectral seriation rather than a numerical bug.

## 3.4 Mosaic La<sub>y</sub>out

Let � be the reordered nonnegative matrix. Let

$$
r _ { i } = \sum _ { j } A _ { i j } , \qquad c _ { j } = \sum _ { i } A _ { i j } , \qquad T = \sum _ { i , j } A _ { i j } > 0 .\tag{10}
$$

Assume positive active marginals, and define normalized marginals $R _ { i } = r _ { i } / T$ and $C _ { j } = { c _ { j } } / { T }$

Definition 5 (Mosaic layout). Set cumulative boundaries

$$
Y _ { 0 } = 0 , \quad Y _ { i } = \sum _ { k = 1 } ^ { i } R _ { k } , \qquad X _ { 0 } = 0 , \quad X _ { j } = \sum _ { \ell = 1 } ^ { j } C _ { \ell } .\tag{11}
$$

In the Mosaic layout, cell $( i , j )$ occupies

$$
\mathscr { R } _ { i j } = [ X _ { j - 1 } , X _ { j } ] \times [ Y _ { i - 1 } , Y _ { i } ] ,\tag{12}
$$

and its fill color is a monotone mapping $g ( A _ { i j } )$ of the observed entry.

Proposition 4 (Tiling and area semantics). The rectangles in Definition 5 cover t<sup>h</sup>e unit square an<sup>d h</sup>ave <sup>d</sup>isjoint interiors. Every ce<sup>ll</sup> i<sub>n row</sub> � h<sub>as</sub> h<sub>e</sub>i<sub>g</sub>ht $R _ { i } ,$ every ce<sup>ll</sup> in co<sup>l</sup>umn � <sup>h</sup>as wi<sup>d</sup>t<sup>h</sup> $C _ { j } ,$ <sub>, an</sub>d

$$
\mathrm { a r e a } ( \mathcal { R } _ { i j } ) = R _ { i } C _ { j } = \frac { r _ { i } c _ { j } } { T ^ { 2 } } .\tag{13}
$$

If � is a contingency table, its expected count under row and column i<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> i<sub>s</sub> $\begin{array} { r } { E _ { i j } = r _ { i } c _ { j } / T _ { : } } \end{array}$ <sub>, so</sub> th<sub>e ce</sub>ll <sub>area equa</sub>l<sub>s</sub> $E _ { i j } / T$

Proof. The normalized marginals satisfy $\begin{array} { r } { \sum _ { i } R _ { i } = \sum _ { j } C _ { j } = 1 } \end{array}$ Their cumulative sums partition each axis into adjacent intervals. Taking all Cartesian products therefore partitions the unit square. The height, width, and area follow directly from the interval lengths. The independence statement follows from the standard expected count formula. □

This proposition separates the visual channels. Geometry encodes the two marginals and therefore the independence baseline. Color encodes the observed cell value. Cell area does not encode $A _ { i j }$ itself. The mathematical layout has no interior gaps. A white region in a rendered figure comes from the color mapping of a zero or small entry, or from optional divider lines. Rare rows or columns can become visually narrow, so Mosaic is intended for overview and mass context rather than precise cell value comparison.

Definition 6 (MHeatmap workflow). MHeatmap first applies Two-Walk seriation to obtain $( \pi _ { R } , \pi _ { C } )$ . It then renders the reordered matrix either as an equal area heatmap or with the Mosaic layout in Definition 5.

## 4 E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub>

## 4.1 Evaluation Questions and Scope

We evaluate the two stages of MHeatmap with evidence matched to their claims. The ordering stage is evaluated quantitatively because it produces explicit row and column permutations. The Mosaic stage is evaluated through controlled visual comparisons because it changes the representation of the same reordered matrix rather than the underlying order.

The evaluation addresses four questions. First, does Two-Walk recover rectangular block and band structure more consistently than representative ordering baselines? Second, does it improve over One-Walk, which uses the same spectral pipeline but ex cludes the length-two terms? Third, on which matrix semantics do shared-neighbor afinities help, and where do they provide little benefit? Fourth, when the row and column order is held fixed, what additional information is made visible by Mosaic geometry?

## 4<sub>.</sub>2 C<sub>on</sub>t<sub>ro</sub>ll<sub>e</sub>d S<sub>yn</sub>th<sub>e</sub>ti<sub>c</sub> B<sub>enc</sub>h<sub>mar</sub>k

The synthetic component contains five structural families at three matrix sizes, with 20 independent random seeds for every family and size combination. This gives

$$
5 \mathrm { f a m i l i e s } \times 3 \mathrm { s i z e s } \times 2 0 \mathrm { s e e d s } = 3 0 0
$$

matrices. The sizes are $1 0 0 \times 9 0 , 2 0 0 \times 1 8 0$ , and $4 0 0 \times 3 6 0$ . All families are generated from a common rectangular block template, after which the rows and columns are randomly permuted before any method is applied. The algorithms receive only the permuted weighted matrix and do not receive block labels.

Family A is a clean one-to-one block control. Each principal row group corresponds to one principal column group, so a direct row to column signal is already strong. Family B adds paired subgroup overlap, which weakens one-to-one separation while preserving shared-neighbor structure. Family C introduces shared super-prototypes, so several subgroups inherit related column profiles and form nested organization. Family D adds noise to the shared-prototype setting. Family E introduces cross-block leakage, so relation mass can spill outside the intended diagonal blocks. The sequence therefore separates an easy direct-link regime from regimes in which shared neighborhoods, noise, and leakage become increasingly important.

## 4<sub>.</sub>3 R<sub>ea</sub>l M<sub>a</sub>t<sub>r</sub>i<sub>x</sub> B<sub>enc</sub>h<sub>mar</sub>k

The real component contains seven nonnegative rectangular matrices with diferent semantics and aspect ratios. Two oficial crosswalks, SIC to NAICS and CIP to SOC, test many-to-many mappings between classification systems [19, 25]. An ACS occupation by industry contingency table tests a small aggregate relation matrix [26]. LODES home county by work county flows test a matrix whose organization is partly geographic [24]. A 20 Newsgroups term by document matrix tests text relations [18]. An MBTA route by sta tion matrix tests route incidence and local network structure [16]. An OpenAlex author by topic matrix tests specialization across distinct entity types [20].

Dataset construction follows fixed external rules rather than a search for submatrices on which Two-Walk performs well. Complete oficial crosswalks are used when available. Geographic, text, transit, and bibliographic subsets are fixed before ordering. Only structurally empty rows and columns are removed. Table 1 reports the final matrix dimensions.

## 4<sub>.</sub>4 C<sub>ompare</sub>d M<sub>e</sub>th<sub>o</sub>d<sub>s</sub>

We compare seven orderings that represent complementary design choices.

Original (O).. This baseline keeps the supplied row and column orders. It measures how much structure is visible without algorithmic intervention.

Marginal sorting (M).. Rows and columns are sorted independently by decreasing row and column sums. This exposes degree or frequency imbalance but does not use relation profiles.

HC with optimal leaf ordering (HC).. Hierarchical clustering is applied separately to row profiles and column profiles, followed by optimal leaf ordering [2]. It represents profile-based clustering that does not construct a joint row and column spectral coordinate.

One-Walk (OW).. One-Walk applies the same Laplacian and Fiedler ordering pipeline as Two-Walk to the ordinary bipartite lift in Eq. (3). It excludes $\bar { B } \bar { B } ^ { \top }$ and $\bar { B } ^ { \top } \bar { B } .$ . This is the central ablation because the diference between OW and TW is exactly the lengthtwo augmentation.

Correspondence analysis SVD (CA).. Correspondence analysis obtains coupled row and column coordinates from a marginally normalized matrix and orders each axis by its first nontrivial coordinate [12]. It represents a factorization-based alternative.

Alternating median (MD).. Alternating median updates row positions from the weighted median positions of adjacent columns and then updates columns from adjacent rows [6]. It represents a local refinement method motivated by weighted bipartite layout.

All methods receive the same nonnegative matrix after removal of structurally empty rows and columns. For spectral methods, reversing the complete coordinate vector gives an equivalent ordering. The reported distance based metrics are invariant when both axes are reversed together.

## 4.5 Structure-Recover<sub>y</sub> Metrics

A rectangular matrix has diferent numbers of row and column positions, so raw integer ranks cannot be compared directly. For row permutation $\pi _ { R }$ and column permutation $\pi c ,$ we map both axes to the same unit interval:

$$
\rho _ { i } = \frac { \pi _ { R } ( i ) - 1 } { \operatorname* { m a x } ( m - 1 , 1 ) } , \qquad \kappa _ { j } = \frac { \pi _ { C } ( j ) - 1 } { \operatorname* { m a x } ( n - 1 , 1 ) } .\tag{14}
$$

Let $\begin{array} { r } { T _ { B } = \sum _ { i , j } B _ { i j } > 0 . } \end{array}$

Normalized rectangular 2-SUM (R2S).. We measure weighted diagonal dispersion by

$$
\mathrm { R } 2 \mathrm { S } ( \pi _ { R } , \pi _ { C } ) = \frac { 1 } { T _ { B } } \sum _ { i , j } B _ { i j } ( \rho _ { i } - \kappa _ { j } ) ^ { 2 } .\tag{15}
$$

R2S lies in [0, 1]. A lower value means that more relation mass is placed near corresponding normalized positions on the two axes. This is the rectangular form of a weighted quadratic arrangement objective. Table 1 multiplies R2S by $1 0 ^ { 2 }$ for readability.

Band concentration. For a normalized half width $\tau ,$ we define

$$
\operatorname { B a n d } _ { \tau } ( \pi _ { R } , \pi _ { C } ) = \frac { 1 } { T _ { B } } \sum _ { i , j } B _ { i j } \mathbb { I } \big \{ | \rho _ { i } - \kappa _ { j } | \leq \tau \big \} .\tag{16}
$$

We report $\tau = 0 . 1 0$ . Higher values indicate that a larger fraction of mass lies in a narrow diagonal band. R2S penalizes distance continuously, whereas Band@10% applies a fixed visual tolerance. The two measures can therefore disagree when a method concentrates most mass tightly but leaves a small number of distant entries.

MWB-AUC.. MWB-AUC summarizes block separability across a sweep of diagonal block cuts. For each cut scale, the benchmark computes the fraction of total mass retained within matched contiguous diagonal blocks. The area under this curve is larger when diagonal blocks remain well separated over a broad range of cut scales. Unlike R2S and Band@10%, MWB-AUC is also used internally to choose � for adaptive Two-Walk.

## 4.6 Parameter Selection and Re<sub>p</sub>ortin<sub>g</sub> Protocol Adaptive Two-Walk evaluates

$$
\alpha \in \{ 1 , 2 , 4 , 6 , 8 , 1 2 \}
$$

for each input matrix and selects the value with the largest internal MWB-AUC. The selection is unsupervised and does not use synthetic block labels or external task labels. Nevertheless, it gives Two-Walk one data-dependent choice that the fixed baselines do not receive. It also makes MWB-AUC a selection-aligned measure.

We retain MWB-AUC in the complete table for transparency, but base the main comparative claim on R2S and Band@10%, neither of which is used to choose �.

For each synthetic family and size, Table 1 reports the arithmetic mean over 20 seeds. Each real matrix is evaluated once at its full reported size. Boldface marks the best method within one row and metric, and underlining marks the runner-up, including ties. The table reports every family and size rather than only an aggregate so that readers can inspect dependence on matrix scale and motif dificulty.

The Mosaic evaluation uses four Tabula Sapiens cell type by cluster contingency tables [23]. For every tissue, the equal-area Two-Walk panel and the Mosaic panel use identical row and column permutations and the same matrix values. The only change is the geometry from equal row heights and column widths to the marginal proportions in Definition 5. This design isolates what can be attributed to Mosaic. The qualitative analysis is intended to establish interpretability of the encoding, not universal perceptual superiority over equal-area heatmaps or marginal bar charts.

## 5 R<sub>esu</sub>lt<sub>s</sub>

Table 1 is the complete quantitative result. The synthetic rows retain all 15 family and size regimes, and the real rows retain all seven datasets. The table also keeps MWB-AUC visible so that the efect of the adaptive selector can be audited rather than hidden.

## 5.1 S<sub>y</sub>nthetic Structure Reco<sub>v</sub>er<sub>y</sub>

Across the synthetic component, Two-Walk obtains the lowest R2S in all 15 family and size regimes. It obtains the highest Band@10% in 14 of 15 regimes and the highest or tied highest MWB-AUC in all 15. The MWB-AUC result is expected to favor Two-Walk because the same score selects �. More importantly, the improvement also appears on the two metrics that are not used for selection.

Averaged over the 15 regimes, the reported 10<sup>2</sup>×R2S decreases from 1.150 for One-Walk to 0.983 for Two-Walk, a relative re duction of approximately 14.5%. Mean Band@10% increases from 0.718 to 0.765, an absolute increase of 0.047. Since One-Walk and Two-Walk share the same normalization, Laplacian construction, eigensolver concept, coordinate split, and sorting rule, this comparison isolates the contribution of the length-two terms.

Family A, clean one-to-one blocks. Direct row to column relations already give One-Walk a strong signal. Two-Walk still improves R2S at all three sizes, from 0.52, 0.55, and 0.60 to 0.43, 0.42, and 0.51, respectively. Band@10% also rises in every size. These mod est gains are consistent with the method design: shared-neighbor afinities reinforce an already separable pattern but are not essential to recover it.

Families B and C, overlapping and nested structure. The benefit is clearer when entities on the same side share relation profiles. In Family B, Two-Walk improves R2S over One-Walk in all sizes, with the largest change at 200 × 180, from 1.15 to 0.82. Band@10% rises from 0.67 to 0.78 in the same regime. Family C shows the same pattern across shared super-prototypes. The results support the mechanism in Theorem 1: the Gram terms add direct smoothness between rows with shared columns and between columns with shared rows.

Family D, shared prototypes with noise. Noise weakens individual entries but does not remove the neighborhood pattern. Two-Walk remains best on both independent metrics at all three sizes. For the largest matrix, R2S decreases from 1.28 for One-Walk to 1.00, and Band@10% increases from 0.63 to 0.70. This suggests that aggregating two-step evidence can preserve block organization when single edges are less reliable.

Family E, cross-block leakage. This family gives the most restrictive synthetic case. Two-Walk retains the lowest R2S at all sizes, but it is not best on Band@10% for the 400 × 360 matrix. MD places 0.60 of the mass within the fixed band, compared with 0.56 for Two-Walk, while Two-Walk has a substantially lower R2S than MD, 1.87 versus 2.65. The disagreement is informative. MD puts slightly more mass inside the chosen band threshold, whereas Two-Walk reduces the overall squared distance of relation mass from the diagonal. No single metric completely describes the visual organization of a reordered matrix.

## 5<sub>.</sub>2 R<sub>ea</sub>l M<sub>a</sub>trix R<sub>esu</sub>lt<sub>s</sub>

The strongest improvements occur in relational matrices where same-side entities are naturally compared through shared neigh bors.

Classification crosswalks. For SIC to NAICS, Two-Walk reduces 10<sup>2</sup>×R2S from 0.68 for One-Walk to 0.18, a reduction of approximately 73.5%. One-Walk has the slightly higher Band@10%, 0.98 versus 0.96, which indicates that both methods form a narrow band but Two-Walk reduces the remaining long-distance mass more strongly. For CIP to SOC, Two-Walk improves both independent metrics. R2S decreases from 2.52 to 1.67, and Band@10% increases from 0.43 to 0.61. Crosswalks are a favorable setting because two source codes may be related precisely when they map to similar target codes, and the reverse relation also holds for target codes.

Text and bibliographic matrices. On 20 Newsgroups, Two-Walk ties One-Walk for the best R2S at 0.43 and raises Band@10% from 0.94 to 0.95. On OpenAlex, R2S improves from 0.98 to 0.92, and Band@10% reaches 1.00. The OpenAlex result is visually consistent with Figure 1, where authors with related topic profiles and topics with related author profiles form coordinated blocks.

Small and spatially organized matrices. The ACS matrix is only 24 × 10. CA gives the best R2S and Band@10%, while Two-Walk is not competitive on R2S. In this regime, a low-dimensional factor coordinate is suficient and the additional shared-neighbor graph does not add useful separation. LODES is governed partly by geographic adjacency. One-Walk and Two-Walk are nearly identical on R2S, and CA has the highest Band@10%. MBTA is the clearest boundary case. Two-Walk ties One-Walk on R2S, but MD has the best Band@10% and MWB-AUC. Route and station order is influenced by local transit topology, which is not equivalent to similarity through shared bipartite neighborhoods.

These results support a scoped conclusion. Two-Walk is useful when latent organization is relational and shared-neighbor profiles carry information beyond direct edges. It is not a universal substitute for geometric, spatial, or very low-dimensional ordering criteria.

<table><tr><td></td><td></td><td></td><td colspan="6">102× R2S ↓</td><td colspan="6"></td><td colspan="6"></td></tr><tr><td></td><td></td><td>0</td><td>M</td><td>HC OW</td><td></td><td>CA</td><td>MD</td><td>TW</td><td>0</td><td>M</td><td>HC OW</td><td>CA</td><td>MD</td><td>TW</td><td></td><td>0</td><td>M HC</td><td>OW</td><td>CA</td><td>MD</td><td>TW</td></tr><tr><td>Synthetic component</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>A: Clean one-to-one</td><td>400×360</td><td>16.70</td><td>13.84</td><td>10.36</td><td>0.59</td><td></td><td></td><td>0.20</td><td>0.21</td><td>0.24</td><td>0.86</td><td>0.85</td><td>0.76</td><td>0.90</td><td>0.44</td><td>0.47</td><td>0.53</td><td>0.92</td><td>0.92</td><td></td><td>0.93</td></tr><tr><td></td><td>200×180</td><td>16.88</td><td>13.72</td><td>10.67</td><td>0.52 0.55</td><td>0.63</td><td>0.79 0.89</td><td>0.43 0.42</td><td>0.20 0.23</td><td>0.26</td><td>0.87</td><td>0.86</td><td>0.78</td><td>0.92</td><td>0.44</td><td>0.48</td><td>0.54</td><td>0.92</td><td>0.92</td><td>0.89 0.89</td><td>0.94</td></tr><tr><td></td><td>100×90</td><td>17.09</td><td>13.25</td><td>9.14</td><td>0.60</td><td>0.79</td><td>0.96</td><td>0.51</td><td>0.21 0.24</td><td>0.34</td><td>0.89</td><td>0.85</td><td>0.88</td><td>0.92</td><td>0.44</td><td>0.49</td><td>0.60</td><td>0.93</td><td>0.91</td><td>0.91</td><td>0.94</td></tr><tr><td>B: Paired subgroup overlap</td><td>400×360</td><td>16.72</td><td>14.04</td><td>8.10</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>200×180</td><td>16.76</td><td>14.15</td><td>8.53</td><td>1.04 1.15</td><td>1.11 1.20</td><td>1.08 1.34</td><td>0.88 0.19 0.82</td><td>0.21 0.20</td><td>0.31 0.27</td><td>0.68</td><td>0.67 0.70</td><td>0.69 0.72</td><td>0.74 0.78</td><td>0.44 0.44</td><td>0.47 0.47</td><td>0.60 0.56</td><td>0.87 0.87</td><td>0.87 0.87</td><td>0.87 0.87</td><td>0.89 0.90</td></tr><tr><td></td><td>100×90</td><td>16.90</td><td>13.68</td><td>10.39</td><td>1.03</td><td>1.32</td><td>1.93</td><td>0.94 0.22</td><td>0.22 0.24</td><td>0.32</td><td>0.67 0.76</td><td>0.72</td><td>0.73</td><td>0.79</td><td>0.44</td><td>0.49</td><td>0.55</td><td>0.89</td><td>0.87</td><td>0.86</td><td>0.90</td></tr><tr><td>C: Shared super-prototype</td><td>400×360</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>200×180</td><td>16.92 16.88</td><td>14.07 13.62</td><td>8.71 9.15</td><td>1.13 1.01</td><td>1.35</td><td>1.78 0.93</td><td>0.19</td><td>0.22</td><td>0.27</td><td>0.67</td><td>0.64</td><td>0.69</td><td>0.73</td><td>0.43</td><td>0.47</td><td>0.58</td><td>0.87</td><td>0.85</td><td>0.85</td><td>0.88</td></tr><tr><td></td><td>100×90</td><td>17.24</td><td>13.12</td><td>7.44</td><td>1.01</td><td>1.24 1.39</td><td>2.07 0.86 3.23</td><td>0.20 0.86 0.21</td><td>0.23 0.25</td><td>0.31 0.37</td><td>0.72</td><td>0.69 0.72</td><td>0.72 0.73</td><td>0.77 0.83</td><td>0.44 0.44</td><td>0.49 0.50</td><td>0.58 0.64</td><td>0.88 0.89</td><td>0.87 0.87</td><td>0.85 0.83</td><td>0.90 0.90</td></tr><tr><td>D: Shared prototype with noise</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.77</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400×360 200×180</td><td>16.63 16.74</td><td>15.12 14.35</td><td>9.85 8.94</td><td>1.28</td><td>1.54</td><td>1.18 1.00</td><td>0.20</td><td>0.22</td><td>0.24</td><td>0.63</td><td>0.61</td><td>0.64</td><td>0.70</td><td>0.44</td><td>0.47</td><td>0.52</td><td>0.85</td><td>0.84</td><td>0.86</td><td>0.88</td></tr><tr><td></td><td>100×90</td><td>17.07</td><td>13.51</td><td>7.70</td><td>1.23 1.14</td><td>1.67</td><td>1.51</td><td>1.04 0.20</td><td>0.23</td><td>0.29</td><td>0.69</td><td>0.62</td><td>0.65</td><td>0.72</td><td>0.44</td><td>0.48</td><td>0.57</td><td>0.87</td><td>0.84</td><td>0.85</td><td>0.88</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>1.65</td><td>2.67</td><td>0.94 0.21</td><td>0.25</td><td>0.40</td><td>0.74</td><td>0.67</td><td>0.72</td><td>0.79</td><td>0.44</td><td>0.50</td><td>0.64</td><td>0.88</td><td>0.85</td><td>0.84</td><td>0.90</td></tr><tr><td>E: Cross-block leakage</td><td>400×360 200×180</td><td>16.65</td><td>14.70</td><td>9.46</td><td>2.09</td><td>2.09</td><td>2.65 1.87</td><td>0.19</td><td>0.22</td><td>0.26</td><td>0.55</td><td>0.57</td><td>0.60</td><td>0.56</td><td>0.44</td><td>0.47</td><td>0.56</td><td>0.81</td><td>0.81</td><td>0.82</td><td>0.82</td></tr><tr><td></td><td>100×90</td><td>16.88 16.87</td><td>13.88 13.15</td><td>7.59 8.36</td><td>1.86</td><td>2.16 3.28</td><td>1.65</td><td>0.20</td><td>0.23</td><td>0.38</td><td>0.60</td><td>0.59</td><td>0.61</td><td>0.63</td><td>0.44</td><td>0.49</td><td>0.63</td><td>0.83</td><td>0.82</td><td>0.80</td><td>0.84</td></tr><tr><td>Real-world component</td><td></td><td></td><td></td><td>1.61</td><td>2.12</td><td></td><td>4.25</td><td>1.60</td><td>0.22</td><td>0.25 0.33</td><td>0.67</td><td>0.64</td><td></td><td>0.67 0.70</td><td>0.44</td><td>0.50</td><td>0.61</td><td>0.85</td><td></td><td>0.83</td><td>0.79 0.86</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SIC → NAICS CIP → SOC</td><td>1004×1176 2143×868</td><td>2.61 15.90</td><td>10.78 16.04</td><td>2.11 12.18</td><td>0.68 2.52 9.77</td><td>2.88 12.37</td><td>0.76 0.18</td><td>0.66 1.67</td><td>0.30 0.19 0.27</td><td>0.52 0.20</td><td>0.98 0.43</td><td>0.51 0.30</td><td>0.90 0.17</td><td>0.96 0.61</td><td>0.82 0.44</td><td>0.56 0.78 0.50 0.53</td><td>0.94 0.77</td><td>0.57</td><td>0.73 0.40</td><td>0.92</td><td>0.97 0.83</td></tr><tr></table>

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> C<sub>om</sub>bi<sub>ne</sub>d <sub>rec</sub>t<sub>angu</sub>l<sub>ar eva</sub>l<sub>ua</sub>ti<sub>on</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>.</sub> B<sub>es</sub>t <sub>en</sub>t<sub>ry per row an</sub>d <sub>me</sub>t<sub>r</sub>i<sub>c</sub> i<sub>s</sub> i<sub>n</sub> b<sub>o</sub>ld<sub>, runner-up</sub> i<sub>s un</sub>d<sub>er</sub>li<sub>ne</sub>d<sub>.</sub> Synthetic component (top): mean over 20 seeds per family–size regime. Real-world component (bottom): datasets at full size. MWB<sub>-</sub>AUC i<sub>s</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erna</sub>l <sub>�-se</sub>l<sub>ec</sub>ti<sub>on me</sub>t<sub>r</sub>i<sub>c</sub> f<sub>or</sub> TW R2S <sub>an</sub>d B<sub>an</sub>d<sub>@</sub>10% <sub>are no</sub>t <sub>use</sub>d f<sub>or � se</sub>l<sub>ec</sub>ti<sub>on.</sub>

## 5<sub>.</sub>3 B<sub>ase</sub>li<sub>ne</sub> B<sub>e</sub>h<sub>av</sub>i<sub>or</sub> A<sub>cross</sub> R<sub>eg</sub>i<sub>mes</sub>

The non-spectral baselines clarify what type of information is needed for the benchmark. Original and marginal orders remain far from the diagonal on every synthetic family. Marginal sorting can expose large rows and columns, but it cannot recover a correspondence between groups on the two axes. HC reduces R2S substantially relative to these two controls, which confirms that profile similarity is useful, but its independently constructed row and column trees do not match the joint spectral methods on the relational motifs.

CA is most competitive when a small number of factor coordinates already describes the matrix. This is visible in Family A and in the ACS table. MD is most competitive when a local neighborhood update agrees with the target structure, including the large leakage regime and the transit example. Its weaker performance on several nested synthetic regimes also shows that local median updates do not consistently replace the global shared-neighbor signal. The diversity of these outcomes is important because it rules out the simpler explanation that any reasonable reordering would obtain the same result.

## 5<sub>.</sub>4 C<sub>o</sub>n<sub>s</sub>i<sub>s</sub>t<sub>e</sub>n<sub>cy</sub> A<sub>c</sub>r<sub>oss</sub> M<sub>a</sub>trix Siz<sub>e</sub>

The synthetic matrix dimensions increase from 100×90 to 400×360, which multiplies the number of cells by sixteen. Two-Walk retains the best R2S in every family at every size. It also retains the best Band@10% except for the largest leakage matrix. The absolute values do not vary monotonically with size because the generator changes the number and realization of weighted relations with the seed. The stable win pattern is the relevant observation: the advantage is not restricted to one selected matrix dimension or one aspect ratio.

The reported synthetic entries are means across matched random instances. The table does not use a hypothesis test to turn these deterministic ordering outputs into binary significance claims. The per-seed records in the anonymous artifact permit paired analysis because every method is evaluated on the same generated matrix for each seed.

## 5<sub>.</sub>5 Wh<sub>a</sub>t th<sub>e</sub> Th<sub>ree</sub> M<sub>e</sub>t<sub>r</sub>i<sub>cs</sub> R<sub>evea</sub>l

The complete table shows why three complementary summaries are useful. R2S measures all weighted distances to the diagonal. Band@10% asks whether mass falls inside one fixed visual tolerance. MWB-AUC rewards persistence ofcontiguous diagonal blocks across its cut sweep. A method can therefore improve one property without improving all three.

SIC to NAICS is one example. Two-Walk has much lower R2S than One-Walk, while One-Walk has slightly higher Band@10%. Family E at 400 × 360 is another example, where MD has the highest Band@10% but Two-Walk has the lowest R2S. We therefore avoid reducing the benchmark to one aggregate rank. The primary evidence for the adaptive method is its consistency across R2S and Band@10%, which are not used for parameter selection.

## 5<sub>.</sub>6 M<sub>osa</sub>i<sub>c</sub> C<sub>ase</sub> A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub>

Figure 2 shows four Tabula Sapiens tissues. The transition from the first row to the second changes only the order. Dominant cell type and cluster correspondences move toward the diagonal, making split, merge, and leakage patterns easier to locate. The transition from the second row to the third keeps that Two-Walk order fixed and changes only the geometry.

In the Mosaic row, a frequent cell type receives more vertical space because its row marginal is larger. A frequent computational cluster receives more horizontal space because its column marginal is larger. A large rectangle therefore indicates a large independencebaseline area $r _ { i } c _ { j } / T ^ { 2 }$ , while a dark fill indicates a large observed entry $A _ { i j }$ . This distinction is important. The layout places a split or merge relation in the context of the total sizes of both participating categories, but rectangle area is not itself the observed count.

Across the four tissues, rare categories contract and high-mass categories expand. This makes it easier to distinguish a visually similar split involving a major cell population from one involving a small population. It also creates a limitation: very small marginals can become dificult to inspect, and unequal geometry weakens exact cell-to-cell comparison. The figure therefore supports the intended overview task but does not establish a general perceptual advantage for every matrix reading task.

## 6 Di<sub>scuss</sub>i<sub>on an</sub>d C<sub>onc</sub>l<sub>us</sub>i<sub>on</sub>

MHeatmap combines a spectral seriation method with a marginal proportional matrix view. The main algorithmic contribution is Two-Walk. Its adjacency retains observed row to column edges and adds the two same side afinities induced by length two walks. Theorem 1 shows that the resulting Fiedler coordinate balances smoothness among rows with shared columns, smoothness among columns with shared rows, and alignment of directly related row and column vertices.

Scope ofthe ordering claim. The complete benchmark provides both positive evidence and boundary conditions. Two-Walk is consistently strong when shared profiles define overlapping or nested groups, and it performs well on crosswalk and bibliographic matrices. Its benefit is smaller when direct one step structure is already suficient, when a matrix is very small, or when spatial and route locality govern the desired order. The two walk terms reward shared bipartite neighborhoods. They do not directly encode geographic distance, temporal order, or route continuity unless those concepts are reflected in the neighborhoods.

The three metrics also define a particular notion of success. They favor arrangements in which relation mass can be organized near matched positions on the normalized axes. This is appropriate for correspondences and block or band motifs, but it is not the only meaningful matrix objective. Hub, checkerboard, and spatial patterns need not become narrowly diagonal. The LODES and MBTA results illustrate this boundary and prevent the benchmark from implying that every matrix should adopt one visual form.

Parameter and computational qualifications. Adaptive selection of � gives Two-Walk more flexibility than a fixed method. MWB-AUC is therefore identified as selection aligned, while the principal comparison uses R2S and Band@10%, which do not choose �. The table evaluates the implemented adaptive procedure rather than a parameter free estimator. A fixed global � or a sensitivity curve would be an additional robustness check, but the present conclusion is restricted to the adaptive method and the independent outcome metrics.

Lemma 1 proves that dense Gram matrices are not required and that one matrix vector product is linear in nnz(�) apart from vector operations. It does not establish wall clock scalability because eigensolver convergence also depends on the eigengap, conditioning, and tolerance. The current experiments evaluate ordering quality, so we make no empirical runtime or memory superiority claim.

Spectral and degree efects. A Fiedler order can be nonunique when the relevant eigenvalue is repeated or coordinate values tie. Deterministic component, sign, and tie handling are therefore part of the algorithm and must be recorded with the submitted permutations. In addition, the Gram terms aggregate weighted shared neighbors and can emphasize high mass vertices. Maximum entry normalization removes arbitrary global units but does not remove degree heterogeneity. A degree normalized variant may be preferable when profile shape matters more than shared mass, but it is outside the present method.

Role of Mosaic. The Mosaic stage is defined separately from ordering. Its complete tiling makes row and column marginals visible inside the matrix body, while color retains observed values. It is useful when marginal context afects the interpretation of a split, merge, or block, but it can compress rare categories and does not replace an equal area heatmap for precise comparison. The paired panels hold the order fixed so that the layout efect is not confused with the ordering efect. They establish interpretability, not universal perceptual superiority. Such a claim would require a task specific study against equal area heatmaps, marginal bars, and classical mosaic displays.

In summary, Two-Walk extends bipartite spectral seriation with explicit shared neighbor structure and produces coordinated orders for weighted rectangular matrices. The full table shows when this structure improves recovery and where alternative assumptions remain competitive. Mosaic provides a precisely defined proportional rendering when row and column mass is part of the analysis.

## R<sub>e</sub>f<sub>erences</sub>

[1] Jonathan E. Atkins, Erik G. Boman, and Bruce Hendrickson. 1998. A Spectral Algorithm for Seriation and the Consecutive Ones Problem. SIAM J. Com ut. 28, 1 (1998), 297–310. doi:10.1137/S0097539795285771

[2] Ziv Bar-Joseph, David K. Giford, and Tommi S. Jaakkola. 2001. Fast optimal leaf ordering for hierarchical clustering. Bioinformatics 17, suppl\_1 (2001), S22–S29. doi:10.1093/bioinformatics/17.suppl\_1.S22

[3] Stephen T. Barnard, Alex Pothen, and Horst D. Simon. 1995. A Spectral Algorithm for Envelope Reduction of Sparse Matrices. Numerical Linear Algebra with A lications 2, 4 (1995), 317–334. doi:10.1002/nla.1680020402

[4] Michael Behrisch, Benjamin Bach, Michael Hund, Michael Delz, Laura von Rüden, Jean-Daniel Fekete, and Tobias Schreck. 2017. Magnostics: Image-Based Search of Interesting Matrix Views for Guided Network Exploration. IEEE Transactions on Visualization and Computer Graphics 23, 1 (2017), 31–40. doi:10.1109/TVCG.2 016.2598467

[5] Vincent D. Blondel, Anahí Gajardo, Maureen Heymans, Pierre Senellart, and Paul Van Dooren. 2004. A Measure of Similarity between Graph Vertices: Applications to Synonym Extraction and Web Searching. SIAM Rev. 46, 4 (2004), 647–666. doi:10.1137/S0036144502415960

[6] Olca A. Cakiroglu, Cesim Erten, Omer Karatas, and Melih Sozdinler. 2009. Crossing minimization in weighted bipartite graphs. Journal ofDiscrete Algorithms 7, 4 (2009), 439–452. doi:10.1016/j.jda.2008.08.003

![](images/d7b65e7bc272f20925c1f643cccc4f676e5269960c7be2ed9ec0c8502a5dd1df.jpg)  
Fi<sub>g</sub>ure 2: Four cell t<sub>yp</sub>e b<sub>y</sub> cluster contin<sub>g</sub>enc<sub>y</sub> matrices from Tabula Sa<sub>p</sub>iens. The first row uses the in<sub>p</sub>ut order with e<sub>q</sub>ual-area <sub>ce</sub>ll<sub>s.</sub> Th<sub>e secon</sub>d <sub>row app</sub>li<sub>es</sub> T<sub>wo-</sub>W<sub>alk an</sub>d k<sub>eeps equa</sub>l<sub>-area ce</sub>ll<sub>s.</sub> Th<sub>e</sub> thi<sub>r</sub>d <sub>row</sub> k<sub>eeps exac</sub>tl<sub>y</sub> th<sub>e same</sub> T<sub>wo-</sub>W<sub>alk row an</sub>d <sub>co</sub>l<sub>umn or</sub>d<sub>ers an</sub>d <sub>app</sub>li<sub>es</sub> M<sub>osaic geome</sub>t<sub>ry.</sub> Th<sub>e secon</sub>d<sub>-</sub>t<sub>o-</sub>thi<sub>r</sub>d<sub>-row compar</sub>i<sub>son</sub> th<sub>ere</sub>f<sub>ore</sub> i<sub>so</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e</sub> l<sub>ayou</sub>t <sub>e</sub>f<sub>ec</sub>t<sub>.</sub>

[7] William S. Cleveland and Robert McGill. 1984. Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods. J. Amer. Statist. Assoc. 79, 387 (1984), 531–554. doi:10.1080/01621459.1984.10478080

[8] Inderjit S. Dhillon. 2001. Co-clustering documents and words using bipartite spectral graph partitioning. In Proceedings ofthe Seventh ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, New York, NY, USA, 269–274. doi:10.1145/502512.502550

[9] Miroslav Fiedler. 1973. Algebraic connectivity of graphs. Czechoslovak Mathematical Journal 23, 2 (1973), 298–305. doi:10.21136/CMJ.1973.101168

[10] Fajwel Fogel, Rodolphe Jenatton, Francis Bach, and Alexandre d’Aspremont. 2015. Convex Relaxations for Permutation Problems. SIAM J. Matrix Anal. Appl. 36, 4 (2015), 1465–1488. doi:10.1137/130947362

[11] Michael Friendly. 1994. Mosaic Displays for Multi-Way Contingency Tables. J. Amer. Statist. Assoc. 89, 425 (1994), 190–200. doi:10.1080/01621459.1994.10476460

[12] Michael Greenacre. 2017. Correspondence Analysis in Practice (3 ed.). Chapman and Hall/CRC, Boca Raton, FL, USA. doi:10.1201/9781315369983

[13] John A. Hartigan and Beat Kleiner. 1981. Mosaics for Contingency Tables. In Computer Science and Statistics: Proceedings ofthe 13th Symposium on the Interface, William F. Eddy (Ed.). Springer, New York, NY, USA, 268–273. doi:10.1007/978-1- 4613-9464-8\_37

[14] David Liben-Nowell and Jon Kleinberg. 2007. The Link-Prediction Problem for Social Networks. Journal ofthe American Societ for Information Science and Technolo y 58, 7 (2007), 1019–1031. doi:10.1002/asi.20591

[15] Innar Liiv. 2010. Seriation and Matrix Reordering Methods: An Historical Overview. Statistical Analysis and Data Mining 3, 2 (2010), 70–91. doi:10.1 002/sam.10071

[16] Massachusetts Bay Transportation Authority. 2026. MBTA GTFS Documentation. https://github.com/mbta/gtfs-documentation. Accessed 2026-04-29.

[17] David Meyer, Achim Zeileis, and Kurt Hornik. 2006. The Strucplot Framework: Visualizing Multi-Way Contingency Tables with vcd. Journal of Statistical Software 17, 3 (2006), 1–48. doi:10.18637/jss.v017.i03

[18] Tom Mitchell. 1997. Twenty Newsgroups. https://archive.ics.uci.edu/dataset/11 3/twenty+newsgroups. doi:10.24432/C5C323 Accessed 2026-04-29.

[19] National Center for Education Statistics and U.S. Bureau of Labor Statistics. 2020. CIP 2020 to SOC 2018 Crosswalk. https://nces.ed.gov/ipeds/cipcode/resources.as px?y=56.

[20] Jason Priem, Heather Piwowar, and Richard Orr. 2022. OpenAlex: A fullyopen index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833 [cs.DL] https://arxiv.org/abs/2205.01833

[21] John K. Reid and Jennifer A. Scott. 2006. Reducing the Total Bandwidth of a Sparse Unsymmetric Matrix. SIAM J. Matrix Anal. Appl. 28, 3 (2006), 805–821. doi:10.1137/050629938

[22] Hans Riedwyl and Michel Schüpbach. 1994. Parquet Diagram to Plot Contingency Tables. In Softstat ’93: Advances in Statistical Software, Frank Faulbaum (Ed.). Gustav Fischer, New York, 293–299.

[23] Tabula Sapiens Consortium. 2022. The Tabula Sapiens: A Multiple-Organ, Single-Cell Transcriptomic Atlas of Humans. Science 376, 6594 (2022), eabl4896. doi:10 .1126/science.abl4896

[24] U.S. Census Bureau. 2022. LEHD Origin-Destination Employment Statistics. https://lehd.ces.census.gov/data/lodes/.

[25] U.S. Census Bureau. 2022. North American Industry Classification System: Concordances. https://www.census.gov/naics/concordances/concordances.html. Accessed 2026-04-29.

[26] U.S. Census Bureau. 2023. 2023 American Community Survey Public Use Microdata Sample Documentation. https://www.census.gov/programssurveys/acs/microdata/documentation/2023.html.

[27] Nathan van Beusekom and Jean-Daniel Fekete. 2022. Reordering for Matrix Visualization with Reorder.js. IEEE VIS Poster. https://inria.hal.science/hal-03786492

[28] Hadley Wickham and Heike Hofmann. 2011. Product Plots. IEEE Transactions on Visualization and Com uter Gra hics 17, 12 (2011), 2223–2230. doi:10.1109/ TVCG.2011.227

[29] Jiangning Zhu, Zheng Wang, Zhiyang Shen, Lai Wei, Fengyuan Tian, Mengchen Liu, and Shixia Liu. 2025. ReorderBench: A Benchmark for Matrix Reordering. IEEE Transactions on Visualization and Computer Graphics 31, 7 (2025), 4126–4140. doi:10.1109/TVCG.2025.3560345

## A Additional Qualitative Benchmark Cases

Figure 3 provides two additional full-workflow examples. The top row uses the SIC to NAICS crosswalk, and the bottom row uses the OpenAlex author by topic matrix. Each example repeats the same controlled sequence used in the main paper: input order with equal-area cells, Two-Walk with equal-area cells, and the same Two-Walk order with Mosaic geometry. These examples are supplementary to the quantitative results in Table 1.

## B R<sub>epro</sub>d<sub>uc</sub>ibilit<sub>y</sub> M<sub>a</sub>t<sub>er</sub>i<sub>a</sub>l<sub>s</sub>

Our code is at https://anonymous.4open.science/r/MHeatm ap-D6FD/. It contains the benchmark implementation, fixed data construction scripts or download instructions, synthetic generators, all baseline wrappers, per-instance permutations and metric records, selected values of �, and scripts that regenerate the complete table and every submitted figure.

![](images/7155c736e77f4693819ecb47dc4c4f3268b06aa74048a02f814f8af9a12e4fc6.jpg)  
Fi<sub>g</sub>ure 3: Additional <sub>q</sub>ualitative exam<sub>p</sub>les. To<sub>p</sub>: SIC to NAICS. Bottom: O<sub>p</sub>enAlex author b<sub>y</sub> to<sub>p</sub>ic. Columns show the in<sub>p</sub>ut <sub>or</sub>d<sub>er,</sub> T<sub>w</sub>o<sub>-</sub>Wa<sub>lk</sub> <sub>w</sub>ith <sub>equa</sub>l<sub>-area</sub> <sub>ce</sub>ll<sub>s,</sub> <sub>an</sub>d th<sub>e</sub> <sub>same</sub> T<sub>w</sub>o<sub>-</sub>Wa<sub>lk</sub> <sub>or</sub>d<sub>er</sub> <sub>w</sub>ith Mosa<sub>i</sub>c <sub>geome</sub>t<sub>ry.</sub>