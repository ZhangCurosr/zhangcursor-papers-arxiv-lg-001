# End-to-End Neural Shrinkage of Indefinite Pairwise Correlation Matrices for Small-Cap-Inclusive Portfolios

Preprint

Christian Bongiorno Université Paris-Saclay, CentraleSupélec Mathématiques et Informatique pour la Complexité et les Systèmes 91190 Gif-sur-Yvette, France christian.bongiorno@centralesupelec.fr

Lorenzo Villassero Université Paris-Saclay, CentraleSupélec Mathématiques et Informatique pour la Complexité et les Systèmes 91190 Gif-sur-Yvette, France

August 2026

## Abstract

Small-cap-inclusive equity universes contain recently listed and intermittently traded securities, so enforcing a common look-back discards a substantial fraction of the available information. Pairwisecomplete estimation preserves the longest overlap for each asset pair, but the resulting correlation matrix can be indefinite because its entries are computed on diferent samples. This prevents direct use in Markowitz optimization and falls outside the assumptions of standard random-matrix shrinkage. We adapt a rotation-invariant neural covariance estimator to this setting. The model computes mask-aware marginal moments and a pairwise correlation matrix proxy, processes its signed spectrum, and uses a bidirectional gated recurrent unit conditioned on factor-aligned efective sample lengths derived from the overlap matrix and eigenvector loadings. It maps all eigenvalues, including negative ones, to a positive inverse spectrum. The reconstructed covariance is positive definite and is trained end-to-end to minimize five-day realized global-minimum-variance risk. We evaluate 26 expanding-window models from 2000 to 2025 on up to 1,500 U.S. equities in a closing-auction simulator with point-in-time selection, commissions, financing, corporate actions, and market impact. Across the 26-year out-of-sample period, the neural estimator reduces annualized five-day volatility by approximately 20% and increases the Sharpe ratio by approximately 40% relative to the next-best covariance estimator. These improvements are consistent across realized risk, risk-adjusted performance, and drawdown control, remain after the modeled execution frictions, and are supported by a 99.9% Model Confidence Set that retains only the neural estimator.

Keywords portfolio optimization · covariance estimation · missing data · nonlinear shrinkage · random matrix theory market impact · neural networks

## 1 Introduction

We consider Markowitz portfolio construction when the investable universe is determined, for example, by an investment mandate or a separate alpha signal. This approach requires estimating a covariance matrix from a rectangular return panel in which all selected assets are observed over a common look-back window. This condition is often violated when the universe includes recent initial public oferings, lower-capitalization securities with intermittent trading, temporarily suspended stocks, or assets with incomplete historical coverage. In these settings, excluding a security solely because its return history is shorter or fragmented would alter the prescribed investment universe and is not a negotiable solution.

A natural alternative is to estimate each covariance or correlation entry from the maximum history shared by the corresponding pair. This pairwise-complete construction uses more information than complete-case deletion and is consistent with the economic motivation for studying unequal return histories [1]. Its numerical consequence is less convenient: because diferent entries are estimated from diferent subsets of dates, the assembled matrix is not necessarily positive semidefinite. Negative eigenvalues imply that the matrix can assign negative variance to some portfolios and therefore cannot be interpreted as a valid covariance matrix. Several methods seek to address this problem. Nearest-correlation approaches project the indefinite estimate onto the positive-semidefinite cone [2], whereas regression and factor-imputation methods construct a valid covariance matrix directly from the incomplete panel [3, 4]. These methods, however, do not jointly address the second challenge faced by large universes. Let � denote the number of assets and $\Delta t _ { \mathrm { i n } }$ the length of a common estimation window. When the concentration ratio $q : = n / \Delta t _ { \mathrm { i n } }$ is not negligible, empirical eigenvalues are strongly afected by sampling noise and require substantial shrinkage [5].

Existing Random Matrix Theory (RMT) based shrinkage methods are derived for covariance matrices computed from a common rectangular return panel. In this setting, all entries share the same sample size and the empirical covariance matrix is positive semidefinite. Pairwise-complete estimation violates both conditions. Each entry is computed from a diferent number of overlapping observations, and the assembled matrix may contain negative eigenvalues. Standard RMT results therefore do not characterize the spectral statistics of this estimator and do not provide a principled rule that simultaneously repairs the negative eigenvalues and shrinks the positive spectrum. To our knowledge, these two operations have not been combined within a consistent analytical framework.

Bongiorno et al. [6] showed that a dimension-agnostic recurrent network can learn a rotation-invariant eigenvalue map directly from a realized Global Minimum Variance (GMV) objective. This approach does not require an explicit analytical model for the empirical spectrum. However, it cannot be applied unchanged when the input matrix may be indefinite and its entries are based on heterogeneous overlap lengths. We extend it to indefinite pairwise cross-moment matrices by providing the network with the signed eigenvalues and factor-specific measures of the available sample information.

Specifically, we condition the spectral map on factor-aligned efective sample lengths derived from pairwise overlaps and squared eigenvector loadings. The network transforms the signed spectrum into a positive inverse spectrum and is trained end-to-end to minimize future GMV variance. For evaluation, the resulting covariance estimates are passed to the same long-only GMV optimizer used for all benchmarks.

The empirical analysis evaluates whether the statistical gains survive realistic implementation costs in a universe extended toward lower-capitalization equities. We use a point-in-time universe of up to 1,500 U.S. stocks, retain short and fragmented return histories, and rebalance every five sessions from 2000 through 2025. Each strategy is evaluated in a continuous broker simulation that accounts for integer-share holdings, closing-auction execution, commissions, regulatory fees, dividends, splits, corporate-action settlements, and financing. Market impact follows a square-root closing-auction specification with coeficients that vary across market-capitalization groups and listing exchanges [7]. This execution layer tests whether reductions in estimated portfolio risk remain economically relevant once concentration, turnover, and trade size are translated into realized trading costs.

## 2 Indefinite Pairwise Risk Estimation

Let $\mathbf { R } \in \mathbb { R } ^ { \Delta t _ { \mathrm { i n } } \times n }$ denote a return panel and $\mathbf { M } \in \{ 0 , 1 \} ^ { \Delta t _ { \mathrm { i n } } \times n }$ its validity mask, where $M _ { t i } = 1$ indicates that $R _ { t i }$ is observed. The overlap matrix is

$$
\mathbf { T } = \mathbf { M } ^ { \top } \mathbf { M } , \qquad T _ { i j } = \sum _ { t = 1 } ^ { \Delta t _ { \mathrm { i n } } } M _ { t i } M _ { t j } .\tag{1}
$$

The marginal sample moments are

$$
\hat { \mu } _ { i } = \frac { 1 } { T _ { i i } } \sum _ { t = 1 } ^ { \Delta t _ { \mathrm { i n } } } M _ { t i } R _ { t i } , \qquad \hat { \sigma } _ { i } ^ { 2 } = \frac { 1 } { T _ { i i } } \sum _ { t = 1 } ^ { \Delta t _ { \mathrm { i n } } } M _ { t i } \big ( R _ { t i } - \hat { \mu } _ { i } \big ) ^ { 2 } .\tag{2}
$$

Define the marginally standardized returns

$$
Z _ { t i } = M _ { t i } \frac { R _ { t i } - \hat { \mu } _ { i } } { \hat { \sigma } _ { i } } .\tag{3}
$$

For $T _ { i j } > 0 .$ , the marginally standardized pairwise cross-moment matrix entry is

$$
C _ { \cap , i j } = \frac { 1 } { T _ { i j } } \sum _ { t = 1 } ^ { \Delta t _ { \mathrm { i n } } } Z _ { t i } Z _ { t j } ,\tag{4}
$$

with entries having no overlap set to zero. Unlike overlap-specific Pearson centering, Eqs. (2)–(4) estimate each asset’s marginal moments once and reuse them for every pair. With

$$
{ \hat { \mathbf { D } } } = \mathrm { D i a g } \big ( { \hat { \sigma } } _ { 1 } , \dots , { \hat { \sigma } } _ { n } \big ) , \qquad { \Sigma } _ { \cap } = { \hat { \mathbf { D } } } { \mathbf { C } } _ { \cap } { \hat { \mathbf { D } } } ,\tag{5}
$$

$\pmb { \Sigma } _ { \cap }$ is the marginally centered pairwise cross-moment matrix.

Although $\mathbf { C } _ { \cap }$ is symmetric with unit diagonal, its entries use diferent subsets and diferent denominators. It is therefore not necessarily positive semidefinite and may have negative eigenvalues [2]. We write

$$
\begin{array} { r l } & { \mathbf { C } _ { \cap } = \mathbf { Q } _ { \cap } \mathbf { \Lambda } _ { \cap } \mathbf { Q } _ { \cap } ^ { \top } , } \\ & { \mathbf { A } _ { \cap } = \operatorname { D i a g } \bigl ( \lambda _ { \cap , 1 } , \ldots , \lambda _ { \cap , n } \bigr ) , \qquad \lambda _ { \cap , 1 } \leq \cdots \leq \lambda _ { \cap , n } . } \end{array}\tag{6}
$$

Since $\hat { \bf D }$ is nonsingular, $\pmb { \Sigma } _ { \cap }$ and $\mathbf { C } _ { \cap }$ are congruent and therefore have the same inertia by Sylvester’s law of inertia $[ 8 ,$ p. 282]. Marginal scaling alone therefore cannot repair an indefinite pairwise estimate.

The conventional pairwise-complete estimator recomputes means and variances on each overlap. Its correlations are therefore bounded in [−1, 1], although the assembled matrix may still be indefinite [2]. By contrast, Eqs. (2)–(5) use marginal moments estimated from each asset’s full available history, so that $\mathbf { C } _ { \cap }$ and $\pmb { \Sigma } _ { \cap }$ collect marginally standardized pairwise cross-moments. Provided that marginal and overlap-conditioned moments converge to the same limits, the two constructions are asymptotically equivalent. We adopt marginal standardization to retain the available information for forecasting [1] and to remain consistent with the standard RMT convention of applying a common normalization to the data matrix [5, 9]. A valid correlation matrix is subsequently reconstructed by our neural network. For brevity, we refer below to $\mathbf { C } _ { \cap }$ and $\pmb { \Sigma } _ { \cap }$ as the pairwise correlation and covariance estimates, respectively.

Once regularization yields a positive-definite covariance estimate ${ \widehat { \pmb { \Sigma } } } .$ , the unconstrained GMV portfolio is

$$
\mathbf { w } = \frac { \widehat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { 1 } } { \mathbf { 1 } ^ { \top } \widehat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { 1 } } .\tag{7}
$$

The proposed method sets $\widehat { \pmb { \Sigma } } = \pmb { \Sigma } _ { \mathrm { N N } }$ , whereas $\pmb { \Sigma } _ { \cap }$ is its indefinite pairwise precursor. At deployment, each valid estimate is used in the common long-only problem

$$
\operatorname* { m i n } _ { \bf w } { \bf w } ^ { \top } \widehat \Sigma { \bf w } \quad \mathrm { s u b j e c t t o } \quad { \bf 1 } ^ { \top } { \bf w } = 1 , \qquad { \bf w } \geq { \bf 0 } .\tag{8}
$$

The required correction must therefore map the signed spectrum in Eq. (6) to a positive, regularized spectrum while preserving the information contained in the heterogeneous overlaps.

## 3 RIEnet for Incomplete Return Panels

The architecture follows the analytical GMV workflow of Ref. [6]. A lag transformation first produces a temporary return panel $\widetilde { \mathbf { R } } .$ The resulting sequence then separates into two branches: a shared multilayer perceptron estimates marginal volatilities, while a recurrent spectral module repairs and shrinks the indefinite correlation estimate. The branches recombine into $\pmb { \Sigma } _ { \mathrm { N N } }$ . The model has approximately 7,400 trainable parameters, independent of � and $\Delta t _ { \mathrm { i n } }$

## 3.1 Lag Transformation

Observed returns are mapped elementwise using the same five-parameter lag transformation as in Ref. [10]:

$$
\begin{array} { r l } & { \widetilde { R } _ { t i } : = a _ { t } R _ { t i } \frac { \operatorname { t a n h } ( b _ { t } R _ { t i } ) } { b _ { t } R _ { t i } } , } \\ & { a _ { t } : = c _ { 0 } t ^ { - c _ { 1 } } , \qquad b _ { t } : = c _ { 2 } - c _ { 3 } e ^ { - c _ { 4 } t } , } \end{array}\tag{9}
$$

$t = 1$ denotes the most recent lag, and $c _ { 0 } , \ldots , c _ { 4 } > 0$ are learned. The coeficient $a _ { t }$ controls the contribution of lag �, whereas $b _ { t }$ controls soft clipping. The mask is retained throughout the transformation, so invalid entries do not become observations. The panel Re is the common input to both subsequent branches.

## 3.2 Marginal Volatility

We apply the marginal-volatility layer of Ref. [10] to the standard deviations in Eq. (2), computed using $\widetilde { \mathbf { R } }$ in place of R. Relative to the original architecture, the only modification is the inclusion of $T _ { i i } ^ { - 1 / \bar { 2 } }$ , which accounts for the asset-specific history length and the resulting heterogeneous sampling uncertainty:

$$
\sigma _ { i , \mathrm { N N } } : = \mathrm { M L P } _ { \sigma } \left( \left[ \hat { \sigma } _ { i } \quad T _ { i i } ^ { - 1 / 2 } \right] ^ { \top } \right) .\tag{10}
$$

The shared MLP has one hidden layer with 8 leaky-ReLU units and a softplus output. The resulting scales are divided by the root mean square of the estimated volatilities to have unit mean square. This common normalization does not afect GMV weights.

## 3.3 Correlation Estimation

In the correlation branch, Eqs. (2)–(6) are evaluated using $\widetilde { \mathbf { R } }$ in place of R, yielding the pairwise cross-moment matrix $\mathbf { C } _ { \cap } .$ , its signed eigenvalues $\lambda _ { \cap , k }$ , and its eigenvectors $\mathbf { Q } _ { \cap }$ . Because the entries of $\mathbf { C } _ { \cap }$ are estimated from diferent overlaps, a single panel length cannot characterize the sampling uncertainty of every spectral factor. We transfer the pairwise sample information to factor � through

$$
\tau _ { k } : = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } Q _ { \cap , i k } ^ { 2 } T _ { i j } Q _ { \cap , j k } ^ { 2 } .\tag{11}
$$

Since the squared eigenvector loadings sum to one, $\tau _ { k }$ is a weighted average of the pairwise overlap counts, where pair $( i , j )$ is weighted by its contribution $\bar { Q } _ { \cap , i k } ^ { 2 } Q _ { \cap , j k } ^ { 2 }$ to factor �. It is invariant to eigenvector signs and asset relabeling and reduces to $\Delta t _ { \mathrm { i n } }$ for a complete panel.

Defining the factor-specific concentration ratio $q _ { k } : = n / \tau _ { k }$ , we represent the �-th ordered spectral factor by the token

$$
\mathbf { x } _ { k } : = \left( \lambda _ { \cap , k } , \sqrt { n } , \sqrt { \tau _ { k } } , q _ { k } \right) ^ { \top } .\tag{12}
$$

The ordered token sequence is processed by a bidirectional GRU following [6]:

$$
\overrightarrow { \mathbf { h } } _ { k } = \mathrm { G R U } _ { \right. } \left( \mathbf { x } _ { k } , \overrightarrow { \mathbf { h } } _ { k - 1 } \right) , \quad \overleftarrow { \mathbf { h } } _ { k } = \mathrm { G R U } _ { \left. } \left( \mathbf { x } _ { k } , \overleftarrow { \mathbf { h } } _ { k + 1 } \right) .\tag{13}
$$

Each directional GRU has 32 hidden units. A shared projection of $\mathbf { h } _ { k } : = \left[ \overrightarrow { \mathbf { h } } _ { k } ; \overleftarrow { \mathbf { h } } _ { k } \right]$ produces a positive inverse eigenvalue:

$$
\widetilde { \lambda } _ { k , \mathrm { N N } } ^ { - 1 } : = \mathrm { s o f t p l u s } \left( \gamma ^ { \top } \mathbf { h } _ { k } + \omega \right) .\tag{14}
$$

The inverse eigenvalues are rescaled by a common positive factor so that the corresponding eigenvalues $\lambda _ { k , \mathrm { N N } }$ have unit mean.

The pairwise eigenvectors are rescaled elementwise as

$$
Q _ { \mathrm { N N } , i k } : = \frac { Q _ { \cap , i k } } { \left( \sum _ { \ell = 1 } ^ { n } \lambda _ { \ell , \mathrm { N N } } Q _ { \cap , i \ell } ^ { 2 } \right) ^ { 1 / 2 } } , \qquad \mathbf { C } _ { \mathrm { N N } } : = \mathbf { Q } _ { \mathrm { N N } } \mathbf { A } _ { \mathrm { N N } } \mathbf { Q } _ { \mathrm { N N } } ^ { \top } .\tag{15}
$$

By construction, $\mathbf { C } _ { \mathrm { N N } }$ has unit diagonal and is positive definite.

## 3.4 End-to-End Training

The marginal and correlation branches are combined to obtain

$$
{ \boldsymbol \Sigma } _ { \mathrm { N N } } = { \bf D } _ { \mathrm { N N } } { \bf C } _ { \mathrm { N N } } { \bf D } _ { \mathrm { N N } } .\tag{16}
$$

Because ${ \bf D } _ { \mathrm { N N } }$ has positive diagonal entries and $\mathbf { C } _ { \mathrm { N N } }$ is positive definite, $\pmb { \Sigma } _ { \mathrm { N N } }$ is a valid covariance estimate. During training, we set $\widehat { \pmb { \Sigma } } = \pmb { \Sigma } _ { \mathrm { N N } }$ and compute the unconstrained GMV portfolio analytically using Eq. (7). The network minimizes its realized five-session variance,

$$
\begin{array} { r } { \mathcal { L } ( \mathbf { w } _ { \mathrm { N N } } , \pmb { \Sigma } _ { \mathrm { o u t } } ) : = n \mathbf { w } _ { \mathrm { N N } } ^ { \top } \pmb { \Sigma } _ { \mathrm { o u t } } \mathbf { w } _ { \mathrm { N N } } . } \end{array}\tag{17}
$$

The loss is diferentiated through the complete estimation pipeline, from the lag transformation to the reconstructed covariance matrix.

Although allocation constraints could be incorporated directly into the training problem [11], we deliberately train the estimator using the unconstrained analytical solution. Portfolio constraints vary across clients, products, and investment mandates, and embedding a specific constraint set in the loss could require retraining the model whenever the allocation problem changes. We instead learn a covariance estimator independently of the deployment constraints and subsequently apply the common long-only optimizer in Eq. (8). This separation evaluates whether the learned risk representation remains useful when the final allocation rule difers from the one used during training.

The architecture is also trained across variable input dimensions. Each training sample contains between 50 and 500 assets (�) and between 600 and 1,200 requested history days $( \Delta t _ { \mathrm { i n } } )$ , whereas the out-of-sample tests contain up to

1,500 assets. This deliberate dimension shift tests whether the architecture generalizes beyond the cross-sectional sizes observed during training. We train 26 expanding-window models: the first uses 1990–1999 to predict the 2000 out-of-sample year, and the final model uses 1990-2024 for 2025. Training uses Adam with an initial learning rate of $1 0 ^ { - 4 }$ and continuous decay 0.99<sup>�/500</sup> after batch �, gradient clipping at one, gradient accumulation over two batches of 32 samples, 500 batches per epoch, and 100 epochs.

## 4 Experimental Setup

## 4.1 Point-in-Time Investable Universe

The data cover U.S. common equities and ADRs on the NYSE, Nasdaq, and NYSE American from 1990 through 2025. At each investment date, selection uses only point-in-time information. We retain securities priced between USD 10 and USD 2,000, require at least five million shares outstanding, and remove duplicate company listings by retaining one security per company. Each security must have a complete price history over the most recent 20 sessions. We also exclude securities whose 5- or 20-session log-volatility falls below the cross-sectional lower 1.5-IQR bound, thereby removing near-zero-volatility stocks that could mechanically reduce the GMV objective and trivialize the portfolio problem. Together, these filters define an investable universe intended to exclude securities with limited execution capacity and to contain transaction costs and market impact, consistent with the use of size, liquidity, and trading-history screens in institutional portfolio construction [12, 13]. The requested universe contains the first 1,500 highest-ranked securities by market capitalization. This construction deliberately retains recently listed and fragmented histories. Across all stock–rebalance observations, 17.15% have fewer than $\Delta t _ { \mathrm { i n } } = 1$ , 200 valid returns, 8.45% have fewer than 600, and 3.48% have fewer than 252. Unequal return histories are therefore a material feature of the evaluated universe.

Portfolio signals are produced every five trading days using a requested look-back of $\Delta t _ { \mathrm { i n } } = 1 , 2 0 0 ~ \mathrm { d a y s }$ . Model inputs consist of the adjusted close-to-close return panel R and its binary validity mask M, constructed using only information available at the signal time. Returns from the execution day are excluded to avoid look-ahead bias, since the closing price is not yet observed when the portfolio is formed. Every covariance estimator is combined with the same long-only GMV optimizer. Target weights are rounded to increments of 0.1% and renormalized before broker simulation. The out-of-sample period is continuous from January 4, 2000 through December 30, 2025, comprising 6,537 sessions.

## 4.2 Compared Estimators

We first consider estimators designed for incomplete return panels. The nearest-correlation estimator applies an Anderson-accelerated implementation of Higham’s projection [2, 14]. Its bootstrap variant repeats pairwise-complete estimation and nearest-correlation projection over 25 resamples and averages the resulting correlation matrices, thereby adding regularization [15]. The factor estimator recovers a low-rank common component from a feasible complete block and uses it to impute missing observations before estimating the covariance matrix [4]. The regression estimator reconstructs the covariance matrix through sequential ridge regressions using the observations available at each step [3, 16]. We refer to these methods as Anderson, Bootstrap, Factor, and Ridge, respectively.

MLE and Quadratic Inverse Shrinkage (QIS) are instead applied to the complete-row subpanel available at each signal date. Importantly, QIS is derived for a sample covariance computed from a single common panel, with a well-defined observation count and degrees of freedom [17]. Covariance matrices obtained through pairwise estimation, nearest-correlation projection, or factor imputation do not satisfy this sampling model. Applying RMT after these methods would therefore require a specific statistical derivation.

All covariance estimators use the same long-only GMV optimizer and execution protocol. Equal Weight is included as the allocation-free benchmark. We omit capitalization weighting because it would mechanically concentrate the portfolio in the largest firms and reduce the small-cap exposure central to the experiment.

## 4.3 Broker Simulation of Closing-Auction Trading

The broker simulation starts with USD 1 million and processes the account chronologically. Each session applies splits and dividends, executes the closing rebalance, settles corporate actions, marks positions, and accrues financing. Target holdings are integer shares sized from a strict pre-execution estimate of Net Liquidation Value (NLV) computed using the execution-day opening prices; closing prices are used only for execution. The account ledger, security identifiers, and corporate-action provenance remain continuous through the full 2000-2025 simulation.

Commissions follow the IBKR Pro tiered U.S. equity schedule, including the per-order minimum and notional cap [18]; sell orders also incur SEC and FINRA charges. Negative cash accrues financing at the Federal Funds Efective Rate [19] plus the broker spread on a 360-day basis. The simulator applies dividends, splits, stock and cash reorganizations, and fractional-share cash-in-lieu.

Table 1: Out-of-sample performance with physical execution, January 2000–December 2025. Vol. 5d is annualized from non-overlapping five-session returns; � is estimated at the same horizon against the Russell 1000 price index; MCS �-values use squared five-session-return losses. All figures are net of execution costs. Methods are ordered by increasing five-session volatility.
<table><tr><td>Method</td><td>Vol. 5d (%)</td><td>CAGR (%)</td><td>Sharpe</td><td>Max DD (%)</td><td> $\beta _ { \mathrm { R U I } }$ </td><td>MCSp</td></tr><tr><td>RIEnet</td><td>11.17</td><td>9.30</td><td>0.814</td><td>-41.3</td><td>0.500</td><td>1.000</td></tr><tr><td>Factor</td><td>14.08</td><td>7.33</td><td>0.580</td><td>-51.5</td><td>0.592</td><td>&lt; 0.001</td></tr><tr><td>Bootstrap</td><td>14.46</td><td>7.22</td><td>0.563</td><td>-49.4</td><td>0.595</td><td>&lt; 0.001</td></tr><tr><td>Anderson</td><td>14.74</td><td>7.55</td><td>0.578</td><td>-50.8</td><td>0.603</td><td>&lt; 0.001</td></tr><tr><td>QIS</td><td>15.54</td><td>8.07</td><td>0.560</td><td>-50.3</td><td>0.799</td><td>&lt; 0.001</td></tr><tr><td>MLE</td><td>16.25</td><td>2.75</td><td>0.248</td><td>-60.3</td><td>0.683</td><td>&lt; 0.001</td></tr><tr><td>Ridge</td><td>18.07</td><td>3.27</td><td>0.272</td><td>-70.0</td><td>0.671</td><td>&lt; 0.001</td></tr><tr><td>Equal Weight</td><td>20.90</td><td>7.53</td><td>0.440</td><td>-58.5</td><td>1.125</td><td>&lt; 0.001</td></tr></table>

Let $\Delta S _ { t i }$ denote the signed closing-auction order in shares, $\langle V _ { t i } \rangle _ { 1 0 \mathrm { d } }$ the strict-prior ten-day average daily volume, and $P _ { t i }$ the raw closing price. Market impact changes the execution price according to

$$
P _ { t i } ^ { \mathrm { e x e c } } = P _ { t i } \left[ 1 + \mathrm { s i g n } ( \Delta S _ { t i } ) \kappa _ { g ( i , t ) } \sqrt { \frac { | \Delta S _ { t i } | } { \langle V _ { t i } \rangle _ { 1 0 d } } } \right] ,\tag{18}
$$

followed by adverse rounding to the observed price precision. The group �(�, �) is defined by the point-in-time size bucket and listing exchange. The NYSE-like coeficients � are 0.45%, 0.74%, and 1.82% for large, small, and micro stocks; Nasdaq adds 0.57, 0.51, and 2.66 percentage points, respectively [7]. These coeficients enter only the ex post execution simulation and do not afect covariance estimation, optimization, or target-weight formation.

## 5 Results

Table 1 reports out-of-sample performance after broker-simulated execution and all modeled trading costs. Five-session realized volatility is the primary endpoint because it is aligned with the horizon of the training objective and the portfolio rebalancing frequency. RIEnet attains the lowest annualized volatility, 11.17%, compared with 14.08% for Factor, the second-best covariance estimator. This corresponds to a 20.7% reduction. We assess the statistical significance of the realized-risk diferences using the Model Confidence Set (MCS) of Hansen et al. [20] with a stationary block bootstrap [21]. Using 10,000 bootstrap replications, RIEnet is the only method retained at the 0.1% significance level for the baseline expected block length of 11 and for all sensitivity values, 5, 10, 20, and 40 blocks.

The secondary performance measures show that the reduction in realized risk is not obtained by sacrificing portfolio growth. RIEnet achieves the highest CAGR, 9.30%, and the highest Sharpe ratio, 0.814, compared with 8.07% and 0.560 for QIS and 7.33% and 0.580 for Factor. Relative to the next-highest Sharpe ratio, the improvement is approximately 40%. RIEnet also records the least severe maximum drawdown among the covariance estimators, at −41.3%, compared with values between −49.4% and −70.0% for the alternatives. Figure 1 shows that these aggregate diferences are not generated by a few favorable episodes. RIEnet maintains a stable performance advantage across the full 26-year experiment and through the principal market drawdowns represented in the sample. The separation becomes particularly pronounced in the final years and does not exhibit the late-sample deterioration that would indicate temporal overfitting. This evidence is especially relevant for a machine-learning estimator, since instability under changing market conditions would be expected to become most visible in the most recent out-of-sample years [22].

RIEnet also has the lowest sensitivity to the Russell 1000 among the evaluated covariance estimators. Its five-session beta is 0.500, compared with 0.592 for Factor, 0.799 for QIS, and 1.125 for Equal Weight. The improvement in realized risk and compounded performance is therefore not obtained by mechanically reproducing the exposure of a broad capitalization-weighted equity index.

The portfolio-concentration and execution diagnostics clarify how these statistical gains translate into implementation costs. Table 2 reports the median efective portfolio size computed from target weights, together with annualized turnover, market impact, explicit fees, and their estimated efect on compounded performance. RIEnet has a median efective size of 57.6 positions, making it substantially more difuse than MLE, Bootstrap, Anderson, Factor, and Ridge, but considerably more concentrated than QIS, whose median efective size is 754.1. RIEnet turns over 31.53 times per year and incurs 2.57 basis points of market impact and 0.92 basis points of explicit fees per dollar traded, corresponding to an estimated total CAGR drag of 1.49 percentage points.

![](images/a67089b8ecba770d27af9041b6e134fc9ffa477b049d7e5cacae3a1603731ab9.jpg)  
Figure 1: NLV under broker-simulated closing-auction execution on a logarithmic scale. QIS represents complete-row RMT shrinkage, Anderson nearest-correlation projection, Factor incomplete-panel imputation, and Equal Weight the non-estimation benchmark. Curves are sampled at the five-session rebalancing frequency; reported statistics use daily data.

Table 2: Execution diagnostics. The $n _ { \mathrm { e f f } }$ column reports the median efective number of assets across rebalancing dates [23]. Turnover is annualized gross traded notional divided by pre-trade NLV; impact and fees are basis points per dollar traded; CAGR drags are annual percentage points. Covariance estimators are ordered by increasing total drag; Equal Weight is reported separately. Boldface identifies the best optimized-portfolio value in each execution-cost column.
<table><tr><td>Method</td><td> $n _ { \mathrm { e f f } }$ </td><td>Turnover Impact</td><td></td><td></td><td>Fees Impact drag Total drag</td><td></td></tr><tr><td>QIS</td><td>754.1</td><td>24.60</td><td>0.87</td><td>2.48</td><td>0.27</td><td>1.17</td></tr><tr><td>RIEnet</td><td>57.6</td><td>31.53</td><td>2.57</td><td>0.92</td><td>1.09</td><td>1.49</td></tr><tr><td>Factor</td><td>22.5</td><td>19.24</td><td>12.33</td><td>1.41</td><td>2.74</td><td>3.06</td></tr><tr><td>Ridge</td><td>14.6</td><td>32.58</td><td>7.85</td><td>1.49</td><td>2.93</td><td>3.46</td></tr><tr><td>MLE</td><td>34.7</td><td>59.50</td><td>3.72</td><td>1.33</td><td>2.79</td><td>3.68</td></tr><tr><td>Anderson</td><td>19.1</td><td>26.07</td><td>11.55</td><td>1.29</td><td>3.50</td><td>3.90</td></tr><tr><td>Bootstrap</td><td>19.7</td><td>28.76</td><td>10.77</td><td>1.27</td><td>3.60</td><td>4.03</td></tr><tr><td>Equal Weight 1496.8</td><td></td><td>3.93</td><td>0.32 17.79</td><td></td><td>0.01</td><td>1.01</td></tr></table>

QIS has the lowest total drag among the optimized portfolios, at 1.17 percentage points, consistent with its broad allocation and smaller individual orders. RIEnet is therefore not the least costly estimator to implement, but its total drag remains well below those of MLE, Bootstrap, Anderson, Factor, and Ridge, which range from 3.06 to 4.03 percentage points. The results show that low turnover does not necessarily imply low implementation costs, since concentrated trades can generate substantial market impact. Execution costs therefore depend jointly on portfolio breadth, turnover, and order concentration. RIEnet preserves its realized-risk and risk-adjusted-performance advantage after broker-simulated execution while incurring substantially lower implementation costs than the more concentrated incomplete-panel estimators.

## 6 Discussion and Conclusions

Incomplete return panels arise naturally in equity universes that include recent listings, lower-capitalization securities, and histories of unequal length. Pairwise-complete estimation preserves this information, but introduces heterogeneous sampling uncertainty and may produce indefinite correlation matrices. The proposed estimator addresses both issues jointly by conditioning a rotation-invariant spectral map on factor-aligned overlap information and transforming the signed spectrum into a valid, regularized covariance estimate.

The empirical results indicate that this treatment is economically relevant. RIEnet achieves the lowest realized risk and the highest risk-adjusted performance among the evaluated covariance estimators, remains the only method retained by the Model Confidence Set, and preserves its advantage after the modeled execution costs. These gains are not obtained by collapsing toward a broadly diversified allocation: the resulting portfolios remain substantially more selective than QIS and exhibit the lowest Russell 1000 beta among the considered covariance estimators.

The empirical analysis evaluates the fixed RIEnet specification within the full portfolio-construction and execution pipeline. Extended component-level attribution is most cleanly conducted first on controlled synthetic benchmarks, where the population covariance and missingness mechanism are known and architectural efects can be isolated without repeatedly reusing the historical backtest [24, 25]. We leave this broader analysis to future work.

## References

[1] Robert F. Stambaugh. Analyzing investments whose histories difer in length. Journal ofFinancial Economics, 45 (3):285–331, 1997. doi: 10.1016/S0304-405X(97)00020-2.

[2] Nicholas J. Higham. Computing the nearest correlation matrix—a problem from finance. IMA Journal of Numerical Analysis, 22(3):329–343, 2002. doi: 10.1093/imanum/22.3.329.

[3] Robert B. Gramacy, Joo Hee Lee, and Ricardo Silva. On estimating covariances between many assets with histories of highly variable length. arXiv preprint arXiv:0710.5837, 2008. doi: 10.48550/arXiv.0710.5837. URL https://arxiv.org/abs/0710.5837.

[4] Ercument Cahan, Jushan Bai, and Serena Ng. Factor-based imputation of missing values and covariances in panel data of large dimensions. Journal ofEconometrics, 233(1):113–131, 2023. doi: 10.1016/j.jeconom.2022.01.006.

[5] Joël Bun, Jean-Philippe Bouchaud, and Marc Potters. Cleaning large correlation matrices: Tools from Random Matrix Theory. Physics Reports, 666:1–109, 2017. doi: 10.1016/j.physrep.2016.10.005.

[6] Christian Bongiorno, Efstratios Manolakis, and Rosario N. Mantegna. End-to-end large portfolio optimization for variance minimization with neural networks through covariance cleaning. The Journal ofFinance and Data Science, 12:100179, 2026. doi: 10.1016/j.jfds.2026.100179. URL https://doi.org/10.1016/j.jfds.2026.100179.

[7] Amit Goyal, Narasimhan Jegadeesh, and Yanbin Wu. Price impact in closing auctions, opening auctions, and continuous markets: A benchmark for cost of trading on anomalies. Journal of Financial and Quantitative Analysis, pages 1–36, 2026. doi: 10.1017/S0022109026102592. First View.

[8] Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge University Press, 2 edition, 2013. ISBN 9781139020411. doi: 10.1017/CBO9781139020411.

[9] Zhidong Bai and Jack W. Silverstein. Spectral Analysis ofLarge Dimensional Random Matrices. Springer Series in Statistics. Springer, New York, 2 edition, 2010. doi: 10.1007/978-1-4419-0661-8.

[10] Christian Bongiorno, Efstratios Manolakis, and Rosario Nunzio Mantegna. Neural network-driven volatility drag mitigation under aggressive leverage. In Proceedings of the 6th ACM International Conference on AI in Finance, ICAIF ’25, pages 449–455, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400722202. doi: 10.1145/3768292.3770370. URL https://doi.org/10.1145/3768292.3770370.

[11] Junhyeong Lee, Haeun Jeon, Hyunglip Bae, and Yongjae Lee. Return prediction for mean-variance portfolio selection: How decision-focused learning shapes forecasting models. In Proceedings ofthe 6th ACM International Conference on AI in Finance, ICAIF ’25, pages 114–122, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400722202. doi: 10.1145/3768292.3770423. URL https://doi.org/10.1145/ 3768292.3770423.

[12] MSCI Inc. MSCI Global Investable Market Indexes Methodology. Index methodology, MSCI Inc., May 2026. URL https://www.msci.com/eqb/methodology/meth\_docs/MSCI\_GIMIMethodology\_May2026.pdf.

[13] Andrea Frazzini, Ronen Israel, and Tobias J. Moskowitz. Trading costs. SSRN Working Paper 3229719, Social Science Research Network, 2018. URL https://ssrn.com/abstract=3229719.

[14] Donald G. Anderson. Iterative procedures for nonlinear integral equations. Journal ofthe ACM, 12(4):547–560, 1965. doi: 10.1145/321296.321305.

[15] Christian Bongiorno. Bootstraps regularize singular correlation matrices. Journal of Computational and Applied Mathematics, 449:115958, 2024. doi: 10.1016/j.cam.2024.115958.

[16] T. W. Anderson. Maximum likelihood estimates for a multivariate normal distribution when some observations are missing. Journal ofthe American Statistical Association, 52(278):200–203, 1957. doi: 10.1080/01621459. 1957.10501379.

[17] Olivier Ledoit and Michael Wolf. Quadratic shrinkage for large covariance matrices. Bernoulli, 28(3):1519–1547, 2022. doi: 10.3150/20-BEJ1315.

[18] Interactive Brokers. Commissions: Stocks, ETFs, warrants and structured products. https://www. interactivebrokers.com/en/pricing/commissions-stocks.php, 2026. Accessed: 2026-07-23.

[19] Board of Governors of the Federal Reserve System (US). Federal funds efective rate [DFF]. Retrieved from FRED, Federal Reserve Bank of St. Louis, https://fred.stlouisfed.org/series/DFF, 2026. Accessed: 2026-07-23.

[20] Peter Reinhard Hansen, Asger Lunde, and James M. Nason. The model confidence set. Econometrica, 79(2): 453–497, 2011. doi: 10.3982/ECTA5771.

[21] Dimitris N. Politis and Joseph P. Romano. The stationary bootstrap. Journal ofthe American Statistical Association, 89(428):1303–1313, 1994. doi: 10.1080/01621459.1994.10476870.

[22] Thomas Fischer and Christopher Krauss. Deep learning with long short-term memory networks for financial market predictions. European Journal ofOperational Research, 270(2):654–669, 2018. doi: 10.1016/j.ejor.2017.11.054.

[23] Walt Woerheide and Don Persson. An index of portfolio diversification. Financial Services Review, 2(2):73–85, 1992. doi: 10.1016/1057-0810(92)90003-U.

[24] Shihao Gu, Bryan Kelly, and Dacheng Xiu. Empirical asset pricing via machine learning. The Review ofFinancial Studies, 33(5):2223–2273, 2020. doi: 10.1093/rfs/hhaa009.

[25] Halbert White. A reality check for data snooping. Econometrica, 68(5):1097–1126, 2000. doi: 10.1111/1468-0262. 00152.