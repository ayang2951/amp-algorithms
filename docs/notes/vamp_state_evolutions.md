## VAMP State Evolutions

We shall use the same setting, i.e. 
$$Y = X \beta_0 + \epsilon$$
where $X \in \mathbb R^{n \times p}$ is right rotationally invariant, ${\beta_0}\_j \overset{iid}\sim \pi_\beta$ for $j \in [p]$, and $\epsilon_i \overset{iid}\sim \mathcal N(0, \sigma^2)$ for $i \in [n]$.

Recall that $X$ is right rotationally invariant, so its SVD can be written as $U \Sigma V^\top$, where $V$ is Haar distributed.

### Bayes VAMP State Evolution

We now give the Bayes VAMP state evolution recursion. Let $s \in \mathbb R^p$ be the vector of top $p$ singular values of design $X$, and let $S$ denote its empirical distribution in the $p$ limit. For the prior on the true signal $\beta_0$ being $\pi_{\beta}$, we shall write the state evolution updates using this distribution. 

<div class="callout algorithm"><span class="label">Algorithm: Bayes VAMP SE Recursion</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For Bayes VAMP, the state evolution recursion is defined by the following updates, in terms of the prior $\pi_\beta$ and the function 
$$f_k(P_k) \triangleq \mathbb E\left[\bar\beta \ | \ \bar\beta + \mathcal N(0, \frac{1}{{\bar\gamma}_{1k}}) = (P_k + \beta)\right] - \beta,$$
where $\beta \sim \pi_\beta$.

<ol type="1">
  <li>${\bar b}_k \triangleq \mathbb E_{P_k \otimes \beta}\left[f_k'(P_k)\right]$</li>
  <li>$\kappa_{2k}^2 \triangleq \left(\frac{1}{1 - \bar b_k}\right)^2 \left(\mathbb E_{P_k \otimes \beta}[f_k^2 (P_k)] - \bar b_k^2 \cdot \kappa_{1k}^2\right)$</li>
  <li>$\bar \gamma_{2k} \triangleq \bar\gamma_{1k} \left(\frac{1}{\bar b_k} - 1\right)$</li>
  <li>$\bar c_k \triangleq \alpha_{k} - \bar c_k \cdot \kappa^2_{2k}$</li>
  <li>$\kappa_{1, k + 1}^2 \triangleq \left(\frac{1}{1 - \bar c_k}\right)^2 \left(\omega_{1k} \cdot \sigma^2 + \omega_{2k} \cdot \kappa_{2k}^2 - \bar c_k^2 \cdot \kappa_{2k}^2\right)$</li>
  <li>$\gamma_{1, k + 1} \triangleq \gamma_{2k} \left(\frac{1}{\bar c_k} - 1\right)$</li>
</ol>

where, if a closed-form solution doesn't exist, we use Monte Carlo sampling to compute $\mathbb E_{P_k \otimes \beta}\left[f_k'(P_k)\right]$ and $\mathbb E_{P_k \otimes \beta}[f_k^2 (P_k)]$. Defining $S$ to be the limit in empirical convergence of the vector of singular values of design matrix $X$, we also use Monte Carlo to compute the quantity
$$\alpha_k \triangleq \mathbb E \left[\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right]$$
for the update of $\bar c_k$ update and quantities
$$\omega_{1k} \triangleq \mathbb E\left[\left(\frac{S}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2\right] \qquad \text{and} \qquad \omega_{2k} \triangleq \mathbb E\left[\left(\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2\right]$$
for the $\kappa_{1k}^2$ update.
</div>



<details class="collapsible">
<summary>Derivation: Explicit Updates</summary>
<div class="collapsible__content">
First, we define a few quantities and functions required for the state evolution. Let us define $\xi \triangleq U^\top \epsilon$ and $s \in \mathbb R^p$ to be the diagonal of the top $p$ singular values of $X$. Define $w \triangleq [\zeta, s]^\top \in \mathbb R^{2 \times p}$.

For fixing $S$ as the limiting empirical distribution in Wasserstein 2 of the vector $s$ and $\Xi$ as the limiting distribution of the components of $\xi$, we can define the following functions (note that the capital letters denote randomness; the actual updates using $f$ and $g$ will require taking an expectation over the functions):
$$f_k(P_k) \triangleq \mathbb E\left[\bar\beta \ | \ \bar\beta + \mathcal N(0, \frac{1}{{\bar\gamma}\_{1k}}) = (P_k + \beta)\right] - \beta,$$
where $\beta \sim \pi_\beta$ and the component-wise function over $[\zeta, s]$ to be
$$g(Q_k) = \frac{S \cdot \Xi + \sigma^2{\bar\gamma}\_{2k} Q_k}{S^2 + \sigma^2 {\bar\gamma}_{2k}},$$
and the auxiliary functions
$$C(x) \triangleq \frac{1}{1 - x}$$
and
$$\Gamma(x, y) \triangleq x\left(\frac{1}{y} - 1\right).$$

We have the base recursion, after initializing $\bar \gamma_{10}$ as the limit of $\gamma_{10}$ and $R_1$ as ???

<ol type="i">
  <li>$\bar b_k \triangleq \mathbb E[f'_k(P_k)]$</li>
  <li>$\kappa_{2k}^2 \triangleq C^2(\bar b_k) \cdot \bigg(\mathbb E[f^2_k(P_k)] - \bar b_k^2 \kappa_{1k}^2\bigg)$</li>
  <li>${\bar\gamma}_{2k} \triangleq \Gamma({\bar\gamma}_{1k}, \bar b_k)$</li>
  <li>$\bar c_k \triangleq \mathbb E[g'_k(Q_k)] - \bar c_k^2 \kappa_{2k}^2$</li>
  <li>$\kappa^2_{1, k + 1} \triangleq C^2(\bar c_k) \cdot \bigg(\mathbb E\left[g_k^2(Q_k)\right] - \bar c_k^2 \kappa^2_{2k}\bigg)$</li>
  <li>$\bar\gamma_{1, k + 1} \triangleq \Gamma(\bar\gamma_{2k}, \bar c_k)$</li>
</ol>

We can then compute the explicit updates (except for $f_k$, as this depends on the prior) using the functions above. We have, for the updates involving $g_k$,

$$g_k^2(Q_k) = \left(\frac{S}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2 \Xi^2 + \left(\frac{S}{S^2 + \sigma^2 \bar\gamma_{2k}}\right) \left(\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right) \Xi \cdot Q_k + \left(\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2 Q_k^2,$$

and

$$g_k'(P_k) = \frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}.$$

Hence we have the updates

$$\mathbb E[g_k^2(P_k)] = \mathbb E\left[\left(\frac{S}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2\right] \mathbb E \left[\Xi^2\right] + \mathbb E\left[\left(\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2\right] \mathbb E\left[Q_k^2\right],$$

where the cross term disappears because $Q_k$ (and in fact $\Xi$ too) is zero-mean.

Simplifying using that $\mathbb E[Q_k^2] = \kappa_{2k}^2$ and that $\mathbb E[\Xi^2] = \sigma^2$, we obtain, in terms of expectations of functions of $S$ (which are much more difficult to evaluate),

$$\mathbb E[g_k^2(Q_k)] = \mathbb E\left[\left(\frac{S}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2\right] \sigma^2 + \mathbb E\left[\left(\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2\right] \kappa_{2k}^2.$$

For simplicity, let us denote

$$\omega_{1k} \triangleq \mathbb E\left[\left(\frac{S}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2\right]$$
and
$$\omega_{2k} \triangleq \mathbb E\left[\left(\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right)^2\right].$$

Next, for the derivative, we have updates
$$\mathbb E\left[g_k'(Q_k)\right] = \mathbb E \left[\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right].$$

Again for simplicity, let us denote

$$\alpha_k \triangleq \mathbb E \left[\frac{\sigma^2 \bar\gamma_{2k}}{S^2 + \sigma^2 \bar\gamma_{2k}}\right].$$
</div>

</details>




<div class="callout algorithm"><span class="label">Algorithm: Algorithm</span><br/>
<hr style="height:0.01px; visibility:hidden;" />


</div>

<ol type="i">
  <li>.</li>
  <li>.</li>
</ol>