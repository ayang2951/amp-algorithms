## Section 1: VAMP Algorithm Implementations

In the regression setting
$$Y = X \beta_0 + \epsilon$$
where $X \in \mathbb R^{n \times p}$ is right rotationally invariant, ${\beta_0}\_j \overset{iid}\sim \pi_\beta$ for $j \in [p]$, and $\epsilon_i \overset{iid}\sim \mathcal N(0, \sigma^2)$ for $i \in [n]$, we give the derivation and implementations for the Bayes, LASSO, and ridge VAMP algorithms.

### Bayes VAMP

We will consider the following priors:
<ol type="i">
  <li>${\beta_0}_j \sim \mathcal N(0, \tau^2)$,</li>
  <li>${\beta_0}_j \sim \text{Bernoulli}(\theta) \cdot \mathcal N(0, \tau^2)$,</li>
  <li>${\beta_0}_j \sim \text{Rademacher}$,</li>
  <li>${\beta_0}_j \in \{-1, 0, 1\} \text{ with probabilities } (\theta_1, \theta_2, 1 - (\theta_1 + \theta_2))$.</li>
</ol>


Defining the functions 
$$\tilde g_k(\mathbf{r}\_2^k) \triangleq \left(\frac{1}{\sigma^2} X^\top X + \gamma_{2k} \mathbf I_p\right)^{-1} \left(\frac{1}{\sigma^2} X^\top Y + \gamma_{2k} \mathbf{r}\_2^k\right)$$
and Bayes denoiser
$$\tilde f_k(\mathbf{r}^k) \triangleq \mathbb E\left[\bar\beta \ | \ \bar\beta + \mathcal N(0, \frac{1}{\gamma_{1k}}) = \mathbf{r}^k\right]$$
where $\bar\beta \sim \pi_\beta$, we can define the iterates below.

<div class="callout algorithm"><span class="label">Algorithm: Bayes VAMP Algorithm</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
For the initialization, let $\mathbf{r}_1^0 \triangleq 0.01 \cdot \mathbf{1}_p$, and let $\gamma_{10} \triangleq 0.05$.

For the denoiser $\tilde f_k$ suitable for the prior specified $\pi_\beta$ on $\beta_0$, we have updates

<ol type="1">
<li>$\hat\beta^k \triangleq \tilde f_k(\mathbf{r}_1^k)$</li>
<li>$b_k \triangleq \frac{1}{p} \sum_{j = 1}^p \tilde f'_k(\mathbf{r}_{1j}^k)$</li>
<li>$\eta_{k} \triangleq \frac{\gamma_{1k}}{b_k}$</li>
<li>$\gamma_{2k} \triangleq \eta_k - \gamma_{1k}$</li>
<li>$\mathbf{r}_2^k \triangleq \frac{1}{\gamma_{2k}} \left(\eta_{k} \hat\beta^k - \gamma_{1k} \mathbf{r}_1^k\right)$</li>
<li>$c_k \triangleq \frac{1}{p} \text{Tr} \left[\gamma_{2k} \left(\frac{1}{\sigma^2} X^\top X + \gamma_{2k} \mathbf{I}_p\right)^{-1} \right]$</li>
<li>$\gamma_{1, k + 1} \triangleq \gamma_{2k} \left(\frac{1}{c_k} - 1\right)$</li>
<li>$\mathbf{r}_1^{k + 1} \triangleq \frac{1}{1 - c_k} \left(\left(\frac{1}{\sigma^2} X^\top X + \gamma_{2k} \mathbf I_p\right)^{-1} \left(\frac{1}{\sigma^2} X^\top Y + \gamma_{2k} \mathbf{r}_2^k\right) - c_k \mathbf{r}_2^k\right)$</li>
</ol>

The specific implementation, i.e. computing the means, inverses, and traces, can be found in the code.

Further, small constants have been added for numerical stability to prevent overflow. Also for numerical stability, for discrete priors, the algorithm terminates early if the estimates are close to some element in the support.
</div>

<details class="collapsible">
<summary>Derivation: Explicit Updates</summary>
<div class="collapsible__content">
We use the starter base algorithm and derive out the updates.
  
The base algorithm is given as, after we initialize $\mathbf{r}\_1^0$ and $\gamma_{10} > 0$, the following updates.
<ol type="1">
<li>$\hat\beta^k \triangleq \tilde f_k(\mathbf{r}_1^k)$</li>
<li>$b_k \triangleq \frac{1}{p} \sum_{j = 1}^p \tilde f'_k(\mathbf{r}_{1j}^k)$</li>
<li>$\eta_{k} \triangleq \frac{\gamma_{1k}}{b_k}$</li>
<li>$\gamma_{2k} \triangleq \eta_k - \gamma_{1k}$</li>
<li>$\mathbf{r}_2^k \triangleq \frac{1}{\gamma_{2k}} \left(\eta_{k} \hat\beta^k - \gamma_{1k} \mathbf{r}_1^k\right)$</li>
<li>$c_k \triangleq \frac{1}{p} \text{Tr} \left[\frac{\partial}{\partial \mathbf{r}_2^k} \tilde{g}_k(\mathbf{r}_2^k)\right]$</li>
<li>$\gamma_{1, k + 1} \triangleq \gamma_{2k} \left(\frac{1}{c_k} - 1\right)$</li>
<li>$\mathbf{r}_1^{k + 1} \triangleq \frac{1}{1 - c_k} \left(\tilde g_k(\mathbf{r}_2^k) - c_k \mathbf{r}_2^k\right)$</li>
</ol>


For $\tilde f_k$, we need to compute the explicit form of the conditional expectation, which is fortunately simple using Tweedie's formula. Since we also need the derivative of $\tilde f_k$, we compute the second derivatives of the marginal densities here as well.
<ol type="i">
<li>For the Gaussian prior ${\beta_0}_j \sim \mathcal N(0, \tau^2)$, we have the simple posterior mean form
$$\tilde f_k(\mathbf{r}_1^k) = \frac{\gamma_{1k} \tau^2}{1 + \gamma_{1k} \tau^2} \mathbf{r}_1^k.$$
Hence, the derivative of the conditional expectation is simply
$$\tilde f'_k(\mathbf{r}_1^k) = \frac{\gamma_{1k} \tau^2}{1 + \gamma_{1k} \tau^2}.$$
</li>
<li>For the Bernoulli-Gaussian prior ${\beta_0}_j \sim \text{Bernoulli}(\theta) \cdot \mathcal N(0, \tau^2)$, we have the form
$$\tilde f_k(\mathbf{r}_1^k) = \mathbf{r}_1^k + \frac{1}{\gamma_{1k}} \frac{h_k'(\mathbf{r}_1^k)}{h_k(\mathbf{r}_1^k)}$$
where $h_k(\mathbf{r}_1^k)$ is the marginal density of (abusing notation) $\bar\beta + \mathcal N(0, \frac{1}{\gamma_{1k}})$, which comes out to be, abusing notation by notation the density as the distribution,
$$h_k(\mathbf{r}_1^k) = \theta \cdot \mathcal N(0, \tau^2 + \frac{1}{\gamma_{1k}}) + (1 - \theta) \cdot \mathcal N(0, \frac{1}{\gamma_{1k}}),$$
and the derivative is
$$h'_k(\mathbf{r}_1^k) = \theta \cdot \mathcal N\left(0, \tau^2 + \frac{1}{\gamma_{1k}}\right) \cdot \left(-\frac{1}{\tau^2 + \frac{1}{\gamma_{1k}}} \mathbf{r}_1^k\right) + (1 - \theta) \cdot \mathcal N\left(0, \frac{1}{\gamma_{1k}}\right) \cdot \left(-\gamma_{1k} \mathbf{r}_1^k\right).$$
The second derivative, which we need for the derivative of the conditional expectation (with the quotient rule), is
$$h''_k(\mathbf{r}_1^k) = \theta \cdot \mathcal N\left(0, \tau^2 + \frac{1}{\gamma_{1k}}\right) \cdot \left(-\frac{1}{\tau^2 + \frac{1}{\gamma_{1k}}} + \left(\frac{\mathbf{r}_1^k}{\tau^2 + \frac{1}{\gamma_{1k}}}\right)^2\right) + (1 - \theta) \cdot \mathcal N\left(0, \frac{1}{\gamma_{1k}}\right) \cdot \left(-\gamma_{1k} + (\gamma_{1k}\mathbf{r}_1^k)^2\right).$$
</li>
<li>For the Rademacher prior, we know the form of the conditional expectation, which is
$$f_k(\mathbf{r}_1^k) = \tanh(\gamma_{1k} \mathbf{r}_1^k).$$
The derivative of the conditional expectation is therefore
$$f'_k(\mathbf{r}_1^k) = \gamma_{1k} \cdot \text{sech}^2(\gamma_{1k} \mathbf{r}_1^k).$$
</li>
<li>For the three-point prior ${\beta_0}_j \in \{-1, 0, 1\} \text{  with probabilities  } (\theta_1, \theta_1, 1 - (\theta_1 + \theta_2))$, using again Tweedie's formula
$$\tilde f_k(\mathbf{r}_1^k) = \mathbf{r}_1^k + \frac{1}{\gamma_{1k}} \frac{h_k'(\mathbf{r}_1^k)}{h_k(\mathbf{r}_1^k)},$$
we have, notating $\theta_3 = 1 - \theta_1 - \theta_2$,
$$h_k(\mathbf{r}_1^k) = \theta_1 \cdot \mathcal N\left(-1, \frac{1}{\gamma_{1k}}\right) + \theta_2 \cdot \mathcal N\left(0, \frac{1}{\gamma_{1k}}\right) + \theta_3 \cdot \mathcal N\left(1, \frac{1}{\gamma_{1k}}\right).$$
The derivative is therefore
$$h'_k(\mathbf{r}_1^k) = -\gamma_{1k} \left[\theta_1 \cdot \mathcal N\left(-1, \frac{1}{\gamma_{1k}}\right) \cdot \left(\mathbf{r}_1^k + 1\right) + \theta_2 \cdot \mathcal N\left(0, \frac{1}{\gamma_{1k}}\right) \cdot \left(\mathbf{r}_1^k \right)+ \theta_3 \cdot \mathcal N\left(1, \frac{1}{\gamma_{1k}}\right) \cdot \left(\mathbf{r}_1^k - 1\right)\right].$$
And the second derivative is
$$h''_k(\mathbf{r}_1^k) = -\gamma_{1k} \left[\theta_1 \cdot \mathcal N\left(-1, \frac{1}{\gamma_{1k}}\right) \cdot \left(1 - \gamma_{1k}(\mathbf{r}_1^k + 1)^2\right) + \theta_2 \cdot \mathcal N\left(0, \frac{1}{\gamma_{1k}}\right) \cdot \left(1 - \gamma_{1k} (\mathbf{r}_1^k)^2 \right)+ \theta_3 \cdot \mathcal N\left(1, \frac{1}{\gamma_{1k}}\right) \cdot \left(1 - \gamma_{1k}(\mathbf{r}_1^k - 1)^2\right)\right].$$
</li>
</ol>

We can now compute out the updates with the $\tilde g_k$ denoiser. We have that
$$\frac{\partial}{\partial \mathbf{r}\_1^k} \tilde g_k(\mathbf{r}\_1^k) = \gamma_{2k} \left(\frac{1}{\sigma^2} X^\top X + \gamma_{2k} \mathbf{I}_p\right)^{-1},$$
which we can place inside the updates for $c_k$.
</div>

</details>



### LASSO VAMP

We may similarly obtain the LASSO VAMP implementation, where the only element changed is the denoiser $\tilde f_k$.

<div class="callout algorithm"><span class="label">Algorithm: LASSO VAMP Algorithm</span><br/>
<hr style="height:0.01px; visibility:hidden;" />

For the denoiser $\tilde f_k$ independent of the prior and instead is the proximal operator for the $\ell_1$ penalty with regularizer $\lambda$, we have
$$\tilde f_k(\mathbf{r}\_1^k) \triangleq \begin{cases}
\mathbf{r}\_1^k - \frac{\lambda}{\gamma_{1k}} & \qquad \mathbf{r}\_1^k > \frac{\lambda}{\gamma_{1k}}
\\\\
0 & \qquad |\mathbf{r}\_1^k| \leq \frac{\lambda}{\gamma\_{1k}}
\\\\
\mathbf{r}\_1^k + \frac{\lambda}{\gamma\_{1k}} & \qquad \mathbf{r}\_1^k < \frac{\lambda}{\gamma_{1k}}
\end{cases}.$$

<ol type="1">
<li>$\hat\beta^k \triangleq \tilde f^{lasso}_k(\mathbf{r}_1^k)$</li>
<li>$b_k \triangleq \frac{1}{p} \sum_{j = 1}^p {{\tilde f}^{lasso}}'_k(\mathbf{r}_{1j}^k)$</li>
<li>$\eta_{k} \triangleq \frac{\gamma_{1k}}{b_k}$</li>
<li>$\gamma_{2k} \triangleq \eta_k - \gamma_{1k}$</li>
<li>$\mathbf{r}_2^k \triangleq \frac{1}{\gamma_{2k}} \left(\eta_{k} \hat\beta^k - \gamma_{1k} \mathbf{r}_1^k\right)$</li>
<li>$c_k \triangleq \frac{1}{p} \text{Tr} \left[\gamma_{2k} \left(\frac{1}{\sigma^2} X^\top X + \gamma_{2k} \mathbf{I}_p\right)^{-1} \right]$</li>
<li>$\gamma_{1, k + 1} \triangleq \gamma_{2k} \left(\frac{1}{c_k} - 1\right)$</li>
<li>$\mathbf{r}_1^{k + 1} \triangleq \frac{1}{1 - c_k} \left(\left(\frac{1}{\sigma^2} X^\top X + \gamma_{2k} \mathbf I_p\right)^{-1} \left(\frac{1}{\sigma^2} X^\top Y + \gamma_{2k} \mathbf{r}_2^k\right) - c_k \mathbf{r}_2^k\right)$</li>
</ol>

Again, the same methods for numerical stability have been applied to the LASSO VAMP implementation.


</div>

### Ridge VAMP

The Ridge VAMP is an easy extension of LASSO, where we simply use the proximal operator for the $\ell_2$ penalty. 

<div class="callout algorithm"><span class="label">Algorithm: Bayes VAMP Algorithm</span><br/>
<hr style="height:0.01px; visibility:hidden;" />

With regularizer $\lambda$, we have that
$$\tilde f_k(\mathbf{r}\_1^k) \triangleq \frac{1}{1 + \frac{\lambda}{\gamma_{1k}}} \mathbf{r}\_1^k.$$ Hence, the algorithm is also given by the updates below.

<ol type="1">
<li>$\hat\beta^k \triangleq \tilde f^{ridge}_k(\mathbf{r}_1^k)$</li>
<li>$b_k \triangleq \frac{1}{p} \sum_{j = 1}^p {{\tilde f}^{ridge}}'_k(\mathbf{r}_{1j}^k)$</li>
<li>$\eta_{k} \triangleq \frac{\gamma_{1k}}{b_k}$</li>
<li>$\gamma_{2k} \triangleq \eta_k - \gamma_{1k}$</li>
<li>$\mathbf{r}_2^k \triangleq \frac{1}{\gamma_{2k}} \left(\eta_{k} \hat\beta^k - \gamma_{1k} \mathbf{r}_1^k\right)$</li>
<li>$c_k \triangleq \frac{1}{p} \text{Tr} \left[\gamma_{2k} \left(\frac{1}{\sigma^2} X^\top X + \gamma_{2k} \mathbf{I}_p\right)^{-1} \right]$</li>
<li>$\gamma_{1, k + 1} \triangleq \gamma_{2k} \left(\frac{1}{c_k} - 1\right)$</li>
<li>$\mathbf{r}_1^{k + 1} \triangleq \frac{1}{1 - c_k} \left(\left(\frac{1}{\sigma^2} X^\top X + \gamma_{2k} \mathbf I_p\right)^{-1} \left(\frac{1}{\sigma^2} X^\top Y + \gamma_{2k} \mathbf{r}_2^k\right) - c_k \mathbf{r}_2^k\right)$</li>
</ol>

</div>




