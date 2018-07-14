---
layout: post
title:  "Implementing Variational Inference"
date:   2018-07-08
visible: true
---

{% include mathjs %}

<a href="../07/relearning-variational-inference.html"> << Previous Post</a>

<a href="../07/relearning-variational-inference.html">Previously</a>, I described the purpose and method of variational inference -- a means to learn a distribution's latent variables by finding a proxy distribution for the latent variables $$q(z \vert v)$$ and maximizing the Evidence Lower Bound (ELBO). Now, we'll discuss how we can maximize the ELBO, and then implement what we've learned.

## Mean Field Variational Inference

*Mean Field Variational Inference* is a specific instance of variational inference in which we choose a $$q(z)$$ that *factorizes*. That is,
<div style="align: center;">$$q(z_{1:m}) = \prod^m_{i=1} q(z_i)$$</div>.
In other words, each variable $$z_i$$ is independent. Now let's use some math to simplify the ELBO. Recall:
<div style="align: center;">$$ELBO = E_q[\log p(z, x)] - E_q[\log q(z)]$$</div>
Expressing the joint probability $$p(z, x)$$ in terms of conditional probabilities,
<div style="align: center;">$$\begin{align*}
p(z, x) &= p(x)\prod^m_{i=1} p(z_i \vert z_{1:(i-1)}, x) \\
E_q[\log p(z, x)] &= E_q[\log p(x)\prod^m_{i=1} p(z_i \vert z_{1:(i-1)}, x)] \\
&= E_q[\log p(x) + \sum^m_{i=1} \log p(z_i \vert z_{1:(i-1)}, x)] \\
&= E_q[\log p(x)] + \sum^m_{i=1} E_q[\log p(z_i \vert z_{1:(i-1)}, x)]
\end{align*}$$</div>
Because $$q(z)$$ factorizes, we also have:
<div style="align: center;">$$E_q[\log q(z)] = \sum^m_{i=1} E_q[\log q(z_i)]$$</div>
Substituting both of these into the ELBO, we get
<div style="align: center;">$$\begin{align*}
ELBO &= E_q[\log p(z, x)] - E_q[\log q(z)] \\
&= E_q[\log p(x)] + \sum^m_{i=1} E_q[\log p(z_i \vert z_{1:(i-1)}, x)] - \sum^m_{i=1} E_q[\log q(z_i)] \\
&= E_q[\log p(x)] + \sum^m_{i=1} (E_q[\log p(z_i \vert z_{1:(i-1)})] - E_q[\log q(z_i)]) 
\end{align*}$$</div>
Consider the ELBO as a function of $$q(z_k)$$. By rearranging how we express $$p(z, x)$$ in conditional probabilities, we can express the ELBO as
<div style="align: center;">$$ELBO = E_q[\log p(z_k \vert z_{-k}, x)] - E_q[\log q(z_k)] + f(z_{-k}, x)$$</div>
where $$f(z_{-k}, x)$$ does not depend on $$z_k$$ and is therefore a constant term. Now, given a fixed distribution for $$q(z_{-k})$$, we can derive the optimal $$q^{\star}(z_k)$$. Re-expressing the expectations, we note that
<div style="align: center;">$$E_q[X] = \int q(z_k) E_{q_{-k}}[X] dz_k$$.</div>
and therefore
<div style="align: center;">$$
\begin{align*}
ELBO &= \int q(z_k)E_{q_{-k}}[\log p(z_k \vert z_{-k}, x)] dz_k - \int q(z_k)E_{q_{-k}}[\log q(z_k)]dz_k + f(z_{-k}, x) \\
%\frac{\partial ELBO}{\partial q(z_k)} &= \int E_{q_{-k}}[\log p(z_k \vert z_{-k}, x)] dz_k - \int (E_{q_{-k}}[\log q(z_k)] + \frac{q(z_k)}{q(z_k)}) dz_k \\
%&= E_{q_{-k}}[\log p(z_k \vert z_{-k}, x)] - E_{q_{-k}}[\log q(z_k)] - 1 \\
%s&= E_{q_{-k}}[\log p(z_k \vert z_{-k}, x)] - q(z_k) - 1
\end{align*}
$$.</div>
When we consider that $$\displaystyle\int q(z_k) dz_k = 1$$, it turns out that we have a optimization problem which allows us to use the integral version of Lagrange multipliers. For those unfamiliar, details about using integral constraints can be found <a href="http://liberzon.csl.illinois.edu/teaching/cvoc/node38.html">here</a> and <a href="http://liberzon.csl.illinois.edu/teaching/cvoc/node39.html">here</a>, and may be covered in this blog at a later date. In essence, for some constant $$\lambda$$, the Lagrangian formulation leads to the condition
<div style="align: center;">$$\begin{align*}
q(z_k)E_{q_{-k}}[\log p(z_k \vert z_{-k}, x)] - q(z_k) E_{q_{-k}}[\log q(z_k)] &= \lambda q(z_k) \\
E_{q_{-k}}[\log p(z_k \vert z_{-k}, x)] - \log q(z_k) &= \lambda \\
q^{\star}(z_k) &= \exp (-\lambda + E_{q_{-k}}[\log p(z_k \vert z_{-k}, x)]) \\
\implies q^{\star}(z_k) & \propto \exp (E_{q_{-k}}[\log p(z_k \vert z_{-k}, x)])
\end{align*}$$</div>
Because $$p(z_k \vert z_{-k}, x) = \displaystyle\frac{p(z_k, z_{-k}, x)}{p(z_{-k}, x)}$$ and $$p(z_{-k}, x)$$ is constant relative to $$z_k$$,
<div style="align: center;">$$\begin{align*}
p(z_k \vert z_{-k}, x) &\propto p(z, x) \\
\implies q^{\star}(z_k) &\propto \exp(E_{q_{-k}}[\log p(z, x)])
\end{align*}$$</div>
Assuming that the normalized distrbution of $$\exp(E_{q_{-k}}[\log p(z, x)])$$ is computable, we now have a means for computing the optimal distribution of $$q(z_k)$$ when the other variables $$z_{-k}$$ are fixed. How do we use this?

### Coordinate Ascent
Work in progress.

## Implementation
An implementation of mean field variational inference on a Bayesian Mixture of Gaussians can be found <a href="https://gist.github.com/yangalexandery/717dcf682f0ab628f43d52f67434d44d">here</a>.