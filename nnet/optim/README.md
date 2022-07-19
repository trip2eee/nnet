# nnet.optim

Optimizers are implemented.

## SGD
$\mu$: momentum (default: 0)

$\tau$: dampening (default: 0)

$\theta$: parameters (weight and bias).

$\gamma$: learning rate


---

$b_t \leftarrow \mu b_{t-1} + (1 - \tau) g_t$

$g_t \leftarrow b_t$

$\theta_t \leftarrow  \theta_{t-1} - \gamma g_t$


---

## Adam

$m_t$: first momentum

$v_t$: second momentum

$\epsilon$ : default: $10^{-8}$

$\theta $: parameters (weight and bias).

$\gamma $: learning rate

---

- if t = 0

  - $m_0 \leftarrow 0$

  - $v_0 \leftarrow 0$

- else

  - $m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1)g_t$
  - $v_t \leftarrow \beta_2 v_{t-1} + (1-\beta_2)g^2_t$

- $\hat{m_t} \leftarrow \frac{m_t}{1 - \beta^t_1}$
- $\hat{v_t} \leftarrow \frac{v_t}{1 - \beta^t_2}$
- $\theta_t \leftarrow \theta_{t-1} - \gamma \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} g_t$

---

## Regularization
### L1 Regularization
The mean of absolute values of weights are added to the loss.

$$ L = L_0 + \frac{ \lambda_1 } {n} \sum \left| w \right| $$

$$ \frac{\partial L}{\partial w} = \frac{\lambda_1}{n}\sum sign\left(w\right) $$
$$ g_t \leftarrow g_t + \frac{\lambda_1}{n}\sum sign\left(w\right) $$

### L2 Regularization
$$ L = L_0 + \frac{ \lambda_2 } {2n} \sum \frac{1}{2} w^2 $$

$$ \frac{\partial L}{\partial w} = \frac{\lambda_2}{n}\sum w $$
$$ g_t \leftarrow g_t + \frac{\lambda_2}{n}\sum w $$