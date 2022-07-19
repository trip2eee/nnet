# nnet.nn

## Sigmoid
### Forward propagation
$$ s(x) = \frac{1}{ 1 + e^{-x} } $$

### Backward propagation
$$ s(x) = \frac{1}{ 1 + e^{-x} } $$

$$ \frac{\partial{s}}{\partial{x}} = \frac{ -(1+e^{-x})' } { (1+e^{-x})^2 }$$
$$ = \frac{ e^{-x} } { (1+e^{-x})^2 } $$

$$ = \frac{ (1+e^{-x}) - 1 }{ (1 + e^{-x})^2 } $$
$$ = \frac{1}{ 1+e^{-x} } \cdot \left( 1 - \frac{1}{ 1 + e^{-x} }  \right)$$
$$ = s(x) \cdot (1 - s(x)) $$


## Linear
### Forward propagation
$Y = XW + B$

### Backward propagation
Let the output of the $j$-th element of the $i$-th batch is computed as follows.

$$ y_{ij} = \sum_k {x_{ik} \cdot w_{kj}} + b_j $$

We take the propagated gradient with shape (I x J) as an input.

$$g_{i,j} = \frac{ \partial{L} } {\partial{y_{ij}}} $$

Then the gradients of the loss with respect to the weights can be computed by the chain rule.

$$\frac{\partial{L}}{\partial{w_{kj}}} = \frac{\partial{L}}{\partial{y_{ij}}} \frac{\partial{{y_{ij}}}}{\partial{w_{kj}}} = g_{i,j} \frac{\partial{{y_{ij}}}}{\partial{w_{kj}}}$$



Since $y_{ij} = x_{i0}w_{0j} + x_{i1}w_{1j} + ... + x_{ik}w_{kj} + b_j$, the gradient of $y_{ij}$ with respect to the weight can be computed as follows.

$$ \frac{\partial{{y_{ij}}}}{\partial{w_{kj}}} = x_{i,k}$$


To update weight, the shape of the gradient has to be the same with the weight (K x J). This can be achieved by transposing $X$, which is of shape (I x K).

$$ \frac{\partial{L}}{\partial{W}} = X^T G $$

where $X^T$ is (K x I).


If we compute gradient of the loss with respect to the bias,

$$ \frac{\partial{{y_{ij}}}}{\partial{b_{j}}} = 1 $$

$$ \frac{\partial{L}}{\partial{b_j}} = \frac{\partial{L}}{\partial{y_{ij}}} \frac{\partial{{y_{ij}}}}{\partial{b_j}} = \sum_i g_{i,j} $$

Since $y_{ij} = \sum_k x_{ik} w_{kj}$,  the derivative of loss with respect to input $x$ is as follows.

$$ \frac{\partial{L}}{\partial{x_{ik}}} = \frac{\partial{L}}{\partial{y_{ij}}} \frac{\partial{{y_{ij}}}}{\partial{x_{ik}}} = g_{i,j} \frac{\partial{{y_{ij}}}}{\partial{x_{ik}}} = g_{i,j}w_{k,j} $$

The equation can be represented as matrix multiplication.

$$ \frac{\partial{L}}{\partial{x}} = G W^T $$

## Conv2d

### Forward propagation
If we convolve a kernel $k$ over an image $x$, the $m$-th channel value at $(r,c)$ is as follows.

$$ y_{n,r,c,m} = \sum_i^{kh} \sum_j^{kw} \sum_k^{xch} = k_{i,j,k} x_{n,r+i-bh,c+j-bw,k} + b_m $$

where $bh = \lfloor kh/2 \rfloor$ and $bw = \lfloor kw/2 \rfloor$

### Backward propagation

The goal of backward propagation is to compute the gradient of loss with respect to the kernel.

$$ \frac{\partial{L}}{\partial{k_{i,j,k}}} = \frac{\partial{L}}{\partial{y_{n,r,c,m}}} \frac{\partial{{y_{n,r,c,m}}}}{\partial{k_{i,j,k}}} = g_{n,r,c,m} \frac{\partial{{y_{n,r,c,m}}}}{\partial{k_{i,j,k}}} $$

The gradient of $y$ with respect to the kernel $k$ is computed as follows.

$$ \frac{\partial{{y_{n,r,c,m}}}}{\partial{k_{i,j,k}}} =  \sum_i^{kh} \sum_j^{kw} \sum_k^{xch} = x_{n,r+i-bh,c+j-bw,k} $$


The gradient of $y$ is of shape (R x C x M). The shape of $x$ is (R x C x XCH). To update kernel, we should have the gradient of shape (KH x KW x XCH x M), which is the shape of the kernel. 

$$ \frac{\partial{L}}{\partial{k_{i,j,k}}} = \sum_r \sum_c g_{n,r,c,m} \cdot x_{n,r+i-bh,c+j-bw,k} $$


The gradient of loss with respect to bias is computed as follows.

$$ \frac{\partial{L}}{\partial{b_m}} = \frac{\partial{L}}{\partial{y_{n,r,c,m}}} \frac{\partial{{y_{n,r,c,m}}}}{\partial{b_m}} = g_{n,r,c,m} \frac{\partial{{y_{n,r,c,m}}}}{\partial{b_m}} $$


$$ \frac{\partial{{y_{n,r,c,m}}}}{\partial{b_m}} = 1$$

Therefore the gradient of loss with respect to bias is summation of gradient over $(n, r, c)$.

$$ \frac{\partial{L}}{\partial{b_m}} = \sum_n \sum_r \sum_c g_{n,r,c,m} $$



## BatchNorm2d
### Forward propagation


During training, moving average and variance of input $x$ is computed.

$$ \hat\mu_{x,t+1} \leftarrow \hat\mu_{x,t} + \alpha \left( \mu_{x, t+1} - \hat\mu_{x,t} \right) $$

$$ \hat\sigma^2_{x,t+1} \leftarrow \hat\sigma^2_{x,t} + \alpha \left( \sigma^2_{x, t+1} - \hat\sigma^2_{x,t} \right) $$

Normalize image over channel.

$$ y = \gamma \frac{x - \mu_x}{\sqrt{\sigma^2_x + \epsilon}} + \beta $$


### Backward propagation
$$ \frac{\partial y}{\partial \gamma} = \hat x $$

where 

$$ \hat x = \frac{x-\mu_x}{\sqrt{\sigma^2_x + \epsilon}} $$

$$ \frac{\partial y}{\partial \beta} = 1 $$

$$ \frac{\partial y}{\partial x} = \frac{1}{\sqrt{\sigma^2_x + \epsilon}} $$


## Dropout
### Forward propagation
During randomly generate mask $m$ with zeros with probability p. Then the output is computed as follows.

$$ y = m\frac{x}{1 - p} $$

### Backward propagation

$$ y = m\frac{x}{1 - p} $$
$$ \frac{\partial y}{\partial x} = \frac{m}{1-p} $$
$$ G_x = \frac{m}{1-p}G_y $$
