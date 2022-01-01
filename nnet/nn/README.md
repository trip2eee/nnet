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
$ Y = XW + B $

### Backward propagation
Let the output of the $j$-th element of the $i$-th batch is computed as follows.

$$ y_{ij} = \sum_k {x_{ik} \cdot w_{kj}} + b_j $$

We take the propagated gradient with shape (I x J) as an input.

$$g_{i,j} = \frac{ \partial{L} } {\partial{y_{ij}}} $$

Then the gradients of the loss with respect to the weights can be computed by the chain rule.
$$ \frac{\partial{L}}{\partial{w_{kj}}} = \frac{\partial{L}}{\partial{y_{ij}}} \frac{\partial{{y_{ij}}}}{\partial{w_{kj}}} = g_{i,j} \frac{\partial{{y_{ij}}}}{\partial{w_{kj}}} $$



Since $ y_{ij} = x_{i0}w_{0j} + x_{i1}w_{1j} + ... + x_{ik}w_{kj} + b_j $, the gradient of $y_{ij}$ with respect to the weight can be computed as follows.

$$ \frac{\partial{{y_{ij}}}}{\partial{w_{kj}}} = x_{i,k}$$


To update weight, the shape of the gradient has to be the same with the weight (K x J). This can be achieved by transposing $X$, which is of shape (I x K).

$$ \frac{\partial{L}}{\partial{W}} = X^T G $$

where $ X^T $ is (K x I).


If we compute gradient of the loss with respect to the bias,
$$ \frac{\partial{{y_{ij}}}}{\partial{b_{j}}} = 1 $$

$$ \frac{\partial{L}}{\partial{b_j}} = \frac{\partial{L}}{\partial{y_{ij}}} \frac{\partial{{y_{ij}}}}{\partial{b_j}} = \sum_i g_{i,j} $$