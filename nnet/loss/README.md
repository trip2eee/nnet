# nnet.loss

## Mean Squared Error (MSE)
### Forward propagation

$$ L = \frac{ \sum_{i}^M{  \sum_{j}^N { \Delta y^2_{ij} }   }} {MN} $$

### Backward propagation

$$ \frac{\partial{L}}{\partial{ \Delta y^2_{i,j} }} = \frac{1}{MN} $$

$$ \frac{\partial{ \Delta y^2_{i,j} }} {y_{ij}} =  \frac{ \partial{ \left( y_{i,j} - y_{target}\right)^2 }} {\partial{ y_{ij }}} = 2y_{ij}$$


$$ \frac{\partial{L}}{\partial{ \Delta y_{i,j} }} = \frac{2y_{ij}}{MN} $$

