# Metrics



### MAE

Mean Absolute Error

$$
\text{MAE}(Y, \hat{Y}) = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|
$$

### MSE

Mean Squared Error

$$
\text{MSE}(Y, \hat{Y}) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
$$


### RMSE

Root Mean Squared Error


$$
\text{RMSE}(Y, \hat{Y}) = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}
$$

### R-Squared



$$
\text{R}^2(Y, \hat{Y}) = 1 - \frac{\sum_{i=1}^N (\hat{y}_i - y_i)^2}{\sum_{i=1}^N (\hat{y}_i - \bar{y})^2}
$$


---
## Percentage Metrics

### MAPE

**Mean Absolute Percent Error**

$$
\text{MAPE}(Y, \hat{Y}) = \frac{1}{N} \sum_{i=1}^N \left|\frac{  \hat{y}_i - y_i}{y_i} \right|
$$

### SMAPE

**Symmetric Mean Absolute Percent Error**

$$
\text{SMAPE}(Y, \hat{Y}) = \frac{100\%}{N} \sum_{i=1}^N \left|\frac{  \hat{y}_i - y_i}{\frac{|\hat{y}_i|+|y_i|}{2}} \right|
$$