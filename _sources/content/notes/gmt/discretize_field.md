# Field Discretization

So we talked about spatial discretization and temporal discretization. However, there is one more discretization concept which might not be obvious at first but it is a fundamental - we can discretize the field itself.


In machine learning, the most obvious example one could think of a pixels within images. The actual pixel value itself can have a value between 0-255. So this is not a continuous representation of values, they are inherently discrete.

Another common problem we choose to solve is one of segmentation.

Lastly, we do the same for classification.

This is very prevalent in applications that require decision-making. We often see extreme event indices like drought or heat whereby we mark a range of values with some semantic meaning. For example, we can say that temperatures that are between 20-30 are normal, temperatures between 30-40 are high and temperatures 40+ are considered extreme. We can do the same for droughts, wildfires and even hurricanes. A more extreme example is when we decide to mark a threshold to say when we act or don't act (a binary classification problem).


---
## Demo Code


**Apply Operations**

```python
# initialize the domain
domain: Domain = ...

# initialize the field
fn: Callable = lambda x: ...
u: Field = Discretization(domain, fn)

# apply operator
du_dx = gradient(u)

# apply operator with predefined params
params: Params = DiscretizationParams(...)
u_grad = gradient(u, params)
```


**Discretization** - Finite Difference

```python
class FDParams:

```
