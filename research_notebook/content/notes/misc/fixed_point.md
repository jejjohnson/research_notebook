# Fixed-Point Methods

* Efficient and Modular Implicit Differentiation - Blondel et al. (2021)


---
## Psuedo-Code


```python
def f(x, params):

    return ...
```

---
##### Loops

```python
def fp_solver(x, params, iterations=100):

    model = model.merge(params)
    
    for _ in range(iterations):

        x = f(x, params)

    return x
```

---
##### Jax Scan

```python
def fp_solver(x, params, iterations=100):


    def body(x, i):

        x = f(x, params)

        return x, i

    # run solver
    x_phi, _ = jax.lax.scan(body, init=x, xs=None, length=fp_iters)

    return x_phi
```

---
##### Fixed-Point Solver


```python

def fp_projection_update_opt(params, model, x, y, mask, fp_fn, **kwargs):
    
    model = model.merge(params)
    
    def T(x, model):
        
        return fn_projection_update(model, x, y, mask)
    
    fpi = fp_fn(fixed_point_fun=T, **kwargs)
    
    sol = fpi.run(x, model)
    return sol.params
```