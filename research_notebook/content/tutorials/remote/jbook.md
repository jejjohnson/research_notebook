# JupyterBook



## Helpful Things


### Labeled Equations

We can use the following syntax

```
$$
\mathbf{x}_t = \boldsymbol{f}(\mathbf{x}_{t-1}, \boldsymbol{\epsilon})
$$ (ssm_eqn)
```

to produce the following:

$$
\mathbf{x}_t = \boldsymbol{f}(\mathbf{x}_{t-1}, \boldsymbol{\epsilon})
$$ (ssm_eqn)


We can also reference these later with this syntax:

```
{eq}`ssm_eqn
```
to get the following:
Recall from: {eq}`ssm_eqn`.


### Proofs


See [this](https://sphinx-proof.readthedocs.io/en/latest/syntax.html) documentation.

### Algorithms


---

## Citations

### Multiple bibs

```
bibtex_bibfiles:
  - references.bib
  - bibliographies/software.bib
```


### Local Citations

Syntax:

```
```{bibliography}
:filter: docname in docnames
:style: alpha
```
```

To get:

```{bibliography}
:filter: docname in docnames
:style: alpha
```