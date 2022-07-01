# Dynamical Systems

Assume we have an imperfect model of reality:

$$
\boldsymbol{\nabla}_t{\color{blue}\mathbf{x}}= \boldsymbol{f} \left( {\color{blue}\mathbf{x}}, t, \mathbf{u}, \boldsymbol{\beta}, \boldsymbol{\epsilon}\right)
$$(dyn_sys)

where 


## Components

### State

${\color{blue}\mathbf{x}} \in \mathbb{R}^D$ is the *state* of the system. This can be meteorological variables such as wind speed, temperature or pressure. It could also be oceanographic like temperature, salinity and pressure. 

### Function, $\boldsymbol{f}$

This is the imperfect model of reality. For example, it could be a linear model, non-linear, stochastic or chaotic.

We treat $\boldsymbol{f}$ as a stochastic model for many reasons:

- **Imperfect Model**: the model $\boldsymbol{f}$ embeds our knowledge about the laws governing the process. This is well known to be partial and/or incorrect.
- **Numerical Discretization**: the model $\boldsymbol{f}$ is a spatial-temporal discretization of the physical laws (e.g. Navier Stokes for fluids) which is expressed as a partial differential equation on continuous domain. This is a finite resolution which is dictated by the computational power/speed availability). This will always induce errors.
- **Chaos**: Many natural systems are chaotic and exhibit extreme sensitivity to initial conditions. This is an error (possibly inevitable) that is inherent to the system at an arbitrary time which contaminates the prediction.

### Time, $t$

This is dynamic so we have some time, $t$ which shows that the system evolves with time.

## State Representation



## Observation Models

We often don't have access to the original signal we are interested in

$$
\mathbf{y} = \boldsymbol{g}(\mathbf{x},t, \boldsymbol{\eta})
$$(obs_model)

where:

- $\mathbf{y} \in \mathbb{R}^P$ - observations
- $\boldsymbol{g}: \mathbb{R}^D\rightarrow \mathbb{R}^P$ - observation model (mapping of the state to the observations)
- $\mathbf{x} \in \mathbb{R}^D$ - state of the system, e.g. meteorological variables (wind speed, temperature, etc) in every single grid point
- $t$ - time of the observations (discrete, continuous)
- $\boldsymbol{\eta}$ - noise model

Here, we assume that the observations are not perfect.

---

### Components

#### Observed Variable, $\mathbf{y}$

These are often things that we can actually observe in nature with some sort of instrument. 

**In Situ** 

These measurements are direct using instruments. They are typically very sparse and irregular in space and time because it's just impossible to obtain more. For example, in an oceanographic setting, these can be hydrographic observations via ships. In a meteorological setting, these are often weather balloons and possibly aircraft. It has to be said that the ARGO project is a grand effort to make these measurements more regular. 

**Satellite**

These measurements are often indirect because they are often not exactly looking at the exact variable of interest, but instead a different variable altogether. For example, in meteorological applications, this is often the radiance even though we are not really interested in the radiance; we're more interested in the state and how it relates to the radiance.

These measurements can also be direct as well because sometimes we do actually have access to satellite measurements which correspond to quantity we are actually interested in. 

 in situ observations or possibly satellite observations. For example, in the case of many meteorological applications, it's the radiance.

 #### Function, $\boldsymbol{g}$

We also treat $\boldsymbol{g}$ as a stochastic model for many reasons (some are the same as the state space model):

- **Instrument Noise**: Each instrument comes with errors that need to be accounted for, e.g. in-situ, satellites, etc.
- Representativity: The models and the observations may have different resolutions.
- Model Specification: The operator could be incorrectly characterized.


##### Example Operators

**Identity**

I

$$\mathbf{y} = \mathbf{x}$$

where:

- $\mathbf{x}$ - variable at time $t$.
- $\mathbf{y}$ - observation at time $t$
- $\boldsymbol{h}$ - this operator is the identity


**Transformation**

$$\mathbf{y} = \boldsymbol{h}_{\boldsymbol \theta}(\mathbf{x})$$

where:

- $\mathbf{x}$ - variable at time $t$
- $\mathbf{y}$ - observation at time $t$
- $\mathbf{h}: \mathbf{x} \in \mathbb{R}^{N_x} \rightarrow \mathbf{y} \in \mathbb{R}^{N_y}$ - this operator is a function that tries to relate the variable $\mathbf{x}$ and $\mathbf{y}$.

**Interpolator**

$$\mathbf{y} = \mathbf{H}\mathbf{x}$$

where:

- $\mathbf{x}$ - variable at time $t$
- $\mathbf{y}$ - observation at time $t$
- $\mathbf{H}\in \mathbb{R}^{N_y \times N_x}$ - this operator is an interpolator that maps the data between the full resolution of $\mathbf{x}$ and partially observed resolution of $\mathbf{y}$.


---
## Challenges

- Unknown $\boldsymbol{f}$
- Non-Linear $\boldsymbol{f}$
- High-Dimensional $\mathbf{x,y}$.
- Multi-scale
- Chaotic
- Latent/Hidden Variables
- Noise/Disturbances
- Uncertainty

**ALL ARE OPTIMIZATION PROBLEMS**!

---
## Applications

- Predict the future (ensemble/statistical)
- Design/Optimization (F1 Jets, Yachts, Engines)
- Control
- Understanding (interpretable, generalizable)