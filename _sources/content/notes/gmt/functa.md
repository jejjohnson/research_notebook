# Functa

---

# Motivation

In my experience, we seem to jump a little bit when talking about models and data representations, especially in the geosciences. We often jump straight to image-to-image applications without really thinking about the data/functa. I think there is a logical progression that describes data by the means of coordinates, data, and models. This is my interpretation of how we can describe data and its representations and what we hope to achieve. Throughout this entire tutorial, I will be using sea surface height (SSH) as my example quantity of interest to showcase how each of these quantities are related.

### My Objective

I hope that this will bring some cohesion between different concepts in terms of *data* and *modeling*. I also think it will help me as a (future) machine learning research engineer to make some things *more* concrete with different abstractions. I sincerely believe that it will make my research skills better and also my programming skills as I try to tame the wildness. Abstraction and Ontology is very important in SWE and I think it should be important in geosciences. Also, those who know me, always know that I like to understand the *information flow* of peoples ideas and that I will almost always ask what the *shape of the data* is when fed into our machines. I have strong feelings about the *information flow* of systems (more later) but I also think that thinking about data representations in geoscience is really an underrated idea…

### Format

The rest of the blog post will go through some of the fundamental concepts we need for modeling in geoscience. From the data side, we have the *space* that defines the coordinate system and the *domain* which defines the range of values for the coordinates. From the modeling side, we have the *physical model* which defines some constraints in the form differential operators in space and time and we also have *parameterized models* which are functions with learnable parameters. Lastly, we come back to reality with observations which can be the same as the data or some other proxy which we believe is related. I end this will some somber news about why these abstracts are only the containers because the real world we live in really limit our ability to learn models based on data.

---

# Space

Almost no quantity of interest (QOI) exist within a vacuum. We almost always assume that our quantities of interest lie within some spatial-temporal space, even if we choose to ignore it. For example, they may spatially lie within a pixel space (e.g. an image) and temporally lie along a time line. In most geoscience applications, we can define some arbitrary coordinate system which varies in space and time. Let


$$
\mathbf{x}_s \in \mathbb{R}^{D_s}, \hspace{5mm} t \in\mathbb{R}^+
$$ (functa_space)

where $\mathbb{R}^{D_s},\mathbb{R}^{+}$ is the domain for all possible queries in space and time. It represents the form of the queries for each of these QOIs. The full domain $\mathbb{R}$ represent the infinite values that the coordinates can have and the exponent defines the properties of the coordinate. In general, a single spatial coordinate is a vector of size $D_s$ and a single temporal coordinate is a positive integer. These coordinates define the entire space where “data” can lie.  For example, we can have the Cartesian coordinate system which is generally rectangular, cubic or hyper-cubic. We could also have the spherical coordinate system which is circular, spherical or hyper-spherical. We can also have some hybrid version of the two, e.g. the cylindrical coordinate system, or some other very funky coordinate systems.

````{admonition} Code Formulation
:class: dropdown idea

One example of a space

```python
x_domain: obj = Space("euclidean")
```

Perhaps we could sample from this domain

```python
x_samples: Array["100"] = x_domain.sample(100)
```

Now the minimum and maximum of these samples would be between $(-\infty,\infty)$.

````

````{admonition} Example: Sea Surface Height
:class: dropdown info

My application is geoscience so everything lives on the Earths core, surface or atmosphere.

````

````{admonition} Better Discussion
:class: dropdown seealso

For more details on the notion of spaces and coordinates, please see [this jbook](https://r-spatial.org/book/02-Spaces.html) where they have a detailed discussion about coordinate reference systems. It is the jupyter-book of the full book: [Spatial Data Science with Applications in R](https://www.routledge.com/Spatial-Data-Science-With-Applications-in-R/Pebesma-Bivand/p/book/9781138311183).

````




---

# Domain

In almost all cases in geoscience, we can take the Earth as the domain where all the geophysical quantities lie. If we were to think about Cartesian coordinate system and the Earth, we could assume that the center of the earth is the origin of the domain and the values are bounded between 0 and the radius of the Earth, e.g. ~6,371 km (perhaps a bit further into space if we account for the atmosphere). We can also think about it in spherical terms which bounds the domain in terms of how many rotations, e.g. $-2\pi,2\pi$. We can even be interested in subregions along the globe like the Mediterranean sea or the open ocean. For the spatial domain, we define $\Omega$ as proper subset of the entire domain $\mathbb{R}^{D_s}$. For the temporal domain, $\mathcal{T}$, it is usually defined along a bounded number line $\mathcal{T}\in[0, T]$. However, we could also use the 24 hour clock as well as cyclic calendars with years, months and days. So more concretely, we define our coordinate system where our quantities of interest lie as

$$
\mathbf{x}_s \in \Omega \subset \mathbb{R}^{D_s}, \hspace{5mm} t\in\mathcal{T}\subset\mathbb{R}^+
$$ (functa_domain)

There are of course simplifications to this system. For example, we could be looking for processes that stationary (don’t change in time) or constant (don’t change in space). However, both are sub-cases and are still covered within this notation!



````{admonition} Code Formulation
:class: dropdown, idea

One example of a domain on a line

```python
t_min: float = 0.0
t_max: float = 1.0
type: str = "line"
time_domain: obj = Domain(bounds=(t_min, t_max))
```

Perhaps we could sample from this domain (infinitely)

```python
samples: Array["inf"] = time_domain.sample(inf)
```

Now the minimum and maximum of these samples would be between $[0,1.0]$. This would be nearly equivalent with a vector of spatial coordinates along a line.

**2D Space Domain**

So let's assume we have a 2D domain that's defined on Euclidean space.

```python
x_space: object = Domain((x_min, x_max))
y_space: object = Domain((y_min, y_max))
```

Now we want to create an object

```python
xy_domain: object = Domain((x_domain, y_domain))
```

````


````{admonition} Example: Sea Surface Height
:class: dropdown, tip

For sea surface height, it can exist on the surface of a sphere (roughly). This has the spatial coordinates as latitude, longitude and we can set some arbitrary temporal some coordinates as days.

$$
\mathbf{x}_s = [\text{latitude},\text{longitude}], \hspace{5mm} t=[0 \text{ day},1 \text{ day}]
$$

We can also transform these coordinates into an x,y plane with a higher frequency. For example, we can use the local tangent coordinate system defined over the Gulfstream where SSH is queried every hour.

$$
\mathbf{x}_s=[x,y], \hspace{5mm} t=[\text{hours}]
$$

This is typically what we do in numerical schemes as it can be quite difficult to march along the entire globe, especially at finer resolutions (see my discussion on discretization).

````


---

# Functa (Data)

Coordinates alone can be thought of as references (or query points) to an actual quantity of interest, $\boldsymbol{f}\in\mathbb{R}^{D}$, which could be a scalar or a vector. Almost all of these quantities live on some domain. I really like the term *functa* to describe these quantities, i.e. the quantity of interest and its context which includes the set of coordinates, the mapping, and the resulting function value. So the function mapping the coordinates to the variable is give as:

$$
\boldsymbol{f}=\boldsymbol{f}(\mathbf{x}_s,t), \hspace{10mm} \boldsymbol{f}:\Omega\times\mathcal{T}\rightarrow\mathbb{R}^{D}
$$ (functa)

In physics, we typically call this a scalar or vector field which means that each value within this field is a scalar or vector value with an associated spatio-temporal coordinate. These *functa* are continuous as there exists infinitely many spatial-temporal queries we can do which lead to infinitely many values of the functa. The actual functa values could also be infinite. I don’t put any restrictions on the actual functa values themselves. I just say that there is a functa value for the infinitely many coordinates we could query which, by association, would result in infinitely many functa values.

**Note**: I did not include any stochastic term in this formulation but one could easily include this here. I don’t know if there exists any stochasticity in nature and it could just be an artefact of our parameterizations. That’s an open debate I have with myself. In either case, I leave this as an artefact and have included a short discussion below.


```{admonition} Emans Thoughts: Storage
:class: dropdown, tip

- it’s cheaper to store the mapping rather than the actual (see functa paper)
- Can do both! - store the coordinates & values, xarray

```


```{admonition} Emans Thoughts: Images vs. Geoscience Data
:class: dropdown, tip

Pixels -  `height x width x channels`
Don’t think about it as an image →  think about it as a mapping

Images don’t have to… - live in an isolated space with implied coordinates

Geosciences don’t have the luxury of images (we need the coordinates as well)
- everything lives in a relative space
- Need transformations from one domain to another

===
I often get these requirements from geoscientists to want to encode the physical relationships (i.e. interactions between variables) and global information (i.e. the relative position of the measurement and its effect on the overall structure).

For example, when using ML to map the state from $f_1$ to $f_2$, they often don’t want to give the model the actual global, $(\mathbf{x}_s,t)$, coordinates because they are afraid that the model will memorize the state based on it’s global position. But they want the model to know the global structure of the state when making predictions. To get global structure, you need a relative position (not a global position).

I think this conversation always comes up because our intuition doesn’t fit the standard mathematical model / objects that we use.

Practically speaking, you can do this my normalizing your input coordinates, i.e. $(\mathbf{x}_s, t) \rightarrow (\hat{\mathbf{x}_s}, \hat{t})$.

```




```{admonition} Example: Sea Surface Height
:class: dropdown idea

For SSH, we are interested in the height of the ocean above the mean sea level. We can denote this as:

$$
\text{Sea Surface Height [m]} \hspace{10mm} \eta = \boldsymbol{\eta}(\mathbf{x}_s,t)
$$

So although we typically take the SSH values as “data”, they are actually fields/functa which have an associated coordinate for every SSH value and (as mentioned above), in nature, there are infinitely queries we could use which would result in potentially infinitely many SSH values.

```



---

# Physical Model

The objective would be to somehow find the true functa. In geosciences, we can write some partial differential equations (PDE) that we believe describe how this field changes in space and time. In most cases, we typically denote this evolving field as $u$ which is some approximation to the true field $\boldsymbol{f}$.

$$
\boldsymbol{f}(\mathbf{x}_s,t)\approx\boldsymbol{u}(\mathbf{x}_s,t)\hspace{15mm}\boldsymbol{u}:\Omega\times\mathcal{T}\rightarrow\mathbb{R}^{D}
$$ (functa_field)

where field $u$ is a representation of the true field $f$. However, we need some way for this field, $u$, to be close to the true field $f$. So this is where the (partial) differential equations come into play. We state that this field needs to satisfy some constraints which we define as space-time operations. The associated PDE constraint is defined by some arbitrary differential operators on the field to describe how it changes in space and time. Therefore, the PDE can be thought of a set of equations that act as constraints to how the field needs to behave in space and time.  So we can add the PDE constraints as:

$$
\begin{aligned}
u &= \boldsymbol{u}(\mathbf{x}_s,t) \\
\text{s.t.  } \partial_tu &=\mathcal{N}[u;\boldsymbol{\theta}](\mathbf{x}_s,t)
\end{aligned}
$$ (functa_pde)

where $\mathcal{N}$ is the differential operator on the field and $\boldsymbol{\theta}$ are the (hyper-) parameters for the PDE. These parameters don’t actually exist in nature. They are artefacts introduced by the PDE which are often unknown and/or assumed based on some prior knowledge.

```{admonition} Example: Sea Surface Height
:class: dropdown idea
For our SSH variable, there are many approximate models we can use to describe the dynamics. One such model is the Quasi-Geostrophic equations given by

$$
\begin{aligned}
\eta &= \boldsymbol{\eta_\theta}(\mathbf{x}_s,t) && && &\boldsymbol{\eta_\theta}:\Omega\times\mathcal{T}\rightarrow\mathbb{R} \\
\text{s.t.   }\dot{\eta} &= -\frac{g}{f} \det\boldsymbol{J}(\eta,\nabla\eta)
\end{aligned}
$$

Through a series of assumptions, we approximate this. For this example, we know that this is a crude approximate of the actual dynamics. However, we can assume that this is “good enough”.

```



## Domain

Because our functa is often defined on a bounded domain, our PDE must also be able to understand the field on the bounded domain. For the spatial domain, we need to describe what happens at the edges (e.g. rectangle) of the domain. For the temporal domain, we need to describe what happens at the beginning of the domain, e.g. $t=0$. We can also define these as operators. Let’s define these as:

$$
\begin{aligned}
\mathcal{BC}[u;\boldsymbol{\theta}](\mathbf{x}_s,t) &=  \boldsymbol{u}_b, && &&
 \mathbf{x}_s\in\partial\Omega &&
 & &t\in\mathcal{T} \\
\mathcal{IC}[u; \boldsymbol{\theta}](\mathbf{x}_s,0) &= \boldsymbol{u}_0, && && \mathbf{x}_s\in\Omega
\end{aligned}
$$  (functa_pde_domain)

where $\mathcal{BC}$, are the boundary conditions on the field, $\mathcal{IC}$ are the initial conditions on the field. The boundary conditions dictate the behaviour on the spatial domain on the boundaries and the initial conditions dictate the behaviour at the initial condition, $t=0$. We find these a lot even in ML applications. For example, whenever we deal with convolutions on images, we need to think about what to do at the boundaries (the solution is almost always padding, a.k.a. ghost points). In toy problems in physics, we also often simplify these to make the problem easier and well-behaved. A common approach is to use periodic boundary conditions; which are very rare in nature; but they are very convenient because they allow us to use simpler solvers like spectral and pseudo-spectral solvers. If we have access to observations, then we can use these as initial and boundary conditions. This is often done in data assimilation fields like gap-filling and reanalysis.

---

# Parameterized Model

We could also assume that we don’t know anything about the physical equations that govern the system. However, we believe that we can learn about it from *data*. Let’s assume (by luck) we can define each pairwise spatial-temporal coordinate and field value of the *functa* that we are interested in. So we have a set of pairwise points which we can call a dataset $\mathcal{D}$ defined as

$$
\mathcal{D} = \left\{(\mathbf{x}_{s,n},t_n),\boldsymbol{f}_n \right\}_n^{\infty}
$$ (functa_model)

I say this is infinite because technically we can sample any continuous function infinitely many times without revisiting any previous samples (even on a bounded domain). The objective would be to find some sort of approximation of the actual function, $\boldsymbol{f}$, which maps each of these coordinate-values to the correct scaler/vector value. So we can define some arbitrary parameterized function, $\boldsymbol{f_\theta}$, which tries to approximate the functa. We can say that:

$$
\boldsymbol{f}(\mathbf{x}_s,t)\approx\boldsymbol{f_\theta}(\mathbf{x}_s,t)
$$

This parameters depend upon the architecture of the function we choose. Again, like the PDE, these parameters are artefacts introduced by the function. So for a linear function, we may have just a few set of parameters (weights and bias), for a basis function we may the same parameters with some additional hyper-parameters for the basis, and neural networks have many weights and biases which we apply compositionally. Now, if we have a flexible enough model and infinite data, we should be able to find such parameters to fit the functa. However, the problem becomes *how* to find those parameters. This is the *learning problem.* We assume the solution exists and we can find it. But the question becomes: 1) how do we find it and 2) how do we know we have found it. However, there is an entire field dedicated to trying to resolve these issues, e.g. optimization for finding the solution and defining the metrics for knowing if we’ve found it. In addition, I stated that we assume the problem exists and we have infinite data which is never true at all. So that only adds more problems…

### Non-Parametric Models

- **Example: Regression**


    Which functions do we use will depend upon the data representation, the assumptions of the solutions, the computational power we have, the amount of datapoints we have, and any additional constraints. We can use neural fields (NerFs) which are coordinate-based neural networks. We can also use Gaussian processes (GPs) which are non-parametric, probabilistic, functional methods.


---

# Observations

Throughout this note, we have assumed that there exists a true field, $f$, for some QOI, e.g. SSH. And we have determined that one can either write down a PDE function, $u$, using physical equations or we can write a parametric function, $\boldsymbol{f_\theta}$, where we can learn from samples from the field. However, in practice, we often never actually observe the true field that we are interested in. In the best case scenario, we observe noisy, sparse or incomplete measurements of the field; symptoms of circumstance. So we define some observation variable $y$ and we can define a function, $\boldsymbol{h}$, that maps the “true” field to the observations.

$$
y = \mathcal{H}[u](\mathbf{x}_s,t)
$$ (functa_obs)

In the worst case scenario, we observe proxy variables which are related to the QOI through some complex, unknown dynamics. This is often the case (especially in geosciences) as we are interested in one variable but for some reason, we cannot observe this variable. However, we can observe another variable which is related in some way, shape or form.

```{admonition} Example: Geosciences
:class: dropdown, hint

The use of proxy variables is very common in geosciences. We have many satellites but we cannot observe all of the interesting quantities on the Earth. In the case of oceanography, we have an abundance of SST and SSH data. SST is the most abundant and it is available at a higher frequency (hourly) and finer spatial resolution (~500 m) whereas SSH is less abundant with a frequency of 1 day with a spatial resolution of 100km. So one natural thing is to find a relationship between SSH and SST which would allow us to to find a function that can fit the SSH field and then...

```


---

# Real Problems

Now, this is all idealized where everything is continuous and accessible. But reality strikes quickly and we realize that much of the stuff above becomes impossible. I alluded to a few of these issues above but I will expand upon a few of them below. I try to go in a “logical” order that increases the impending doom and the ill-posedness of the problem we are trying to solve.

---
## Discretization

This is easily the cause of most of our problems with real-world data. The world may be continuous but the society we live in and how we operate is discrete. In terms of computation, continuous is very expensive and thus impractical (and possibly unnecessary). Continuous is also expensive. The way that we do computation doesn’t work this way. The way that we collect measurements also are fundamentally discrete.

```{admonition} Sparse Observations
:class: dropdown, tip

The idea of limited or sparse observations is an artefact of discretization.

```

---
## Model Specification

This is the second thing that causes the most problems. Even if we had continuous/infinite data, there is no guarantees that our assumed model actually captures the underlying processes of the system. From a physical perspective, this can be attributed to unknown physics. From a parametric modeling perspective, this can be due to not a sufficiently flexible model. And even if we have a non-parametric model, the problem of limited observations (discretization) causes this to be a moot point.

```{admonition} Stochasticity & Noise
:class: dropdown, tip

We often put *stochasticity* and/or *noise* as an inherent problem in the system. I disagree. I think that this is an artefact of model specification.

```

```{admonition} Chaos
:class: dropdown, tip

We often put *chaos* as an inherent problem in the system. I disagree. I think that this is an artefact of model specification.

```


---
## Learning Problem

This is the final icing on the cake. Even if we were to have sufficient observations and the most flexible model possible, there are no guarantees that we can actual *learn* the appropriate function. Similarly, if we have
