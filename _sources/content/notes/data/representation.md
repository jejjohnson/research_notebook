# Representation


---
## Field Representation


We basically present a 'raveled' version of the state whereby we measure the entire field. This can be shown as:


$$
\mathbf{x} = \left[\text{lon}_1, \ldots, \text{lon}_{D_\text{lat}}, \text{lat}_1, \ldots, \text{lat}_{D_\text{lon}},, \text{time}_1, \ldots, \text{time}_{D_\text{time}} \right] \in \mathbb{R}^{D_\text{lat} \times D_\text{lon} \times D_\text{time}}
$$

**Example**: if we have a *full* spatial lat-lon grid of `30x30` points and `30` time steps, then the vector is `30x30x30` which is `27,000`-dimensional vector! This is compounded if we wish to calculate correlations between each of the grid points which would result in a matrix of size `27,000 x 27,000` points. As we see below, this is a very high dimensional problem.


$$
D_\mathbf{x} = [\text{lat}_1, \ldots, \text{lat}_D, \text{lon}_1, \ldots, \text{lon}_D, \text{time}_1, \ldots, \text{time}_D]
$$

And the final vector, $\mathbf{x}$, can be massive for this unrolled spatio-temporal vector. So stacking all of these together gives us a very large vector of $\mathbf{X} \in \mathbb{R}^{D_\mathbf{x}}$. Estimating the covariance between each of coordinates would results in a massive matrix, $\mathbf{C}_{XX} \in \mathbb{R}^{D_x \times D_x}$. In the above algorithm, we need to do a matrix inversion in conjunction which is very expensive. Below you have the computational complexity when considering the state, $\mathbf{x}$:

State $\mathbf{x}$: 

- computational complexity - $\mathcal{O}(D_{\mathbf{x}}^3)$
- memory $\mathcal{O}(D_{\mathbf{x}}^2)$

---
## Coordinate Representation

The coordinate representation assumes the input vector, $\boldsymbol{x}$, is a single set of coordinates.

$$
D_\phi = [\text{lat,lon,time}]
$$

If we assume we have a large number of sparsely distributed coordinate values. This gives us a large number of samples, $N$. Stacked together, we get a matrix of samples and features (coordinates), $\boldsymbol{X} \in \mathbb{R}^{N \times D_\phi}$. 

**Example**: Take the full grid from the field representation, i.e. `30x30x30=27,000`-dimensional vector. Under this representation, we would have a vector which is three times the size, i.e. `N x D = 27,000 x 3` because for every grid point, we have a lat, lon, time coordinate. However, in our specific application, we have very sparse observations which means that we will never have to have a full grid of that size in memory (or during compute). If we assume there to be only 20% of the grid observed, then we have: `0.20 * 27,000 = 5,400` and a covariance of `5,400x5,400`. This is significantly less data that the full grid. This allows us to push the upper limit of the amount of observations where we can learn the parameters. For example, if we have a budget of `20,000` points for our memory (including the covariance), then we could potentially have a grid size of `46x46x46` if we wanted an evenly distribued spatio-temporal grid, `54x54x30` for a spatially dense grid and `30x30x74` for a temporal dense grid. This would allow the methods to capture processes at a finer scale without sacrificing computability.


This operation is still very expensive however, we assume that the observations are 

State $\boldsymbol{x}$:
- computational complexity $\mathcal{O}(N^3)$
- memory $\mathcal{O}(N^2)$

As you can see, the covariance matrix is still very expensive to calculate. However, the observations are very sparse compared to the full coordinate grid, i.e. $N << D_\mathbf{x}$. 

#### Pros and Cons

**Coordinate Transformations**: We have direct access to the lat-lon-time coordinates. This gives us the flexibility to perform transformations such that this is reflected within the input representation. 3 coordinates might lack information and it might be useful to transform these coordinates into a high representation. For example, if we transform the spatial coordinates from lat-lon to spherical coordinates, it goes from 2 to 3 dimensions. If we transform the time coordinate to a cyclic coordinate that encodes the minute, hour, day, month, year, then we go from a 1D vector to a 5D vector which could potentially encode some multi-scale dynamics.

**Large Upper Limit**: The data is sparse so that enables us to see more observations in space and time. The more data seen, the better the interpolation will be.

<span style="color:green">**Complete Space**</span>: We seen the complete space (all of the grid points). In many state-space methods, it is not possibly to put the entire dataset in memory. In addition, we can query observations which are not in a fine grid.



<span style="color:red">**No Physical Models**</span>: We are mapping a set of coordinates to a physical quantity of interest. This is different than the field representation which removes the physical sense. In this case, we are using a pure interpolation / smoothing setting. Many of the methods are based in proximity, e.g. nearest neighbour calculation. There is no direct physical interpretation between the space of coordinates to the physical quantity. When we consider the state-space representation, this becomes more feasible because we assume the 


---
## Data 

We assume that we have some set of input and output data points, $\mathcal{D} = \left\{ \mathbf{x}_n, \mathbf{y}_n \right\}_{n=1}^N$. This dataset, $\mathcal{D}$, will be used for training to find the parameters of interest, $\boldsymbol{\theta}$.

