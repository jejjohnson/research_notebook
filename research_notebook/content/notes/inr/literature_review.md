# Literature Review

**Source**: [Awesome-Neural-Rendering](https://github.com/weihaox/awesome-neural-rendering)


## Best Tutorials

* Fourier Feature Networks and Neural Volume Rendering - [Video](https://www.youtube.com/watch?v=Al6NTbgka1o)

> Shows how to learning a signal based on Fourier features. Does time series, 2D spatial and 2D volume.


---
## Basics



---
## Most Interesting



---
## Best Code



---


* Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields - [Project](https://jonbarron.info/mipnerf/)
* nerf2D - [Project](https://www.matthewtancik.com/nerf) | [Paper](https://arxiv.org/abs/2003.08934) | [Paper Explained](https://www.youtube.com/watch?v=CRlN-cYFxTk) | [Code](https://github.com/ankurhanda/nerf2D)


---

* Plenoxels: Radiance Fields without Neural Networks - Fridovich-Keil & Yu et al (2022) - [Project](https://alexyu.net/plenoxels/) 
> They used a sparse voxel grid along with a trilinear interpolation field to fill in the missing data. Superfast convergence

pixelNeRF: Neural Radiance Fields from One or Few Images - Yu et al (2021) - [Project](https://alexyu.net/pixelnerf/)

> They use convolutions to get some global structure. Apparently it works on very few samples.

RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs - [Project](https://m-niemeyer.github.io/regnerf/index.html)

> They learn from very few samples. They regularize with a normalizing flow.

NeRF-VAE: A Geometry Aware 3D Scene Generative Model - [Paper](https://arxiv.org/abs/2104.00587) | [Talk](https://papertalk.org/papertalks/32326)

Learned Initializations for Optimizing Coordinate-Based Neural Representation - [Project]()

> Talks about meta-learning which can be used for some more advanced problems for faster convergence.

---
### Spatio-Temporals

* Neural Radiance Flow for 4D View Synthesis and Video Processing - Du et al (2021) - [Project](https://yilundu.github.io/nerflow/)
* Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes - Li et al (2021) - [Project](https://www.cs.cornell.edu/~zl548/NSFF/)

---
## Generative Models

### Normalizing Flows

Neural Radiance Flow for 4D View Synthesis and Video Processing - Du et al (2021) - ICCV - [Project](https://yilundu.github.io/nerflow/)


---
### Random Fourier Features

* Random Feature Expansions for Deep Gaussian Processes - Cutajar et al - [Thesis](https://www.google.com/search?q=Kurt+Cutajar+thesis&oq=Kurt+Cutajar+thesis&aqs=chrome..69i57j33i160.2459j0j4&sourceid=chrome&ie=UTF-8)
* On the Error of Random Fourier Features - [Paper](https://arxiv.org/abs/1506.02785) | 


#### Fourier Operators

* Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains - [Project](https://bmild.github.io/fourfeat/) | [Code Demo](https://github.com/GlassyWing/fourier-feature-networks)
* Fourier Neural Operator for Parametric Partial Differential Equations - [Paper](https://arxiv.org/abs/2010.08895) | [Code](https://github.com/zongyi-li/fourier_neural_operator) | [Project](https://zongyi-li.github.io/neural-operator/) | [Paper Explained](https://www.youtube.com/watch?v=IaS72aHrJKE) | [Anima AnandKumar](https://www.youtube.com/watch?v=Bd4KvlmGbY4)
* SIREN: Implicit Neural Representations with Periodic Activation Functions - [Paper Explained](https://www.youtube.com/watch?v=Q5g3p9Zwjrk) | [Project](https://www.vincentsitzmann.com/siren/) | [Jax](https://github.com/KeunwooPark/siren-jax) | [PyTorch](https://github.com/lucidrains/siren-pytorch)

