---
title: Overview
subject: Modern 4DVar
subtitle: How to think about modern 4DVar formulations
short_title: Overview
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CNRS
      - MEOM
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: data-assimilation, open-science
abbreviations:
    GP: Gaussian Process
---

+++ {"part": "abstract"}

This is a top-level document that introduces the Four-Dimensional Variational (4DVar) Formulation from a modern perspective. The purpose of this document is to consolidate all of the 4DVarNet related stuff under one roof.
It will also serve as a nice springboard for us to use it when explaining 4DVar for different groups, i.e. oceanographers, data assimilation experts and machine learning researchers.
For the core foundations of 4DVar, this document will host links to the formulation, some specific details involving spatiotemporal discretization, and an introduction to the 4DVarNet algorithm using all of the notation described in the previous two documents.
For things related to research, there is the document on stochasticity and document consolidating all of the papers that were published (or will be published) related to 4DVar.

+++




## Motivation

We want an abstraction for how we think about the solving problems with modern tools.
The research and software world is enormous and it is very hard to see how everything is connected.
In many cases, we see instances of people starting with the algorithm and then defining the usecase.
In other cases, we see a lot of instances of people choosing the problem and then
I propose to do it via a problem-oriented approach whereby we build up the necessary abstraction.

**Wide Audience.**
I want to reach a large audience.
Nowadays problems are very complicated and it requires many different people from many different disciplines.
I personally find that we are very bad at communicating with each other irrespective of the audience because we don't take the time or energy to create a common language between each other.

**Easy Entry Point**.
I want there to be a relatively easy entry point for these people to.
I personally find that ML literature tends to be too focused on displaying novelty by highlighting differences rather than similarities.
I want this to highlight similarities rather than differences.
Hopefully it will serve as a springboard for newcomers to get inspired as well as other experts alike to find even more commonalities.

**Software Oriented.**
I firmly believe software will save science and engineering.
It gives us the necessary computational power to be able to solve new tasks or ask new questions that was impossible before.
However, it's not enough to have such a powerful tool: we need a place to use it.
It's easier to build end-to-end tools once the levels of abstractions have been set.
So having this clean abstraction framework might help people get an idea of how we can make all-inclusive software piece together to build an end-to-end framework.


---
## Framework



### [Task Identification](./framework/tasks.md) (**TODO**)

> There are many geosciences tasks that can be described in terms of ML tasks. The most major tasks include interpolation and forecasting and almost all *tasks* should be somewhere underneath this umbrella. However, there are some other tasks that can occur either. Some could be subtasks that are needed to solve the original task, e.g., denoising, latent embeddings, and conditional density estimation. There are also more "pure" tasks which are less on the application side and more on the *learning* or *discovery* side, e.g. attribution, process discovery, etc. Ultimately, any problem within the geoscience realm can be expressed underneath the Digital Twin umbrella which provides a useful framework to encompass most model-driven developments.


---

### [Problem Structure](./framework/problem.md)


> This provides the core abstraction for how we're going to concretize the problems we want to solve.
> We outline some key components that we need to identity like the Quantity of Interest we wish to estimate, the observations we have access to, the controls we wish to utilize and the overall state we wish to define.
> We also will put this under a single umbrella by using graphical models to represent the relationships between all of the components.
> This will help us to define which pieces we wish to estimate based on what we have and do not have.
> Many concepts outlined here are heavily inspired by this talk by [Karen Willcox](https://www.youtube.com/watch?v=ZuSx0pYAZ_I&t=2767s) where it presents a similar formulation and proposes the use of Bayesian Graphical Models.


$$
\begin{aligned}
\text{Quantity of Interest}: &&
\boldsymbol{u} = \boldsymbol{u}(\vec{\mathbf{x}},t)
&& \boldsymbol{u}: \boldsymbol{\Omega}_u \times \mathcal{T}_u \rightarrow \mathbb{R}^{D_u} \\
\text{State Space}: &&
\boldsymbol{z} = \boldsymbol{z}(\vec{\mathbf{x}},t),
&& \boldsymbol{z}: \boldsymbol{\Omega}_z \times \mathcal{T}_z \rightarrow \mathbb{R}^{D_z} \\
\text{Observations}: &&
\boldsymbol{y} = \boldsymbol{y}(\vec{\mathbf{x}},t)
&& \boldsymbol{y}: \boldsymbol{\Omega}_y \times \mathcal{T}_y \rightarrow \mathbb{R}^{D_y}
\end{aligned}
$$

---

### [Estimation Problem](./framework/estimation.md) (**In Progress**)

> This formulation states the learning problem from a Bayesian perspective.
> We cover the different elements we may want to estimate based on the graphical model we stated above.
> Namely, we discuss state estimation, parameter estimation, and both (i.e. Bi-Level Optimization).
> We also discuss gradient learning schemes which provide an all-encompassing end-to-end learning framework for learning model parameters, estimating the state space and estimating the path towards the best solution(s).

$$
\begin{aligned}
\text{Parameter Estimation}: &&
\boldsymbol{\theta}^* &=
\underset{\boldsymbol{\theta}}{\text{argmin}}
\hspace{2mm}
\boldsymbol{L}(\boldsymbol{\theta};\mathcal{D}) \\
\text{State Estimation}: &&
\boldsymbol{z}^*(\boldsymbol{\theta}) &=
\underset{\boldsymbol{z}}{\text{argmin}}
\hspace{2mm}
\boldsymbol{J}(\boldsymbol{z};\boldsymbol{\theta}) \\
\text{Gradient Learning}: &&
\boldsymbol{z}^{(k+1)} &= \boldsymbol{z}^{(k)} + \boldsymbol{g}_k \\
&& [\boldsymbol{g}_k, \boldsymbol{h}_{k+1}] &= \boldsymbol{g}(\boldsymbol{\nabla_z}\boldsymbol{J},\boldsymbol{h}_k, k; \boldsymbol{\phi})
\end{aligned}
$$

---

### [Hierarchical Decisions](./framework/problem_decisions.md) (**TODO**)

```{mermaid}
graph LR
    Processes --> Representation
    Representation --> Relationships
    Relationships --> Uncertainty
    Uncertainty --> Solution-Methodology
```

> In order to have a general ontology of how one can include information into a model, we need some standards for how we can describe the decisions we make.
> This outlines the notion of a Hierarchical system where a user can outline all of their assumptions from the idea down to the modeling decisions.
> This attempts at an ontology of generic decisions one has to make to create a model and find a solution.
> This includes defining the *processes* involved, choosing an approprate *representation* based on the data and the resources, choosing how the *process relationships*, including the *uncertainty* through every level, and the *solution* methodology chosen.
> Many concepts here stem from this excellent talk by [Hoshin Gupta](https://www.youtube.com/watch?v=eH6vwiukIsA&t=3541s&pp=ygUYaW5mb3JtYXRpb24gaG9zaGluIGd1cHRh) that outlines an instance of Hierarchical system of model choices one can use.
> This section also serves as a precursor to the following section where we go into more details.






<!-- # GMT of Learning

These are the notes for my *Grand Master Theory* of Learning. It's very superfluous but that's by design. It's just that I see many learning problems over and over and over again. Everything is connected but the papers explain everything as if it were novel, unique and disconnected. I try to synthesize everything from my perspective.

[**Hierarchical Representations**](./hierarchical_rep.md) [**TODO**]. These notes come from an [excellent talk](https://www.youtube.com/watch?v=eH6vwiukIsA) by Hoshin Gupta. He explains this from. These are a sequence of hierarchical decisions by which we can follow whenever we're trying to solve a problem. They include 1) choosing the processes to include, 2) choosing the system architecture, 3) choosing the process parameterization, 4) the specification of uncertainty, and 5) choosing a solution procedure. I outline some key points of this talk and give my own spin on a few.


[**Data (Functa)**](./functa.md). This is my attempt to try and explain data from a *functional* perspective. No data exists in a vacuum, especially geoscience. I start with spaces/coordinates, then I talk about data (functa), followed by different perspectives on learning from data (parametric or physical). Lastly, I mention how we connect things to the *real world* through observations.


**[Spatial](./discretize_space.md), [Temporal](./discretize_time.md) & [**Field**](./discretize_field.md) Discretization**. Discretization is inevitable as it breaks down the continuous fantasy world to the real world. For geoscience, all data lives in the spatial-temporal plane where we need to make choices about discretization for both coordinates. This is my attempt to explain discretization and showcase some different choices we can make for both space and time. The [**spatial discretization page**](./discretize_space.md) attempts to showcase how this is done with a few common schemes including auto-differentiation, finite difference and spectral methods. It also briefly describes ODEs as a sort of special case of spatial discretization.

[**Learning**](./learning.md). This tries to identify the different learning strategies for the scientific machine learning problems.
Typically, we can break them down into defining a model to make predictions that match the observations and we learn the parameters of said model, i.e [parameter estimation](./state_est.md).
The second problem is where we define some minimization problem with a set of constraints and we need to find the minimum value that satisfies said constraints, i.e. [state estimation](./state_est.md).
This is often found in geoscience (however it doesn't have to be).
Of course, we can always do a combination of both, whereby we do the minimization problem in conjunction with learning the parameters of the model, i.e. [bi-level optimization](bilevel_opt.md)
<!-- I will outline each of these approaches and also introduce a new explanation that defines both -->


[**Uncertainty**]() [**TODO**]. This gives a breakdown -->
