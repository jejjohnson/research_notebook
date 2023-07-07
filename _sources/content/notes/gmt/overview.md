# GMT of Learning

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


[**Uncertainty**]() [**TODO**]. This gives a breakdown
