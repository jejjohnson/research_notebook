# Neural Fields

* [Formulation](./formulation.md)
* [Literature Review](./literature_review.md)



## **Coordinate-Based Methods**

Here I talk about what is a *field* and how does this approach differ from the standard grid-based methods. I'll try to highlight some of the advantages and disadvantages of using this coordinate-based approach.



## **Naive Neural Networks**

*Standard* neural networks don't do very well with high-frequency signals - which is something that inevitable occurs in certain fields. There are three main remedies that exist in the literature to overcome this problem: 1) High-Frequency-Based Encoders, e.g. Fourier Neural Networks, 2)  Modified activation functions, e.g. SIREN, and 3) Multiplicative Filter Networks, e.g. FourierNet, GaborNet. Each have their pros and cons when used and one should choose carefully based on what field they are modeling and properties they wish to have.


## **Derivative Properties**

Because we are using a coordinate-based method, we can take the derivative with respect to the input coordinates, e.g. space, time. This will affect how we are able to train these methods. It also differs greatly when we look at each of the listed above flavours of NerFs. Again, depending on the field we wish to model, this will be an important factor to consider when choosing a model configuration.


## **Modulation**

Naive NerFs are kind of closed because they can model a single field based on coordinates and that is it.
However, we can generalize their usage by adding modulation.
It's a very similar concept to *hypernetworks* accept we do a simple affine transformation on the weights instead of directly parameterizing the weights with a neural network.
This allows us to condition on external factors which opens up the possibilities.

## **Spatial-Temporal Considerations**

Most of the NerF research is centered around spatial fields.
However, adding temporal coordinates can cause problems because they are fundamentally different than spatial coordinates.
I showcase how one can directly add the temporal coordinate into the scheme.
I also demonstrate a better way to do it via positional encodings which allow us to use the modulations mentioned in the previous section.


## **Conditioning**

We will make use the modulation so that we can condition on external factors.
This will help us to utilize external information to help *guide* the representation.
This even offers us an opportunity to incorporate embeddings from grid-based methods, e.g. CNNs and Flow Models (AEs, VAEs, NFs, Diffusion, etc).


## **PDE Constraints**

Physics-Informed Neural Networks (PINNs) is an entire field in itself.
These are algorithms that modify the loss function to comply with PDE constraints.
It turns out that the base network used in PINNs is a NerF but the solution they obtain is inevitably better because they constrain it with known physics.

## **Predictive Uncertainty**

We often don't actually observe the exact values of the field so we need to account for some of this uncertainty. Here we outline some methods we can utilize to incorporate some of this uncertainty within the observations.

## **Training Issues**

All of the stuff above offer many methods to improve the theoretical solution we would like to obtain.
However, there are many practical aspects we need to incorporate in order to actually obtain said solution.
Here, I outline some of the most common ones including 1) learning rate schedulers, 2) batch sizes, 3) fine-tuning, 4) transfer learning, 5) temporal causality, and 6) curriculum learning.
