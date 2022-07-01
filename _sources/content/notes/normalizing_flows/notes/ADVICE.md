# Advice

**Normalization**

* BatchNorm -> ActNorm

**Linear Layers**

* Permutations
* Random Init (Orthogonal, Permutations, SVD, ICA)
* HouseHolder
* 1x1Conv (Ortho, QR)
* Invertible Convolutions
* Convolutional Exponential

**MultiScale**

* Reshape -> Easy
* iRev -> Spatially Aware + Cheap
* Wavelet Haar -> Spatially Aware + Cheap


**Conditioners**

* Linear -> Tabular Data
* Convolutions -> MultiScale Data
* ResNet
* Attention
* Bottleneck (e.g. AutoEncoder) -> Anomaly Detection

**Masks**

* Partition
* 