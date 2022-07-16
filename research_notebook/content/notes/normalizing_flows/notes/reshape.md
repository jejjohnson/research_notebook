# Reshaping

* Reshaping
  * [Squeeze2D](https://github.com/matejgrcic/DenseFlow/blob/main/denseflow/transforms/bijections/squeeze.py) | [Unsqueeze2D](https://github.com/matejgrcic/DenseFlow/blob/main/denseflow/transforms/bijections/unsqueeze.py)
* IRev-Net - [Freia](https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/reshapes.py#L12)
* Haar Wavelet - [Freia](https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/reshapes.py#L191) | [DenseFlow](https://github.com/matejgrcic/DenseFlow/blob/main/denseflow/transforms/bijections/wavelet.py) | [Other](https://github.com/JingyunLiang/HCFlow/blob/main/codes/models/modules/Basic.py#L450) |
* Rotation [Source](https://github.com/matejgrcic/DenseFlow/blob/main/denseflow/transforms/bijections/rotate.py)
* Orthogonal Squeeze - [DenseFlow](https://github.com/matejgrcic/DenseFlow/blob/main/denseflow/transforms/bijections/orth_squeeze_pgd.py)

---
## Reshaping

* [Source](https://github.com/matejgrcic/DenseFlow/blob/main/denseflow/transforms/bijections/reshape.py)

```python
class Reshape(Bijection):

    def __init__(self, input_shape, output_shape):
        super(Reshape, self).__init__()
        self.input_shape = torch.Size(input_shape)
        self.output_shape = torch.Size(output_shape)
        assert self.input_shape.numel() == self.output_shape.numel()

    def forward(self, x):
        batch_size = (x.shape[0],)
        z = x.reshape(batch_size + self.output_shape)
        ldj = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        batch_size = (z.shape[0],)
        x = z.reshape(batch_size + self.input_shape)
        return x
```

---
## Haar Wavelet Transform

**Kernel**

```python
def _create_kernel(n_channels):
    kernel = torch.ones(4, 1, 2, 2)
    kernel[1, 0, 0, 1] = -1
    kernel[1, 0, 1, 1] = -1

    kernel[2, 0, 1, 0] = -1
    kernel[2, 0, 1, 1] = -1

    kernel[3, 0, 1, 0] = -1
    kernel[3, 0, 0, 1] = -1
    kernel *= 0.5

    kernel = np.concatenate([kernel] * n_channels, axis=0)

    return kernel
```

### DownSampling


```python
class WaveletDownSampling:
    """Works on Images (C, H, W)"""
    def __init__(self):
        pass

    def forward(self, x):
        output = F.conv2d(x, kernel, bias=None, stride=2, groups=n_channels)
        return output, 0

    def inverse(self, x):
        output = F.conv2d_transpose(x, kernel, bias=None, stride=2, groups=n_channels)
        return output, 0

```


---

## IRev Reshaping

### Upsampling

**Source**:
* https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/reshapes.py
* github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py

**calculate kernel**

```python

def init_kernel(n_channels: int=3):

    kernel = np.zeros(4, 1, 2, 2)

    kernel[0, 0, 0, 0] = 1
    kernel[1, 0, 0, 1] = 1
    kernel[2, 0, 1, 0] = 1
    kernel[3, 0, 1, 1] = 1

    kernel = np.concatenate([kernel] * n_channels, axis=0)

    return kernel
```



**calculate output dims**

```python
def calculate_output_dims(shape):

    c, w, h = shape

    c2, w2, h2 = c * 4, w //2, h // 2

    if c * h * w != c2 * h2 * w2:
        print("NOPE! Odd number of inputs")

    return c2, w2, h2
```

```python
class IRevDownSampling:
    def __init__(self):
        pass

    def forward(self, x):
        output = F.conv2d(x, kernel, stride=2, groups=n_channels)
        return output, 0

    def inverse(self, x):
        output = F.conv2d_transpose(x, kernel, stride=2, groups=n_channels)
        return output, 0

```

### DownSampling

**Source**: https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/reshapes.py



**calculate output dims**

```python
def calculate_output_dims(shape):

    c, w, h = shape

    c2, w2, h2 = c // 4, w * 2, h * 2

    if c * h * w != c2 * h2 * w2:
        print("NOPE! Odd number of inputs")

    return c2, w2, h2
```

**Bijection**


```python
class IRevUpSampling(IRevDownSampling):
    def __init__(self):
        pass

    def forward(self, x):
        return super().inverse(x)

    def inverse(self, x):
        return super().forward(x)

```
