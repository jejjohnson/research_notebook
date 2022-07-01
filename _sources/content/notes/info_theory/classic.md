# Classic Methods


---

## Parametric

### Assume Gaussian

---

#### Single Variate

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left( \frac{-x^2}{2\sigma^2} \right)
$$


##### Entropy

$$
h(X) = \frac{1}{2}\log (2\pi e\sigma^2)
$$



$$
\begin{aligned}
h(X) 
&= - \int_\mathcal{X} f(X) \log f(X) dx \\
&= - \int_\mathcal{X} f(X) \log \left( \frac{1}{\sqrt{2\pi}\sigma}\exp\left( \frac{-x^2}{2\sigma^2} \right) \right)dx \\
&= - \int_\mathcal{X} f(X)
\left[ -\frac{1}{2}\log (2\pi \sigma^2) - \frac{x^2}{2\sigma^2}\log e \right]dx \\
&= \frac{1}{2} \log (2\pi\sigma^2) + \frac{\sigma^2}{2\sigma^2}\log e \\
&= \frac{1}{2} \log (2\pi e \sigma^2)
\end{aligned}
$$






**From Scratch**

```python
def entropy_gauss(sigma: float) -> float:
    return np.log(2 * np.pi * np.e * sigma**2)
```

**Numpy**

```python
from scipy import stats

H_g = stats.norm(scale=sigma).entropy()
```


* Lecture 8: Density Estimation: Parametric Approach  - [Lecture Notes](http://faculty.washington.edu/yenchic/18W_425/Lec8_parametric.pdf)




---

## Histogram



* [Astropy Histograms](https://docs.astropy.org/en/stable/visualization/histogram.html)




---
## Kernel Density Estimation



* Intro to Kernel Density Estimation - [Video](https://www.youtube.com/watch?v=x5zLaWT5KPs)
* Lecture 6: Density Estimation: Histogram and Kernel Density Estimator - [Lecture](http://faculty.washington.edu/yenchic/18W_425/Lec6_hist_KDE.pdf)
* [KDEPy Literature](https://kdepy.readthedocs.io/en/latest/literature.html)
* [Viz Demo of KDE](https://mathisonian.github.io/kde/)
* [A Tutorial on KDE and Recent Advances](https://arxiv.org/pdf/1704.03924.pdf) - arxiv (2017)
* [KDE From Scratch](https://jduncstats.com/post/2019-03-16_kde-scratch/) - w Julia
* [In Depth KDE](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html) - Jake
* [KDE Tutorial](http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/r3_kde.html)
* [KDE: How to compute gaussian KDE w. Python](https://gsalvatovallverdu.gitlab.io/python/kernel_density_estimation/)
* [Statsmodels Tutorial](https://www.statsmodels.org/stable/examples/notebooks/generated/kernel_density.html)

**Software**

* [kdepy](https://kdepy.readthedocs.io/en/latest/)
* [StatsModels](https://www.statsmodels.org/stable/examples/notebooks/generated/kernel_density.html)
* [Numba Implementation](https://numba.pydata.org/numba-examples/examples/density_estimation/kernel/results.html)
* [KDE Numba](https://github.com/ablancha/kde_numba)
* [Wrapper for Scipy](https://github.com/DTOcean/dtocean-core/blob/578129d52ecb0a6bc02270fe3cad4d6083c1da0a/dtocean_core/utils/stats.py)
* [pyqt - KDE Wrapper](https://github.com/sergeyfarin/pyqt-fit/blob/master/pyqt_fit/kde_methods.py)


---
## K-Nearest Neighbours






* [Paper](https://content.sciendo.com/view/journals/tmmp/50/1/article-p39.xml?language=en) - k-NEAREST NEIGHBOUR KERNEL DENSITY ESTIMATION, THE CHOICE OF OPTIMAL k
* [Helpful Presentation](http://www.cs.haifa.ac.il/~rita/ml_course/lectures_old/KNN.pdf)
* Lecture 7: Density Estimation: k-Nearest Neighbor and Basis Approach - [Prezi](http://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf)
* KNN Density Estimation, a slecture by Qi Wang - [Vid](https://www.youtube.com/watch?v=CG2Z8gaLDpA)
* Non-parametric density estimation - 3: k nearest neighbor - [Video](https://www.youtube.com/watch?v=GlZ8_rG3zXk)
* Mod-05 Lec-12 Nonparametric estimation, Parzen Windows, nearest neighbour methods - [Video](https://www.youtube.com/watch?v=esoVuEG-X1I)
* Modal-set Estimation using kNN graphs, and Applications to Clustering - [Video](https://www.youtube.com/watch?v=gKpTSmpejMg)



#### Entropy

The full entropy expression:

$$
\hat{H}(\mathbf{X}) =
\psi(N) -
\psi(k) +
\log{c_d} +
\frac{d}{N}\sum_{i=1}^{N}
\log{\epsilon(i)}
$$

where:
* $\psi$ - the digamma function.
* $c_d=\frac{\pi^{\frac{d}{2}}}{\Gamma(1+\frac{d}{2})}$
* $\Gamma$ - is the gamma function
* $\epsilon(i)$ is the distance to the $i^{th}$ sample to its $k^{th}$ neighbour.