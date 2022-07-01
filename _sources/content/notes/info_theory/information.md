# Information Theory

In this report, I will be outlining what exactly Information theory is and what it means in the machine learning context. Many times when people ask me what I do, I say that I look at Information theory (IT) measures in the context of a generative network. But sometimes I have a difficult time convincing myself that these measures are actually useful. I think this is partially because I don't fully understand the magnitude of IT measures and what they can do. So this post is designed to help me (and others) really dig deep into the space of information measures and I hope this will help someone else who is also interested in understanding IT without any formal classes. This post will not be formula heavy and will instead focus on concepts. See the end of the post for some additional references where you can explore each of these ideas even further.


---
### Information (Revisited)

Now we will start to do a deep dive into IT measures and define them in a more concrete sense.

$$I(x)= \underset{Intuition}{\log \frac{1}{p(x)}} = \underset{Simplified}{- \log p(x)}$$

The intuitive definition is important because it really showcases how the heuristic works in the end. I'm actually not entirely sure if there is a mathematical way to formalate this without some sort of axiom that we mentioned before about **surprise** and **uncertainty**.

We use logs because...

**Example Pt I: Delta Function, Uniform Function, Binomial Curve, Gaussian Curve**



---
### PDF Estimation

For almost all of these measures to work, we need to have a really good PDF estimation of our dataset, $\mathcal{X}$. This is a hard problem and should not be taken lightly. There is an entire field of methods that can be used, e.g. autoregressive models, generative networks, and Gaussianization. One of the simplest techniques (and often fairly effective) is just histogram transformation. 
I work specifically with Gaussianization methods and we have found that a simple histogram transformation works really well. It also led to some properties which allow one to estimate some IT measures in unison with PDF estimation. Another way of estimating PDFs would be to look at kernel methods (Parezen Windows). A collaborator works with this methodology and has found success in utitlize kernel methods and have also been able to provide good IT measures through these techniques.





---
## Supplementary Material

### GPs and IT








---
## References

#### Gaussian Processes and Information Theory


#### Information Theory


* Information Theory Tutorial: The Manifold Things Information Measures - [YouTube](https://www.youtube.com/watch?v=34mONTTxoTE)
* [On Measures of Entropy and Information](http://threeplusone.com/on_information.pdf) - 
* [Understanding Interdependency Through Complex Information Sharing](https://pdfs.semanticscholar.org/de0b/e2001efc6590bf28f895bc4c42231c6101da.pdf) - Rosas et. al. (2016)
* The Information Bottleneck of Deep Learning - [Youtube](https://www.youtube.com/watch?v=XL07WEc2TRI)
* Maximum Entropy Distributions - [blog](http://bjlkeng.github.io/posts/maximum-entropy-distributions/)
* 



**Articles**

* A New Outlook on Shannon's Information Measures - Yeung (1991) - [pdf](https://pdfs.semanticscholar.org/a37e/ab85f532cdc027260777815d78f164eb93aa.pdf)
* A Brief Introduction to Shannon's Information Theory - Chen (2018) - [arxiv](https://arxiv.org/pdf/1612.09316.pdf)
* Information Theory for Intelligent People - DeDeo (2018) - [pdf](http://tuvalu.santafe.edu/~simon/it.pdf)

**Blogs**

* Visual Information Theory - Colah (2015) - [blog](https://colah.github.io/posts/2015-09-Visual-Information/)
* Better Intuition for Information Theory - Kirsch (2019) - [blog](https://www.blackhc.net/blog/2019/better-intuition-for-information-theory/)
* A Brief History of Information Theory - Vasiloudis  (2019) - [blog](http://tvas.me/articles/2018/04/30/Information-Theory-History.html)
* Information Theory of Deep Learning - Sharma (2019) - [blog](https://adityashrm21.github.io/Information-Theory-In-Deep-Learning/)

