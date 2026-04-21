---
title: "Intensity zoo — λ(t) kernels for methane sources"
---

# Intensity zoo

A temporal point process is a mathematical framework used to model random events occurring over continuous time. In the context of atmospheric monitoring, it is fully defined by its **conditional intensity function, λ(t)**, which represents the infinitesimal rate at which satellite-detected methane (CH₄) plumes are expected to occur given the history of past detections.

The notebook [04_intensity_gallery](04_intensity_gallery.ipynb) plots every module in this catalog side-by-side using the `methane_pod.intensity` library.

Here is a progression of point process models, starting from the foundational baseline and building up to the most complex, showing how each addresses the limitations of the last while applying them to satellite observations of global methane sources.

---

### 1. The Simplest Baseline: Homogeneous Poisson Process (HPP)

The Homogeneous Poisson Process is the bedrock of temporal modeling. It is completely "memoryless," meaning the future is entirely independent of the past.

**Physical Intuition (Methane Context):** Imagine a vast, stable network of aging natural gas distribution pipelines. Small, uncoordinated leaks happen entirely at random. A satellite passing overhead detects these transient plumes at a constant average background rate.

**Detailed Assumptions:**

1. **Constant Rate (Stationarity):** Plume events happen at a constant average rate, **λ > 0** (e.g., λ = 2 plumes/day). The probability of a satellite detecting a new plume in a tiny time interval (dt) is exactly λ·dt.
2. **Independent Increments:** The number of plumes in any two disjoint time intervals are completely independent of each other. Knowing that 100 plumes were detected in Texas yesterday tells you absolutely nothing about how many will be detected today.
3. **Orderliness (No Simultaneous Events):** The probability of two or more distinct leak events spontaneously occurring in the exact same infinitesimally small fraction of a second is zero.

**Equation:**
λ(t) = λ

*(The intensity is a strict constant, completely independent of time or past emission events.)*

**Visualization: Intensity & Distribution**

```text
Intensity λ(t) [Plumes/Day]               Inter-arrival Distribution P(Δt)
^                                         ^
|  λ = 2 (constant background leak rate)  | *
|---------------------------------------  |  *
|                                         |   *
|                                         |     * +---x-------x---x-----------x-------x-->  +-------*--------*------------->
   t₁      t₂  t₃          t₄      t₅       0              Δt [Days]
   (Plumes detected completely randomly)    (Exponentially distributed wait times)


```

---

### 2. Inhomogeneous Poisson Process (IPP)

**Limitation Addressed:** The HPP assumes a constant event rate. In the real world, methane emission rates fluctuate over time. Agricultural emissions, wetlands, and permafrost thawing are highly dependent on temperature, sunlight, and seasonal cycles.

**The Evolution:**
We relax the "Stationarity" assumption. The intensity is no longer a constant λ, but a deterministic function of time: **λ(t)**.

* The process is still completely memoryless.
* Satellite detections in disjoint intervals are still independent.
* The only difference is that the underlying "background rate" is allowed to rise and fall according to a set schedule (like a seasonal sine wave modeling rice paddy emissions peaking in summer).

**Equation:**
λ(t) = g(t)

*(The intensity g(t) changes over time t, but is still strictly deterministic and independent of past emission history.)*

**Visualization: Intensity & Distribution**

```text
Intensity λ(t) [Plumes/Day]               Inter-arrival Distribution P(Δt)
^                                         ^
|       .---.                   .---      | (Depends on the season/time t. 
|      /     \                 /          |  During summer warming peaks,
|     /       \               /           |  the exponential decay is 
|    /         \             /            |  steeper, meaning shorter 
+---x--x-x--x---x-----------x--x-x--x-->  +  wait times between plumes.)
   t₁ t₂t₃ t₄   t₅         t₆ t₇t₈ t₉         
   (Plumes cluster during warm seasons)           


```

---

### 3. Log Gaussian Cox Process (LGCP)

**Limitation Addressed:** The IPP forces us to assume the fluctuating rate `g(t)` is a perfectly known, deterministic function. In reality, atmospheric and operational conditions are inherently noisy and uncertain. We do not just have uncertainty in the *events*; we have uncertainty in the *rate itself*.

**Physical Intuition:** Instead of a rigid seasonal curve, imagine the background leak rate is driven by unobserved, fluctuating subsurface pressures or shifting regional economics. The rate itself becomes a random, undulating variable. This creates a "doubly stochastic" system: first, nature rolls the dice to determine what the leak rate is today, and second, nature rolls the dice to see if a leak actually happens at that rate.

**The Evolution:**
We elevate the deterministic `g(t)` to a stochastic Gaussian Process (GP).

* To guarantee the emission rate remains mathematically positive, we place the GP prior on the *logarithm* of the intensity function.
* The smoothness and volatility of the rate are strictly governed by the GP's covariance kernel (e.g., Exponentiated Quadratic), allowing us to infer the uncertainty of the underlying physics directly from the sparse satellite data.

**Equation:**
λ(t) = exp(f(t))
where f(t) ~ GP(0, K(t, t'))

*(The intensity is the exponential of a Gaussian Process, defined by a mean of zero and a covariance function K dictating how smoothly the physical rate evolves over time.)*

**Visualization: Intensity & Distribution**

```text
Intensity λ(t) [Plumes/Day]               Inter-arrival Distribution
^                                         ^
|           _..._                         | (Highly variable. High variance 
|         /       \      _                |  in the GP leads to intense 
|       /           \___/  \              |  clustering of events, heavily 
|     /                     \___          |  skewing the wait times compared 
+----x-x-x---------x----------x------>    +  to a standard Poisson model.)
     t₁t₂t₃        t₄         t₅
   (Plumes cluster strictly under the
    peaks of the latent GP curve)

```

---

### 4. Renewal Process

**Limitation Addressed:** All Poisson processes (HPP, IPP, LGCP) enforce strictly independent inter-arrival times modeled by the Exponential distribution. This means the time until the *next* plume detection never depends on how long you've already been waiting. In reality, consider intermittent venting from an industrial facility (e.g., a liquid natural gas storage tank). Pressure builds up steadily over time; the longer it has been since the last deliberate vent, the more likely the pressure relief valve will open in the next hour.

**The Evolution:**
We relax the Exponential distribution requirement.

* The times between satellite-detected plumes (Δt) are still independent and identically distributed (i.i.d.), but they can follow **any** probability distribution (e.g., Weibull, Gamma, Log-Normal).
* This allows you to model regular, predictable spacing (like routine maintenance venting schedules) or bursty spacing, giving the model a "local memory" regarding the time elapsed since the last immediate venting event.

**Equation:**
λ(t) = h(t - t_last)

*(The intensity is defined by a hazard function h, which depends purely on the time elapsed since the most recent plume detection, t_last.)*

**Visualization: Intensity & Distribution**

```text
Intensity λ(t) (Increasing Pressure)      Inter-arrival Dist P(Δt) (e.g., Normal)
^                                         ^
|     /|      /|           /|             |       ***
|    / |     / |          / |             |      * *
|   /  |    /  |         /  |             |     * *
|  /   |   /   |        /   |             |    * *
+-x----+--x----+-------x----+---------->  +---*---------*---------------->
 t₁       t₂           t₃                   0              Δt [Days]
(Vent risk grows as pressure builds)        (Venting happens at regular intervals)


```

---

### 5. Hawkes Process (Self-Exciting Point Process)

**Limitation Addressed:** All previous models assume events are strictly independent. However, in many industrial domains, **events cause more events**. An earthquake damages oil infrastructure; a major pipeline rupture stresses the surrounding network, causing pressure surges that trigger secondary valve failures and subsequent methane leaks in nearby connected infrastructure.

**The Evolution:**
We introduce explicit history dependence. The intensity function λ(t) jumps up whenever a plume is detected, and then decays back down over time as the system stabilizes or operators rush to patch the leaks.

* **μ (Base Rate):** The constant, background rate of spontaneous leaks (plumes/day).
* **tᵢ:** The timestamps of strictly past leak events.
* **φ(t - tᵢ) (Excitation Kernel):** A decay function (often exponential) that defines how much a past rupture at time tᵢ boosts the probability of new leaks at time t.

**Equation:**
λ(t) = μ + Σ_{tᵢ < t} α·exp(-β(t - tᵢ))

*(The intensity is the sum of a baseline μ plus an exponentially decaying boost α [boost in plumes/day] for every single infrastructure failure tᵢ that occurred in the past, decaying at rate β [1/days].)*

**Visualization: Intensity & Distribution**

```text
Intensity λ(t) [Plumes/Day]               Inter-arrival Distribution
^                                         ^ 
|       |\                                | (Highly complex and multimodal. 
|       | \         |\                    |  Wait times are a mix of long 
|       |  \        | \                   |  waits for background leaks 
|   |\  |   \       |  \                  |  and very short waits for 
|___| \_|____\______|___\_______ Base(μ)  |  rapidly cascading failures.)
+---x---|----x------x------------------>  +
    t₁  |    t₂     t₃
        Triggered secondary leak 
        (caused by pressure surge from t₁)


```

---

### 6. Mechanistic ODE Point Process

**Limitation Addressed:** Purely statistical point processes (like LGCP or Hawkes) are phenomenological—they map curves to data without understanding *why* the curves behave the way they do. They lack absolute physical constraints. If we know the thermodynamic laws governing a facility, guessing the rate with a statistical kernel is a severe waste of prior knowledge.

**Physical Intuition:** Methane buildup inside a sealed tank is governed by fluid dynamics. If we model the internal state (pressure, volume, temperature) as an Ordinary Differential Equation (ODE), we force our point process to obey the laws of physics. The intensity of satellite detections becomes a direct mapping of the simulated internal stress of the pipeline.

**The Evolution:**
We replace the statistical prior with a mechanistic vector field.

* We define a latent state variable `z(t)` governed by an ODE that balances deterministic forcing (like diurnal solar heating expanding the gas) against proportional decay (the leakage itself).
* The emission rate `λ(t)` is derived by clamping this physical state to strictly positive values using a Softplus transformation.
* Inference no longer finds arbitrary curves; it finds the exact physical coefficients (e.g., thermal expansion rates, mechanical friction) of the specific pipeline being observed.

**Equation:**
dz(t)/dt = α · (1 + sin(ω · t)) - β · z(t)
λ(t) = ln(1 + exp(z(t)))

*(The intensity is mathematically tethered to an underlying differential equation, where α is the accumulation amplitude, ω is the diurnal frequency, and β is the structural dissipation constant.)*

**Visualization: Intensity & Distribution**

```text
Latent State z(t) & Intensity λ(t)        Inter-arrival Distribution
^                                         ^
|  [ODE Models Internal Pressure]         | (Highly structured. Wait times
|         __                    __        |  are entirely dictated by the 
|       /    \                /    \      |  resonance and frequency of 
|      /      \              /      \     |  the underlying physical 
|-----/--------\------------/--------\->  |  differential equations.)
|    /          \          /          \   |
|   / (Negative  \        /            \  +
|  /   Pressure)  \      /
| /                \    /
=============================================================================

```

---

### 7. Neural / Deep Point Processes

**Limitation Addressed:** The Hawkes process and ODE models rely on strict mathematical assumptions about *how* the past influences the present or *how* the physics operate. They force you to choose a specific parametric shape. But what if the true relationship involves unmodeled, complex, non-linear interactions, or long-term inhibitions (where an event *prevents* future events)?

**Physical Intuition:** Consider a massive, multi-source region like the Permian Basin. Plume detections depend on a massive interplay of factors: oil price fluctuations driving extraction rates, complex facility maintenance schedules, weather patterns obscuring satellite visibility, and regulatory crackdowns. A massive super-emitter event might attract regulatory fines, actively *inhibiting* (lowering) the rate of future leaks as operators scramble to tighten their systems.

**The Evolution:**
We abandon hand-crafted mathematical kernels and physics equations, relying entirely on Deep Learning to model λ(t).

* Models like Recurrent Neural Networks (RNNs), LSTMs, or Transformers are used to encode the entire history of past plume detections into a hidden state vector, **h(t)**.
* The intensity is then calculated as a non-linear function of this hidden state.
* This allows the model to learn incredibly complex patterns straight from the satellite data—capturing both excitation (cascading failures boosting the rate) and inhibition (regulatory crackdowns lowering the rate), as well as complex seasonalities and long-term dependencies that previous mathematical frameworks could never fit.

**Equation:**
h(t) = RNN(h(t_{i-1}), t - t_{i-1})
λ(t) = f(Wᵀ · h(t) + b)

*(The hidden state h(t) continuously evolves based on neural network weights capturing regional dynamics, and the expected leak intensity is mapped from this highly non-linear, historically aware state.)*

**Visualization: Intensity & Distribution**

```text
Intensity λ(t) [Plumes/Day]               Inter-arrival Distribution
^                                         ^
|     .                                   | (Completely arbitrary and learned
|    / \          .---.                   |  directly from the satellite 
|   /   |   /\   /     \                  |  training data. Adapts to 
|__/    \__/  \_/       \___              |  complex human/nature cycles.)
+--x----x--x--x---------x-------------->  +
   t₁   t₂ t₃ t₄        t₅
(Intensity curves freely, factoring in 
 economics, weather, and past leaks)


```
