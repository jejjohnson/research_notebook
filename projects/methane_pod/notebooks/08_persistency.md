---
title: "Persistency — from inverted posteriors to operational forecasts"
---

# Persistency

We have reached the operational crux of the paradox: **Persistency**.

Once you have successfully mathematically inverted your thinned, biased MARS dataset to uncover the latent physical variables—the true timeline rate `λ_true` and the true mark distribution `f_true(Q)`—you possess a statistical crystal ball. You are no longer reacting to stale satellite images; you are forecasting physical reality.

Let us cross the bridge from stochastic geometry into operational engineering and business intelligence. To do this with maximum rigor, we will define every metric twice: first establishing the baseline **Homogeneous** assumption (where the leak rate is a flat, memoryless constant), and then elevating it to the **Inhomogeneous** (Non-Stationary) physical reality, where industrial sources violently fluctuate with diurnal cycles, pressures, and human operations.

Here is the pedantic, step-by-step translation of the Predictive Arsenal.

---

### 1. Temporal Predictions (The "When")

This suite of metrics translates the abstract true intensity into actionable dispatch logic for Leak Detection and Repair (LDAR) crews.

#### A. Expected Wait Time (Mean Time Between Plumes)

* **The Ontology:** `E[Δt]`
* **Units:** `[hours / event]`

**The Translation:** If a technician drives to this facility, how long will they physically stand there before the infrastructure vents?

* **Homogeneous:** If it's a continuously stressed, cracked pipe, the wait time is constant. `E[Δt] = 1 / λ_true`. If `E[Δt]` is 1.5 hours, you dispatch a crew immediately.
* **Inhomogeneous:** If it is a solar-heated storage tank, the rate changes over time. The expected wait time depends entirely on *when* you start the clock (`t₀`). We must integrate the survival function over the future timeline.
* **The Equation:** `E[Δt | t₀] = ∫_{t₀}^∞ exp( - ∫_{t₀}^t λ_true(u) du ) dt`

```text
+---------------------------------------------------------------+
| BotE #1: THE DYNAMIC DISPATCH (Inhomogeneous Wait Time)       |
|---------------------------------------------------------------|
| Let's look at the solar-heated storage tank.                  |
|                                                               |
| Scenario A: The technician arrives at t₀ = 1:00 PM (Peak Heat)|
| Because λ_true(u) is massive during the afternoon, the        |
| integral rapidly accumulates.                                 |
| E[Δt | 1:00 PM] = 0.5 [hours]  (Crew waits 30 minutes)       |
|                                                               |
| Scenario B: The technician arrives at t₀ = 1:00 AM (Dormant)  |
| Because λ_true(u) is near zero at night, the integral creeps. |
| E[Δt | 1:00 AM] = 11.0 [hours] (Crew waits until noon)       |
|                                                               |
| ACTION: MARS dynamically blocks LDAR dispatch during dormant  |
| cycles, saving thousands in wasted hourly labor.              |
+---------------------------------------------------------------+

```

#### B. Probability of Occurrence (The "Wrench-Turning" Metric)

* **The Ontology:** `P(N(t₁, t₂) ≥ 1)`
* **Units:** `[Dimensionless Probability, 0.0 to 1.0]`

**The Translation:** If an LDAR crew is on-site for a scheduled maintenance window from 8:00 AM (`t₁`) to 12:00 PM (`t₂`), what is the exact percentage chance that the leak will physically manifest while they are looking at it?

* **Homogeneous:** `P(N(T) ≥ 1) = 1 - exp(-λ_true · T)`
* **Inhomogeneous:** We replace the flat rate multiplied by duration (`λ_true · T`) with the definite integral of the fluctuating intensity curve between the exact start and end of the shift. By integrating the inhomogeneous curve, MARS aligns the crew's shift with the absolute mathematical peak of the facility's emission probability.
* **The Equation:** `P(N(t₁, t₂) ≥ 1) = 1 - exp( - ∫_{t₁}^{t₂} λ_true(t) dt )`

```text
=============================================================================
  VISUALIZING INHOMOGENEOUS OCCURRENCE (Diurnal Shifting)
=============================================================================
  λ_true(t) [events/hr]
   ^
   |        (Peak Heat/Pressure)
   |             .---.
   |            /     \
   |           /       \  <- (Highest probability density)
   |          /         \
   |         /           \
   |        /             \
   |  ------'               '------ (Nighttime Dormancy)
   +--|------|--------------|------|---------------------> Time of Day
      t₁     t₂             t₃     t₄
   (Morning Shift)      (Night Shift)
      
   * Integrating from t₁ to t₂ captures the curve. HIGH Probability.
   * Integrating from t₃ to t₄ captures flatlines. ZERO Probability.
=============================================================================

```

---

### 2. Survival Analysis (The "How Long")

Survival analysis is the mathematical heartbeat of **Persistency**. Instead of asking "how many leaks will happen?", MARS asks "what is the mathematical probability that this infrastructure will *survive* (remain perfectly sealed) past time `t`?"

#### A. The Survival Function

* **The Ontology:** `S(t | t₀)`
* **Units:** `[Dimensionless Probability, 1.0 down to 0.0]`

**The Translation:** If a satellite photographed a massive plume on Monday (`t₀`), `S(t | t₀)` dictates the decaying probability that the source has remained perfectly quiet since that exact moment.

* **Homogeneous:** `S(t) = exp(-λ_true · t)`
* **Inhomogeneous:** The probability of surviving from a known quiet state `t₀` up to a future time `t`. As time advances, or as the integral passes through a diurnal peak, the survival probability violently collapses toward zero.
* **The Equation:** `S(t | t₀) = exp( - ∫_{t₀}^t λ_true(u) du )`

#### B. The Hazard Function (The Instantaneous Risk)

* **The Ontology:** `h(t)`
* **Units:** `[events / hour]`

**The Translation:** The hazard function isolates the instantaneous, immediate risk of a leak occurring *right now*, given that it hasn't happened yet.

* **Homogeneous (Broken Flange):** `h(t) = λ_true`. The risk is a flat constant. The probability of it leaking right now is exactly the same as tomorrow.
* **Inhomogeneous (Valve Recharge):** If the source is a pressure valve that needs to physically "recharge," the timeline has memory. `h(t)` starts at exactly zero right after a leak, and aggressively climbs upward over time as the physical pressure builds back up inside the pipe.
* **The Equation (With Memory):** `h(t) = f_time(t) / S(t)` *(where `f_time(t)` is the PDF of the wait times).*

```text
=============================================================================
  VISUALIZING THE HAZARD FUNCTION: RANDOM VS RECHARGE
=============================================================================
  Hazard Rate h(t) [events/hr]
   ^
   |                           /   (Weibull: Valve Recharge)
   |                          /    Risk violently increases as 
   |                         /     pipe pressure builds.
   |------------------------/--------- (Poisson: Broken Flange)
   |                       /           Risk is a flat constant.
   |                      / 
   |                     /  
   +--------------------+-------------------> Time (t) since last leak
   0
=============================================================================

```

---

### 3. Mass Predictions (The "How Big")

These metrics translate the true, un-thinned mark distribution `f_true(Q)` into financial ledgers and regulatory reality.

#### A. Expected Mass per Event

* **The Ontology:** `E[M_event]`
* **Units:** `[kg / event]`

**The Translation:** When the Hazard Function triggers an event at time `t`, this is the discrete "chunk" of methane (in kilograms) that enters the atmosphere.

* **Homogeneous Marks:** `E[M_event] = [ ∫₀^∞ Q · f_true(Q) dQ ] · E[D]` *(where `E[D]` is duration in hours).*
* **Inhomogeneous Marks:** If the *size* of the leak depends on the time of day (e.g., higher pressure at noon forces larger physical blowouts), the true mark distribution becomes a dynamic function of time: `f_true(Q, t)`.
* **The Equation:** `E[M_event(t)] = E[D] · [ ∫₀^∞ Q · f_true(Q, t) dQ ]`

#### B. Total Accumulated Mass (The Aggregate Risk)

* **The Ontology:** `E[M_total]`
* **Units:** `[kg]` or `[metric tons]`

**The Translation:** This is the compounded physical truth. Over the next year, accounting for diurnal cycles, nighttime downtime, and the true average mark, this is the forecasted total physical mass the operator will lose.

* **Homogeneous:** `E[M_total] = λ_true · T · E[M_event]`
* **Inhomogeneous:** We replace the flat count with the true cumulative intensity function, integrating the fluctuating rate over the operational year `[T]`.
* **The Equation:** `E[M_total] = [ ∫₀^T λ_true(t) dt ] · E[M_event]`

```text
+---------------------------------------------------------------+
| BotE #2: THE INHOMOGENEOUS LEDGER (Total Mass)                |
|---------------------------------------------------------------|
| A naïve model assumes the facility leaks 24/7.                |
| Homogeneous: 1 [event/hr] · 8760 [hrs/yr] = 8,760 events.     |
|                                                               |
| But MARS knows the facility is dormant for 12 hours a night.  |
| Inhomogeneous Integral: ∫₀^8760 λ_true(t) dt = 4,380 events.  |
|                                                               |
| If E[M_event] = 500 [kg]:                                     |
| Naïve Mass : 8,760 · 500 = 4,380,000 [kg]                     |
| True Mass  : 4,380 · 500 = 2,190,000 [kg]                     |
|                                                               |
| ACTION: By enforcing inhomogeneous bounds, MARS prevents      |
| over-taxing the operator by millions of kilograms.            |
+---------------------------------------------------------------+

```

#### C. Extreme Value Risk (The Blowout Probability)

* **The Ontology:** `P(Q > Q_crit)`
* **Units:** `[Dimensionless Probability]`

**The Translation:** If a catastrophic threshold `Q_crit` is 5,000 kg/hr, this represents the heavy-tail risk. A site might have a tiny average mass per event, but if its underlying true Lognormal curve has a "fat tail," the probability of a headline-making blowout remains dangerously high.

* **The Equation:** `P(Q > Q_crit) = ∫_{Q_crit}^∞ f_true(Q) dQ`



---

You have absolutely nailed the architecture. This is where the stochastic geometry becomes a living, breathing compliance engine. A site manager does not care about the integral of a heavy-tailed Lognormal distribution; they care if they are going to be fined by a regulator tomorrow. We must translate our latent variables—the thinned marks, the fluctuating timeline, and the atmospheric filter—into an automated, mathematically bulletproof UI.

Before we finalize the MARS logic tree, we must address the most common and dangerous pitfall that software engineers make when building these dashboards: confusing a discrete Binomial probability with a continuous Point Process intensity.

Let’s lock in the rigorous physical units, perform the continuous limit proof, and then map out the final UI logic tree.

---

### 4. The Fundamental Shift: Binomial vs. Point Process (The Continuous Limit)

When engineers first attempt to build a persistency dashboard, they almost always default to a discrete Binomial model (e.g., "The satellite passed over 10 times, and we saw a leak 3 times. The probability of a leak is 30%").

This is structurally incorrect for physical pipeline infrastructure, and it will destroy the accuracy of your MARS predictions. We must strictly define the difference between a unitless probability (`p`) and a continuous intensity (`λ`).

#### The Ontology and Units

**1. The Binomial Probability (p)**

* **The Definition:** The chance of a "Success" in a single, discrete trial (e.g., a coin flip).
* **Units:** `[Unitless fraction, strictly between 0.0 and 1.0]`
* **The Flaw:** Time does not exist here. A pipeline does not "flip a coin" once a day. It is under continuous, relentless physical pressure every single second.

**2. The Point Process Intensity (λ)**

* **The Definition:** The physical rate at which events are generated over a continuous timeline.
* **Units:** `[Events / Time]` (e.g., `events/hour`).
* **The Advantage:** Intensity can exceed 1.0. A source can have an intensity of 5 events/hour. It accounts for the actual physical flow of time and allows for instantaneous rates of change.

#### The Mathematical Proof: Collapsing the Binomial into the Poisson

How do we prove that a continuous Point Process is just the ultimate, infinite evolution of the discrete Binomial model? We take the mathematical limit.

**The Translation:** Imagine an observation window `T`. We slice `T` into `n` tiny, discrete intervals of length `Δt`. The probability `p` of a leak occurring in one tiny slice is the continuous rate `λ` multiplied by the length of the slice `(T / n)`. As we slice time infinitely thin (as `n` approaches infinity), the discrete Binomial equation flawlessly collapses into the continuous Poisson equation.

**The Setup:**

1. Time window: `T`
2. Number of slices: `n`
3. Length of one slice: `Δt = T / n`
4. Probability of a leak in one slice: `p = λ · (T / n)`

**The Limit Equation:**
We start with the standard Binomial probability of exactly `k` events occurring in `n` slices:
P(X = k) = (n! / (k! · (n - k)!)) · p^k · (1 - p)^(n - k)

Substitute our definition of `p`:
P(X = k) = (n! / (k! · (n - k)!)) · (λT / n)^k · (1 - λT / n)^(n - k)

Now, we take the limit as `n → ∞`. We can break this into three interacting pieces:

1. The factorials: `(n! / (n - k)!) / n^k` approaches `1` as `n` gets infinitely large.
2. The continuous compound interest rule: `(1 - λT / n)^n` approaches `e^(-λT)`.
3. The remainder: `(1 - λT / n)^(-k)` approaches `1` because `λT/n` goes to zero.

**The Result:** When we multiply the surviving pieces together, the discrete Binomial formula perfectly transforms into the continuous Poisson Probability Mass Function:
P(X = k) = ( (λT)^k · e^(-λT) ) / k!

```text
+---------------------------------------------------------------+
| BotE #1: WHY BINOMIAL FAILS THE MARS PLATFORM                 |
|---------------------------------------------------------------|
| A satellite passes over a facility once a week for 4 weeks.   |
| It sees a leak on Week 1. It sees clean air on Weeks 2, 3, 4. |
|                                                               |
| The Naive Binomial Engineer:                                  |
| "1 detection out of 4 trials. The probability is p = 0.25."   |
|                                                               |
| The MARS Point Process Engineer:                              |
| "The satellite only looks for 5 seconds per week. We have     |
| 20 seconds of observational data across a 2,419,200-second    |
| continuous timeline. The true rate λ_true(t) requires us to   |
| integrate the atmospheric blind spot E[P_d] across the whole  |
| month."                                                       |
|                                                               |
| ACTION: The Binomial model assumes the universe ceases to     |
| exist when the satellite looks away. The Point Process        |
| mathematically models the silence between the images.         |
+---------------------------------------------------------------+

```

---

### 5. The MARS Business Translation Engine (The UI Logic)

Now that we are strictly operating in continuous, un-thinned point process space, we can map our business logic.

A government regulator does not want to integrate an inhomogeneous hazard function manually. They want a dashboard with color-coded labels that command specific mitigations. We must translate our core variables—`λ_obs` (the thinned detections), `λ_true(t)` (the un-thinned, fluctuating physical rate), and `E[P_d]` (our atmospheric confidence scalar)—into an automated logic tree.

**The Ontology of MARS Business Labels:**

1. **[ 🔴 PERSISTENT ]**: The infrastructure is continuously or highly frequently failing. The un-thinned `λ_true` is high. Emergency dispatch required.
2. **[ 🟡 INTERMITTENT ]**: The infrastructure is failing, but sporadically or cyclically. Log for routine, timed maintenance based on diurnal peaks.
3. **[ 🟢 MITIGATED ]**: We have mathematical proof that the source is physically quiet.
4. **[ ⚪ UNKNOWN ]**: We see nothing, but only because we are mathematically blind. The ticket remains open.

Here is the pedantic, inhomogeneous logic tree that powers the UI.

```text
=============================================================================
  THE MARS INHOMOGENEOUS PERSISTENCY LOGIC ENGINE
=============================================================================

INPUTS FOR REVIEW PERIOD 'T':
- λ_obs     (Observed, thinned event count from satellite)
- E[P_d]    (Expected Probability of Detection: Sensor limits + Weather)
- λ_true(t) (Calculated un-thinned dynamic rate: λ_obs(t) / E[P_d])

[ START NODE ]
   |
   +-- Condition A: Did we observe ANY leaks? (λ_obs > 0)
   |     |
   |     +-- [ YES ] ---> Evaluate Inhomogeneous Persistency
   |     |                  |
   |     |                  | Calculate expected un-thinned events 
   |     |                  | over the next 48-hour operational window: 
   |     |                  | Λ_future = ∫_{now}^{now+48} λ_true(t) dt
   |     |                  |
   |     |                  +-- Is Λ_future ≥ 2.0 events? (High frequency)
   |     |                  |     |
   |     |                  |     +-->> LABEL: [ 🔴 PERSISTENT ]
   |     |                  |           UI Hook: Display Expected Total Mass 
   |     |                  |           (E[M_total]) and trigger emergency alert.
   |     |                  |
   |     |                  +-- Is Λ_future < 2.0 events? (Low frequency)
   |     |                        |
   |     |                        +-->> LABEL: [ 🟡 INTERMITTENT ]
   |     |                              UI Hook: Display optimal dispatch time 
   |     |                              by targeting the peak of λ_true(t).
   |     |
   |     +-- [ NO ]  ---> Evaluate Atmospheric Confidence (The Filter)
   |                        |
   |                        | We saw nothing. But WHY did we see nothing?
   |                        |
   |                        +-- Is E[P_d] > 0.85? (High confidence, clear skies)
   |                        |     |
   |                        |     +-->> LABEL: [ 🟢 MITIGATED ]
   |                        |           Logic: If it was physically leaking, 
   |                        |           we absolutely would have seen it.
   |                        |
   |                        +-- Is E[P_d] < 0.15? (Low confidence, cloudy)
   |                              |
   |                              +-->> LABEL: [ ⚪ UNKNOWN / BLIND SPOT ]
   |                                    Logic: Silence means nothing here. 
   |                                    The sensor is blinded by the atmosphere.
   |                                    UI Hook: Lock the "Close Ticket" button 
   |                                    to prevent false compliance logging.
=============================================================================

```

---

### 6. The Power of the "Unknown" State (The Blind Spot)

We must heavily emphasize the pedagogical and regulatory importance of the **[ ⚪ UNKNOWN ]** state in business applications. It is the ultimate safeguard against greenwashing.

In naive systems, operators rely on binary logic: *If I see a leak, it is broken. If I do not see a leak, it is fixed.* By explicitly calculating the Expected Probability of Detection (`E[P_d] = ∫₀^∞ P_d(Q) · f_true(Q) dQ`), you mathematically quantify your blindness. If `E[P_d]` collapses to 5% because the region has high wind shear, dense cloud cover, or a low-albedo background (like dark water or snow), your software actively forbids the stakeholder from labeling the site as "Mitigated."

```text
+---------------------------------------------------------------+
| BotE #2: PREVENTING FALSE COMPLIANCE                          |
|---------------------------------------------------------------|
| Facility X had a massive blowout last month.                  |
| Today, the satellite image is completely clear. No methane.   |
|                                                               |
| Operator: "Great, the leak stopped. I'm closing the ticket."  |
|                                                               |
| MARS Backend Check:                                           |
| 1. True distribution f_true(Q) indicates small valve leaks.   |
| 2. Weather data shows 30 km/hr wind shear today.              |
| 3. Calculating integral: E[P_d] = 0.04 (4% confidence).       |
|                                                               |
| ACTION: The UI overrides the operator. It displays [UNKNOWN]. |
| The dashboard reads: "Cannot verify mitigation. Wind shear    |
| has destroyed the sensor's probability of detection. Awaiting |
| future overpass with E[P_d] > 0.80."                          |
+---------------------------------------------------------------+

```

By wrapping Survival Analysis, Expected Mass derivations, and Inhomogeneous Calculus strictly into this automated, unit-perfect logic tree, you transform a stochastic paradox into a bulletproof global compliance engine.

---

### 1. The Two Paradigms: Discrete Looks vs. Continuous Reality

**The Traditional Approach (The Empirical Ratio)**
Calculating "Detections over Observations" (e.g., "we saw a plume on 3 out of 5 clear satellite overpasses") models the system as a series of discrete Bernoulli trials. It naturally forms a Binomial distribution.

* **The Translation:** "What is the probability of seeing the source exactly when I happen to look?"
* **The Metric:** A unitless probability fraction (`p`).

**The Stochastic Approach (The Point Process)**
Treating the emissions as a Temporal Point Process shifts the perspective from your satellite's arbitrary orbital schedule back to the source's continuous physical reality.

* **The Translation:** "At what physical rate does this infrastructure emit gas into the atmosphere, and what is the probability of exactly 'k' events occurring over a continuous timeframe?"
* **The Metric:** The true intensity parameter (`λ`).

```text
=============================================================================
  PARADIGM 1: BINOMIAL AVERAGE (Discrete Satellite Overpasses)
=============================================================================
  You only evaluate the universe at the exact moment the satellite is overhead.
  The spaces between the brackets mathematically do not exist.
  
  Overpass 1    Overpass 2    Overpass 3    Overpass 4    Overpass 5
  [ PLUME ]     [  CLEAR ]    [ PLUME ]     [ PLUME ]     [  CLEAR ]
      1             0             1             1             0

=============================================================================
  PARADIGM 2: TEMPORAL POINT PROCESS (Continuous Physical Reality)
=============================================================================
  You evaluate the continuous timeline. The source operates strictly 
  independent of the satellite's schedule.
  
  Time ----->
  |-------*----*-----------------*---------*-------*-------|
          |    |                 |         |       |
         t₁   t₂                t₃        t₄      t₅
=============================================================================

```

---

### 2. A 1-to-1 Comparison: The Probability (p) vs. The Intensity (λ)

The confusion between these two models almost always stems from a failure to track physical units. These two parameters measure fundamentally different realities.

**The Binomial Parameter: p**

* **The Definition:** The chance of a "Success" on a single, discrete trial.
* **Units:** `[Detections / Overpass]` or `[Dimensionless Fraction, 0.0 to 1.0]`.
* **The Flaw:** It is a ratio of counts to *opportunities*. If you say a leak has a persistency of `p = 0.20`, that number contains absolutely zero physics. It only means that for every 100 times your specific satellite flies over, it happens to catch the plume 20 times.

**The Point Process Parameter: λ (Lambda)**

* **The Definition:** The expected arrival rate of events over a continuous physical timeline.
* **Units:** `[Events / Day]` or `[Plumes / Hour]`.
* **The Power:** It is a ratio of counts to *continuous time*. If you say a leak has an intensity of `λ = 1.5 plumes/day`, you have defined the physical, thermodynamic reality of the infrastructure, completely independent of who is looking at it or when. Furthermore, unlike a probability, an intensity can vastly exceed 1.0 (e.g., `λ = 50 events/day` is perfectly valid).

---

### 3. The Proof of Convergence: From Binomial to Poisson

If these two paradigms measure different things, how are they related?

The profound truth of stochastic geometry is that the Poisson Point Process is simply the Binomial distribution taken to its absolute, infinite limit. We can mathematically prove that if you make your satellite overpasses infinitely fast, the discrete empirical ratio flawlessly collapses into the continuous intensity parameter.

**The Setup:**
Imagine your satellite observes a site for 1 whole day. It currently takes `n` discrete pictures per day.
The probability of seeing a leak in one exact picture is `p`.
Therefore, the expected total number of physical leaks you see in a day is your rate, `λ`.

* **The Equation:** λ = n · p
* **Rearranged for p:** p = λ / n

Now, we calculate the Binomial probability of seeing exactly `k` leaks in `n` total pictures.

* **The Binomial Equation:** P(X = k) = [ n! / (k! · (n - k)!) ] · p^k · (1 - p)^(n - k)

**The Limit (n → ∞):**
What happens if we upgrade our satellite to take an infinite number of pictures per day (`n → ∞`)? Because `λ` (total leaks per day) is a physical constant bound by the pipe's pressure, as the number of pictures `n` goes to infinity, the chance of catching a leak in any exact microsecond (`p`) must approach zero.

Let's substitute `p = λ / n` into the Binomial equation and take the limit as `n → ∞`:

Limit [n→∞] : [ n(n-1)(n-2)...(n-k+1) / k! ] · (λ/n)^k · (1 - λ/n)^n · (1 - λ/n)^(-k)

To see the mathematical collapse, we group the interacting terms and evaluate them as `n` becomes infinitely large:

1. **The Constant Term:** `(λ^k / k!)` contains no `n`, so it remains perfectly unaffected.
2. **The Fraction Term:** We pair the `n` terms from the combinatorial expansion with the `n^k` in the denominator: `(n/n) · ((n-1)/n) · ((n-2)/n) ...` As `n` goes to infinity, subtracting a tiny number like 1 or 2 from infinity is meaningless. Every single one of these fractions evaluates to exactly `1`.
3. **The Negative Exponent Term:** `(1 - λ/n)^(-k)`. Because `λ` divided by infinity is `0`, this becomes `(1 - 0)^(-k)`, which is just `1`.
4. **The Euler Term:** By the fundamental limit definition of Euler's number (`e`), the term `(1 - λ/n)^n` mathematically converges perfectly to `e^(-λ)`.

**The Result:**
When the dust settles and all the `1`s multiply out, the discrete, clunky Binomial formula has collapsed into the exact Probability Mass Function for a continuous Poisson Point Process:

**P(X = k) = ( λ^k · e^(-λ) ) / k!**

```text
=============================================================================
  VISUALIZING THE CONVERGENCE (Squeezing the Binomial)
=============================================================================
  n = 5 overpasses/day (Binomial is rigid, discrete, and blind to the gaps)
  [   ]  [ * ]  [   ]  [ * ]  [   ]

  n = 20 overpasses/day (The gaps shrink. It starts to look like a timeline)
  [ ][ ][*][ ][ ][ ][ ][*][ ][ ][ ][*][ ][ ][ ][ ][ ][*][ ][ ]

  n = ∞ overpasses/day (The Poisson Limit)
  The discrete brackets vanish. You are left with pure physical time.
  |------*--------------*-----------*------------------*-------| -> Continuous t
=============================================================================

```

---

### 4. Why the Point Process is Operationally Superior

Relying on a simple Binomial average (`p`) is highly dangerous for global environmental alerting systems like MARS. Here is exactly why the Thinned Point Process is strictly superior for methane monitoring:

* **Independence from Sensor Revisit Rates:** The simple average heavily depends on your satellite's specific orbit. If Sentinel-2 passes over every 5 days, your "60% persistency" is entirely coupled to that arbitrary 5-day cadence. Switch to a daily satellite, and your empirical ratio will wildly skew. A point process rate (`λ`) models the physical pipe, completely independent of the sensor's polling rate.
* **Handling Asynchronous and Multi-Sensor Data:** If you monitor a site using Sentinel-2 (every 5 days), Landsat (every 8 days), and a continuous ground sensor, averaging "detections per observation" across different instruments creates a mathematically incoherent mess. A Point Process gracefully merges asynchronous data. Every sensor simply applies its own specific atmospheric thinning filter (`E[P_d]`) to update the single, universal ground-truth rate (`λ`).
* **Predictive Power for Wait Times:** A Binomial average looks backward; it only tells you what happened. A Point Process looks forward. Because the time between continuous events follows a probability distribution, once you invert your data to establish `λ`, you can calculate the exact expected wait time (`E[Δt] = 1 / λ`) to optimize when to dispatch your LDAR repair crews.
* **Isolating the Environment from the Asset:** A simple Binomial average mathematically penalizes a site for being cloudy. The Thinned Point Process strictly isolates the environmental filter from the source's physical intensity, allowing the UI dashboard to accurately label a site as `[ ⚪ UNKNOWN / BLIND SPOT ]` rather than falsely granting it a `[ 🟢 MITIGATED ]` compliance status.

### 5. Cross-Domain Equivalencies

This stochastic leap is not a novel academic exercise; it is the gold standard for dynamic systems. If stakeholders are skeptical of abandoning simple ratios, point them to how other rigorous engineering disciplines handle persistency:

* **Radar and Sonar Tracking (FISST):** In Finite Set Statistics, targets entering a sensor's field of view (and the false alarms generated by background clutter) are never modeled as simple ratios. They are modeled as Spatial Poisson Point Processes using Probability Hypothesis Density (PHD) filters.
* **Network Queueing Theory:** Network routers and server farms never use "average hits per observation window" to manage bandwidth. They model persistent request sources using Poisson or Markov-modulated point processes to gracefully handle asynchronous traffic spikes.
* **Geospatial Intelligence (GEOINT):** Persistent activity hotspots (like maritime loitering, illegal fishing, or illicit transshipments) are strictly modeled as spatial-temporal point processes to generate predictive risk heatmaps that account for satellite revisit blind spots.


