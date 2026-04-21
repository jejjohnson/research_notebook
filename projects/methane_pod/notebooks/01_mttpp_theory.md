---
title: "MTTPP — thinned marked temporal point processes"
---

# Thinned Marked Temporal Point Processes

Let's build this mathematical architecture from the bedrock up. To truly understand a "Thinned Marked Temporal Point Process" in the context of global methane infrastructure, we must assemble it piece by piece. We will define the physical reality, assign the strict mathematical variables, and visualize how each atmospheric layer violently transforms the one below it.

Here is the pedantic, step-by-step construction of the model, expanded to reveal the rigorous mechanics operating beneath the surface of the MARS platform.

---

### Step 1: The Temporal Point Process (The "When")

Before we care about how big a leak is, or if a satellite can see it, we must first mathematically model the sheer existence of events in time.

Physically, imagine a pressure-relief valve on a liquid natural gas storage tank. As pressure builds and drops, the valve periodically "burps" methane into the atmosphere. Each burp is a discrete, instantaneous event occurring at a specific moment.

**The Ontology (The Temporal Truth):**

* **t:** The continuous timeline `[hours]`. This operates on the domain [0, ∞).
* **t_i:** The exact timestamp of the *i*-th emission event.
* **N_true(t):** The True Counting Process `[events]`. The total accumulated number of physical events that have occurred from time 0 up to time `t`. Mathematically, it is a right-continuous step function that jumps by exactly +1 at every `t_i`.
* **λ_true(t):** The True Intensity Function `[events / hour]`. This governs how rapidly events are physically arriving. If the tank heats up at noon, `λ_true(t)` increases. If it cools at night, `λ_true(t)` decreases.

**The Translation:** The True Cumulative Intensity Function `[events]` is the expected total number of physical events up to time `t`. It is the fundamental link between the rate and the count, defined strictly as the integral of the intensity over the time window.
**The Equation:** Λ_true(t) = ∫₀^t λ_true(u) du

```text
=============================================================================
  STEP 1: THE TEMPORAL POINT PROCESS (The Pure Chronology)
=============================================================================
  The pure chronology of the methane valve venting. 
  Every "x" is an event occurring at a specific time t_i.
  
  Timeline:
  --x-------x-----------x----x--x----------x-----------------x------x---> t [hours]
    t₁      t₂          t₃   t₄ t₅         t₆                t⇇     t₈

  NOTICE: We know exactly WHEN they happened, but we know nothing else.
  The points have no mass. They are purely temporal coordinates.
=============================================================================

```

---

### Step 2: Adding the "Marks" (The "What")

A timestamp alone is physically meaningless for a greenhouse gas inventory. To model reality, every temporal event must carry a physical payload. In stochastic geometry, this payload is called a "Mark."

Physically, the mark is the severity of the valve's burp—the exact mass flux or emission rate of the methane plume.

**The Ontology (The Physical Truth):**

* **Q_i:** The True Mark `[kg/hr]`. The specific, actual emission rate associated with the event at time `t_i`. It exists in the Mark Space (a set of all possible emission rates, typically all real numbers > 0).
* **f_true(Q):** The True Mark Distribution (PDF) `[unitless]`. The underlying probability distribution that dictates the physical size of these burps. For methane, this is famously a heavy-tailed Lognormal distribution—most burps are tiny, a few are massive blowouts.
* *Constraint:* The total area under the curve must equal exactly 1.0 (∫₀^∞ f_true(Q) dQ = 1.0).



**The Translation:** The True Expected Mark `[kg/hr]` is the mathematical center of mass of the true physical leaks. It is the integral of the flux rate multiplied by its true probability distribution.
**The Equation:** E[Q_true] = ∫₀^∞ Q · f_true(Q) dQ

* **λ_true(t, Q):** The True Joint Intensity Function. Assuming the size of the leak is independent of when it happens, the entire physical system is defined by multiplying the temporal rate by the mark distribution: `λ_true(t, Q) = λ_true(t) · f_true(Q)`.

```text
=============================================================================
  STEP 2: THE MARKED TEMPORAL POINT PROCESS (Adding Mass)
=============================================================================
  Every event t_i now carries a true physical weight Q_i.
  Our 1D timeline gains a Y-axis. Every event becomes a vertical stem.
  
  Q [kg/hr]
   ^
   |        [Q₂]                                             [Q₇]
   |         |                                                |
   |         |          [Q₃]                                  |
   |         |           |   [Q₄]                             |
   |   [Q₁]  |           |    |            [Q₆]               |
   |    |    |           |    |  [Q₅]       |                 |     [Q₈]
 --+----x----x-----------x----x---x---------x-----------------x------x---> t [hours]
        t₁   t₂          t₃   t₄  t₅        t₆                t⇇     t₈
=============================================================================

```

---

### Step 3: Independent Thinning (The "Filter")

Now we introduce the observer (e.g., the MARS satellite network). The observer is imperfect. It cannot see every plume. The physical reality must pass through an observational filter.

Physically, a tiny puff of methane (a small mark) will be immediately dispersed by wind shear and fall below the satellite's pixel resolution. A massive blowout (a large mark) will almost certainly trigger an anomaly.

This process of selective deletion is called **Thinning**. Mathematically, it invokes a beautiful theorem: Independent Thinning splits the original point process into two entirely separate, independent point processes—the "Observed" process and the "Hidden" process.

**The Ontology (The Sensor Limit):**

* **P_d(Q):** The Probability of Detection `[unitless fraction]`. This is a conditional probability function. Given a true plume of size `Q`, what is the mathematical probability [0.0 to 1.0] that the sensor registers it?

**The Bernoulli Trial:** For every single marked point `(t_i, Q_i)` generated by the source, the universe flips a weighted coin. The probability of "Heads" (surviving the atmospheric filter) is exactly `P_d(Q_i)`. If it lands "Tails", the point is permanently deleted from our observed dataset and banished to the Hidden process.

```text
=============================================================================
  STEP 3: THE THINNING PROCESS (The Observational Sieve)
=============================================================================
  The sensor applies the P_d(Q) filter. 
  Small leaks have a near-zero chance of survival.
  
  Q [kg/hr]
   ^
   |        [Q₂] <---- (Massive leak. P_d = 0.99. SURVIVES)  [Q₇] <-(SURVIVES)
   |         |                                                |
   |         |          [Q₃] <---- (Medium. P_d = 0.40. LOST) |
   |         |           |   [Q₄] <--- (Small. P_d = 0.05. LOST)
   |   [Q₁]  |           |    |            [Q₆] <--- (Medium. SURVIVES)
   |    |    |           |    |  [Q₅]       |                 |     [Q₈]
 --+----x----x-----------x----x---x---------x-----------------x------x---> t [hours]
       LOST SURVIVED    LOST LOST LOST   SURVIVED          SURVIVED LOST
=============================================================================

```

---

### Step 4: The Final State (The Thinned Marked Process)

We have arrived at the final dataset sitting on the MARS servers.

The physical process has been temporally sparsified (events appear to occur much less frequently) and its marks have been violently biased (only the large burps remain). To calculate exactly how the data is warped, we must integrate out the dependencies and compare them to the True ontology defined in Steps 1 and 2.

**The Ontology (The Observed Reality):**

**The Translation:** The Expected Probability of Detection `[unitless scalar]` is the denominator that anchors our skewed distribution, found by integrating the detection curve against the true physical leak sizes.
**The Equation:** E[P_d] = ∫₀^∞ P_d(Q) · f_true(Q) dQ

* **λ_obs(t):** The Observed Intensity `[events / hour]`. The rate at which MARS actually records detections. It is the true intensity severely crippled by the overall expectation of detection: `λ_obs(t) = λ_true(t) · E[P_d]`

**The Translation:** The Observed Mark Distribution `[unitless]` is the skewed distribution of the plumes MARS caught. We multiply the physical truth by the hardware filter, and divide by the expectation scalar to force the area under the new curve to remain 1.0.
**The Equation:** f_obs(Q) = ( P_d(Q) · f_true(Q) ) / E[P_d]

**The Translation:** The Observed Expected Mark `[kg/hr]` is the center of mass of the leaks the satellite *actually saw*. Because the small leaks were deleted by the Bernoulli trial, this is mathematically forced to be drastically larger than `E[Q_true]`.
**The Equation:** E[Q_obs] = ∫₀^∞ Q · f_obs(Q) dQ

```text
=============================================================================
  STEP 4: THE THINNED MARKED TEMPORAL POINT PROCESS (The MARS Data)
=============================================================================
  This is the final mathematical object you are forced to work with. 
  The source appears to burp rarely, but when it does, the burps 
  appear massive. 
  
  Q_obs [kg/hr]
   ^
   |        [Q₂]                                             [Q₇]
   |         |                                                |
   |         |                                                |
   |         |                                                |
   |         |                             [Q₆]               |
   |         |                              |                 |     
 --+---------x------------------------------x-----------------x----------> t [hours]
             t₂                             t₆                t⇇     
=============================================================================

```

### The Pedagogical Summary

* **Temporal:** The physical infrastructure creates a true, un-thinned timeline of events (`t`).
* **Marked:** Physics dictates that every event must carry a true, physical emission mass (`Q`).
* **Thinned:** The atmosphere and the satellite sensor act as a probabilistic sieve, discarding events based on their mass (`P_d`), leaving MARS with a mathematically warped Observed timeline and heavily biased Mark distribution.

To successfully mitigate a methane facility, you cannot trust Step 4. You must take the biased, sparse data from Step 4 and use stochastic inversion to mathematically reverse-engineer it all the way back to the physical reality of Step 2.

---

This primer successfully builds the model from the ground up, providing the foundational context needed before hitting them with the Missing Mass Paradox.
