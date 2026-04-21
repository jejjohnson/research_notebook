---
name: Research / Comparative Analysis
about: Investigate an external repo, paper, or prior art and map it onto this project. Produces a prioritized roadmap of follow-up issues.
title: "research: <short topic>"
labels: ["type:research"]
---

<!--
STYLE NOTES
- Use unicode math inline in prose (σ², E₁, ∑, ⊗, ≈, Λ⁻¹, ∂/∂x, ℝ, O(d³))
  rather than LaTeX / MathJax blocks. Keeps the issue readable in the GH UI
  and in plain-text tools. Reach for `text`-tagged code fences only for
  multi-line equations or when subscripts are too dense for unicode:

      ```text
      s_next   = (1 - ρ) * s   + ρ * (s₀   - h)
      η_next   = (1 - ρ) * η   + ρ * (η₀ + g - h * m)
      ```

- Numbered top-level sections (## 1. ..., ## 2. ..., etc.) give the issue
  a citeable table of contents.
- Letter-then-number subsections (### A., ### B., then #### A1., #### A2.,
  #### B1., ...) make findings individually linkable ("see A3" reads
  naturally across the project).
- Lead tables and code-first sections with the concrete content; keep
  prose short and after the snippet / table.
- Delete sections that don't apply. Every placeholder is optional.
- Rename sections if the content type warrants (e.g. rename §1 to
  "What `<paper>` Proves" for a theory survey, or §2 to
  "Comparison with Prior Art" when surveying multiple sources).

Research issues produce the plan; the follow-up feature / design issues
they open do the actual work.
-->

# <Title — e.g. "Comparative Analysis: `<external-repo>` vs `<this project>`">

## Context

<!--
Two short paragraphs. What are we investigating, what's the motivating
question, and what decision / roadmap does this research inform?
-->

---

## 1. What `<subject>` Contains

### Package / Project Structure
```
<tree or component map, e.g.>
<subject>/
├── core/
│   └── <module>.py
├── algorithms/
│   └── <module>.py
└── utils.py
```

### Core Data Structures

| Structure | Fields | Purpose |
|---|---|---|
| `<name>` | `<fields>` | <role — e.g. Gaussian in square-root form, μ + cholΣ> |

### Core Algorithms / Features

#### A. <Algorithm area — e.g. "Square-root filtering">
- `<function>(args)` — <what it does>. Complexity O(d³) per step. Returns `(m, cholP)`.
- `<function>(args)` — <what it does>. Uses the Bonnet–Price identity ∇μ 𝔼[ℓ] = g − Hm.

#### B. <Algorithm area>
- `<function>(args)` — <what it does>. Parallel via `jax.lax.associative_scan`, gives O(d³ log N) depth on P processors.

---

## 2. Comparison with `<this project>`

### A. Already in `<this project>` (direct equivalents)

| Subject feature | This project's equivalent | Path | Notes |
|---|---|---|---|
| `<feature>` | `<our name>` | `src/<path>:<line>` | <gap or divergence> |

### B. Already in `<this project>` but missing enhancements from `<subject>`

#### B1. <Enhancement name> (HIGH PRIORITY)
- **`<subject>`**: uses <approach>; e.g. σ² = (1/Nd) ∑ rᵢᵀ rᵢ for posterior calibration.
- **This project**: <current state — what's different, what regresses>.
- **What's needed**: <concrete change, file paths, function signatures>.
- **Impact**: <why this matters — correctness / speed / stability>.

#### B2. <Enhancement name> (MEDIUM PRIORITY)
- ...

### C. Missing completely from `<this project>`

#### C1. <Feature name> (HIGH PRIORITY)
- **What it is**: <description — e.g. Integrated Wiener Process prior with Nordsieck preconditioner P = diag(dtq+½ / q!) for numerical stability on stiff problems>.
- **Why useful**: <motivation>.
- **Where in this project**: proposed module `src/<package>/<submodule>.py`.

#### C2. <Feature name> (MEDIUM PRIORITY)
- ...

### D. `<this project>` has it, `<subject>` doesn't (context only — no action)
<!-- Useful to note we haven't regressed on things the subject omits. -->
- `<our feature>` — <why the subject doesn't need it>.

---

## 3. Summary Table

| Feature | `<subject>` | `<this project>` | Status |
|---|---|---|---|
| Sequential filter | ✓ | ✓ | **Already have** |
| Parallel filter (associative scan) | ✓ | ✗ (dense only) | **Enhancement needed** |
| Square-root form | ✓ | ✗ | **Missing** |
| `tria()` QR utility | ✓ | ✗ | **Missing** (dependency for sq-root work) |
| Block-tridiag operators | ✗ | ✓ | We're ahead |

---

## 4. Recommended Integration Priority

<!-- Phased roadmap. Each phase should be shippable on its own. -->

### Phase 1: Foundations (enables everything else)
1. <item — e.g. `tria()` lower-triangular QR utility>
2. <item>

### Phase 2: <theme — e.g. "Square-root Kalman">
3. <item — e.g. sqrt_predict, sqrt_update, sqrt_smooth primitives>

### Phase 3: <theme — e.g. "Parallel-in-time">
4. <item>

### Phase 4: <theme>
5. <item>

### Phase 5: Advanced (nice-to-have)
6. <item>

---

## 5. Proposed Follow-up Issues

<!--
Concrete issues to open from this research. These are what actually get
done. Each should cite the phase / letter-number it resolves, and link
back to this issue via `Parent research: #<this>`.
-->

- [ ] `feat(<scope>): add tria() QR utility` — covers Phase 1 item 1 / C8
- [ ] `feat(<scope>): square-root Kalman primitives` — covers Phase 2 / B2
- [ ] `[Design] associative-scan element representation` — resolves open question for Phase 3 / C3
- [ ] `feat(<scope>): IWP prior with Nordsieck preconditioner` — covers Phase 4 / C1

---

## 6. Key Synergies (optional)

<!--
Narrative section for cross-cutting observations — how combining X with
Y unlocks Z, architectural implications, dependency chains between the
missing pieces. Delete if not useful.
-->

1. **<Synergy name>**: <one paragraph>.
2. **<Synergy name>**: <one paragraph>.

---

## References
- <paper / repo / doc> — `<url>`
- <paper / repo / doc> — `<url>`

## Relationships
<!--
Apply native GitHub links after the issue is opened.

  Parent:      → Sub-issue link (research issue nested under a theme epic).
                 make gh-sub PARENT=<parent#> CHILDREN="<this#>"
  Blocks:      → Each follow-up issue from §5 should reference THIS research
                 issue as its blocker. From the follow-up:
                 make gh-block ISSUE=<follow-up#> BLOCKED_BY=<this#>
  Blocked by:  → make gh-block ISSUE=<this#> BLOCKED_BY=<other#>
  Related:     → Prose only; no native feature.

Helper: `.github/scripts/link-issues.sh`.
-->
- Parent (theme epic, if any): #
- Blocked by: #
- Blocks (follow-up issues from §5 reference this as parent research): #
- Related: #
