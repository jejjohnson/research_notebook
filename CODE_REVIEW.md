# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify master >/dev/null 2>&1 && echo master || echo main)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

### 1. Code Style and Readability

- Clear, descriptive naming (variables, functions, classes, modules)
- Appropriate function/method length (single responsibility)
- Logical code organization and flow
- Avoidance of deeply nested structures
- Linting via **ruff** (`pixi run lint`)
- Type-hint checking via **ty** (`pixi run typecheck`)

> **Rule of thumb**: Sacrifice *cleverness* for *clarity*. Sacrifice *brevity* for *explicitness*.
> Don't worry about formatting — our CI pipeline (ruff format, pre-commit) handles that automatically.

### 2. Modern Python Idioms (Python ≥ 3.12)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- Walrus operator (`:=`) only when it genuinely improves readability
- `match` statements for pattern matching where appropriate
- Structural pattern matching for complex conditionals
- Context managers (`with` statements) for resource handling
- `dataclasses` or `attrs` for data containers
- `Enum` for fixed sets of constants
- Modern union syntax (`X | Y` instead of `Union[X, Y]`)
- Modern optional syntax (`X | None` instead of `Optional[X]`)
- Built-in generics (`list[int]`, `dict[str, Any]` instead of `List[int]`, `Dict[str, Any]`)

### 3. Packaging and Project Structure

- Proper `pyproject.toml` configuration (PEP 621)
- Appropriate use of `__init__.py` exports
- Clear module boundaries and dependencies
- Correct use of relative vs absolute imports
- Entry points defined properly for CLI tools
- `src/` layout enforced

### 4. Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for **all** public APIs (Google style — be consistent)
- Inline comments explaining *why*, not *what* — except for complex logic or function calls where a brief *what* comment aids comprehension
- Complex algorithms should have step-by-step explanations
- All scientific algorithms should include Unicode equations in docstrings and inline where appropriate (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- All docstrings for public classes and functions should include 2–3 example use cases
- Type hints serve as documentation — ensure they are accurate and complete

### 5. Error Handling

- Specific exception types (never bare `except:`)
- Custom exceptions for domain-specific errors
- Helpful error messages with context
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

### 6. Testing Considerations

- Functions should be easily testable (pure functions where possible)
- Dependencies should be injectable
- Side effects should be isolated and explicit
- Consider edge cases and boundary conditions

### 7. Performance (when relevant)

- Appropriate data structures for the use case
- Generator expressions for large sequences
- Avoid premature optimization
- Note O(n) implications for critical paths

### 8. Security

- No hardcoded secrets or credentials
- Input validation and sanitization
- Safe handling of file paths (no path traversal vulnerabilities)
- Appropriate use of `subprocess` (avoid `shell=True`)

---

## Package Preferences

When reviewing dependency choices or suggesting alternatives, prefer these libraries:

| Purpose | Preferred Package |
|---------|-------------------|
| Logging | `loguru` |
| CLI | `cyclopts` |
| Data containers | `dataclasses` (stdlib) or `attrs` |
| Configuration | `hydra-core` / `omegaconf` |
| Path handling | `pathlib` (stdlib) |
| HTTP | `httpx` |
| Testing | `pytest` |

---

## Python-Specific Checks

When reviewing, specifically verify the patterns below.

### Type Hints

```python
# ❌ Missing type hints
def process_data(items, threshold):
    ...

# ✅ Complete type hints
def process_data(items: list[DataItem], threshold: float) -> ProcessedResult:
    ...
```

### Modern Syntax

```python
# ❌ Old-style
from typing import Optional, Union, List, Dict

def fetch(id: Optional[int] = None) -> Union[Data, None]:
    result: Dict[str, List[int]] = {}

# ✅ Modern (Python 3.12+)
from __future__ import annotations

def fetch(id: int | None = None) -> Data | None:
    result: dict[str, list[int]] = {}
```

### Dataclasses for Data Containers

```python
# ❌ Plain class with boilerplate
class Config:
    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout

# ✅ Dataclass
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Config:
    host: str
    port: int
    timeout: float = 30.0
```

### Path Handling

```python
# ❌ os.path
import os
path = os.path.join(base_dir, "data", filename)
if os.path.exists(path):
    with open(path) as f:
        ...

# ✅ pathlib
from pathlib import Path
path = base_dir / "data" / filename
if path.exists():
    content = path.read_text()
```

### Exception Handling

```python
# ❌ Bare except, poor chaining
try:
    result = parse(data)
except:
    raise RuntimeError("Failed")

# ✅ Specific exceptions, proper chaining
try:
    result = parse(data)
except json.JSONDecodeError as e:
    raise ParseError(f"Invalid JSON in {source}") from e
```

### Explanatory Comments for Complex Logic

```python
# ❌ No explanation for non-obvious algorithm
def calculate_score(items):
    return sum(i.weight * (1 - i.age / 365) for i in items if i.active)

# ✅ Clear explanation of the logic
def calculate_score(items: list[Item]) -> float:
    """Calculate weighted score with time decay.

    Score computation:
        1. Filter to only active items
        2. Apply time decay: items lose relevance linearly over one year
        3. Weight each item's contribution by its assigned weight
        4. Sum all weighted, decayed values
    """
    total = 0.0
    for item in items:
        if not item.active:
            continue
        # Time decay factor: 1.0 for new items → 0.0 after 365 days
        decay_factor = 1 - (item.age / 365)
        total += item.weight * decay_factor
    return total
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
