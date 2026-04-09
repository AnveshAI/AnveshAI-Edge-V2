# AnveshAI Edge V2

**A fully offline, hybrid AI tutor for JEE Advanced — Physics, Chemistry, and Mathematics.**

AnveshAI Edge is a terminal-based AI assistant designed to run entirely on-device (CPU only). It combines dedicated deterministic engines for Physics, Chemistry, and symbolic Mathematics with a local knowledge base and a compact 1.5B-parameter language model into a unified hierarchical pipeline. A **chain-of-thought reasoning engine** decomposes every problem before the LLM is invoked, significantly reducing hallucinations for supported deterministic problems and improving answer quality.

---

## Links

- Hugging Face: *[Click Here!](https://huggingface.co/AnveshAI/AnveshAI-Edge-V2)*
- Zenodo: *Coming Soon!*

---

## Why AnveshAI Edge?

- Works fully offline — no internet required after the one-time model download
- Gives step-by-step JEE Advanced-style solutions
- Uses far less compute than cloud AI tools — hybrid deterministic solving avoids sending every query through a large model, reducing compute cost significantly for supported problem types
- Designed for students on low-resource hardware

---

## Benchmark Snapshot (V2)

| Metric | Result |
|--------|--------|
| Overall pass rate | 81.1% |
| Routing accuracy | 100% |
| Average latency | 0.3 ms |
| Main remaining bottleneck | Primarily solver coverage gaps |

---

## Table of Contents

1. [Links](#links)
2. [Architecture Overview](#architecture-overview)
3. [Components](#components)
4. [JEE Advanced Domain Coverage](#jee-advanced-domain-coverage)
5. [Question Bank](#question-bank)
6. [Advanced Math Engine](#advanced-math-engine)
7. [Reasoning Engine](#reasoning-engine)
8. [Getting Started](#getting-started)
9. [Usage Examples](#usage-examples)
10. [Commands](#commands)
11. [Design Principles](#design-principles)
12. [Current Limitations](#current-limitations)
13. [File Structure](#file-structure)
14. [Technical Details](#technical-details)

---

## Architecture Overview

AnveshAI Edge uses a **hierarchical fallback pipeline** with reasoning at every non-trivial stage:

```
User Input
    │
    ├── [/command]      ──► System Handler (instant)
    │
    ├── [arithmetic]    ──► Math Engine (AST safe-eval, instant)
    │
    ├── [physics]       ──► Reasoning Engine: analyze()
    │                          │  (problem type, strategy selection)
    │                          ▼
    │                       Physics Engine (formula solver)
    │                          │  EXACT answer computed
    │                          ▼
    │                       Reasoning Engine: build_math_prompt()
    │                          │  (CoT plan embedded in LLM prompt)
    │                          ▼
    │                       LLM (Qwen2.5-1.5B) → Step-by-step explanation
    │
    ├── [chemistry]     ──► Reasoning Engine: analyze()
    │                          ▼
    │                       Chemistry Engine (stoichiometry, equilibrium, kinetics…)
    │                          ▼
    │                       LLM with CoT context → User
    │
    ├── [advanced math] ──► Reasoning Engine: analyze()
    │                          ▼
    │                       Advanced Math Engine (SymPy — exact symbolic answer)
    │                          ▼
    │                       LLM with CoT context → User
    │
    ├── [knowledge]     ──► Knowledge Engine (local KB)
    │                          ├── match found → User
    │                          └── no match → LLM with CoT context → User
    │
    ├── [logic]         ──► Inference Engine (propositional logic)
    │                          └── LLM with CoT context → User
    │
    └── [conversation]  ──► Conversation Engine (pattern rules)
                               ├── matched → User
                               └── no match → LLM with CoT context → User
```

### Key Design Principle

> **Correctness-first:** For all deterministic problems (math, physics, chemistry), the domain engine computes the exact answer *before* the LLM is called. The LLM's only task is to explain the working — it cannot invent a wrong answer.

---

## Components

| Module | File | Role |
|--------|------|------|
| **Intent Router** | `router.py` | Rule-based intent classifier. Outputs: `system`, `physics`, `chemistry`, `advanced_math`, `math`, `knowledge`, `logic`, `conversation`. Checked in priority order. |
| **Math Engine** | `math_engine.py` | Safe AST-based evaluator for plain arithmetic (`2 + 3 * (4^2)`). No `eval()` — uses a whitelist of allowed AST node types. |
| **Advanced Math Engine** | `advanced_math_engine.py` | SymPy symbolic computation engine. 31+ operation types. Returns `(success, result_str, latex_str)`. |
| **Physics Engine** | `physics_engine.py` | Deterministic formula solver for JEE Advanced Physics: mechanics, SHM, electrostatics, magnetism, modern physics, optics, thermodynamics. |
| **Chemistry Engine** | `chemistry_engine.py` | Deterministic solver for JEE Advanced Chemistry: stoichiometry, equilibrium (Kc/Kp), kinetics, colligative properties, solid state, acid-base. |
| **Reasoning Engine** | `reasoning_engine.py` | Structured reasoning planner using chain-of-thought decomposition. Identifies 64 problem types across 15+ domains, selects strategy, generates ordered sub-steps, assigns confidence, flags warnings. Builds structured LLM prompts. |
| **Inference Engine** | `inference_engine.py` | Propositional logic and deductive reasoning handler. |
| **Knowledge Engine** | `knowledge_engine.py` | Local knowledge-base lookup from `knowledge.txt`. Returns `(response, found: bool)`. |
| **Conversation Engine** | `conversation_engine.py` | Pattern-matching response rules from `conversation.txt`. Returns `(response, matched: bool)`. |
| **LLM Engine** | `llm_engine.py` | Lazy-loading `Qwen2.5-1.5B-Instruct` (GGUF, Q4_K_M, ~1 GB) via `llama-cpp-python`. CPU-only, no GPU required. JEE Advanced system prompt tuned for exam-level rigor. |
| **Memory** | `memory.py` | SQLite-backed conversation history. Powers the `/history` command. |
| **Main** | `main.py` | Terminal REPL loop. Orchestrates all engines. Displays colour-coded output. |

---

## JEE Advanced Domain Coverage

AnveshAI Edge currently supports core JEE Advanced topics, with ongoing solver expansion.

### Physics Topics Handled

| Domain | Problem Types | Example Queries |
|--------|--------------|----------------|
| Mechanics | `physics_problem`, `shm_problem` | Kinetic energy, time period of spring-mass system |
| SHM & Oscillations | `shm_problem` | Spring constant, pendulum period, angular frequency |
| Rotational Mechanics | `rotational_mechanics` | Moment of inertia, angular momentum, rolling motion |
| Modern Physics | `modern_physics` | Photoelectric effect, de Broglie wavelength, Bohr model |
| Electrostatics | `electrostatics` | Electric field, Coulomb's law, potential |
| Electromagnetism | `electromagnetism` | Self-inductance, solenoid, Faraday's law |
| Thermodynamics | `ideal_gas_thermo` | Ideal gas, internal energy, isothermal/adiabatic |

### Chemistry Topics Handled

| Domain | Problem Types | Example Queries |
|--------|--------------|----------------|
| Chemical Equilibrium | `chemical_equilibrium` | Kc, Kp, delta n, ICE tables |
| Chemical Kinetics | `chemical_kinetics` | Rate law, half-life, Arrhenius equation |
| Solid State | `solid_state` | BCC/FCC/SC lattice, packing fraction, atomic radius |
| Colligative Properties | `colligative_properties` | Boiling point elevation, osmotic pressure, van't Hoff |
| Acid-Base / pH | `chemistry_problem` | pH calculations, buffer solutions, Nernst equation |
| General Chemistry | `chemistry_problem` | Stoichiometry, molarity, enthalpy |

### Mathematics Topics Handled

| Domain | Problem Types | Example Queries |
|--------|--------------|----------------|
| Calculus | `integration`, `differentiation`, `limit_evaluation`, `maxima_minima`, `monotonicity` | Integration by parts, implicit diff, L'Hôpital |
| Coordinate Geometry | `conic_parabola`, `conic_ellipse`, `conic_hyperbola`, `circle_geometry`, `straight_line` | Tangent to parabola, foci of ellipse, asymptotes |
| 3D Geometry & Vectors | `3d_geometry`, `vector_algebra` | Shortest distance between skew lines, scalar triple product |
| Trigonometry | `trigonometric_equation`, `inverse_trig`, `trig_identities`, `triangle_trig` | General solution of sin 2x = cos x |
| Algebra | `sequences_series`, `binomial_theorem`, `equation_solving` | AP/GP sums, middle term in binomial expansion |
| Probability & Stats | `bayes_probability`, `probability_problem`, `statistical_analysis` | Bayes theorem, probability distributions |
| ODE & Transforms | `ode_solving`, `laplace_transform`, `fourier_transform` | Solve y'' + y = 0, Laplace of sin(t) |
| Linear Algebra | `matrix_operation`, `eigenvalue` | Determinant, eigenvalues, matrix inverse |

---

## Question Bank

AnveshAI Edge includes **3,000 tagged benchmark questions** (`datasheet.csv`) spanning all three subjects.

| Subject | Questions | Topics Covered |
|---------|-----------|---------------|
| **Physics** | 1,000 | SHM, Mechanics, Rotational Mechanics, Gravitation, Electrostatics, Electromagnetism, Current Electricity, Thermodynamics, Optics, Wave Optics, Waves, Modern Physics, Semiconductors, AC Circuits, Fluid Mechanics, Electromagnetic Waves |
| **Chemistry** | 1,000 | Chemical Equilibrium, Chemical Kinetics, Solid State, Colligative Properties, Acid-Base/pH, Electrochemistry, Thermochemistry, Organic Chemistry, Coordination Chemistry, Periodic Table, Chemical Bonding, Solutions, Surface Chemistry, Industrial Chemistry, Biochemistry, Polymers, Analytical Chemistry, Environmental Chemistry, Metallurgy |
| **Mathematics** | 1,000 | Integration, Differentiation, Limits, Maxima-Minima, Sequences & Series, Binomial Theorem, Complex Numbers, Quadratics/Polynomials, Probability & Distributions, Matrices & Determinants, Differential Equations, Laplace/Fourier Transforms, Number Theory, Statistics, Combinatorics, Coordinate Geometry (Straight Lines, Circles, Parabola, Ellipse, Hyperbola), 3D Geometry, Vectors, Trigonometry |

### Dataset Format (`datasheet.csv`)

```
id, question, subject, topic, expected_intent, expected_problem_type
```

Each row contains a unique question ID, the full question text, subject, topic tag, the expected intent label (used by the router), and the expected problem type (used by the reasoning engine). This dataset is used by `benchmark.py` to evaluate end-to-end routing and classification accuracy.

### Running the Benchmark

```bash
python benchmark.py
```

Outputs `report.csv` with per-question results and a summary block including overall accuracy, per-subject breakdown, and weak-topic identification.

---

## Advanced Math Engine

The engine supports **31+ symbolic mathematics operations** across 11 categories:

### Calculus
| Operation | Example Input |
|-----------|--------------|
| Indefinite integration | `integrate x^2 sin(x)` |
| Definite integration | `definite integral of x^2 from 0 to 3` |
| Differentiation (any order) | `second derivative of sin(x) * e^x` |
| Limits (including ±∞) | `limit of sin(x)/x as x approaches 0` |

### Algebra & Equations
| Operation | Example Input |
|-----------|--------------|
| Equation solving | `solve x^2 - 5x + 6 = 0` |
| Factorisation | `factor x^3 - 8` |
| Expansion | `expand (x + y)^4` |
| Simplification | `simplify (x^2 - 1)/(x - 1)` |
| Partial fractions | `partial fraction 1/(x^2 - 1)` |
| Trig simplification | `simplify trig sin^2(x) + cos^2(x)` |

### Differential Equations
| Operation | Example Input |
|-----------|--------------|
| ODE solving (dsolve) | `solve differential equation y'' + y = 0` |
| First-order ODEs | `solve ode dy/dx = y` |

### Series & Transforms
| Operation | Example Input |
|-----------|--------------|
| Taylor / Maclaurin series | `taylor series of e^x around 0 order 6` |
| Laplace transform | `laplace transform of sin(t)` |
| Inverse Laplace transform | `inverse laplace of 1/(s^2 + 1)` |
| Fourier transform | `fourier transform of exp(-x^2)` |

### Linear Algebra
| Operation | Example Input |
|-----------|--------------|
| Determinant | `determinant of [[1,2],[3,4]]` |
| Matrix inverse | `inverse matrix [[2,1],[5,3]]` |
| Eigenvalues & eigenvectors | `eigenvalue [[4,1],[2,3]]` |
| Matrix rank | `rank of matrix [[1,2,3],[4,5,6]]` |
| Matrix trace | `trace of matrix [[1,2],[3,4]]` |

### Number Theory
| Operation | Example Input |
|-----------|--------------|
| GCD | `gcd of 48 and 18` |
| LCM | `lcm of 12 and 15` |
| Prime factorisation | `prime factorization of 360` |
| Modular arithmetic | `17 mod 5` |
| Modular inverse | `modular inverse of 3 mod 7` |

### Statistics
| Operation | Example Input |
|-----------|--------------|
| Descriptive stats | `mean of 2, 4, 6, 8, 10` |
| Standard deviation | `standard deviation of 1, 2, 3, 4, 5` |

### Combinatorics
| Operation | Example Input |
|-----------|--------------|
| Factorial | `factorial of 10` |
| Binomial coefficient | `binomial coefficient 10 choose 3` |
| Permutations | `permutation 6 P 2` |

### Summations & Products
| Operation | Example Input |
|-----------|--------------|
| Finite sum | `sum of k^2 for k from 1 to 10` |
| Infinite series | `summation of 1/n^2 for n from 1 to infinity` |

### Complex Numbers
| Operation | Example Input |
|-----------|--------------|
| All properties | `modulus of 3 + 4*I` |

---

## Reasoning Engine

The **ReasoningEngine** adds structured chain-of-thought reasoning at every stage.

### Reasoning Pipeline (4 Stages)

**Stage 1 — Problem Analysis**
- Detects domain (15+ domains: calculus, linear algebra, coordinate_geometry, 3d_geometry, trigonometry, statistics, physics, chemistry, computer_science, …)
- Classifies problem type (64 problem types via priority-ordered regex)
- Identifies sub-questions implicit in the problem

**Stage 2 — Strategy Selection**
- Chooses the optimal solution method (u-substitution, L'Hôpital, ICE table, Kp-Kc relation, parallel axis theorem, …)
- Decomposes the problem into an ordered list of numbered reasoning steps

**Stage 3 — Verification & Confidence**
- Assigns confidence: `HIGH` (symbolic/deterministic answer available), `MEDIUM`, or `LOW`
- Detects warnings: missing bounds, undetected variables, potential singularities

**Stage 4 — Prompt Engineering**
- Builds a structured LLM prompt that embeds the full reasoning plan
- For math/physics/chemistry: forces the LLM to follow exact numbered steps toward the verified answer
- For knowledge: guides the LLM through the identified sub-questions in order

### ReasoningPlan Data Structure

```python
@dataclass
class ReasoningPlan:
    problem_type:    str        # e.g. "conic_parabola", "chemical_kinetics"
    domain:          str        # e.g. "coordinate_geometry", "chemistry"
    sub_problems:    list[str]  # ordered reasoning steps
    strategy:        str        # solution method description
    expected_form:   str        # what the answer should look like
    assumptions:     list[str]  # stated assumptions
    confidence:      str        # HIGH / MEDIUM / LOW
    warnings:        list[str]  # potential issues flagged
```

### Problem Types Covered (64 types across 15+ domains)

| Category | Problem Types |
|----------|--------------|
| Calculus | `integration`, `differentiation`, `limit_evaluation`, `ode_solving`, `maxima_minima`, `monotonicity`, `mean_value_theorem` |
| Coordinate Geometry | `conic_parabola`, `conic_ellipse`, `conic_hyperbola`, `conic_section`, `circle_geometry`, `straight_line`, `pair_of_lines` |
| 3D Geometry & Vectors | `3d_geometry`, `vector_algebra` |
| Trigonometry | `trigonometric_equation`, `inverse_trig`, `trig_identities`, `triangle_trig` |
| Algebra | `sequences_series`, `binomial_theorem`, `equation_solving`, `factorization`, `algebraic_expansion`, `simplification`, `summation` |
| Probability & Stats | `bayes_probability`, `probability_problem`, `probability_distribution`, `statistical_analysis` |
| Transforms & Series | `laplace_transform`, `fourier_transform`, `series_expansion` |
| Linear Algebra | `matrix_operation`, `eigenvalue` |
| Number Theory | `number_theory`, `combinatorics` |
| Physics | `shm_problem`, `rotational_mechanics`, `modern_physics`, `electrostatics`, `electromagnetism`, `ideal_gas_thermo`, `physics_problem` |
| Chemistry | `chemical_equilibrium`, `chemical_kinetics`, `solid_state`, `colligative_properties`, `chemistry_problem` |
| Logic & Reasoning | `propositional_logic`, `logical_reasoning`, `analogical_reasoning`, `causal_analysis`, `comparative_analysis` |
| Knowledge | `knowledge_retrieval`, `conceptual_explanation`, `procedural_knowledge` |

### Why Chain-of-Thought Matters for a Small LLM

The Qwen2.5-1.5B model has 1.5 billion parameters. Without structured guidance it can still:
- Skip algebraic steps
- Confuse method with result on multi-step problems
- Apply correct techniques incorrectly

By embedding a detailed reasoning plan in every prompt — and for deterministic problems, by **providing the correct answer upfront** — the model's role becomes that of a *step-by-step explainer* rather than an *unsupervised solver*. This dramatically improves output quality without requiring a larger model.

---

## Getting Started

### System Requirements

| Requirement | Minimum |
|-------------|---------|
| RAM | 4 GB (8 GB recommended) |
| OS | Linux, macOS, Windows (WSL recommended on Windows) |
| Python | 3.9+ |
| Disk space | ~2 GB (model + dependencies) |

> **First run:** The Qwen2.5-1.5B model (~1 GB) is downloaded automatically from HuggingFace and cached locally. Expect 5–15 minutes depending on your connection speed. Subsequent runs load from cache instantly.

### Installation

```bash
pip install sympy llama-cpp-python colorama
```

> **Note for `llama-cpp-python`:** This package compiles C++ code during installation. You may need:
> - A C++ compiler (`gcc` / `g++` on Linux, Xcode CLI tools on macOS)
> - At least 2 GB of free RAM during the build
> - If the build fails, try: `pip install llama-cpp-python --no-cache-dir`

> **Replit users:** Use the AnveshAI Edge workflow — dependencies are pre-configured. RAM may be limited on free-tier instances; the model may load slowly or require a paid plan with higher memory.

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `llama-cpp-python` build fails | Install `build-essential` (Linux) or Xcode CLI tools (macOS) |
| Out of memory on model load | Close other apps; ensure at least 3 GB free RAM |
| Model download stalls | Check your internet connection; delete the partial cache and retry |
| Slow inference | Normal on CPU — first-token latency is ~2–5 s on typical hardware |

### Running

```bash
cd anveshai
python main.py
# try: "integrate x^2 sin(x)"
```

Or use the **AnveshAI Edge** workflow in the Replit environment.

---

## Usage Examples

```
You › A spring of constant 200 N/m has a mass of 0.5 kg attached. Find the time period.

  Physics Engine → T = 2π√(m/k) = 2π√(0.5/200) = 0.314 s
  [Reasoning] domain=physics | type=shm_problem | strategy=SHM formula | confidence=HIGH

AnveshAI [Physics+CoT+LLM] › T ≈ 0.314 s
  Step 1: Identify the spring-mass system parameters: k = 200 N/m, m = 0.5 kg
  Step 2: Apply the SHM time period formula: T = 2π√(m/k)
  Step 3: Substitute: T = 2π√(0.5/200) = 2π × 0.05 = 0.314 s
```

```
You › Kc for N2 + 3H2 ⇌ 2NH3 is 0.5 at 400K. Find Kp.

  Chemistry Engine → Kp = Kc(RT)^Δn,  Δn = 2−4 = −2
  [Reasoning] domain=chemistry | type=chemical_equilibrium | confidence=HIGH

AnveshAI [Chem+CoT+LLM] › Kp = Kc × (RT)^(−2) ≈ 3.27 × 10⁻⁵
  Step 1: Write the balanced equation and count Δn = moles product gas − moles reactant gas
  Step 2: Δn = 2 − (1 + 3) = −2
  Step 3: Apply Kp = Kc(RT)^Δn = 0.5 × (0.0821 × 400)^(−2)
```

```
You › Find the equation of tangent to parabola y²=8x at point (2,4).

  SymPy → Tangent computed
  [Reasoning] domain=coordinate_geometry | type=conic_parabola | confidence=HIGH

AnveshAI [AdvMath+CoT+LLM] › y = x + 2
  Step 1: Identify standard form y²=4ax, so a = 2
  Step 2: Slope of tangent at (x₁, y₁): m = 2a/y₁ = 4/4 = 1
  Step 3: Tangent equation: yy₁ = 2a(x + x₁) → 4y = 4(x + 2) → y = x + 2
```

```
You › integrate x^2 sin(x)

  SymPy → ∫ (x**2*sin(x)) dx = -x**2*cos(x) + 2*x*sin(x) + 2*cos(x) + C
  [Reasoning] domain=calculus | strategy=Apply integration by parts (IBP) | confidence=HIGH

AnveshAI [AdvMath+CoT+LLM] › -x**2*cos(x) + 2*x*sin(x) + 2*cos(x) + C
  Step 1: Use Integration by Parts twice: u = x², dv = sin(x)dx
  …
```

---

## Commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands and usage examples |
| `/history` | Display the last 10 conversation turns |
| `/exit` | Quit the assistant |

---

## Design Principles

1. **Correctness-first for deterministic problems** — The domain engine (SymPy / Physics / Chemistry) always runs before the LLM. The LLM explains, it does not compute.

2. **Offline-first** — All computation runs locally. No API keys, no internet connection required after the one-time model download (~1 GB).

3. **JEE Advanced focus** — System prompts, strategy libraries, and reasoning steps are tuned for JEE Advanced level. Domain scoring prioritises correct sub-domain classification (SHM vs rotational, equilibrium vs kinetics, conics vs 3D).

4. **Transparency** — The system prints its internal reasoning trace to the console (engine used, reasoning plan summary, confidence, warnings).

5. **Graceful degradation** — Every engine has a fallback: SymPy failures fall back to CoT-guided LLM, KB misses fall back to reasoning-guided LLM.

6. **Safety** — Arithmetic uses AST-based safe-eval (no `eval()`). Matrix parsing uses a validated bracket pattern before `eval()`. The LLM prompt explicitly forbids inventing a different answer.

7. **Modularity** — Every engine is independent and communicates through simple return types. Adding a new math operation requires only a new handler function and a keyword entry.

---

## Current Limitations

- Some advanced Physics and Chemistry formula handlers are still being added; solver gaps account for all current benchmark failures
- Open-ended conceptual explanations depend on the 1.5B LLM and may vary in depth or phrasing
- CPU inference speed depends on hardware — responses may take several seconds on lower-end machines
- The model requires at least ~3 GB of free RAM to load comfortably

---

## File Structure

```
anveshai/
├── main.py                 # REPL loop, orchestration, response composer
├── router.py               # Rule-based intent classifier (8 intents)
├── math_engine.py          # Safe AST arithmetic evaluator
├── advanced_math_engine.py # SymPy symbolic engine (31+ operations)
├── physics_engine.py       # JEE Advanced Physics formula solver
├── chemistry_engine.py     # JEE Advanced Chemistry solver
├── reasoning_engine.py     # Structured reasoning planner (64 problem types)
├── llm_engine.py           # Qwen2.5-1.5B-Instruct GGUF loader, JEE tuned prompts
├── inference_engine.py     # Propositional logic / deductive reasoning
├── knowledge_engine.py     # Local KB lookup (knowledge.txt)
├── conversation_engine.py  # Pattern-response engine (conversation.txt)
├── memory.py               # SQLite conversation history
├── knowledge.txt           # Local knowledge base paragraphs
├── conversation.txt        # PATTERN|||RESPONSE rule pairs
└── anveshai_memory.db      # Auto-created SQLite DB (gitignored)
```

---

## Technical Details

### Model
- **Name:** Qwen2.5-1.5B-Instruct
- **Format:** GGUF (Q4_K_M quantisation, ~1 GB)
- **Runtime:** llama-cpp-python (CPU-only via llama.cpp)
- **Context window:** 16,384 tokens
- **Parameters:** 1.5B (3× the original 0.5B for noticeably better reasoning)
- **Threads:** 4 CPU threads
- **System prompt:** JEE Advanced tutor — exam-focused, notation-correct, concise

### Symbolic Engine
- **Library:** SymPy 1.x
- **Parsing:** `sympy.parsing.sympy_parser` with implicit multiplication and XOR-to-power transforms
- **Supported variables:** x, y, z, t, n, k, a, b, c, m, n, p, q, r, s
- **Special constants:** π, e, i (imaginary), ∞

### Chain-of-Thought Reasoning
- **Domain detection:** 15+ domain categories with keyword matching
- **Problem type classification:** 64 problem types via priority-ordered regex
- **Strategy library:** Pre-defined strategies for 64 problem types
- **Decomposition:** Problem-specific step generators for 50+ operation types
- **Confidence levels:** HIGH (deterministic result available) / MEDIUM / LOW
- **JEE-specific types:** SHM, rotational mechanics, modern physics, electrostatics, electromagnetism, chemical equilibrium, chemical kinetics, solid state, colligative properties, conic sections, 3D geometry, Bayes probability, binomial theorem, sequences/series, trig equations, inverse trig, maxima/minima

### Response Labels

| Label | Meaning |
|-------|---------|
| `Math` | Instant arithmetic result |
| `AdvMath+CoT+LLM` | SymPy exact answer + CoT plan + LLM explanation |
| `AdvMath+CoT` | CoT-guided LLM fallback (SymPy failed) |
| `Physics+CoT+LLM` | Physics engine result + CoT plan + LLM explanation |
| `Chem+CoT+LLM` | Chemistry engine result + CoT plan + LLM explanation |
| `Knowledge` | Local KB answer |
| `LLM+CoT-KB` | KB miss → reasoning-guided LLM |
| `Chat` | Conversation pattern match |
| `LLM+CoT` | Reasoning-guided LLM for open conversation |

### Intent Routing

Benchmark routing accuracy on full 3,000-question dataset: **100%**

| Intent | Sample Triggers |
|--------|----------------|
| `physics` | spring constant, moment of inertia, photoelectric effect, de Broglie, electrostatics |
| `chemistry` | Kc, Kp, rate constant, half-life, BCC lattice, boiling point elevation |
| `advanced_math` | parabola, binomial theorem, general solution, shortest distance, Bayes theorem |
| `math` | plain numeric arithmetic expressions |
| `knowledge` | "what is", "explain", "define" — general knowledge |
| `logic` | "if…then…", "given that", logical deduction |
| `conversation` | greetings, small talk |
| `system` | `/help`, `/exit`, `/history` |
