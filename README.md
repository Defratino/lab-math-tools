# Lab Math Tools

A lightweight, robust Python library for numerical calculus and experimental error propagation tailored for physics, chemistry, and engineering laboratory computations.

---

## Features

- **Numerical Differentiation**:
  - Central difference derivatives ($O(h^2)$) for scalar ($\mathbb{R} \to \mathbb{R}$) and vector-valued ($\mathbb{R} \to \mathbb{R}^n$) functions.
  - Partial derivatives, gradients ($\nabla f$), divergence ($\nabla \cdot \mathbf{f}$), and Jacobian matrices ($\mathbf{J}$).
- **Numerical Integration**:
  - 1D composite Trapezoidal rule supporting both scalar and vector-valued functions.
  - $n$-dimensional Monte Carlo volume integration over arbitrary geometric shapes with optional vectorized evaluation and reproducible RNG seeds.
- **Statistical Error & Covariance Propagation**:
  - Sensitivity-based uncertainty propagation using Root-Sum-Square (RSS / statistical) or worst-case absolute addition.
  - Multi-output covariance matrix propagation via Jacobian transformation ($\mathbf{\Sigma}_y = \mathbf{J} \mathbf{\Sigma}_x \mathbf{J}^T$).
  - Variance budget decomposition calculating the fractional contribution of each variable to total variance.
  - Relative / fractional uncertainty calculations.
- **Modern Python Standards**:
  - Full typing annotations with PEP 561 (`py.typed`) support for Pyright/MyPy.
  - Cross-platform automated CI running on Python 3.10–3.12.

---

## Installation

### Directly from GitHub (No cloning needed)

```bash
pip install git+https://github.com/Defratino/lab-math-tools.git
```

To install from a specific branch (e.g. `main` or `dev`):

```bash
pip install git+https://github.com/Defratino/lab-math-tools.git@main
```

### Local / Editable Development Installation

```bash
git clone https://github.com/Defratino/lab-math-tools.git
cd lab-math-tools
pip install -e .
```

To run the test suite:

```bash
pytest
```

---

## Quickstart Guide

### 1. Direct Package Import

All core functions are exported at top-level:

```python
import numpy as np
import lab_math_tools as lmt
```

### 2. Numerical Differentiation

```python
# 1D Derivative: d/dx (x^2) at x = 2.0 -> 4.0
df = lmt.derivative_saap(lambda x: x**2, x=2.0)

# Vector-valued function derivative: d/dx [x^2, x^3] at x = 2.0 -> [4.0, 12.0]
v_df = lmt.derivative_saap(lambda x: np.array([x**2, x**3]), x=2.0)

# Gradient: nabla(x^2 * y) at [2.0, 3.0] -> [12.0, 4.0]
grad = lmt.av_gradient_saap(lambda v: v[0]**2 * v[1], av_x=np.array([2.0, 3.0]))

# Jacobian matrix of R^2 -> R^2: [x + y, x * y]
jac = lmt.am_jacobian_saap(lambda v: np.array([v[0] + v[1], v[0] * v[1]]), av_x=np.array([2.0, 3.0]))
```

### 3. Numerical Integration

```python
# 1D Trapezoidal Integration of x^2 from 0 to 3 -> 9.0
area = lmt.integral_trapezoidal(lambda x: x**2, a=0.0, b=3.0, n_steps=1000)

# Vector-valued Integration: [x, 2x] from 0 to 1 -> [0.5, 1.0]
v_area = lmt.integral_trapezoidal(lambda x: np.array([x, 2*x]), a=0.0, b=1.0)

# Monte Carlo Integration over a custom 2D shape (unit square)
bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
def shape(v): return 1.0 if (0.0 <= v[0] <= 1.0 and 0.0 <= v[1] <= 1.0) else 0.0
mc_vol = lmt.integral_over_shape(lambda v: v[0] + v[1], shape, bounds, n_samples=50000, rng=42)
```

### 4. Error & Covariance Propagation

```python
# Uncertainty of f(x, y) = x * y with errors dx = 0.5, dy = 0.2 at [10.0, 5.0]
vals = np.array([10.0, 5.0])
errs = np.array([0.5, 0.2])

# Statistical (RSS): sqrt((5 * 0.5)^2 + (10 * 0.2)^2) = sqrt(10.25)
delta_f = lmt.propagate_uncertainty_saap(lambda v: v[0] * v[1], vals, errs, method="statistical")

# Fractional variance contribution of each input
weights = lmt.error_contribution_saap(lambda v: v[0] * v[1], vals, errs)

# Covariance matrix propagation: Vy = J * Vx * J^T
vx = np.diag([0.1, 0.2])
vy = lmt.propagate_covariance_saap(lambda v: np.array([v[0] + v[1], v[0] * v[1]]), vals, vx)
```

---

## Package Naming & Type Conventions

### General Writing Conventions
1. All multidimensional variables are defined as `numpy.ndarray`.
    * **Vectors:** Prefixed with `v{n}_`, where `n` is the dimension (e.g., `v3_velocity`).
    * **Matrices:** Prefixed with `m{n}{l}_`, where `n` and `l` are the dimensions (e.g., `m33_rotation`).
    * **Abstract/Dynamic Dimensions:** If dimensions are not pre-defined or accept a range, use `av_` (abstract vector) or `am_` (abstract matrix).

### Custom Function Conventions
Function names indicate their mathematical mapping domains:
1. **Single Input -> Single Output ($\mathbb{R} \to \mathbb{R}$):** Standard naming, `f(x)`
2. **Single Input -> Vector Output ($\mathbb{R} \to \mathbb{R}^n$):** Prefixed with output dimension, `v{n}_f(x)`
3. **Vector Input -> Single Output ($\mathbb{R}^n \to \mathbb{R}$):** Argument prefixed with input dimension, `f(v{n}_x)`
4. **Vector Input -> Vector Output ($\mathbb{R}^n \to \mathbb{R}^m$):** Prefixed with output dimension and argument prefixed with input dimension, `v{m}_f(v{n}_x)`

*(Note: If dimensions are abstract, `av_` replaces the `{n}` or `{m}` prefix).*

### Functions as Inputs (Callbacks)
1. For predefined scripts requiring functional arguments, the argument names will strictly follow the mapping signatures above (e.g., `def optimize(f, v3_x):`).
2. For scripts accepting multiple function inputs of the same type, they will be numbered: `f1`, `f2`, `f3`, etc.
3. If the functional argument must be a vector-valued function, the argument name will be `av_f` or `v{n}_f` depending on whether the vector's dimensions are abstract or pre-defined.
4. If the functional argument is a scalar-valued function, the argument name will be `s_f`.

### Abbreviations & Terminology
* **saap:** Simple Approximation At Point (e.g., `derivative_saap` == Simple Approximation At Point derivative)