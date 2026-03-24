# ❇️ Emerald: Adaptive Spectral Regularization

[![JAX](https://img.shields.io/badge/JAX-Accelerated-blue?logo=google)](https://github.com/google/jax)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Emerald** is a JAX-accelerated framework for the closed-loop stabilization of continuous-time nonlinear dynamical systems. 

Standard simulations of high-dimensional energy transfer, fluid dynamics, and continuous-depth neural networks (Neural ODEs) frequently suffer from pathological spectral bias—resulting in rapid ultraviolet (UV) divergence and finite-time `NaN` collapse. Emerald solves this by introducing the **Adaptive $\Omega-\Sigma$ Governor**, a dynamic constraint mechanism that monitors the temporal derivative of spectral variance and autonomously scales effective dissipation to quench instability before it destroys the forward pass.

## 🚀 Key Features
* **Closed-Loop Spectral Control:** Replaces static weight decay and artificial low-pass truncation with a state-aware, dynamic regularizer.
* **$10^{12}$ Variance Suppression:** Empirically proven to arrest pathological cascade growth, suppressing high-frequency energy concentration by 12 orders of magnitude.
* **JAX-Native Architecture:** Fully vectorized 4th-Order Runge-Kutta (RK4) integration, utilizing Just-In-Time (`@jit`) compilation for maximum hardware utilization on TPUs and GPUs.
* **Continuous Active Equilibrium:** Stabilizes highly parameterized state spaces while preserving the expressivity of the underlying nonlinear transport.

---

## 📦 Installation

Emerald requires Python 3.10+ and JAX. 

```bash
# Clone the repository
git clone https://github.com/andrewkkim58-afk/emerald-research.git
cd emerald-research

# Install dependencies (CPU version by default, see JAX docs for CUDA/TPU)
pip install jax jaxlib matplotlib

### Expected Output
The script will compute the dyadic forward passes and generate `stability_diagnostics.png`. 

1. **Top Graph (Spectral Variance):** You will observe the Unregularized baseline rapidly hit numerical infinity (`NaN`), disappearing from the plot. The Protected engine will successfully repel this trajectory, maintaining a bounded variance of $\mathcal{O}(10^1)$.
2. **Bottom Graph (System Stability Index):** Demonstrates the adaptive constraint finding a continuous active equilibrium (typically $\approx 0.119$), proving the system is stabilized without being artificially frozen.

![System Stability Diagnostics](stability_diagnostics.png)