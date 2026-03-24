import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from emerald.core.spectrum import compute_shell_energies
from emerald.core.constraints import compute_spectral_variance, adaptive_omega_sigma

class DyadicSimulator:
    def __init__(self, n_shells=16, nu=2e-5, c_trans=45.0, dt=5e-5):
        self.n_shells = n_shells
        self.nu = nu
        self.c_trans = c_trans
        self.dt = dt
        self.weights = 4.0 ** jnp.arange(n_shells)

    @partial(jit, static_argnums=(0, 3))
    def _compute_derivatives(self, a, prev_variance, use_constraint):
        # 1. Transport & Forcing
        inflow = self.c_trans * jnp.roll(a**1.2, 1).at[0].set(0.0)
        outflow = self.c_trans * (a**1.2)
        forcing = jnp.zeros(self.n_shells).at[1].set(8.0).at[2].set(4.0)

        # 2. Adaptive Ω-Σ Logic
        effective_nu = self.nu
        current_var = 0.0
        
        if use_constraint:
            _, p_q = compute_shell_energies(a, self.n_shells)
            current_var = compute_spectral_variance(p_q, self.weights)
            # Use the Adaptive L7 Logic: Penalty depends on Velocity
            penalty = adaptive_omega_sigma(current_var, prev_variance, self.dt)
            effective_nu = self.nu * penalty

        dissipation = -effective_nu * self.weights * a
        return forcing + (inflow - outflow) + dissipation, current_var

    def step(self, a, penalty_state, use_constraint=True):
        """
        L7 Step: Requires 'penalty_state' to track variance history.
        """
        dt = self.dt
        prev_var = penalty_state['prev_variance']
        
        # RK4 Integration with state tracking
        k1, var_k1 = self._compute_derivatives(a, prev_var, use_constraint)
        k2, _ = self._compute_derivatives(a + 0.5 * dt * k1, var_k1, use_constraint)
        k3, _ = self._compute_derivatives(a + 0.5 * dt * k2, var_k1, use_constraint)
        k4, _ = self._compute_derivatives(a + dt * k3, var_k1, use_constraint)
        
        new_a = jnp.maximum(a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 1e-16)
        
        # Update state for the next step
        new_state = {'prev_variance': var_k1}
        return new_a, new_state