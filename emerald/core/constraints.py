import jax
import jax.numpy as jnp
from jax import jit

@jit
def compute_spectral_variance(p_q, weights):
    mu = jnp.sum(p_q * weights)
    return jnp.sum(p_q * (weights - mu)**2)

@jit
def adaptive_omega_sigma(variance, prev_variance, dt=5e-5, base_strength=1000.0):
    """
    L7 Adaptive Gain Control: Scales arrest force based on the 
    velocity of the spectral collapse.
    """
    # Calculate 'Spectral Velocity'
    velocity = (variance - prev_variance) / dt
    
    # Increase strength if the hallucination is accelerating
    dynamic_boost = jnp.maximum(0.0, velocity * 0.1) 
    effective_strength = base_strength + dynamic_boost
    
    ratio = variance / 1e11
    penalty = 1.0 + effective_strength * jnp.log1p(jnp.maximum(0.0, ratio - 1.0))
    return penalty

@jit
def omega_sigma_penalty(variance, var_limit=1e11, beta=2.0, strength=2000.0):
    """
    Legacy Wrapper: Keeps the engine running using a fixed-strength 
    arrest while we transition to fully adaptive state.
    """
    ratio = variance / var_limit
    multiplier = 1.0 + strength * jnp.log1p(jnp.maximum(0.0, ratio - 1.0))
    return multiplier

@jit
def get_stability_score(variance, var_limit=1e11):
    score = jax.nn.sigmoid(2.0 * (variance / var_limit - 1.0))
    return jnp.clip(score, 0.0, 1.0)