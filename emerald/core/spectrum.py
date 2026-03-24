import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnums=(0, 1))
def get_dyadic_mask(n_size, shell_idx):
    freqs = jnp.fft.fftfreq(n_size) * n_size
    abs_freqs = jnp.abs(freqs)
    lower = 2.0**shell_idx
    upper = 2.0**(shell_idx + 1)
    mask = (abs_freqs >= lower) & (abs_freqs < upper)
    return mask.astype(jnp.float32)

@partial(jit, static_argnums=(1,))
def compute_shell_energies(signal, n_shells=16):
    n = signal.shape[-1]
    f_signal = jnp.fft.fft(signal)
    
    # We use a standard Python loop here because n_shells is small (16)
    # This avoids the complex 'jax.lax.scan' tracer issues entirely
    energies = []
    for q in range(n_shells):
        mask = get_dyadic_mask(n, q)
        e = jnp.sum(jnp.abs(f_signal * mask)**2) / n
        energies.append(e)
    
    energies_array = jnp.array(energies)
    p_q = energies_array / (jnp.sum(energies_array) + 1e-12)
    return energies_array, p_q