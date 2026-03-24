import jax.numpy as jnp
import matplotlib.pyplot as plt
from emerald.engine.simulator import DyadicSimulator
from emerald.engine.controller import StabilityController

def main():
    print("=== EMERALD RESEARCH: DYNAMICAL SYSTEM STABILIZATION ===")
    
    # Initialize the Dyadic State Space Simulator
    sim = DyadicSimulator(n_shells=16, nu=2e-5)
    controller = StabilityController(sim)
    
    # Initial state (Tensor of ones)
    initial_state = jnp.ones(16)
    
    print("\n[Executing Run 1]: Unregularized Baseline...")
    _, baseline_telemetry = controller.run_mission(initial_state, steps=2000, use_arrest=False)
    
    print("[Executing Run 2]: Adaptive Ω-Σ Regularized Flow...")
    _, protected_telemetry = controller.run_mission(initial_state, steps=2000, use_arrest=True)
    
    print("\nGenerating Stability Diagnostics...")
    
    # --- Plotting the DeepMind-Style Dashboard ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    
    # Top Graph: Spectral Variance
    ax1.plot(baseline_telemetry["variance"], label="Unregularized (Divergent)", color="#81D4FA")
    ax1.plot(protected_telemetry["variance"], label="Ω-Σ Regularized", color="#FFF59D")
    ax1.set_yscale("log")
    ax1.set_title("Spectral Variance (UV Divergence Risk)")
    ax1.legend()
    
    # Bottom Graph: System Stability Index
    ax2.plot(protected_telemetry["health"], label="Ω-Σ Governor Active", color="#FFF59D", linewidth=2)
    ax2.set_title("System Stability Index (Ω-Σ Constraint)")
    ax2.set_ylabel("Critical Risk Factor")
    
    plt.tight_layout()
    plt.savefig('stability_diagnostics.png', dpi=300)
    print("Evaluation Complete. Diagnostics saved to 'stability_diagnostics.png'.")

if __name__ == "__main__":
    main()