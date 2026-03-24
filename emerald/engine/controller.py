import jax.numpy as jnp
from emerald.core.constraints import get_stability_score

class StabilityController:
    def __init__(self, simulator):
        self.sim = simulator

    def run_mission(self, initial_state, steps=2000, use_arrest=True):
        current_state = initial_state
        # Initialize the penalty state (L7 memory for the Governor)
        penalty_state = {'prev_variance': 0.0}
        
        # Telemetry logs for the dashboard
        history_variance = []
        history_health = []

        for i in range(steps):
            # Pass the state in, get the new state out
            current_state, penalty_state = self.sim.step(
                current_state, 
                penalty_state, 
                use_constraint=use_arrest
            )
            
            # Extract the variance from the engine's memory
            current_var = penalty_state['prev_variance']
            
            # Log the telemetry
            history_variance.append(current_var)
            history_health.append(get_stability_score(current_var))
            
        # Package the data exactly how run_emerald.py expects it
        telemetry = {
            "variance": jnp.array(history_variance),
            "health": jnp.array(history_health)
        }
            
        return current_state, telemetry