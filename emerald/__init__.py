# Emerald Research Group: Spectral Stability Framework
__version__ = "1.2.0"

from emerald.core.spectrum import compute_shell_energies
from emerald.core.constraints import omega_sigma_penalty, get_stability_score
from emerald.engine.simulator import DyadicSimulator
from emerald.engine.controller import StabilityController