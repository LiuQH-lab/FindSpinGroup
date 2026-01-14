from dataclasses import dataclass

@dataclass(frozen=True)
class Tolerances:
    space: float = 0.02 # Angstrom
    moment: float = 0.02 # mu_B
    m_eig: float = 0.004
    occupancy: float = 0.002

DEFAULT_TOL = Tolerances()