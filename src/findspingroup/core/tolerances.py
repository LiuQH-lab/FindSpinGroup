from dataclasses import dataclass

@dataclass(frozen=True)
class Tolerances:
    space: float = 0.02 # Angstrom
    moment: float = 0.02 # mu_B
    m_eig: float = 0.004
    occupancy: float = 0.002
    m_matrix_tol: float = 0.01
DEFAULT_TOL = Tolerances()