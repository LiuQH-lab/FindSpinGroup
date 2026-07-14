from dataclasses import dataclass


DEFAULT_KPOINT_TOL = 1e-5


@dataclass(frozen=True)
class Tolerances:
    space: float = 0.02 # Angstrom
    moment: float = 0.02 # mu_B
    m_eig: float = 0.00002
    occupancy: float = 0.002
    m_matrix_tol: float = 0.01
DEFAULT_TOL = Tolerances()
