from dataclasses import dataclass
import math


DEFAULT_KPOINT_TOL = 1e-5


@dataclass(frozen=True)
class Tolerances:
    space: float = 0.02 # Angstrom
    moment: float = 0.02 # mu_B
    m_eig: float = 0.00002
    occupancy: float = 0.002
    m_matrix_tol: float = 0.01

    def __post_init__(self):
        for name in ("space", "moment", "m_eig", "occupancy", "m_matrix_tol"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} tolerance must be a finite number.") from exc
            allows_zero = name in {"moment", "occupancy"}
            if not math.isfinite(value) or value < 0 or (value == 0 and not allows_zero):
                sign = "nonnegative" if allows_zero else "positive"
                raise ValueError(f"{name} tolerance must be finite and {sign}.")
            object.__setattr__(self, name, value)

DEFAULT_TOL = Tolerances()
