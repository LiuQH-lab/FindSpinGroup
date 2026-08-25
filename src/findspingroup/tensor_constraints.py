"""FindSpinGroup-specific response-tensor constraint contracts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from spintensor import build_bcd_extra_constraints, solve_tensor_constraints


def solve_wpd_qmd(
    operations: Sequence[Any],
    n: int = 3,
    tol: float = 1e-10,
    spin_axes: tuple[int, ...] | None = None,
    det_tol: float = 1e-6,
    symbol: str = r"Q^{WPD}",
) -> tuple[np.ndarray, np.ndarray, list[str], list]:
    """Solve the WPD/Qiang interband QMD conductivity constraints.

    This solver intentionally coexists with ``spintensor.solve_qmd``:

    * both tensors are rank-3, time-reversal odd, transform with ``det(Rs)``
      times the rank-3 tensor action of ``Rr``, and are symmetric in the two
      electric-field indices;
    * the legacy Zhu/Das-QK tensor imposes no further intrinsic permutation
      identity and therefore also contains a fully symmetric, potentially
      longitudinal BCPD sector;
    * the Gao wave-packet-dynamics (WPD) interband QMD additionally belongs to
      the cyclic mixed-symmetry sector::

        W[i,j,k] = W[i,k,j]
        W[i,j,k] + W[j,k,i] + W[k,i,j] = 0

    Consequently ``W[i,i,i] = 0`` and ``j(E) . E = 0`` for the WPD sector.
    As representation spaces, legacy QK-QMD decomposes into this WPD sector
    plus a fully symmetric sector.  The latter has the same symmetry constraints
    as IMD, but is physically QK BCPD rather than an inverse-mass contribution.

    References
    ----------
    Gao, Yang, and Niu, Phys. Rev. Lett. 112, 166601 (2014),
    doi:10.1103/PhysRevLett.112.166601.
    Das et al., Phys. Rev. B 108, L201405 (2023),
    doi:10.1103/PhysRevB.108.L201405.
    Zhu et al., Nat. Commun. 16, 4882 (2025),
    doi:10.1038/s41467-025-60128-2.
    Qiang et al., Adv. Sci. 13, e14818 (2026),
    doi:10.1002/advs.202514818.

    This routine implements only the symmetry-allowed WPD tensor subspace.  It
    does not evaluate a microscopic Brillouin-zone kernel or claim that WPD is
    an exhaustive theory of every intrinsic quantum-metric contribution.
    """

    return solve_tensor_constraints(
        transformations=operations,
        n=n,
        r=3,
        symmetries=[(0, 2, 1)],
        tol=tol,
        spin_axes=spin_axes,
        T_constraint=1,
        det_tol=det_tol,
        symbol=symbol,
        extra_constraints=build_bcd_extra_constraints(n),
    )


__all__ = ["solve_wpd_qmd"]
