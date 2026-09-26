"""Numerical kernels of accepted three-vector operation constraints."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VectorConstraintSpace:
    basis: np.ndarray
    physical_basis: np.ndarray
    singular_values: np.ndarray
    rank: int
    threshold: float
    max_operation_residual: float
    distinct_blocks: int
    frame_condition: float
    roundoff_allowance: float

    @property
    def dimension(self):
        return 3 - self.rank

    def diagnostics(self):
        minimum_nonzero = float(self.singular_values[self.rank-1]) if self.rank else None
        maximum_zero = float(self.singular_values[self.rank]) if self.rank < len(self.singular_values) else None
        near_boundary = ((minimum_nonzero is not None and minimum_nonzero < 10*self.threshold)
                         or (maximum_zero is not None and maximum_zero > self.threshold/10))
        return {
            "norm": "maximum_unit_vector_action_error_per_operation",
            "stack_weighting": "rms_distinct_nonzero_operation_blocks",
            "rank": self.rank,
            "dimension": self.dimension,
            "threshold": self.threshold,
            "singular_values": self.singular_values.tolist(),
            "min_constrained_singular": minimum_nonzero,
            "max_null_singular": maximum_zero,
            "separation_status": "near_threshold" if near_boundary else "well_separated",
            "singular_gap": (minimum_nonzero/maximum_zero
                             if minimum_nonzero is not None and maximum_zero and maximum_zero > 0 else None),
            "max_operation_residual": self.max_operation_residual,
            "acceptance_margin": self.threshold-self.max_operation_residual,
            "distinct_blocks": self.distinct_blocks,
            "frame_condition": self.frame_condition,
            "roundoff_allowance": self.roundoff_allowance,
        }


def solve_vector_constraints(stacked, *, tol, frame=None):
    """Solve C_i v=0 with one physical budget for each 3x3 action block.

    ``frame`` maps supplied vector components to an orthonormal physical frame.
    Duplicate/zero blocks do not vote on rank. An RMS singular threshold proposes
    the kernel, then every original block must satisfy the maximum action-error
    budget on that kernel. A failed check is diagnosed, never silently repaired
    by widening the threshold or dropping an operation.

    A short unframed equation matrix is padded with zero rows for legacy callers.
    Framed inputs must contain complete operation blocks, not arbitrary scalar
    equations whose codomain metric is unspecified.
    """
    tolerance = float(tol)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Constraint tolerance must be finite and nonnegative.")
    matrix = np.asarray(stacked, dtype=float)
    if matrix.size == 0:
        matrix = np.zeros((0, 3))
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.all(np.isfinite(matrix)):
        raise ValueError("Vector constraints must be a finite matrix with three columns.")
    basis_frame = np.eye(3) if frame is None else np.asarray(frame, dtype=float)
    if basis_frame.shape != (3,3) or not np.all(np.isfinite(basis_frame)):
        raise ValueError("Constraint frame must be a finite 3x3 matrix.")
    condition = float(np.linalg.cond(basis_frame))
    if not math.isfinite(condition) or condition*np.finfo(float).eps >= 1:
        raise ValueError("Constraint frame must be numerically nonsingular.")
    inverse = np.linalg.inv(basis_frame)
    if len(matrix) % 3:
        if frame is not None:
            raise ValueError("Framed constraints require complete 3x3 operation blocks.")
        matrix = np.pad(matrix, ((0, 3-len(matrix)%3), (0,0)))
    blocks = matrix.reshape(-1,3,3)
    physical = basis_frame @ blocks @ inverse
    scale = max(1., float(np.max(np.abs(physical))) if physical.size else 0.)
    roundoff = 64*np.finfo(float).eps*scale
    distinct = np.unique(physical.reshape(-1,9), axis=0).reshape(-1,3,3)
    distinct = distinct[np.max(np.abs(distinct), axis=(1,2)) > roundoff]
    if len(distinct):
        weighted = distinct.reshape(-1,3)/np.sqrt(len(distinct))
        # There are always at least three rows; do not allocate a square U.
        _, singular, vh = np.linalg.svd(weighted, full_matrices=False)
        rank = int(np.count_nonzero(singular > tolerance))
        physical_basis = vh[rank:].T
    else:
        singular, rank, physical_basis = np.zeros(3), 0, np.eye(3)
    if rank == 3 or not len(physical):
        residual = 0.
    else:
        residual = float(np.max(np.linalg.norm(physical @ physical_basis, ord=2, axis=(1,2))))
    if residual > tolerance + roundoff:
        raise ValueError(
            "Unresolved vector-constraint rank: the RMS candidate kernel violates "
            f"a full-operation budget (residual={residual:.12g}, tol={tolerance:g}, "
            f"rank={rank}, singular_values={singular.tolist()})."
        )
    return VectorConstraintSpace(inverse @ physical_basis, physical_basis, singular,
                                 rank, tolerance, residual, len(distinct), condition, roundoff)
