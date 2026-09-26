"""Periodic distances in a physical lattice, without modifying that lattice."""

from functools import lru_cache
import itertools
import math

import numpy as np
from spglib import niggli_reduce


@lru_cache(maxsize=256)
def _distance_context(lattice_key):
    lattice = np.asarray(lattice_key, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(lattice)):
        raise ValueError("Periodic distances require a finite lattice.")
    singular = np.linalg.svd(lattice, compute_uv=False)
    if singular[-1] <= np.finfo(float).eps * singular[0]:
        raise ValueError("Periodic distances require a numerically nonsingular lattice.")

    # Reduction only chooses an auxiliary basis for the same translation
    # lattice. It is not a standardization or idealization of an atomic cell.
    reduced = niggli_reduce(lattice, eps=1e-10)
    if reduced is None:
        reduced = lattice
    change = np.linalg.solve(lattice.T, reduced.T).T
    integer_change = np.rint(change)
    if not np.allclose(change, integer_change, atol=1e-7, rtol=0) or abs(round(np.linalg.det(integer_change))) != 1:
        reduced = lattice
    else:
        reduced = integer_change @ lattice
    to_reduced = np.linalg.solve(reduced.T, lattice.T).T
    triangular = np.linalg.qr(reduced.T)[1]
    return lattice, reduced, to_reduced, triangular, float(singular[-1])


def _context(lattice):
    matrix = np.asarray(lattice, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Periodic distances require a 3x3 row-vector lattice.")
    return _distance_context(tuple(matrix.ravel()))


def periodic_cartesian_distance(left, right, lattice) -> float:
    """Return min_n ||(left-right-n) @ lattice||, for integer n in Z^3."""
    original, reduced, to_reduced, triangular, minimum_scale = _context(lattice)
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("Periodic positions must be finite fractional three-vectors.")
    wrapped = delta - np.rint(delta)
    distance = float(np.linalg.norm(wrapped @ original))
    # Every nonzero lattice vector has length >= sigma_min(lattice).
    if distance <= minimum_scale / 2:
        return distance
    coordinates = (np.mod(delta, 1.0) @ to_reduced) % 1.0
    initial = coordinates - np.rint(coordinates)
    best = float(np.dot(initial @ reduced, initial @ reduced))
    if best == 0.0:
        return 0.0
    chosen = np.zeros(3, dtype=int)
    visits = 0

    def search(axis, used):
        nonlocal best, visits
        if axis < 0:
            best = min(best, used)
            return
        remaining = max(best - used, 0.0)
        center = coordinates[axis] + float(
            triangular[axis, axis + 1:] @ (coordinates[axis + 1:] - chosen[axis + 1:])
        ) / triangular[axis, axis]
        radius = math.sqrt(remaining) / abs(triangular[axis, axis])
        slack = 16 * np.finfo(float).eps * max(1.0, abs(center), radius)
        lower = math.ceil(center - radius - slack)
        upper = math.floor(center + radius + slack)
        nearest = int(round(center))
        extent = max(abs(nearest - lower), abs(upper - nearest))
        candidates = itertools.chain(
            [nearest],
            itertools.chain.from_iterable((nearest - step, nearest + step) for step in range(1, extent + 1)),
        )
        for integer in candidates:
            if integer < lower or integer > upper:
                continue
            visits += 1
            if visits > 100000:
                raise ValueError("Periodic distance search exceeded its bound; inspect lattice conditioning.")
            chosen[axis] = integer
            residual = triangular[axis, axis] * (center - integer)
            cost = used + residual * residual
            if cost <= best + 16 * np.finfo(float).eps * max(best, 1e-300):
                search(axis - 1, cost)

    search(2, 0.0)
    return math.sqrt(max(best, 0.0))


def positions_within_cartesian_tolerance(left, right, lattice, tolerance) -> bool:
    """Test periodic site proximity with a tolerance in the lattice's length unit."""
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Position tolerance must be finite and nonnegative.")
    matrix, _, _, _, minimum_scale = _context(lattice)
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("Periodic positions must be finite fractional three-vectors.")
    wrapped = delta - np.rint(delta)
    distance = float(np.linalg.norm(wrapped @ matrix))
    slack = 64 * np.finfo(float).eps * max(1.0, distance, tolerance)
    if distance <= tolerance + slack:
        return True
    if distance <= minimum_scale / 2:
        return False
    return periodic_cartesian_distance(left, right, matrix) <= tolerance + slack


def fractional_search_radius(lattice, tolerance) -> float:
    """Conservative fractional infinity-norm radius for a physical tolerance."""
    return float(tolerance) / _context(lattice)[-1]
