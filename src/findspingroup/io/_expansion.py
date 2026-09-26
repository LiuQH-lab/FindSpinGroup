"""Physical error budgets for atoms reconstructed from file symmetry loops."""

from collections import defaultdict
from itertools import product
import math

import numpy as np

from ..utils.periodic import (
    fractional_search_radius,
    positions_within_cartesian_tolerance,
)


class ExpansionSites:
    """Keep one representative per typed periodic site, without idealizing it.

    Geometry uses the file lattice's length unit; moment residuals use physical
    Cartesian vectors. Unspecified moments are placeholders, not zero-valued
    observations. A supplied zero is an observation and is checked normally.
    """

    def __init__(self, lattice, moment_to_cartesian, *, position_atol, moment_atol,
                 occupancy_atol, format_name):
        self.lattice = np.asarray(lattice, dtype=float)
        self.moment_to_cartesian = np.asarray(moment_to_cartesian, dtype=float)
        self.position_atol = self._budget(position_atol, "position_atol")
        self.moment_atol = self._budget(moment_atol, "atol")
        self.occupancy_atol = self._budget(occupancy_atol, "occupancy_atol")
        self.format_name = format_name
        radius = fractional_search_radius(self.lattice, self.position_atol + 64*np.finfo(float).eps)
        self.bins = max(1, min(10**9, math.floor(1 / radius))) if radius else 10**9
        self.buckets = defaultdict(list)
        self.positions, self.elements, self.occupancies = [], [], []
        self.labels, self.moments, self.known = [], [], []

    @staticmethod
    def _budget(value, name):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        return value

    def _key(self, position):
        return tuple(np.floor(position * self.bins).astype(np.int64) % self.bins)

    def add(self, position, element, occupancy, label, moment, *, moment_known):
        position = np.asarray(position, dtype=float)
        moment = np.asarray(moment, dtype=float)
        if (position.shape != (3,) or moment.shape != (3,)
                or not np.all(np.isfinite(position)) or not np.all(np.isfinite(moment))
                or not math.isfinite(float(occupancy))):
            raise ValueError(f"{self.format_name} atom {label!r} has nonfinite or malformed data.")
        position = position % 1.0
        key = self._key(position)
        neighbor_keys = {tuple((key[i] + shift[i]) % self.bins for i in range(3))
                         for shift in product((-1, 0, 1), repeat=3)}
        matches = []
        for neighbor in neighbor_keys:
            for index in self.buckets.get((element, neighbor), ()):
                if (abs(occupancy - self.occupancies[index]) <= self.occupancy_atol
                        and positions_within_cartesian_tolerance(
                            position, self.positions[index], self.lattice, self.position_atol)):
                    matches.append(index)
        if len(matches) > 1:
            raise ValueError(
                f"{self.format_name} expansion has ambiguous site matches for {label!r} "
                f"within position_atol={self.position_atol:g} lattice length units."
            )
        if matches:
            index = matches[0]
            if moment_known and self.known[index]:
                residual = float(np.linalg.norm(
                    self.moment_to_cartesian @ (moment - self.moments[index])))
                slack = 64*np.finfo(float).eps*max(
                    1.0, np.linalg.norm(self.moment_to_cartesian @ moment),
                    np.linalg.norm(self.moment_to_cartesian @ self.moments[index]))
                if residual > self.moment_atol + slack:
                    raise ValueError(
                        f"{self.format_name} expansion produced inconsistent moments for "
                        f"the same atomic site ({self.labels[index]!r}, {label!r}): "
                        f"physical residual={residual:.12g}, atol={self.moment_atol:g}. "
                        "Check the declared spin frame, operations and moment precision. "
                        "An explicitly justified moment budget can be supplied with "
                        "find_spin_group(..., parser_atol=...) or parse_scif_file(..., atol=...) "
                        "/ parse_cif_file(..., atol=...); do not broaden it to hide a contradictory model."
                    )
            elif moment_known:
                self.moments[index] = moment.copy()
                self.known[index] = True
            return
        index = len(self.positions)
        self.buckets[(element, key)].append(index)
        self.positions.append(position)
        self.elements.append(element)
        self.occupancies.append(occupancy)
        self.labels.append(label)
        self.moments.append(moment.copy())
        self.known.append(bool(moment_known))

    def parsed(self, lattice_factors):
        return (lattice_factors, self.positions, self.elements, self.occupancies,
                self.labels, self.moments)
