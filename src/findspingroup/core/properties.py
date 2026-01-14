from __future__ import annotations

import numpy as np

from findspingroup.core.identify_symmetry_from_ops import deduplicate_matrix_pairs
from findspingroup.structure import *
from findspingroup.utils.matrix_utils import rref_with_tolerance



def combine_parametric_solutions(rref_matrix, tol=1e-3):
    import numpy as np

    A = np.array(rref_matrix, dtype=float)
    rows, cols = A.shape
    pivot_cols = []
    free_vars = []

    # 找出主元列
    for i in range(rows):
        for j in range(cols):
            if abs(A[i, j]) > tol:
                pivot_cols.append(j)
                break

    pivot_cols = set(pivot_cols)
    free_vars = [j for j in range(cols) if j not in pivot_cols]

    # 构造每个自由变量对应的解向量
    symbols = ['Sx', 'Sy', 'Sz']
    vector_expr = ['0'] * cols

    for free_idx, var_col in enumerate(free_vars):
        var_name = symbols[free_idx]
        coeffs = [0] * cols
        coeffs[var_col] = 1
        for row_idx in range(rows):
            row = A[row_idx]
            pivot_col = next((j for j in range(cols) if abs(row[j]) > tol), None)
            if pivot_col is not None and abs(row[var_col]) > tol:
                coeffs[pivot_col] = -row[var_col]

        # 累加表达式向量
        for i in range(cols):
            c = coeffs[i]
            if abs(c) < tol:
                continue
            if vector_expr[i] == '0':
                if abs(c - 1) < tol:
                    vector_expr[i] = var_name
                elif abs(c + 1) < tol:
                    vector_expr[i] = f"-{var_name}"
                else:
                    vector_expr[i] = f"{round(c, 3)}*{var_name}"
            else:
                if abs(c - 1) < tol:
                    vector_expr[i] += f" + {var_name}"
                elif abs(c + 1) < tol:
                    vector_expr[i] += f" - {var_name}"
                elif c > 0:
                    vector_expr[i] += f" + {round(c, 3)}*{var_name}"
                else:
                    vector_expr[i] += f" - {abs(round(c, 3))}*{var_name}"

    return vector_expr

def tensor_constraint(symmetry_ops, spin_rank, spatial_rank, T_odd=False, extra_constraint=None):
    # Placeholder for tensor constraint logic
    return {"tensor_constraints": (symmetry_ops, spin_rank, spatial_rank, T_odd, extra_constraint)}


class TensorConstraint:
    """
    A class to represent tensor constraints for a specific physical tensor by some symmetry.

    is returned by TensorConstraintsAnalyzer.analyze()
    """
    def __init__(self, symmetry_ops,eqs,symbols,extra_constraint=None):
        self.tensor_name = ''
        self.symmetry_ops = symmetry_ops
        self.extra_constraint = extra_constraint

        self.eqs = eqs
        self.symbols = symbols


    def __repr__(self):
        return ''




class TensorConstraintsAnalyzer:
    """
    A class to analyze tensor constraints for various physical tensors.
    """

    def __init__(self, symmetries):
        self.symmetries = symmetries

        self._conductivity = None
        self._AHE = None
        self._SHE = None
        self._spin_polarization = None
        self._piezoelectric = None
        self._piezomagnetic = None
        self._magnetoelectric = None
        self._elasticity = None
        self._stiffness = None
        self._compliance = None
        self._permittivity = None
        self._susceptibility = None

    @property
    def conductivity(self):
        if self._conductivity is None:
            self._conductivity = TensorConstraint(self.symmetries.ssg.operations, spin_rank=0, spatial_rank=2, T_odd=True, extra_constraint=self.extra_constraint)
        return self._conductivity

    @property
    def AHE(self):
        if self._AHE is None:
            self._AHE = TensorConstraint(self.symmetries.ssg.operations, spin_rank=0, spatial_rank=2, T_odd=True, extra_constraint=self.extra_constraint)
        return self._AHE

    @property
    def SHE(self):
        if self._SHE is None:
            self._SHE = TensorConstraint(self.symmetries.ssg.operations, spin_rank=1, spatial_rank=2, T_odd=False, extra_constraint=self.extra_constraint)
        return self._SHE

    @property
    def spin_polarization(self):
        if self._spin_polarization is None:
            self._spin_polarization = TensorConstraint(self.symmetries.ssg.operations, spin_rank=1, spatial_rank=1, T_odd=False, extra_constraint=self.extra_constraint)
        return self._spin_polarization

    @property
    def piezoelectric(self):
        if self._piezoelectric is None:
            self._piezoelectric = TensorConstraint(self.symmetries.ssg.operations, spin_rank=0, spatial_rank=3, T_odd=False, extra_constraint=self.extra_constraint)
        return self._piezoelectric

    @property
    def piezomagnetic(self):
        if self._piezomagnetic is None:
            self._piezomagnetic = TensorConstraint(self.symmetries.ssg.operations, spin_rank=1, spatial_rank=3, T_odd=True, extra_constraint=self.extra_constraint)



    def analyze(self,extra_constraint=None):
        eqs = []
        symbols = []

        return TensorConstraint(self.symmetries, eqs, symbols, extra_constraint=extra_constraint)



class SymmetryProperties:
    """
    A class to analyze and store symmetry-related properties of a crystal structure.

    """
    def __init__(self, structure:CrystalCell, ssg :SpinSpaceGroup = None,sg = None, msg = None):
        self.tol = 1e-2
        self.structure = structure
        self.ssg = ssg
        self.sg = sg
        self.msg = msg
        self.spin_constraint = self.get_spin_constraint()
        self._magnetic_phase = None
        self._spin_polarization_wSOC = None
        self._spin_polarization_woSOC = None
        self._AHE_wSOC = None
        self._AHE_woSOC = None

        self._tensor_constraints = None

        # remain to be extended



    @property
    def magnetic_phase(self):
        if self._magnetic_phase is None:
            self._magnetic_phase = self._determine_magnetic_phase()
        return self._magnetic_phase

    @property
    def spin_polarization_wSOC(self):
        if self._spin_polarization_wSOC is None:
            self._spin_polarization_wSOC = self._determine_spin_polarization(wSOC=True)
        return self._spin_polarization_wSOC

    @property
    def spin_polarization_woSOC(self):
        if self._spin_polarization_woSOC is None:
            self._spin_polarization_woSOC = self._determine_spin_polarization(wSOC=False)
        return self._spin_polarization_woSOC

    @property
    def AHE_wSOC(self):
        if self._AHE_wSOC is None:
            self._AHE_wSOC = self._determine_AHE(wSOC=True)
        return self._AHE_wSOC

    @property
    def AHE_woSOC(self):
        if self._AHE_woSOC is None:
            self._AHE_woSOC = self._determine_AHE(wSOC=False)
        return self._AHE_woSOC

    @property
    def tensor_constraints(self):
        if self._tensor_constraints is None:
            self._tensor_constraints = TensorConstraintsAnalyzer(self.ssg)
        return self._tensor_constraints

    def _determine_magnetic_phase(self):
        # Placeholder for magnetic phase determination logic
        return "MagneticPhase"

    def _determine_spin_polarization(self, wSOC=True):
        # Placeholder for spin polarization determination logic
        return f"SpinPolarization_wSOC_{wSOC}"

    def _determine_AHE(self, wSOC=True):
        # Placeholder for AHE determination logic
        return f"AHE_wSOC_{wSOC}"

    def is_spin_splitting(self):
        spinsplitting = []
        for little_group in self.ssg.little_groups:
            spinmatrices = np.vstack(deduplicate_matrix_pairs([op[0]-np.eye(3) for op in little_group],tol=self.tol))

            if all(abs(x) > 1e-3 for x in np.linalg.svd(spinmatrices.astype(np.float32))[1]):
                # 无自旋极化，故无ss
                spinsplitting.append('no spin splitting')
            else:
                spinsplitting.append('spin splitting')
        return spinsplitting

    def get_spin_constraint(self):
        constraints = []
        for little_group in self.ssg.little_groups:
            spinmatrices = np.vstack(deduplicate_matrix_pairs([op[0]-np.eye(3) for op in little_group],tol=self.tol))
            constraints.append(combine_parametric_solutions(rref_with_tolerance(spinmatrices)))

        return constraints