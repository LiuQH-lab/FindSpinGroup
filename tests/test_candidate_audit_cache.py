import numpy as np

from findspingroup.core.identify_spin_space_group import (
    _candidate_audit_failure,
    _candidate_audit_failure_cached,
)
from findspingroup.core.tolerances import Tolerances
from findspingroup.structure import SpinSpaceGroupOperation


def _translation_group_ops():
    return [
        SpinSpaceGroupOperation(np.eye(3), np.eye(3), np.zeros(3)),
        SpinSpaceGroupOperation(np.eye(3), np.eye(3), np.array([0.5, 0.0, 0.0])),
    ]


def test_candidate_audit_cache_is_order_independent_and_tolerance_sensitive():
    _candidate_audit_failure_cached.cache_clear()
    ops = _translation_group_ops()
    default_tol = Tolerances()

    assert _candidate_audit_failure(ops, group_tol=default_tol) is None
    first = _candidate_audit_failure_cached.cache_info()
    assert first.misses == 1

    assert _candidate_audit_failure(reversed(ops), group_tol=default_tol) is None
    reordered = _candidate_audit_failure_cached.cache_info()
    assert reordered.hits == first.hits + 1
    assert reordered.misses == first.misses

    changed_tol = Tolerances(m_matrix_tol=default_tol.m_matrix_tol / 2.0)
    assert _candidate_audit_failure(ops, group_tol=changed_tol) is None
    tolerance_changed = _candidate_audit_failure_cached.cache_info()
    assert tolerance_changed.misses == reordered.misses + 1


def test_candidate_audit_cache_distinguishes_exact_operation_payloads():
    _candidate_audit_failure_cached.cache_clear()
    ops = _translation_group_ops()

    assert _candidate_audit_failure(ops) is None
    first = _candidate_audit_failure_cached.cache_info()

    changed_ops = _translation_group_ops()
    changed_ops[1].translation[0] = np.nextafter(0.5, 1.0)
    _candidate_audit_failure(changed_ops)
    changed = _candidate_audit_failure_cached.cache_info()

    assert changed.misses == first.misses + 1
