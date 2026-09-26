import json

import numpy as np

from findspingroup.batch_mcif import _normalize_jsonable
from findspingroup.find_spin_group import NumpyEncoder
from findspingroup.structure import SpinSpaceGroupOperation


def test_operation_json_retains_resolved_matrix_terms_and_lifted_translations():
    angle = 3.24557960535796e-7
    c, s = np.cos(angle), np.sin(angle)
    spin = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    rotation = np.array([[1., 3.1e-8, 0.], [0., -1., 0.], [0., 0., -1.]])
    translation = np.array([1.000000123456789, -2., .499999876543211])
    op = SpinSpaceGroupOperation(spin, rotation, translation)
    original = [op.spin_rotation.copy(), op.rotation.copy(), op.translation.copy()]
    for payload in (op.tolist(), _normalize_jsonable(op)):
        decoded = json.loads(json.dumps(payload))
        for actual, expected in zip(decoded, original):
            np.testing.assert_array_equal(actual, expected)
    structured = json.loads(json.dumps(op, cls=NumpyEncoder))
    for key, expected in zip(('spin_rotation', 'real_rotation', 'translation'), original):
        np.testing.assert_array_equal(structured[key], expected)
    for actual, expected in zip((op.spin_rotation, op.rotation, op.translation), original):
        np.testing.assert_array_equal(actual, expected)
