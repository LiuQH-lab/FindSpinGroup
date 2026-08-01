import itertools
import re

from findspingroup.data.POINT_GROUP_MATRIX import (
    operations,
    operations_hex,
    operations_mix,
)


def _matrix_key(matrix):
    return tuple(value for row in matrix for value in row)


def test_cubic_point_operation_table_contains_every_signed_permutation_once():
    expected = set()
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = [[0, 0, 0] for _ in range(3)]
            for row, column in enumerate(permutation):
                matrix[row][column] = signs[row]
            expected.add(_matrix_key(matrix))

    actual = {_matrix_key(matrix) for matrix, _, _ in operations}

    assert len(operations) == 48
    assert actual == expected


def test_point_operation_descriptions_and_tokens_use_the_same_hm_symbol():
    for table in (operations, operations_hex, operations_mix):
        for _, description, token in table:
            description_symbol = re.match(r"-?\d+|m", description).group()
            token_symbol = re.match(r"-?\d+|m", token).group()
            assert token_symbol == description_symbol
