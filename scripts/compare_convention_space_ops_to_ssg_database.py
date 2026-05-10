#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable


CONF_DIRS = {
    "Collinear": "Collinear",
    "Coplanar": "Coplanar",
    "Noncoplanar": "Non-coplanar",
    "Non-coplanar": "Non-coplanar",
}


def _fraction_from_float(value: float, *, max_denominator: int, tol: float) -> Fraction:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    frac = Fraction(float(value)).limit_denominator(max_denominator)
    if abs(float(frac) - float(value)) > tol:
        return Fraction(round(float(value) / tol), round(1.0 / tol))
    return frac


def _parse_fraction_token(token: object) -> Fraction:
    if isinstance(token, int):
        return Fraction(token, 1)
    if isinstance(token, float):
        return Fraction(token).limit_denominator(48)

    raw = str(token).strip()
    if not raw:
        raise ValueError("empty numeric token")
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return Fraction(int(numerator.strip()), int(denominator.strip()))
    if "." in raw:
        return Fraction(float(raw)).limit_denominator(48)
    return Fraction(int(raw), 1)


def _normalize_translation(value: Fraction) -> Fraction:
    denominator = value.denominator
    return Fraction(value.numerator % denominator, denominator)


def _op_key(
    rotation: Iterable[Iterable[float]],
    translation: Iterable[float],
    *,
    max_denominator: int,
    tol: float,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    rot = tuple(
        _fraction_from_float(float(item), max_denominator=max_denominator, tol=tol)
        for row in rotation
        for item in row
    )
    trans = tuple(
        _normalize_translation(
            _fraction_from_float(float(item), max_denominator=max_denominator, tol=tol)
        )
        for item in translation
    )
    return rot, trans


def _db_op_key(
    rotation_tokens: Iterable[object],
    translation_tokens: Iterable[object],
    *,
    normalize_translation: bool = True,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    rot = tuple(_parse_fraction_token(item) for item in rotation_tokens)
    parsed_trans = tuple(_parse_fraction_token(item) for item in translation_tokens)
    trans = (
        tuple(_normalize_translation(item) for item in parsed_trans)
        if normalize_translation
        else parsed_trans
    )
    return rot, trans


def _matrix_identity() -> list[list[Fraction]]:
    return [
        [Fraction(1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1)],
    ]


def _matrix_mul(
    left: list[list[Fraction]],
    right: list[list[Fraction]],
) -> list[list[Fraction]]:
    return [
        [
            sum(left[row][inner] * right[inner][col] for inner in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def _matrix_vec_mul(
    matrix: list[list[Fraction]],
    vector: list[Fraction],
) -> list[Fraction]:
    return [
        sum(matrix[row][col] * vector[col] for col in range(3))
        for row in range(3)
    ]


def _matrix_det(matrix: list[list[Fraction]]) -> Fraction:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _matrix_inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    det = _matrix_det(matrix)
    if det == 0:
        raise ValueError(f"Singular translational generator matrix: {matrix!r}")
    cofactors = [
        [
            matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
            -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]),
            matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
        ],
        [
            -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]),
            matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
            -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]),
        ],
        [
            matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
            -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]),
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        ],
    ]
    return [
        [cofactors[col][row] / det for col in range(3)]
        for row in range(3)
    ]


def _parse_fraction_vector(text: str) -> list[Fraction]:
    return [_parse_fraction_token(item.strip()) for item in text.split(",")]


def _read_translational_generator_matrix(file1_path: Path) -> list[list[Fraction]]:
    if not file1_path.exists():
        return _matrix_identity()
    pattern = re.compile(
        r"Translational group generators:\s*"
        r"a = \(([^)]*)\),\s*b = \(([^)]*)\),\s*c = \(([^)]*)\)"
    )
    with file1_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            match = pattern.search(raw_line)
            if not match:
                continue
            columns = [_parse_fraction_vector(match.group(i)) for i in range(1, 4)]
            if any(len(column) != 3 for column in columns):
                raise ValueError(f"Unexpected translational generators in {file1_path}")
            return [
                [columns[col][row] for col in range(3)]
                for row in range(3)
            ]
    return _matrix_identity()


def _read_file1_transformation_matrix(
    file1_path: Path,
) -> tuple[list[list[Fraction]], list[Fraction]]:
    if not file1_path.exists():
        return _matrix_identity(), [Fraction(0), Fraction(0), Fraction(0)]
    pattern = re.compile(r"transformation_matrix M:\s*(\[\[.*\]\])")
    with file1_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            match = pattern.search(raw_line)
            if not match:
                continue
            text = match.group(1).strip()
            if not (text.startswith("[[") and text.endswith("]]")):
                raise ValueError(f"Unexpected file1 transformation matrix in {file1_path}")
            inner = text[2:-2]
            parts = re.split(r"\]\s*,\s*\[", inner, maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Unexpected file1 transformation matrix in {file1_path}")
            values = [_parse_fraction_token(item.strip()) for item in parts[0].split(",")]
            shift = [_parse_fraction_token(item.strip()) for item in parts[1].split(",")]
            if len(values) != 9 or len(shift) != 3:
                raise ValueError(f"Unexpected file1 transformation matrix in {file1_path}")
            return (
                [values[0:3], values[3:6], values[6:9]],
                shift,
            )
    return _matrix_identity(), [Fraction(0), Fraction(0), Fraction(0)]


def _affine_inverse(
    transform: tuple[list[list[Fraction]], list[Fraction]],
) -> tuple[list[list[Fraction]], list[Fraction]]:
    matrix, shift = transform
    inverse = _matrix_inverse(matrix)
    return inverse, [-item for item in _matrix_vec_mul(inverse, shift)]


def _affine_compose(
    first: tuple[list[list[Fraction]], list[Fraction]],
    second: tuple[list[list[Fraction]], list[Fraction]],
) -> tuple[list[list[Fraction]], list[Fraction]]:
    first_matrix, first_shift = first
    second_matrix, second_shift = second
    return (
        _matrix_mul(second_matrix, first_matrix),
        [
            item + second_shift[index]
            for index, item in enumerate(_matrix_vec_mul(second_matrix, first_shift))
        ],
    )


def _transform_space_key(
    key: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    transform: tuple[list[list[Fraction]], list[Fraction]],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    rotation_flat, translation = key
    rotation = [
        [rotation_flat[3 * row + col] for col in range(3)]
        for row in range(3)
    ]
    matrix, shift = transform
    inverse = _matrix_inverse(matrix)
    transformed_rotation = _matrix_mul(_matrix_mul(matrix, rotation), inverse)
    transformed_translation_linear = _matrix_vec_mul(matrix, list(translation))
    transformed_shift = _matrix_vec_mul(transformed_rotation, shift)
    transformed_translation = [
        transformed_translation_linear[index] + shift[index] - transformed_shift[index]
        for index in range(3)
    ]
    return (
        tuple(item for row in transformed_rotation for item in row),
        tuple(_normalize_translation(item) for item in transformed_translation),
    )


def _axis_collapse_output_candidates(
    output_ops: list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
) -> list[tuple[str, list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]]]]:
    """Try quotienting a redundant half-translation coset in the output basis.

    This is intentionally a comparison-side normalization.  It does not claim
    that the full spin-space operation list is smaller; it only tests whether
    the real-space operation set is equivalent after shrinking one conventional
    axis by a factor of two.
    """
    zero = [Fraction(0), Fraction(0), Fraction(0)]
    axis_transforms = [
        (
            "output-axis-collapse-x2",
            [
                [Fraction(2), Fraction(0), Fraction(0)],
                [Fraction(0), Fraction(1), Fraction(0)],
                [Fraction(0), Fraction(0), Fraction(1)],
            ],
        ),
        (
            "output-axis-collapse-y2",
            [
                [Fraction(1), Fraction(0), Fraction(0)],
                [Fraction(0), Fraction(2), Fraction(0)],
                [Fraction(0), Fraction(0), Fraction(1)],
            ],
        ),
        (
            "output-axis-collapse-z2",
            [
                [Fraction(1), Fraction(0), Fraction(0)],
                [Fraction(0), Fraction(1), Fraction(0)],
                [Fraction(0), Fraction(0), Fraction(2)],
            ],
        ),
    ]
    candidates = []
    original_unique_count = len(set(output_ops))
    for name, matrix in axis_transforms:
        transformed_ops = [
            _transform_space_key(key, (matrix, zero))
            for key in output_ops
        ]
        # Keep the candidate focused on the observed failure mode: one redundant
        # index-2 real-space coset.  Broader reductions should be reviewed before
        # becoming comparison defaults.
        if len(set(transformed_ops)) * 2 == original_unique_count:
            candidates.append((name, transformed_ops))
    return candidates


def _record_identify_transformation_matrix(
    result: dict,
    *,
    max_denominator: int,
    tol: float,
) -> tuple[list[list[Fraction]], list[Fraction]]:
    details = result.get("identify_index_details") or {}
    raw = details.get("transformation_matrix")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not isinstance(raw[0], list)
        or len(raw[0]) != 3
        or len(raw[1]) != 3
    ):
        raise ValueError("record has no identify_index_details.transformation_matrix")
    matrix = [
        [
            _fraction_from_float(
                float(raw[0][row][col]),
                max_denominator=max_denominator,
                tol=tol,
            )
            for col in range(3)
        ]
        for row in range(3)
    ]
    shift = [
        _fraction_from_float(float(item), max_denominator=max_denominator, tol=tol)
        for item in raw[1]
    ]
    return matrix, shift


def _record_identify_space_group_transformation(
    result: dict,
    *,
    max_denominator: int,
    tol: float,
) -> tuple[list[list[Fraction]], list[Fraction]] | None:
    details = result.get("identify_index_details") or {}
    raw = details.get("space_group_transformation")
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not isinstance(raw[0], list)
        or len(raw[0]) != 3
        or len(raw[1]) != 3
    ):
        raise ValueError("unexpected identify_index_details.space_group_transformation")
    matrix = [
        [
            _fraction_from_float(
                float(raw[0][row][col]),
                max_denominator=max_denominator,
                tol=tol,
            )
            for col in range(3)
        ]
        for row in range(3)
    ]
    shift = [
        _fraction_from_float(float(item), max_denominator=max_denominator, tol=tol)
        for item in raw[1]
    ]
    return matrix, shift


def _transform_db_key(
    key: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    generator_matrix: list[list[Fraction]],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    if generator_matrix == _matrix_identity():
        return key
    rotation_flat, translation = key
    rotation = [
        [rotation_flat[3 * row + col] for col in range(3)]
        for row in range(3)
    ]
    inverse = _matrix_inverse(generator_matrix)
    transformed_rotation = _matrix_mul(_matrix_mul(inverse, rotation), generator_matrix)
    transformed_translation = _matrix_vec_mul(inverse, list(translation))
    return (
        tuple(item for row in transformed_rotation for item in row),
        tuple(_normalize_translation(item) for item in transformed_translation),
    )


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_key(key: tuple[tuple[Fraction, ...], tuple[Fraction, ...]]) -> dict:
    rotation, translation = key
    return {
        "rotation": [
            [_format_fraction(rotation[3 * row + col]) for col in range(3)]
            for row in range(3)
        ],
        "translation": [_format_fraction(item) for item in translation],
    }


def _parse_db_space(space_text: str) -> tuple[tuple[object, ...], tuple[object, ...]]:
    text = space_text.strip()
    if not (text.startswith("[[") and text.endswith("]]")):
        raise ValueError(f"Unexpected database space string: {space_text!r}")
    inner = text[2:-2]
    matrix_text, translation_text = inner.split("], [", 1)
    matrix = tuple(item.strip() for item in matrix_text.split(","))
    translation = tuple(item.strip() for item in translation_text.split(","))
    if len(matrix) != 9 or len(translation) != 3:
        raise ValueError(f"Unexpected database space dimensions: {space_text!r}")
    return matrix, translation


def _read_db_ops(
    file3_path: Path,
    *,
    coordinate_mode: str,
) -> list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]]:
    ops: list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]] = []
    in_main_section = False
    generator_matrix = (
        _read_translational_generator_matrix(file3_path.with_name("file1.txt"))
        if coordinate_mode == "translational-generators"
        else _matrix_identity()
    )

    with file3_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[No.,"):
                in_main_section = True
                continue
            if line.startswith("[extra"):
                in_main_section = False
                continue
            if not in_main_section or not line.startswith("["):
                continue

            try:
                row = ast.literal_eval(line)
            except Exception as exc:
                raise ValueError(f"Could not parse {file3_path}:{line_number}: {line}") from exc

            if not isinstance(row, list) or len(row) < 4 or not isinstance(row[0], int):
                continue
            matrix_tokens, translation_tokens = _parse_db_space(str(row[2]))
            key = _db_op_key(
                matrix_tokens,
                translation_tokens,
                normalize_translation=(coordinate_mode == "raw"),
            )
            ops.append(_transform_db_key(key, generator_matrix))

    if not ops:
        raise ValueError(f"No main file3 operations parsed from {file3_path}")
    return ops


def _candidate_db_paths(ssg_database_root: Path, conf: str, index: str) -> list[Path]:
    conf_dir = CONF_DIRS.get(conf, conf)
    parts = index.split(".")
    if len(parts) < 2:
        return []
    g0, l0 = parts[0], parts[1]
    base = ssg_database_root / "OUTPUT" / conf_dir
    return [
        base / type_dir / g0 / l0 / index / "file3.txt"
        for type_dir in ("typeI", "typeII", "typeIII")
    ]


def _find_db_path(ssg_database_root: Path, conf: str, index: str) -> tuple[Path | None, list[Path]]:
    candidates = _candidate_db_paths(ssg_database_root, conf, index)
    matches = [path for path in candidates if path.exists()]
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def _counter_delta(left: Counter, right: Counter) -> tuple[list, list]:
    missing = list((right - left).elements())
    extra = list((left - right).elements())
    return missing, extra


def _comparison_status(
    output_ops: list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
    db_ops: list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
) -> str:
    if output_ops == db_ops:
        return "same_sequence"
    if Counter(output_ops) == Counter(db_ops):
        return "same_multiset_order_diff"
    if set(output_ops) == set(db_ops):
        return "same_unique_set_multiplicity_diff"
    return "different_set"


def _load_record_ops(
    payload: dict,
    *,
    max_denominator: int,
    tol: float,
) -> list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]]:
    result = payload.get("result") or {}
    ops = result.get("convention_ssg_ops")
    if not isinstance(ops, list):
        raise ValueError("record has no convention_ssg_ops list")
    parsed = []
    for op in ops:
        if not isinstance(op, list) or len(op) != 3:
            raise ValueError(f"Unexpected convention operation shape: {op!r}")
        _spin_rotation, space_rotation, translation = op
        parsed.append(
            _op_key(
                space_rotation,
                translation,
                max_denominator=max_denominator,
                tol=tol,
            )
        )
    return parsed


def _sample(items: list, limit: int) -> list:
    return items[: max(0, limit)]


def compare(
    *,
    full_results_jsonl: Path,
    ssg_database_root: Path,
    output_dir: Path,
    coordinate_mode: str,
    max_denominator: int,
    tol: float,
    sample_limit: int,
    limit: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mismatches_path = output_dir / "space_op_mismatches.jsonl"
    missing_path = output_dir / "missing_database_paths.jsonl"
    errors_path = output_dir / "comparison_errors.jsonl"

    summary_counter: Counter[str] = Counter()
    by_conf: dict[str, Counter] = defaultdict(Counter)
    by_setting: dict[str, Counter] = defaultdict(Counter)
    by_db_type: dict[str, Counter] = defaultdict(Counter)
    by_candidate: Counter[str] = Counter()
    op_count_pairs: Counter[str] = Counter()
    unique_indices: set[str] = set()
    mismatch_indices: set[str] = set()
    started = time.time()

    db_ops_cache: dict[
        tuple[Path, str],
        list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
    ] = {}
    file1_transform_cache: dict[Path, tuple[list[list[Fraction]], list[Fraction]]] = {}

    with (
        full_results_jsonl.open("r", encoding="utf-8") as records,
        mismatches_path.open("w", encoding="utf-8") as mismatches_file,
        missing_path.open("w", encoding="utf-8") as missing_file,
        errors_path.open("w", encoding="utf-8") as errors_file,
    ):
        for zero_index, line in enumerate(records):
            if limit is not None and zero_index >= limit:
                break
            if not line.strip():
                continue

            summary_counter["records_seen"] += 1
            try:
                record = json.loads(line)
                result = record.get("result") or {}
                if record.get("status") != "ok":
                    summary_counter["skipped_non_ok_record"] += 1
                    continue

                index = result.get("index")
                conf = result.get("conf")
                setting = result.get("convention_ssg_setting")
                file_name = record.get("file_name")
                case_id = record.get("case_id")
                if not index or not conf:
                    raise ValueError("record is missing result.index or result.conf")

                unique_indices.add(str(index))
                output_ops = _load_record_ops(
                    record,
                    max_denominator=max_denominator,
                    tol=tol,
                )
                db_path, db_matches = _find_db_path(ssg_database_root, str(conf), str(index))
                if db_path is None:
                    status = "missing_database_path" if not db_matches else "ambiguous_database_path"
                    summary_counter[status] += 1
                    by_conf[str(conf)][status] += 1
                    by_setting[str(setting)][status] += 1
                    missing_file.write(
                        json.dumps(
                            {
                                "status": status,
                                "case_id": case_id,
                                "file_name": file_name,
                                "index": index,
                                "conf": conf,
                                "setting": setting,
                                "candidate_count": len(db_matches),
                                "candidates": [path.as_posix() for path in db_matches],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue

                candidate_name = coordinate_mode
                if coordinate_mode == "candidate-basis":
                    effective_output_ops = output_ops
                    raw_cache_key = (db_path, "raw")
                    tgen_cache_key = (db_path, "translational-generators")
                    if raw_cache_key not in db_ops_cache:
                        db_ops_cache[raw_cache_key] = _read_db_ops(
                            db_path,
                            coordinate_mode="raw",
                        )
                    if tgen_cache_key not in db_ops_cache:
                        db_ops_cache[tgen_cache_key] = _read_db_ops(
                            db_path,
                            coordinate_mode="translational-generators",
                        )
                    candidates: list[
                        tuple[str, list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]]]
                    ] = [
                        ("raw", db_ops_cache[raw_cache_key]),
                        ("translational-generators", db_ops_cache[tgen_cache_key]),
                    ]
                    file1_path = db_path.with_name("file1.txt")
                    if file1_path not in file1_transform_cache:
                        file1_transform_cache[file1_path] = _read_file1_transformation_matrix(
                            file1_path
                        )
                    identify_transform = _record_identify_transformation_matrix(
                        result,
                        max_denominator=max_denominator,
                        tol=tol,
                    )
                    database_to_convention = _affine_compose(
                        file1_transform_cache[file1_path],
                        _affine_inverse(identify_transform),
                    )
                    candidates.append(
                        (
                            "file1-m-identify-inverse",
                            [
                                _transform_space_key(key, database_to_convention)
                                for key in db_ops_cache[raw_cache_key]
                            ],
                        )
                    )
                    space_group_transform = _record_identify_space_group_transformation(
                        result,
                        max_denominator=max_denominator,
                        tol=tol,
                    )
                    if space_group_transform is not None:
                        inverse_space_group_transform = _affine_inverse(space_group_transform)
                        candidates.extend(
                            [
                                (
                                    "space-group-transformation(raw)",
                                    [
                                        _transform_space_key(key, space_group_transform)
                                        for key in db_ops_cache[raw_cache_key]
                                    ],
                                ),
                                (
                                    "space-group-transformation(tgen)",
                                    [
                                        _transform_space_key(key, space_group_transform)
                                        for key in db_ops_cache[tgen_cache_key]
                                    ],
                                ),
                                (
                                    "space-group-transformation-inverse(raw)",
                                    [
                                        _transform_space_key(key, inverse_space_group_transform)
                                        for key in db_ops_cache[raw_cache_key]
                                    ],
                                ),
                                (
                                    "space-group-transformation-inverse(tgen)",
                                    [
                                        _transform_space_key(key, inverse_space_group_transform)
                                        for key in db_ops_cache[tgen_cache_key]
                                    ],
                                ),
                            ]
                        )

                    winner: tuple[
                        str,
                        list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
                        list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
                    ] | None = None
                    for preferred_status in (
                        "same_sequence",
                        "same_multiset_order_diff",
                        "same_unique_set_multiplicity_diff",
                    ):
                        for name, candidate_ops in candidates:
                            if _comparison_status(output_ops, candidate_ops) == preferred_status:
                                winner = name, output_ops, candidate_ops
                                break
                        if winner is not None:
                            break
                    if winner is None:
                        output_axis_candidates = _axis_collapse_output_candidates(output_ops)
                        for preferred_status in (
                            "same_sequence",
                            "same_multiset_order_diff",
                            "same_unique_set_multiplicity_diff",
                        ):
                            for output_name, candidate_output_ops in output_axis_candidates:
                                for db_name, candidate_db_ops in candidates:
                                    if (
                                        _comparison_status(candidate_output_ops, candidate_db_ops)
                                        == preferred_status
                                    ):
                                        winner = (
                                            f"{output_name}+{db_name}",
                                            candidate_output_ops,
                                            candidate_db_ops,
                                        )
                                        break
                                if winner is not None:
                                    break
                            if winner is not None:
                                break
                    if winner is None:
                        scored_candidates: list[
                            tuple[
                                str,
                                list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
                                list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
                            ]
                        ] = [
                            (name, output_ops, candidate_ops)
                            for name, candidate_ops in candidates
                        ]
                        for output_name, candidate_output_ops in _axis_collapse_output_candidates(output_ops):
                            scored_candidates.extend(
                                (
                                    f"{output_name}+{db_name}",
                                    candidate_output_ops,
                                    candidate_db_ops,
                                )
                                for db_name, candidate_db_ops in candidates
                            )
                        winner = min(
                            scored_candidates,
                            key=lambda item: len(set(item[2]) - set(item[1]))
                            + len(set(item[1]) - set(item[2])),
                        )
                    candidate_name, effective_output_ops, db_ops = winner
                else:
                    effective_output_ops = output_ops
                    use_file1_identify = coordinate_mode == "file1-m-identify-inverse" or (
                        coordinate_mode == "file1-m-identify-inverse-l0std"
                        and str(setting) == "L0std"
                    )
                    if use_file1_identify:
                        raw_cache_key = (db_path, "raw")
                        if raw_cache_key not in db_ops_cache:
                            db_ops_cache[raw_cache_key] = _read_db_ops(
                                db_path,
                                coordinate_mode="raw",
                            )
                        file1_path = db_path.with_name("file1.txt")
                        if file1_path not in file1_transform_cache:
                            file1_transform_cache[file1_path] = _read_file1_transformation_matrix(
                                file1_path
                            )
                        identify_transform = _record_identify_transformation_matrix(
                            result,
                            max_denominator=max_denominator,
                            tol=tol,
                        )
                        database_to_convention = _affine_compose(
                            file1_transform_cache[file1_path],
                            _affine_inverse(identify_transform),
                        )
                        db_ops = [
                            _transform_space_key(key, database_to_convention)
                            for key in db_ops_cache[raw_cache_key]
                        ]
                    else:
                        effective_coordinate_mode = (
                            "translational-generators"
                            if coordinate_mode == "file1-m-identify-inverse-l0std"
                            else coordinate_mode
                        )
                        cache_key = (db_path, effective_coordinate_mode)
                        if cache_key not in db_ops_cache:
                            db_ops_cache[cache_key] = _read_db_ops(
                                db_path,
                                coordinate_mode=effective_coordinate_mode,
                            )
                        db_ops = db_ops_cache[cache_key]

                output_counter = Counter(effective_output_ops)
                db_counter = Counter(db_ops)
                output_set = set(effective_output_ops)
                db_set = set(db_ops)
                same_sequence = effective_output_ops == db_ops
                same_multiset = output_counter == db_counter
                same_set = output_set == db_set
                db_type = db_path.parts[-5]
                by_candidate[candidate_name] += 1
                by_db_type[db_type]["records_seen"] += 1
                op_count_pairs[f"{len(output_ops)}:{len(db_ops)}"] += 1

                if same_sequence:
                    status = "same_sequence"
                elif same_multiset:
                    status = "same_multiset_order_diff"
                elif same_set:
                    status = "same_unique_set_multiplicity_diff"
                else:
                    status = "different_set"
                    mismatch_indices.add(str(index))
                    missing_from_output = sorted(db_set - output_set)
                    extra_in_output = sorted(output_set - db_set)
                    row = {
                        "case_id": case_id,
                        "file_name": file_name,
                        "index": index,
                        "conf": conf,
                        "setting": setting,
                        "db_path": db_path.as_posix(),
                        "candidate": candidate_name,
                        "output_op_count": len(output_ops),
                        "db_op_count": len(db_ops),
                        "output_unique_op_count": len(output_counter),
                        "db_unique_op_count": len(db_counter),
                        "missing_from_output_count": len(missing_from_output),
                        "extra_in_output_count": len(extra_in_output),
                        "missing_from_output_sample": [
                            _format_key(item) for item in _sample(missing_from_output, sample_limit)
                        ],
                        "extra_in_output_sample": [
                            _format_key(item) for item in _sample(extra_in_output, sample_limit)
                        ],
                    }
                    mismatches_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

                summary_counter[status] += 1
                by_conf[str(conf)][status] += 1
                by_setting[str(setting)][status] += 1
                by_db_type[db_type][status] += 1

            except Exception as exc:
                summary_counter["comparison_error"] += 1
                errors_file.write(
                    json.dumps(
                        {
                            "record_number": zero_index + 1,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    summary = {
        "full_results_jsonl": full_results_jsonl.as_posix(),
        "ssg_database_root": ssg_database_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "database_coordinate_mode": coordinate_mode,
        "elapsed_seconds": round(time.time() - started, 3),
        "max_denominator": max_denominator,
        "tolerance": tol,
        "counts": dict(summary_counter),
        "unique_index_count": len(unique_indices),
        "mismatch_unique_index_count": len(mismatch_indices),
        "by_conf": {key: dict(counter) for key, counter in sorted(by_conf.items())},
        "by_setting": {key: dict(counter) for key, counter in sorted(by_setting.items())},
        "by_database_type": {key: dict(counter) for key, counter in sorted(by_db_type.items())},
        "by_candidate": dict(sorted(by_candidate.items())),
        "operation_count_pairs": dict(sorted(op_count_pairs.items())),
        "artifacts": {
            "mismatches": mismatches_path.as_posix(),
            "missing_database_paths": missing_path.as_posix(),
            "comparison_errors": errors_path.as_posix(),
        },
    }
    (output_dir / "space_op_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FindSpinGroup convention real-space operations against "
            "file3.txt operations in an SSG database checkout."
        )
    )
    parser.add_argument("--full-results-jsonl", required=True, type=Path)
    parser.add_argument(
        "--ssg-database-root",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--database-coordinate-mode",
        choices=[
            "raw",
            "translational-generators",
            "file1-m-identify-inverse",
            "file1-m-identify-inverse-l0std",
            "candidate-basis",
        ],
        default="raw",
        help=(
            "Use raw file3 coordinates, or transform file3 operations into the "
            "basis defined by file1 translational group generators. "
            "file1-m-identify-inverse composes file1 transformation_matrix M "
            "with identify_index_details.transformation_matrix^-1. "
            "file1-m-identify-inverse-l0std applies that composition only to "
            "L0std records and uses translational-generators for the others. "
            "candidate-basis accepts the best exact set match among raw, "
            "translational-generators, file1-m-identify-inverse, and identify "
            "space_group_transformation variants."
        ),
    )
    parser.add_argument("--max-denominator", type=int, default=48)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    summary = compare(
        full_results_jsonl=args.full_results_jsonl,
        ssg_database_root=args.ssg_database_root,
        output_dir=args.output_dir,
        coordinate_mode=args.database_coordinate_mode,
        max_denominator=args.max_denominator,
        tol=args.tol,
        sample_limit=args.sample_limit,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
