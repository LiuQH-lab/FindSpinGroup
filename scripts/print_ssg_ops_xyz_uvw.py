#!/usr/bin/env python3

from __future__ import annotations

import argparse
from fractions import Fraction
from typing import Iterable

import numpy as np

from findspingroup import find_spin_group
from findspingroup.structure.group import SpinSpaceGroup, SpinSpaceGroupOperation
from findspingroup.utils.international_symbol import (
    _canonical_spin_symbol_map,
    _compose_setting_transform,
    _find_real_operation,
    _parse_sg_generator_ops,
    _transport_standard_generators_to_current_basis,
)


XYZ_VARS = ("x", "y", "z")
UVW_VARS = ("u", "v", "w")


def _format_number(value: float, tol: float = 1e-8) -> str | None:
    if abs(value) < tol:
        return None
    if abs(value - round(value)) < tol:
        return str(int(round(value)))
    frac = Fraction(float(value)).limit_denominator(12)
    if abs(float(frac) - float(value)) < 1e-6:
        return f"{frac.numerator}/{frac.denominator}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_affine(matrix: np.ndarray, translation: np.ndarray, symbols: Iterable[str]) -> str:
    labels = tuple(symbols)
    parts: list[str] = []
    for i in range(3):
        expr: list[str] = []
        for j, label in enumerate(labels):
            coeff = float(matrix[i, j])
            if abs(coeff) < 1e-8:
                continue
            if abs(coeff - 1.0) < 1e-8:
                expr.append(f"+{label}")
            elif abs(coeff + 1.0) < 1e-8:
                expr.append(f"-{label}")
            else:
                expr.append(f"+{_format_number(coeff)}{label}")
        tau = float(np.mod(translation[i], 1.0))
        if tau > 1e-8 and abs(tau - 1.0) > 1e-8:
            expr.append(f"+{_format_number(tau)}")
        token = "".join(expr)
        if not token:
            token = "0"
        elif token[0] == "+":
            token = token[1:]
        parts.append(token)
    return ", ".join(parts)


def _get_layer_ops(result, layer: str) -> list:
    mapping = {
        "g0": result.g0_standard_ssg_ops,
        "l0": result.l0_standard_ssg_ops,
        "acc_primitive": result.acc_primitive_ssg_ops,
        "acc_conventional": result.acc_conventional_ssg_ops,
        "convention": result.convention_ssg_ops,
    }
    if layer not in mapping:
        raise ValueError(f"Unsupported layer: {layer}")
    return mapping[layer]


def _print_ops(result, layer: str, indices: list[int] | None) -> None:
    ssg = SpinSpaceGroup(_get_layer_ops(result, layer))
    ops = list(ssg.ops)
    descriptions = ssg.seitz_descriptions

    selected = range(len(ops)) if not indices else [i - 1 for i in indices]
    print(f"LAYER {layer}")
    print(f"TOTAL_OPS {len(ops)}")
    for idx in selected:
        op = ops[idx]
        desc = descriptions[idx]
        print(f"OP {idx + 1}")
        print(f"  seitz: {desc['symbol']}")
        print(f"  real xyz: {_format_affine(np.asarray(op.rotation, dtype=float), np.asarray(op.translation, dtype=float), XYZ_VARS)}")
        print(f"  spin uvw: {_format_affine(np.asarray(op.spin_rotation, dtype=float), np.zeros(3), UVW_VARS)}")


def _print_current_g0_generator_audit(result, layer: str, tol: float) -> None:
    ssg = SpinSpaceGroup(_get_layer_ops(result, layer))
    current_to_standard, current_to_standard_shift = _compose_setting_transform(
        np.asarray(ssg.transformation_to_G0std, dtype=float),
        np.asarray(ssg.origin_shift_to_G0std, dtype=float),
        np.asarray(ssg.transformation_to_G0std_id, dtype=float),
        np.asarray(ssg.origin_shift_to_G0std_id, dtype=float),
    )
    named_ops, centering = _parse_sg_generator_ops(int(ssg.G0_num))
    named_ops_cur, centering_cur = _transport_standard_generators_to_current_basis(
        named_ops,
        centering,
        current_to_standard,
        current_to_standard_shift,
    )
    spin_map = _canonical_spin_symbol_map(ssg)

    print(f"CURRENT_G0_GENERATOR_AUDIT layer={layer}")
    print(f"current_to_standard={current_to_standard.tolist()}")
    print(f"current_to_standard_shift={current_to_standard_shift.tolist()}")
    print(f"centering_cur={[np.mod(v, 1.0).tolist() for v in centering_cur]}")
    for i, (rotation, translation) in enumerate(named_ops_cur, 1):
        matched = _find_real_operation(ssg.nssg, rotation, translation, tol=tol)
        print(f"GEN {i}")
        print(f"  target real xyz: {_format_affine(rotation, translation, XYZ_VARS)}")
        if matched is None:
            print("  matched: NONE")
            print("  spin uvw: <fallback 1>")
        else:
            print(
                f"  matched real xyz: {_format_affine(np.asarray(matched.rotation, dtype=float), np.asarray(matched.translation, dtype=float), XYZ_VARS)}"
            )
            print(
                f"  matched spin uvw: {_format_affine(np.asarray(matched.spin_rotation, dtype=float), np.zeros(3), UVW_VARS)}"
            )
            print(f"  spin symbol: {spin_map.get(id(matched), 'FALLBACK_1')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print SSG operations as real-space xyz and spin-space uvw affine forms."
    )
    parser.add_argument("source", help="Input structure file, typically an .mcif path")
    parser.add_argument(
        "--layer",
        choices=["g0", "l0", "acc_primitive", "acc_conventional", "convention"],
        default="convention",
        help="Which result SSG layer to inspect",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="1-based operation indices to print; omit to print all ops in the layer",
    )
    parser.add_argument(
        "--current-g0-generator-audit",
        action="store_true",
        help="Audit transported current-basis G0 generators against the selected layer's current nssg ops",
    )
    parser.add_argument("--tol", type=float, default=1e-4, help="Operation matching tolerance")
    args = parser.parse_args()

    result = find_spin_group(args.source)
    print(f"CASE {args.source}")
    print(f"INDEX {result.index}")
    print(f"ACC {result.acc}")
    print(f"CONVENTION_SETTING {result.convention_ssg_setting}")
    print(f"CONVENTION_SSG {result.convention_ssg_international_linear}")
    print(f"GSPG {result.gspg_symbol_linear}")
    print("---")

    if args.current_g0_generator_audit:
        _print_current_g0_generator_audit(result, args.layer, args.tol)
    else:
        _print_ops(result, args.layer, args.indices)


if __name__ == "__main__":
    main()
