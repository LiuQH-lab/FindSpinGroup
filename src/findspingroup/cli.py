import argparse
import json
import re
import sys
from pathlib import Path

from .find_spin_group import (
    NumpyEncoder,
    find_spin_group,
    find_spin_group_acc_primitive,
    find_spin_group_basic,
    find_spin_group_input_ssg,
    find_spin_group_poscar_ssg,
    write_poscar_ssg_symmetry_dat,
    write_ssg_operation_matrices,
)


_AUTO_INPUT_EXTENSIONS = {".scif", ".mcif", ".cif", ".vasp", ".poscar"}
_AUTO_IGNORE_NAMES = {"ssg_symm.json", "input_poscar.vasp", "magnetic_primitive_poscar.vasp"}
_POSCAR_MAGMOM_PATTERN = re.compile(r"^\s*#*\s*magmom\s*=", re.IGNORECASE | re.MULTILINE)
_INCAR_MAGMOM_PATTERN = re.compile(r"^\s*MAGMOM\s*=", re.IGNORECASE | re.MULTILINE)
_CIF_MOMENT_TAG_MARKERS = (
    "_atom_site_moment.",
    "_atom_site_moment_",
    "_atom_site_spin_moment.",
    "_atom_site_spin_moment_",
    "_atom_site_orbital_moment.",
    "_atom_site_orbital_moment_",
)


def _read_text_for_auto_detect(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _has_embedded_poscar_magmom(path: Path) -> bool:
    try:
        return _POSCAR_MAGMOM_PATTERN.search(_read_text_for_auto_detect(path)) is not None
    except OSError:
        return False


def _has_sibling_incar_magmom(path: Path) -> bool:
    incar_path = path.with_name("INCAR")
    if not incar_path.is_file():
        return False
    try:
        logical_text = []
        for raw_line in _read_text_for_auto_detect(incar_path).splitlines():
            line = re.split(r"[#!]", raw_line, maxsplit=1)[0].strip()
            if line:
                logical_text.append(line)
        return _INCAR_MAGMOM_PATTERN.search("\n".join(logical_text)) is not None
    except OSError:
        return False


def _has_cif_moment_tags(path: Path) -> bool:
    try:
        text = _read_text_for_auto_detect(path).lower()
    except OSError:
        return False
    return any(marker in text for marker in _CIF_MOMENT_TAG_MARKERS)


def _discover_structure_candidates(cwd: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(cwd.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if name_lower in _AUTO_IGNORE_NAMES:
            continue
        if name_lower in {"poscar", "contcar"} or path.suffix.lower() in _AUTO_INPUT_EXTENSIONS:
            if _candidate_priority(path) is None:
                continue
            candidates.append(path)
    return candidates


def _candidate_priority(path: Path) -> tuple[int, str] | None:
    name_lower = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".scif":
        return (0, name_lower)
    if suffix == ".mcif":
        return (1, name_lower)
    if suffix == ".cif" and _has_cif_moment_tags(path):
        return (2, name_lower)
    if name_lower == "poscar" and _has_sibling_incar_magmom(path):
        return (3, name_lower)
    if name_lower == "poscar" and _has_embedded_poscar_magmom(path):
        return (4, name_lower)
    if suffix in {".vasp", ".poscar"} and _has_embedded_poscar_magmom(path):
        return (5, name_lower)
    if name_lower == "contcar":
        return (6, name_lower)
    return None


def _select_structure_file(explicit_file: str | None) -> str:
    if explicit_file:
        return explicit_file

    cwd = Path.cwd()
    candidates = _discover_structure_candidates(cwd)
    if not candidates:
        raise ValueError("No readable structure file was found in the current directory.")

    ordered = sorted(candidates, key=_candidate_priority)
    selected = ordered[0]
    if len(ordered) == 1:
        print(f"[fsg] Auto-selected structure file: {selected.name}", file=sys.stderr)
    else:
        others = ", ".join(path.name for path in ordered[1:])
        print(
            f"[fsg] Multiple structure files found. Using {selected.name}. Other candidates: {others}",
            file=sys.stderr,
        )
    return str(selected)


def _to_serializable_payload(result):
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


def _resolve_show_path(value, path: str):
    current = value
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                raise KeyError(path)
            current = current[segment]
            continue
        if hasattr(current, segment):
            current = getattr(current, segment)
            continue
        raise KeyError(path)
    return current


def _emit_payload(payload, show_paths: list[str] | None):
    if not show_paths:
        print(json.dumps(payload, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        return

    resolved = {}
    missing = []
    for path in show_paths:
        try:
            resolved[path] = _resolve_show_path(payload, path)
        except KeyError:
            resolved[path] = None
            missing.append(path)

    if missing:
        print(f"[fsg] Missing fields: {', '.join(missing)}", file=sys.stderr)

    if len(show_paths) == 1:
        value = resolved[show_paths[0]]
        if isinstance(value, (dict, list)):
            print(json.dumps(value, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        else:
            print(value)
        return

    print(json.dumps(resolved, indent=2, ensure_ascii=False, cls=NumpyEncoder))


def _write_input_ssg_output_dir(directory: Path, payload: dict) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = [write_poscar_ssg_symmetry_dat(directory / "ssg_symm.json", payload)]
    input_poscar = payload.get("input_poscar")
    if input_poscar:
        path = directory / "input_poscar.vasp"
        path.write_text(input_poscar, encoding="utf-8")
        written.append(path)
    magnetic_primitive_poscar = payload.get("magnetic_primitive_poscar")
    if magnetic_primitive_poscar:
        path = directory / "magnetic_primitive_poscar.vasp"
        path.write_text(magnetic_primitive_poscar, encoding="utf-8")
        written.append(path)
    return written


def _uses_full_route(args) -> bool:
    return bool(args.all or args.mode == "full")


def _validate_route_options(args) -> None:
    if args.mode is not None:
        if args.all or args.write or args.show:
            raise ValueError("Use either legacy `--mode` or the new `--all/--show/-w` flags, not both.")
    elif args.all and args.write:
        raise ValueError("`--all` and `-w/--write` cannot be used together.")

    if args.write_ssg_matrices and args.mode != "acc-primitive":
        raise ValueError("`--write-ssg-matrices` is only valid with `--mode acc-primitive`.")
    if args.write_symmetry_dat and args.mode not in {"input-ssg", "poscar-ssg"}:
        raise ValueError("`--write-symmetry-dat` is only valid with `--mode input-ssg` or `--mode poscar-ssg`.")

    if args.calculation_mode != "3d" and not _uses_full_route(args):
        raise ValueError("`--calculation-mode` is only supported by the full route; use `--all` or `--mode full`.")
    if args.vacuum_axis != "c" and not _uses_full_route(args):
        raise ValueError("`--vacuum-axis` is only supported by the full route; use `--all` or `--mode full`.")


def _legacy_mode_payload(args):
    poscar_magmom_kwargs = {
        "poscar_allow_incar_magmom": True,
        "poscar_prefer_incar_magmom": True,
    }
    if args.mode == "basic":
        return find_spin_group_basic(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            parser_atol=args.parser_atol,
            **poscar_magmom_kwargs,
        )
    if args.mode == "acc-primitive":
        payload = find_spin_group_acc_primitive(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            parser_atol=args.parser_atol,
            **poscar_magmom_kwargs,
        )
        if args.write_ssg_matrices:
            key = (
                "acc_primitive_ssg_operation_matrices"
                if args.ssg_matrix_setting == "acc-primitive"
                else "acc_primitive_poscar_spin_frame_ssg_operation_matrices"
            )
            write_ssg_operation_matrices(args.write_ssg_matrices, payload[key])
        return payload
    if args.mode in {"poscar-ssg", "input-ssg"}:
        payload = find_spin_group_input_ssg(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            **poscar_magmom_kwargs,
        )
        if args.write_symmetry_dat:
            write_poscar_ssg_symmetry_dat(args.write_symmetry_dat, payload)
        return payload
    result = find_spin_group(
        args.structure_file,
        space_tol=args.space_tol,
        mtol=args.mtol,
        meigtol=args.meigtol,
        matrix_tol=args.matrix_tol,
        parser_atol=args.parser_atol,
        calculation_mode=args.calculation_mode,
        vacuum_axis=args.vacuum_axis,
        **poscar_magmom_kwargs,
    )
    return _to_serializable_payload(result)


def main():
    parser = argparse.ArgumentParser(description="Calculate Spin Space Groups from magnetic structure files.")
    parser.epilog = (
        "For POSCAR-like inputs, the CLI prefers MAGMOM from a sibling INCAR "
        "when present. Direct Python calls read only embedded POSCAR MAGMOM "
        "unless explicitly opted into INCAR reading."
    )
    parser.add_argument("structure_file", nargs="?", help="Path to the magnetic structure file")
    parser.add_argument(
        "--mode",
        choices=["full", "basic", "acc-primitive", "poscar-ssg", "input-ssg"],
        default=None,
        help="Legacy route selector. Prefer the default/basic flow, `--all`, or `-w` for new usage.",
    )
    parser.add_argument("--all", action="store_true", help="Run the full MagSymmetryResult route.")
    parser.add_argument(
        "--show",
        action="append",
        default=[],
        metavar="FIELD",
        help="Show only selected field(s). Supports dot paths like `summary.input_ssg_index`.",
    )
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Run the input-SSG route and write `ssg_symm.json` plus optional POSCAR files to the current directory.",
    )
    parser.add_argument(
        "--write-ssg-matrices",
        help="When used with --mode acc-primitive, write the selected SSG operation matrices to a JSON file.",
    )
    parser.add_argument(
        "--write-symmetry-dat",
        help="Legacy single-file writer for --mode input-ssg/poscar-ssg.",
    )
    parser.add_argument(
        "--ssg-matrix-setting",
        choices=["acc-primitive", "poscar-spin-frame"],
        default="acc-primitive",
        help="Which SSG setting to export when --write-ssg-matrices is used.",
    )
    parser.add_argument(
        "--space-tol",
        "--space_tol",
        dest="space_tol",
        type=float,
        default=0.02,
        help="Spatial tolerance",
    )
    parser.add_argument("--mtol", type=float, default=0.02, help="Magnetic tolerance")
    parser.add_argument("--meigtol", type=float, default=0.00002, help="Point-group eigenvalue tolerance")
    parser.add_argument(
        "--matrix-tol",
        "--matrix_tol",
        dest="matrix_tol",
        type=float,
        default=0.01,
        help="Point-group standardization tolerance",
    )
    parser.add_argument(
        "--parser-atol",
        "--parser_atol",
        dest="parser_atol",
        type=float,
        default=0.02,
        help="CIF/SCIF parser expansion tolerance",
    )
    parser.add_argument(
        "--calculation-mode",
        choices=["auto", "quasi2d", "2d", "3d", "bulk", "slab", "layer"],
        default="3d",
        help="Quasi-2D interpretation mode for additive diagnostics.",
    )
    parser.add_argument(
        "--vacuum-axis",
        choices=["a", "b", "c", "x", "y", "z", "0", "1", "2"],
        default="c",
        help="Input-cell vacuum axis for --calculation-mode quasi2d/2d.",
    )

    args = parser.parse_args()

    try:
        args.structure_file = _select_structure_file(args.structure_file)
        _validate_route_options(args)

        if args.mode is not None:
            payload = _legacy_mode_payload(args)
            print(json.dumps(payload, indent=2, ensure_ascii=False, cls=NumpyEncoder))
            return

        if args.write:
            payload = find_spin_group_input_ssg(
                args.structure_file,
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
                poscar_allow_incar_magmom=True,
                poscar_prefer_incar_magmom=True,
            )
            written = _write_input_ssg_output_dir(Path.cwd(), payload)
            print(
                json.dumps(
                    {
                        "written_files": [path.name for path in written],
                        "summary": payload["summary"],
                    },
                    indent=2,
                    ensure_ascii=False,
                    cls=NumpyEncoder,
                )
            )
            return

        if args.all:
            payload = _to_serializable_payload(
                find_spin_group(
                    args.structure_file,
                    space_tol=args.space_tol,
                    mtol=args.mtol,
                    meigtol=args.meigtol,
                    matrix_tol=args.matrix_tol,
                    parser_atol=args.parser_atol,
                    calculation_mode=args.calculation_mode,
                    vacuum_axis=args.vacuum_axis,
                    poscar_allow_incar_magmom=True,
                    poscar_prefer_incar_magmom=True,
                )
            )
        else:
            payload = find_spin_group_basic(
                args.structure_file,
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
                parser_atol=args.parser_atol,
                poscar_allow_incar_magmom=True,
                poscar_prefer_incar_magmom=True,
            )

        _emit_payload(payload, args.show)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
