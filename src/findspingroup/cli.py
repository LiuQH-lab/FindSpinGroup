import argparse
import json
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


def _discover_structure_candidates(cwd: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(cwd.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if name_lower in _AUTO_IGNORE_NAMES:
            continue
        if name_lower in {"poscar", "contcar"} or path.suffix.lower() in _AUTO_INPUT_EXTENSIONS:
            candidates.append(path)
    return candidates


def _candidate_priority(path: Path) -> tuple[int, str]:
    name_lower = path.name.lower()
    if name_lower == "poscar":
        return (0, name_lower)
    if name_lower == "contcar":
        return (1, name_lower)
    suffix = path.suffix.lower()
    if suffix == ".scif":
        return (2, name_lower)
    if suffix == ".mcif":
        return (3, name_lower)
    if suffix == ".cif":
        return (4, name_lower)
    if suffix in {".vasp", ".poscar"}:
        return (5, name_lower)
    return (99, name_lower)


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


def _legacy_mode_payload(args):
    if args.mode == "basic":
        return find_spin_group_basic(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            parser_atol=args.parser_atol,
        )
    if args.mode == "acc-primitive":
        payload = find_spin_group_acc_primitive(
            args.structure_file,
            space_tol=args.space_tol,
            mtol=args.mtol,
            meigtol=args.meigtol,
            matrix_tol=args.matrix_tol,
            parser_atol=args.parser_atol,
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
    )
    return _to_serializable_payload(result)


def main():
    parser = argparse.ArgumentParser(description="Calculate Spin Space Groups from magnetic structure files.")
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
    parser.add_argument("--space_tol", type=float, default=0.02, help="Spatial tolerance")
    parser.add_argument("--mtol", type=float, default=0.02, help="Magnetic tolerance")
    parser.add_argument("--meigtol", type=float, default=0.00002, help="Point-group eigenvalue tolerance")
    parser.add_argument("--matrix_tol", type=float, default=0.01, help="Point-group standardization tolerance")
    parser.add_argument("--parser_atol", type=float, default=0.02, help="CIF/SCIF parser expansion tolerance")

    args = parser.parse_args()

    try:
        args.structure_file = _select_structure_file(args.structure_file)

        if args.mode is not None:
            if args.all or args.write or args.show:
                raise ValueError("Use either legacy `--mode` or the new `--all/--show/-w` flags, not both.")
            payload = _legacy_mode_payload(args)
            print(json.dumps(payload, indent=2, ensure_ascii=False, cls=NumpyEncoder))
            return

        if args.all and args.write:
            raise ValueError("`--all` and `-w/--write` cannot be used together.")

        if args.write:
            payload = find_spin_group_input_ssg(
                args.structure_file,
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
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
            )

        _emit_payload(payload, args.show)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
