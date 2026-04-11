import argparse
import json
import sys
from .find_spin_group import (
    find_spin_group,
    find_spin_group_acc_primitive,
    find_spin_group_basic,
    find_spin_group_poscar_ssg,
    write_ssg_operation_matrices,
)


def main():
    parser = argparse.ArgumentParser(description="Calculate Spin Space Groups from CIF files.")
    parser.add_argument("cif_file", help="Path to the CIF file")
    parser.add_argument(
        "--mode",
        choices=["full", "basic", "acc-primitive", "poscar-ssg"],
        default="full",
        help="Choose the full pipeline or the lightweight identification route.",
    )
    parser.add_argument(
        "--write-ssg-matrices",
        help="When used with --mode acc-primitive, write the selected SSG operation matrices to a JSON file.",
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
        if args.mode == "basic":
            payload = find_spin_group_basic(
                args.cif_file,
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
                parser_atol=args.parser_atol,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        elif args.mode == "acc-primitive":
            payload = find_spin_group_acc_primitive(
                args.cif_file,
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
            print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        elif args.mode == "poscar-ssg":
            payload = find_spin_group_poscar_ssg(
                args.cif_file,
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            result = find_spin_group(
                args.cif_file,
                space_tol=args.space_tol,
                mtol=args.mtol,
                meigtol=args.meigtol,
                matrix_tol=args.matrix_tol,
                parser_atol=args.parser_atol,
            )
            print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
