import argparse
import sys
from .find_spin_group import find_spin_group


def main():
    parser = argparse.ArgumentParser(description="Calculate Spin Space Groups from CIF files.")
    parser.add_argument("cif_file", help="Path to the CIF file")
    parser.add_argument("--space_tol", type=float, default=0.02, help="Spatial tolerance")
    parser.add_argument("--mtol", type=float, default=0.02, help="Magnetic tolerance")
    parser.add_argument("--meigtol", type=float, default=0.00002, help="Point-group eigenvalue tolerance")
    parser.add_argument("--matrix_tol", type=float, default=0.01, help="Point-group standardization tolerance")
    parser.add_argument("--parser_atol", type=float, default=0.02, help="CIF/SCIF parser expansion tolerance")

    args = parser.parse_args()

    try:
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
