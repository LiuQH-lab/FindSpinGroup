import argparse
import sys
from .find_spin_group import find_spin_groups


def main():
    parser = argparse.ArgumentParser(description="Calculate Spin Space Groups from CIF files.")
    parser.add_argument("cif_file", help="Path to the CIF file")
    parser.add_argument("--space_tol", type=float, default=0.02, help="Spatial tolerance")
    parser.add_argument("--mtol", type=float, default=0.02, help="Magnetic tolerance")

    args = parser.parse_args()

    try:
        result = find_spin_groups(args.cif_file, space_tol=args.space_tol, mtol=args.mtol)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()