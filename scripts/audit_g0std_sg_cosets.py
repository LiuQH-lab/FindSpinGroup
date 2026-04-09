#!/usr/bin/env python3

from __future__ import annotations

import argparse

from findspingroup.g0std_sg_cosets import analyze_g0std_space_group_cosets_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ordinary SG operations on the G0std cell and partition them by the SSG real-space subgroup.",
    )
    parser.add_argument("source", help="Input structure file, typically an .mcif path.")
    parser.add_argument("--symprec", type=float, default=0.02, help="spglib symprec for the G0std cell.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Operation matching tolerance.")
    args = parser.parse_args()

    print(analyze_g0std_space_group_cosets_json(args.source, symprec=args.symprec, tol=args.tol))


if __name__ == "__main__":
    main()
