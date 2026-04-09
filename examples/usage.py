from pathlib import Path

from findspingroup import find_spin_group


def main():
    # Swap this path to any supported `.mcif` or repo-generated `.scif`.
    path = Path(__file__).resolve().parent / "Fe.mcif"
    result = find_spin_group(str(path))

    print("index:", result.index)
    print("conf:", result.conf)
    print("convention_ssg:", result.convention_ssg_international_linear)

    # Default public `.scif` output remains the legacy profile in the
    # oriented-G0 export setting.
    scif_default = result.scif
    print("\n=== default .scif (legacy + g0std_oriented) ===")
    print("\n".join(scif_default.splitlines()[:18]))

    # The same content is also available explicitly through `to_scif(...)`.
    scif_legacy = result.to_scif(profile="legacy", cell_mode="g0std_oriented")
    assert scif_legacy == result.scif

    # Draft-aligned experimental profile.
    scif_working = result.to_scif(profile="spincif_working", cell_mode="g0std_oriented")
    print("\n=== spincif_working header ===")
    print("\n".join(scif_working.splitlines()[:12]))

    # Additional export-cell modes exist for internal audit and roundtrip work.
    scif_primitive = result.to_scif(profile="legacy", cell_mode="magnetic_primitive")
    print("\n=== magnetic_primitive .scif header ===")
    print("\n".join(scif_primitive.splitlines()[:12]))

    print("\nNotes:")
    print("- repo-local FINDSPINGROUP metadata now uses CIF-legal `_space_group_spin.fsg_*` tags")
    print("- symmetry-operation and transform coefficients prefer exact-looking forms such as `1/3` or `sqrt(6)/3` when numerically justified")
    print("- default public `.scif` output is still `legacy + g0std_oriented`")


if __name__ == "__main__":
    main()
