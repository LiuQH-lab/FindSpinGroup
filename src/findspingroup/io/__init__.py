from pathlib import Path

from .cif_parser import (
    parse_cif_file,
    parse_cif_metadata,
    parse_scif_file,
    parse_scif_metadata,
    parse_scif_text,
)
from .poscar_parser import parse_poscar_file


def parse_structure_file(
    filename,
    atol=0.02,
    return_metadata=False,
    *,
    poscar_allow_incar_magmom: bool = False,
    poscar_prefer_incar_magmom: bool = False,
):
    path = Path(filename)
    suffix = path.suffix.lower()
    basename = path.name.lower()
    if suffix == '.scif':
        if return_metadata:
            parsed, metadata = parse_scif_file(filename, atol=atol, return_metadata=True)
            enriched = {} if metadata is None else dict(metadata)
            enriched.setdefault("source_format", "scif")
            spinframe_abc = (
                enriched.get("space_group_spin", {}).get("transform_spinframe_P_abc")
                if isinstance(enriched.get("space_group_spin"), dict)
                else None
            )
            normalized_spinframe = None if spinframe_abc is None else "".join(str(spinframe_abc).split())
            enriched.setdefault(
                "spin_setting",
                "in_lattice" if normalized_spinframe in {None, "a,b,c"} else "cartesian",
            )
            return parsed, enriched
        return parse_scif_file(filename, atol=atol)
    if suffix in {'.vasp', '.poscar'} or basename in {'poscar', 'contcar'}:
        parsed = parse_poscar_file(
            filename,
            allow_incar_magmom=poscar_allow_incar_magmom,
            prefer_incar_magmom=poscar_prefer_incar_magmom,
        )
        if return_metadata:
            return parsed, {
                "source_format": "poscar",
                "spin_setting": "cartesian",
                "poscar_allow_incar_magmom": poscar_allow_incar_magmom,
                "poscar_prefer_incar_magmom": poscar_prefer_incar_magmom,
            }
        return parsed
    if return_metadata:
        parsed, metadata = parse_cif_file(filename, atol=atol, return_metadata=True)
        enriched = {} if metadata is None else dict(metadata)
        enriched.setdefault("source_format", "cif")
        enriched.setdefault("spin_setting", "in_lattice")
        return parsed, enriched
    return parse_cif_file(filename, atol=atol)


__all__ = [
    'parse_cif_file',
    'parse_cif_metadata',
    'parse_scif_file',
    'parse_scif_metadata',
    'parse_scif_text',
    'parse_poscar_file',
    'parse_structure_file',
]
