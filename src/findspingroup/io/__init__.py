from pathlib import Path

from .cif_parser import (
    parse_cif_file,
    parse_cif_metadata,
    parse_scif_file,
    parse_scif_metadata,
    parse_scif_text,
)
from .poscar_parser import parse_poscar_file


def parse_structure_file(filename, atol=0.02, return_metadata=False):
    path = Path(filename)
    suffix = path.suffix.lower()
    basename = path.name.lower()
    if suffix == '.scif':
        return parse_scif_file(filename, atol=atol, return_metadata=return_metadata)
    if suffix in {'.vasp', '.poscar'} or basename in {'poscar', 'contcar'}:
        if return_metadata:
            return parse_poscar_file(filename), None
        return parse_poscar_file(filename)
    if return_metadata:
        return parse_cif_file(filename, atol=atol, return_metadata=True)
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
