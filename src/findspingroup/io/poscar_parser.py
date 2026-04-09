from pathlib import Path
import re

import numpy as np

from ..structure.cell import calculate_lattice_params, transform_moments
from ..utils.matrix_utils import evaluate_numeric_expression


def _read_text(path: str) -> str:
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _parse_scalar(token: str) -> float:
    return float(evaluate_numeric_expression(token))


def _parse_vector(tokens: list[str]) -> np.ndarray:
    if len(tokens) < 3:
        raise ValueError("POSCAR vector line has fewer than 3 numeric entries.")
    return np.array([_parse_scalar(tokens[0]), _parse_scalar(tokens[1]), _parse_scalar(tokens[2])], dtype=float)


def _parse_scale_factors(scale_line: str, base_lattice: np.ndarray) -> np.ndarray:
    tokens = scale_line.split()
    if len(tokens) not in {1, 3}:
        raise ValueError("POSCAR scaling line must contain one or three numbers.")

    if len(tokens) == 3:
        scale_vector = np.array([_parse_scalar(token) for token in tokens], dtype=float)
        if np.any(scale_vector <= 0):
            raise ValueError("Three-component POSCAR scaling factors must all be positive.")
        return scale_vector

    scale = _parse_scalar(tokens[0])
    if scale == 0:
        raise ValueError("POSCAR scaling factor cannot be zero.")
    if scale > 0:
        return np.array([scale, scale, scale], dtype=float)

    desired_volume = abs(scale)
    base_volume = abs(float(np.linalg.det(base_lattice)))
    if base_volume <= 0:
        raise ValueError("POSCAR base lattice has non-positive volume.")
    inferred_scale = (desired_volume / base_volume) ** (1.0 / 3.0)
    return np.array([inferred_scale, inferred_scale, inferred_scale], dtype=float)


def _expand_species(species: list[str], counts: list[int]) -> list[str]:
    expanded = []
    for symbol, count in zip(species, counts):
        expanded.extend([symbol] * count)
    return expanded


def _synthesize_labels(elements: list[str]) -> list[str]:
    per_species_count: dict[str, int] = {}
    labels = []
    for symbol in elements:
        per_species_count[symbol] = per_species_count.get(symbol, 0) + 1
        labels.append(f"{symbol}_{per_species_count[symbol]}")
    return labels


def _parse_magmom_payload(payload: str, n_sites: int) -> np.ndarray:
    expanded_tokens = []
    for token in payload.split():
        match = re.fullmatch(r"(\d+)\*(.+)", token)
        if match is None:
            expanded_tokens.append(token)
            continue
        repeat_count = int(match.group(1))
        repeated_value = match.group(2)
        expanded_tokens.extend([repeated_value] * repeat_count)

    flat = np.array([_parse_scalar(token) for token in expanded_tokens], dtype=float)
    if flat.size == 3 * n_sites:
        return flat.reshape(n_sites, 3)
    if flat.size == n_sites:
        vectors = np.zeros((n_sites, 3), dtype=float)
        # Project scalar collinear MAGMOM values onto the parser's default z axis.
        vectors[:, 2] = flat
        return vectors
    raise ValueError("MAGMOM entry count does not match either NIONS or 3*NIONS.")


def _read_incar_magmom_payload(poscar_filename: str) -> str | None:
    incar_path = Path(poscar_filename).with_name("INCAR")
    if not incar_path.is_file():
        return None

    text = _read_text(str(incar_path))
    logical_lines = []
    buffer = ""
    for raw_line in text.splitlines():
        line = re.split(r"[#!]", raw_line, maxsplit=1)[0].rstrip()
        if not line.strip():
            continue
        if line.endswith("\\"):
            buffer += line[:-1].rstrip() + " "
            continue
        logical_lines.append(buffer + line)
        buffer = ""
    if buffer.strip():
        logical_lines.append(buffer.strip())

    payload = None
    pattern = re.compile(r"^\s*MAGMOM\s*=\s*(.*)$", re.IGNORECASE)
    for logical_line in logical_lines:
        for statement in logical_line.split(";"):
            match = pattern.match(statement.strip())
            if match is not None:
                payload = match.group(1).strip()
    return payload


def _extract_magmom_vectors(lines: list[str], n_sites: int, *, poscar_filename: str) -> np.ndarray:
    pattern = re.compile(r"^\s*#*\s*magmom\s*=\s*(.*)$", re.IGNORECASE)
    magmom_index = None
    magmom_text = None
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match is not None:
            magmom_index = index
            magmom_text = match.group(1).strip()
            break
    if magmom_index is None or magmom_text is None:
        magmom_text = _read_incar_magmom_payload(poscar_filename)
        if magmom_text is None:
            return np.zeros((n_sites, 3), dtype=float)
        return _parse_magmom_payload(magmom_text, n_sites)

    trailing_lines = [segment.strip() for segment in lines[magmom_index + 1 :] if segment.strip()]
    if trailing_lines:
        magmom_text = " ".join([magmom_text, *trailing_lines]).strip()

    return _parse_magmom_payload(magmom_text, n_sites)


def parse_poscar_file(filename):
    text = _read_text(filename)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError("POSCAR file is too short.")

    base_lattice = np.array(
        [_parse_vector(lines[2].split()), _parse_vector(lines[3].split()), _parse_vector(lines[4].split())],
        dtype=float,
    )
    scale_vector = _parse_scale_factors(lines[1], base_lattice)
    lattice_matrix = base_lattice * scale_vector

    species = lines[5].split()
    counts_line_index = 6
    if all(token.lstrip("+-").isdigit() for token in species):
        raise ValueError(
            "POSCAR species names are omitted. The current parser requires species names because it does not read POTCAR."
        )

    counts = [int(token) for token in lines[counts_line_index].split()]
    if not species or len(species) != len(counts):
        raise ValueError("POSCAR species/count lines are inconsistent.")
    n_sites = sum(counts)

    mode_index = counts_line_index + 1
    selective_dynamics = False
    maybe_mode = lines[mode_index].strip().lower()
    if maybe_mode.startswith("s"):
        selective_dynamics = True
        mode_index += 1

    coord_mode = lines[mode_index].strip().lower()
    cartesian_mode = coord_mode.startswith(("c", "k"))
    direct_mode = not cartesian_mode

    position_lines = lines[mode_index + 1 : mode_index + 1 + n_sites]
    if len(position_lines) != n_sites:
        raise ValueError("POSCAR coordinate block length does not match the atom count.")

    coordinate_rows = np.array([_parse_vector(line.split()) for line in position_lines], dtype=float)
    if cartesian_mode:
        cartesian_positions = coordinate_rows * scale_vector
        positions = cartesian_positions @ np.linalg.inv(lattice_matrix)
    elif direct_mode:
        positions = coordinate_rows
    else:
        raise ValueError("POSCAR coordinate mode is not recognized.")

    if selective_dynamics:
        # The current parser intentionally ignores T/F flags after the first three coordinates.
        pass

    positions = positions % 1.0
    magmom_vectors = _extract_magmom_vectors(lines[mode_index + 1 + n_sites :], n_sites, poscar_filename=str(filename))
    # POSCAR MAGMOM vectors are written in the file's Cartesian spin frame. Convert
    # them back into the parser's in-lattice convention expected by downstream APIs.
    moments_in_lattice = transform_moments(magmom_vectors, calculate_lattice_params(lattice_matrix), inverse=True)

    elements = _expand_species(species, counts)
    occupancies = [1.0] * n_sites
    labels = _synthesize_labels(elements)
    lattice_factors = np.array(calculate_lattice_params(lattice_matrix), dtype=float)

    return lattice_factors, positions, elements, occupancies, labels, moments_in_lattice
