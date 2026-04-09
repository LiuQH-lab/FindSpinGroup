from pathlib import Path

import numpy as np

from findspingroup import find_spin_group, find_spin_group_from_data
from findspingroup.io import parse_poscar_file, parse_structure_file
def _rewrite_generated_poscar_to_selective_dynamics(text: str) -> str:
    lines = text.splitlines()
    atom_count = sum(int(token) for token in lines[6].split())
    rewritten = lines[:7] + ["Selective dynamics", lines[7]]
    for index in range(atom_count):
        rewritten.append(f"{lines[8 + index]} T F T")
    rewritten.extend(lines[8 + atom_count :])
    return "\n".join(rewritten) + "\n"


def _rewrite_generated_poscar_to_cartesian(text: str) -> str:
    lines = text.splitlines()
    atom_count = sum(int(token) for token in lines[6].split())
    lattice = np.array([[float(x) for x in lines[2].split()], [float(x) for x in lines[3].split()], [float(x) for x in lines[4].split()]])
    rewritten = lines[:7] + ["Cartesian"]
    for index in range(atom_count):
        frac = np.array([float(x) for x in lines[8 + index].split()[:3]], dtype=float)
        cart = frac @ lattice
        rewritten.append(" ".join(f"{value:.8f}" for value in cart))
    rewritten.extend(lines[8 + atom_count :])
    return "\n".join(rewritten) + "\n"


def _rewrite_generated_poscar_to_three_scale(text: str, scale_vector=(2.0, 3.0, 4.0)) -> str:
    lines = text.splitlines()
    lattice = np.array([[float(x) for x in lines[2].split()], [float(x) for x in lines[3].split()], [float(x) for x in lines[4].split()]])
    scale_vector = np.array(scale_vector, dtype=float)
    base_lattice = lattice / scale_vector
    rewritten = lines.copy()
    rewritten[1] = " ".join(str(value) for value in scale_vector)
    for row_index in range(3):
        rewritten[2 + row_index] = " ".join(f"{value:.8f}" for value in base_lattice[row_index])
    return "\n".join(rewritten) + "\n"


def _rewrite_generated_poscar_to_negative_volume(text: str) -> str:
    lines = text.splitlines()
    lattice = np.array([[float(x) for x in lines[2].split()], [float(x) for x in lines[3].split()], [float(x) for x in lines[4].split()]])
    actual_volume = abs(float(np.linalg.det(lattice)))
    base_lattice = lattice / 2.0
    rewritten = lines.copy()
    rewritten[1] = f"{-actual_volume:.8f}"
    for row_index in range(3):
        rewritten[2 + row_index] = " ".join(f"{value:.8f}" for value in base_lattice[row_index])
    return "\n".join(rewritten) + "\n"


def _rewrite_generated_poscar_without_magmom(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:-1]) + "\n"


def _rewrite_generated_poscar_with_relaxed_magmom_comment(text: str) -> str:
    lines = text.splitlines()
    magmom_payload = lines[-1].split("=", 1)[1].strip()
    lines[-1] = f"##\t MAGMOM \t = \t{magmom_payload}"
    return "\n".join(lines) + "\n"


def _write_incar(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_parse_poscar_file_roundtrips_generated_mnte_poscar(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(
        str(poscar_path),
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_parse_poscar_file_roundtrips_generated_non_orthogonal_acc_primitive_poscar(tmp_path):
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.250_(NH2(CH3)2)(FeCo(HCOO)6).mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(
        str(poscar_path),
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_parse_poscar_file_roundtrips_generated_precision_sensitive_pyrochlore_poscar(tmp_path):
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.2_Cd2Os2O7.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(
        str(poscar_path),
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_parse_poscar_file_roundtrips_generated_precision_sensitive_tetragonal_poscar(tmp_path):
    original = find_spin_group("tests/testset/mcif_241130_no2186/0.151_Tm2Mn2O7.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(
        str(poscar_path),
        lattice_factors,
        positions,
        elements,
        occupancies,
        moments,
    )

    assert roundtrip.index == original.index
    assert roundtrip.conf == original.conf


def test_parse_structure_file_dispatches_poscar_basename(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    parsed = parse_structure_file(poscar_path)
    direct = parse_poscar_file(poscar_path)

    assert parsed[0].tolist() == direct[0].tolist()
    assert len(parsed[1]) == len(direct[1])
    assert parsed[2] == direct[2]
    assert parsed[3] == direct[3]
    assert parsed[4] == direct[4]


def test_parse_structure_file_dispatches_poscar_suffix(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "mnte.poscar"
    poscar_path.write_text(original.acc_primitive_magnetic_cell_poscar, encoding="utf-8")

    parsed = parse_structure_file(poscar_path)
    roundtrip = find_spin_group_from_data(
        str(poscar_path),
        parsed[0],
        parsed[1],
        parsed[2],
        parsed[3],
        parsed[5],
    )

    assert roundtrip.index == original.index


def test_parse_poscar_file_supports_selective_dynamics(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(
        _rewrite_generated_poscar_to_selective_dynamics(original.acc_primitive_magnetic_cell_poscar),
        encoding="utf-8",
    )

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(str(poscar_path), lattice_factors, positions, elements, occupancies, moments)

    assert roundtrip.index == original.index


def test_parse_poscar_file_supports_cartesian_coordinates(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(
        _rewrite_generated_poscar_to_cartesian(original.acc_primitive_magnetic_cell_poscar),
        encoding="utf-8",
    )

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(str(poscar_path), lattice_factors, positions, elements, occupancies, moments)

    assert roundtrip.index == original.index


def test_parse_poscar_file_supports_three_component_scale_factors(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(
        _rewrite_generated_poscar_to_three_scale(original.acc_primitive_magnetic_cell_poscar),
        encoding="utf-8",
    )

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(str(poscar_path), lattice_factors, positions, elements, occupancies, moments)

    assert roundtrip.index == original.index


def test_parse_poscar_file_supports_negative_volume_scale(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(
        _rewrite_generated_poscar_to_negative_volume(original.acc_primitive_magnetic_cell_poscar),
        encoding="utf-8",
    )

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(str(poscar_path), lattice_factors, positions, elements, occupancies, moments)

    assert roundtrip.index == original.index


def test_parse_poscar_file_defaults_missing_magmom_to_zero(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(
        _rewrite_generated_poscar_without_magmom(original.acc_primitive_magnetic_cell_poscar),
        encoding="utf-8",
    )

    parsed = parse_poscar_file(poscar_path)

    assert all(np.allclose(moment, np.zeros(3), atol=1e-12) for moment in parsed[5])


def test_parse_poscar_file_tolerates_relaxed_magmom_comment_format(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(
        _rewrite_generated_poscar_with_relaxed_magmom_comment(original.acc_primitive_magnetic_cell_poscar),
        encoding="utf-8",
    )

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(str(poscar_path), lattice_factors, positions, elements, occupancies, moments)

    assert roundtrip.index == original.index


def test_parse_poscar_file_reads_official_incar_magmom_vector_syntax(tmp_path):
    original = find_spin_group("examples/0.800_MnTe.mcif")
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(
        _rewrite_generated_poscar_without_magmom(original.acc_primitive_magnetic_cell_poscar),
        encoding="utf-8",
    )
    _write_incar(
        Path(tmp_path) / "INCAR",
        "ISPIN = 2 ; SIGMA = 0.05\n"
        "MAGMOM = -2.3 -3.9837 0 \\\n"
        "          2.3 3.9837 0 \\\n"
        "          6*0.0  # trailing comment\n",
    )

    lattice_factors, positions, elements, occupancies, labels, moments = parse_poscar_file(poscar_path)
    roundtrip = find_spin_group_from_data(str(poscar_path), lattice_factors, positions, elements, occupancies, moments)

    assert roundtrip.index == original.index


def test_parse_poscar_file_reads_official_incar_magmom_scalar_syntax(tmp_path):
    poscar_text = "\n".join(
        [
            "Fe",
            "1.0",
            "1 0 0",
            "0 1 0",
            "0 0 1",
            "Fe",
            "2",
            "Direct",
            "0 0 0",
            "0.5 0.5 0.5",
        ]
    ) + "\n"
    poscar_path = Path(tmp_path) / "POSCAR"
    poscar_path.write_text(poscar_text, encoding="utf-8")
    _write_incar(Path(tmp_path) / "INCAR", "MAGMOM = 1.5 -1.5\n")

    parsed = parse_poscar_file(poscar_path)

    assert len(parsed[5]) == 2
    assert np.allclose(parsed[5][0], np.array([0.0, 0.0, 1.5]), atol=1e-12)
    assert np.allclose(parsed[5][1], np.array([0.0, 0.0, -1.5]), atol=1e-12)
