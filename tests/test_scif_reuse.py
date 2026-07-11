import importlib

from findspingroup import find_spin_group


find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")
scif_module = importlib.import_module("findspingroup.io.scif_generator")


def test_full_route_reuses_scif_loops_and_skips_supplied_symbol_transform(
    monkeypatch,
):
    loop_calls = 0
    transform_strip_flags = []
    original_loops = find_spin_group_module._resolve_scif_operation_loops
    original_transform = scif_module._resolve_transform_chen_parts

    def counted_loops(ssg):
        nonlocal loop_calls
        loop_calls += 1
        return original_loops(ssg)

    def counted_transform(*args, **kwargs):
        transform_strip_flags.append(
            kwargs.get("strip_spin_lattice_lengths", True)
        )
        return original_transform(*args, **kwargs)

    monkeypatch.setattr(
        find_spin_group_module,
        "_resolve_scif_operation_loops",
        counted_loops,
    )
    monkeypatch.setattr(
        scif_module,
        "_resolve_transform_chen_parts",
        counted_transform,
    )

    result = find_spin_group("examples/0.800_MnTe.mcif")

    assert len(result.scif_outputs) == 8
    assert loop_calls < len(result.scif_outputs)
    assert transform_strip_flags.count(False) == 1
    assert transform_strip_flags.count(True) == len(result.scif_outputs)
