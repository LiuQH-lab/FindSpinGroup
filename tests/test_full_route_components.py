import importlib

import pytest

from findspingroup import find_spin_group


find_spin_group_module = importlib.import_module("findspingroup.find_spin_group")


def test_full_route_default_component_plan_preserves_all_optional_outputs():
    result = find_spin_group("examples/0.800_MnTe.mcif")

    assert result.tensor_outputs
    assert result.scif_outputs
    assert result.operation_views
    assert result.spin_texture_config_no_soc is not None
    assert result.spin_texture_config_soc is not None


def test_full_route_empty_component_plan_skips_independent_heavy_outputs(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("disabled full-route component was evaluated")

    monkeypatch.setattr(find_spin_group_module, "_compute_tensor_outputs", unexpected)
    monkeypatch.setattr(find_spin_group_module, "_build_operation_views", unexpected)
    monkeypatch.setattr(find_spin_group_module, "_spin_texture_config_from_ossg_convention", unexpected)
    monkeypatch.setattr(find_spin_group_module, "generate_scif", unexpected)

    result = find_spin_group("examples/0.800_MnTe.mcif", components=())

    assert result.tensor_outputs == {}
    assert result.scif is None
    assert result.scif_outputs == {}
    assert result.scif_cell_modes == []
    assert result.operation_views is None
    assert result.spin_texture_config_database is None
    assert result.spin_texture_config_no_soc is None
    assert result.spin_texture_config_soc is None
    assert result.index == "194.164.1.1.L"
    assert result.KPOINTS


def test_full_route_component_plan_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown full-route components: unknown"):
        find_spin_group("examples/0.800_MnTe.mcif", components={"unknown"})
