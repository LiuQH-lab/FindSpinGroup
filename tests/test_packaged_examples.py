from findspingroup import example_path, find_spin_group


def test_packaged_example_path_resolves_and_runs():
    result = find_spin_group(example_path("0.800_MnTe.mcif"))

    assert result.index is not None
    assert result.convention_ssg_international_linear
