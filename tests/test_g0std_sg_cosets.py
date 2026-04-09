from findspingroup.g0std_sg_cosets import analyze_g0std_space_group_cosets


def test_analyze_g0std_space_group_cosets_for_1048_mnse2():
    payload = analyze_g0std_space_group_cosets(
        "tests/testset/mcif_241130_no2186/1.0.48_MnSe2.mcif"
    )

    assert payload["expected_g0_number"] == 29
    assert payload["expected_g0_symbol"] == "Pca2_1"
    assert payload["expected_g0_hall_number"] == 143
    assert payload["expected_g0_database_op_count"] == 4
    assert payload["g0std_spglib_number"] == 205
    assert payload["g0std_spglib_symbol"] == "Pa-3"
    assert payload["matches_expected_g0_number"] is False
    assert payload["ops_identified_number"] == 61
    assert payload["ops_identified_symbol"] == "Pbca"
    assert payload["sg_op_count"] == 24
    assert payload["ssg_real_op_count"] == 4
    assert payload["ssg_real_subset_of_sg"] is True
    assert payload["right_coset_count"] == 6
    assert payload["left_coset_count"] == 6
    assert payload["right_coset_sizes"] == [4, 4, 4, 4, 4, 4]
    assert payload["left_coset_sizes"] == [4, 4, 4, 4, 4, 4]
    assert payload["left_equals_right"] is False
