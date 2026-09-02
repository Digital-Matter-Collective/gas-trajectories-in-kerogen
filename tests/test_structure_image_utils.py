from pathlib import Path

import pytest

from scripts.structure_image_utils import (
    collect_indexes,
    generate_indexes_by_mode,
    generate_indexes_from_available_structures,
    parse_indexes,
)


def test_parse_indexes_splits_commas_and_whitespace() -> None:
    assert parse_indexes(["25000", "50000,75000", "100000 125000"]) == [
        25000,
        50000,
        75000,
        100000,
        125000,
    ]


def test_collect_indexes_parses_cli_strings_to_ints(tmp_path: Path) -> None:
    indexes_file = tmp_path / "indexes.txt"
    indexes_file.write_text("300000,325000\n350000\n")

    result = collect_indexes(["25000", "50000,75000"], indexes_file)

    assert result == [25000, 50000, 75000, 300000, 325000, 350000]
    assert all(isinstance(index, int) for index in result)


def test_collect_indexes_deduplicates_and_requires_at_least_one() -> None:
    assert collect_indexes(["1", "1", "2"], None) == [1, 2]
    with pytest.raises(ValueError, match="Pass at least one"):
        collect_indexes([], None)


def test_generate_indexes_by_mode_all_returns_exact_count_with_endpoints() -> (
    None
):
    # Reproduces the P1-05 report: 6,560-step trajectory, 500 requested
    # structures must not silently become 501.
    indexes = generate_indexes_by_mode(
        start_step=25000,
        step_size=250000,
        full_count_steps=6560,
        count_structures=500,
        mode="all",
    )

    assert len(indexes) == 500
    assert len(set(indexes)) == 500
    assert indexes[0] == 25000
    assert indexes[-1] == 25000 + 250000 * 6560


def test_generate_indexes_by_mode_all_handles_fewer_positions_than_requested() -> (
    None
):
    # Only 11 distinct integer positions exist between 0 and 10 inclusive.
    indexes = generate_indexes_by_mode(
        start_step=0,
        step_size=1,
        full_count_steps=10,
        count_structures=500,
        mode="all",
    )

    assert len(indexes) == 11
    assert indexes[0] == 0
    assert indexes[-1] == 10


def test_generate_indexes_by_mode_all_single_structure_is_the_start() -> None:
    indexes = generate_indexes_by_mode(
        start_step=25000,
        step_size=250000,
        full_count_steps=6560,
        count_structures=1,
        mode="all",
    )
    assert indexes == [25000]


def test_generate_indexes_by_mode_part_returns_first_consecutive_steps() -> (
    None
):
    indexes = generate_indexes_by_mode(
        start_step=0,
        step_size=250000,
        full_count_steps=6560,
        count_structures=10,
        mode="part",
    )
    assert indexes == [250000 * i for i in range(10)]


def test_generate_indexes_from_available_structures_all_returns_exact_count(
    tmp_path: Path,
) -> None:
    available = list(range(0, 6561 * 250000, 250000))
    from unittest.mock import patch

    with patch(
        "scripts.structure_image_utils.list_available_structure_indexes",
        return_value=available,
    ):
        indexes = generate_indexes_from_available_structures(
            tmp_path, mode="all", count_slices=500
        )

    assert len(indexes) == 500
    assert len(set(indexes)) == 500
    assert indexes[0] == available[0]
    assert indexes[-1] == available[-1]
