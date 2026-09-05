from pathlib import Path

import numpy as np
import numpy.lib.npyio as npyio
import pytest

from base.kerogendata import AtomData
from scripts.structure_image_utils import (
    collect_indexes,
    generate_indexes_by_mode,
    generate_indexes_from_available_structures,
    load_structure,
    parse_indexes,
    save_structure,
)


def _make_atoms(count: int) -> list[AtomData]:
    return [
        AtomData(
            struct_number=i,
            struct_type="KRG",
            atom_id=f"C{i}",
            type_id=i % 5,
            pos=np.array([i * 0.1, i * 0.2, i * 0.3], dtype=np.float32),
        )
        for i in range(count)
    ]


def test_load_structure_round_trips_save_structure(tmp_path: Path) -> None:
    atoms = _make_atoms(7)
    path = tmp_path / "struct.npz"
    save_structure(path, (42, 12.5, np.array(atoms), (1.0, 2.0, 3.0)))

    num, time_ps, loaded_atoms, size = load_structure(path)

    assert (num, time_ps, size) == (42, 12.5, (1.0, 2.0, 3.0))
    assert len(loaded_atoms) == 7
    assert loaded_atoms[3].struct_number == 3
    assert loaded_atoms[3].atom_id == "C3"
    np.testing.assert_allclose(loaded_atoms[3].pos, [0.3, 0.6, 0.9], atol=1e-6)


def test_load_structure_reads_each_array_once_not_once_per_atom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: NpzFile.__getitem__ re-decompresses the whole array
    on every call (no caching) - indexing data[key][i] inside a per-atom
    loop is O(atom_count^2) and was observed to exhaust memory on a real
    ~66k-atom structure. Guard against that pattern coming back."""
    atom_count = 50
    path = tmp_path / "struct.npz"
    save_structure(
        path, (1, 0.0, np.array(_make_atoms(atom_count)), (1.0, 1.0, 1.0))
    )

    call_count = 0
    original_getitem = npyio.NpzFile.__getitem__

    def counting_getitem(self: npyio.NpzFile, key: str) -> object:
        nonlocal call_count
        call_count += 1
        return original_getitem(self, key)

    monkeypatch.setattr(npyio.NpzFile, "__getitem__", counting_getitem)

    load_structure(path)

    # 7 top-level keys (num, time_ps, size, struct_numbers, struct_types,
    # atom_ids, type_ids, positions), each read once - not scaled by
    # atom_count (which would be atom_count * 5 = 250 for this fixture).
    assert call_count <= 10, (
        f"load_structure() accessed the npz archive {call_count} times "
        f"for {atom_count} atoms - looks like it's re-reading an array "
        "per atom again instead of once"
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
