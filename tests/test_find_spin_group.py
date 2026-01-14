import pytest
from pathlib import Path
from findspingroup import find_spin_group
from findspingroup.version import __version__
import shutil
import datetime

time = datetime.datetime.now().strftime("%y%m%d%H%M%S")


cif_dir = Path("../examples/skipped_v0.11.0")
cif_files = sorted(cif_dir.glob("*.mcif"))

# log
base_dir = Path(f"fsg_test_log_{time}_v{__version__}")
failed_dir = base_dir / "failed"
skipped_dir = base_dir / "skipped"
log_file = base_dir / "test.log"
id_file = base_dir / "id_index.txt"


def write_id_index(id_msg: str):
    with open(id_file, "a", encoding="utf-8") as f:
        f.write(id_msg + "\n")


def write_log(msg: str):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)




def test_find_spin_group_specific():
    specific_file = Path("../examples/0.800_MnTe.mcif")
    result = find_spin_group.find_spin_groups(str(specific_file))
    print(result)






@pytest.mark.parametrize("cif_path", cif_files, ids=lambda p: p.name)
def test_find_spin_group_index_batch(cif_path: Path):
    for d in (failed_dir, skipped_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        result = find_spin_group.find_spin_groups(str(cif_path))
    except Exception as e:
        dst = skipped_dir / cif_path.name
        shutil.copy2(cif_path, dst)
        msg = f"[SKIP] {cif_path.name}: {type(e).__name__}: {e}"
        write_log(msg)
        pytest.skip(msg)
    else:
        msg = (
            f"[PASS]{cif_path.name}\t{result['index']}\t{result['spin_part_pg']}"
            f"\t{result['conf']}\t{result['acc']}"
        )
        write_log(msg)
        write_id_index(result["id_index_info"])


@pytest.mark.parametrize("cif_path", cif_files, ids=lambda p: p.name)
def test_error_batch(cif_path: Path):
    """
    only test for errors
    """
    try:
        result = find_spin_group.find_spin_groups(str(cif_path))
        assert result is not None
    except Exception as e:
        pytest.fail(f"[ERROR] {cif_path.name}\n{type(e).__name__}: {e}")
