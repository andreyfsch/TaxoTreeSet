"""A single-head regeneration must not erase the other heads' manifest entries."""
import json
from pathlib import Path
from types import SimpleNamespace

from taxotreeset.core._orchestration._manifest import _persist_scheduling_artifacts


def _ctx(tmp_path):
    return SimpleNamespace(output_dir=str(tmp_path))


def test_single_head_rerun_preserves_other_entries(tmp_path):
    # A full run records three heads.
    _persist_scheduling_artifacts(_ctx(tmp_path), "viruses", {
        "master_manifest": {"1": {"num_leaves": 3}, "2": {"num_leaves": 5},
                            "3": {"num_leaves": 7}},
        "passthrough_map": {"9": "1"},
        "virtual_id_registry": {},
    })
    # Then one head is regenerated on its own -- the path that silently wiped the
    # 60-head pilot's metadata on 2026-08-03.
    _persist_scheduling_artifacts(_ctx(tmp_path), "viruses", {
        "master_manifest": {"2": {"num_leaves": 99}},
        "passthrough_map": {},
        "virtual_id_registry": {},
    })
    m = json.loads((tmp_path / "manifest_viruses.json").read_text())
    assert set(m) == {"1", "2", "3"}       # siblings survive
    assert m["2"]["num_leaves"] == 99      # the rerun's entry wins
    assert m["1"]["num_leaves"] == 3
    p = json.loads((tmp_path / "passthroughs_viruses.json").read_text())
    assert p == {"9": "1"}                 # passthroughs likewise preserved
