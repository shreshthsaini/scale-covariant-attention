import json

import pytest

from repro.merge_evaluation_caches import merge


def test_equal_duplicates_merge_but_conflicts_leave_destination_unchanged(tmp_path):
    dest, first, second = [tmp_path / name for name in ("dest.json", "first.json", "second.json")]
    dest.write_text(json.dumps({"a": {"scores": {"musiq": 45}, "seconds": 12}}))
    first.write_text(json.dumps({"a": {"scores": {"musiq": 45}, "seconds": 12}, "b": {"scores": {"musiq": 48}}}))
    assert merge(dest, [first]) == {"initial_entries": 1, "merged_entries": 2, "added_entries": 1}
    before = dest.read_bytes()
    second.write_text(json.dumps({"new": {}, "a": {"scores": {"musiq": 46}}}))
    with pytest.raises(ValueError, match="destination unchanged"):
        merge(dest, [second])
    assert dest.read_bytes() == before
