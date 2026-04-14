"""JSON Schema validation for results.json."""
from __future__ import annotations

import pytest

from hermes_sci.results import (
    ResultsSchemaError,
    Table,
    from_dict,
    validate,
)


def _minimal():
    return {"metrics": [{"name": "BLEU", "value": 28.3}]}


def test_minimal_valid():
    validate(_minimal())  # must not raise


def test_full_valid():
    doc = {
        "setup": {
            "hardware": "M2 MPS",
            "hyperparams": {"lr": 1e-4, "epochs": 20, "use_amp": True},
        },
        "metrics": [
            {"name": "BLEU", "value": 28.3, "unit": "",
             "method": "HAMR", "split": "test"},
        ],
        "tables": [
            {"id": "complexity", "caption": "Buckets",
             "headers": ["C", "L"], "rows": [["S", "19.4"]],
             "owning_section": "experiments"},
        ],
        "raw_log": "stdout",
    }
    validate(doc)


def test_metrics_required():
    with pytest.raises(ResultsSchemaError) as ei:
        validate({"setup": {"hardware": "M2"}})
    assert "metrics" in str(ei.value)


def test_metric_missing_value():
    with pytest.raises(ResultsSchemaError) as ei:
        validate({"metrics": [{"name": "BLEU"}]})
    assert "value" in ei.value.message


def test_bad_table_id_rejected():
    with pytest.raises(ResultsSchemaError) as ei:
        validate({
            "metrics": [],
            "tables": [{"id": "bad id", "headers": ["x"], "rows": []}],
        })
    assert "tables" in ei.value.path and "id" in ei.value.path


def test_invalid_owning_section_rejected():
    with pytest.raises(ResultsSchemaError):
        validate({
            "metrics": [],
            "tables": [{"id": "t", "headers": ["x"], "rows": [],
                        "owning_section": "not_a_real_section"}],
        })


def test_additional_top_level_rejected():
    with pytest.raises(ResultsSchemaError):
        validate({"metrics": [], "extraneous": 1})


def test_from_dict_roundtrip_owning_section():
    doc = {
        "metrics": [],
        "tables": [{"id": "t1", "caption": "c", "headers": ["a"], "rows": [],
                    "owning_section": "method"}],
    }
    r = from_dict(doc)
    assert r.tables[0].owning_section == "method"
    assert isinstance(r.tables[0], Table)


def test_from_dict_default_owning_section_is_empty_string():
    r = from_dict({"metrics": [], "tables": [
        {"id": "t", "caption": "", "headers": ["x"], "rows": []}]})
    assert r.tables[0].owning_section == ""
