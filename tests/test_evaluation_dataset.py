import json
from pathlib import Path


DATASET_PATH = Path(__file__).parents[1] / "evaluation" / "dataset.json"
REQUIRED_RECORD_FIELDS = {"id", "language", "text", "reference_summary", "entities"}
REQUIRED_ENTITY_FIELDS = {"text", "type"}
VALID_LANGUAGES = {"en", "hi", "mr"}
VALID_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "DATE"}


def test_evaluation_dataset_has_required_fields_and_expected_coverage():
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)

    assert len(records) == 15
    assert {record["language"] for record in records} == VALID_LANGUAGES
    assert len({record["id"] for record in records}) == len(records)

    for record in records:
        assert REQUIRED_RECORD_FIELDS <= record.keys()
        assert record["language"] in VALID_LANGUAGES
        assert record["text"]
        assert record["reference_summary"]
        assert record["entities"]
        for entity in record["entities"]:
            assert REQUIRED_ENTITY_FIELDS <= entity.keys()
            assert entity["text"] in record["text"]
            assert entity["type"] in VALID_ENTITY_TYPES
