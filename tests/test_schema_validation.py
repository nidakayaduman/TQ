from src.schema import validate_schema


def test_schema_validation_accepts_existing_sample(sample_df):
    result = validate_schema(sample_df)
    assert result.valid
    assert not result.missing_columns
    assert result.row_count > 0

