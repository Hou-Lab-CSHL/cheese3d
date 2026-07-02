import pandas as pd
import pytest

from cheese3d.backends.lightning_pose import read_lightning_pose_predictions

@pytest.mark.unit
def test_read_lightning_pose_predictions_filters_metadata_columns(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    columns = pd.MultiIndex.from_tuples([
        ("set", "", ""),
        ("model", "nose", "x"),
        ("model", "nose", "y"),
        ("model", "nose", "likelihood"),
        ("model", "nose", "z"),
    ])
    df = pd.DataFrame([["test", 1.0, 2.0, 0.9, 3.0]], columns=columns)
    df.to_csv(csv_path)
    parsed = read_lightning_pose_predictions(csv_path, scorer="lp")
    assert list(parsed.columns) == [
        ("lp", "nose", "x"),
        ("lp", "nose", "y"),
        ("lp", "nose", "likelihood"),
    ]
    assert parsed.iloc[0].tolist() == [1.0, 2.0, 0.9]
