import pytest
import torch
import pandas as pd
from unittest.mock import patch

from src.dataset import AudioArtifactsDataset, calculate_class_weights


@pytest.fixture
def tmp_csv(tmp_path):
    df = pd.DataFrame({"path": ["a.wav", "b.wav", "c.wav"], "class": ["no_artifact", "artifact", "artifact"]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, sep=";", index=False)
    return csv_path


@pytest.fixture
def mock_load_and_resample():
    with patch("src.dataset.load_and_resample") as mock_load:
        mock_load.return_value = torch.randn(32000)  # exactly 1 second @ 32kHz
        yield mock_load


@pytest.mark.dataset
def test_dataset_length(tmp_csv, mock_load_and_resample):
    ds = AudioArtifactsDataset(str(tmp_csv), data_path="/dummy", interval=1)

    error_msg = "Incorrect dataset length."

    assert len(ds) == 3, error_msg


@pytest.mark.dataset
def test_label_encoding(tmp_csv, mock_load_and_resample):
    ds = AudioArtifactsDataset(str(tmp_csv), data_path="/dummy")
    _, y0 = ds[0]
    _, y1 = ds[1]
    _, y2 = ds[2]

    error_msg = "Incorrect label encoding."

    assert y0 == 0, error_msg  # "no_artifact"
    assert y1 == 1, error_msg  # "artifact"
    assert y2 == 1, error_msg


@pytest.mark.dataset
def test_padding(tmp_csv):
    # Return waveform shorter than needed
    with patch("src.dataset.load_and_resample") as mock_load:
        mock_load.return_value = torch.randn(10000)

        ds = AudioArtifactsDataset(str(tmp_csv), data_path="/dummy", interval=1)
        waveform, _ = ds[0]

        error_msg = f"Incorrect length of padded track, should be 32000 but is {waveform.shape[0]}."

        assert waveform.shape[0] == 32000, error_msg


@pytest.mark.dataset
def test_truncation(tmp_csv):
    with patch("src.dataset.load_and_resample") as mock_load:
        mock_load.return_value = torch.randn(64000)  # 2 seconds

        ds = AudioArtifactsDataset(str(tmp_csv), data_path="/dummy", interval=1)
        waveform, _ = ds[0]

        error_msg = f"Incorrect length of truncated track, should be 32000 but is {waveform.shape[0]}."

        assert waveform.shape[0] == 32000, error_msg


@pytest.mark.dataset
def test_class_weights(tmp_csv, mock_load_and_resample):
    ds = AudioArtifactsDataset(str(tmp_csv), data_path="/dummy")
    device = torch.device("cpu")

    weights = calculate_class_weights(ds, device=device)

    assert isinstance(weights, torch.Tensor), "Incorrect data type of weights."
    assert weights.shape == (2,), "Incorrect shape of weights."
    assert torch.all(weights > 0), "Negative weights."
    assert weights.device == device, "Wrong device."
