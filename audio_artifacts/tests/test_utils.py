import pytest

import torch
import torchaudio

from src.utils import split_wav_into_chunks, convert_stereo_to_mono, load_and_resample


@pytest.fixture
def stereo_waveform():
    return torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])


@pytest.fixture
def tmp_wav_file(tmp_path):
    waveform = torch.randn(1, 48000)
    wav_path = tmp_path / "test.wav"
    torchaudio.save(str(wav_path), waveform, sample_rate=48000)
    return str(wav_path)


@pytest.mark.utils
def test_load_and_resample(tmp_wav_file):
    result = load_and_resample(tmp_wav_file, sample_rate=32000)

    assert isinstance(result, torch.Tensor), "Incorrect data type, should be Tensor."
    assert result.ndim == 1, "Incorrect dimension."
    assert result.shape[0] == 32000, "Incorrect shape."


@pytest.mark.utils
def test_convert_stereo_to_mono(stereo_waveform):
    mono = convert_stereo_to_mono(stereo_waveform)
    expected = torch.tensor([2.0, 2.0, 2.0])

    error_msg = "Incorrect conversion from stero to mono."

    assert torch.allclose(mono, expected), error_msg


@pytest.mark.utils
def test_split_wav_into_chunks():
    sr = 32000
    waveform = torch.randn(sr * 3)
    chunks = split_wav_into_chunks(waveform, sr, interval=5.0, overlapping_ratio=0.25)

    error_msg = "Length of chunk incorrect."

    for chunk in chunks:
        assert chunk.shape[0] == sr, error_msg
