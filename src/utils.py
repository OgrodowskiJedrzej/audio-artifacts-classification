import os
from typing import Literal

import torch
from torch import Tensor
import torchaudio
import matplotlib.pyplot as plt

from classifier_module import PANNBasedClassifier

def convert_stereo_to_mono(waveform: Tensor) -> Tensor:
    """Conversion from stereo to mono via averaging channels."""
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    else:
        waveform = waveform.squeeze(0)

    return waveform

def load_and_resample(path: str, sample_rate: int = 32000) -> Tensor:
    """Loads wav file, converts to mono and resamples to given sample rate."""
    waveform, sr = torchaudio.load(path)
    waveform = convert_stereo_to_mono(waveform)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform

def split_wav_into_chunks(waveform: Tensor, sample_rate: int, interval: float, overlapping_ratio: float) -> list:
    """Splits wav file into chunks of given length overlapping by given ratio."""
    segment_len = sample_rate * interval
    # Overlap ratio (e.g. 0.25 means each new chunk starts 75% into previous -> 25% overlap kept)
    overlap_ratio = overlapping_ratio
    hop_len = int(segment_len * (1 - overlap_ratio))
    if hop_len <= 0:
        hop_len = segment_len

    chunks = []
    for start in range(0, max(0, waveform.shape[0] - segment_len + 1), hop_len):
        chunk = waveform[start:start + segment_len]
        if chunk.shape[0] < segment_len:
            chunk = torch.nn.functional.pad(chunk, (0, segment_len - chunk.shape[0]))
        chunks.append(chunk)

    return chunks

def load_model(model_path: str, model_type=Literal["wavegram_logmel", "resnet"]) -> PANNBasedClassifier:
    """Loads model and sets up for evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PANNBasedClassifier(num_classes=2, model_type=model_type)
    print("Model object created")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def generate_spectrogram(path: str) -> str | None:
    """Generate spectrograme of given wav file."""
    output_dir = "spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    try:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.numpy()

        n_channels, _ = waveform.shape

        NFFT = 1024
        noverlap = 512

        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 10 * n_channels), squeeze=False)

        for c in range(n_channels):
            ax = axes[c, 0]
            ax.specgram(waveform[c], Fs=sample_rate, NFFT=NFFT, noverlap=noverlap)
            ax.set_ylabel(f"Channel {c+1} Frequency [Hz]")
            ax.set_xlabel("Time [s]")
            ax.set_ylim(0, 14000)

        fig.suptitle(f"Spectrogram for {os.path.basename(path)}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        base_filename = os.path.splitext(os.path.basename(path))[0]
        output_filename = os.path.join(output_dir, f"{base_filename}_spectrogram.png")
        plt.savefig(output_filename, dpi=300)
        plt.close(fig)
        print(f"Spectrogram saved to {output_filename}")
        return output_filename

    except Exception as e:
        print(f"Could not generate spectrogram for {path}: {e}")
        if "fig" in locals():
            plt.close(fig)
        return None
