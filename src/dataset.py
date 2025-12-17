import torch
from torch import Tensor
from torch.utils.data import Dataset
import os

import pandas as pd
import random

from src.config import cfg
from src.utils import load_and_resample

class AudioArtifactsDataset(Dataset):
    def __init__(self, csv_path: str, data_path: str | None = None, interval: int = 1):
        self.dataset_file = pd.read_csv(csv_path, sep=";")
        self.data_path = data_path
        self.interval = interval
        self.dataset_file["class"] = self.dataset_file["class"].map(lambda x: 0 if x == "no_artifact" else 1)

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, idx):
        relative_path = self.dataset_file.iloc[idx]["path"]
        full_path = os.path.join(self.data_path, relative_path)
        target = self.dataset_file.iloc[idx]["class"]
        waveform = load_and_resample(full_path, cfg["data"]["sample_rate"])
        target_length = cfg["data"]["sample_rate"] * self.interval
        if waveform.shape[-1] >= target_length:
            max_start = waveform.shape[-1] - target_length
            start = random.randint(0, max_start) if max_start > 0 else 0
            waveform = waveform[start:start + target_length]
        else:
            pad = target_length - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform, target  # 1D waveform of length sample_rate * interval


def calculate_class_weights(train_dataset: AudioArtifactsDataset, device) -> Tensor:
    """Calculates weights of every class to minimalize unbalanced dataset effects."""
    with torch.no_grad():
        train_labels = torch.tensor([train_dataset.dataset_file.iloc[i]["class"] for i in range(len(train_dataset))], dtype=torch.long)
        counts = torch.bincount(train_labels, minlength=2).to(torch.float32)
        total = counts.sum()
        class_weights = total / (len(counts) * counts)
        class_weights = class_weights * (len(counts) / class_weights.sum())

    return class_weights.to(device)
