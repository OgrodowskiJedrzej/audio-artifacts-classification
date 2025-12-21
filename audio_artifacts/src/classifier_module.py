from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from src.models.wavegram import Wavegram_Logmel_Cnn14
from src.models.resnet import ResNet54
from src.config import cfg


class PANNBasedClassifier(nn.Module):
    def __init__(
        self,
        model_type: Literal["wavegram_logmel", "resnet"] = "wavegram_logmel",
        num_classes: int = 2,
        freeze_panns: bool = True,
        unfreeze_last_layers: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.panns_model = None

        if model_type == "wavegram_logmel":
            panns_weights_path = cfg["weights"]["wavegram_logmel_weights"]
            self.panns_model = Wavegram_Logmel_Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=200,
                fmax=14000,
                classes_num=527,
            )

        if model_type == "resnet":
            panns_weights_path = cfg["weights"]["resnet_weights"]
            self.panns_model = ResNet54(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=200,
                fmax=8000,
                classes_num=527,
            )

        try:
            checkpoint = torch.load(panns_weights_path, map_location="cpu")
            self.panns_model.load_state_dict(checkpoint["model"])
        except Exception as e:
            print(f"Error loading weights: {e}")

        self.panns_model.to(self.device)

        if freeze_panns:
            for param in self.panns_model.parameters():
                param.requires_grad = False

        layers_to_unfreeze = []
        if unfreeze_last_layers > 0:
            if model_type == "wavegram_logmel":
                if unfreeze_last_layers >= 1:
                    layers_to_unfreeze.append("conv_block6")
                if unfreeze_last_layers >= 2:
                    layers_to_unfreeze.append("conv_block5")
                if unfreeze_last_layers >= 3:
                    layers_to_unfreeze.append("conv_block4")

            elif model_type == "resnet":
                if unfreeze_last_layers >= 1:
                    layers_to_unfreeze.append("resnet.layer4")
                if unfreeze_last_layers >= 2:
                    layers_to_unfreeze.append("resnet.layer3")
                if unfreeze_last_layers >= 3:
                    layers_to_unfreeze.append("resnet.layer2")

        for name, param in self.panns_model.named_parameters():
            for layer in layers_to_unfreeze:
                if layer in name:
                    param.requires_grad = True

        panns_embedding_size = self.panns_model.fc_audioset.in_features

        self.fc1 = nn.Linear(panns_embedding_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        self.fc1.to(self.device)
        self.dropout1.to(self.device)
        self.fc2.to(self.device)

    def forward(self, waveform):
        waveform = waveform.to(self.device)

        output = self.panns_model(waveform)
        embedding = output["embedding"]

        x = F.relu(self.fc1(embedding))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x
