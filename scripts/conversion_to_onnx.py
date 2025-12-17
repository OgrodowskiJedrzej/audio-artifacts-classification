import torch

from src.utils import load_model

model = load_model(
    "models/wavegram_logmel.pth", model_type="wavegram_logmel")
example_input = (torch.randn(1,32000*5),)
onnx_model = torch.onnx.export(model, example_input, dynamo=True)
onnx_model.save("models/wavegram_logmel.onnx")