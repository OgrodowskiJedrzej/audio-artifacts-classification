import pytest
import torch

from src.classifier_module import PANNBasedClassifier

@pytest.mark.model
@pytest.mark.parametrize("model_type", ["wavegram_logmel", "resnet"])
def test_real_weights_loading(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PANNBasedClassifier(
        model_type=model_type,
        num_classes=2,
        freeze_panns=True,
        unfreeze_last_layers=0,
        device=device
    )

    any_param = next(model.panns_model.parameters())

    error_msg = "PANN model parameters should not be None"

    assert any_param is not None, error_msg

@pytest.mark.model
@pytest.mark.parametrize("model_type", ["wavegram_logmel", "resnet"])
def test_freeze_unfreeze_layers(model_type):
    """
    Test that layers are frozen/unfrozen correctly based on unfreeze_last_layers argument.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PANNBasedClassifier(
        model_type=model_type,
        num_classes=2,
        freeze_panns=True,
        unfreeze_last_layers=1,
        device=device
    )

    trainable_found = False
    frozen_found = False

    for _, param in model.panns_model.named_parameters():
        if param.requires_grad:
            trainable_found = True
        else:
            frozen_found = True

    assert trainable_found, "No layers were unfrozen, but unfreeze_last_layers > 0"
    assert frozen_found, "All layers are unfrozen, freeze_panns=True but some should be frozen"

@pytest.mark.model
@pytest.mark.parametrize("model_type", ["wavegram_logmel", "resnet"])
def test_forward_pass(model_type):
    """
    Test that forward pass works and output has correct shape.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PANNBasedClassifier(
        model_type=model_type,
        num_classes=2,
        freeze_panns=True,
        unfreeze_last_layers=0,
        device=device
    )

    model.eval()
    dummy_input = torch.randn(2, 32000, device=device)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 2), f"Expected output shape (2,2), got {output.shape}"
