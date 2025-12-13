import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
import mlflow

from classifier_module import PANNBasedClassifier

def training_loop(
    train_dataloader: DataLoader,
    validate_dataloader: DataLoader,
    epochs: int,
    *,
    pretrained_lr: float,
    head_lr: float,
    max_lr_pt: float = 5e-4,
    max_lr_hd: float = 8e-3,
    weight_decay: float,
    model: PANNBasedClassifier,
    criterion
):
    model.train()
    device = model.device

    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters() if "panns_model" in n and p.requires_grad], "lr": pretrained_lr, "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if "panns_model" not in n], "lr": head_lr, "weight_decay": weight_decay}
    ])

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=[max_lr_pt, max_lr_hd],
        steps_per_epoch=len(train_dataloader), epochs=epochs)

    for epoch in range(epochs):
        total_loss = 0
        correct_train = 0
        total_train = 0
        for batch_idx, (waveform, label) in enumerate(train_dataloader):
            waveform = waveform.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(waveform)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == label).sum().item()
                total_train += label.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0.0
        print(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%")

        val_loss, val_accuracy = validate(model, validate_dataloader, criterion, device)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")
        print("-" * 50)
        mlflow.log_metrics(
            {
                "avg_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }
        )

    return model

def validate(model, validate_dataloader: DataLoader, criterion, device):
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveform, label in validate_dataloader:
            waveform = waveform.to(device)
            label = label.to(device)

            outputs = model(waveform)
            loss = criterion(outputs, label)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_val_loss = total_val_loss / len(validate_dataloader)
    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    model.train()
    return avg_val_loss, val_accuracy
