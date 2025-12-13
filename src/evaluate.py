import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import mlflow

def test(test_dataloader: DataLoader, model, criterion, device):
    model.eval()
    total_test_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for waveform, label in test_dataloader:
            waveform = waveform.to(device)
            label = label.to(device)

            outputs = model(waveform)
            loss = criterion(outputs, label)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_dataloader)

    accuracy = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["no_artefact", "artefact"], zero_division=0))
    print("=" * 50)

    metrics = {
        "test_loss": avg_test_loss,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1
    }
    mlflow.log_metrics(metrics)

    return metrics
