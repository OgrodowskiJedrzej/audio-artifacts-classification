import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import mlflow

from dataset import AudioArtifactsDataset, calculate_class_weights
from classifier_module import PANNBasedClassifier
from train import training_loop
from evaluate import test

default_params = {
            "epochs": 1,
            "batch_size": 1,
            "unfreeze_last_layers": 3,
            "high_pass_filter": 200,
            "pretrained_lr": 1e-4,
            "head_lr": 5e-3,
            "max_lr_pretrained": 5e-4,
            "max_lr_head": 8e-3,
            "weight_decay": 0.008,
            "interval": 5
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")

    parser.add_argument("--epochs", type=int, default=default_params["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=default_params["batch_size"], help="Batch size")
    parser.add_argument("--pretrained_lr", type=float, default=default_params["pretrained_lr"], help="LR for frozen / pretrained backbone layers")
    parser.add_argument("--head_lr", type=float, default=default_params["head_lr"], help="LR for classification head")
    parser.add_argument("--max_lr_pretrained", type=float, default=default_params["max_lr_pretrained"], help="OneCycle max LR for pretrained part")
    parser.add_argument("--max_lr_head", type=float, default=default_params["max_lr_head"], help="OneCycle max LR for head")
    parser.add_argument("--unfreeze_last_layers", type=int, default=default_params["unfreeze_last_layers"], choices=[0,1,2,3], help="How many last PANNs conv blocks to unfreeze")
    parser.add_argument("--high_pass_filter", type=int, default=default_params["high_pass_filter"], help="High-pass filter cutoff (currently not applied if code doesn't use it)")
    parser.add_argument("--weight_decay", type=float, default=default_params["weight_decay"], help="L2 regularization")

    return parser.parse_args()

def main():
    args = parse_args()
    default_params["epochs"] = args.epochs
    default_params["batch_size"] = args.batch_size
    default_params["pretrained_lr"] = args.pretrained_lr
    default_params["head_lr"] = args.head_lr
    default_params["max_lr_pretrained"] = args.max_lr_pretrained
    default_params["max_lr_head"] = args.max_lr_head
    default_params["unfreeze_last_layers"] = args.unfreeze_last_layers
    default_params["high_pass_filter"] = args.high_pass_filter
    default_params["weight_decay"] = args.weight_decay

    train_csv = os.path.join(args.data_path, "train_5s_aug.csv")
    test_csv = os.path.join(args.data_path, "test_5s.csv")
    val_csv = os.path.join(args.data_path, "val_5s.csv")

    train_dataset = AudioArtifactsDataset(
        csv_path=train_csv,
        data_path=args.data_path,
        interval=default_params["interval"]
    )

    test_dataset = AudioArtifactsDataset(
        csv_path=test_csv,
        data_path=args.data_path,
        interval=default_params["interval"]
    )

    val_dataset = AudioArtifactsDataset(
        csv_path=val_csv,
        data_path=args.data_path,
        interval=default_params["interval"]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=default_params["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=default_params["batch_size"], shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=default_params["batch_size"], shuffle=False)

    with mlflow.start_run() as run:
        mlflow.log_params(default_params)
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow run id: {run.info.run_id}")
        print(f"MLflow version: {mlflow.__version__}")

        model = PANNBasedClassifier(num_classes=2, model_type="wavegram_logmel", unfreeze_last_layers=default_params["unfreeze_last_layers"])
        device = model.device
        class_weights = calculate_class_weights(train_dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        print(f"Device: {device}")
        print("Starting training...")
        model = training_loop(
            train_dataloader,
            val_dataloader,
            epochs=default_params["epochs"],
            pretrained_lr=default_params["pretrained_lr"],
            head_lr=default_params["head_lr"],
            model=model,
            criterion=criterion,
            max_lr_pt=default_params["max_lr_pretrained"],
            max_lr_hd=default_params["max_lr_head"],
            weight_decay=default_params["weight_decay"]
        )
        print("\nStarting testing...")
        test(test_dataloader, model, criterion, device)

        try:
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/model.pth")
        except Exception as e:
            print(f"Could not save file: {e}")

        try:
            mlflow.pytorch.log_model(model, name="wavegram_logmel_trained")
        except Exception as e:
            print(f"Could not log model: {e}")

if __name__ == "__main__":
    main()
