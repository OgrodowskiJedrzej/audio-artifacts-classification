import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import mlflow

from src.dataset import AudioArtifactsDataset, calculate_class_weights
from src.classifier_module import PANNBasedClassifier
from src.train import training_loop
from src.evaluate import test
from src.config import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=cfg["data"]["root"])
    parser.add_argument("--epochs", type=int, default=cfg["training"]["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=cfg["training"]["batch_size"], help="Batch size")
    parser.add_argument("--pretrained_lr", type=float, default=cfg["optimizer"]["pretrained_lr"], help="LR for frozen / pretrained backbone layers")
    parser.add_argument("--head_lr", type=float, default=cfg["optimizer"]["head_lr"], help="LR for classification head")
    parser.add_argument("--max_lr_pretrained", type=float, default=cfg["optimizer"]["max_lr_pretrained"], help="OneCycle max LR for pretrained part")
    parser.add_argument("--max_lr_head", type=float, default=cfg["optimizer"]["max_lr_head"], help="OneCycle max LR for head")
    parser.add_argument("--unfreeze_last_layers", type=int, default=cfg["model"]["unfreeze_last_layers"], choices=[0, 1, 2, 3], help="How many last PANNs conv blocks to unfreeze")
    parser.add_argument("--weight_decay", type=float, default=cfg["optimizer"]["weight_decay"], help="L2 regularization")
    parser.add_argument("--train", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    train_csv = os.path.join(args.data_path, cfg["data"]["train_csv"])
    val_csv = os.path.join(args.data_path, cfg["data"]["val_csv"])
    test_csv = os.path.join(args.data_path, cfg["data"]["test_csv"])

    train_dataset = AudioArtifactsDataset(csv_path=train_csv, data_path=args.data_path, interval=cfg["training"]["interval"])

    test_dataset = AudioArtifactsDataset(csv_path=test_csv, data_path=args.data_path, interval=cfg["training"]["interval"])

    val_dataset = AudioArtifactsDataset(csv_path=val_csv, data_path=args.data_path, interval=cfg["training"]["interval"])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow run id: {run.info.run_id}")
        print(f"MLflow version: {mlflow.__version__}")

        model = PANNBasedClassifier(num_classes=2, model_type=cfg["model"]["model_type"], unfreeze_last_layers=args.unfreeze_last_layers)
        device = model.device
        class_weights = calculate_class_weights(train_dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        print(f"Device: {device}")
        if args.train:
            print("Starting training...")
            model = training_loop(
                train_dataloader,
                val_dataloader,
                epochs=args.epochs,
                pretrained_lr=args.pretrained_lr,
                head_lr=args.head_lr,
                model=model,
                criterion=criterion,
                max_lr_pt=args.max_lr_pretrained,
                max_lr_hd=args.max_lr_head,
                weight_decay=args.weight_decay,
            )
        print("\nStarting testing...")
        test(test_dataloader, model, criterion, device)

        try:
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/model.pth")
        except Exception as e:
            print(f"Could not save file: {e}")

        try:
            mlflow.pytorch.log_model(model, name=f"{cfg["model"]["model_type"]}_trained")
        except Exception as e:
            print(f"Could not log model: {e}")


if __name__ == "__main__":
    main()
