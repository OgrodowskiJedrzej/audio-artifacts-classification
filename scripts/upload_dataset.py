from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from src.config import cfg

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=cfg.subscription_id,
    resource_group_name=cfg.resource_group_name,
    workspace_name=cfg.workspace_name
)

dataset = Data(
    name="audio_artifacts_dataset",
    path="./dataset",
    type=AssetTypes.URI_FOLDER,
    description="Wav files + CSV for train, validation and test set.",
    version="V2_corrected_paths"
)

ml_client.data.create_or_update(dataset)
