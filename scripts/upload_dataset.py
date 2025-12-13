from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from src.config import AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME, AZURE_COMPUTER_TARGET

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=AZURE_SUBSCRIPTION_ID,
    resource_group_name=AZURE_RESOURCE_GROUP,
    workspace_name=AZURE_WORKSPACE_NAME
)

dataset = Data(
    name="audio_artifacts_dataset",
    path="./dataset",
    type=AssetTypes.URI_FOLDER,
    description="Wav files + CSV for train, validation and test set.",
    version="V2_corrected_paths"
)

ml_client.data.create_or_update(dataset)
