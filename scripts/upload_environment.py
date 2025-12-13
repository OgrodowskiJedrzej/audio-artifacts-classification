
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

from src.config import AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME, AZURE_COMPUTER_TARGET

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=AZURE_SUBSCRIPTION_ID,
    resource_group_name=AZURE_RESOURCE_GROUP,
    workspace_name=AZURE_WORKSPACE_NAME
)

env = Environment(
    name="pytorch-audio-env",
    conda_file="./environment.yml",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:42",
    description="Custom PyTorch audio env with PANNs & MLflow."
)

env = ml_client.environments.create_or_update(env)
