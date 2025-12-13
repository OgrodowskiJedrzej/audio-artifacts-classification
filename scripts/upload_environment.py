
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

from src.config import cfg

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=cfg.subscription_id,
    resource_group_name=cfg.resource_group_name,
    workspace_name=cfg.workspace_name
)

env = Environment(
    name="pytorch-audio-env",
    conda_file="./environment.yml",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:42",
    description="Custom PyTorch audio env with PANNs & MLflow."
)

env = ml_client.environments.create_or_update(env)
