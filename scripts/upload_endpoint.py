
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineDeployment, CodeConfiguration, ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential

from src.config import cfg

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=cfg.subscription_id,
    resource_group_name=cfg.resource_group_name,
    workspace_name=cfg.workspace_name
)

endpoint = ManagedOnlineEndpoint(
    name = "test-endpoint",
    description="this is a sample endpoint",
    auth_mode="key"
)

model = Model(path="./models/wavegram_logmel.pth")
deployment = ManagedOnlineDeployment(
    name="Wavegram_logmel_deployment",
    endpoint_name=endpoint.name,
    model=model,
    environment="pytorch-audio-env:4",
    code_configuration=CodeConfiguration(
        code="./src/", scoring_script="score.py"
    ),
    instance_count=1,
    instance_type="Standard_DS2_v2"
)

res_endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)
res_deployment = ml_client.online_deployments.begin_create_or_update(
    deployment=deployment, local=True
)

print(res_endpoint)
print(res_deployment)
