
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, command

from src.config import cfg

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=cfg.subscription_id,
    resource_group_name=cfg.resource_group_name,
    workspace_name=cfg.workspace_name
)

job = command(
    code="./src/",
    command="python main.py --data_path ${{inputs.data}} ",
    environment="pytorch-audio-env@latest",
    compute=cfg.compute_target,
    display_name="cluster-gpu-training-test",
    experiment_name="cluster-training-wavegram_5s",
    instance_count=1,
    inputs={
        "data": Input(type="uri_folder", path="azureml:audio_artifacts_dataset:V2_corrected_paths")
    }
)

returned_job = ml_client.jobs.create_or_update(job)
print("Job submitted:", returned_job.name)
print("Portal: https://ml.azure.com/experiments/{}/runs/{}".format(job.experiment_name, returned_job.name))
