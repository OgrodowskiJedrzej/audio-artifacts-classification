from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.sweep import Choice, Uniform
from azure.ai.ml.sweep import MedianStoppingPolicy

from src.config import AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME, AZURE_COMPUTER_TARGET

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=AZURE_SUBSCRIPTION_ID,
    resource_group_name=AZURE_RESOURCE_GROUP,
    workspace_name=AZURE_WORKSPACE_NAME
)

base_command = (
    "python main.py "
    "--data_path ${{inputs.data}} "
    "--pretrained_lr ${{inputs.pretrained_lr}} "
    "--head_lr ${{inputs.head_lr}} "
    "--unfreeze_last_layers ${{inputs.unfreeze_last_layers}} "
    "--weight_decay ${{inputs.weight_decay}} "
    "--batch_size ${{inputs.batch_size}} "
)

command_job = command(
    code="./src/",
    command=base_command,
    environment="pytorch-audio-env@latest",
    compute=AZURE_COMPUTER_TARGET,
    inputs={
        "data": Input(type="uri_folder", path="azureml:audio_dataset:1"),
        "pretrained_lr": 1e-4,
        "head_lr": 5e-3,
        "unfreeze_last_layers": 3,
        "weight_decay": 0.008,
        "batch_size": 256
    },
)

command_job_for_sweep = command_job(
    pretrained_lr=Choice(values=[5e-5, 1e-4, 2e-4, 1e-5]),
    head_lr=Uniform(max_value=1e-3, min_value=8e-3),
    weight_decay=Choice(values=[1e-2, 1e-3, 8e-3]),
)

sweep_job = command_job_for_sweep.sweep(
    compute=AZURE_COMPUTER_TARGET,
    sampling_algorithm="bayesian",
    primary_metric="test_f1_score",
    goal="Maximize",
)

sweep_job.display_name = "hpo-audio-artifacts"
sweep_job.experiment_name = "hpo-audio-artifacts"
sweep_job.description = "Hyperparameter sweep for audio classification training."

sweep_job.set_limits(max_total_trials=10, max_concurrent_trials=2)
sweep_job.early_termination = MedianStoppingPolicy(
    delay_evaluation=2, evaluation_interval=1
)

returned_job = ml_client.jobs.create_or_update(sweep_job)

print("Sweep submitted:", returned_job.name)
print(
    f"Portal: https://ml.azure.com/experiments/{sweep_job.experiment_name}/runs/{returned_job.name}"
)
