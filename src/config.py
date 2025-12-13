import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path="config.yml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)

AZURE_SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("RESOURCE_GROUP_NAME")
AZURE_WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
AZURE_COMPUTER_TARGET = os.getenv("COMPUTE_TARGET")

cfg = load_config()
