from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Config(BaseModel):
    sample_rate: int = 32000
    wavegram_logmel_weights_path: str = "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
    resnet_weights_path: str = "ResNet54_mAP=0.429.pth"

    # compute_target: str = Field(default_factory=lambda: os.environ["CLUSTER_NAME"])
    # subscription_id: str = Field(default_factory=lambda: os.environ["SUBSCRIPTION_ID"])
    # resource_group_name: str = Field(default_factory=lambda: os.environ["RESOURCE_GROUP_NAME"])
    # workspace_name: str = Field(default_factory=lambda: os.environ["WORKSPACE_NAME"])

cfg = Config()
