import pydantic
import wandb
from typing import List, Optional


class WandbRunConfig(pydantic.BaseModel):
    project: str
    run_name: Optional[str] = None
    entity: Optional[str] = None
    group_name: Optional[str] = None
    tags: Optional[List[str]] = None
    log_dir: Optional[str] = None

    # Check that there is a valid wandb entity
    @pydantic.field_validator("entity")
    def valid_entity(cls, entity: Optional[str]) -> str:
        return entity if entity is not None else wandb.Api().default_entity


class WandbCheckpointConfig(WandbRunConfig):

    checkpoint_version: str = "latest"

    @property
    def artifact_name(self) -> str:
        return f"{self.entity}/{self.project}/model-{self.checkpoint_version}"

    
class WandbRunCheckpointConfig(WandbCheckpointConfig):

    run_id: str
    @property
    def full_run_name(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"

    @property
    def artifact_suffix(self) -> str:
        return f"model-{self.run_id}:{self.checkpoint_version}"

    @property
    def artifact_name(self) -> str:
        return f"{self.entity}/{self.project}/{self.artifact_suffix}"


class TrainConfig(pydantic.BaseModel):
    wandb_run_config: Optional[WandbRunConfig] = None
    batch_size: int = 64
    max_epoch_number: int = 10  # Fixed typo here
    warmup_epochs: int = 10
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    seed: int = 42
    gpu_num: int = 0
    log_steps: int = 10
    num_workers: int = 16
    checkpoint_every_n_epochs: int = 5
    precision: int = 32
    profiler: str = "simple"
