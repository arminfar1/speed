from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from pathlib import Path
import yaml
from datetime import datetime


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    bucket_path: str
    file_pattern: str
    ship_methods: List[str]
    outlier_quantiles: Tuple[float, float]
    validation_split: float = 0.2


@dataclass
class HyperParameters:
    """Hyperparameters for different models."""
    GBM: Dict[str, Any]
    XGB: Dict[str, Any]


@dataclass
class ModelConfig:
    """Main model configuration."""
    name: str
    version: str
    target_quantiles: List[float]
    label: str
    features: Dict[str, List[str]]
    test_size: float
    random_state: int
    time_limit: int
    num_cpus: int
    hyperparameters: HyperParameters

    @property
    def get_all_features(self) -> List[str]:
        """Combine all feature categories into a flat list."""
        all_features = []
        for feature_list in self.features.values():
            all_features.extend(feature_list)
        return all_features


@dataclass
class TrainConfig:
    """Training specific configuration."""
    random_state: int


@dataclass
class PredictConfig:
    """Prediction specific configuration."""
    random_state: int


@dataclass
class EvaluateConfig:
    """Evaluation specific configuration."""
    random_state: int


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    save_path: str = "logs/"
    file_name: str = "df-shipment-speed"
    file_extension: str = ".log"
    metrics: List[str] = field(default_factory=lambda: ["quantile_loss", "coverage_error"])


@dataclass
class PathConfig:
    """Path configuration."""
    model_save_dir: str = "models/"
    results_save_dir: str = "results/"


@dataclass
class Config:
    """Master configuration class."""
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    predict: PredictConfig
    evaluate: EvaluateConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'Config':
        """Load configuration from YAML file."""
        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            # Create nested configuration objects
            data_config = DataConfig(**config_dict['data'])
            hyperparams = HyperParameters(**config_dict['model']['hyperparameters'])

            # Remove hyperparameters from model dict and create ModelConfig
            model_dict = config_dict['model'].copy()
            del model_dict['hyperparameters']
            model_dict['hyperparameters'] = hyperparams
            model_config = ModelConfig(**model_dict)

            train_config = TrainConfig(**config_dict['train'])
            predict_config = PredictConfig(**config_dict['predict'])
            evaluate_config = EvaluateConfig(**config_dict['evaluate'])

            # Create logging and paths configs if they exist in yaml
            logging_config = LoggingConfig(**config_dict.get('logging', {}))
            paths_config = PathConfig(**config_dict.get('paths', {}))

            return cls(
                data=data_config,
                model=model_config,
                train=train_config,
                predict=predict_config,
                evaluate=evaluate_config,
                logging=logging_config,
                paths=paths_config
            )
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {str(e)}")

    def save(self, save_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'data': {k: v for k, v in self.data.__dict__.items()},
            'model': {
                **{k: v for k, v in self.model.__dict__.items()
                   if k != 'hyperparameters'},
                'hyperparameters': self.model.hyperparameters.__dict__
            },
            'train': self.train.__dict__,
            'predict': self.predict.__dict__,
            'evaluate': self.evaluate.__dict__,
            'logging': self.logging.__dict__,
            'paths': self.paths.__dict__
        }

        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def validate(self) -> None:
        """Validate configuration values."""
        # Validate data config
        if not 0 <= self.data.validation_split <= 1:
            raise ValueError("validation_split must be between 0 and 1")

        # Validate model config
        if not 0 < self.model.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.model.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if not all(0 <= q <= 1 for q in self.model.target_quantiles):
            raise ValueError("All quantiles must be between 0 and 1")

        # Validate outlier quantiles
        if not 0 <= self.data.outlier_quantiles[0] < self.data.outlier_quantiles[1] <= 1:
            raise ValueError("Invalid outlier quantiles range")

    def get_model_save_path(self) -> Path:
        """Generate model save path with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.paths.model_save_dir) / f"{self.model.name}_{timestamp}"

    def get_results_save_path(self) -> Path:
        """Generate results save path with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.paths.results_save_dir) / f"results_{timestamp}"


