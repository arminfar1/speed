import logging
from typing import Dict, Tuple
import pandas as pd
from autogluon.tabular import TabularPredictor
from datetime import datetime
from pathlib import Path

from fastcore.basics import num_cpus

from direct_fulfillment_speed_forecast_model.config.model_config import Config


# Set logger
logger = logging.getLogger()


class ModelTrainer:

    def __init__(self, config: Config):
        self.config = config

    @property
    def configure_hyperparameters(self) -> Dict:
        """Returns model hyperparameters for quantile regression."""
        return self.config.model.hyperparameters.__dict__

    def train(self, train_data: pd.DataFrame) -> Tuple[TabularPredictor, str]:
        """Trains the model using AutoGluon."""
        save_path = Path(f'models/quantile_model_{datetime.now().strftime("%Y%m%d_%H%M")}')
        save_path.mkdir(parents=True, exist_ok=True)

        predictor = TabularPredictor(
            label=self.config.model.label,
            path=str(save_path),
            problem_type='quantile',
            quantile_levels=self.config.model.target_quantiles,
            eval_metric='pinball_loss'
        )

        logger.info("Starting model training...")
        predictor.fit(
            train_data=train_data,
            hyperparameters=self.configure_hyperparameters,
            num_bag_folds=5,
            presets='best_quality',
            time_limit=self.config.model.time_limit,
            num_cpus=self.config.model.num_cpus,
        )

        logger.info(f"Model training complete. Model saved at {save_path}.")
        return predictor, str(save_path)