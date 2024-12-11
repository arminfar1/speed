import logging
from typing import Dict, Tuple
import pandas as pd
from autogluon.tabular import TabularPredictor

from direct_fulfillment_speed_forecast_model.config.model_config import Config
from direct_fulfillment_speed_forecast_model.data.data_processor import DataProcessor
from direct_fulfillment_speed_forecast_model.evaluation.metrics import ModelEvaluator
from direct_fulfillment_speed_forecast_model.models.trainer import ModelTrainer

# Set logger
logger = logging.getLogger()


class ModelPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator()

    def run(self, data: pd.DataFrame) -> Tuple[TabularPredictor, Dict]:
        """Main production pipeline for model training and evaluation."""
        try:
            logger.info("Starting main production pipeline...")

            # Data processing
            data = self.data_processor.validate_data(data)
            data = self.data_processor.feature_engineering(data)
            data = self.data_processor.handle_outliers(data)
            train_data, val_data = self.data_processor.split_data(
                data, strat_cols=['month_of', 'dow_of']
            )

            # Model training
            predictor, save_path = self.trainer.train(train_data)

            # Model evaluation
            results = self.evaluate_model(predictor, val_data)

            logger.info("Pipeline execution complete.")
            return predictor, results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def evaluate_model(self, predictor: TabularPredictor,
                      val_data: pd.DataFrame) -> Dict:
        """Evaluates model performance and returns metrics."""
        val_features = val_data.drop(columns=[self.config.model.label])
        val_targets = val_data[self.config.model.label]
        predictions = predictor.predict(val_features)

        results = {}
        for i, quantile in enumerate(self.config.model.target_quantiles):
            metrics = self.evaluator(
                predictions.iloc[:, i], val_targets, quantile
            )
            results[quantile] = {'metrics': metrics}

        coverage_metrics = self.evaluator.calculate_coverage_metrics(
            predictions, val_targets, self.config.model.target_quantiles
        )

        return {
            'quantile_metrics': results,
            'coverage_metrics': coverage_metrics
        }

