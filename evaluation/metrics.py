import logging
from typing import List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEvaluator:
    @staticmethod
    def calculate_coverage_metrics(predictions: pd.DataFrame,
                                   targets: pd.Series,
                                   target_quantiles: List[float]) -> Dict:
        """
        Calculates coverage metrics for quantile predictions.

        Args:
            predictions: DataFrame with predictions for each quantile
            targets: Series with actual target values
            target_quantiles: List of quantiles used for predictions

        Returns:
            Dictionary containing coverage metrics for each interval
        """
        coverage_metrics = {}

        # Calculate coverage for each symmetric interval
        for i in range(len(target_quantiles) // 2):
            low_idx, high_idx = i, len(target_quantiles) - i - 1
            low_q, high_q = target_quantiles[low_idx], target_quantiles[high_idx]

            # Calculate actual coverage
            within_interval = (
                    (predictions.iloc[:, low_idx] <= targets) &
                    (targets <= predictions.iloc[:, high_idx])
            ).astype(float)
            actual_coverage = within_interval.mean()

            # Calculate desired coverage
            desired_coverage = high_q - low_q

            # Store metrics
            coverage_metrics[f"{low_q:.2f}-{high_q:.2f}"] = {
                'desired_coverage': desired_coverage,
                'actual_coverage': actual_coverage,
                'coverage_error': actual_coverage - desired_coverage,
                'absolute_coverage_error': abs(actual_coverage - desired_coverage),
                'relative_coverage_error': (actual_coverage - desired_coverage) / desired_coverage,
                'num_samples': len(targets),
                'num_within_interval': within_interval.sum()
            }

            logger.info(
                f"Coverage {low_q:.2f}-{high_q:.2f}: "
                f"Desired={desired_coverage:.3f}, "
                f"Actual={actual_coverage:.3f}, "
                f"Error={actual_coverage - desired_coverage:.3f}"
            )

        # Add summary metrics
        coverage_metrics['summary'] = {
            'mean_absolute_coverage_error': np.mean([
                m['absolute_coverage_error']
                for m in coverage_metrics.values()
                if isinstance(m, dict) and 'absolute_coverage_error' in m
            ]),
            'max_absolute_coverage_error': max([
                m['absolute_coverage_error']
                for m in coverage_metrics.values()
                if isinstance(m, dict) and 'absolute_coverage_error' in m
            ]),
            'total_samples': len(targets)
        }

        return coverage_metrics

    @staticmethod
    def calculate_quantile_metrics(predictions: pd.Series,
                                   targets: pd.Series,
                                   quantile: float) -> Dict:
        """
        Calculates metrics for a single quantile.

        Args:
            predictions: Series with predictions for this quantile
            targets: Series with actual target values
            quantile: The quantile level

        Returns:
            Dictionary containing metrics for this quantile
        """
        # Calculate quantile loss
        errors = targets - predictions
        quantile_loss = np.mean(
            np.maximum(
                quantile * errors,
                (quantile - 1) * errors
            )
        )

        return {
            'mean_prediction': predictions.mean(),
            'std_prediction': predictions.std(),
            'min_prediction': predictions.min(),
            'max_prediction': predictions.max(),
            'quantile_loss': quantile_loss,
            'skewness': predictions.skew(),
            'kurtosis': predictions.kurtosis(),
            'nan_percentage': (predictions.isna().sum() / len(predictions)) * 100
        }

