import logging

import pandas as pd
from typing import Tuple, List, Optional

from direct_fulfillment_speed_forecast_model.config.model_config import Config



# Set logger
logger = logging.getLogger()

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validates data for consistency and missing values."""
        logger.info("Starting data validation...")
        missing_cols = set([self.config.model.label] + self.config.model.get_all_features) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        data = data[[self.config.model.label] + self.config.model.get_all_features].copy()
        if data.isnull().any().any():
            logger.info("Dropping rows with missing values.")
            data.dropna(inplace=True)
        return data

    @staticmethod
    def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
        """Performs feature engineering and type conversions."""
        logger.info("Starting feature engineering...")

        # Perform type conversions
        data['dow_of'] = data['dow_of'].astype(int)
        data['month_of'] = data['month_of'].astype(int)
        data['unpadded_c2p_weekofyear'] = data['unpadded_c2p_weekofyear'].astype(int)
        data['c2d_days_1'] = data['c2d_days_1'].astype(float)
        data['vendor_primary_gl_description'] = data['vendor_primary_gl_description'].astype(str)
        data['warehouse_id'] = data['warehouse_id'].astype(str)
        data['destination_zip3'] = data['destination_zip3'].astype(str)

        # Fix nullable integer columns (convert Int32 to int64)
        nullable_int_cols = data.select_dtypes(include=['Int32']).columns
        for col in nullable_int_cols:
            logger.info(f"Converting nullable integer column '{col}' to int64")
            data[col] = data[col].astype('int64')

        # Verify all object columns are strings
        object_cols = data.select_dtypes(include=['object']).columns
        for col in object_cols:
            logger.info(f"Converting object column '{col}' to string")
            data[col] = data[col].astype(str)

        logger.info("Feature engineering complete.")
        return data

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles outliers by capping values at specified quantiles."""
        lower, upper = df[self.config.model.label].quantile(list(self.config.data.outlier_quantiles))
        original_mean = df[self.config.model.label].mean()
        logger.info(f"Capping outliers for {self.config.model.label}: [{lower:.2f}, {upper:.2f}]")
        df[self.config.model.label] = df[self.config.model.label].clip(lower=lower, upper=upper)
        new_mean = df[self.config.model.label].mean()
        logger.info(f"Outlier handling changed mean from {original_mean:.2f} to {new_mean:.2f}")
        return df

    def split_data(self, data: pd.DataFrame, strat_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits data into training and validation sets."""
        from sklearn.model_selection import train_test_split

        if strat_cols and all(col in data.columns for col in strat_cols):
            data['strat_col'] = data[strat_cols].astype(str).agg('_'.join, axis=1)
            train_data, val_data = train_test_split(
                data, test_size=self.config.model.test_size,
                random_state=self.config.model.random_state
            )
            data.drop(columns=['strat_col'], inplace=True)
        else:
            train_data, val_data = train_test_split(
                data, test_size=self.config.model.test_size,
                random_state=self.config.model.test_size
            )

        logger.info(f"Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
        return train_data, val_data