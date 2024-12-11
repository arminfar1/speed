import logging
from typing import List, Optional

import awswrangler as wr
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class to read inputs from S3.
    """

    @staticmethod
    def read_raw_data_from_s3(
            bucket_path: str,
            file_pattern: str,
            needed_columns: List[str],
            ship_methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and process data from S3.
        :param bucket_path: S3 bucket path.
        :param file_pattern: File pattern to match in the bucket.
        :param needed_columns: List of required columns to retain.
        :param ship_methods: List of ship methods to filter data (optional).
        :return: Concatenated DataFrame.
        """
        try:
            combined_df = pd.read_parquet("s3://df-shipment-speed/inputs/shipment_data/*_Two_Years_test_data_2024-11-02_*.parquet")
            # # List files matching the bucket path and file pattern
            # logger.info(f"Listing files from S3 path: {bucket_path} with pattern: {file_pattern}")
            # file_paths = wr.s3.list_objects(f"{bucket_path}/{file_pattern}")
            #
            # if not file_paths:
            #     raise FileNotFoundError(f"No files found at {bucket_path} with pattern {file_pattern}")
            #
            # dfs = []
            # for path in file_paths:
            #     logger.info(f"Reading file from S3: {path}")
            #     df = wr.s3.read_parquet(path)
            #
            #     # Retain only needed columns
            #     df = df[needed_columns]
            #
            #     # Filter by ship methods if provided
            #     if ship_methods:
            #         logger.info(f"Filtering by ship methods: {ship_methods}")
            #         df = df[df["ship_method"].isin(ship_methods)]
            #
            #     dfs.append(df)
            #
            # # Concatenate all DataFrames
            # combined_df = pd.concat(dfs, ignore_index=True)
            # logger.info(f"Successfully loaded and processed data. Shape: {combined_df.shape}")
            return combined_df

        except Exception as e:
            logger.error(f"Error loading data from S3: {str(e)}")
            raise