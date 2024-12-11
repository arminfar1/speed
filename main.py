import argparse
import logging
from pathlib import Path

from direct_fulfillment_speed_forecast_model.config.model_config import Config
from direct_fulfillment_speed_forecast_model.data.data_loader import DataLoader
from direct_fulfillment_speed_forecast_model.pipeline.pipeline import ModelPipeline
from direct_fulfillment_speed_forecast_model.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Direct Fulfillment Speed Forecast Model')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'configs' / 'base.yaml',  # Use absolute path
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'evaluate'],
        default='train',
        help='Mode to run the model in'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        help='Path to saved model for predict/evaluate mode',
        required=False
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()

    # Load and validate configuration
    config = Config.from_yaml(args.config)
    config.validate()

    # Setup logging configuration
    setup_logging(
        level=config.logging.level,
        save_path=config.logging.save_path,
        file_name=config.logging.file_name,
        file_extension=config.logging.file_extension
    )

    logging.info(f"Starting {args.mode} mode with config from {args.config}")

    try:
        # Initialize ModelPipeline
        model_pipeline = ModelPipeline(config)

        # Run the appropriate mode
        if args.mode == 'train':
            # Load raw data
            logging.info("Loading raw data...")
            data_loader = DataLoader()
            raw_data = data_loader.read_raw_data_from_s3(
                bucket_path=config.data.bucket_path,
                file_pattern=config.data.file_pattern,
                needed_columns=config.model.get_all_features + [config.model.label],
                ship_methods=config.data.ship_methods
            )

            # Execute the training pipeline
            predictor, results = model_pipeline.run(raw_data)

            # Save results
            results_path = config.get_results_save_path()
            results_path.mkdir(parents=True, exist_ok=True)
            # Optionally, save `results` or the trained model
            logging.info(f"Training completed. Results saved to {results_path}")

        elif args.mode == 'predict':
            if not args.model_path:
                raise ValueError("Model path is required for prediction mode")
            # Implement prediction pipeline logic (if applicable)

        elif args.mode == 'evaluate':
            if not args.model_path:
                raise ValueError("Model path is required for evaluation mode")
            # Implement evaluation pipeline logic (if applicable)

    except Exception as e:
        logging.error(f"Error in {args.mode} mode: {str(e)}")
        raise


if __name__ == "__main__":
    main()