import logging
from pathlib import Path


def setup_logging(level: str, save_path: str, file_name: str, file_extension: str) -> None:
    """Set up logging configuration."""
    log_dir = Path(save_path)
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    log_file = log_dir / f"{file_name}{file_extension}"

    logging.basicConfig(
        level=level.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ]
    )

