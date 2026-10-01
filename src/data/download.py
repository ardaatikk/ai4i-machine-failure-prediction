from pathlib import Path
from zipfile import ZipFile

import requests

from src.utils.config import (
    DATASET_URL,
    DATASET_ZIP_FILENAME,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
)


def download_file(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    destination.write_bytes(response.content)


def extract_zip(zip_path: Path, destination: Path) -> None:
    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(destination)


def download_dataset() -> None:
    if RAW_DATA_FILE.exists():
        print(f"Dataset already exists: {RAW_DATA_FILE}")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DATA_DIR / DATASET_ZIP_FILENAME

    print("Downloading dataset...")
    download_file(DATASET_URL, zip_path)

    print("Extracting dataset...")
    extract_zip(zip_path, RAW_DATA_DIR)

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Expected dataset file was not found: {RAW_DATA_FILE}"
        )

    zip_path.unlink(missing_ok=True)

    print(f"Dataset ready: {RAW_DATA_FILE}")


if __name__ == "__main__":
    download_dataset()