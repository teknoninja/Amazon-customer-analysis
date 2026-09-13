"""Download the Customer Support on Twitter dataset via Kaggle.

Auth (pick one):
  - KAGGLE_API_TOKEN in .env  (recommended; KGAT_... tokens)
  - Legacy: KAGGLE_USERNAME + KAGGLE_KEY or ~/.kaggle/kaggle.json

Note: PyPI ``kaggle`` 1.7.x (the newest installable on Python 3.9) only
understands username/key. This script downloads with a Bearer token over HTTP
so KGAT tokens work without upgrading Python.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_OWNER = "thoughtvector"
DATASET_SLUG = "customer-support-on-twitter"
DATA_DIR = Path("data")
ZIP_FILE = DATA_DIR / f"{DATASET_SLUG}.zip"
DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_OWNER}/{DATASET_SLUG}"
)


def _dataset_already_present() -> bool:
    return (DATA_DIR / "twcs.csv").exists() or (DATA_DIR / "twcs" / "twcs.csv").exists()


def _load_legacy_json() -> tuple[str, str] | None:
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    user, key = data.get("username"), data.get("key")
    if user and key:
        return str(user), str(key)
    return None


def _auth_headers_and_auth():
    """Return (headers, auth) for requests based on available credentials."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}, None

    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if user and key:
        return {}, (user, key)

    legacy = _load_legacy_json()
    if legacy:
        return {}, legacy

    raise RuntimeError(
        "No Kaggle credentials found. Set KAGGLE_API_TOKEN=KGAT_... in .env "
        "(https://www.kaggle.com/settings/api), or use legacy username/key."
    )


def _extract_zip(zip_path: Path) -> None:
    logger.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    try:
        zip_path.unlink()
    except OSError:
        pass


def download_and_extract() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _dataset_already_present():
        logger.info("Dataset already exists under data/ — skipping download.")
        return

    headers, auth = _auth_headers_and_auth()
    logger.info(
        "Downloading %s/%s from Kaggle...",
        DATASET_OWNER,
        DATASET_SLUG,
    )

    with requests.get(
        DOWNLOAD_URL,
        headers=headers,
        auth=auth,
        stream=True,
        allow_redirects=True,
        timeout=120,
    ) as resp:
        if resp.status_code == 401:
            raise RuntimeError(
                "Kaggle returned 401 Unauthorized. Regenerate KAGGLE_API_TOKEN at "
                "https://www.kaggle.com/settings/api and update .env."
            )
        if resp.status_code == 403:
            raise RuntimeError(
                "Kaggle returned 403 Forbidden. Open the dataset page in a browser, "
                "accept any terms, then retry:\n"
                f"https://www.kaggle.com/datasets/{DATASET_OWNER}/{DATASET_SLUG}"
            )
        resp.raise_for_status()

        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        last_logged_pct = -10
        with open(ZIP_FILE, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(100.0 * downloaded / total)
                    if pct >= last_logged_pct + 10 or downloaded >= total:
                        logger.info(
                            "Download progress: %s%% (%s / %s bytes)",
                            pct,
                            downloaded,
                            total,
                        )
                        last_logged_pct = pct
                elif downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                    logger.info("Downloaded %s bytes...", downloaded)

    _extract_zip(ZIP_FILE)

    if not _dataset_already_present():
        raise FileNotFoundError(
            "Download finished but twcs.csv was not found under data/. "
            "Inspect the extracted files and accept dataset rules on Kaggle if needed."
        )

    logger.info("Dataset ready.")


if __name__ == "__main__":
    download_and_extract()
