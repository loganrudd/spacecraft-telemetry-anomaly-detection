"""Zenodo data downloader for the ESA Anomaly Dataset.

Downloads mission zip files from a Zenodo record, verifies MD5 checksums,
and extracts to data/raw/{mission}/. Handles 429 rate-limiting with
exponential backoff.

Usage:
    from spacecraft_telemetry.ingest.download import ZenodoDownloader

    downloader = ZenodoDownloader(record_id="12528696", dest_dir=Path("data/raw"))
    files = downloader.get_file_list()
    downloader.download_mission("ESA-Mission1")
"""

from __future__ import annotations

import contextlib
import hashlib
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)

_CHUNK_SIZE = 65_536  # 64 KB


class ZenodoFile(BaseModel):
    """Metadata for a single file in a Zenodo record."""

    filename: str
    size: int  # bytes
    checksum: str  # "md5:<hexdigest>" as returned by the Zenodo API
    url: str  # direct download URL

    @property
    def md5(self) -> str:
        """Hex digest extracted from the 'md5:<hexdigest>' checksum string."""
        return self.checksum.removeprefix("md5:")


class ZenodoDownloader:
    """Downloads files from a Zenodo record with retry and MD5 verification.

    Args:
        record_id: Zenodo record ID (e.g. "12528696").
        dest_dir:  Directory where downloaded files are written.
        client:    Optional pre-built httpx.Client (injected for tests).
    """

    ZENODO_API = "https://zenodo.org/api/records"
    MAX_RETRIES = 5
    RETRY_BASE_DELAY = 2.0  # seconds; actual delay = base * 2^attempt

    def __init__(
        self,
        record_id: str,
        dest_dir: Path,
        client: httpx.Client | None = None,
    ) -> None:
        self._record_id = record_id
        self._dest_dir = dest_dir
        self._client = client or httpx.Client(timeout=60.0, follow_redirects=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_file_list(self) -> list[ZenodoFile]:
        """Return all files listed in the Zenodo record.

        Returns:
            List of ZenodoFile objects sorted by filename.

        Raises:
            httpx.HTTPStatusError: On non-429 HTTP errors.
            RuntimeError: If rate-limited beyond MAX_RETRIES attempts.
        """
        url = f"{self.ZENODO_API}/{self._record_id}"
        log.info("fetching file list", record_id=self._record_id)

        response = self._get_with_backoff(url)
        data: dict[str, Any] = response.json()

        files = [
            ZenodoFile(
                filename=f["key"],
                size=f["size"],
                checksum=f["checksum"],
                url=f["links"]["self"],
            )
            for f in data.get("files", [])
        ]

        log.info("file list fetched", count=len(files), record_id=self._record_id)
        return sorted(files, key=lambda f: f.filename)

    def download_file(self, file: ZenodoFile) -> Path:
        """Download a single file to dest_dir, verifying MD5 on completion.

        Skips the download if the file already exists with a matching checksum.
        On MD5 mismatch the partial file is deleted before raising.

        Args:
            file: ZenodoFile describing what to fetch.

        Returns:
            Path to the downloaded file.

        Raises:
            ValueError: If the downloaded file's MD5 doesn't match.
            httpx.HTTPStatusError: On non-429 HTTP errors.
        """
        dest = self._dest_dir / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and self._verify_md5(dest, file.md5):
            log.info("already exists, skipping", filename=file.filename)
            return dest

        size_mb = round(file.size / 1e6, 1)
        log.info("starting download", filename=file.filename, size_mb=size_mb)

        hasher = hashlib.md5()
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(file.filename, total=file.size)
            with self._stream_with_backoff(file.url) as response:
                with dest.open("wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                        fh.write(chunk)
                        hasher.update(chunk)
                        progress.advance(task, len(chunk))

        actual = hasher.hexdigest()
        if actual != file.md5:
            dest.unlink(missing_ok=True)
            raise ValueError(
                f"MD5 mismatch for {file.filename!r}: "
                f"expected {file.md5!r}, got {actual!r}"
            )

        log.info("download verified", filename=file.filename, md5=actual)
        return dest

    def download_mission(self, mission: str) -> Path:
        """Download and extract all zip files matching a mission name.

        Files are extracted to dest_dir/{mission}/.

        Args:
            mission: Mission name substring to match against filenames
                     (e.g. "ESA-Mission1"). Case-insensitive.

        Returns:
            Path to the mission directory containing extracted files.

        Raises:
            ValueError: If no files in the record match the mission name.
        """
        mission_dir = self._dest_dir / mission
        mission_dir.mkdir(parents=True, exist_ok=True)

        all_files = self.get_file_list()
        mission_files = [f for f in all_files if mission.lower() in f.filename.lower()]

        if not mission_files:
            raise ValueError(
                f"No files found for mission {mission!r} in record {self._record_id!r}. "
                f"Available files: {[f.filename for f in all_files]}"
            )

        for zf in mission_files:
            zip_path = self.download_file(zf)
            log.info("extracting", filename=zf.filename, dest=str(mission_dir))
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(mission_dir)

        return mission_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_with_backoff(self, url: str) -> httpx.Response:
        """GET a URL, retrying on 429 with exponential backoff."""
        response: httpx.Response | None = None
        for attempt in range(self.MAX_RETRIES):
            response = self._client.get(url)
            if response.status_code != 429:
                response.raise_for_status()
                return response
            delay = self.RETRY_BASE_DELAY * (2**attempt)
            log.warning(
                "rate limited, retrying",
                attempt=attempt + 1,
                max_retries=self.MAX_RETRIES,
                delay_s=delay,
                url=url,
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Rate limit exceeded after {self.MAX_RETRIES} attempts on {url!r}"
        )

    @contextlib.contextmanager
    def _stream_with_backoff(self, url: str) -> Iterator[httpx.Response]:
        """Context manager for a streaming GET with 429 backoff."""
        for attempt in range(self.MAX_RETRIES):
            with self._client.stream("GET", url) as response:
                if response.status_code == 429:
                    delay = self.RETRY_BASE_DELAY * (2**attempt)
                    log.warning(
                        "rate limited on stream, retrying",
                        attempt=attempt + 1,
                        delay_s=delay,
                    )
                    time.sleep(delay)
                    continue  # closes this response and retries
                response.raise_for_status()
                yield response
                return

        raise RuntimeError(
            f"Rate limit exceeded after {self.MAX_RETRIES} attempts on {url!r}"
        )

    @staticmethod
    def _verify_md5(path: Path, expected: str) -> bool:
        """Return True if the file at path has the expected MD5 hex digest."""
        hasher = hashlib.md5()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest() == expected
