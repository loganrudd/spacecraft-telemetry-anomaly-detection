"""Tests for ingest.download — mocked HTTP, checksum logic."""

from __future__ import annotations

import contextlib
import hashlib
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from spacecraft_telemetry.ingest.download import ZenodoDownloader, ZenodoFile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RECORD_ID = "12528696"

_FAKE_API_RESPONSE = {
    "files": [
        {
            "key": "ESA-Mission1.zip",
            "size": 1_000,
            "checksum": "md5:abc123def456",
            "links": {"self": "https://zenodo.org/files/ESA-Mission1.zip"},
        },
        {
            "key": "ESA-Mission2.zip",
            "size": 2_000,
            "checksum": "md5:deadbeef0000",
            "links": {"self": "https://zenodo.org/files/ESA-Mission2.zip"},
        },
    ]
}


def _mock_get_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


def _make_streaming_client(content: bytes) -> MagicMock:
    """Return a mock httpx.Client whose .stream() yields the given bytes."""

    @contextlib.contextmanager
    def fake_stream(method: str, url: str, **kwargs: object):  # type: ignore[misc]
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.iter_bytes.return_value = iter([content])
        yield resp

    client = MagicMock(spec=httpx.Client)
    client.stream = fake_stream
    return client


def _make_zip(files: dict[str, bytes]) -> bytes:
    """Return an in-memory zip containing the given filename→content mapping."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ZenodoFile
# ---------------------------------------------------------------------------


class TestZenodoFile:
    def test_md5_strips_prefix(self) -> None:
        f = ZenodoFile(
            filename="test.zip",
            size=100,
            checksum="md5:abc123def456",
            url="https://example.com/test.zip",
        )
        assert f.md5 == "abc123def456"

    def test_md5_with_no_prefix(self) -> None:
        # If the API ever returns a bare hash, removeprefix is a no-op.
        f = ZenodoFile(
            filename="test.zip",
            size=100,
            checksum="abc123def456",
            url="https://example.com/test.zip",
        )
        assert f.md5 == "abc123def456"


# ---------------------------------------------------------------------------
# _verify_md5 (pure function)
# ---------------------------------------------------------------------------


class TestVerifyMd5:
    def test_correct_md5_returns_true(self, tmp_path: Path) -> None:
        content = b"hello spacecraft"
        p = tmp_path / "file.bin"
        p.write_bytes(content)
        assert ZenodoDownloader._verify_md5(p, hashlib.md5(content).hexdigest())

    def test_wrong_md5_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "file.bin"
        p.write_bytes(b"some content")
        assert not ZenodoDownloader._verify_md5(p, "0" * 32)

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert ZenodoDownloader._verify_md5(p, hashlib.md5(b"").hexdigest())


# ---------------------------------------------------------------------------
# get_file_list
# ---------------------------------------------------------------------------


class TestGetFileList:
    def test_returns_zenodo_files_sorted_by_name(self, tmp_path: Path) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_get_response(200, _FAKE_API_RESPONSE)

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        files = downloader.get_file_list()

        assert len(files) == 2
        assert files[0].filename == "ESA-Mission1.zip"
        assert files[1].filename == "ESA-Mission2.zip"
        assert files[0].md5 == "abc123def456"
        assert files[0].size == 1_000

    def test_retries_on_429_then_succeeds(self, tmp_path: Path) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.side_effect = [
            _mock_get_response(429),
            _mock_get_response(200, _FAKE_API_RESPONSE),
        ]

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        with patch("time.sleep"):
            files = downloader.get_file_list()

        assert len(files) == 2
        assert client.get.call_count == 2

    def test_raises_after_max_retries(self, tmp_path: Path) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_get_response(429)

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        with patch("time.sleep"), pytest.raises(RuntimeError, match="Rate limit exceeded"):
            downloader.get_file_list()

    def test_empty_record_returns_empty_list(self, tmp_path: Path) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_get_response(200, {"files": []})

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        assert downloader.get_file_list() == []


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    def _make_zenodo_file(self, filename: str, content: bytes) -> ZenodoFile:
        return ZenodoFile(
            filename=filename,
            size=len(content),
            checksum=f"md5:{hashlib.md5(content).hexdigest()}",
            url=f"https://example.com/{filename}",
        )

    def test_successful_download_writes_file(self, tmp_path: Path) -> None:
        content = b"mission telemetry data"
        zf = self._make_zenodo_file("mission.zip", content)
        client = _make_streaming_client(content)

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        result = downloader.download_file(zf)

        assert result == tmp_path / "mission.zip"
        assert result.read_bytes() == content

    def test_skips_download_if_file_exists_with_correct_md5(self, tmp_path: Path) -> None:
        content = b"existing content"
        zf = self._make_zenodo_file("existing.zip", content)
        (tmp_path / "existing.zip").write_bytes(content)

        client = MagicMock(spec=httpx.Client)
        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        result = downloader.download_file(zf)

        assert result == tmp_path / "existing.zip"
        client.stream.assert_not_called()

    def test_redownloads_if_existing_md5_wrong(self, tmp_path: Path) -> None:
        content = b"correct content"
        zf = self._make_zenodo_file("data.zip", content)
        (tmp_path / "data.zip").write_bytes(b"corrupted content")

        client = _make_streaming_client(content)
        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        result = downloader.download_file(zf)

        assert result.read_bytes() == content

    def test_raises_and_deletes_file_on_md5_mismatch(self, tmp_path: Path) -> None:
        content = b"some bytes"
        wrong_md5 = "0" * 32
        zf = ZenodoFile(
            filename="bad.zip",
            size=len(content),
            checksum=f"md5:{wrong_md5}",
            url="https://example.com/bad.zip",
        )
        client = _make_streaming_client(content)

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        with pytest.raises(ValueError, match="MD5 mismatch"):
            downloader.download_file(zf)

        assert not (tmp_path / "bad.zip").exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        content = b"data"
        zf = ZenodoFile(
            filename="nested/dir/file.zip",
            size=len(content),
            checksum=f"md5:{hashlib.md5(content).hexdigest()}",
            url="https://example.com/file.zip",
        )
        client = _make_streaming_client(content)

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        result = downloader.download_file(zf)

        assert result.exists()
        assert result.parent.is_dir()


# ---------------------------------------------------------------------------
# download_mission
# ---------------------------------------------------------------------------


class TestDownloadMission:
    def test_raises_if_no_matching_files(self, tmp_path: Path) -> None:
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_get_response(200, {"files": []})

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        with pytest.raises(ValueError, match="No files found for mission"):
            downloader.download_mission("ESA-Mission1")

    def test_extracts_zip_to_mission_dir(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({"channels/A-1.pkl": b"fake pickle", "labels.csv": b"ch,start,end"})
        md5 = hashlib.md5(zip_bytes).hexdigest()

        api_resp = {
            "files": [
                {
                    "key": "ESA-Mission1.zip",
                    "size": len(zip_bytes),
                    "checksum": f"md5:{md5}",
                    "links": {"self": "https://example.com/ESA-Mission1.zip"},
                }
            ]
        }
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_get_response(200, api_resp)

        @contextlib.contextmanager
        def fake_stream(method: str, url: str, **kwargs: object):  # type: ignore[misc]
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.iter_bytes.return_value = iter([zip_bytes])
            yield resp

        client.stream = fake_stream

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        mission_dir = downloader.download_mission("ESA-Mission1")

        assert mission_dir.is_dir()
        assert (mission_dir / "channels" / "A-1.pkl").exists()
        assert (mission_dir / "labels.csv").exists()

    def test_mission_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({"labels.csv": b"data"})
        md5 = hashlib.md5(zip_bytes).hexdigest()

        api_resp = {
            "files": [
                {
                    "key": "esa-mission1.zip",  # lowercase in Zenodo
                    "size": len(zip_bytes),
                    "checksum": f"md5:{md5}",
                    "links": {"self": "https://example.com/esa-mission1.zip"},
                }
            ]
        }
        client = MagicMock(spec=httpx.Client)
        client.get.return_value = _mock_get_response(200, api_resp)

        @contextlib.contextmanager
        def fake_stream(method: str, url: str, **kwargs: object):  # type: ignore[misc]
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.iter_bytes.return_value = iter([zip_bytes])
            yield resp

        client.stream = fake_stream

        downloader = ZenodoDownloader(_RECORD_ID, tmp_path, client=client)
        # Upper-case "ESA-Mission1" should match lower-case filename
        mission_dir = downloader.download_mission("ESA-Mission1")
        assert mission_dir.is_dir()
