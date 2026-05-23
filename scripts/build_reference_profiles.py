"""Build and upload Evidently reference profiles for all channels in a mission.

Usage:
    # Build locally (reads from settings.monitoring.reference_profiles_dir):
    python scripts/build_reference_profiles.py --mission ESA-Mission2

    # Read channels from a text file (local or gs://):
    python scripts/build_reference_profiles.py \\
        --mission ESA-Mission2 \\
        --channels-from gs://my-project-processed-data/ESA-Mission2/channels.txt

    # Build and upload to GCS:
    python scripts/build_reference_profiles.py \\
        --mission ESA-Mission2 \\
        --upload gs://my-project-artifacts/reference_profiles

    # Both:
    python scripts/build_reference_profiles.py \\
        --env cloud \\
        --mission ESA-Mission2 \\
        --channels-from gs://my-project-processed-data/ESA-Mission2/channels.txt \\
        --upload gs://my-project-artifacts/reference_profiles

Requires: .[tracking,gcp] installed (evidently, gcsfs, google-cloud-storage).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.evidently_monitoring import (
    build_reference_profile,
    reference_profile_path,
    save_reference_profile,
)
from spacecraft_telemetry.ray_training import discover_channels

log = get_logger(__name__)


def _read_channels_from_file(path: str) -> list[str]:
    """Read channel IDs from a local or GCS text file, one per line."""
    import fsspec

    with fsspec.open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]  # type: ignore[union-attr]


def _upload_to_gcs(local_path: Path, gcs_prefix: str, mission: str, channel: str) -> None:
    """Upload a local Parquet file to GCS under the canonical profile path."""
    import gcsfs

    remote_path = f"{gcs_prefix.rstrip('/')}/{mission}/{channel}/reference.parquet"
    fs = gcsfs.GCSFileSystem()
    fs.put(str(local_path), remote_path)
    log.info("profile.uploaded", channel=channel, dest=remote_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Evidently reference profiles.")
    parser.add_argument("--env", default="local", help="Config env (local, cloud, test).")
    parser.add_argument("--mission", required=True, help="Mission name, e.g. ESA-Mission2.")
    parser.add_argument(
        "--channels-from",
        default=None,
        metavar="PATH",
        help="Path to a text file (local or gs://) with one channel ID per line. "
        "Defaults to discover_channels() from the processed-data directory.",
    )
    parser.add_argument(
        "--upload",
        default=None,
        metavar="GCS_PREFIX",
        help="GCS prefix to upload profiles to, e.g. gs://project-artifacts/reference_profiles. "
        "Profiles are also saved locally unless --upload-only is set.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip saving profiles locally; upload directly to GCS (uses a BytesIO buffer).",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Cap at this many channels (useful for smoke tests).",
    )
    args = parser.parse_args()

    settings = load_settings(args.env)

    # Resolve channel list.
    if args.channels_from:
        channels = _read_channels_from_file(args.channels_from)
        log.info("channels.from_file", path=args.channels_from, n=len(channels))
    else:
        channels = discover_channels(settings, args.mission)
        if not channels:
            print(
                f"No preprocessed channels found for {args.mission}. "
                "Run `spacecraft-telemetry spark preprocess` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        log.info("channels.discovered", n=len(channels))

    if args.max_channels:
        channels = channels[: args.max_channels]

    ok = err = 0
    for channel in channels:
        try:
            df = build_reference_profile(settings, args.mission, channel)

            if not args.upload_only:
                path = reference_profile_path(settings, args.mission, channel)
                save_reference_profile(df, path)
                log.info("profile.saved", channel=channel, path=str(path))

            if args.upload:
                if args.upload_only:
                    # Write to an in-memory buffer and push to GCS directly.
                    import gcsfs

                    buf = io.BytesIO()
                    df.to_parquet(buf, index=False)
                    buf.seek(0)
                    remote = (
                        f"{args.upload.rstrip('/')}/{args.mission}/{channel}/reference.parquet"
                    )
                    fs = gcsfs.GCSFileSystem()
                    with fs.open(remote, "wb") as f:
                        f.write(buf.read())
                    log.info("profile.uploaded", channel=channel, dest=remote)
                else:
                    path = reference_profile_path(settings, args.mission, channel)
                    _upload_to_gcs(path, args.upload, args.mission, channel)

            ok += 1
        except Exception as exc:
            log.error("profile.failed", channel=channel, error=str(exc))
            err += 1

    print(f"Mission  : {args.mission}")
    print(f"Channels : {len(channels)}")
    print(f"OK       : {ok}")
    print(f"Errors   : {err}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
