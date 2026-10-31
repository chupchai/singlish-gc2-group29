"""
download_common_voice.py

Download a Common Voice release from Mozilla Data Collective (MDC).

Purpose
-------
Common Voice is distributed only through Mozilla Data Collective. This
script downloads one dataset archive into <project root>/raw/ using the
official `datacollective` SDK. It does NOT extract the archive; the next
scripts stream it directly.

Before the first run
--------------------
    1. Log in at Mozilla Data Collective, open the dataset page and
       accept its terms (required once per dataset).
    2. Create an API key in your MDC profile.
    3. Copy .env.example to .env (project root) and set MDC_API_KEY=...
       (or set the MDC_API_KEY environment variable).

Usage
-----
    python download_common_voice.py                  # full English (~95 GB)
    python download_common_voice.py --list_segments  # known accent segments

Downloads resume automatically if interrupted: the SDK keeps
<archive>.part and <archive>.checksum next to the target file and
continues from there when you re-run the same command.

Outputs
-------
    raw/<archive file>   -- the downloaded archive (never extracted)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from cv_utils import ENV_FILE, RAW_DIR


# ============================================================
# KNOWN DATASETS
# ============================================================

DEFAULT_SLUG = "common-voice-scripted-speech-27-0-englis-e4a13e08"

KNOWN_SEGMENTS = {
    "common-voice-scripted-speech-27-0-englis-e4a13e08": (
        "~95 GB",
        "Full English 27.0. Most Asian accents (Chinese, Korean, Filipino, "
        "Japanese, Hong Kong, ...) are ONLY here, filtered on accent metadata.",
    ),
    "common-voice-scripted-speech-26-0-south-9d6029b6": (
        "4.1 GB",
        "South Asian English accent segment (26.0).",
    ),
}

API_KEY_ENV = "MDC_API_KEY"


# ============================================================
# HELPERS
# ============================================================

def print_segments() -> None:
    print("=" * 70)
    print("KNOWN COMMON VOICE DATASETS ON MOZILLA DATA COLLECTIVE")
    print("=" * 70)
    for slug, (size, desc) in KNOWN_SEGMENTS.items():
        default = "  (default)" if slug == DEFAULT_SLUG else ""
        print(f"{slug}{default}")
        print(f"    size : {size}")
        print(f"    info : {desc}")
    print()
    print("Accept each dataset's terms on the MDC website before downloading.")


def load_api_key(env_file: Path) -> Optional[str]:
    """Load env_file (if present) without overriding existing env vars."""
    if env_file.is_file():
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    key = os.environ.get(API_KEY_ENV, "")
    return key.strip() or None


def print_missing_key_help(env_file: Path) -> None:
    print("=" * 70)
    print(f"ERROR: {API_KEY_ENV} is not set")
    print("=" * 70)
    print("The Mozilla Data Collective SDK needs an API key. To set it up:")
    print("  1. Log in at Mozilla Data Collective, open the dataset page and")
    print("     accept the dataset terms (required once per dataset).")
    print("  2. Create an API key in your MDC profile / API keys section.")
    print(f"  3. Copy .env.example to {env_file}")
    print(f"     and fill in the line  {API_KEY_ENV}=<your key>")
    print(f"     (or set the {API_KEY_ENV} environment variable).")
    print("Never commit .env; it is listed in .gitignore.")


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a Common Voice archive from Mozilla Data Collective "
                    "into raw/ (no extraction)."
    )
    p.add_argument("--slug", default=DEFAULT_SLUG,
                   help=f"MDC dataset id (default: {DEFAULT_SLUG}, full English).")
    p.add_argument("--list_segments", action="store_true",
                   help="Print the known accent-segment slugs and exit.")
    p.add_argument("--raw_dir", type=Path, default=RAW_DIR,
                   help="Download directory (default: <project root>/raw).")
    p.add_argument("--env_file", type=Path, default=ENV_FILE,
                   help="Path of the .env file holding MDC_API_KEY "
                        "(default: <project root>/.env).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download even if the archive already exists.")
    return p.parse_args(argv)


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_segments:
        print_segments()
        return 0

    if not load_api_key(args.env_file):
        print_missing_key_help(args.env_file)
        return 1

    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOWNLOADING FROM MOZILLA DATA COLLECTIVE")
    print("=" * 70)
    print(f"Dataset    : {args.slug}")
    if args.slug in KNOWN_SEGMENTS:
        print(f"Size       : {KNOWN_SEGMENTS[args.slug][0]}")
    print(f"Target dir : {raw_dir}")
    print("API key    : loaded (not shown)")
    print()
    print("Note: interrupted downloads resume automatically when you re-run")
    print("      this command (<archive>.part + <archive>.checksum in raw/).")
    print()

    # The SDK progress bar prints non-ASCII characters; on a redirected or
    # non-UTF-8 Windows console that raises UnicodeEncodeError mid-download.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        from datacollective import download_dataset
        archive = download_dataset(
            args.slug,
            download_directory=str(raw_dir),
            show_progress=True,
            overwrite_existing=args.overwrite,
        )
    except Exception as exc:
        name = type(exc).__name__
        print()
        print("=" * 70)
        print(f"DOWNLOAD FAILED ({name})")
        print("=" * 70)
        print(str(exc).replace(os.environ.get(API_KEY_ENV, "") or "\0", "<redacted>"))
        print()
        print("Common causes:")
        print("  - Dataset terms not accepted yet on the MDC dataset page.")
        print(f"  - {API_KEY_ENV} invalid or expired (create a new key on MDC).")
        print("  - Wrong --slug (see --list_segments).")
        print("  - Network interruption: re-run the same command to resume.")
        return 1

    archive = Path(archive)
    size_gb = archive.stat().st_size / 1e9 if archive.exists() else float("nan")

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Archive : {archive}")
    print(f"Size    : {size_gb:.2f} GB")
    print("The archive is NOT extracted. Next step:")
    print("  python scripts/scan_common_voice_metadata.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
