from __future__ import annotations

import argparse
import os
import subprocess
import shutil
import sys
import zipfile
from pathlib import Path


def copy_or_extract(path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out)
    elif path.is_dir():
        for item in path.iterdir():
            dest = out / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    else:
        shutil.copy2(path, out / path.name)


def download_with_kagglehub(competition: str, out: Path, copy: bool, force: bool) -> Path:
    if os.environ.get("KAGGLEHUB_TOKEN") and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLEHUB_TOKEN"]

    try:
        import kagglehub
    except ImportError as exc:
        message = str(exc)
        if "get_web_endpoint" in message or "kagglesdk" in message:
            raise RuntimeError(
                "Your kagglehub/kagglesdk versions are incompatible. "
                "Run: pip install --force-reinstall kagglehub==0.4.3"
            ) from exc
        raise

    if copy:
        try:
            path = Path(kagglehub.competition_download(competition, output_dir=str(out), force_download=force))
            return path
        except TypeError:
            path = Path(kagglehub.competition_download(competition, force_download=force))
            copy_or_extract(path, out)
            return out

    return Path(kagglehub.competition_download(competition, force_download=force))


def download_with_kaggle_cli(competition: str, out: Path, force: bool) -> Path:
    try:
        __import__("kaggle")
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle CLI fallback requires the `kaggle` package. Run: pip install kaggle"
        ) from exc

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(out),
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)
    for zip_path in out.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BirdCLEF+ 2026 competition data with kagglehub.")
    parser.add_argument("--competition", default="birdclef-2026")
    parser.add_argument("--out", type=Path, default=Path("data/birdclef-2026"), help="Directory to copy files into.")
    parser.add_argument("--copy", action="store_true", help="Copy downloaded files from kagglehub cache to --out.")
    parser.add_argument("--force", action="store_true", help="Force re-download if the backend supports it.")
    parser.add_argument("--method", choices=["auto", "kagglehub", "kaggle-cli"], default="auto")
    args = parser.parse_args()

    if args.method in {"auto", "kagglehub"}:
        try:
            path = download_with_kagglehub(args.competition, args.out, args.copy, args.force)
            print(f"Path to competition files: {path}")
            return
        except Exception as exc:
            if args.method == "kagglehub":
                raise
            print(f"kagglehub failed: {exc}")
            print("Trying Kaggle CLI fallback...")

    path = download_with_kaggle_cli(args.competition, args.out, args.force)
    print(f"Path to competition files: {path.resolve()}")
    print("If this fails with permission/403, accept the competition rules on Kaggle and configure Kaggle credentials.")


if __name__ == "__main__":
    main()
