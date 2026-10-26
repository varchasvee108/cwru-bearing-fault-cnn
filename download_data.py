"""Download all CWRU dataset .mat files required by cwru.py into data/raw/cwru/."""

import urllib.request
from pathlib import Path
from cwru import CWRU_FILES

BASE_URL = "https://engineering.case.edu/sites/default/files"
TARGET_DIR = Path("data/raw/cwru")


def download_cwru_data():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    file_numbers = sorted({num for numbers in CWRU_FILES.values() for num in numbers})
    total = len(file_numbers)
    print(f"Preparing to download {total} CWRU files into {TARGET_DIR.resolve()}...\n")

    for i, num in enumerate(file_numbers, 1):
        filename = f"{num}.mat"
        dest_path = TARGET_DIR / filename
        url = f"{BASE_URL}/{filename}"

        if dest_path.is_file() and dest_path.stat().st_size > 0:
            print(f"[{i}/{total}] Skipping {filename} (already exists)")
            continue

        print(f"[{i}/{total}] Downloading {filename} from {url}...", end=" ", flush=True)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
                f.write(resp.read())
            print(f"Done ({dest_path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"FAILED: {e}")
            if dest_path.exists():
                dest_path.unlink()

    print("\nDownload process completed.")


if __name__ == "__main__":
    download_cwru_data()
