"""Download MMS burst-mode L2 CDF files from the LASP Science Data Center.

The original version of this script had three URLs hard-coded. This version queries
the SDC file-name API for whatever burst intervals fall in a date range and downloads
them, so the streaming trainer can be pointed at a much larger, more diverse dataset.

The SDC public API (no auth needed for public L2):
  list:     https://lasp.colorado.edu/mms/sdc/public/files/api/v1/file_names/science?<query>
  download: https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?file=<name>

Products of interest
  des-dist : FPI electron 3-D velocity distribution functions  (the model input)
  des-moms : FPI electron moments (density, velocity, pressure...) -- ground truth for
             moments-level validation
  fgm      : fluxgate magnetometer B-field -- needed for pitch-angle / log|B| features

Usage
-----
    python download_cdfs.py --start 2024-03-02 --end 2024-03-03 \
        --sc 1 --products des-dist des-moms fgm --out data_2024_03_02

    # dry run -- just list what would be downloaded:
    python download_cdfs.py --start 2024-03-02 --end 2024-03-03 --list-only

Notes
-----
* Burst data is bursty: a date range may contain anywhere from zero to dozens of
  short segments. Check ``--list-only`` first; each des-dist file is ~0.3-0.6 GB.
* Files already present (by name) are skipped, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

SDC = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"

# product -> (instrument_id, descriptor or None)
_PRODUCTS = {
    "des-dist": ("fpi", "des-dist"),
    "des-moms": ("fpi", "des-moms"),
    "dis-dist": ("fpi", "dis-dist"),
    "dis-moms": ("fpi", "dis-moms"),
    "fgm": ("fgm", None),
}


def list_files(sc: int, product: str, start: str, end: str) -> list[str]:
    instrument_id, descriptor = _PRODUCTS[product]
    params = {
        "sc_id": f"mms{sc}",
        "instrument_id": instrument_id,
        "data_rate_mode": "brst",
        "data_level": "l2",
        "start_date": start,
        "end_date": end,
    }
    if descriptor:
        params["descriptor"] = descriptor
    r = requests.get(f"{SDC}/file_names/science", params=params, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    return [f for f in text.split(",") if f.endswith(".cdf")]


def product_subdir(product: str) -> str:
    return "fgm" if product == "fgm" else "fpi"


def download(name: str, dest_dir: str) -> bool:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, name)
    if os.path.exists(path):
        print(f"  skip (exists): {name}")
        return False
    print(f"  downloading: {name}")
    with requests.get(f"{SDC}/download/science", params={"file": name}, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        os.replace(tmp, path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive-ish; SDC convention)")
    ap.add_argument("--sc", type=int, default=1, help="spacecraft 1-4")
    ap.add_argument("--products", nargs="+", default=["des-dist", "fgm"],
                    choices=sorted(_PRODUCTS), help="which products to fetch")
    ap.add_argument("--out", default=None, help="output dir (default data_<start with _>)")
    ap.add_argument("--list-only", action="store_true", help="just print the file list, download nothing")
    args = ap.parse_args()

    out_root = args.out or ("data_" + args.start.replace("-", "_"))
    total = 0
    for product in args.products:
        try:
            names = list_files(args.sc, product, args.start, args.end)
        except requests.HTTPError as e:
            print(f"[{product}] query failed: {e}", file=sys.stderr)
            continue
        print(f"[{product}] {len(names)} file(s) in {args.start}..{args.end}")
        for n in names:
            print(f"   {n}")
        if args.list_only:
            continue
        dest = os.path.join(out_root, product_subdir(product))
        for n in names:
            if download(n, dest):
                total += 1
    if not args.list_only:
        print(f"done: downloaded {total} new file(s) into {out_root}/")


if __name__ == "__main__":
    main()
