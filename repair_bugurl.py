from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from azure_devops.client import ORG, PROJECT_URL_SAFE


def _project_base() -> Path:
    return Path(__file__).resolve().parent


def _normalize_bug_id(raw: object) -> str:
    s = str(raw).strip()
    if not s:
        return ""
    # Common pandas/Excel representation for whole numbers.
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        s = s[:-2]
    return s


def _build_bug_url(bug_id: str) -> str:
    if not bug_id:
        return ""
    return f"https://dev.azure.com/{ORG}/{PROJECT_URL_SAFE}/_workitems/edit/{bug_id}"


def repair_bugurl_column(metadata_csv: Path) -> tuple[int, Path]:
    """Ensure metadata CSV has BugUrl and fill missing values.

    Rules:
    - Never calls Azure DevOps API.
    - Only writes/updates the BugUrl column.
    - Uses ORG and PROJECT_URL_SAFE from azure_devops.client.

    Returns:
        (repaired_rows, saved_path)

        saved_path will be the original metadata_csv if the file was updated
        in-place; otherwise it will be a sidecar repaired copy if the original
        CSV was locked.
    """

    metadata_csv = Path(metadata_csv)
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")

    df = pd.read_csv(metadata_csv, dtype=str)

    # Decide which column contains the bug ID.
    id_col: str
    if "BugID" in df.columns:
        id_col = "BugID"
    elif "WorkItemId" in df.columns:
        id_col = "WorkItemId"
    else:
        raise ValueError("Metadata CSV must contain either 'BugID' or 'WorkItemId'")

    if "BugUrl" not in df.columns:
        df["BugUrl"] = ""

    bugurl_series = df["BugUrl"].fillna("").astype(str)
    missing_mask = bugurl_series.astype(str).str.strip() == ""
    if not bool(missing_mask.any()):
        return (0, metadata_csv)

    ids = df.loc[missing_mask, id_col].map(_normalize_bug_id)
    df.loc[missing_mask, "BugUrl"] = ids.map(_build_bug_url)

    repaired = int((df.loc[missing_mask, "BugUrl"].fillna("").astype(str).str.strip() != "").sum())

    # Atomic-ish write: write temp then replace. Retries help on Windows locks.
    tmp_path: Path | None = None
    last_exc: Exception | None = None
    saved_path = metadata_csv

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(metadata_csv.parent),
            prefix=metadata_csv.stem + "_",
            suffix=metadata_csv.suffix or ".csv",
        ) as f:
            tmp_path = Path(f.name)
            df.to_csv(f, index=False)
            f.flush()

        for _ in range(10):
            try:
                os.replace(str(tmp_path), str(metadata_csv))
                last_exc = None
                tmp_path = None
                break
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.5)

        if last_exc is not None:
            # Fallback: keep a repaired copy next to the original.
            fallback = metadata_csv.with_name(metadata_csv.stem + "_repaired" + metadata_csv.suffix)
            try:
                df.to_csv(fallback, index=False)
                saved_path = fallback
            except Exception:
                pass
            # Do not raise here; caller can decide what to do next.
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return (repaired, saved_path)


def main(argv: list[str]) -> int:
    csv_path = Path(argv[1]) if len(argv) > 1 else (_project_base() / "data" / "bug_metadata.csv")
    repaired, saved_path = repair_bugurl_column(csv_path)
    if saved_path == csv_path:
        print(f"Repaired BugUrl for {repaired} row(s): {csv_path}")
        return 0

    print(f"Repaired BugUrl for {repaired} row(s). CSV was locked: {csv_path}")
    print(f"Wrote repaired copy to: {saved_path}")
    print("\nClose Excel (or any program) holding the CSV, then swap it in with:")
    print(
        "  [System.IO.File]::Replace('data\\bug_metadata_repaired.csv'," \
        "'data\\bug_metadata.csv','data\\bug_metadata.csv.bak',$true)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
