from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import sys
import os
import threading
import tempfile
from typing import TYPE_CHECKING

import runtime_paths


if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer


# -----------------------------
# Lazy, offline-only model loading
# -----------------------------
# The transformer model is the slowest part of startup (especially on Windows).
# We intentionally do NOT load it at app startup. Instead, we load it once on
# the first search and reuse it for all subsequent searches.

_MODEL_LOCK = threading.Lock()
_MODEL: "SentenceTransformer | None" = None
_MODEL_LOADING = False
_MODEL_WARMED = False
_TORCH_CONFIGURED = False


# -----------------------------
# Embeddings loading strategy
# -----------------------------
# Memory-mapping keeps the .npy on disk and pages data in as needed.
# This significantly reduces RAM usage for larger corpora.
USE_MMAP = True


# -----------------------------
# Pre-normalized embeddings
# -----------------------------
# When enabled, cosine similarity becomes a simple dot product.
USE_PRENORMALIZED = True
_PRENORM_LOGGED = False


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _app_base_path() -> Path:
    """Base directory for runtime read/write files.

    Requirements:
    - If frozen: Path(sys.executable).parent
    - Else: Path(__file__).resolve().parent
    """

    if _is_frozen():
        try:
            return Path(sys.executable).resolve().parent
        except Exception:
            return Path.cwd().resolve()
    return Path(__file__).resolve().parent


def _resolve_runtime_path(path: str | Path) -> Path:
    """Resolve a path for runtime artifacts.

    If `path` is relative, it is treated as inside the runtime writable area:
      - DEV: <repo>/data/...
      - EXE: %LOCALAPPDATA%\\BugSearchApp\\data/...
    """

    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return runtime_paths.get_runtime_file(str(p))


def _close_memmap_if_needed(arr: object) -> None:
    """Best-effort close for numpy memmap-backed arrays (Windows file locks)."""

    try:
        if isinstance(arr, np.memmap):
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()
    except Exception:
        return


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from a .npy file.

    - When USE_MMAP=True, uses numpy memory-mapping (read-only).
    - Otherwise, loads the array eagerly into RAM.

    Raises:
        FileNotFoundError: if the file does not exist.
    """

    p = _resolve_runtime_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {p}")

    if USE_MMAP:
        arr = np.load(str(p), mmap_mode="r")
        logging.info("Embeddings loaded using memory-mapping")
        return arr

    return np.load(str(p))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize a 2D matrix.

    - Works with ndarray or memmap input.
    - Avoids division by zero.
    - Returns float32.
    """

    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.ndim != 2:
        raise ValueError("l2_normalize expects a 1D or 2D matrix")

    norms = np.linalg.norm(m, axis=1, keepdims=True)
    denom = np.clip(norms, 1e-12, None)
    return (m / denom).astype(np.float32, copy=False)


def is_normalized(matrix: np.ndarray, sample_size: int = 1000) -> bool:
    """Heuristically check if rows are unit-normalized.

    Samples up to `sample_size` rows to avoid a full scan on large files.
    """

    try:
        rows = int(matrix.shape[0])
    except Exception:
        return False

    if rows <= 0:
        return True

    n = max(1, min(int(sample_size), rows))
    try:
        rng = np.random.default_rng()
        idx = rng.choice(rows, size=n, replace=False)
        sample = np.asarray(matrix[idx], dtype=np.float32)
        norms = np.linalg.norm(sample, axis=1)
        if not np.isfinite(norms).all():
            return False
        return bool(np.allclose(norms, 1.0, atol=1e-3, rtol=0.0))
    except Exception:
        return False


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """Atomically overwrite a .npy file (write temp then replace)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(path.parent),
            prefix=path.stem + "_",
            suffix=path.suffix or ".npy",
        ) as f:
            tmp_path = Path(f.name)
            np.save(f, array)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path is not None and tmp_path.exists() and tmp_path != path:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _ensure_prenormalized_embeddings(path: Path) -> np.ndarray:
    """Load embeddings and ensure they are unit-normalized if enabled.

    If embeddings are not normalized and USE_PRENORMALIZED is True, performs
    a one-time migration:
      - materialize into RAM
      - normalize row-wise to float32
      - atomically overwrite the .npy
      - reload via load_embeddings (mmap if enabled)
    """

    global _PRENORM_LOGGED

    emb = load_embeddings(str(path))
    if not USE_PRENORMALIZED:
        return emb

    if not _PRENORM_LOGGED:
        logging.info("Using pre-normalized embeddings (cosine via dot product)")
        _PRENORM_LOGGED = True

    if is_normalized(emb):
        return emb

    logging.info("Normalizing embeddings (one-time migration)")

    # Materialize into RAM first (safe even for memmap).
    materialized = np.array(emb)

    # Release the mmap handle before overwriting the file (Windows).
    _close_memmap_if_needed(emb)

    # Normalize, then atomically overwrite.
    normalized = l2_normalize(materialized)
    _atomic_write_npy(path, normalized)

    # Reload (will use mmap if enabled)
    return load_embeddings(str(path))


def is_model_loaded() -> bool:
    return _MODEL is not None


def is_model_loading() -> bool:
    return _MODEL_LOADING


def _configure_torch_for_startup() -> None:
    """Reduce PyTorch startup overhead on Windows.

    Setting threads to 1 avoids PyTorch spinning up a large threadpool during
    model initialization, which can noticeably slow startup on some machines.
    """

    global _TORCH_CONFIGURED
    if _TORCH_CONFIGURED:
        return

    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        # If torch isn't available for some reason, don't block the app.
        pass

    _TORCH_CONFIGURED = True


def get_model(local_model_dir: str) -> "SentenceTransformer":
    """Singleton-style model accessor.

    IMPORTANT: This function only loads from a local directory path to guarantee
    fully offline behavior and avoid HuggingFace cache scanning / downloads.
    The model directory is expected to be bundled with PyInstaller under:
      ./models/all-MiniLM-L6-v2
    """

    global _MODEL, _MODEL_LOADING, _MODEL_WARMED

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        model_path = Path(local_model_dir)
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(
                "Local model directory not found. Expected a bundled folder at: "
                f"{model_path}"
            )

        # Force offline behavior (prevents accidental network calls).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        _configure_torch_for_startup()

        _MODEL_LOADING = True
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(model_path))

            # One-time warm-up to avoid first real search feeling "stuck".
            if not _MODEL_WARMED:
                try:
                    model.encode(["warm up"], normalize_embeddings=True)
                except Exception:
                    pass
                _MODEL_WARMED = True

            _MODEL = model
            return _MODEL
        finally:
            _MODEL_LOADING = False


class SemanticBugSearch:
    def __init__(
        self,
        embeddings_path="data/bug_embeddings.npy",
        metadata_path="data/bug_metadata.csv",
        model_name="all-MiniLM-L6-v2",
        source_xlsx_path="data/bugs_semantic.xlsx",
    ):
        self._embeddings_path = _resolve_runtime_path(embeddings_path)
        self._metadata_path = _resolve_runtime_path(metadata_path)
        self._source_xlsx_path = _resolve_runtime_path(source_xlsx_path) if str(source_xlsx_path).strip() else None

        # Load data
        self.embeddings = _ensure_prenormalized_embeddings(self._embeddings_path)
        self.metadata = pd.read_csv(self._metadata_path, dtype={"WorkItemId": str})

        # Defensive validation: embeddings and metadata must align 1:1.
        # If they don't, searches can crash (or worse: return incorrect matches).
        emb_rows = int(self.embeddings.shape[0])
        meta_rows = int(len(self.metadata))
        if emb_rows != meta_rows:
            raise RuntimeError(
                "Embeddings/metadata are out of sync. "
                f"Embeddings rows={emb_rows}, metadata rows={meta_rows}. "
                "Rebuild both files together by running: python build_embeddings.py"
            )

        # Store model name; resolve/check local directory only when the model is
        # actually initialized (keeps startup fast and lets the UI show even if
        # the model folder hasn't been bundled yet).
        self._model_name = model_name
        self._local_model_dir: str | None = None

        # Best-effort: if metadata CSV doesn't contain a link column, try to pull it
        # from the source Excel by matching WorkItemId (keeps embeddings intact).
        if self._source_xlsx_path is not None and self._source_xlsx_path.is_file():
            self._try_augment_links_from_xlsx(str(self._source_xlsx_path))

    def reload_index(self) -> None:
        """Reload embeddings + metadata from disk.

        Intended to be called after Refresh, which atomically replaces the
        embeddings/metadata files on disk.
        """

        # Support older instances created via __new__ that may not have paths.
        if not hasattr(self, "_embeddings_path"):
            self._embeddings_path = _resolve_runtime_path("data/bug_embeddings.npy")
        if not hasattr(self, "_metadata_path"):
            self._metadata_path = _resolve_runtime_path("data/bug_metadata.csv")
        if not hasattr(self, "_source_xlsx_path"):
            self._source_xlsx_path = None

        # Release any open mmap handle before reloading (important on Windows).
        _close_memmap_if_needed(getattr(self, "embeddings", None))

        self.embeddings = _ensure_prenormalized_embeddings(self._embeddings_path)
        self.metadata = pd.read_csv(self._metadata_path, dtype={"WorkItemId": str})

        emb_rows = int(self.embeddings.shape[0])
        meta_rows = int(len(self.metadata))
        if emb_rows != meta_rows:
            raise RuntimeError(
                "Embeddings/metadata are out of sync after reload. "
                f"Embeddings rows={emb_rows}, metadata rows={meta_rows}. "
                "Rebuild both files together by running: python build_embeddings.py"
            )

        if self._source_xlsx_path is not None and self._source_xlsx_path.is_file():
            try:
                self._try_augment_links_from_xlsx(str(self._source_xlsx_path))
            except Exception:
                pass

    def _resource_base_dir(self) -> Path:
        # For this app, bundled runtime resources live next to the executable.
        # We avoid sys._MEIPASS here so the model is loaded from a stable path.
        return _app_base_path()

    def _resolve_local_model_dir(self, model_name: str) -> str:
        """Resolve a strictly-local model directory.

        We intentionally do NOT allow falling back to a HuggingFace model name
        because that may scan caches or download files at runtime.
        """

        candidate = Path(model_name)
        if candidate.exists() and candidate.is_dir():
            return str(candidate)

        bundled = runtime_paths.get_model_dir() / model_name
        if bundled.exists() and bundled.is_dir():
            return str(bundled)

        raise FileNotFoundError(
            "Model directory not found. Expected a bundled local folder at: "
            f"{bundled}"
        )

    def _get_local_model_dir(self) -> str:
        if self._local_model_dir is None:
            self._local_model_dir = self._resolve_local_model_dir(self._model_name)
        return self._local_model_dir

    def load_model(self) -> None:
        """Explicit model initializer (safe to call multiple times)."""

        get_model(self._get_local_model_dir())

    def _try_augment_links_from_xlsx(self, xlsx_path: str = "data/bugs_semantic.xlsx") -> None:
        try:
            xlsx_path = str(_resolve_runtime_path(xlsx_path))
        except Exception:
            pass

        # Normalize metadata to always expose a lower-case 'link' column if possible.
        if "link" in self.metadata.columns:
            self.metadata["link"] = self._clean_link_series(self.metadata["link"])
            # If we already have real links, don't touch anything.
            if (self.metadata["link"].astype(str).str.strip() != "").any():
                return
        if "Link" in self.metadata.columns:
            self.metadata["link"] = self._clean_link_series(self.metadata["Link"].fillna("").astype(str))
            if (self.metadata["link"].astype(str).str.strip() != "").any():
                return

        # If we don't have any link info in the CSV, attempt to read from Excel.
        link_map = self._read_xlsx_workitemid_to_link_map(xlsx_path)
        if not link_map:
            return

        if "WorkItemId" not in self.metadata.columns:
            return

        # Map by WorkItemId (robust even if ordering/row counts differ).
        work_ids = pd.to_numeric(self.metadata["WorkItemId"], errors="coerce").astype("Int64")
        self.metadata["link"] = self._clean_link_series(
            work_ids.map(lambda x: link_map.get(int(x), "") if pd.notna(x) else "")
        )

    def _clean_link_series(self, s: "pd.Series") -> "pd.Series":
        """Clean common bogus link values (e.g., cached formula results like 0).

        Some Excel exports store hyperlink formulas whose cached value is `0`.
        We treat those as missing links.
        """

        cleaned = s.fillna("").astype(str).str.strip()
        bogus = {"0", "0.0", "0.0.0.0", "nan", "None"}
        return cleaned.map(lambda v: "" if v in bogus else v)

    def _read_xlsx_workitemid_to_link_map(self, xlsx_path: str) -> dict[int, str]:
        """Read {WorkItemId -> link} from the first sheet of an .xlsx using stdlib only."""

        path = Path(xlsx_path)
        if not path.exists():
            return {}

        ns = {
            "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
            "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
        }

        def col_to_index(cell_ref: str) -> int:
            letters = []
            for ch in cell_ref:
                if ch.isalpha():
                    letters.append(ch.upper())
                else:
                    break
            idx = 0
            for ch in letters:
                idx = idx * 26 + (ord(ch) - ord("A") + 1)
            return idx

        def get_text(el: ET.Element | None) -> str:
            return (el.text or "") if el is not None else ""

        def read_cell_value(cell: ET.Element, shared: list[str]) -> str:
            cell_type = cell.attrib.get("t")
            v = cell.find("main:v", ns)
            if cell_type == "s" and v is not None:
                try:
                    s_idx = int(get_text(v))
                    return shared[s_idx] if 0 <= s_idx < len(shared) else ""
                except Exception:
                    return ""
            if cell_type == "inlineStr":
                parts = [get_text(t) for t in cell.findall(".//main:is/main:t", ns)]
                return "".join(parts)
            if v is not None:
                return get_text(v)
            return ""

        def parse_cell_ref(ref: str) -> tuple[int, int] | None:
            # Returns (col_idx, row_idx)
            if not ref:
                return None
            m = re.match(r"^\$?([A-Za-z]{1,3})\$?(\d+)$", ref)
            if not m:
                return None
            col_letters, row_digits = m.group(1), m.group(2)
            return col_to_index(col_letters + row_digits), int(row_digits)

        def try_extract_hyperlink_from_formula(
            formula: str,
            current_row_number: int,
            cells_by_col: dict[int, ET.Element],
            shared: list[str],
            fallback_work_id: str,
        ) -> str:
            f = (formula or "").strip()
            # Some parsers (e.g., openpyxl) represent formulas with a leading '='.
            # The XML formula text usually omits it, but handle both forms.
            if f.startswith("="):
                f = f[1:].lstrip()
            if not f.upper().startswith("HYPERLINK("):
                return ""

            # Grab first argument expression inside HYPERLINK(expr, ...)
            inner = f[len("HYPERLINK(") :]
            # Find the first comma not inside quotes or parentheses.
            depth = 0
            in_quotes = False
            cut = None
            for i, ch in enumerate(inner):
                if ch == '"':
                    in_quotes = not in_quotes
                elif not in_quotes:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        if depth == 0:
                            # reached end of function, stop
                            break
                        depth -= 1
                    elif ch == ',' and depth == 0:
                        cut = i
                        break
            expr = inner[:cut].strip() if cut is not None else inner.strip().rstrip(')')

            # Evaluate very small subset: concatenation with '&' of string literals and cell refs.
            parts = [p.strip() for p in expr.split('&') if p.strip()]
            out = []
            for p in parts:
                if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
                    out.append(p[1:-1])
                    continue

                # cell ref like R2 or $R$2
                m = re.match(r"^\$?([A-Za-z]{1,3})\$?(\d+)$", p)
                if m:
                    col_letters = m.group(1)
                    # Use the current row for relative refs; formulas here use same-row refs.
                    ref = f"{col_letters}{current_row_number}"
                    parsed = parse_cell_ref(ref)
                    if parsed is not None:
                        col_idx, _ = parsed
                        cell = cells_by_col.get(col_idx)
                        if cell is not None:
                            val = read_cell_value(cell, shared).strip()
                            if val and val not in {"0", "0.0"}:
                                out.append(val)
                            else:
                                out.append(fallback_work_id)
                            continue
                    out.append(fallback_work_id)
                    continue

                # If we can't understand the token, ignore it.
            candidate = "".join(out).strip()

            # If we only managed to extract the base edit URL, attach the WorkItemId.
            # This handles cases where the formula references an empty/missing cell.
            if candidate.startswith(("http://", "https://")):
                normalized = candidate.rstrip("/")
                if normalized.endswith("/edit") and not normalized.endswith("/edit/" + str(fallback_work_id)):
                    # Only append if the last segment is not already a number.
                    last_segment = normalized.split("/")[-1]
                    if not last_segment.isdigit():
                        candidate = candidate + str(fallback_work_id)

            return candidate

        try:
            with zipfile.ZipFile(path) as z:
                # shared strings (optional)
                shared: list[str] = []
                try:
                    ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
                    for si in ss_root.findall("main:si", ns):
                        parts = [get_text(t) for t in si.findall(".//main:t", ns)]
                        shared.append("".join(parts))
                except KeyError:
                    shared = []

                wb_root = ET.fromstring(z.read("xl/workbook.xml"))
                sheet_el = wb_root.find("main:sheets/main:sheet", ns)
                if sheet_el is None:
                    return {}

                rid = sheet_el.attrib.get(f"{{{ns['rel']}}}id")
                if not rid:
                    return {}

                rels_root = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
                target = None
                for rel in rels_root.findall("pkgrel:Relationship", ns):
                    if rel.attrib.get("Id") == rid:
                        target = rel.attrib.get("Target")
                        break
                if not target:
                    return {}

                sheet_path = "xl/" + target.lstrip("/")
                sheet_root = ET.fromstring(z.read(sheet_path))

                # Hyperlinks can be stored as relationships rather than cell text.
                hyperlink_by_ref: dict[str, str] = {}
                try:
                    rels_path = "/".join(sheet_path.split("/")[:-1]) + "/_rels/" + sheet_path.split("/")[-1] + ".rels"
                    sheet_rels_root = ET.fromstring(z.read(rels_path))
                    hyperlink_targets: dict[str, str] = {}
                    for rel in sheet_rels_root.findall("pkgrel:Relationship", ns):
                        if rel.attrib.get("Type", "").endswith("/hyperlink"):
                            hyperlink_targets[rel.attrib.get("Id", "")] = rel.attrib.get("Target", "")

                    for h in sheet_root.findall(".//main:hyperlinks/main:hyperlink", ns):
                        ref = h.attrib.get("ref")
                        rid2 = h.attrib.get(f"{{{ns['rel']}}}id")
                        if ref and rid2 and rid2 in hyperlink_targets:
                            hyperlink_by_ref[ref] = hyperlink_targets[rid2]
                except KeyError:
                    hyperlink_by_ref = {}

                sheet_data = sheet_root.find("main:sheetData", ns)
                if sheet_data is None:
                    return {}

                # Shared formulas: store template text by shared index (si)
                shared_formulas: dict[str, tuple[str, int]] = {}

                # Header row
                header_row = sheet_data.find("main:row[@r='1']", ns)
                if header_row is None:
                    header_row = sheet_data.find("main:row", ns)
                if header_row is None:
                    return {}

                header_by_col: dict[int, str] = {}
                for c in header_row.findall("main:c", ns):
                    ref = c.attrib.get("r")
                    if not ref:
                        continue
                    header_by_col[col_to_index(ref)] = read_cell_value(c, shared).strip()

                # Find column indices for WorkItemId and link
                work_col = None
                link_col = None
                for col_idx, name in header_by_col.items():
                    if name == "WorkItemId":
                        work_col = col_idx
                    if name in ("link", "Link"):
                        link_col = col_idx
                if work_col is None or link_col is None:
                    return {}

                result: dict[int, str] = {}
                for row in sheet_data.findall("main:row", ns):
                    # skip header
                    if row.attrib.get("r") == "1":
                        continue

                    cells_by_col: dict[int, ET.Element] = {}
                    for c in row.findall("main:c", ns):
                        ref = c.attrib.get("r")
                        if not ref:
                            continue
                        cells_by_col[col_to_index(ref)] = c

                    # Capture shared formula templates for the link column even if the row's
                    # WorkItemId cell is missing/invalid (some sheets have sparse data).
                    row_number = int(row.attrib.get("r", "0") or "0")
                    link_cell = cells_by_col.get(link_col)
                    if link_cell is not None:
                        f_el = link_cell.find("main:f", ns)
                        if f_el is not None and f_el.attrib.get("t") == "shared":
                            si = f_el.attrib.get("si")
                            f_text = get_text(f_el).strip()
                            if si is not None and f_text:
                                shared_formulas[si] = (f_text, row_number)

                    work_cell = cells_by_col.get(work_col)
                    if work_cell is None:
                        continue

                    work_raw = read_cell_value(work_cell, shared).strip()
                    if not work_raw:
                        continue
                    try:
                        work_id = int(float(work_raw))
                    except Exception:
                        continue

                    link_val = ""
                    if link_cell is not None:
                        # 1) Prefer hyperlink relationship targets (if present)
                        link_ref = link_cell.attrib.get("r", "")
                        if link_ref and link_ref in hyperlink_by_ref:
                            link_val = hyperlink_by_ref[link_ref].strip()
                        else:
                            # 2) If the cell is a HYPERLINK() formula, extract URL
                            f_el = link_cell.find("main:f", ns)
                            f_text = get_text(f_el).strip() if f_el is not None else ""
                            if f_el is not None and f_el.attrib.get("t") == "shared":
                                si = f_el.attrib.get("si")
                                if si is not None and not f_text and si in shared_formulas:
                                    template, base_row = shared_formulas[si]
                                    # Adjust same-row references from base_row -> row_number.
                                    f_text = re.sub(
                                        rf"(\$?[A-Za-z]{{1,3}}\$?){base_row}\b",
                                        rf"\\g<1>{row_number}",
                                        template,
                                    )

                            if f_text:
                                link_val = try_extract_hyperlink_from_formula(
                                    f_text,
                                    current_row_number=row_number,
                                    cells_by_col=cells_by_col,
                                    shared=shared,
                                    fallback_work_id=str(work_id),
                                )

                            # 3) Fallback: raw cell value (may be 0 for formulas)
                            if not link_val:
                                link_val = read_cell_value(link_cell, shared).strip()

                    # Filter common bogus cached values from formulas
                    if link_val in {"0", "0.0", "0.0.0.0"}:
                        link_val = ""

                    if link_val:
                        result[work_id] = link_val

                return result
        except Exception:
            return {}

    def search(self, query, top_k=5):
        """
        Returns top_k similar bugs for a given query
        """

        # Lazy-load the model on first search.
        model = get_model(self._get_local_model_dir())

        # Encode query
        query_embedding = model.encode([query], normalize_embeddings=True)

        # Ensure L2 normalized regardless of model settings.
        if USE_PRENORMALIZED:
            query_embedding = l2_normalize(np.asarray(query_embedding))

        # Fully vectorized similarity (works with memmap and EmbeddingsStore)
        # scores shape: (1, N) -> flatten to (N,)
        emb_mat = np.asarray(self.embeddings)
        similarities = (np.asarray(query_embedding) @ emb_mat.T).ravel()

        # -----------------------------
        # Hybrid scoring: semantic + keyword boosts (metadata-only)
        # -----------------------------
        q = str(query).strip()
        q_lower = q.lower()

        try:
            meta_len = int(len(self.metadata))
        except Exception:
            meta_len = int(len(similarities))

        keyword_scores = np.zeros(int(len(similarities)), dtype=float)
        final_scores = np.zeros(int(len(similarities)), dtype=float)

        for i in range(int(len(similarities))):
            semantic_score = float(similarities[i])

            # Safely access metadata row (pandas DataFrame expected).
            try:
                meta = self.metadata.iloc[i] if hasattr(self.metadata, "iloc") else self.metadata[i]
            except Exception:
                meta = None

            wid = ""
            title = ""
            semantic_text = ""
            if meta is not None:
                try:
                    wid = str(meta.get("WorkItemId", ""))
                except Exception:
                    try:
                        wid = str(meta["WorkItemId"])  # type: ignore[index]
                    except Exception:
                        wid = ""

                try:
                    title = str(meta.get("Title", ""))
                except Exception:
                    title = ""

                # May not exist in metadata; guard with get(default).
                try:
                    semantic_text = str(meta.get("SemanticText", ""))
                except Exception:
                    semantic_text = ""

            keyword_score = 0.0
            if q and wid and q == wid:
                keyword_score += 1.0

            if q_lower and title and q_lower in str(title).lower():
                keyword_score += 0.5

            if q_lower and semantic_text and q_lower in str(semantic_text).lower():
                keyword_score += 0.3

            keyword_scores[i] = float(keyword_score)
            final_scores[i] = (0.75 * semantic_score) + (0.25 * float(keyword_score))

        # Get top results by final score
        top_indices = final_scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row = self.metadata.iloc[idx]
            bug_url = row.get("BugUrl", "")
            link_val = bug_url if str(bug_url).strip() else row.get("link", row.get("Link", ""))
            try:
                link_val = "" if str(link_val).strip() in {"0", "0.0", "0.0.0.0", "nan", "None"} else str(link_val)
            except Exception:
                link_val = ""

            wid_raw = row.get("WorkItemId", "")
            wid_str = str(wid_raw).strip()
            try:
                wid_int = int(wid_str) if wid_str.isdigit() else int(float(wid_str))
            except Exception:
                wid_int = 0

            semantic_score = float(similarities[idx])
            kw = float(keyword_scores[idx])
            final_score = float(final_scores[idx])
            print(f"[HYBRID] ID={wid_str} semantic={semantic_score:.3f} keyword={kw:.3f} final={final_score:.3f}")
            results.append({
                "WorkItemId": wid_int,
                "Title": row["Title"],
                "BugUrl": link_val,
                "link": link_val,
                "Score": final_score,
                # Explainability fields (UI-only consumers). Keep `Score` for backwards compatibility.
                "SemanticScore": semantic_score,
                "KeywordScore": kw,
                "FinalScore": final_score,
            })

        return results
