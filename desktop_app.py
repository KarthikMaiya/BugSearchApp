"""Tkinter desktop app for semantic bug search.

Run:
    python desktop_app.py
"""

from __future__ import annotations

import queue
import os
import sys
import threading
import time
import traceback
import webbrowser
import logging
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from embedding_text import build_embedding_text

import runtime_paths

if TYPE_CHECKING:
    import pandas as pd


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


def _resource_base_path() -> Path:
    """Base directory for bundled read-only resources.

    In a frozen app, PyInstaller exposes sys._MEIPASS; in one-folder mode it's
    typically the same directory as the executable, but we don't assume that.
    """

    try:
        if _is_frozen() and hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    except Exception:
        pass
    return _app_base_path()


def _data_dir() -> Path:
    return runtime_paths.get_runtime_data_dir()


def _ensure_runtime_data_layout() -> None:
    """Ensure the runtime ./data layout exists and migrate legacy files.

    The app reads/writes embeddings + metadata inside:
      base_path / 'data'

    For backwards compatibility (older dev layouts), if embeddings were stored
    under ./embeddings we copy them into ./data on first run.
    """

    import shutil

    base = _app_base_path()
    res_base = _resource_base_path()
    data = _data_dir()
    data.mkdir(parents=True, exist_ok=True)

    legacy_dir = base / "embeddings"
    legacy_res_dir = res_base / "embeddings"

    migrations: list[tuple[Path, Path]] = [
        (legacy_dir / "bug_embeddings.npy", data / "bug_embeddings.npy"),
        (legacy_dir / "bug_metadata.csv", data / "bug_metadata.csv"),
    ]

    # If running frozen and files ended up under sys._MEIPASS/embeddings, also migrate.
    migrations += [
        (legacy_res_dir / "bug_embeddings.npy", data / "bug_embeddings.npy"),
        (legacy_res_dir / "bug_metadata.csv", data / "bug_metadata.csv"),
    ]

    for src, dst in migrations:
        try:
            if dst.exists():
                continue
            if src.exists() and src.is_file():
                shutil.copy2(src, dst)
                logging.info("[BOOT] Migrated %s -> %s", src, dst)
        except Exception as exc:  # noqa: BLE001
            logging.exception("[BOOT] Migration failed (%s -> %s): %s", src, dst, exc)

    # Ensure fingerprints file exists (refresh writes atomically later).
    fp = data / "bug_fingerprints.json"
    if not fp.exists():
        # Copy bundled template if present, else create empty.
        try:
            bundled_fp = (res_base / "data" / "bug_fingerprints.json")
            if bundled_fp.exists() and bundled_fp.is_file():
                shutil.copy2(bundled_fp, fp)
            else:
                fp.write_text("{}\n", encoding="utf-8")
        except Exception:
            pass


def _atomic_write_embeddings_and_metadata(
    *,
    embeddings_path: Path,
    metadata_path: Path,
    write_embeddings: Callable[[Path], None],
    write_metadata: Callable[[Path], None],
    retries: int = 3,
    retry_delay_seconds: float = 0.25,
) -> None:
    """Atomically write embeddings + metadata to disk.

    On Windows, writing directly to an in-use CSV (e.g., open in Excel) can raise
    PermissionError. This helper writes to temp files first, then performs an
    atomic replace with a lightweight rollback strategy to avoid leaving
    embeddings/metadata out of sync.
    """

    import os

    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)

    emb_parent = embeddings_path.parent
    meta_parent = metadata_path.parent
    emb_parent.mkdir(parents=True, exist_ok=True)
    meta_parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None

    for _attempt in range(max(1, int(retries))):
        tmp_emb: Path | None = None
        tmp_meta: Path | None = None
        emb_bak: Path | None = None
        meta_bak: Path | None = None

        try:
            # 1) Write temps first (never touch the live files yet).
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=str(emb_parent),
                prefix=embeddings_path.stem + "_",
                suffix=embeddings_path.suffix or ".npy",
            ) as f:
                tmp_emb = Path(f.name)

            # Ensure the handle is closed before numpy writes.
            write_embeddings(tmp_emb)

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(meta_parent),
                prefix=metadata_path.stem + "_",
                suffix=metadata_path.suffix or ".csv",
            ) as f:
                tmp_meta = Path(f.name)

            write_metadata(tmp_meta)

            # 2) Commit both files with rollback safety.
            emb_bak = embeddings_path.with_suffix((embeddings_path.suffix or ".npy") + ".bak")
            meta_bak = metadata_path.with_suffix((metadata_path.suffix or ".csv") + ".bak")

            # Remove stale backups if any.
            try:
                emb_bak.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                meta_bak.unlink(missing_ok=True)
            except Exception:
                pass

            # Move live files to backups (cheap rename). This also surfaces locks.
            if embeddings_path.exists():
                os.replace(str(embeddings_path), str(emb_bak))
            if metadata_path.exists():
                os.replace(str(metadata_path), str(meta_bak))

            # Replace live paths with newly written temps.
            os.replace(str(tmp_emb), str(embeddings_path))
            os.replace(str(tmp_meta), str(metadata_path))

            # Success: clean up backups.
            try:
                emb_bak.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                meta_bak.unlink(missing_ok=True)
            except Exception:
                pass

            return

        except PermissionError as exc:
            last_exc = exc

            # Best-effort rollback if we moved originals to .bak.
            try:
                if emb_bak is not None and emb_bak.exists():
                    os.replace(str(emb_bak), str(embeddings_path))
            except Exception:
                pass
            try:
                if meta_bak is not None and meta_bak.exists():
                    os.replace(str(meta_bak), str(metadata_path))
            except Exception:
                pass

            time.sleep(float(retry_delay_seconds))

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            raise

        finally:
            # Always try to clean up temps.
            for p in (tmp_emb, tmp_meta):
                if p is None:
                    continue
                try:
                    if p.exists() and p not in {embeddings_path, metadata_path}:
                        p.unlink(missing_ok=True)
                except Exception:
                    pass

    # Retries exhausted.
    if isinstance(last_exc, PermissionError):
        raise PermissionError(
            "Permission denied while updating refresh artifacts. "
            f"Close '{metadata_path.name}' if it is open in Excel (or any other program) "
            "and try Refresh again."
        ) from last_exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to save refresh artifacts")


@dataclass(frozen=True)
class BugResult:
    work_item_id: int
    title: str
    link: str
    semantic_score: float
    keyword_score: float
    final_score: float


class BugSearchService:
    """Thin wrapper around SemanticBugSearch to keep UI separate from search logic."""

    def __init__(self) -> None:
        # Import here (not at module import time) to keep app startup snappy.
        from semantic_engine import SemanticBugSearch

        data_dir = _data_dir()
        self._embeddings_path = data_dir / "bug_embeddings.npy"
        self._metadata_path = data_dir / "bug_metadata.csv"
        self._source_xlsx_path = data_dir / "bugs_semantic.xlsx"

        self._engine = SemanticBugSearch(
            embeddings_path=str(self._embeddings_path),
            metadata_path=str(self._metadata_path),
            source_xlsx_path=str(self._source_xlsx_path) if self._source_xlsx_path.is_file() else "",
        )

    @classmethod
    def from_loaded(
        cls,
        embeddings: Any,
        metadata: Any,
        *,
        base_dir: Path,
        model_name: str = "all-MiniLM-L6-v2",
        source_xlsx_path: Path | None = None,
    ) -> "BugSearchService":
        """Create a service from already loaded embeddings/metadata.

        This avoids re-loading from disk and prevents multiple parallel embedding
        states from being created at startup.
        """

        from semantic_engine import SemanticBugSearch

        data_dir = _data_dir()

        if source_xlsx_path is None:
            source_xlsx_path = data_dir / "bugs_semantic.xlsx"

        emb_rows = int(getattr(embeddings, "shape", [len(embeddings)])[0])
        meta_rows = int(len(metadata))
        if emb_rows != meta_rows:
            raise ValueError(
                "Embeddings/metadata are out of sync. "
                f"Embeddings rows={emb_rows}, metadata rows={meta_rows}."
            )

        engine = SemanticBugSearch.__new__(SemanticBugSearch)
        engine.embeddings = embeddings
        engine.metadata = metadata
        engine._model_name = model_name
        engine._local_model_dir = None
        try:
            engine._embeddings_path = (data_dir / "bug_embeddings.npy").resolve()
            engine._metadata_path = (data_dir / "bug_metadata.csv").resolve()
            engine._source_xlsx_path = source_xlsx_path.resolve() if source_xlsx_path is not None else None
        except Exception:
            pass
        try:
            engine._try_augment_links_from_xlsx(str(source_xlsx_path))
        except Exception:
            pass

        service = cls.__new__(cls)
        service._engine = engine
        service._embeddings_path = data_dir / "bug_embeddings.npy"
        service._metadata_path = data_dir / "bug_metadata.csv"
        service._source_xlsx_path = source_xlsx_path
        return service

    def is_model_loaded(self) -> bool:
        from semantic_engine import is_model_loaded

        return is_model_loaded()

    def is_model_loading(self) -> bool:
        from semantic_engine import is_model_loading

        return is_model_loading()

    def load_model(self) -> None:
        # Triggers lazy-load of the local model directory (offline-only).
        self._engine.load_model()

    def has_local_model_files(self) -> bool:
        # The app enforces local/offline model loading from ./models/<model_name>/.
        return (runtime_paths.get_model_dir() / "all-MiniLM-L6-v2").is_dir()

    def search(self, query: str, top_k: int) -> List[BugResult]:
        raw_results = self._engine.search(query, top_k=top_k)
        results: List[BugResult] = []
        for r in raw_results:
            final_score = float(r.get("FinalScore", r.get("Score", 0.0) or 0.0))
            results.append(
                BugResult(
                    work_item_id=int(r.get("WorkItemId", 0) or 0),
                    title=str(r.get("Title", "") or ""),
                    link=str(r.get("BugUrl", r.get("link", r.get("Link", "")))),
                    semantic_score=float(r.get("SemanticScore", 0.0) or 0.0),
                    keyword_score=float(r.get("KeywordScore", 0.0) or 0.0),
                    final_score=final_score,
                )
            )
        return results

    def reload_index(self) -> None:
        """Reload embeddings + metadata from disk.

        This is used after Refresh, which atomically replaces the artifacts.
        """

        self._engine.reload_index()

    def replace_index(self, embeddings: Any, metadata: Any) -> None:
        """Replace the in-memory search index.

        Used by the Refresh workflow to swap in newly updated embeddings/metadata
        without reloading from disk.
        """
        self._engine.embeddings = embeddings
        self._engine.metadata = metadata

    def get_index(self) -> tuple[Any, Any]:
        """Return the current in-memory embeddings and metadata."""
        return self._engine.embeddings, self._engine.metadata


class EmbeddingsStore:
    """Stable in-memory container for embeddings.

    Numpy arrays are fixed-size; "appending" produces a new array object.
    This wrapper keeps a stable object identity (`id(self)`) while allowing the
    underlying array to grow.
    """

    def __init__(self, array: Any) -> None:
        self._array = array

    @property
    def array(self) -> Any:
        return self._array

    @array.setter
    def array(self, value: Any) -> None:
        self._array = value

    @property
    def shape(self) -> Any:
        return getattr(self._array, "shape", (len(self._array),))

    def __len__(self) -> int:
        try:
            return int(self.shape[0])
        except Exception:
            return int(len(self._array))

    def __array__(self, dtype: Any | None = None) -> Any:  # numpy hook
        import numpy as np

        return np.asarray(self._array, dtype=dtype)

    def __getitem__(self, idx: Any) -> Any:
        return self._array[idx]

    def __setitem__(self, idx: Any, value: Any) -> None:
        self._array[idx] = value


class DesktopBugSearchApp:
    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self.root = root
        self._root.title("BugSearchApp — Semantic Bug Search")
        self._root.geometry("1000x600")
        self._root.minsize(1000, 600)

        # Config (created/loaded on startup)
        self._project_root = self._get_project_root_dir()
        self.config: dict[str, Any] = self._load_or_create_config()

        self._font = ("Segoe UI", 10)
        self._header_font = ("Segoe UI", 10, "bold")
        self._title_font = ("Segoe UI", 16, "bold")
        self._subtle_font = ("Segoe UI", 9)

        self.last_index_time: datetime | None = None
        self.last_indexed_time: datetime | None = None
        self.last_refresh_stats: dict[str, int] | None = None
        self._current_excel_path: Path | None = None

        self.metadata_path = (_data_dir() / "bug_metadata.csv").resolve()
        self.metadata_df: "pd.DataFrame | None" = None
        self.total_indexed: int = 0

        # Service is loaded once at startup
        self._service: Optional[BugSearchService] = None

        # If a user clicks Search before the ML model is ready, we keep the most
        # recent request here and auto-run it once initialization completes.
        self._pending_search: tuple[str, int] | None = None

        # Threading / communication
        self._result_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._active_worker: Optional[threading.Thread] = None

        # Single authoritative in-memory index state.
        self.embeddings: EmbeddingsStore | None = None
        self.metadata: Any = None
        self._embeddings_obj_id: int | None = None

        # State vars
        self._top_k_var = tk.IntVar(value=5)
        # Bottom status bar vars
        self.status_var = tk.StringVar(value="Ready")
        self.state_var = tk.StringVar(value="Ready")
        self._bug_count_var = tk.StringVar(value="Total Bugs in Database: —")
        self._found_results_var = tk.StringVar(value="Found — results")

        # Data Source Status vars
        self._ds_source_var = tk.StringVar(value="—")
        self._ds_excel_update_var = tk.StringVar(value="—")
        self._ds_last_indexed_var = tk.StringVar(value="—")
        self._ds_total_indexed_var = tk.StringVar(value="—")

        # Info popover text
        self._info_tooltip_text_var: tk.StringVar | None = None

        # Public UI refs
        self.bug_count_label: ttk.Label | None = None
        self._status_label: ttk.Label | None = None

        # Toolbar info popover
        self._info_btn: ttk.Button | None = None
        self._info_popover: tk.Toplevel | None = None
        self._info_popover_hide_after: str | None = None

        # Diagnostics
        self._diagnostics_btn: ttk.Button | None = None
        self._diagnostics_thread: threading.Thread | None = None

        # Statusbar progress indicator
        self._progress: ttk.Progressbar | None = None

        # Search placeholder
        self._placeholder_text = "Paste bug description, error, or steps to reproduce..."
        self._placeholder_active = False

        # Treeview UX
        self._tree_menu: tk.Menu | None = None
        self._hover_row_id: str | None = None

        # Treeview sort direction toggles per column (UI-only state).
        self._tree_sort_desc: dict[str, bool] = {}

        # Splash UI (shown while model loads)
        self._splash: tk.Toplevel | None = None
        self._splash_label: ttk.Label | None = None
        self._splash_phase: str = ""

        # Show splash immediately; we will deiconify the main window when ready.
        self._root.withdraw()
        self._show_splash()

        self._set_splash_text("Loading UI...")

        self._build_ui()

        # Load the search index (embeddings + metadata) first.
        # The ML model is NOT loaded at startup.
        self._load_index_async()

        # Periodically process results from worker threads.
        self._root.after(100, self._poll_queue)

    def _get_project_root_dir(self) -> Path:
        """Return the directory treated as the app/project root.

        - In dev runs, this is the folder containing this file.
        - In a frozen (PyInstaller) app, this is the AppData root so config.json can be
          created/edited by users without requiring admin rights.
        """

        return runtime_paths.get_appdata_root()

    def _load_or_create_config(self) -> dict[str, Any]:
        """Load config.json from project root, creating it if missing."""

        config_path = self._project_root / "config.json"
        default_config: dict[str, Any] = {
            "excel_source": "local",
            "excel_path": "data/bugs_live.xlsx",
        }

        if not config_path.is_file():
            try:
                config_path.write_text(
                    json.dumps(default_config, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                # If we cannot write a config file, fall back to defaults in-memory.
                logging.exception("[CONFIG] Failed to create config.json at %s: %s", config_path, exc)
                return dict(default_config)

        try:
            raw = config_path.read_text(encoding="utf-8")
            parsed = json.loads(raw) if raw.strip() else {}
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception as exc:  # noqa: BLE001
            # If config is invalid/corrupted, keep the app running with defaults.
            logging.exception("[CONFIG] Failed to read/parse config.json at %s: %s", config_path, exc)
            parsed = {}

        merged = dict(default_config)
        merged.update(parsed)
        return merged

    def _save_config(self) -> None:
        """Persist current config to config.json.

        Avoid logging config contents because it may include secrets.
        """

        config_path = (self._project_root / "config.json").resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
        payload = json.dumps(self.config, indent=2, ensure_ascii=False) + "\n"
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(config_path)

    def get_latest_excel(self) -> str:
        source = self.config.get("excel_source", "local")
        path = self.config.get("excel_path")

        logging.info("[CONFIG] Excel source: %s", source)
        logging.info("[CONFIG] Excel path: %s", path)

        if source != "local":
            raise ValueError(f"Unsupported excel_source: {source!r}. Only 'local' is supported for now.")

        if not path or not str(path).strip():
            raise FileNotFoundError("Excel path not set in config.json (excel_path).")

        excel_path = Path(str(path))
        if not excel_path.is_absolute():
            excel_path = (self._project_root / excel_path)

        excel_path = excel_path.resolve()
        if not excel_path.is_file():
            raise FileNotFoundError(f"Configured Excel file does not exist: {excel_path}")

        return str(excel_path)

    def _show_splash(self) -> None:
        splash = tk.Toplevel(self._root)
        splash.title("Loading")
        splash.resizable(False, False)
        splash.transient(self._root)

        # Keep it on top so the user sees it immediately.
        try:
            splash.attributes("-topmost", True)
        except tk.TclError:
            pass

        frame = ttk.Frame(splash, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")

        label = ttk.Label(frame, text="Loading...")
        label.grid(row=0, column=0, sticky="w")

        # Center on screen
        splash.update_idletasks()
        w = splash.winfo_width()
        h = splash.winfo_height()
        x = (splash.winfo_screenwidth() // 2) - (w // 2)
        y = (splash.winfo_screenheight() // 2) - (h // 2)
        splash.geometry(f"{w}x{h}+{x}+{y}")

        self._splash = splash
        self._splash_label = label

    def _set_splash_text(self, text: str) -> None:
        if self._splash_label is None:
            return
        try:
            self._splash_label.configure(text=text)
        except tk.TclError:
            pass

    def _close_splash(self) -> None:
        if self._splash is not None:
            try:
                self._splash.destroy()
            except tk.TclError:
                pass

        self._splash = None
        self._splash_label = None

        # Show main window
        try:
            self._root.deiconify()
            self._root.lift()
            self._center_root_window()
        except tk.TclError:
            pass

    def _center_root_window(self) -> None:
        """Center the main window on the current screen."""

        try:
            self._root.update_idletasks()
            w = self._root.winfo_width()
            h = self._root.winfo_height()
            sw = self._root.winfo_screenwidth()
            sh = self._root.winfo_screenheight()
            x = max(0, (sw // 2) - (w // 2))
            y = max(0, (sh // 2) - (h // 2))
            self._root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    # -------------------- UI construction --------------------

    def _build_ui(self) -> None:
        style = ttk.Style(self._root)

        # Palette
        bg_root = "#f4f6f8"
        bg_surface = "#ffffff"
        text_primary = "#111827"
        text_secondary = "#6b7280"
        accent = "#2563eb"
        shadow = "#cbd5e1"

        # Prefer a theme that respects custom colors.
        for theme in ("clam", "vista"):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue

        # Root background (best-effort across ttk themes)
        try:
            self._root.configure(background=bg_root)
        except Exception:
            pass

        style.configure("App.TFrame", background=bg_root)
        style.configure("Toolbar.TFrame", background=bg_surface)
        style.configure("Statusbar.TFrame", background=bg_surface)
        style.configure("Surface.TFrame", background=bg_surface)

        style.configure("TLabel", font=self._font)
        style.configure("Title.TLabel", font=self._title_font, foreground=text_primary, background=bg_surface)
        style.configure("ToolbarTitle.TLabel", font=("Segoe UI", 12, "bold"), foreground=text_primary, background=bg_surface)
        style.configure("Body.TLabel", font=self._font, foreground=text_primary, background=bg_root)
        style.configure("Secondary.TLabel", font=self._font, foreground=text_secondary)
        style.configure("StatusLeft.TLabel", font=self._subtle_font, foreground=text_primary, background=bg_surface)
        style.configure("StatusRight.TLabel", font=self._subtle_font, foreground=text_secondary, background=bg_surface)

        # Tooltip / hover panel styles (flat white, no default ttk grey)
        info_font = ("Segoe UI", 9)
        style.configure("Info.TFrame", background=bg_surface)
        style.configure("InfoKey.TLabel", font=info_font, foreground=text_secondary, background=bg_surface)
        style.configure("InfoVal.TLabel", font=info_font, foreground=text_primary, background=bg_surface)

        # Toolbar info icon button (subtle, flat, non-accent)
        style.configure(
            "InfoIcon.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(6, 4),
            background=bg_surface,
            foreground=text_secondary,
            borderwidth=0,
            relief="flat",
        )
        try:
            style.map(
                "InfoIcon.TButton",
                background=[("active", "#f3f4f6"), ("pressed", "#e5e7eb")],
                foreground=[("active", text_primary)],
            )
        except Exception:
            pass

        style.configure("TButton", font=self._font)
        style.configure(
            "Accent.TButton",
            font=self._font,
            padding=(12, 8),
            background=accent,
            foreground="#ffffff",
            borderwidth=0,
        )
        try:
            style.map(
                "Accent.TButton",
                background=[("active", "#1d4ed8"), ("pressed", "#1e40af")],
                foreground=[("disabled", "#e5e7eb")],
            )
        except Exception:
            pass

        style.configure("Treeview", font=self._font, rowheight=32, background=bg_surface, fieldbackground=bg_surface)
        style.configure(
            "Treeview.Heading",
            font=self._header_font,
            background=bg_root,
            foreground=text_primary,
            relief="flat",
        )
        try:
            style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", text_primary)])
        except Exception:
            pass

        # Status label variants (used by set_status)
        style.configure("StatusNeutral.TLabel", font=self._subtle_font, foreground=text_secondary, background=bg_surface)
        style.configure("StatusInProgress.TLabel", font=self._subtle_font, foreground=accent, background=bg_surface)
        style.configure("StatusSuccess.TLabel", font=self._subtle_font, foreground="#059669", background=bg_surface)
        style.configure("StatusError.TLabel", font=self._subtle_font, foreground="#b91c1c", background=bg_surface)

        container = ttk.Frame(self._root, padding=0, style="App.TFrame")
        container.grid(row=0, column=0, sticky="nsew")

        self._root.grid_rowconfigure(0, weight=1)
        self._root.grid_columnconfigure(0, weight=1)

        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(2, weight=1)

        # ROW 0: Top toolbar
        toolbar = ttk.Frame(container, style="Toolbar.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.grid_columnconfigure(0, weight=1)
        toolbar.grid_propagate(False)
        try:
            toolbar.configure(height=48)
        except Exception:
            pass

        toolbar_inner = ttk.Frame(toolbar, style="Toolbar.TFrame", padding=(16, 10))
        toolbar_inner.grid(row=0, column=0, sticky="nsew")
        toolbar_inner.grid_columnconfigure(0, weight=1)

        ttk.Label(toolbar_inner, text="BugSearchApp — Semantic Bug Search", style="ToolbarTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        # Top-right toolbar buttons (right-aligned): Diagnostics, Info
        self._diagnostics_btn = ttk.Button(
            toolbar_inner,
            text="Run Diagnostics",
            command=self.run_diagnostics,
        )
        self._diagnostics_btn.grid(row=0, column=1, sticky="e", padx=(0, 8))
        try:
            self._diagnostics_btn.configure(cursor="hand2")
        except Exception:
            pass

        self._info_btn = ttk.Button(toolbar_inner, text="ⓘ", width=3, style="InfoIcon.TButton", takefocus=False)
        self._info_btn.grid(row=0, column=2, sticky="e")
        self._info_btn.configure(command=lambda: None)
        self._info_btn.bind("<Enter>", lambda _e: self._show_info_popover())
        self._info_btn.bind("<Leave>", lambda _e: self._schedule_hide_info_popover())

        ttk.Separator(container, orient="horizontal").grid(row=1, column=0, sticky="ew")

        # ROW 1: Search area (compact)
        search_area = ttk.Frame(container, style="App.TFrame", padding=(16, 14))
        search_area.grid(row=2, column=0, sticky="nsew")
        search_area.grid_columnconfigure(0, weight=1)
        search_area.grid_rowconfigure(0, weight=0)
        search_area.grid_rowconfigure(1, weight=0)
        search_area.grid_rowconfigure(2, weight=1)

        input_frame = ttk.Frame(search_area, style="Surface.TFrame", padding=12)
        input_frame.grid(row=0, column=0, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_rowconfigure(0, weight=1)

        self._query_text = tk.Text(
            input_frame,
            height=6,
            wrap="word",
            font=self._font,
            relief="flat",
            borderwidth=0,
            background=bg_surface,
            foreground=text_primary,
            insertbackground=text_primary,
            highlightthickness=1,
            highlightbackground="#e5e7eb",
            highlightcolor=accent,
        )
        self._query_text.grid(row=0, column=0, sticky="nsew")

        # Placeholder UX + Enter-to-search
        self._apply_search_placeholder()
        self._query_text.bind("<FocusIn>", self._on_search_focus_in)
        self._query_text.bind("<FocusOut>", self._on_search_focus_out)
        self._query_text.bind("<Return>", self._on_search_return)

        query_scroll = ttk.Scrollbar(input_frame, command=self._query_text.yview)
        query_scroll.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self._query_text.configure(yscrollcommand=query_scroll.set)

        controls = ttk.Frame(search_area, style="App.TFrame", padding=(0, 12, 0, 0))
        controls.grid(row=1, column=0, sticky="ew")
        controls.grid_columnconfigure(6, weight=1)

        ttk.Label(controls, text="Results:", style="Secondary.TLabel", background=bg_root).grid(row=0, column=0, sticky="w")

        self._topk_spin = ttk.Spinbox(
            controls,
            from_=1,
            to=20,
            textvariable=self._top_k_var,
            width=6,
            font=self._font,
        )
        self._topk_spin.grid(row=0, column=1, sticky="w", padx=(10, 18))

        self._search_btn = ttk.Button(
            controls,
            text="🔍  Search",
            command=self._on_search_clicked,
            style="Accent.TButton",
        )
        self._search_btn.grid(row=0, column=2, sticky="w", padx=(0, 10))
        try:
            self._search_btn.configure(cursor="hand2")
        except Exception:
            pass

        self._refresh_btn = ttk.Button(
            controls,
            text="⟳  Refresh",
            command=self._on_refresh_clicked,
            style="Accent.TButton",
        )
        self._refresh_btn.grid(row=0, column=3, sticky="w")
        try:
            self._refresh_btn.configure(cursor="hand2")
        except Exception:
            pass

        # Refresh pulls from Azure DevOps using environment/.env configuration.
        self._set_refresh_enabled(True)

        # ROW 2: Results table occupies all remaining space
        results_frame = ttk.Frame(search_area, style="Surface.TFrame", padding=12)
        results_frame.grid(row=2, column=0, sticky="nsew", pady=(14, 0))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            results_frame,
            columns=("Rank", "Match", "WorkItemId", "Title", "Link"),
            show="headings",
        )
        self._tree.grid(row=0, column=0, sticky="nsew")

        # Heading-click sorting is UI-only and does not affect search ranking unless user clicks.
        self._tree.heading("Rank", text="#", anchor="center", command=lambda: self._sort_tree("Rank"))
        self._tree.heading("Match", text="Match", anchor="center", command=lambda: self._sort_tree("Match"))
        self._tree.heading("WorkItemId", text="WorkItemId", anchor="center", command=lambda: self._sort_tree("WorkItemId"))
        self._tree.heading("Title", text="Title", command=lambda: self._sort_tree("Title"))
        self._tree.heading("Link", text="Link", command=lambda: self._sort_tree("Link"))

        self._tree.column("Rank", width=50, anchor="center", stretch=False)
        self._tree.column("Match", width=80, anchor="center", stretch=False)
        self._tree.column("WorkItemId", width=120, anchor="center", stretch=False)
        self._tree.column("Title", width=520, anchor="w", stretch=True)
        self._tree.column("Link", width=280, anchor="w", stretch=True)

        self._tree.tag_configure("odd", background="#F6F8FB")
        self._tree.tag_configure("hover", background="#EAF2FF")

        self._tree.bind("<Button-1>", self._on_tree_click)
        self._tree.bind("<Double-1>", self._on_result_double_click)
        self._tree.bind("<Motion>", self._on_tree_motion)

        results_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self._tree.yview)
        results_scroll.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self._tree.configure(yscrollcommand=results_scroll.set)

        # Auto column sizing + professional table UX
        self._tree.bind("<Configure>", self._resize_tree_columns)
        self._tree.bind("<Motion>", self._on_tree_motion)
        self._tree.bind("<Leave>", self._on_tree_leave)
        self._tree.bind("<Double-1>", self._on_result_double_click)
        self._tree.bind("<Button-3>", self._on_tree_right_click)

        # Context menu
        self._tree_menu = tk.Menu(self._root, tearoff=0)
        self._tree_menu.add_command(label="Open bug", command=self._open_selected_bug)
        self._tree_menu.add_separator()
        self._tree_menu.add_command(label="Copy WorkItemId", command=lambda: self._copy_selected_cell("WorkItemId"))
        self._tree_menu.add_command(label="Copy Title", command=lambda: self._copy_selected_cell("Title"))
        self._tree_menu.add_command(label="Copy Link", command=lambda: self._copy_selected_cell("Link"))

        # Subtle separator above status bar
        ttk.Separator(container, orient="horizontal").grid(row=3, column=0, sticky="ew")

        # ROW 3: Status bar
        statusbar = ttk.Frame(container, style="Statusbar.TFrame")
        statusbar.grid(row=4, column=0, sticky="ew")
        statusbar.grid_propagate(False)
        try:
            statusbar.configure(height=34)
        except Exception:
            pass
        statusbar.grid_columnconfigure(0, weight=1)

        status_inner = ttk.Frame(statusbar, style="Statusbar.TFrame", padding=(16, 8))
        status_inner.grid(row=0, column=0, sticky="nsew")
        status_inner.grid_columnconfigure(0, weight=1)

        ttk.Label(status_inner, textvariable=self.status_var, style="StatusLeft.TLabel").grid(row=0, column=0, sticky="w")

        self._progress = ttk.Progressbar(status_inner, mode="indeterminate")
        self._progress.grid(row=0, column=1, sticky="e")
        try:
            self._progress.grid_remove()
        except Exception:
            pass

        # Keep a right-side label reference for styling compatibility (not visible in final layout).
        self._status_label = ttk.Label(status_inner, textvariable=self.state_var, style="StatusRight.TLabel", anchor="e")

        # Global keyboard shortcuts
        self._root.bind_all("<Control-r>", lambda _e: self._on_refresh_clicked())
        self._root.bind_all("<Control-R>", lambda _e: self._on_refresh_clicked())
        self._root.bind_all("<Control-f>", lambda _e: self._focus_search_box())
        self._root.bind_all("<Control-F>", lambda _e: self._focus_search_box())
        self._root.bind_all("<Control-c>", self._on_global_copy)
        self._root.bind_all("<Control-C>", self._on_global_copy)

        # Start disabled until engine loads successfully.
        self._set_search_enabled(False)
        self._set_refresh_enabled(False)

        # Initialize label text.
        self.update_bug_count_display()
        self.update_data_source_status(None)

    def _format_dt(self, dt: datetime) -> str:
        return dt.strftime("%d %b %Y %I:%M %p")

    def _shorten_display_path(self, p: Path, max_len: int = 60) -> str:
        s = str(p)
        if len(s) <= max_len:
            return s

        parts = list(p.parts)
        if len(parts) <= 4:
            return s[: max_len - 1] + "…"

        # Windows-friendly shortening: keep drive + first 2 dirs + last filename.
        drive = parts[0]
        head_parts = parts[1:3]
        tail = parts[-1]
        head = "\\".join([drive.rstrip("\\"), *head_parts])
        shortened = f"{head}\\...\\{tail}"
        if len(shortened) <= max_len:
            return shortened
        return "…" + shortened[-(max_len - 1) :]

    def update_data_source_status(self, excel_path: str | Path | None) -> None:
        """Update Data Source Status UI fields (Tk thread only)."""

        # Total bugs indexed
        try:
            if self.metadata_df is not None:
                total = len(self.metadata_df)
            else:
                total = len(self.metadata) if self.metadata is not None else 0
            self._ds_total_indexed_var.set(str(int(total)))
        except Exception:
            self._ds_total_indexed_var.set("—")

        # Last indexed timestamp
        ts = self.last_indexed_time or self.last_index_time
        if ts is None:
            self._ds_last_indexed_var.set("—")
        else:
            try:
                self._ds_last_indexed_var.set(self._format_dt(ts))
            except Exception:
                self._ds_last_indexed_var.set("—")

        # Keep a numeric field available for the dynamic info tooltip.
        try:
            self.total_indexed = int(len(self.metadata_df)) if self.metadata_df is not None else int(total)
        except Exception:
            self.total_indexed = 0

        # Excel path + Excel modified time
        if excel_path is None:
            self._ds_source_var.set("Not available")
            self._ds_excel_update_var.set("—")
            return

        try:
            p = Path(excel_path).resolve()
            if not p.is_file():
                self._ds_source_var.set("Not available")
                self._ds_excel_update_var.set("—")
                return

            self._ds_source_var.set(self._shorten_display_path(p))
            modified_ts = p.stat().st_mtime
            formatted_time = datetime.fromtimestamp(modified_ts).strftime("%d %b %Y %I:%M %p")
            self._ds_excel_update_var.set(formatted_time)
        except Exception:
            self._ds_source_var.set("Not available")
            self._ds_excel_update_var.set("—")

    def reload_metadata(self) -> None:
        """Reload bug_metadata.csv from disk and refresh index statistics."""

        try:
            import pandas as pd

            self.metadata_df = pd.read_csv(self.metadata_path, dtype={"WorkItemId": str})
            self.total_indexed = int(len(self.metadata_df))
        except Exception:
            # Keep previous values if reload fails.
            return

        now = datetime.now()
        self.last_indexed_time = now
        self.last_index_time = now
        try:
            self.update_data_source_status(self._current_excel_path)
        except Exception:
            pass

    def build_info_tooltip_text(self) -> str:
        """Build the ℹ️ info panel text dynamically from current state."""

        total = int(self.total_indexed or 0)

        ts = self.last_indexed_time
        last_indexed = "—"
        if ts is not None:
            try:
                last_indexed = self._format_dt(ts)
            except Exception:
                last_indexed = "—"

        if not self.last_refresh_stats:
            return (
                f"Total bugs indexed: {total}\n"
                f"Last indexed: {last_indexed}\n\n"
                "Last refresh: —"
            )

        added = int(self.last_refresh_stats.get("added", 0) or 0)
        updated = int(self.last_refresh_stats.get("updated", 0) or 0)
        skipped = int(self.last_refresh_stats.get("skipped", 0) or 0)

        return (
            f"Total bugs indexed: {total}\n"
            f"Last indexed: {last_indexed}\n\n"
            "Last refresh:\n"
            f"  New bugs added: {added}\n"
            f"  Bugs updated: {updated}\n"
            f"  Unchanged/skipped: {skipped}"
        )

    def update_info_tooltip(self) -> None:
        """Refresh the ℹ️ info panel contents in the UI."""

        if self._info_tooltip_text_var is None:
            return
        try:
            self._info_tooltip_text_var.set(self.build_info_tooltip_text())
        except Exception:
            pass

    def _build_info_popover(self) -> None:
        if self._info_popover is not None:
            return

        pop = tk.Toplevel(self._root)
        pop.withdraw()
        pop.overrideredirect(True)
        try:
            pop.attributes("-topmost", True)
        except Exception:
            pass

        # Shadow + border illusion: use the toplevel background as shadow color,
        # and leave a small margin on right/bottom.
        shadow_color = "#cbd5e1"
        try:
            pop.configure(background=shadow_color)
        except Exception:
            pass

        # Inner white panel with 10px internal spacing.
        frame = ttk.Frame(pop, style="Info.TFrame", padding=10)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=(0, 2))
        frame.columnconfigure(0, weight=1)

        self._info_tooltip_text_var = tk.StringVar(value="")
        ttk.Label(
            frame,
            textvariable=self._info_tooltip_text_var,
            style="InfoVal.TLabel",
            takefocus=False,
            justify="left",
            wraplength=520,
        ).grid(row=0, column=0, sticky="w")

        pop.bind("<Enter>", lambda _e: self._cancel_hide_info_popover())
        pop.bind("<Leave>", lambda _e: self._schedule_hide_info_popover())

        self._info_popover = pop

    def _refresh_info_popover_values(self) -> None:
        # Always refresh on show to avoid stale values.
        self.update_info_tooltip()

    def _show_info_popover(self) -> None:
        if self._info_btn is None:
            return
        self._build_info_popover()
        if self._info_popover is None:
            return

        self._cancel_hide_info_popover()
        self._refresh_info_popover_values()

        try:
            bx = self._info_btn.winfo_rootx()
            by = self._info_btn.winfo_rooty()
            bh = self._info_btn.winfo_height()
        except Exception:
            bx, by, bh = 0, 0, 0

        pop = self._info_popover
        try:
            pop.update_idletasks()
            w = pop.winfo_reqwidth()
            h = pop.winfo_reqheight()
            x = bx - w + 24
            y = by + bh + 8
            pop.geometry(f"{w}x{h}+{max(8, x)}+{max(8, y)}")
            pop.deiconify()
        except Exception:
            try:
                pop.deiconify()
            except Exception:
                pass

    def _hide_info_popover(self) -> None:
        if self._info_popover is None:
            return
        try:
            self._info_popover.withdraw()
        except Exception:
            pass

    def _cancel_hide_info_popover(self) -> None:
        if self._info_popover_hide_after is not None:
            try:
                self._root.after_cancel(self._info_popover_hide_after)
            except Exception:
                pass
            self._info_popover_hide_after = None

    def _schedule_hide_info_popover(self) -> None:
        self._cancel_hide_info_popover()
        # Small delay prevents flicker moving between icon and popover.
        try:
            self._info_popover_hide_after = self._root.after(180, self._hide_info_popover)
        except Exception:
            self._hide_info_popover()

    # -------------------- Startup / engine loading --------------------

    def _load_index_async(self) -> None:
        """Load embeddings + metadata without freezing the UI."""
        self.start_progress("Loading search index...")
        self._set_splash_text("Loading search index...")

        def worker() -> None:
            try:
                _ensure_runtime_data_layout()
                # Cold start: construct the service; it loads artifacts from disk.
                # This ensures mmap loading is used when enabled by the engine.
                service = BugSearchService()
                self._result_queue.put(("engine_loaded", service))
            except Exception as exc:  # noqa: BLE001 - show friendly message
                self._result_queue.put(("engine_failed", (exc, traceback.format_exc())))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _init_model_async(self) -> None:
        """Initialize the ML model in the background (first time only)."""

        if self._service is None:
            return
        if self._service.is_model_loaded() or self._service.is_model_loading():
            return

        # If the model directory isn't bundled/present, don't pop an error on startup.
        # We'll show a clear error the first time the user searches.
        if self._pending_search is None and not self._service.has_local_model_files():
            return

        # Model load progress should be visible in the bottom status bar.
        self.start_progress("Loading semantic model...")

        def worker() -> None:
            try:
                self._service.load_model()
                self._result_queue.put(("model_loaded", None))
            except Exception as exc:  # noqa: BLE001
                self._result_queue.put(("model_failed", (exc, traceback.format_exc())))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # -------------------- Search workflow --------------------

    def _on_search_clicked(self) -> None:
        try:
            if str(self._search_btn.cget("state")) == "disabled":
                return
        except Exception:
            pass

        if self._service is None:
            messagebox.showerror("Not ready", "Search engine is not loaded yet.")
            return

        query = self._query_text.get("1.0", "end").strip()
        if not query:
            messagebox.showwarning("Missing query", "Please enter a bug description to search.")
            return

        # Fast path: exact numeric WorkItemId match.
        q = str(query).strip()
        if q.isdigit() and self.metadata is not None:
            try:
                if hasattr(self.metadata, "columns") and "WorkItemId" in self.metadata.columns:
                    work_ids = self.metadata["WorkItemId"].fillna("").astype(str).str.strip()
                    matches = self.metadata[work_ids == q]
                    if not matches.empty:
                        row = matches.iloc[0]
                        title = str(row.get("Title", "") or "")
                        link_val = row.get("BugUrl", row.get("link", row.get("Link", "")))
                        try:
                            link_val = (
                                ""
                                if str(link_val).strip() in {"0", "0.0", "0.0.0.0", "nan", "None"}
                                else str(link_val)
                            )
                        except Exception:
                            link_val = ""

                        self._clear_results()
                        self._render_results(
                            [
                                BugResult(
                                    work_item_id=int(q),
                                    title=title,
                                    link=str(link_val),
                                    semantic_score=1.0,
                                    keyword_score=1.0,
                                    final_score=1.0,
                                )
                            ]
                        )
                        self.set_status("Exact WorkItemId match found.")
                        return
            except Exception:
                # If anything goes wrong, fall back to normal search.
                pass

        try:
            top_k = int(self._top_k_var.get())
        except Exception:
            top_k = 5

        top_k = max(1, min(20, top_k))
        self._top_k_var.set(top_k)

        # Lazy-load ML model on first search.
        if not self._service.is_model_loaded():
            self._pending_search = (query, top_k)
            if self._service.is_model_loading():
                self.start_progress("Loading semantic model...")
                self._set_search_enabled(False)
            else:
                self._clear_results()
                self._set_search_enabled(False)
                self._init_model_async()
            return

        self._clear_results()
        self.start_progress("Searching...")

        def worker(q2: str, k: int, service: BugSearchService) -> None:
            try:
                results = service.search(q2, top_k=k)
                self._result_queue.put(("search_ok", results))
            except Exception as exc:  # noqa: BLE001
                self._result_queue.put(("search_failed", (exc, traceback.format_exc())))

        self._active_worker = threading.Thread(
            target=worker,
            args=(query, top_k, self._service),
            daemon=True,
        )
        self._active_worker.start()

    def _on_refresh_clicked(self) -> None:
        try:
            if str(self._refresh_btn.cget("state")) == "disabled":
                return
        except Exception:
            pass

        self.start_progress("Refreshing data...")

        # Azure incremental refresh does not depend on an Excel path.
        self._current_excel_path = None
        self.update_data_source_status(None)

        # Run refresh orchestration in a background thread; UI updates are posted
        # back to the Tk main thread via the internal queue.
        self._active_worker = threading.Thread(target=self.refresh_data, daemon=True)
        self._active_worker.start()

    # -------------------- Diagnostics --------------------

    def _set_diagnostics_enabled(self, enabled: bool) -> None:
        try:
            if self._diagnostics_btn is None:
                return
            self._diagnostics_btn.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass

    def run_diagnostics(self) -> None:
        """Run local integrity checks without blocking the UI."""

        try:
            if self._diagnostics_btn is not None and str(self._diagnostics_btn.cget("state")) == "disabled":
                return
        except Exception:
            pass

        # Prevent re-entrancy.
        try:
            if self._diagnostics_thread is not None and self._diagnostics_thread.is_alive():
                return
        except Exception:
            pass

        self.set_status("Running diagnostics...")
        self._set_diagnostics_enabled(False)

        def thread_main() -> None:
            results: list[tuple[str, bool, str]] = []
            try:
                results = self._diagnostics_worker()
            except Exception as exc:  # noqa: BLE001
                results = [("Diagnostics worker", False, f"Unexpected error: {exc}")]

            def ui_done() -> None:
                try:
                    self._show_diagnostics_report(results)
                except Exception:
                    pass
                self._set_diagnostics_enabled(True)
                self.set_status("Ready")

            try:
                self._root.after(0, ui_done)
            except Exception:
                # Worst-case fallback: don't crash if Tk is shutting down.
                return

        t = threading.Thread(target=thread_main, daemon=True)
        self._diagnostics_thread = t
        t.start()

    def _diagnostics_worker(self) -> list[tuple[str, bool, str]]:
        """Background diagnostics worker.

        Returns a list of (name, ok, message).
        Must not perform UI operations.
        """

        import pandas as pd
        import numpy as np

        results: list[tuple[str, bool, str]] = []

        def add(name: str, ok: bool, message: str) -> None:
            results.append((str(name), bool(ok), str(message)))

        data_dir = None
        metadata_df = None
        embeddings = None

        # 1) Runtime data directory exists and writable
        try:
            data_dir = runtime_paths.get_runtime_data_dir()
            probe_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    delete=False,
                    dir=str(data_dir),
                    prefix="diag_",
                    suffix=".tmp",
                ) as f:
                    f.write("ok")
                    probe_path = Path(f.name)
                add("Runtime data directory writable", True, str(data_dir))
            finally:
                if probe_path is not None:
                    try:
                        probe_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            add("Runtime data directory writable", False, f"Failed: {exc}")

        # 2) Required files exist
        required_files = [
            "bug_embeddings.npy",
            "bug_metadata.csv",
            "bug_fingerprints.json",
            "refresh_state.json",
        ]
        try:
            dd = Path(data_dir) if data_dir is not None else runtime_paths.get_runtime_data_dir()
            for name in required_files:
                p = (dd / name).resolve()
                ok = p.is_file()
                add(f"Required file: {name}", ok, ("Found" if ok else f"Missing ({p})"))
        except Exception as exc:  # noqa: BLE001
            add("Required runtime files", False, f"Failed: {exc}")

        # 3) Load metadata CSV + validate columns
        try:
            md_path = runtime_paths.get_runtime_file("data/bug_metadata.csv")
            metadata_df = pd.read_csv(md_path, dtype={"WorkItemId": str})

            cols = set([str(c) for c in getattr(metadata_df, "columns", [])])
            missing_cols: list[str] = []
            if not ("BugID" in cols or "WorkItemId" in cols):
                missing_cols.append("BugID or WorkItemId")
            if "Title" not in cols:
                missing_cols.append("Title")
            if "BugUrl" not in cols:
                missing_cols.append("BugUrl")

            if missing_cols:
                add(
                    "Metadata loaded + schema valid",
                    False,
                    f"Loaded {len(metadata_df)} rows, missing column(s): {', '.join(missing_cols)}",
                )
            else:
                add("Metadata loaded + schema valid", True, f"Loaded {len(metadata_df)} rows")
        except Exception as exc:  # noqa: BLE001
            add("Metadata loaded + schema valid", False, f"Failed: {exc}")

        # 4) Load embeddings using semantic_engine loader
        try:
            from semantic_engine import load_embeddings, USE_MMAP

            emb_path = runtime_paths.get_runtime_file("data/bug_embeddings.npy")
            embeddings = load_embeddings(str(emb_path))
            shape = getattr(embeddings, "shape", None)
            add(
                "Embeddings loaded",
                True,
                f"Loaded shape={shape} (USE_MMAP={bool(USE_MMAP)})",
            )
        except Exception as exc:  # noqa: BLE001
            add("Embeddings loaded", False, f"Failed: {exc}")

        # 5) Row alignment
        try:
            if embeddings is None or metadata_df is None:
                raise RuntimeError("Embeddings or metadata not loaded")
            emb_rows = int(getattr(embeddings, "shape", [0])[0])
            meta_rows = int(len(metadata_df))
            ok = emb_rows == meta_rows
            add("Row alignment", ok, f"Embeddings rows={emb_rows}, metadata rows={meta_rows}")
        except Exception as exc:  # noqa: BLE001
            add("Row alignment", False, f"Failed: {exc}")

        # 6) Confirm memory-mapping
        try:
            if embeddings is None:
                raise RuntimeError("Embeddings not loaded")
            ok = isinstance(embeddings, np.memmap)
            add(
                "Embeddings are memory-mapped",
                ok,
                ("np.memmap" if ok else f"Type={type(embeddings).__name__}"),
            )
        except Exception as exc:  # noqa: BLE001
            add("Embeddings are memory-mapped", False, f"Failed: {exc}")

        # 7) Model directory exists (do NOT load model)
        try:
            model_root = runtime_paths.get_model_dir()
            model_dir = (model_root / "all-MiniLM-L6-v2").resolve()
            ok = model_dir.is_dir()
            add("Model directory exists", ok, str(model_dir))
        except Exception as exc:  # noqa: BLE001
            add("Model directory exists", False, f"Failed: {exc}")

        # 8) Refresh state readable + valid timestamp
        try:
            state_path = runtime_paths.get_runtime_file("data/refresh_state.json")
            raw = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
            if not isinstance(raw, dict):
                raise RuntimeError("refresh_state.json missing or invalid JSON")
            ts = str(raw.get("last_refresh_time", "") or "").strip()
            if not ts:
                raise RuntimeError("Missing last_refresh_time")
            # datetime.fromisoformat doesn't accept 'Z'
            normalized = ts.replace("Z", "+00:00")
            _ = datetime.fromisoformat(normalized)
            add("Refresh state readable", True, f"last_refresh_time={ts}")
        except Exception as exc:  # noqa: BLE001
            add("Refresh state readable", False, f"Failed: {exc}")

        # 9) Sample semantic search test
        try:
            if self._service is None:
                raise RuntimeError("Search service is not initialized")
            res = self._service.search("test error", top_k=1)
            ok = bool(res) and len(res) >= 1
            if ok:
                first = res[0]
                add(
                    "Sample semantic search",
                    True,
                    f"Returned 1+ result(s). Top WorkItemId={int(first.work_item_id)}",
                )
            else:
                add("Sample semantic search", False, "Returned 0 results")
        except Exception as exc:  # noqa: BLE001
            add("Sample semantic search", False, f"Failed: {exc}")

        # Logging: always log the full diagnostics payload.
        try:
            payload = [
                {"name": n, "ok": ok, "message": m}
                for (n, ok, m) in results
            ]
            overall_ok = all(bool(x["ok"]) for x in payload) if payload else False
            logging.info(
                "[DIAGNOSTICS] overall_ok=%s results=%s",
                overall_ok,
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception:
            pass

        return results

    def _show_diagnostics_report(self, results: list[tuple[str, bool, str]]) -> None:
        """Render a scrollable diagnostics report dialog (Tk thread only)."""

        overall_ok = all(bool(ok) for (_n, ok, _m) in results) if results else False

        win = tk.Toplevel(self._root)
        win.title("Diagnostics Report")
        win.transient(self._root)
        try:
            win.geometry("760x520")
            win.minsize(640, 420)
        except Exception:
            pass

        outer = ttk.Frame(win, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        text = tk.Text(
            outer,
            wrap="word",
            height=20,
            borderwidth=1,
            relief="solid",
        )
        scroll = ttk.Scrollbar(outer, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)

        text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns", padx=(8, 0))

        try:
            text.tag_configure("ok", foreground="#059669")
            text.tag_configure("fail", foreground="#b91c1c")
        except Exception:
            pass

        for name, ok, message in results:
            prefix = "✔" if ok else "❌"
            line = f"{prefix} {name} — {message}\n"
            tag = "ok" if ok else "fail"
            text.insert("end", line, (tag,))

        text.configure(state="disabled")

        bottom = ttk.Frame(outer)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=0)

        summary_text = "Diagnostics PASSED" if overall_ok else "Diagnostics FAILED"
        summary = ttk.Label(bottom, text=summary_text)
        summary.grid(row=0, column=0, sticky="w")
        try:
            summary.configure(foreground=("#059669" if overall_ok else "#b91c1c"))
        except Exception:
            pass

        close_btn = ttk.Button(bottom, text="Close", command=win.destroy)
        close_btn.grid(row=0, column=1, sticky="e")

        try:
            win.grab_set()
            win.focus_set()
        except Exception:
            pass

    def refresh_data(self) -> None:
        """Orchestrate the refresh workflow.

        Runs in a background thread. UI updates must be posted back to the Tk main
        thread via the internal result queue.
        """

        result: dict[str, object] = {
            "ok": False,
            "processed": 0,
            "stats": None,
            "error": "",
        }

        try:
            self._post_status("Refreshing data...")

            self._post_status("Updating embeddings...")
            # Import here so missing Azure env config doesn't prevent app startup.
            from refresh.refresh_manager import refresh as incremental_refresh

            stats = incremental_refresh(
                self.update_embeddings_from_dataframe,
                self.reload_search_index,
            )

            result["ok"] = True
            result["stats"] = dict(stats or {})
            processed = int((stats or {}).get("added", 0) or 0) + int((stats or {}).get("updated", 0) or 0)
            result["processed"] = processed

        except Exception as exc:  # noqa: BLE001
            result["error"] = f"Refresh failed: {exc}"
            logging.exception("Refresh worker failed")
        finally:
            # Always re-enable UI controls exactly once (on the Tk thread).
            self._result_queue.put(("refresh_done", result))

    def update_embeddings_from_dataframe(self, df: "pd.DataFrame") -> dict[str, int]:
        """Incrementally update embeddings + metadata from a DataFrame.

        Expected input columns (from Azure refresh pipeline):
          - BugID (int/str)
          - Title
          - SemanticText

        This function:
        - updates existing WorkItemId rows in-place
        - appends new WorkItemIds
        - persists updated embeddings/metadata to disk

        Runs in the refresh worker thread.
        """

        # Heavy deps are imported lazily to keep UI startup fast.
        import numpy as np
        import pandas as pd

        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            return {"added": 0, "updated": 0, "skipped": 0}

        required = {"BugID", "Title", "SemanticText"}
        missing = sorted([c for c in required if c not in df.columns])
        if missing:
            raise ValueError(f"Missing required column(s) in refresh DataFrame: {missing}")

        def normalize_work_item_id(raw: object) -> str:
            s = str(raw).strip()
            if s.endswith(".0") and s.replace(".0", "").isdigit():
                s = s[:-2]
            return s

        _ensure_runtime_data_layout()
        data_dir = _data_dir()
        embeddings_path = data_dir / "bug_embeddings.npy"
        metadata_path = data_dir / "bug_metadata.csv"

        if self._service is None:
            raise RuntimeError("Search service is not initialized")

        # If the app is using memory-mapped embeddings for search, we must
        # materialize a writable in-memory copy for refresh updates. This also
        # ensures the .npy file isn't held open (Windows atomic replace).
        if self.embeddings is None or self.metadata is None or not isinstance(self.embeddings, EmbeddingsStore):
            svc_embeddings, svc_metadata = self._service.get_index()

            # Normalize metadata to a DataFrame for the refresh update path.
            if self.metadata is None:
                self.metadata = svc_metadata

            emb_obj = svc_embeddings
            try:
                import numpy as np

                if isinstance(emb_obj, EmbeddingsStore):
                    emb_obj = emb_obj.array

                # If memmap, copy to RAM and close mmap handle.
                if isinstance(emb_obj, np.memmap):
                    emb_in_mem = np.array(emb_obj)
                    try:
                        mm = getattr(emb_obj, "_mmap", None)
                        if mm is not None:
                            mm.close()
                    except Exception:
                        pass
                else:
                    # Avoid copies unless needed for writeability.
                    try:
                        emb_in_mem = emb_obj if getattr(emb_obj, "flags", None) is None or emb_obj.flags.writeable else np.array(emb_obj)
                    except Exception:
                        emb_in_mem = np.array(emb_obj)
            except Exception as exc:
                raise RuntimeError(f"Failed to prepare embeddings for refresh: {exc}") from exc

            self.embeddings = EmbeddingsStore(emb_in_mem)
            self._embeddings_obj_id = id(self.embeddings)

            # Keep the service engine aligned to the writable in-memory state
            # while the refresh worker updates and writes new artifacts.
            self._service.replace_index(self.embeddings, self.metadata)

        embeddings = self.embeddings
        metadata = self.metadata

        # Normalize to the schema expected by build_embedding_text.
        changed_df = df.copy()
        changed_df["WorkItemId"] = changed_df["BugID"].map(normalize_work_item_id)
        changed_df["WorkItemId"] = changed_df["WorkItemId"].astype(str).str.strip()
        changed_df["Title"] = changed_df["Title"].fillna("").astype(str).str.strip()
        changed_df["SemanticText"] = changed_df["SemanticText"].fillna("").astype(str).str.strip()
        changed_df.loc[changed_df["SemanticText"] == "", "SemanticText"] = changed_df.loc[
            changed_df["SemanticText"] == "", "Title"
        ]

        changed_df = changed_df.dropna(subset=["WorkItemId"]).copy()
        changed_df = changed_df[changed_df["WorkItemId"].astype(str).str.strip() != ""].copy()
        changed_df = changed_df[changed_df["SemanticText"].astype(str).str.strip() != ""].copy()
        if changed_df.empty:
            return {"added": 0, "updated": 0, "skipped": int(len(df))}

        if "WorkItemId" not in metadata.columns:
            raise ValueError("Metadata file is missing required column: WorkItemId")

        index_by_id: dict[str, int] = {}
        work_ids_series = metadata["WorkItemId"].fillna("").astype(str).map(normalize_work_item_id)
        for idx, val in enumerate(work_ids_series.tolist()):
            wid = str(val).strip()
            if wid and wid not in index_by_id:
                index_by_id[wid] = idx

        meta_link_col: str | None
        if "link" in metadata.columns:
            meta_link_col = "link"
        elif "Link" in metadata.columns:
            meta_link_col = "Link"
        else:
            meta_link_col = None

        has_bugurl_col = "BugUrl" in getattr(metadata, "columns", [])

        texts: list[str] = []
        rows_for_apply: list[tuple[str, str, str]] = []  # (work_id, title, bug_url)
        # Keep a fingerprint for each processed work item so the existing
        # incremental fingerprinting stays consistent.
        fingerprints_for_ids: dict[str, str] = {}
        changed_df = changed_df.sort_values("WorkItemId")
        for row in changed_df.itertuples(index=False):
            wid = str(getattr(row, "WorkItemId") or "").strip()
            if not wid:
                continue
            title = str(getattr(row, "Title") or "").strip()
            semantic_text = str(getattr(row, "SemanticText") or "").strip()
            bug_url = ""
            try:
                bug_url = str(getattr(row, "BugUrl") or "").strip()
            except Exception:
                bug_url = ""
            payload = f"{wid}|{title}|{semantic_text}".encode("utf-8", errors="replace")
            fingerprints_for_ids[wid] = hashlib.sha256(payload).hexdigest()
            combined_text = build_embedding_text(row)
            texts.append(combined_text)
            rows_for_apply.append((wid, title, bug_url))

        if not texts:
            return {"added": 0, "updated": 0, "skipped": int(len(df))}

        from semantic_engine import get_model
        from semantic_engine import USE_PRENORMALIZED, l2_normalize

        model_dir = (runtime_paths.get_model_dir() / "all-MiniLM-L6-v2").resolve()
        model = get_model(str(model_dir))

        new_embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        if USE_PRENORMALIZED:
            new_embeddings = l2_normalize(new_embeddings)

        new_embeddings = np.asarray(new_embeddings)
        if new_embeddings.ndim != 2:
            raise ValueError("Model returned embeddings with unexpected shape")

        if embeddings.shape[0] > 0 and new_embeddings.shape[1] != embeddings.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Existing dim={embeddings.shape[1]}, new dim={new_embeddings.shape[1]}"
            )

        if embeddings.shape[0] == 0:
            embeddings.array = embeddings.array.reshape((0, new_embeddings.shape[1]))

        added = 0
        updated = 0
        for i, (work_id, title, bug_url) in enumerate(rows_for_apply):
            if not work_id:
                continue

            if work_id in index_by_id:
                idx = index_by_id[work_id]
                embeddings[idx] = new_embeddings[i]
                if "Title" in metadata.columns:
                    metadata.at[idx, "Title"] = title
                if has_bugurl_col:
                    metadata.at[idx, "BugUrl"] = bug_url
                updated += 1
            else:
                vec = np.asarray(new_embeddings[i]).reshape(1, -1)
                embeddings.array = np.vstack([embeddings.array, vec])

                new_row: dict[str, object] = {col: "" for col in metadata.columns}
                new_row["WorkItemId"] = work_id
                if "Title" in metadata.columns:
                    new_row["Title"] = title
                if meta_link_col is not None:
                    # Maintain legacy link column only if present.
                    new_row[meta_link_col] = ""
                if has_bugurl_col:
                    new_row["BugUrl"] = bug_url

                metadata.loc[len(metadata)] = new_row
                index_by_id[work_id] = int(len(metadata) - 1)
                added += 1

        if int(embeddings.shape[0]) != int(len(metadata)):
            raise ValueError(
                "Embeddings/metadata length mismatch before saving. "
                f"Embeddings rows={int(embeddings.shape[0])}, metadata rows={int(len(metadata))}"
            )

        data_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_embeddings_and_metadata(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            write_embeddings=lambda p: np.save(p, embeddings.array),
            write_metadata=lambda p: metadata.to_csv(p, index=False),
        )

        # Update fingerprints atomically (best-effort, but should not be skipped).
        fingerprints_path = data_dir / "bug_fingerprints.json"
        previous_fingerprints: dict[str, str] = {}
        if fingerprints_path.is_file():
            try:
                loaded = json.loads(fingerprints_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    previous_fingerprints = {str(k).strip(): str(v).strip() for k, v in loaded.items()}
            except Exception:
                previous_fingerprints = {}

        updated_fingerprints = dict(previous_fingerprints)
        for wid, fp in fingerprints_for_ids.items():
            if wid and fp:
                updated_fingerprints[wid] = fp

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(fingerprints_path.parent),
                prefix=fingerprints_path.stem + "_",
                suffix=".tmp",
            ) as f:
                json.dump(updated_fingerprints, f, indent=2, sort_keys=True)
                f.flush()
                try:
                    import os

                    os.fsync(f.fileno())
                except Exception:
                    pass
                tmp_path = Path(f.name)

            tmp_path.replace(fingerprints_path)
        finally:
            if tmp_path is not None and tmp_path.exists() and tmp_path != fingerprints_path:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        skipped = max(0, int(len(df)) - (added + updated))
        return {"added": int(added), "updated": int(updated), "skipped": int(skipped)}

    def set_status(self, message: str) -> None:
        """Central status updater (safe for worker threads)."""

        # Thread-safe: marshal to the Tk thread if called from a worker.
        try:
            if threading.current_thread() is not threading.main_thread():
                self.root.after(0, lambda: self.set_status(message))
                return
        except Exception:
            pass

        msg = str(message)
        try:
            self.status_var.set(msg)
            self.root.update_idletasks()
        except Exception:
            return

        # Keep a small coarse-grained state on the right.
        lowered = msg.lower().strip()
        state = self.state_var.get()
        if "loading search index" in lowered:
            state = "Loading"
        elif "searching" in lowered:
            state = "Searching"
        elif any(k in lowered for k in ("loading semantic model", "initializing model", "model")) and any(
            k in lowered for k in ("loading", "initializing")
        ):
            state = "Loading model"
        elif any(k in lowered for k in ("refreshing", "fetching", "updating", "reloading")):
            state = "Refreshing"
        elif any(k in lowered for k in ("update complete", "no changes")):
            state = "Index updated"
        elif any(k in lowered for k in ("failed", "error")):
            state = "Error"
        elif lowered in {"ready", "model ready"} or lowered.startswith("found "):
            state = "Ready"

        try:
            self.state_var.set(state)
        except Exception:
            pass

    def start_progress(self, message: str) -> None:
        """Show indeterminate progress + disable actions (thread-safe)."""

        try:
            if threading.current_thread() is not threading.main_thread():
                self.root.after(0, lambda: self.start_progress(message))
                return
        except Exception:
            pass

        self.set_status(str(message))

        try:
            if self._progress is not None:
                self._progress.grid()
                self._progress.start(10)
        except Exception:
            pass

        try:
            self._set_search_enabled(False)
            self._set_refresh_enabled(False)
        except Exception:
            pass

    def stop_progress(self, message: str | None = "Ready") -> None:
        """Hide progress + enable actions (thread-safe)."""

        try:
            if threading.current_thread() is not threading.main_thread():
                self.root.after(0, lambda: self.stop_progress(message))
                return
        except Exception:
            pass

        try:
            if self._progress is not None:
                self._progress.stop()
                self._progress.grid_remove()
        except Exception:
            pass

        try:
            self._set_search_enabled(True)
            self._set_refresh_enabled(True)
        except Exception:
            pass

        if message is not None:
            self.set_status(str(message))

    def update_bug_count_display(self) -> None:
        """Update the 'Total Bugs in Database' label."""
        try:
            count = len(self.embeddings) if self.embeddings is not None else 0
        except Exception:
            count = 0
        try:
            self._bug_count_var.set(f"Total Bugs in Database: {int(count)}")
        except Exception:
            pass

    # -------------------- Search box UX --------------------

    def _apply_search_placeholder(self) -> None:
        if self._query_text is None:
            return
        try:
            current = self._query_text.get("1.0", "end").strip()
        except Exception:
            current = ""
        if current:
            return

        try:
            self._query_text.delete("1.0", "end")
            self._query_text.insert("1.0", self._placeholder_text)
            self._query_text.configure(foreground="#6b7280")
            self._placeholder_active = True
        except Exception:
            pass

    def _clear_search_placeholder(self) -> None:
        if not self._placeholder_active:
            return
        try:
            self._query_text.delete("1.0", "end")
            self._query_text.configure(foreground="#111827")
            self._placeholder_active = False
        except Exception:
            pass

    def _on_search_focus_in(self, _event: tk.Event) -> None:
        self._clear_search_placeholder()

    def _on_search_focus_out(self, _event: tk.Event) -> None:
        try:
            current = self._query_text.get("1.0", "end").strip()
        except Exception:
            current = ""
        if not current:
            self._apply_search_placeholder()

    def _on_search_return(self, event: tk.Event) -> str | None:
        # Shift+Enter should insert a newline.
        try:
            if int(getattr(event, "state", 0)) & 0x0001:
                return None
        except Exception:
            return None

        # Enter triggers Search.
        self._on_search_clicked()
        return "break"

    def _focus_search_box(self) -> None:
        try:
            self._query_text.focus_set()
            self._query_text.see("1.0")
        except Exception:
            pass

    def _on_global_copy(self, _event: tk.Event) -> None:
        # If focus is in the query textbox, keep native copy behavior.
        try:
            if self._root.focus_get() is self._query_text:
                return
        except Exception:
            pass

        self._copy_selected_cell("Title")

    # -------------------- Treeview UX --------------------

    def _resize_tree_columns(self, _event: tk.Event | None = None) -> None:
        try:
            total_w = int(self._tree.winfo_width())
        except Exception:
            return
        if total_w <= 50:
            return

        # Leave a little breathing room for the scrollbar.
        avail = max(200, total_w - 26)

        # Fixed-ish columns
        w_rank = 50
        w_match = 90
        w_wid = 120

        remaining = max(200, avail - (w_rank + w_match + w_wid))
        w_title = int(remaining * 0.58)
        w_link = remaining - w_title

        # Minimums
        w_title = max(320, w_title)
        w_link = max(200, w_link)

        try:
            self._tree.column("Rank", width=w_rank, stretch=False, anchor="center")
            self._tree.column("Match", width=w_match, stretch=False, anchor="center")
            self._tree.column("WorkItemId", width=w_wid, stretch=False, anchor="center")
            self._tree.column("Title", width=w_title, stretch=True, anchor="w")
            self._tree.column("Link", width=w_link, stretch=True, anchor="w")
        except Exception:
            pass

    def _on_tree_leave(self, _event: tk.Event) -> None:
        self._clear_hover_row()

    def _set_hover_row(self, row_id: str | None) -> None:
        if row_id == self._hover_row_id:
            return
        self._clear_hover_row()
        if not row_id:
            return
        try:
            # Don't override selection visuals.
            if row_id in set(self._tree.selection() or ()):  # type: ignore[arg-type]
                self._hover_row_id = None
                return
        except Exception:
            pass

        try:
            tags = list(self._tree.item(row_id, "tags") or ())
            if "hover" not in tags:
                tags.append("hover")
                self._tree.item(row_id, tags=tuple(tags))
        except Exception:
            return
        self._hover_row_id = row_id

    def _clear_hover_row(self) -> None:
        if not self._hover_row_id:
            return
        try:
            row_id = self._hover_row_id
            tags = [t for t in (self._tree.item(row_id, "tags") or ()) if t != "hover"]
            self._tree.item(row_id, tags=tuple(tags))
        except Exception:
            pass
        finally:
            self._hover_row_id = None

    def _on_tree_motion(self, event: tk.Event) -> None:
        """Hover effects + hand cursor on Link column."""

        try:
            # Hover row highlight
            row_id = self._tree.identify_row(event.y)
            self._set_hover_row(row_id)
        except Exception:
            pass

        # Existing link-hover cursor logic
        try:
            region = self._tree.identify("region", event.x, event.y)
            if region != "cell":
                self._tree.configure(cursor="")
                return
            col = self._tree.identify_column(event.x)
            link_col = self._tree_column_number("Link")
            if link_col and col == link_col:
                self._tree.configure(cursor="hand2")
            else:
                self._tree.configure(cursor="")
        except Exception:
            try:
                self._tree.configure(cursor="")
            except Exception:
                pass

    def _on_result_double_click(self, event: tk.Event) -> None:
        # Double-click anywhere on a row opens the bug link.
        try:
            row_id = self._tree.identify_row(event.y)
            if row_id:
                self._tree.selection_set(row_id)
        except Exception:
            pass
        self._open_selected_bug()

    def _open_selected_bug(self) -> None:
        selection = self._tree.selection()
        if not selection:
            return
        values = self._tree.item(selection[0], "values")
        if not values:
            return
        link = self._tree_value_at(values, "Link").strip()
        if not link:
            messagebox.showinfo("No link", "No link is available for this item.")
            return
        try:
            webbrowser.open_new_tab(link)
        except Exception:
            messagebox.showwarning("Open link", "Could not open the link in your browser.")

    def _copy_selected_cell(self, col_id: str) -> None:
        selection = self._tree.selection()
        if not selection:
            return
        values = self._tree.item(selection[0], "values")
        if not values:
            return
        text = self._tree_value_at(values, col_id)
        try:
            self._root.clipboard_clear()
            self._root.clipboard_append(str(text))
        except Exception:
            pass

    def _on_tree_right_click(self, event: tk.Event) -> None:
        try:
            row_id = self._tree.identify_row(event.y)
            if row_id:
                self._tree.selection_set(row_id)
        except Exception:
            return

        if self._tree_menu is None:
            return
        try:
            self._tree_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self._tree_menu.grab_release()
            except Exception:
                pass

    def debug_refresh_state(self, excel_path: Path) -> None:
        """Temporary read-only diagnostics for refresh fingerprinting.

        Prints detailed information about:
        - Excel parsing and required columns
        - WorkItemId normalization
        - Fingerprint file contents
        - SHA-256 computation and comparisons

        This function must not modify any application state.
        """

        print("\n===== DEBUG REFRESH STATE =====")
        try:
            # 1) In-memory counts
            embeddings_count: int
            try:
                embeddings_count = len(self.embeddings) if self.embeddings is not None else 0
            except Exception:
                embeddings_count = 0

            metadata_count: int
            try:
                metadata_count = len(self.metadata) if self.metadata is not None else 0
            except Exception:
                metadata_count = 0

            print("[In-memory]")
            print(f"  len(self.embeddings): {embeddings_count}")
            print(f"  len(self.metadata): {metadata_count}")

            fingerprints_path = _data_dir() / "bug_fingerprints.json"

            # 2) Load fingerprints
            fingerprints: dict[str, str] = {}
            if fingerprints_path.is_file():
                try:
                    loaded = json.loads(fingerprints_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        # Ensure keys/values are strings for consistent comparison.
                        fingerprints = {str(k).strip(): str(v).strip() for k, v in loaded.items()}
                except Exception as exc:  # noqa: BLE001
                    print(f"[Fingerprints] Failed to load {fingerprints_path}: {exc}")
                    fingerprints = {}

            print("[Fingerprints]")
            print(f"  Path: {fingerprints_path}")
            print(f"  Entries: {len(fingerprints)}")

            # 3) Read Excel
            import pandas as pd

            print("[Excel]")
            print(f"  Path: {excel_path}")
            df = pd.read_excel(excel_path, header=0)
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")

            required_columns = ["WorkItemId", "Title", "SemanticText"]
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                print(f"  MISSING required columns: {missing}")
                print("===== END DEBUG REFRESH STATE =====\n")
                return

            def norm_str(val: object) -> str:
                try:
                    # pandas NA / NaN
                    if val is None:
                        return ""
                    # Avoid importing numpy; pandas provides isna.
                    if pd.isna(val):
                        return ""
                except Exception:
                    pass
                return str(val).strip()

            def norm_work_item_id_simple(raw: object) -> str:
                # As requested: wid = str(...).strip()
                return norm_str(raw)

            def norm_work_item_id_app(raw: object) -> str:
                # Mirror the refresh logic normalization (helps diagnose .0 issues).
                s = norm_str(raw)
                if s.endswith(".0") and s.replace(".0", "").isdigit():
                    s = s[:-2]
                return s

            def sha256_hex(payload: str) -> str:
                return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()

            print("[Rows]")
            for idx, row in df.iterrows():
                wid = norm_work_item_id_simple(row["WorkItemId"])
                wid_app = norm_work_item_id_app(row["WorkItemId"])

                title = norm_str(row["Title"])
                semantic_text = norm_str(row["SemanticText"])

                # Required by spec: SHA-256(wid + title + semantic_text)
                computed_hash = sha256_hex(f"{wid}{title}{semantic_text}")

                # Additional (helpful) comparison: the app's current fingerprint format.
                computed_hash_app = sha256_hex(f"{wid_app}|{title}|{semantic_text}")

                fp_exists_simple = wid in fingerprints
                fp_exists_app = wid_app in fingerprints

                stored_fp_simple = fingerprints.get(wid)
                stored_fp_app = fingerprints.get(wid_app)

                match_simple = (stored_fp_simple == computed_hash) if stored_fp_simple is not None else False
                match_app = (stored_fp_app == computed_hash_app) if stored_fp_app is not None else False

                print(f"- Row {int(idx)}")
                print(f"    WorkItemId: {wid!r} (app_norm={wid_app!r})")
                print(f"    SHA256(wid+title+semantic): {computed_hash}")
                print(f"    Exists in fingerprint file (wid): {fp_exists_simple}")
                if fp_exists_simple:
                    print(f"    Hash matches stored (wid): {match_simple}")
                print(f"    Exists in fingerprint file (app_norm wid): {fp_exists_app}")
                if fp_exists_app:
                    print(f"    Hash matches stored (app_norm wid, app_format): {match_app}")

            print("===== END DEBUG REFRESH STATE =====\n")

        except Exception as exc:  # noqa: BLE001
            # Never block refresh due to diagnostics.
            print(f"[DEBUG] debug_refresh_state failed: {exc}")
            print(traceback.format_exc())
            print("===== END DEBUG REFRESH STATE =====\n")

    def fetch_latest_excel(self) -> Path:
        """Return the local "live" Excel file path used for refresh.

        This simulates fetching a newly refreshed Excel file from an external source
        (e.g., SharePoint) by reading a single local file that can be manually
        updated over time.

        Notes:
        - This function intentionally does not parse the Excel.
        - It raises a clear error if the file is missing.
        """

        excel_path = (_data_dir() / "bugs_live.xlsx").resolve()
        logging.info("[refresh] Fetching Excel from: %s", excel_path)

        if not excel_path.is_file():
            raise FileNotFoundError(
                "Live Excel file not found. Expected a local file at:\n\n"
                f"  {excel_path}\n\n"
                "Create or copy the file there (data/bugs_live.xlsx) to simulate live updates."
            )

        return excel_path

    def update_embeddings_from_excel(self, excel_path: Path) -> tuple[int, int, bool]:
        """Incrementally update embeddings and metadata from the Excel snapshot.

        Uses lightweight fingerprinting to detect which bugs are new or modified
        (by WorkItemId) and only re-embeds those items.

        Fingerprints are persisted to:
          data/bug_fingerprints.json

        Constraints:
        - Does not delete embeddings for removed bugs (handled later).
        - Avoids refactoring existing search logic; updates on-disk artifacts.
        """

        # Import heavy deps lazily to keep UI startup fast.
        import os

        import numpy as np
        import pandas as pd

        def normalize_work_item_id(raw: object) -> str:
            """Normalize WorkItemId to a stable string identifier."""
            s = str(raw).strip()
            # Common Excel/pandas representation for whole numbers: '12345.0'
            if s.endswith(".0") and s.replace(".0", "").isdigit():
                s = s[:-2]
            return s

        _ensure_runtime_data_layout()
        data_dir = _data_dir()
        embeddings_path = data_dir / "bug_embeddings.npy"
        metadata_path = data_dir / "bug_metadata.csv"
        fingerprints_path = data_dir / "bug_fingerprints.json"

        if not excel_path.is_file():
            raise FileNotFoundError(f"Excel snapshot not found: {excel_path}")

        # 1) Load Excel snapshot
        logging.info("[refresh] Reading Excel snapshot: %s", excel_path)
        df = pd.read_excel(excel_path, header=0)

        required_columns = ["WorkItemId", "Title", "SemanticText"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column in Excel: {col}")

        # Normalize required fields early.
        df = df.dropna(subset=["WorkItemId", "Title"]).copy()
        df["WorkItemId"] = df["WorkItemId"].map(normalize_work_item_id)
        df = df[df["WorkItemId"].astype(str).str.strip() != ""].copy()
        df["WorkItemId"] = df["WorkItemId"].astype(str).str.strip()

        df["Title"] = df["Title"].fillna("").astype(str).str.strip()
        df["SemanticText"] = df["SemanticText"].fillna("").astype(str).str.strip()

        # SemanticText safety: fall back to Title when SemanticText is blank.
        df.loc[df["SemanticText"] == "", "SemanticText"] = df.loc[df["SemanticText"] == "", "Title"]

        # Final guard: ensure every embedded row has non-empty SemanticText.
        df = df[df["SemanticText"].astype(str).str.strip() != ""].copy()

        # Determine which link column we can use from the Excel (if any).
        excel_link_col: str | None = None
        if "link" in df.columns:
            excel_link_col = "link"
        elif "Link" in df.columns:
            excel_link_col = "Link"

        # 2) Load existing fingerprints
        previous_fingerprints: dict[str, str] = {}
        if fingerprints_path.is_file():
            try:
                previous_fingerprints = json.loads(fingerprints_path.read_text(encoding="utf-8"))
                if not isinstance(previous_fingerprints, dict):
                    previous_fingerprints = {}
            except Exception:
                # If fingerprints are corrupted, fall back to re-fingerprinting during this run.
                previous_fingerprints = {}

        # 3) Compute fingerprints for the Excel snapshot
        def compute_fingerprint(work_item_id: str, title: str, semantic_text: str) -> str:
            payload = f"{work_item_id}|{title}|{semantic_text}".encode("utf-8", errors="replace")
            return hashlib.sha256(payload).hexdigest()

        current_fingerprints: dict[str, str] = {}
        for row in df.itertuples(index=False):
            work_id = str(getattr(row, "WorkItemId") or "").strip()
            title = str(getattr(row, "Title") or "").strip()
            semantic_text = str(getattr(row, "SemanticText") or "").strip()
            current_fingerprints[work_id] = compute_fingerprint(work_id, title, semantic_text)

        new_ids: list[str] = []
        modified_ids: list[str] = []
        unchanged_ids: list[str] = []

        for work_id, fp in current_fingerprints.items():
            prev = previous_fingerprints.get(work_id)
            if prev is None:
                new_ids.append(work_id)
            elif prev != fp:
                modified_ids.append(work_id)
            else:
                unchanged_ids.append(work_id)

        logging.info("[refresh] New bugs: %s", len(new_ids))
        logging.info("[refresh] Modified bugs: %s", len(modified_ids))
        logging.info("[refresh] Unchanged bugs: %s", len(unchanged_ids))

        changed_ids = new_ids + modified_ids
        if not changed_ids:
            # Still persist fingerprints for newly seen bugs set? (none) and return.
            return (0, 0, False)

        # 4) Use the single authoritative in-memory state.
        # Disk is write-only during refresh.
        data_dir.mkdir(parents=True, exist_ok=True)

        if self.embeddings is None or self.metadata is None:
            raise RuntimeError("In-memory embeddings/metadata are not loaded")

        if self._embeddings_obj_id is not None and id(self.embeddings) != self._embeddings_obj_id:
            raise AssertionError("Embedding state divergence detected before refresh")

        if not isinstance(self.embeddings, EmbeddingsStore):
            raise RuntimeError("In-memory embeddings are not an EmbeddingsStore")

        embeddings = self.embeddings
        metadata = self.metadata

        before_count = int(getattr(embeddings, "shape", [0])[0])
        logging.info("[REFRESH] embeddings before=%s", before_count)
        logging.info("[APPEND] embeddings_store id=%s len=%s", id(self.embeddings), len(self.embeddings))

        if "WorkItemId" not in metadata.columns:
            raise ValueError("Metadata file is missing required column: WorkItemId")

        # Build index by normalized WorkItemId for in-place updates.
        index_by_id: dict[str, int] = {}
        work_ids_series = metadata["WorkItemId"].fillna("").astype(str).map(normalize_work_item_id)
        for idx, val in enumerate(work_ids_series.tolist()):
            wid = str(val).strip()
            if not wid:
                continue
            if wid not in index_by_id:
                index_by_id[wid] = idx

        # Determine which metadata link column to maintain.
        if "link" in metadata.columns:
            meta_link_col = "link"
        elif "Link" in metadata.columns:
            meta_link_col = "Link"
        else:
            meta_link_col = None

        # 5) Encode only new + modified bugs
        changed_df = df[df["WorkItemId"].isin(changed_ids)].copy()
        changed_df = changed_df.sort_values("WorkItemId")

        texts: list[str] = []
        for row in changed_df.itertuples(index=False):
            wid = str(getattr(row, "WorkItemId") or "").strip()
            combined_text = build_embedding_text(row)
            logging.info("[EMBED] WorkItemId %s | Text length: %s", wid, len(combined_text))
            texts.append(combined_text)
        if not texts:
            return (len(new_ids), len(modified_ids), False)

        logging.info("[refresh] Re-embedding changed bugs: %s", len(texts))

        from semantic_engine import get_model

        model_dir = (runtime_paths.get_model_dir() / "all-MiniLM-L6-v2").resolve()
        model = get_model(str(model_dir))

        new_embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        new_embeddings = np.asarray(new_embeddings)
        if new_embeddings.ndim != 2:
            raise ValueError("Model returned embeddings with unexpected shape")

        if embeddings.shape[0] > 0 and new_embeddings.shape[1] != embeddings.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Existing dim={embeddings.shape[1]}, new dim={new_embeddings.shape[1]}"
            )

        if embeddings.shape[0] == 0:
            # Initialize correct dimension from the model output.
            embeddings.array = embeddings.array.reshape((0, new_embeddings.shape[1]))

        # 6) Apply updates in-place for modified, append for new.
        # Keep metadata aligned 1:1 with embeddings at all times.

        for i, row in enumerate(changed_df.itertuples(index=False)):
            work_id = str(getattr(row, "WorkItemId") or "").strip()
            title = str(getattr(row, "Title") or "").strip()

            link_val = ""
            if excel_link_col is not None:
                try:
                    link_val = str(getattr(row, excel_link_col) or "").strip()
                except Exception:
                    link_val = ""

            if work_id in index_by_id:
                idx = index_by_id[work_id]
                embeddings[idx] = new_embeddings[i]
                # Update metadata fields without disturbing other columns.
                if "Title" in metadata.columns:
                    metadata.at[idx, "Title"] = title
                if meta_link_col is not None:
                    metadata.at[idx, meta_link_col] = link_val
            else:
                logging.info("[refresh][NEW] %s %s", work_id, title)
                logging.info("[refresh] appended embedding for %s", work_id)

                vec = np.asarray(new_embeddings[i]).reshape(1, -1)
                embeddings.array = np.vstack([embeddings.array, vec])

                new_row: dict[str, object] = {col: "" for col in metadata.columns}
                new_row["WorkItemId"] = work_id
                if "Title" in metadata.columns:
                    new_row["Title"] = title
                if meta_link_col is not None:
                    new_row[meta_link_col] = link_val
                metadata.loc[len(metadata)] = new_row

                index_by_id[work_id] = int(len(metadata) - 1)
                logging.info("[APPEND] embeddings_store id=%s len=%s", id(self.embeddings), len(self.embeddings))

        # 7) Persist updated artifacts (write-only)
        after_count = int(getattr(embeddings, "shape", [0])[0])
        logging.info("[REFRESH] embeddings before=%s after=%s", before_count, after_count)

        if int(after_count) != int(len(metadata)):
            raise ValueError(
                "Embeddings/metadata length mismatch before saving. "
                f"Embeddings rows={int(after_count)}, metadata rows={int(len(metadata))}"
            )

        logging.info("[refresh] saving embeddings: %s", after_count)

        _atomic_write_embeddings_and_metadata(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            write_embeddings=lambda p: np.save(p, embeddings.array),
            write_metadata=lambda p: metadata.to_csv(p, index=False),
        )

        # 8) Persist fingerprints atomically (only after embeddings update succeeded)
        updated_fingerprints = dict(previous_fingerprints)
        for work_id in changed_ids:
            updated_fingerprints[work_id] = current_fingerprints.get(work_id, updated_fingerprints.get(work_id, ""))

        fingerprints_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(fingerprints_path.parent),
                prefix=fingerprints_path.stem + "_",
                suffix=".tmp",
            ) as f:
                json.dump(updated_fingerprints, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
                tmp_path = Path(f.name)

            tmp_path.replace(fingerprints_path)
        finally:
            if tmp_path is not None and tmp_path.exists() and tmp_path != fingerprints_path:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return (len(new_ids), len(modified_ids), True)

    def reload_search_index(self) -> None:
        """Reload the in-memory search index so new artifacts are used."""
        logging.info("[refresh] Reloading search index...")
        if self._service is None:
            raise RuntimeError("Search service is not initialized")

        # Reload from disk so memory-mapping is used (and the new artifacts are picked up).
        self._service.reload_index()
        self.embeddings, self.metadata = self._service.get_index()
        self._embeddings_obj_id = id(self.embeddings)

    def _post_status(self, text: str) -> None:
        """Post a status update to be applied on the Tk main thread."""
        self._result_queue.put(("status", text))

    # -------------------- UI helpers --------------------

    def _set_search_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._search_btn.configure(state=state)
        self._topk_spin.configure(state=state)

    def _set_refresh_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._refresh_btn.configure(state=state)

    def _clear_results(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

    def _tree_column_number(self, col_id: str) -> str | None:
        """Return Treeview column number string (e.g. '#5') for a column id."""
        try:
            cols = list(self._tree["columns"])
            idx = cols.index(col_id)
            return f"#{idx + 1}"
        except Exception:
            return None

    def _tree_value_at(self, values: object, col_id: str) -> str:
        """Fetch a column's value from a Treeview 'values' tuple safely."""
        try:
            cols = list(self._tree["columns"])
            idx = cols.index(col_id)
        except Exception:
            return ""

        if not values or not isinstance(values, (list, tuple)):
            return ""
        try:
            return str(values[idx]) if idx < len(values) else ""
        except Exception:
            return ""

    def _sort_tree(self, col_id: str) -> None:
        """Sort Treeview rows by a column; toggles ascending/descending."""

        def parse(v: object) -> tuple[int, object]:
            s = str(v).strip()
            if not s:
                return (2, "")
            if s.endswith("%"):
                try:
                    return (0, float(s[:-1]))
                except Exception:
                    return (1, s.lower())
            try:
                return (0, float(s))
            except Exception:
                return (1, s.lower())

        try:
            col_idx = list(self._tree["columns"]).index(col_id)
        except Exception:
            return

        items = list(self._tree.get_children(""))
        sort_desc = bool(self._tree_sort_desc.get(col_id, False))

        def item_key(item_id: str) -> tuple[int, object]:
            vals = self._tree.item(item_id, "values")
            try:
                v = vals[col_idx] if vals and col_idx < len(vals) else ""
            except Exception:
                v = ""
            return parse(v)

        items.sort(key=item_key, reverse=sort_desc)
        for i, item_id in enumerate(items):
            self._tree.move(item_id, "", i)

        # Re-apply striping after sorting.
        for i, item_id in enumerate(items, start=1):
            current_tags = set(self._tree.item(item_id, "tags") or ())
            current_tags.discard("odd")
            if i % 2 == 1:
                current_tags.add("odd")
            self._tree.item(item_id, tags=tuple(current_tags))

        self._tree_sort_desc[col_id] = not sort_desc

    def _render_results(self, results: List[BugResult]) -> None:
        self._clear_results()

        if not results:
            self.set_status("Found 0 result(s)")
            return

        # NOTE:
        # Cosine similarity is a *similarity score*, not a literal “percent match”.
        # In many real datasets scores often cluster around ~0.4–0.7 even for good hits,
        # which can look like “40–70%” to users. To make the UI more intuitive, we
        # display a relative percentage vs the best hit in the current search:
        #   Match% = score / best_score * 100
        scores: list[float] = []
        for r in results:
            s = float(r.final_score)
            if s < 0.0:
                s = 0.0
            if s > 1.0:
                s = 1.0
            scores.append(s)
        best_score = max(scores) if scores else 0.0

        for rank, (r, score) in enumerate(zip(results, scores), start=1):
            if best_score > 0.0:
                match_pct = (score / best_score) * 100.0
            else:
                match_pct = 0.0
            if match_pct > 100.0:
                match_pct = 100.0
            match_display = f"{match_pct:.1f}%"

            tag = "odd" if (rank % 2 == 1) else ""
            self._tree.insert(
                "",
                "end",
                values=(
                    rank,
                    match_display,
                    r.work_item_id,
                    r.title,
                    r.link,
                ),
                tags=(tag,) if tag else (),
            )

        self.set_status(f"Found {len(results)} result(s)")

    def _on_tree_click(self, event: tk.Event) -> None:
        # Open link when user clicks the Link cell.
        region = self._tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        col = self._tree.identify_column(event.x)  # e.g. '#1'
        link_col = self._tree_column_number("Link")
        if not link_col or col != link_col:
            return

        row_id = self._tree.identify_row(event.y)
        if not row_id:
            return

        values = self._tree.item(row_id, "values")
        if not values:
            return

        link = self._tree_value_at(values, "Link").strip()
        if not link:
            return

        try:
            webbrowser.open(link)
        except Exception:
            messagebox.showwarning("Open link", "Could not open the link in your browser.")

    # -------------------- Thread-to-UI bridge --------------------

    def _poll_queue(self) -> None:
        """Process worker messages on the Tk main thread."""
        try:
            while True:
                msg_type, payload = self._result_queue.get_nowait()

                if msg_type == "engine_loaded":
                    self._service = payload
                    try:
                        self.embeddings, self.metadata = self._service.get_index()
                    except Exception:
                        self.embeddings, self.metadata = None, None

                    # Initialize index stats immediately on startup.
                    now = datetime.now()
                    self.last_index_time = now
                    self.last_indexed_time = now
                    try:
                        self.metadata_df = self.metadata
                        self.total_indexed = int(len(self.metadata_df)) if self.metadata_df is not None else 0
                    except Exception:
                        self.metadata_df = None
                        self.total_indexed = 0

                    self.stop_progress("Ready")
                    self._embeddings_obj_id = id(self.embeddings)
                    if self.embeddings is not None:
                        print("[BOOT]", id(self.embeddings), len(self.embeddings))
                    self.update_bug_count_display()
                    try:
                        startup_excel = self.get_latest_excel()
                        self.update_data_source_status(startup_excel)
                    except FileNotFoundError:
                        self.update_data_source_status(None)
                    self._close_splash()

                    # After the UI is visible, warm up the ML model in the background.
                    # This improves perceived responsiveness without blocking startup.
                    self._root.after(200, self._init_model_async)

                elif msg_type == "engine_failed":
                    exc, tb = payload
                    self.stop_progress("Failed to load search index")
                    self._set_search_enabled(False)
                    self._set_refresh_enabled(False)
                    self._close_splash()
                    messagebox.showerror(
                        "Startup error",
                        "Failed to load the search index.\n\n"
                        f"Error: {exc}\n\n"
                        "Details were printed to the console.",
                    )
                    print(tb)

                elif msg_type == "model_loaded":
                    # If the user already clicked Search while the model was loading,
                    # run that pending search now.
                    pending = self._pending_search
                    self._pending_search = None
                    if pending is None:
                        self.stop_progress("Ready")
                    else:
                        q, k = pending
                        self._run_search_async(q, k)

                elif msg_type == "model_failed":
                    exc, tb = payload
                    # Stop any visible progress started for model load.
                    self.stop_progress("Model init failed")

                    # Only show a blocking dialog if the user actually requested a search.
                    if self._pending_search is not None:
                        if isinstance(exc, FileNotFoundError):
                            messagebox.showerror(
                                "Model files missing",
                                "The app is configured for fully offline use and requires a local\n"
                                "SentenceTransformer model folder at:\n\n"
                                "  models\\all-MiniLM-L6-v2\\\n\n"
                                "How to create it (run once from the project folder):\n"
                                "  python -c \"from sentence_transformers import SentenceTransformer; "
                                "m=SentenceTransformer('all-MiniLM-L6-v2'); m.save('models/all-MiniLM-L6-v2')\"\n\n"
                                "Then restart the app (and rebuild the EXE if you are packaging with PyInstaller).",
                            )
                        else:
                            messagebox.showerror(
                                "Model initialization error",
                                "Failed to initialize the ML model.\n\n"
                                f"Error: {exc}\n\n"
                                "Details were printed to the console.",
                            )
                        self._pending_search = None
                    else:
                        # Background pre-load failed (usually missing model folder).
                        # Keep the app usable; user will see an error on Search.
                        self.set_status("Ready")
                    print(tb)

                elif msg_type == "search_ok":
                    self.stop_progress(None)
                    self._render_results(payload)

                elif msg_type == "search_failed":
                    exc, tb = payload
                    self.stop_progress("Search failed")
                    messagebox.showerror(
                        "Search error",
                        f"An error occurred while searching:\n\n{exc}\n\n"
                        "Details were printed to the console.",
                    )
                    print(tb)

                elif msg_type == "status":
                    self.set_status(str(payload))

                elif msg_type == "refresh_done":
                    self.stop_progress(None)

                    info = payload if isinstance(payload, dict) else {}
                    ok = bool(info.get("ok"))
                    if ok:
                        now = datetime.now()
                        self.last_index_time = now
                        self.last_indexed_time = now
                        stats = info.get("stats") if isinstance(info.get("stats"), dict) else None
                        if stats is not None:
                            self.last_refresh_stats = {
                                "added": int(stats.get("added", 0) or 0),
                                "updated": int(stats.get("updated", 0) or 0),
                                "skipped": int(stats.get("skipped", 0) or 0),
                            }
                        processed = int(info.get("processed") or 0)
                        if processed <= 0:
                            self.set_status("No changes found")
                            messagebox.showinfo("Refresh", "No new or updated bugs found")
                        else:
                            self.set_status(f"Update complete — {processed} bugs processed")
                            messagebox.showinfo("Refresh", f"Refresh complete: {processed} bugs processed")
                        self.update_bug_count_display()
                        self.update_data_source_status(None)
                        # Always reload from disk so counts are never stale.
                        self.reload_metadata()
                        self.update_info_tooltip()
                    else:
                        err = str(info.get("error") or "Refresh failed")
                        self.set_status(err)
                        messagebox.showerror("Refresh error", err)

        except queue.Empty:
            pass
        finally:
            self._root.after(100, self._poll_queue)

    def _run_search_async(self, query: str, top_k: int) -> None:
        if self._service is None:
            return

        self._clear_results()
        self.start_progress("Searching...")

        def worker(q: str, k: int, service: BugSearchService) -> None:
            try:
                results = service.search(q, top_k=k)
                self._result_queue.put(("search_ok", results))
            except Exception as exc:  # noqa: BLE001
                self._result_queue.put(("search_failed", (exc, traceback.format_exc())))

        self._active_worker = threading.Thread(
            target=worker,
            args=(query, top_k, self._service),
            daemon=True,
        )
        self._active_worker.start()


def main() -> None:
    frozen = bool(getattr(sys, "frozen", False))
    base = _app_base_path()
    data = _data_dir()

    log_file = runtime_paths.get_runtime_file("logs/app.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.captureWarnings(True)
    logging.info("Application started")

    def _log_unhandled(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        logging.exception("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _log_unhandled

    logging.info("[BOOT] Running in frozen mode: %s", frozen)
    logging.info("[BOOT] Base path: %s", base)
    try:
        data_exists = bool(data.exists())
    except Exception:
        data_exists = False
    logging.info("[BOOT] Data path exists: %s", data_exists)

    # Ensure data dir + legacy migrations before UI loads.
    _ensure_runtime_data_layout()

    root = tk.Tk()

    def _tk_report_callback_exception(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        logging.exception("Tkinter callback exception", exc_info=(exc_type, exc, tb))

    try:
        root.report_callback_exception = _tk_report_callback_exception  # type: ignore[assignment]
    except Exception:
        pass

    app = DesktopBugSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
