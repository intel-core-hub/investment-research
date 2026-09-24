"""Backups of the real-money data, which git (and so GitHub) never holds.

`python src/live/run_live.py backup` writes one zip per run to `backup.dir`
(set it in config/live.local.toml; default ~/investment-research-backups). To
survive losing this PC, point it at a private folder that is copied elsewhere,
such as a personal OneDrive folder. Never use a public or shared location.

Each zip holds data/live/ (market data only on request, since it can be
downloaded again) and config/live.local.toml, plus manifest.json with the SHA-256
of every file. The zip is read back and checked against the manifest before it is
kept. Old backups are never deleted automatically.

Restore: unzip into the repository root; it recreates data/live/ and config/live.local.toml.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

from settings import LOCAL_CONFIG_PATH, ROOT, LiveStore

DEFAULT_DIR = Path.home() / "investment-research-backups"
MANIFEST = "manifest.json"


class BackupError(RuntimeError):
    pass


def backup_dir(settings: dict) -> Path:
    configured = settings.get("backup", {}).get("dir", "")
    return Path(configured).expanduser() if configured else DEFAULT_DIR


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def files_to_back_up(store: LiveStore, local_config: Path | None, include_market: bool) -> dict[str, Path]:
    """Archive name -> file. Names mirror the repository layout, so unzipping in the root restores them."""
    files = {}
    if store.root.exists():
        for path in sorted(p for p in store.root.rglob("*") if p.is_file()):
            relative = path.relative_to(store.root)
            if relative.parts[0] == "market" and not include_market:
                continue
            if path.suffix == ".tmp":
                continue
            files[f"data/live/{relative.as_posix()}"] = path
    if local_config is not None and local_config.exists():
        files["config/live.local.toml"] = local_config
    return files


def create_backup(store: LiveStore, destination: Path, local_config: Path | None = LOCAL_CONFIG_PATH,
                  include_market: bool = False, now: datetime | None = None) -> Path:
    destination = Path(destination).expanduser().resolve()
    if destination == ROOT.resolve() or ROOT.resolve() in destination.parents:
        raise BackupError(f"{destination} is inside the repository; choose a folder outside it")
    files = files_to_back_up(store, local_config, include_market)
    if not any(name.startswith("data/live/") for name in files):
        raise BackupError(f"nothing to back up in {store.root}")

    now = now or datetime.now()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"live-backup-{now:%Y%m%d-%H%M%S}.zip"
    if target.exists():
        raise BackupError(f"{target} already exists")
    contents = {name: path.read_bytes() for name, path in files.items()}
    manifest = {"created_at": now.isoformat(timespec="seconds"), "source": str(store.root),
                "files": {name: sha256(data) for name, data in contents.items()}}

    tmp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in contents.items():
            z.writestr(name, data)
        z.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
    try:
        verify(tmp)
    except BackupError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, target)
    return target


def verify(path: Path) -> dict:
    """Read the zip back: every file must be intact and match the manifest. Returns the manifest."""
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise BackupError(f"{path} is corrupt")
        manifest = json.loads(z.read(MANIFEST))
        names = set(z.namelist()) - {MANIFEST}
        if names != set(manifest["files"]):
            raise BackupError(f"{path}: files do not match the manifest")
        for name, digest in manifest["files"].items():
            if sha256(z.read(name)) != digest:
                raise BackupError(f"{path}: {name} does not match its checksum")
    return manifest


def list_backups(destination: Path) -> list[Path]:
    destination = Path(destination).expanduser()
    return sorted(destination.glob("live-backup-*.zip")) if destination.exists() else []
