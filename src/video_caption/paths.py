import hashlib
import os
import re
import string
from functools import cache
from pathlib import Path, PurePosixPath

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts"}

_DRIVE_PATH = re.compile(r"^([A-Za-z]):[\\/]?(.*)$")
_WSL_UNC = re.compile(r"^[\\/]{2}wsl(?:\.localhost|\$)[\\/][^\\/]+[\\/]?(.*)$", re.IGNORECASE)
_MNT_DRIVE = re.compile(r"^/mnt/([a-z])(?:/(.*))?$")


@cache
def is_wsl() -> bool:
    if os.getenv("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def to_local_path(raw: str) -> Path:
    """Turn a user-supplied path into a path this process can open.

    On WSL, Windows paths are mapped to their Linux mount:
      C:\\Users\\me\\a.mp4              -> /mnt/c/Users/me/a.mp4
      \\\\wsl.localhost\\Ubuntu\\home\\a  -> /home/a
    """
    raw = raw.strip().strip('"').strip("'")
    if is_wsl():
        if m := _DRIVE_PATH.match(raw):
            drive, rest = m.groups()
            return Path("/mnt", drive.lower(), *[p for p in re.split(r"[\\/]+", rest) if p])
        if m := _WSL_UNC.match(raw):
            return Path("/", *[p for p in re.split(r"[\\/]+", m.group(1)) if p])
    return Path(raw).expanduser()


def to_display_path(path: Path) -> str:
    """Show /mnt/<drive>/... as a Windows path on WSL, so it matches what the user sees in Explorer."""
    posix = PurePosixPath(path).as_posix()
    if is_wsl() and (m := _MNT_DRIVE.match(posix)):
        drive, rest = m.groups()
        return f"{drive.upper()}:\\" + (rest or "").replace("/", "\\")
    return str(path)


def browse_roots() -> list[Path]:
    """Starting points for the folder browser: Windows drives on WSL, plus the home folder."""
    roots: list[Path] = []
    if is_wsl():
        roots += [Path("/mnt", d) for d in string.ascii_lowercase if Path("/mnt", d).is_dir()]
    roots.append(Path.home())
    return roots


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def work_dir_for(video_path: Path, temp_dir: str) -> Path:
    """Folder for intermediate files (audio, transcript, translation checkpoint).

    Keyed by stem + a short hash of the full path, so two videos with the same
    name in different folders do not share a cache.
    """
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return Path(temp_dir) / f"{video_path.stem}-{digest}"
