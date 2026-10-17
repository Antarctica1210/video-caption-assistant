import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import AppConfig
from ..logger import get_logger
from ..paths import browse_roots, is_video, is_wsl, to_display_path, to_local_path
from ..pipeline import PipelineError, run_pipeline, validate_video

log = get_logger("video_caption.web")

_STATIC = Path(__file__).parent / "static"
_MAX_LOG_LINES = 500


@dataclass
class Job:
    id: str
    video_path: str
    status: str = "queued"  # queued | transcribing | translating | done | failed
    error: str | None = None
    outputs: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


class _JobLogHandler(logging.Handler):
    """Copies pipeline log lines into the running job so the page can show them."""

    def __init__(self) -> None:
        super().__init__()
        self.job: Job | None = None
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        if self.job is not None:
            self.job.logs.append(self.format(record))
            del self.job.logs[:-_MAX_LOG_LINES]


class ResolveRequest(BaseModel):
    path: str


class JobRequest(BaseModel):
    video_path: str
    lang: str = "zh"
    source: str | None = None
    fast_mode: bool = False
    format: str = "both"
    title: str | None = None


def create_app(cfg: AppConfig) -> FastAPI:
    app = FastAPI(title="Video Caption Assistant")
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    # One job at a time: Whisper on GPU cannot share memory well, and LM Studio is a single server
    executor = ThreadPoolExecutor(max_workers=1)
    log_handler = _JobLogHandler()
    logging.getLogger("video_caption").addHandler(log_handler)

    def _run(job: Job, req: JobRequest) -> None:
        log_handler.job = job
        try:
            job.outputs = [
                to_display_path(Path(p))
                for p in run_pipeline(
                    cfg,
                    Path(job.video_path),
                    lang=req.lang,
                    source=req.source or None,
                    fast_mode=req.fast_mode,
                    fmt=req.format,
                    title=req.title or None,
                    on_phase=lambda phase: setattr(job, "status", phase),
                )
            ]
            job.status = "done"
        except PipelineError as e:
            job.status, job.error = "failed", str(e)
            log.error("%s", e)
        except Exception as e:
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            log.exception("Job %s crashed", job.id)
        finally:
            log_handler.job = None

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/info")
    def info() -> dict:
        return {"wsl": is_wsl()}

    @app.get("/api/browse")
    def browse(path: str | None = None) -> dict:
        if not path:
            return {
                "path": None,
                "display": "Computer",
                "parent": None,
                "dirs": [{"path": str(p), "name": to_display_path(p)} for p in browse_roots()],
                "videos": [],
            }
        folder = to_local_path(path)
        if not folder.is_dir():
            raise HTTPException(404, f"Folder not found: {path}")
        dirs, videos = [], []
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            raise HTTPException(403, f"No permission to read: {to_display_path(folder)}")
        for entry in entries:
            if entry.name.startswith((".", "$")):
                continue
            try:
                if entry.is_dir():
                    dirs.append({"path": str(entry), "name": entry.name})
                elif is_video(entry):
                    videos.append({"path": str(entry), "name": entry.name, "size": entry.stat().st_size})
            except OSError:
                continue  # broken links and system files Windows will not let us stat
        is_root = folder.parent == folder or folder in browse_roots()
        return {
            "path": str(folder),
            "display": to_display_path(folder),
            "parent": None if is_root else str(folder.parent),
            "dirs": dirs,
            "videos": videos,
        }

    @app.post("/api/resolve")
    def resolve(req: ResolveRequest) -> dict:
        video = to_local_path(req.path)
        try:
            validate_video(video)
        except PipelineError as e:
            raise HTTPException(400, str(e))
        video = video.resolve()
        return {
            "path": str(video),
            "display": to_display_path(video),
            "output_dir": to_display_path(video.parent),
            "existing": [p.name for p in (video.with_suffix(".srt"), video.with_suffix(".ass")) if p.exists()],
        }

    @app.post("/api/jobs")
    def create_job(req: JobRequest) -> dict:
        if req.format not in ("srt", "ass", "both"):
            raise HTTPException(400, "format must be srt, ass, or both")
        video = to_local_path(req.video_path)
        try:
            validate_video(video)
        except PipelineError as e:
            raise HTTPException(400, str(e))
        job = Job(id=uuid.uuid4().hex[:12], video_path=str(video.resolve()))
        with jobs_lock:
            jobs[job.id] = job
        executor.submit(_run, job, req)
        log.info("Queued job %s for %s", job.id, job.video_path)
        return {"id": job.id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return {
            "id": job.id,
            "video": to_display_path(Path(job.video_path)),
            "status": job.status,
            "error": job.error,
            "outputs": job.outputs,
            "logs": job.logs[-200:],
        }

    return app
