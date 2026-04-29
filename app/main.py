from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from app.models import JobRecord, JobStatus
from app.services.converter import PDFToLatexConverter
from app.services.latex_compiler import LatexCompiler

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage" / "jobs"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PDF to LaTeX Platform", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
converter = PDFToLatexConverter()
latex_compiler = LatexCompiler()
MAX_DEEPSEEK_LIVE_OUTPUT_CHARS = 16000

jobs: dict[str, JobRecord] = {}
job_lock = Lock()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    recognition_mode: str = Form("local"),
    deepseek_api_key: str = Form(""),
) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    normalized_mode = recognition_mode.strip().lower()
    if normalized_mode not in {"local", "deepseek"}:
        raise HTTPException(status_code=400, detail="recognition_mode must be local or deepseek.")

    use_deepseek = normalized_mode == "deepseek"
    key_for_job = deepseek_api_key.strip()

    if use_deepseek and not converter.supports_refiner(key_for_job):
        raise HTTPException(
            status_code=400,
            detail="DeepSeek mode requires DEEPSEEK_API_KEY.",
        )

    job_id = uuid.uuid4().hex
    job_dir = STORAGE_DIR / job_id
    output_dir = job_dir / "output"
    input_pdf = job_dir / "input.pdf"

    job_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    input_pdf.write_bytes(content)

    now = datetime.utcnow()
    record = JobRecord(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=now,
        updated_at=now,
        input_pdf=input_pdf,
        used_deepseek=use_deepseek,
        progress=1,
        current_step="Job created, waiting in queue",
        deepseek_live_output="",
    )

    with job_lock:
        jobs[job_id] = record

    background_tasks.add_task(
        _process_job,
        job_id,
        input_pdf,
        output_dir,
        use_deepseek,
        key_for_job,
    )
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    record = _get_job_or_404(job_id)
    return record.to_dict()


@app.get("/api/jobs/{job_id}/download")
def download_output(job_id: str) -> FileResponse:
    record = _get_job_or_404(job_id)
    if record.status != JobStatus.COMPLETED or not record.output_zip or not record.output_zip.exists():
        raise HTTPException(status_code=409, detail="Result is not ready yet.")

    return FileResponse(
        path=str(record.output_zip),
        media_type="application/zip",
        filename=f"latex_result_{job_id}.zip",
    )


@app.get("/api/jobs/{job_id}/download-pdf")
def download_compiled_pdf(job_id: str) -> FileResponse:
    record = _get_job_or_404(job_id)
    if record.status != JobStatus.COMPLETED or not record.compiled_pdf or not record.compiled_pdf.exists():
        raise HTTPException(status_code=409, detail="Compiled PDF is not ready yet.")

    return FileResponse(
        path=str(record.compiled_pdf),
        media_type="application/pdf",
        filename=f"latex_result_{job_id}.pdf",
    )


def _get_job_or_404(job_id: str) -> JobRecord:
    with job_lock:
        record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return record


def _process_job(
    job_id: str,
    input_pdf: Path,
    output_dir: Path,
    use_deepseek: bool,
    deepseek_api_key: str,
) -> None:
    _set_job_state(
        job_id,
        status=JobStatus.PROCESSING,
        progress=3,
        current_step="Initializing conversion pipeline",
    )

    def report_progress(
        progress: int,
        step: str,
        current_page: int | None = None,
        total_pages: int | None = None,
    ) -> None:
        _set_job_state(
            job_id,
            status=JobStatus.PROCESSING,
            progress=progress,
            current_step=step,
            current_page=current_page,
            total_pages=total_pages,
        )

    def report_deepseek_stream(chunk: str) -> None:
        if not chunk:
            return
        _append_deepseek_live_output(job_id, chunk)

    try:
        stats = converter.convert(
            input_pdf,
            output_dir,
            use_refiner=use_deepseek,
            deepseek_api_key=deepseek_api_key,
            progress_callback=report_progress,
            live_output_callback=report_deepseek_stream if use_deepseek else None,
        )

        compile_success = False
        compile_engine = ""
        compile_log = ""
        compiled_pdf: Path | None = None
        _set_job_state(
            job_id,
            status=JobStatus.PROCESSING,
            progress=94,
            current_step="Compiling LaTeX to PDF",
        )
        try:
            compile_result = latex_compiler.compile_main_tex(output_dir / "main.tex")
            compile_success = True
            compile_engine = compile_result.engine
            compile_log = compile_result.log_excerpt
            compiled_pdf = compile_result.pdf_path
            _set_job_state(
                job_id,
                status=JobStatus.PROCESSING,
                progress=96,
                current_step=f"LaTeX compiled successfully ({compile_engine})",
                compiled_pdf=compiled_pdf,
                compile_success=True,
                compile_engine=compile_engine,
                compile_log=compile_log,
            )
        except Exception as compile_exc:  # noqa: BLE001
            compile_success = False
            compile_engine = ""
            compile_log = str(compile_exc)
            _set_job_state(
                job_id,
                status=JobStatus.PROCESSING,
                progress=96,
                current_step="LaTeX compile failed, packaging source project",
                compile_success=False,
                compile_engine=compile_engine,
                compile_log=compile_log,
            )

        _set_job_state(
            job_id,
            status=JobStatus.PROCESSING,
            progress=97,
            current_step="Packaging LaTeX output",
        )
        zip_base = output_dir.parent / "latex_output"
        zip_path = Path(
            shutil.make_archive(
                base_name=str(zip_base),
                format="zip",
                root_dir=str(output_dir),
            )
        )

        _set_job_state(
            job_id,
            status=JobStatus.COMPLETED,
            output_zip=zip_path,
            pages=stats.get("pages"),
            extracted_images=stats.get("extracted_images"),
            compiled_pdf=compiled_pdf,
            compile_success=compile_success,
            compile_engine=compile_engine,
            compile_log=compile_log,
            progress=100,
            current_step="Finished. Downloads are ready",
        )
    except Exception as exc:  # noqa: BLE001
        _set_job_state(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            current_step="Conversion failed",
        )


def _set_job_state(
    job_id: str,
    status: JobStatus,
    output_zip: Path | None = None,
    error: str | None = None,
    pages: int | None = None,
    extracted_images: int | None = None,
    compiled_pdf: Path | None = None,
    compile_success: bool | None = None,
    compile_engine: str | None = None,
    compile_log: str | None = None,
    progress: int | None = None,
    current_step: str | None = None,
    current_page: int | None = None,
    total_pages: int | None = None,
    deepseek_live_output: str | None = None,
) -> None:
    with job_lock:
        record = jobs[job_id]
        record.status = status
        record.updated_at = datetime.utcnow()
        if output_zip is not None:
            record.output_zip = output_zip
        if error is not None:
            record.error = error
        if pages is not None:
            record.pages = pages
        if extracted_images is not None:
            record.extracted_images = extracted_images
        if compiled_pdf is not None:
            record.compiled_pdf = compiled_pdf
        if compile_success is not None:
            record.compile_success = compile_success
        if compile_engine is not None:
            record.compile_engine = compile_engine
        if compile_log is not None:
            record.compile_log = compile_log
        if progress is not None:
            record.progress = max(0, min(100, int(progress)))
        if current_step is not None:
            record.current_step = current_step
        if current_page is not None:
            record.current_page = current_page
        if total_pages is not None:
            record.total_pages = total_pages
        if deepseek_live_output is not None:
            record.deepseek_live_output = deepseek_live_output


def _append_deepseek_live_output(job_id: str, chunk: str) -> None:
    with job_lock:
        record = jobs[job_id]
        merged = f"{record.deepseek_live_output}{chunk}"
        if len(merged) > MAX_DEEPSEEK_LIVE_OUTPUT_CHARS:
            merged = merged[-MAX_DEEPSEEK_LIVE_OUTPUT_CHARS:]
        record.deepseek_live_output = merged
        record.updated_at = datetime.utcnow()
