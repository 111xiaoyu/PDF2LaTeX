from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    input_pdf: Path
    used_deepseek: bool = False
    progress: int = 0
    current_step: str = "Waiting to start"
    current_page: int | None = None
    total_pages: int | None = None
    deepseek_live_output: str = ""
    output_zip: Path | None = None
    compiled_pdf: Path | None = None
    compile_success: bool | None = None
    compile_engine: str | None = None
    compile_log: str | None = None
    error: str | None = None
    pages: int | None = None
    extracted_images: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        payload["input_pdf"] = str(self.input_pdf)
        payload["output_zip"] = str(self.output_zip) if self.output_zip else None
        payload["compiled_pdf"] = str(self.compiled_pdf) if self.compiled_pdf else None
        return payload
