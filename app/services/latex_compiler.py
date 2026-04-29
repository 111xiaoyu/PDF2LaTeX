from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LatexCompileResult:
    success: bool
    engine: str
    pdf_path: Path | None
    command: str
    log_excerpt: str


@dataclass
class _CompilePlan:
    engine: str
    steps: list[list[str]]


class LatexCompiler:
    def __init__(self, timeout_seconds: int | None = None) -> None:
        raw_timeout = os.getenv("LATEX_COMPILE_TIMEOUT", "240").strip()
        env_timeout = 240
        if raw_timeout:
            try:
                env_timeout = int(raw_timeout)
            except ValueError:
                env_timeout = 240

        effective_timeout = timeout_seconds if timeout_seconds is not None else env_timeout
        self._timeout_seconds = max(30, min(1800, int(effective_timeout)))

    def compile_main_tex(self, tex_path: Path) -> LatexCompileResult:
        if not tex_path.exists():
            raise RuntimeError(f"LaTeX source file not found: {tex_path}")

        if tex_path.suffix.lower() != ".tex":
            raise RuntimeError(f"Expected a .tex file, got: {tex_path.name}")

        work_dir = tex_path.parent
        tex_name = tex_path.name
        pdf_path = work_dir / f"{tex_path.stem}.pdf"
        plans = self._build_plans(tex_name=tex_name)

        if not plans:
            raise RuntimeError(
                "No LaTeX compiler was found. Install TeX Live (or MiKTeX) and ensure latexmk/xelatex/pdflatex is in PATH."
            )

        failure_logs: list[str] = []
        for plan in plans:
            result = self._run_plan(plan=plan, work_dir=work_dir)
            if result.success and pdf_path.exists():
                return LatexCompileResult(
                    success=True,
                    engine=plan.engine,
                    pdf_path=pdf_path,
                    command=result.command,
                    log_excerpt=result.log_excerpt,
                )

            failure_logs.append(
                f"[{plan.engine}]\n{result.log_excerpt.strip() or 'Compilation failed without output.'}"
            )

        merged_failures = "\n\n".join(failure_logs)
        raise RuntimeError(
            "LaTeX compilation failed with all available engines. "
            "Please inspect the compile log for the exact TeX line-level error.\n\n"
            f"{merged_failures}"
        )

    def _build_plans(self, tex_name: str) -> list[_CompilePlan]:
        plans: list[_CompilePlan] = []

        if shutil.which("latexmk"):
            plans.append(
                _CompilePlan(
                    engine="latexmk(xelatex)",
                    steps=[
                        [
                            "latexmk",
                            "-xelatex",
                            "-interaction=nonstopmode",
                            "-halt-on-error",
                            "-file-line-error",
                            tex_name,
                        ]
                    ],
                )
            )

        if shutil.which("xelatex"):
            xelatex_cmd = [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                tex_name,
            ]
            plans.append(_CompilePlan(engine="xelatex", steps=[xelatex_cmd, xelatex_cmd]))

        if shutil.which("pdflatex"):
            pdflatex_cmd = [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                tex_name,
            ]
            plans.append(_CompilePlan(engine="pdflatex", steps=[pdflatex_cmd, pdflatex_cmd]))

        return plans

    def _run_plan(self, plan: _CompilePlan, work_dir: Path) -> LatexCompileResult:
        collected_output: list[str] = []
        joined_commands: list[str] = []

        for command in plan.steps:
            joined_commands.append(" ".join(command))
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return LatexCompileResult(
                    success=False,
                    engine=plan.engine,
                    pdf_path=None,
                    command=" ; ".join(joined_commands),
                    log_excerpt=(
                        f"Compilation timed out after {self._timeout_seconds} seconds. "
                        "Try increasing LATEX_COMPILE_TIMEOUT or simplify the document."
                    ),
                )

            output = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
            collected_output.append(output)

            if proc.returncode != 0:
                excerpt = self._tail_lines("\n".join(collected_output), max_lines=140)
                return LatexCompileResult(
                    success=False,
                    engine=plan.engine,
                    pdf_path=None,
                    command=" ; ".join(joined_commands),
                    log_excerpt=excerpt,
                )

        excerpt = self._tail_lines("\n".join(collected_output), max_lines=80)
        return LatexCompileResult(
            success=True,
            engine=plan.engine,
            pdf_path=None,
            command=" ; ".join(joined_commands),
            log_excerpt=excerpt,
        )

    def _tail_lines(self, text: str, max_lines: int) -> str:
        lines = [line.rstrip() for line in str(text or "").splitlines()]
        if len(lines) <= max_lines:
            return "\n".join(lines).strip()
        tail = lines[-max_lines:]
        return "\n".join(tail).strip()
