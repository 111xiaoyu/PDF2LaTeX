# PDF to LaTeX Platform

A web platform that accepts a PDF upload and generates a ZIP package containing:

- `main.tex`
- `figures/` (extracted images)
- `main.pdf` (when LaTeX compilation succeeds)

## Features

- FastAPI backend with asynchronous job execution
- Browser UI for upload, progress polling, and one-click download
- Real-time progress indicator (percentage, current step, current page/total pages)
- PDF text extraction with improved paragraph reconstruction and heading detection
- Figure extraction from PDF image objects
- Two recognition modes: local recognition or DeepSeek recognition
- DeepSeek API key can be provided per upload job from the frontend
- In DeepSeek mode, structural LaTeX generation is done by DeepSeek directly (no local structure reconstruction)
- In DeepSeek mode, conversion runs in direct page-by-page DeepSeek recognition mode by default
- In DeepSeek mode, a side panel shows live DeepSeek output text during generation
- In DeepSeek mode, live output supports both `content` and `reasoning_content` stream fields and emits heartbeat text when stream is temporarily silent
- In DeepSeek mode, figures are exported as `figures/figure1.png`, `figures/figure2.png`, ...
- In DeepSeek mode, `\includegraphics` width is normalized to `0.5\linewidth`
- In DeepSeek mode, page-by-page recovery is used directly to reduce full-document truncation risk on long papers
- In DeepSeek mode, figure labels/references are normalized to `\label{fig:N}` and `\ref{fig:N}`
- In DeepSeek mode, table labels/references are normalized to `\label{tab:N}` and `\ref{tab:N}`
- In DeepSeek mode, common `tabular` column-count mismatch issues are auto-corrected
- In DeepSeek mode, algorithm pseudocode packages are auto-injected when `algorithm`/`algorithmic`/`\State`-style commands are detected
- ZIP export of LaTeX project files
- Automatic LaTeX compilation (`latexmk`/`xelatex`/`pdflatex`) with compile log feedback

## Project Structure

- `app/main.py`: FastAPI app and job APIs
- `app/services/converter.py`: PDF-to-LaTeX conversion pipeline
- `app/models.py`: In-memory job record model
- `app/templates/index.html`: Frontend page
- `storage/jobs/`: Runtime job workspace

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:

   ```bash
   python run.py
   ```

4. Open browser:

   - http://127.0.0.1:8000

## Optional: DeepSeek Configuration

DeepSeek mode can work in either way:

- Frontend prompt input per upload (recommended)
- Server environment variable fallback

Environment variables (optional):

- DEEPSEEK_API_KEY: your API key
- DEEPSEEK_MODEL: optional, default is deepseek-v4-pro
- DEEPSEEK_BASE_URL: optional, default is https://api.deepseek.com
- DEEPSEEK_MAX_TOKENS: optional, default is 8192 (valid range 1-8192)
- LATEX_COMPILE_TIMEOUT: optional, default is 240 seconds

PowerShell example:

   $env:DEEPSEEK_API_KEY = "your_key_here"
   $env:DEEPSEEK_MODEL = "deepseek-v4-pro"
   python run.py

## LaTeX Compiler Requirements

To enable automatic PDF output, install one of the following and add it to `PATH`:

- `latexmk` (recommended, uses `xelatex` internally)
- `xelatex`
- `pdflatex`

If no compiler is found, or compilation fails, you can still download the ZIP and compile `main.tex` manually.

## API Endpoints

- `POST /api/jobs` : upload a PDF file
   - form field `recognition_mode=local|deepseek`
   - optional form field `deepseek_api_key=...` (required in deepseek mode unless server env key is set)
- `GET /api/jobs/{job_id}` : query conversion status
- `GET /api/jobs/{job_id}/download` : download ZIP output
- `GET /api/jobs/{job_id}/download-pdf` : download compiled PDF (when available)

## Output Details

The generated ZIP includes:

- `main.tex`
- `figures/*`
- `main.pdf` (if compilation succeeds)

Current conversion strategy:

- Uses line merging, de-hyphenation, and heading heuristics for better structure
- Exports images that are embedded as PDF image objects
- Escapes LaTeX special characters in extracted text
- DeepSeek mode processes PDF pages sequentially with DeepSeek and then applies unified LaTeX post-processing

## Limitations

- Complex layouts (multi-column, dense formulas, nested tables) are not fully reconstructed
- Scanned PDFs (image-only pages) are not OCR-processed in this version
- Figure captions are auto-generated and should be manually revised

## Roadmap

- Add OCR fallback for scanned PDFs
- Better multi-column reading order reconstruction
- Mathematical expression reconstruction via OCR + LaTeX models
- Optional bibliography extraction to `.bib`
