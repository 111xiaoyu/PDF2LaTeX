# PDF to LaTeX Platform

A web platform that accepts a PDF upload and generates a ZIP package containing:

- `main.tex`
- `figures/` (extracted images)

## Features

- FastAPI backend with asynchronous job execution
- Browser UI for upload, progress polling, and one-click download
- Real-time progress indicator (percentage, current step, current page/total pages)
- PDF text extraction with improved paragraph reconstruction and heading detection
- Figure extraction from PDF image objects
- Two recognition modes: local recognition or DeepSeek recognition
- DeepSeek API key can be provided per upload job from the frontend
- In DeepSeek mode, structural LaTeX generation is done by DeepSeek directly (no local structure reconstruction)
- In DeepSeek mode, a side panel shows live DeepSeek output text during generation
- In DeepSeek mode, figures are exported as `figures/figure1.png`, `figures/figure2.png`, ...
- In DeepSeek mode, `\includegraphics` width is normalized to `0.5\linewidth`
- In DeepSeek mode, if full-document output is detected as incomplete, the system retries and falls back to page-by-page recovery automatically
- In DeepSeek mode, figure labels/references are normalized to `\label{fig:N}` and `\ref{fig:N}`
- In DeepSeek mode, table labels/references are normalized to `\label{tab:N}` and `\ref{tab:N}`
- In DeepSeek mode, common `tabular` column-count mismatch issues are auto-corrected
- ZIP export of LaTeX project files

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
- DEEPSEEK_MODEL: optional, default is deepseek-chat
- DEEPSEEK_BASE_URL: optional, default is https://api.deepseek.com
- DEEPSEEK_MAX_TOKENS: optional, default is 8192 (valid range 1-8192)

PowerShell example:

   $env:DEEPSEEK_API_KEY = "your_key_here"
   $env:DEEPSEEK_MODEL = "deepseek-chat"
   python run.py

## API Endpoints

- `POST /api/jobs` : upload a PDF file
   - form field `recognition_mode=local|deepseek`
   - optional form field `deepseek_api_key=...` (required in deepseek mode unless server env key is set)
- `GET /api/jobs/{job_id}` : query conversion status
- `GET /api/jobs/{job_id}/download` : download ZIP output

## Output Details

The generated ZIP includes:

- `main.tex`
- `figures/*`

Current conversion strategy:

- Uses line merging, de-hyphenation, and heading heuristics for better structure
- Exports images that are embedded as PDF image objects
- Escapes LaTeX special characters in extracted text
- DeepSeek mode sends full-document raw text + figure manifest to DeepSeek for full LaTeX generation

## Limitations

- Complex layouts (multi-column, dense formulas, nested tables) are not fully reconstructed
- Scanned PDFs (image-only pages) are not OCR-processed in this version
- Figure captions are auto-generated and should be manually revised

## Roadmap

- Add OCR fallback for scanned PDFs
- Better multi-column reading order reconstruction
- Mathematical expression reconstruction via OCR + LaTeX models
- Optional bibliography extraction to `.bib`
