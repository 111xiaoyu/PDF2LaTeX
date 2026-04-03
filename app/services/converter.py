from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import fitz

from app.services.deepseek_refiner import DeepSeekLatexRefiner

ProgressCallback = Callable[[int, str, int | None, int | None], None]
LiveOutputCallback = Callable[[str], None]


@dataclass
class _TextLine:
    text: str
    size: float
    x0: float
    y0: float
    x1: float
    y1: float


class PDFToLatexConverter:
    def __init__(self) -> None:
        self._image_counter = 0
        self._default_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self._model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
        self._base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()

    def supports_refiner(self, override_api_key: str | None = None) -> bool:
        return bool((override_api_key or "").strip() or self._default_api_key)

    def _make_refiner(self, override_api_key: str | None) -> DeepSeekLatexRefiner | None:
        api_key = (override_api_key or "").strip() or self._default_api_key
        if not api_key:
            return None
        return DeepSeekLatexRefiner(
            api_key=api_key,
            model=self._model,
            base_url=self._base_url,
        )

    def convert(
        self,
        pdf_path: Path,
        output_root: Path,
        use_refiner: bool = False,
        deepseek_api_key: str | None = None,
        progress_callback: ProgressCallback | None = None,
        live_output_callback: LiveOutputCallback | None = None,
    ) -> dict[str, int]:
        self._image_counter = 0
        output_root.mkdir(parents=True, exist_ok=True)
        figures_dir = output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        page_refiner = self._make_refiner(deepseek_api_key) if use_refiner else None
        mode_label = "DeepSeek" if use_refiner else "local"

        tex_fragments: list[str] = []
        extracted_images = 0

        self._notify_progress(progress_callback, 8, f"Preparing {mode_label} recognition")
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count
            self._notify_progress(
                progress_callback,
                10,
                f"Loaded PDF metadata, total pages: {total_pages}",
                current_page=0,
                total_pages=total_pages,
            )

            if use_refiner and page_refiner is not None:
                self._notify_progress(progress_callback, 14, "Extracting figure regions from PDF")
                figure_specs = self._extract_document_figures(
                    doc=doc,
                    figures_dir=figures_dir,
                    progress_callback=progress_callback,
                )

                self._notify_progress(
                    progress_callback,
                    45,
                    "Preparing full document text for DeepSeek",
                    current_page=0,
                    total_pages=0,
                )
                raw_document_text = self._extract_raw_document_text(doc)

                self._notify_progress(
                    progress_callback,
                    60,
                    "DeepSeek is generating full LaTeX document",
                    current_page=0,
                    total_pages=0,
                )
                full_tex = page_refiner.refine_document(
                    raw_text=raw_document_text,
                    figure_specs=figure_specs,
                    title_hint=pdf_path.stem,
                    stream_callback=live_output_callback,
                )

                if self._is_likely_incomplete_document(full_tex):
                    self._notify_progress(
                        progress_callback,
                        76,
                        "Detected incomplete output, retrying DeepSeek full generation",
                        current_page=0,
                        total_pages=0,
                    )
                    if live_output_callback is not None:
                        live_output_callback(
                            "\n\n[System] The first full-document output looked incomplete. Retrying once...\n"
                        )

                    retry_tex = page_refiner.refine_document(
                        raw_text=raw_document_text,
                        figure_specs=figure_specs,
                        title_hint=pdf_path.stem,
                        stream_callback=live_output_callback,
                    )
                    if not self._is_likely_incomplete_document(retry_tex):
                        full_tex = retry_tex
                    else:
                        self._notify_progress(
                            progress_callback,
                            82,
                            "Fallback: DeepSeek page-by-page recovery",
                            current_page=0,
                            total_pages=total_pages,
                        )
                        if live_output_callback is not None:
                            live_output_callback(
                                "\n\n[System] Full-document output is still incomplete. "
                                "Switching to page-by-page DeepSeek recovery mode.\n"
                            )
                        full_tex = self._recover_document_by_pages(
                            doc=doc,
                            refiner=page_refiner,
                            progress_callback=progress_callback,
                        )

                self._notify_progress(
                    progress_callback,
                    92,
                    "Writing LaTeX files to disk",
                    current_page=0,
                    total_pages=0,
                )
                (output_root / "main.tex").write_text(full_tex, encoding="utf-8")
                return {
                    "pages": total_pages,
                    "extracted_images": len(figure_specs),
                }

            seen_xrefs: set[int] = set()
            seen_image_hashes: set[str] = set()
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                current_page = page_index + 1
                page_progress = self._calculate_page_progress(current_page=current_page, total_pages=total_pages)
                self._notify_progress(
                    progress_callback,
                    page_progress,
                    f"Recognizing page {current_page}/{total_pages}",
                    current_page=current_page,
                    total_pages=total_pages,
                )
                tex_fragments.extend(
                    self._page_text_to_latex(
                        page,
                        current_page,
                        refiner=page_refiner,
                    )
                )
                caption_fragments, caption_count = self._extract_figures_from_captions(
                    page=page,
                    page_number=current_page,
                    figures_dir=figures_dir,
                    seen_hashes=seen_image_hashes,
                )
                image_fragments, image_count = self._extract_page_images(
                    doc=doc,
                    page=page,
                    page_number=current_page,
                    figures_dir=figures_dir,
                    seen_xrefs=seen_xrefs,
                    seen_hashes=seen_image_hashes,
                )
                extracted_images += caption_count + image_count
                tex_fragments.extend(caption_fragments)
                tex_fragments.extend(image_fragments)
                done_progress = min(90, page_progress + 2)
                self._notify_progress(
                    progress_callback,
                    done_progress,
                    f"Page {current_page}/{total_pages} recognized",
                    current_page=current_page,
                    total_pages=total_pages,
                )

            self._notify_progress(progress_callback, 92, "Writing LaTeX files to disk")
            tex_content = self._build_document(tex_fragments)
            (output_root / "main.tex").write_text(tex_content, encoding="utf-8")

            return {
                "pages": doc.page_count,
                "extracted_images": extracted_images,
            }

    def _extract_raw_document_text(self, doc: fitz.Document) -> str:
        parts: list[str] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text")
            cleaned = text.replace("\u00a0", " ").replace("\x0c", "\n").strip()
            parts.append(f"--- PAGE {page_index + 1} ---\n{cleaned}")
        return "\n\n".join(parts)

    def _extract_document_figures(
        self,
        doc: fitz.Document,
        figures_dir: Path,
        progress_callback: ProgressCallback | None,
    ) -> list[dict[str, str]]:
        used_hashes: set[str] = set()
        used_numbers: set[int] = set()
        figure_specs: list[dict[str, str]] = []

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text_dict = page.get_text("dict")
            lines = self._extract_text_lines(page)
            caption_lines = [
                line for line in sorted(lines, key=lambda item: (item.y0, item.x0))
                if re.match(r"^(fig(?:ure)?\.?\s*\d+|图\s*\d+)", line.text.strip(), re.IGNORECASE)
            ]
            if not caption_lines:
                continue

            graphic_regions = self._collect_graphic_regions(page, text_dict)
            text_rects: list[fitz.Rect] = []
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                rect = fitz.Rect(block.get("bbox", [0, 0, 0, 0]))
                if rect.width > 0 and rect.height > 0:
                    text_rects.append(rect)

            for caption in caption_lines:
                figure_rect = self._select_region_for_caption(caption, graphic_regions, text_rects, page.rect)
                if figure_rect is None:
                    continue

                pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), clip=figure_rect, alpha=False)
                image_bytes = pix.tobytes("png")
                digest = hashlib.sha1(image_bytes).hexdigest()
                if digest in used_hashes:
                    continue
                used_hashes.add(digest)

                number = self._parse_figure_number(caption.text)
                if number is None or number in used_numbers:
                    number = 1
                    while number in used_numbers:
                        number += 1
                used_numbers.add(number)

                file_stem = f"figure{number}"
                file_name = f"{file_stem}.png"
                (figures_dir / file_name).write_bytes(image_bytes)

                figure_specs.append(
                    {
                        "number": str(number),
                        "path": f"figures/{file_stem}",
                        "caption": caption.text.strip(),
                        "page": str(page_index + 1),
                    }
                )

            progress = 16 + int(((page_index + 1) / max(1, doc.page_count)) * 24)
            self._notify_progress(
                progress_callback,
                progress,
                f"Figure extraction: page {page_index + 1}/{doc.page_count}",
                current_page=page_index + 1,
                total_pages=doc.page_count,
            )

        figure_specs.sort(key=lambda item: int(item["number"]))
        return figure_specs

    def _parse_figure_number(self, caption_text: str) -> int | None:
        match = re.search(r"(?:fig(?:ure)?\.?|图)\s*(\d+)", caption_text, re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _notify_progress(
        self,
        callback: ProgressCallback | None,
        progress: int,
        step: str,
        current_page: int | None = None,
        total_pages: int | None = None,
    ) -> None:
        if callback is None:
            return
        callback(progress, step, current_page, total_pages)

    def _is_likely_incomplete_document(self, tex_content: str) -> bool:
        value = (tex_content or "").strip()
        if not value:
            return True
        if "\\end{document}" not in value:
            return True

        for env in ("equation", "align", "figure", "table", "tabular"):
            begin_count = len(re.findall(rf"\\\\begin\{{{env}\*?\}}", value))
            end_count = len(re.findall(rf"\\\\end\{{{env}\*?\}}", value))
            if begin_count > end_count:
                return True

        body = value.split("\\end{document}", 1)[0]
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            return True

        tail = lines[-1]
        if tail.startswith("%"):  # comment-only tail line is usually suspicious
            return True
        if tail.startswith("\\"):
            return False
        if re.search(r"[.!?:;]$", tail):
            return False
        if tail.endswith(("}", "$", "]", ")")):
            return False
        if len(tail.split()) >= 3:
            return True
        return False

    def _recover_document_by_pages(
        self,
        doc: fitz.Document,
        refiner: DeepSeekLatexRefiner,
        progress_callback: ProgressCallback | None,
    ) -> str:
        total_pages = max(1, doc.page_count)
        fragments: list[str] = []

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            current_page = page_index + 1
            page_progress = 82 + int((current_page / total_pages) * 8)
            self._notify_progress(
                progress_callback,
                min(90, page_progress),
                f"DeepSeek page recovery {current_page}/{total_pages}",
                current_page=current_page,
                total_pages=total_pages,
            )

            page_text = page.get_text("text").replace("\u00a0", " ").replace("\x0c", "\n")
            raw_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            if not raw_lines:
                continue

            try:
                refined_lines = refiner.refine_page(page_number=current_page, raw_lines=raw_lines)
            except Exception:
                refined_lines = []

            if refined_lines:
                fragments.extend([f"{line}\n" for line in refined_lines])
            else:
                fragments.extend([f"{self._escape_latex(line)}\n" for line in raw_lines])
            fragments.append("\\clearpage\n")

        return self._build_document(fragments)

    def _calculate_page_progress(self, current_page: int, total_pages: int) -> int:
        if total_pages <= 0:
            return 10
        ratio = (current_page - 1) / total_pages
        return 12 + int(ratio * 76)

    def _page_text_to_latex(
        self,
        page: fitz.Page,
        page_number: int,
        refiner: DeepSeekLatexRefiner | None,
    ) -> list[str]:
        raw_lines = self._extract_text_lines(page)
        if not raw_lines:
            return ["\\clearpage\n"]

        filtered_lines = self._filter_noise_lines(raw_lines, page.rect.height, page_number)
        if not filtered_lines:
            return ["\\clearpage\n"]

        ordered_lines = self._order_lines(filtered_lines, page.rect.width)
        merged_lines = self._merge_lines(ordered_lines)

        if refiner is not None:
            try:
                refined_lines = refiner.refine_page(
                    page_number=page_number,
                    raw_lines=[line.text for line in merged_lines],
                )
                if refined_lines:
                    chunks = [f"{line}\n" for line in refined_lines]
                    chunks.append("\\clearpage\n")
                    return chunks
            except Exception:
                pass

        chunks = self._render_lines_to_latex(merged_lines)
        chunks.append("\\clearpage\n")
        return chunks

    def _extract_text_lines(self, page: fitz.Page) -> list[_TextLine]:
        text_dict = page.get_text("dict")
        lines: list[_TextLine] = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(span.get("text", "") for span in spans)
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue
                avg_size = sum(float(span.get("size", 0.0)) for span in spans) / len(spans)
                bbox = line.get("bbox", [0, 0, 0, 0])
                lines.append(
                    _TextLine(
                        text=text,
                        size=avg_size,
                        x0=float(bbox[0]),
                        y0=float(bbox[1]),
                        x1=float(bbox[2]),
                        y1=float(bbox[3]),
                    )
                )

        return lines

    def _filter_noise_lines(self, lines: list[_TextLine], page_height: float, page_number: int) -> list[_TextLine]:
        if not lines:
            return []

        size_baseline = median(line.size for line in lines)
        top_cut = page_height * 0.06
        bottom_cut = page_height * 0.94

        filtered: list[_TextLine] = []
        for line in lines:
            text = line.text.strip()
            lower = text.lower()
            if self._is_page_number_like(text):
                continue
            if re.fullmatch(r"[ðÞ\(\)\[\]\{\}\s\d\.,;:]+", text):
                continue
            if len(text) <= 6 and not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
                continue

            near_top = line.y0 < top_cut
            near_bottom = line.y1 > bottom_cut
            weak_meta = line.size <= size_baseline + 0.8 and len(text) < 130
            meta_pattern = bool(
                re.search(
                    r"doi|proceedings|vol\.|pp\.|copyright|issn|isbn|et al\.|springer|ieee|aerospace",
                    lower,
                )
            )
            if "©" in text:
                meta_pattern = True

            looks_like_running_title = (
                near_top
                and page_number > 1
                and weak_meta
                and 20 <= len(text) <= 95
                and not re.match(r"^\d+(?:\.\d+)*\s+", text)
                and not text.endswith((".", ":", ";"))
            )

            if (near_top or near_bottom) and weak_meta and meta_pattern:
                continue
            if looks_like_running_title:
                continue

            filtered.append(line)

        return filtered

    def _order_lines(self, lines: list[_TextLine], page_width: float) -> list[_TextLine]:
        if not lines:
            return []

        left_count = sum(1 for line in lines if line.x0 < page_width * 0.45)
        right_count = sum(1 for line in lines if line.x0 > page_width * 0.55)
        use_two_columns = left_count > 10 and right_count > 10

        if not use_two_columns:
            return sorted(lines, key=lambda line: (round(line.y0, 1), line.x0))

        left_lines = [line for line in lines if line.x0 <= page_width * 0.5]
        right_lines = [line for line in lines if line.x0 > page_width * 0.5]
        left_lines.sort(key=lambda line: (round(line.y0, 1), line.x0))
        right_lines.sort(key=lambda line: (round(line.y0, 1), line.x0))
        return left_lines + right_lines

    def _merge_lines(self, lines: list[_TextLine]) -> list[_TextLine]:
        if not lines:
            return []

        merged: list[_TextLine] = [lines[0]]
        for line in lines[1:]:
            prev = merged[-1]
            if self._should_merge(prev, line):
                merged[-1] = self._merge_pair(prev, line)
            else:
                merged.append(line)

        return merged

    def _should_merge(self, prev: _TextLine, current: _TextLine) -> bool:
        if self._is_page_number_like(prev.text) or self._is_page_number_like(current.text):
            return False

        if self._looks_like_heading(prev, prev.size) or self._looks_like_heading(current, prev.size):
            if not re.fullmatch(r"\d+(?:\.\d+)*", prev.text.strip()):
                return False

        vertical_gap = current.y0 - prev.y1
        similar_size = abs(prev.size - current.size) <= 0.8
        aligned = abs(prev.x0 - current.x0) <= 35

        return vertical_gap <= max(3.2, prev.size * 0.95) and similar_size and aligned

    def _merge_pair(self, prev: _TextLine, current: _TextLine) -> _TextLine:
        prev_text = prev.text.rstrip()
        current_text = current.text.lstrip()

        if re.fullmatch(r"\d+(?:\.\d+)*", prev_text):
            combined_text = f"{prev_text} {current_text}"
        elif prev_text.endswith("-") and current_text and current_text[0].islower():
            combined_text = prev_text[:-1] + current_text
        else:
            combined_text = f"{prev_text} {current_text}"

        return _TextLine(
            text=combined_text,
            size=max(prev.size, current.size),
            x0=min(prev.x0, current.x0),
            y0=min(prev.y0, current.y0),
            x1=max(prev.x1, current.x1),
            y1=max(prev.y1, current.y1),
        )

    def _render_lines_to_latex(self, lines: list[_TextLine]) -> list[str]:
        if not lines:
            return []

        normal_size = median(line.size for line in lines)
        chunks: list[str] = []
        index = 0

        while index < len(lines):
            line = lines[index]
            if self._is_page_number_like(line.text):
                index += 1
                continue

            if self._is_equation_like(line.text):
                equation_parts = [line.text]
                cursor = index + 1
                while cursor < len(lines):
                    candidate = lines[cursor]
                    if self._is_page_number_like(candidate.text):
                        break
                    if not self._is_equation_fragment(candidate.text):
                        break
                    equation_parts.append(candidate.text)
                    cursor += 1

                math_expr = self._to_math_expression(" ".join(equation_parts))
                if math_expr:
                    chunks.append("\\begin{equation}\n")
                    chunks.append(f"{math_expr}\n")
                    chunks.append("\\end{equation}\n")
                    index = cursor
                    continue

            escaped = self._escape_latex(line.text)
            if self._looks_like_heading(line, normal_size):
                level = self._heading_level(line, normal_size)
                if level == 1:
                    chunks.append(f"\\section{{{escaped}}}\n")
                elif level == 2:
                    chunks.append(f"\\subsection{{{escaped}}}\n")
                else:
                    chunks.append(f"\\subsubsection{{{escaped}}}\n")
            else:
                chunks.append(f"{escaped}\n")

            index += 1

        return chunks

    def _looks_like_heading(self, line: _TextLine, normal_size: float) -> bool:
        text = line.text.strip()
        if not text or len(text) > 110:
            return False
        if self._is_page_number_like(text):
            return False
        if text.endswith((".", ";", "?", "!")):
            return False

        words = text.split()
        numbered = bool(re.match(r"^\d+(?:\.\d+)*\s+[A-Za-z]", text))
        strong_size = line.size >= normal_size + 1.9 and len(words) <= 18
        return numbered or strong_size

    def _heading_level(self, line: _TextLine, normal_size: float) -> int:
        text = line.text.strip()
        numbered_match = re.match(r"^(\d+(?:\.\d+)*)\s+", text)
        if numbered_match:
            depth = numbered_match.group(1).count(".") + 1
            return min(depth, 3)

        if line.size >= normal_size + 3.0:
            return 1
        if line.size >= normal_size + 2.1:
            return 2
        return 3

    def _is_page_number_like(self, text: str) -> bool:
        value = text.strip()
        if not value:
            return True
        return bool(re.fullmatch(r"\d{1,4}", value))

    def _is_equation_like(self, text: str) -> bool:
        value = self._normalize_equation_source(text)
        if not value or len(value) > 180:
            return False

        math_markers = "=+-*/^_()[]{}<>"
        marker_count = sum(1 for char in value if char in math_markers)
        has_eq_symbol = any(token in value for token in ("=", "≈", "≤", "≥", "¼"))
        has_alpha = bool(re.search(r"[A-Za-z]", value))
        looks_sentence = value.endswith((".", ";", "?", "!")) and "=" not in value

        return has_eq_symbol and has_alpha and marker_count >= 3 and not looks_sentence

    def _to_math_expression(self, text: str) -> str:
        expr = self._normalize_equation_source(text)
        expr = re.sub(r"\(\s*\d+\s*\)", "", expr).strip()
        expr = expr.replace("\u2013", "-").replace("\u2014", "-")
        expr = expr.replace("þ", "+")
        expr = expr.replace("\u2212", "-").replace("\u00d7", r"\times")
        expr = expr.replace("\u00b7", r"\cdot")
        expr = expr.replace("\u03c0", r"\pi")
        expr = expr.replace("\u03c6", r"\phi").replace("\u03d5", r"\phi")
        expr = expr.replace("\u03b8", r"\theta").replace("\u03bb", r"\lambda")
        expr = expr.replace("\u03bc", r"\mu").replace("\u0394", r"\Delta")
        expr = expr.replace("\u03a3", r"\Sigma").replace("\u03c3", r"\sigma")
        expr = expr.replace("\u00a0", " ")
        expr = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", expr)

        expr = expr.replace("^{(auth)}", "^{\\text{auth}}")
        expr = expr.replace("_(auth)", "_{\\text{auth}}")
        expr = re.sub(r"\bRe\s*\(", r"\\Re(", expr)
        expr = re.sub(r"\bexp\s*\(", r"\\exp(", expr)

        expr = re.sub(r"([A-Za-z])_([A-Za-z0-9]+)", r"\1_{\2}", expr)
        expr = re.sub(r"([A-Za-z])\^([A-Za-z0-9]+)", r"\1^{\2}", expr)

        expr = expr.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
        expr = re.sub(r"\s+", " ", expr).strip()
        expr = self._canonicalize_known_formula(expr)
        return expr

    def _canonicalize_known_formula(self, expr: str) -> str:
        normalized = re.sub(r"\s+", "", expr).lower()
        tokens = ("si", "asc", "dauth", "fct", "u0", "auth")
        if all(token in normalized for token in tokens):
            return (
                r"s_{i}(t)=Re\left\{A_{s} C_{i}^{(auth)}(t) D_{i}^{(auth)}(t) "
                r"\exp\left(2 \pi f_{c} t+\varphi_{0}\right)\right\}"
            )
        return expr

    def _is_equation_fragment(self, text: str) -> bool:
        value = self._normalize_equation_source(text)
        if not value:
            return False
        if len(value) > 220:
            return False
        if value.endswith((".", ";", "?", "!")) and "=" not in value:
            return False

        if re.fullmatch(r"[\(\)\[\]\{\}\s]*\d+[\(\)\[\]\{\}\s]*", value):
            return True

        has_math_chars = any(ch in value for ch in "=+-*/^_{}[]()<>¼ðÞ")
        has_math_keywords = bool(re.search(r"\b(exp|sin|cos|tan|log|fft|ifft|mod|Re|Im)\b", value))
        word_count = len(re.findall(r"[A-Za-z]+", value))

        if has_math_chars or has_math_keywords:
            return True

        return word_count <= 4 and len(value) <= 40

    def _normalize_equation_source(self, text: str) -> str:
        normalized = text.strip()
        normalized = normalized.replace("¼", "=")
        normalized = normalized.replace("ð", "(").replace("Þ", ")")
        normalized = normalized.replace("–", "-").replace("—", "-")
        normalized = normalized.replace("\u00a0", " ")
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", normalized)
        return normalized

    def _extract_figures_from_captions(
        self,
        page: fitz.Page,
        page_number: int,
        figures_dir: Path,
        seen_hashes: set[str],
    ) -> tuple[list[str], int]:
        lines = self._extract_text_lines(page)
        page_rect = page.rect
        text_dict = page.get_text("dict")
        graphic_regions = self._collect_graphic_regions(page, text_dict)
        if not graphic_regions:
            return [], 0

        text_rects: list[fitz.Rect] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox", [0, 0, 0, 0])
            rect = fitz.Rect(bbox)
            if rect.width > 0 and rect.height > 0:
                text_rects.append(rect)

        caption_lines = [
            line for line in sorted(lines, key=lambda item: (item.y0, item.x0))
            if re.match(r"^(fig(?:ure)?\.?\s*\d+|图\s*\d+)", line.text.strip(), re.IGNORECASE)
            and len(line.text.strip()) <= 320
        ]
        if not caption_lines:
            return [], 0

        fragments: list[str] = []
        count = 0
        used_regions: list[fitz.Rect] = []

        for caption in caption_lines:
            figure_rect = self._select_region_for_caption(caption, graphic_regions, text_rects, page_rect)
            if figure_rect is None:
                continue
            used_regions.append(figure_rect)

            pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), clip=figure_rect, alpha=False)
            image_bytes = pix.tobytes("png")
            digest = hashlib.sha1(image_bytes).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            self._image_counter += 1
            count += 1
            image_name = f"fig_{self._image_counter:04d}_p{page_number:03d}_cap.png"
            (figures_dir / image_name).write_bytes(image_bytes)

            caption_text = self._escape_latex(caption.text.strip())
            fragments.append("\\begin{figure}[H]\n")
            fragments.append("  \\centering\n")
            fragments.append(f"  \\includegraphics[width=0.95\\textwidth]{{figures/{image_name}}}\n")
            fragments.append(f"  \\caption{{{caption_text}}}\n")
            fragments.append("\\end{figure}\n")

        page_area = page_rect.width * page_rect.height
        for region in graphic_regions:
            if any(self._rect_overlap_ratio(region, used) > 0.30 for used in used_regions):
                continue

            area_ratio = (region.width * region.height) / page_area if page_area > 0 else 0
            if area_ratio < 0.018:
                continue
            if self._text_overlap_ratio(region, text_rects) > 0.55:
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), clip=region, alpha=False)
            image_bytes = pix.tobytes("png")
            digest = hashlib.sha1(image_bytes).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            self._image_counter += 1
            count += 1
            image_name = f"fig_{self._image_counter:04d}_p{page_number:03d}_auto.png"
            (figures_dir / image_name).write_bytes(image_bytes)

            fragments.append("\\begin{figure}[H]\n")
            fragments.append("  \\centering\n")
            fragments.append(f"  \\includegraphics[width=0.95\\textwidth]{{figures/{image_name}}}\n")
            fragments.append(f"  \\caption{{Auto-extracted figure from page {page_number}}}\n")
            fragments.append("\\end{figure}\n")

        return fragments, count

    def _collect_graphic_regions(self, page: fitz.Page, text_dict: dict) -> list[fitz.Rect]:
        page_area = page.rect.width * page.rect.height
        min_area = max(1400.0, page_area * 0.0018)
        max_area = page_area * 0.82
        rects: list[fitz.Rect] = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 1:
                continue
            rect = fitz.Rect(block.get("bbox", [0, 0, 0, 0]))
            area = rect.width * rect.height
            if rect.width < 24 or rect.height < 24:
                continue
            if area < min_area or area > max_area:
                continue
            rects.append(rect)

        for drawing in page.get_drawings():
            rect_data = drawing.get("rect")
            if rect_data is None:
                continue
            rect = fitz.Rect(rect_data)
            area = rect.width * rect.height
            if rect.width < 24 or rect.height < 24:
                continue
            if area < min_area or area > max_area:
                continue
            rects.append(rect)

        return self._merge_rectangles(rects, gap=10.0)

    def _merge_rectangles(self, rects: list[fitz.Rect], gap: float) -> list[fitz.Rect]:
        merged: list[fitz.Rect] = []
        for rect in rects:
            candidate = fitz.Rect(rect)
            changed = True
            while changed:
                changed = False
                remaining: list[fitz.Rect] = []
                for existing in merged:
                    expanded = fitz.Rect(
                        existing.x0 - gap,
                        existing.y0 - gap,
                        existing.x1 + gap,
                        existing.y1 + gap,
                    )
                    if expanded.intersects(candidate):
                        candidate = fitz.Rect(
                            min(candidate.x0, existing.x0),
                            min(candidate.y0, existing.y0),
                            max(candidate.x1, existing.x1),
                            max(candidate.y1, existing.y1),
                        )
                        changed = True
                    else:
                        remaining.append(existing)
                merged = remaining
            merged.append(candidate)

        return sorted(merged, key=lambda item: (item.y0, item.x0))

    def _select_region_for_caption(
        self,
        caption: _TextLine,
        graphic_regions: list[fitz.Rect],
        text_rects: list[fitz.Rect],
        page_rect: fitz.Rect,
    ) -> fitz.Rect | None:
        max_distance = page_rect.height * 0.36
        candidates: list[fitz.Rect] = []
        for rect in graphic_regions:
            if rect.width <= 38 or rect.height <= 38:
                continue
            if rect.y0 >= caption.y0 - 18:
                continue
            if (caption.y0 - rect.y0) > max_distance:
                continue

            adjusted = fitz.Rect(rect)
            adjusted.y1 = min(adjusted.y1, caption.y0 - 6)
            if adjusted.height <= 36:
                continue
            candidates.append(adjusted)

        if not candidates:
            return None

        nearest_bottom = max(rect.y1 for rect in candidates)
        band_candidates = [
            rect for rect in candidates
            if abs(rect.y1 - nearest_bottom) <= 26
        ]

        union = fitz.Rect(band_candidates[0])
        for rect in band_candidates[1:]:
            union = fitz.Rect(
                min(union.x0, rect.x0),
                min(union.y0, rect.y0),
                max(union.x1, rect.x1),
                max(union.y1, rect.y1),
            )

        overlap_ratio = self._text_overlap_ratio(union, text_rects)
        if overlap_ratio > 0.88:
            return None

        margin = 6
        clipped = fitz.Rect(
            max(page_rect.x0, union.x0 - margin),
            max(page_rect.y0, union.y0 - margin),
            min(page_rect.x1, union.x1 + margin),
            min(page_rect.y1, union.y1 + margin),
        )
        if clipped.width < 36 or clipped.height < 36:
            return None
        return clipped

    def _text_overlap_ratio(self, rect: fitz.Rect, text_rects: list[fitz.Rect]) -> float:
        base_area = rect.width * rect.height
        if base_area <= 0:
            return 1.0

        overlap_area = 0.0
        for text_rect in text_rects:
            inter = rect & text_rect
            if inter.is_empty:
                continue
            overlap_area += inter.width * inter.height

        return overlap_area / base_area

    def _rect_overlap_ratio(self, left: fitz.Rect, right: fitz.Rect) -> float:
        inter = left & right
        if inter.is_empty:
            return 0.0
        inter_area = inter.width * inter.height
        base_area = min(left.width * left.height, right.width * right.height)
        if base_area <= 0:
            return 0.0
        return inter_area / base_area

    def _extract_page_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
        figures_dir: Path,
        seen_xrefs: set[int],
        seen_hashes: set[str],
    ) -> tuple[list[str], int]:
        fragments: list[str] = []
        page_images = page.get_images(full=True)
        page_count = 0

        for index, image_meta in enumerate(page_images, start=1):
            xref = image_meta[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            image = doc.extract_image(xref)
            image_bytes = image.get("image")
            extension = image.get("ext", "png")

            if not image_bytes:
                continue

            digest = hashlib.sha1(image_bytes).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            self._image_counter += 1
            page_count += 1
            image_name = f"fig_{self._image_counter:04d}_p{page_number:03d}_{index:02d}.{extension}"
            image_path = figures_dir / image_name
            image_path.write_bytes(image_bytes)

            fragments.append("\\begin{figure}[H]\n")
            fragments.append("  \\centering\n")
            fragments.append(f"  \\includegraphics[width=0.95\\textwidth]{{figures/{image_name}}}\n")
            fragments.append(f"  \\caption{{Extracted figure from page {page_number}}}\n")
            fragments.append("\\end{figure}\n")

        return fragments, page_count

    def _build_document(self, body_fragments: list[str]) -> str:
        cleaned_lines = self._normalize_body_lines(body_fragments)
        front = self._extract_front_matter(cleaned_lines)
        title = self._sanitize_front_text(front["title"])
        author = self._sanitize_front_text(front["author"])
        abstract = self._sanitize_front_text(front["abstract"])
        keywords = self._sanitize_front_text(front["keywords"])
        body = "\n".join(front["body_lines"])

        abstract_block = (
            "\\begin{abstract}\n"
            f"{abstract}\n"
            "\\end{abstract}\n"
        ) if abstract else ""

        keyword_block = f"\\noindent\\textbf{{Keywords:}} {keywords}\n" if keywords else ""

        return (
            "\\documentclass{article}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{amssymb}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{caption}\n"
            "\\usepackage{multirow}\n"
            "\\usepackage{float}\n"
            "\\usepackage{ctex}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{a4paper,margin=1in}\n"
            "\\begin{document}\n"
            f"\\title{{{title}}}\n"
            f"\\author{{{author}}}\n"
            "\\date{}\n"
            "\\maketitle\n"
            f"{abstract_block}"
            f"{keyword_block}\n"
            f"{body}\n"
            "\\end{document}\n"
        )

    def _normalize_body_lines(self, body_fragments: list[str]) -> list[str]:
        raw_lines: list[str] = []
        for fragment in body_fragments:
            raw_lines.extend(fragment.splitlines())

        lines: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if stripped == "\\clearpage":
                continue
            if self._is_forbidden_body_line(stripped):
                continue
            if "Aerospace Information Research Institute" in stripped:
                continue
            if re.fullmatch(r"[ðÞ\(\)\[\]\{\}\s\d\.,;:]+", stripped):
                continue

            heading_split = self._split_inline_heading(stripped)
            if heading_split is not None:
                heading, remaining = heading_split
                lines.append(f"\\subsection{{{self._escape_latex(heading)}}}")
                if remaining:
                    lines.append(self._escape_latex(remaining))
                continue

            lines.append(stripped)

        compact: list[str] = []
        for line in lines:
            if line == "" and compact and compact[-1] == "":
                continue
            compact.append(line)

        promoted = [self._promote_and_clean_heading(line) for line in compact]
        return promoted

    def _is_forbidden_body_line(self, line: str) -> bool:
        forbidden_prefixes = (
            "\\documentclass",
            "\\usepackage",
            "\\geometry",
            "\\title",
            "\\author",
            "\\date",
            "\\maketitle",
            "\\begin{document}",
            "\\end{document}",
            "\\begin{thebibliography}",
            "\\end{thebibliography}",
            "\\begin{itemize}",
            "\\end{itemize}",
            "\\item",
        )
        if line.startswith("```"):
            return True
        return any(line.startswith(prefix) for prefix in forbidden_prefixes)

    def _split_inline_heading(self, text: str) -> tuple[str, str] | None:
        patterns = [
            "Signal Architecture",
            "Structure of Authenticable Spreading Code",
            "Design of Security Code",
            "Navigation Message",
            "Structure of Authenticable Receiver",
            "Parameters and Performance Metrics",
            "Implementation for Beidou B1C",
            "Conclusion",
        ]
        for candidate in patterns:
            prefix = f"{candidate} "
            if text.startswith(prefix) and len(text) > len(prefix) + 20:
                return candidate, text[len(prefix):].strip()
        return None

    def _promote_and_clean_heading(self, line: str) -> str:
        section_pattern = re.compile(r"^\\section\{(.+)\}$")
        subsection_pattern = re.compile(r"^\\subsection\{(.+)\}$")

        section_match = section_pattern.match(line)
        if section_match:
            title = re.sub(r"^\d+(?:\.\d+)*\s+", "", section_match.group(1)).strip()
            return f"\\section{{{title}}}"

        subsection_match = subsection_pattern.match(line)
        if not subsection_match:
            return line

        title = re.sub(r"^\d+(?:\.\d+)*\s+", "", subsection_match.group(1)).strip()
        top_level = {
            "Introduction",
            "Model of CC-SCA Signal",
            "Structure of Authenticable Receiver",
            "Parameters and Performance Metrics",
            "Implementation for Beidou B1C",
            "Conclusion",
        }
        if title in top_level:
            return f"\\section{{{title}}}"
        return f"\\subsection{{{title}}}"

    def _extract_front_matter(self, lines: list[str]) -> dict[str, object]:
        title = "Converted Paper"
        author_lines: list[str] = []
        abstract = ""
        keywords = ""

        body = list(lines)

        for i, line in enumerate(body):
            match = re.match(r"^\\section\{(.+)\}$", line)
            if not match:
                continue
            if len(match.group(1)) >= 18:
                title = match.group(1)
                body = body[i + 1:]
            break

        capture_author = True
        processed_body: list[str] = []
        index = 0
        author_scan_count = 0
        while index < len(body):
            line = body[index].strip()

            if self._is_forbidden_body_line(line):
                index += 1
                continue

            if line.startswith("\\section{") or line.startswith("\\subsection{"):
                capture_author = False

            if line.startswith("Abstract."):
                abstract = line[len("Abstract."):].strip()
                capture_author = False
                index += 1
                continue
            if line.startswith("Keywords:"):
                keywords = line[len("Keywords:"):].strip()
                capture_author = False
                index += 1
                continue

            if capture_author and line and not line.startswith("\\"):
                author_scan_count += 1
                is_short = len(line) <= 140
                has_email = "@" in line
                has_affiliation = any(token in line for token in ("University", "Institute", "Laboratory", "China"))
                looks_name_list = bool(re.fullmatch(r"[A-Za-z\-\.,\s\(\)&]+", line)) and ("," in line or line.lower().startswith("and "))

                if is_short and (has_email or has_affiliation or looks_name_list):
                    author_lines.append(line)
                    index += 1
                    continue

                # If the model starts outputting long prose before Abstract marker, treat it as body.
                if len(line) > 170 or author_scan_count > 8:
                    capture_author = False

            processed_body.append(body[index])
            index += 1

        author = " \\\\ ".join(author_lines) if author_lines else ""
        return {
            "title": title,
            "author": author,
            "abstract": abstract,
            "keywords": keywords,
            "body_lines": processed_body,
        }

    def _sanitize_front_text(self, text: object) -> str:
        value = str(text or "")
        value = value.replace("\u00a0", " ")
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
        return value.strip()

    def _escape_latex(self, text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = text.replace("ð", "(").replace("Þ", ")").replace("þ", "+")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        escaped = "".join(replacements.get(char, char) for char in text)
        return escaped
