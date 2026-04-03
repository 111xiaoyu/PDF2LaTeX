from __future__ import annotations

import os
import re
from typing import Callable, Sequence

from openai import OpenAI


class DeepSeekLatexRefiner:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-reasoner",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 180.0,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model
        env_tokens = os.getenv("DEEPSEEK_MAX_TOKENS", "8192").strip()
        self._max_tokens = self._resolve_max_tokens(max_tokens=max_tokens, env_tokens=env_tokens)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _resolve_max_tokens(self, max_tokens: int | None, env_tokens: str) -> int | None:
        raw: int | None = max_tokens
        if raw is None and env_tokens:
            try:
                raw = int(env_tokens)
            except ValueError:
                raw = 8192

        if raw is None:
            return None
        return max(1, min(8192, int(raw)))

    def refine_page(self, page_number: int, raw_lines: Sequence[str]) -> list[str]:
        payload_text = "\n".join(raw_lines).strip()
        if not payload_text:
            return []

        system_prompt = (
            "You are an expert LaTeX editor. "
            "Given noisy text extracted from one PDF page, output only valid LaTeX body content "
            "without preamble or document environment. Keep original wording and order as much as possible. "
            "Use plain paragraphs for body text. Use section/subsection only when clearly a heading. "
            "Do not wrap every sentence as headings. Remove page headers, footers, page numbers, and running titles. "
            "Repair line breaks and hyphenated words split by line endings. "
            "For formulas, use \\begin{equation} ... \\end{equation}. "
            "Do not use \\tag{...}, \\eqno, \\begin{cases}, or \\end{cases}. "
            "Do not add unsupported claims or hallucinated content."
        )
        user_prompt = (
            f"Page number: {page_number}\n"
            "Return only LaTeX fragment. No markdown fences.\n"
            "Input lines:\n"
            f"{payload_text}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            stream=False,
        )

        content = response.choices[0].message.content or ""
        normalized = self._strip_markdown_fence(content)
        normalized = self._sanitize_latex(normalized)

        lines = [line.rstrip() for line in normalized.splitlines()]
        return [line for line in lines if line.strip()]

    def refine_document(
        self,
        raw_text: str,
        figure_specs: Sequence[dict[str, str]],
        title_hint: str,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        figure_prompt = "\n".join(
            f"- Figure {item['number']}: path={item['path']} caption={item['caption']}"
            for item in figure_specs
        )
        if not figure_prompt:
            figure_prompt = "- No figures found."

        system_prompt = (
            "You are a strict LaTeX conversion engine. "
            "Output a full compilable LaTeX paper and preserve original content as closely as possible. "
            "Never output markdown fences. Never output [cite] or any citation placeholders. "
            "In \\includegraphics, always use width=0.5\\linewidth. "
            "For figure paths, only use the provided figures/figureN paths exactly. "
            "Every figure environment must include a label in the form \\label{fig:N} and all figure references must use \\ref{fig:N}. "
            "Every table environment must include a label in the form \\label{tab:N} and all table references must use \\ref{tab:N}. "
            "For every tabular environment, ensure column definition count exactly matches each row's column count. "
            "If you use \\toprule/\\midrule/\\bottomrule, include \\usepackage{booktabs}. "
            "Do not add extra sections not present in source text. "
            "Formulas must be compilable LaTeX and should use equation/align environments when needed. "
            "Do not stop early. Output the complete paper content from beginning to end."
        )
        user_prompt = (
            f"Title hint: {title_hint}\n"
            "Please fully recognize this paper and output LaTeX code for direct compilation.\n"
            "Do not delete content.\n"
            "Figure assets available:\n"
            f"{figure_prompt}\n\n"
            "Raw extracted document text:\n"
            f"{raw_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        content = ""
        if stream_callback is not None:
            try:
                content = self._stream_document_response(messages, stream_callback)
            except Exception:
                content = ""

        if not content:
            request_kwargs: dict = {
                "model": self._model,
                "messages": messages,
                "temperature": 0.0,
                "stream": False,
            }
            if self._max_tokens is not None:
                request_kwargs["max_tokens"] = self._max_tokens

            response = self._client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            if stream_callback is not None and content:
                stream_callback(content)

        normalized = self._strip_markdown_fence(content)
        sanitized = self._sanitize_full_document_output(normalized)
        return sanitized

    def _stream_document_response(
        self,
        messages: list[dict[str, str]],
        stream_callback: Callable[[str], None],
    ) -> str:
        request_kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "stream": True,
        }
        if self._max_tokens is not None:
            request_kwargs["max_tokens"] = self._max_tokens

        stream = self._client.chat.completions.create(**request_kwargs)

        chunks: list[str] = []
        for event in stream:
            choices = getattr(event, "choices", None)
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            piece = self._normalize_stream_piece(getattr(delta, "content", ""))
            if not piece:
                continue

            chunks.append(piece)
            stream_callback(piece)

        return "".join(chunks)

    def _normalize_stream_piece(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            merged: list[str] = []
            for item in value:
                if isinstance(item, str):
                    merged.append(item)
                    continue
                text_value = ""
                if isinstance(item, dict):
                    text_value = str(item.get("text") or item.get("content") or "")
                else:
                    text_value = str(getattr(item, "text", "") or getattr(item, "content", "") or "")
                if text_value:
                    merged.append(text_value)
            return "".join(merged)
        return ""

    def _strip_markdown_fence(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        parts = stripped.splitlines()
        if len(parts) < 3:
            return stripped.replace("```", "").strip()

        if parts[0].startswith("```") and parts[-1].strip() == "```":
            return "\n".join(parts[1:-1]).strip()
        return stripped

    def _sanitize_latex(self, text: str) -> str:
        sanitized = text

        doc_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", sanitized, re.S)
        if doc_match:
            sanitized = doc_match.group(1)

        # Remove document-level commands if the model outputs a full template.
        forbidden_line_starts = (
            r"\documentclass",
            r"\usepackage",
            r"\geometry",
            r"\title",
            r"\author",
            r"\date",
            r"\maketitle",
            r"\begin{document}",
            r"\end{document}",
        )
        kept_lines: list[str] = []
        for line in sanitized.splitlines():
            stripped = line.strip()
            if not stripped:
                kept_lines.append("")
                continue
            if any(stripped.startswith(prefix) for prefix in forbidden_line_starts):
                continue
            if stripped.startswith("```"):
                continue
            kept_lines.append(line)
        sanitized = "\n".join(kept_lines)

        # Remove bibliography/itemize blocks that should not appear in per-page output.
        sanitized = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", sanitized, flags=re.S)
        sanitized = re.sub(r"\\begin\{itemize\}.*?\\end\{itemize\}", "", sanitized, flags=re.S)

        sanitized = re.sub(r"\\begin\{equation\*?\}", r"\\begin{equation}", sanitized)
        sanitized = re.sub(r"\\end\{equation\*?\}", r"\\end{equation}", sanitized)
        sanitized = re.sub(r"\\begin\{cases\}", "", sanitized)
        sanitized = re.sub(r"\\end\{cases\}", "", sanitized)
        sanitized = re.sub(r"\\eqno\b", "", sanitized)
        sanitized = re.sub(r"\\tag\{[^{}]*\}", "", sanitized)
        sanitized = re.sub(r"\\label\{[^{}]*\}", "", sanitized)
        sanitized = re.sub(r"\\varphi", r"\\phi", sanitized)
        if sanitized.count("$$") >= 2:
            parts = sanitized.split("$$")
            rebuilt: list[str] = []
            for index, chunk in enumerate(parts):
                rebuilt.append(chunk)
                if index == len(parts) - 1:
                    continue
                rebuilt.append("\\begin{equation}" if index % 2 == 0 else "\\end{equation}")
            sanitized = "".join(rebuilt)
        return sanitized

    def _sanitize_full_document_output(self, text: str) -> str:
        value = text.strip()
        value = re.sub(r"\[cite[^\]]*\]", "", value, flags=re.IGNORECASE)

        doc_match = re.search(r"(\\documentclass[\s\S]*?\\end\{document\})", value)
        if doc_match:
            value = doc_match.group(1)
        else:
            # Fallback: wrap as body if model forgot preamble.
            value = (
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
                f"{value}\n"
                "\\end{document}\n"
            )

        # Normalize equation variants.
        value = re.sub(r"\\begin\{equation\*?\}", r"\\begin{equation}", value)
        value = re.sub(r"\\end\{equation\*?\}", r"\\end{equation}", value)
        value = re.sub(r"\\tag\{[^{}]*\}", "", value)
        value = re.sub(r"\\eqno\b", "", value)

        # Force figure width and clean figure path style.
        def _fix_includegraphics(match: re.Match[str]) -> str:
            path = match.group(1).strip().replace("\\", "/")
            number_match = re.search(r"figure\s*([0-9]+)", path, re.IGNORECASE)
            if number_match:
                figure_path = f"figures/figure{int(number_match.group(1))}"
            else:
                figure_path = path
                figure_path = re.sub(r"\.(png|jpg|jpeg|pdf)$", "", figure_path, flags=re.IGNORECASE)
            return f"\\includegraphics[width=0.5\\linewidth]{{{figure_path}}}"

        value = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", _fix_includegraphics, value)
        value = self._normalize_figure_labels_and_refs(value)
        value = self._normalize_table_labels_and_refs(value)
        value = self._fix_tabular_column_mismatch(value)

        # Keep a single document block.
        value = re.sub(r"\\begin\{document\}.*?\\begin\{document\}", r"\\begin{document}\n", value, flags=re.S)
        value = re.sub(r"\\end\{document\}.*", r"\\end{document}\n", value, flags=re.S)
        value = self._ensure_required_packages(value)
        return value

    def _normalize_figure_labels_and_refs(self, text: str) -> str:
        figure_env_pattern = re.compile(r"\\begin\{figure\}[\s\S]*?\\end\{figure\}")

        seq_counter = 1
        seen_numbers: list[int] = []

        def _rewrite_figure_block(match: re.Match[str]) -> str:
            nonlocal seq_counter
            block = match.group(0)

            number = seq_counter
            path_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)
            if path_match:
                path = path_match.group(1)
                num_match = re.search(r"figure\s*([0-9]+)", path, re.IGNORECASE)
                if num_match:
                    number = int(num_match.group(1))
            seq_counter += 1
            seen_numbers.append(number)

            label = f"fig:{number}"
            cleaned_block = re.sub(r"\\label\{[^{}]*\}", "", block)
            if "\\end{figure}" in cleaned_block:
                cleaned_block = cleaned_block.replace("\\end{figure}", f"\\label{{{label}}}\n\\end{{figure}}", 1)
            return cleaned_block

        value = figure_env_pattern.sub(_rewrite_figure_block, text)

        def _normalize_ref(match: re.Match[str]) -> str:
            ref_no = int(match.group(1))
            return f"\\ref{{fig:{ref_no}}}"

        value = re.sub(r"\\ref\{fig\s*:\s*0*([0-9]+)\}", _normalize_ref, value, flags=re.IGNORECASE)
        value = re.sub(r"([Ff]igure\s*~?\s*)\\ref\{0*([0-9]+)\}", r"\1\\ref{fig:\2}", value)

        if seen_numbers:
            max_no = max(seen_numbers)

            def _bound_ref(match: re.Match[str]) -> str:
                ref_no = int(match.group(1))
                if ref_no < 1:
                    ref_no = 1
                if ref_no > max_no:
                    ref_no = max_no
                return f"\\ref{{fig:{ref_no}}}"

            value = re.sub(r"\\ref\{fig:0*([0-9]+)\}", _bound_ref, value)

        return value

    def _normalize_table_labels_and_refs(self, text: str) -> str:
        table_env_pattern = re.compile(r"\\begin\{table\}[\s\S]*?\\end\{table\}")

        seq_counter = 1
        seen_numbers: list[int] = []

        def _rewrite_table_block(match: re.Match[str]) -> str:
            nonlocal seq_counter
            block = match.group(0)

            number = seq_counter
            caption_no = re.search(
                r"\\caption\{[^{}]*?(?:table|tab\.?|表)\s*([0-9]+)",
                block,
                re.IGNORECASE,
            )
            if caption_no:
                number = int(caption_no.group(1))
            seq_counter += 1
            seen_numbers.append(number)

            label = f"tab:{number}"
            cleaned_block = re.sub(r"\\label\{[^{}]*\}", "", block)
            if "\\end{table}" in cleaned_block:
                cleaned_block = cleaned_block.replace("\\end{table}", f"\\label{{{label}}}\n\\end{{table}}", 1)
            return cleaned_block

        value = table_env_pattern.sub(_rewrite_table_block, text)

        def _normalize_ref(match: re.Match[str]) -> str:
            ref_no = int(match.group(1))
            return f"\\ref{{tab:{ref_no}}}"

        value = re.sub(r"\\ref\{tab\s*:\s*0*([0-9]+)\}", _normalize_ref, value, flags=re.IGNORECASE)
        value = re.sub(r"([Tt]able\s*~?\s*)\\ref\{0*([0-9]+)\}", r"\1\\ref{tab:\2}", value)

        if seen_numbers:
            max_no = max(seen_numbers)

            def _bound_ref(match: re.Match[str]) -> str:
                ref_no = int(match.group(1))
                if ref_no < 1:
                    ref_no = 1
                if ref_no > max_no:
                    ref_no = max_no
                return f"\\ref{{tab:{ref_no}}}"

            value = re.sub(r"\\ref\{tab:0*([0-9]+)\}", _bound_ref, value)

        return value

    def _fix_tabular_column_mismatch(self, text: str) -> str:
        begin_token = "\\begin{tabular}"
        end_token = "\\end{tabular}"

        idx = 0
        out: list[str] = []
        while True:
            begin_at = text.find(begin_token, idx)
            if begin_at < 0:
                out.append(text[idx:])
                break

            out.append(text[idx:begin_at])
            brace_start = begin_at + len(begin_token)
            if brace_start >= len(text) or text[brace_start] != "{":
                out.append(begin_token)
                idx = brace_start
                continue

            spec, spec_end = self._read_braced_content(text, brace_start)
            body_start = spec_end + 1
            end_at = text.find(end_token, body_start)
            if end_at < 0:
                out.append(text[begin_at:])
                break

            body = text[body_start:end_at]
            defined_cols = self._count_tabular_columns(spec)
            max_cells = self._max_tabular_cells(body)
            fixed_spec = spec
            if max_cells > defined_cols:
                fixed_spec = f"{spec}{'l' * (max_cells - defined_cols)}"

            out.append(f"{begin_token}{{{fixed_spec}}}{body}{end_token}")
            idx = end_at + len(end_token)

        return "".join(out)

    def _read_braced_content(self, source: str, open_brace_index: int) -> tuple[str, int]:
        if open_brace_index >= len(source) or source[open_brace_index] != "{":
            return "", open_brace_index

        depth = 0
        collected: list[str] = []
        cursor = open_brace_index
        while cursor < len(source):
            ch = source[cursor]
            if ch == "{":
                depth += 1
                if depth > 1:
                    collected.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(collected), cursor
                collected.append(ch)
            else:
                if depth >= 1:
                    collected.append(ch)
            cursor += 1
        return "".join(collected), len(source) - 1

    def _count_tabular_columns(self, spec: str) -> int:
        compact = re.sub(r"\s+", "", spec)
        count = 0
        i = 0
        while i < len(compact):
            ch = compact[i]
            if ch in "|!":
                i += 1
                continue

            if ch in "@<>":
                if i + 1 < len(compact) and compact[i + 1] == "{":
                    _, end = self._read_braced_content(compact, i + 1)
                    i = end + 1
                else:
                    i += 1
                continue

            if ch == "*" and i + 1 < len(compact) and compact[i + 1] == "{":
                rep_text, rep_end = self._read_braced_content(compact, i + 1)
                repeat = 1
                try:
                    repeat = max(1, int(rep_text.strip() or "1"))
                except ValueError:
                    repeat = 1
                block_start = rep_end + 1
                if block_start < len(compact) and compact[block_start] == "{":
                    inner, inner_end = self._read_braced_content(compact, block_start)
                    count += repeat * self._count_tabular_columns(inner)
                    i = inner_end + 1
                    continue

            if ch in "lcrXS":
                count += 1
                i += 1
                continue

            if ch in "pmb":
                count += 1
                if i + 1 < len(compact) and compact[i + 1] == "{":
                    _, end = self._read_braced_content(compact, i + 1)
                    i = end + 1
                else:
                    i += 1
                continue

            if ch == "D":
                count += 1
                i += 1
                for _ in range(3):
                    if i < len(compact) and compact[i] == "{":
                        _, end = self._read_braced_content(compact, i)
                        i = end + 1
                continue

            i += 1

        return max(1, count)

    def _max_tabular_cells(self, body: str) -> int:
        max_cells = 0
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if "\\\\" not in line:
                continue
            if any(token in line for token in ("\\\\toprule", "\\\\midrule", "\\\\bottomrule", "\\\\hline", "\\\\cline", "\\\\cmidrule", "\\\\multicolumn", "\\\\multirow")):
                continue

            row_head = line.split("\\\\", 1)[0]
            amp_count = len(re.findall(r"(?<!\\\\)&", row_head.replace(r"\\&", "")))
            max_cells = max(max_cells, amp_count + 1)

        return max_cells

    def _ensure_required_packages(self, text: str) -> str:
        needs_booktabs = bool(re.search(r"\\\\toprule|\\\\midrule|\\\\bottomrule", text))
        has_booktabs = "\\usepackage{booktabs}" in text

        value = text
        if needs_booktabs and not has_booktabs:
            if "\\begin{document}" in value:
                value = value.replace("\\begin{document}", "\\usepackage{booktabs}\n\\begin{document}", 1)
            else:
                value = "\\usepackage{booktabs}\n" + value
        return value
