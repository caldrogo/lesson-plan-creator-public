"""Layout-aware extraction, auditable continuation links and lossless bank loading."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .models import Curriculum, Fragment, Objective, Question

REF = re.compile(r"\b[CE]\d+\.\d+\b")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def objectives(text: str) -> list[Objective]:
    matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", text))
    if not matches:
        return [Objective(text=text)] if text.strip() else []
    result = []
    if text[: matches[0].start()].strip():
        result.append(Objective(text=text[: matches[0].start()].strip()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result.append(Objective(number=match[1], text=text[match.end() : end].strip()))
    return result


def join_rows(rows: list[Fragment], metadata: dict) -> tuple[list[Curriculum], list[dict]]:
    """Forward-only carry: an orphan never inherits a later visible reference."""
    records, unresolved = [], []
    active = None
    previous_page = None
    for row in rows:
        # A separator deliberately breaks continuity across non-table material.
        if row.reference_text == "__BREAK__":
            active = None
            previous_page = None
            continue
        ref_text = row.reference_text
        positive_text = re.split(r"\(?note\b", ref_text, flags=re.IGNORECASE)[0]
        refs = list(dict.fromkeys(REF.findall(positive_text)))
        if refs:
            scope = (
                "Core/Extended"
                if {r[0] for r in refs} == {"C", "E"}
                else "Extended"
                if refs[0].startswith("E")
                else "Core"
            )
            record_id = hashlib.sha256(
                f"{metadata['file_hash']}:{row.page}:{row.bbox}:{refs}".encode()
            ).hexdigest()[:32]
            active = Curriculum(
                **metadata,
                record_id=record_id,
                references=refs,
                scope=scope,
                objectives=[],
                objective_text="",
                activity_text="",
                source_notes=ref_text,
                fragments=[],
                warnings=[
                    (
                        "Mathematical notation: consult attached source crop; "
                        "text alone may flatten fractions and superscripts."
                    )
                ],
            )
            records.append(active)
        elif ref_text.strip():
            unresolved.append(
                {"reason": "Unrecognised reference cell", "fragment": row.model_dump()}
            )
            active = None
            continue
        elif active is None or (previous_page is not None and row.page > previous_page + 1):
            unresolved.append(
                {
                    "reason": "Orphan continuation; no preceding adjacent table row",
                    "fragment": row.model_dump(),
                }
            )
            active = None
            continue
        else:
            active.warnings.append(
                f"Continuation on physical page {row.page}; "
                "associated with preceding reference, not next reference."
            )
        active.fragments.append(row)
        if row.objective_text:
            active.objective_text = "\n".join(
                filter(None, [active.objective_text, row.objective_text])
            )
        if row.activity_text:
            active.activity_text = "\n".join(
                filter(None, [active.activity_text, row.activity_text])
            )
        active.objectives = objectives(active.objective_text)
        previous_page = row.page
    accepted = []
    for record in records:
        if not record.objectives or not record.activity_text.strip():
            unresolved.append(
                {"reason": "Missing objectives or activities", "record": record.model_dump()}
            )
        else:
            accepted.append(record)
    return accepted, unresolved


def raster_tables(page):
    """Recover ruled tables from OCR pages whose grid exists only in the raster image."""
    import numpy as np

    image = page.to_image(resolution=144).original.convert("RGB")
    pixels = np.asarray(image)
    ink = pixels.min(axis=2) < 180

    def boundaries(mask, scale):
        indices = np.flatnonzero(mask)
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        result = []
        for group in groups:
            if not len(group):
                continue
            if len(group) > 6:  # Solid header bands: preserve both edges.
                result.extend([float(group[0]) / scale, float(group[-1]) / scale])
            else:
                result.append(float(group.mean()) / scale)
        return result

    xs = boundaries(ink.mean(axis=0) > 0.30, image.width / page.width)
    ys = boundaries(ink.mean(axis=1) > 0.60, image.height / page.height)
    if len(xs) < 4 or len(ys) < 3:
        return []
    return page.find_tables(
        {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "explicit",
            "explicit_vertical_lines": xs,
            "explicit_horizontal_lines": ys,
        }
    )


def extract_pdf(source: Path, output: Path, *, ocr: bool = False) -> dict:
    import pdfplumber

    output.mkdir(parents=True, exist_ok=True)
    original_hash = digest(source)
    extraction_path = source
    if ocr:
        # Never modifies the original. OCR pages still need table association review.
        extraction_path = output / f"{original_hash[:12]}-ocr.pdf"
        subprocess.run(
            ["ocrmypdf", "--skip-text", "--output-type", "pdf", str(source), str(extraction_path)],
            check=True,
            timeout=600,
        )
    rows, rejected, pages_report = [], [], []
    asset_dir = output / "assets"
    asset_dir.mkdir(exist_ok=True)
    with pdfplumber.open(extraction_path) as pdf:
        cover = pdf.pages[0].extract_text() or ""
        copyright_text = pdf.pages[1].extract_text() or "" if len(pdf.pages) > 1 else ""
        years = re.search(r"For examination[^\n]*", cover, re.IGNORECASE)
        edition = re.search(r"(?:edition|version)\s*[^\n]+", cover, re.IGNORECASE)
        metadata = {
            "document_id": "0607-sow-" + original_hash[:16],
            "file_hash": original_hash,
            "title": "\n".join(
                line for line in cover.splitlines() if not line.startswith("For examination")
            ).strip(),
            "edition": edition[0] if edition else None,
            "syllabus_years": years[0] if years else None,
            "copyright": next((s for s in copyright_text.splitlines() if "©" in s), None),
            "source_filename": str(source),
        }
        for page in pdf.pages:
            page_no = page.page_number
            text = page.extract_text() or ""
            if "How to get the most out of this scheme of work" in text:
                pages_report.append(
                    {"page": page_no, "status": "annotated introduction example excluded"}
                )
                rows.append(
                    Fragment(
                        page=page_no,
                        bbox=[],
                        reference_text="__BREAK__",
                        objective_text="",
                        activity_text="",
                    )
                )
                continue
            detected_tables = page.find_tables()
            if ocr and not detected_tables:
                detected_tables = raster_tables(page)
            tables = sorted(
                detected_tables, key=lambda t: -(t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
            )
            table = next(
                (
                    t
                    for t in tables
                    if "Syllabus ref." in str(t.extract())
                    and "Suggested teaching activities" in str(t.extract())
                ),
                None,
            )
            if table is None:
                pages_report.append({"page": page_no, "status": "no curriculum table"})
                rows.append(
                    Fragment(
                        page=page_no,
                        bbox=[],
                        reference_text="__BREAK__",
                        objective_text="",
                        activity_text="",
                    )
                )
                if not text.strip() or "Syllabus ref." in text:
                    rejected.append(
                        {
                            "page": page_no,
                            "reason": "No readable three-column table; "
                            "run --ocr for scans, then inspect or supply reviewed rows.",
                        }
                    )
                continue
            # Word-export PDFs contain decorative narrow cells in the header. Infer the
            # three actual column boxes from a tall body row, excluding those cells.
            body = next(
                (
                    r
                    for r in table.rows
                    if r.bbox[3] - r.bbox[1] > 25
                    and len([c for c in r.cells if c and c[2] - c[0] > 60]) == 3
                ),
                None,
            )
            if body is None:
                rejected.append({"page": page_no, "reason": "Ambiguous table columns"})
                rows.append(
                    Fragment(
                        page=page_no,
                        bbox=[],
                        reference_text="__BREAK__",
                        objective_text="",
                        activity_text="",
                    )
                )
                continue
            boxes = sorted([c for c in body.cells if c and c[2] - c[0] > 60])
            cuts = [boxes[0][0], boxes[1][0], boxes[2][0], boxes[-1][2]]
            footer = page.crop((0, page.height - 35, page.width, page.height)).extract_text() or ""
            label = next((s for s in footer.splitlines() if s.strip().isdigit()), None)
            count = 0
            for index, row in enumerate(table.rows):
                top, bottom = row.bbox[1], row.bbox[3]
                if bottom - top < 3:
                    continue
                cells = [
                    page.crop((cuts[i], top, cuts[i + 1], bottom)).extract_text(
                        x_tolerance=1, y_tolerance=2
                    )
                    or ""
                    for i in range(3)
                ]
                if "Syllabus ref." in cells[0] or not any(c.strip() for c in cells):
                    continue
                if any(
                    "Past and specimen papers" in c or "Past/specimen papers" in c for c in cells
                ):
                    rows.append(
                        Fragment(
                            page=page_no,
                            bbox=[],
                            reference_text="__BREAK__",
                            objective_text="",
                            activity_text="",
                        )
                    )
                    continue
                bbox = [cuts[0], top, cuts[-1], bottom]
                stem = f"{original_hash[:12]}-p{page_no}-r{index}"
                crop = page.crop(tuple(bbox))
                crop.to_image(resolution=130).save(asset_dir / f"{stem}.png")
                glyphs = [
                    {k: c.get(k) for k in ("text", "x0", "top", "x1", "bottom", "size")}
                    for c in crop.chars
                ]
                (asset_dir / f"{stem}.json").write_text(json.dumps(glyphs, ensure_ascii=False))
                rows.append(
                    Fragment(
                        page=page_no,
                        printed_label=label,
                        bbox=bbox,
                        reference_text=cells[0],
                        objective_text=cells[1],
                        activity_text=cells[2],
                        image=f"assets/{stem}.png",
                        glyph_file=f"assets/{stem}.json",
                    )
                )
                count += 1
            pages_report.append({"page": page_no, "status": "extracted", "rows": count})
    records, unresolved = join_rows(rows, metadata)
    if ocr:
        for record in records:
            record.warnings.append(
                "OCR text and inferred grid require human review against source crops."
            )
    report = {
        "document": metadata,
        "extraction": "pdfplumber ruled-table v1",
        "ocr_used": ocr,
        "records": len(records),
        "pages": pages_report,
        "rejected_records": rejected,
        "unresolved_associations": unresolved,
        "sample_records": [r.model_dump() for r in records[:3]],
        "warnings": [
            "Edition is unknown unless explicitly printed in the source.",
            "Crop PNGs and positioned glyphs preserve original mathematical notation.",
            "No objective-specific activity mapping is inferred inside a shared cell.",
        ],
    }
    write_jsonl(output / "curriculum.jsonl", records)
    (output / "extraction-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def write_jsonl(path: Path, records):
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def load_questions(source: Path) -> tuple[list[Question], dict]:
    paths = sorted(source.rglob("*.json")) if source.is_dir() else [source]
    result, errors, duplicates, seen = [], [], [], {}
    for path in paths:
        try:
            if path.suffix == ".jsonl":
                values = [
                    json.loads(line) for line in path.read_text().splitlines() if line.strip()
                ]
            else:
                value = json.loads(path.read_text())
                values = value.get("questions", [value]) if isinstance(value, dict) else value
            if not isinstance(values, list):
                raise TypeError("Expected a list of question objects")
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"file": str(path), "reason": type(exc).__name__})
            continue
        for index, value in enumerate(values):
            try:
                if not isinstance(value, dict):
                    raise TypeError("Question is not an object")
                if "question_id" in value:
                    # A normalized input is already authoritative, including explicit nulls.
                    question = Question.model_validate(value)
                else:
                    number = value.get("question_number")
                    if number is None or not isinstance(value.get("question_text"), str):
                        raise ValueError("Missing question_number or question_text")
                    marks = value.get("marks", value.get("marks_available"))
                    match = re.fullmatch(r"\[?(\d+)\]?", str(marks).strip())
                    question = Question(
                        question_id=f"{path.stem}:{number}",
                        question_text=value["question_text"],
                        topic=value.get("topic"),
                        subtopic=value.get("subtopic"),
                        marks=int(match[1]) if match else None,
                        exam_board=value.get("exam_board"),
                        year=value.get("year"),
                        paper=value.get("paper"),
                        question_type=value.get("question_type"),
                        mark_scheme=value.get("mark_scheme"),
                        source_filename=value.get("source_filename", str(path)),
                        raw_metadata=value,
                    )
                if not question.question_text.strip() or not question.question_id.strip():
                    raise ValueError("Empty ID or wording")
                if question.question_id in seen:
                    if seen[question.question_id] != question:
                        raise ValueError("Conflicting duplicate question ID")
                    duplicates.append(question.question_id)
                    continue
                seen[question.question_id] = question
                result.append(question)
            except (ValueError, TypeError) as exc:
                errors.append({"file": str(path), "index": index, "reason": str(exc)})
    return result, {"records": len(result), "rejected_records": errors, "duplicates": duplicates}
