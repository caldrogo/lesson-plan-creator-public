"""Regression on the supplied real source; extraction outputs are generated in a temp directory."""

from pathlib import Path

import pytest

from lesson_plan_ai.v2.ingestion import extract_pdf
from lesson_plan_ai.v2.models import Curriculum

PDF = Path("data/raw/0607_Scheme_of_Work_(for_examination_from_2025).pdf")


@pytest.mark.skipif(not PDF.exists(), reason="Real curriculum PDF unavailable")
def test_real_pdf_associations_and_notation(tmp_path):
    report = extract_pdf(PDF, tmp_path)
    assert report["records"] == 76
    assert not report["rejected_records"]
    assert not report["unresolved_associations"]
    assert report["document"]["syllabus_years"] == "For examination from 2025"
    assert report["document"]["edition"] is None
    records = [
        Curriculum.model_validate_json(x)
        for x in (tmp_path / "curriculum.jsonl").read_text().splitlines()
    ]
    assert all(f.page != 6 for c in records for f in c.fragments)
    surds = next(c for c in records if c.references == ["E1.17"])
    assert [f.page for f in surds.fragments] == [33, 34]
    assert [o.number for o in surds.objectives] == ["1", "2"]
    proportion = next(c for c in records if c.references == ["E2.8"] and c.fragments[0].page == 33)
    assert proportion.scope == "Extended"
    assert "Once the basics are secure" not in proportion.activity_text
    formulae = next(c for c in records if "Once the basics are secure" in c.activity_text)
    assert formulae.references == ["C2.5", "E2.5"]
    assert [f.page for f in formulae.fragments] == [31, 32, 33]
    for record in [surds, proportion, formulae]:
        for fragment in record.fragments:
            assert (tmp_path / fragment.image).is_file()
            assert (tmp_path / fragment.glyph_file).is_file()
    assert "√" in surds.activity_text
