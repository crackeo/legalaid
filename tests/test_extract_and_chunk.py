from ingest.chunk import chunk_document, MAX_CHUNK_CHARS
from ingest.extract import pdf_to_clean_pages


def test_extract_repairs_hyphenation_and_strips_headers(fixture_pdf):
    pages = pdf_to_clean_pages(fixture_pdf)
    assert len(pages) == 4
    assert "enactment" in pages[0]          # "enact-\nment" repaired
    # repeated header/footer stripped from body pages
    assert "PENAL CODE OF BHUTAN (FIXTURE)" not in pages[1]
    assert "Penal Code of Bhutan" not in pages[2]


def test_chunker_one_chunk_per_section_with_metadata(fixture_pdf):
    pages = pdf_to_clean_pages(fixture_pdf)
    chunks = chunk_document(pages, doc_title="Penal Code (fixture)",
                            source_url="https://example/x.pdf", doc_type="act")
    numbers = [c.section_number for c in chunks]
    assert numbers == ["1", "2", "92", "93", "94", "95", "96"]

    s92 = next(c for c in chunks if c.section_number == "92")
    assert s92.page == 2
    assert "CHAPTER 3" in s92.section_heading
    assert "Murder" in s92.section_heading
    assert "(a) The homicide is premeditated" in s92.text
    assert s92.source_url == "https://example/x.pdf"

    s94 = next(c for c in chunks if c.section_number == "94")
    assert s94.page == 3
    assert "Assault" in s94.section_heading
    assert "CHAPTER 3" in s94.section_heading  # chapter persists across sections

    s96 = next(c for c in chunks if c.section_number == "96")
    assert "CHAPTER 4" in s96.section_heading  # chapter switches


def test_chunker_handles_articles():
    pages = ["Article 7: Fundamental Rights\n"
             "All persons shall have the right to life, liberty and security.\n"
             "Article 8: Fundamental Duties\n"
             "A Bhutanese citizen shall preserve the environment."]
    chunks = chunk_document(pages, doc_title="Constitution (fixture)")
    assert [c.section_number for c in chunks] == ["Article 7", "Article 8"]
    assert "right to life" in chunks[0].text


def test_oversized_section_is_split_with_continuation_marker():
    clause = "\n".join(f"({i}) A very long sub-clause about procedure number {i}. "
                       + "x" * 200 for i in range(1, 40))
    pages = [f"CHAPTER 2\nPROCEDURE\n10. The following rules apply:\n{clause}"]
    chunks = chunk_document(pages, doc_title="Fixture Act")
    assert len(chunks) > 1
    assert all(c.section_number == "10" for c in chunks)
    assert all(len(c.text) <= MAX_CHUNK_CHARS + 300 for c in chunks)
    assert chunks[1].text.startswith("[10 (continued)]")
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
