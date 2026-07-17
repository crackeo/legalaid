from ingest.crawler import find_pdf_links, _safe_filename

INDEX_HTML = """
<html><body>
<a href="/wp-content/uploads/2010/05/Penal-Code-2004.pdf">Penal Code of Bhutan 2004</a>
<a href="https://oag.gov.bt/files/Evidence%20Act%202005.pdf">Evidence Act 2005</a>
<a href="/wp-content/uploads/2010/05/Penal-Code-2004.pdf">duplicate link</a>
<a href="https://other-site.example/external.pdf">external PDF</a>
<a href="/about-us/">not a pdf</a>
<a href="/files/report.PDF?ver=2">Uppercase ext with query</a>
</body></html>
"""


def test_find_pdf_links_absolute_dedup_and_same_host():
    links = find_pdf_links(INDEX_HTML, "https://oag.gov.bt/language/en/resources/acts-2/")
    urls = [l["url"] for l in links]
    assert urls == [
        "https://oag.gov.bt/wp-content/uploads/2010/05/Penal-Code-2004.pdf",
        "https://oag.gov.bt/files/Evidence%20Act%202005.pdf",
        "https://oag.gov.bt/files/report.PDF?ver=2",
    ]
    assert links[0]["title"] == "Penal Code of Bhutan 2004"


def test_find_pdf_links_can_allow_external_hosts():
    links = find_pdf_links(INDEX_HTML, "https://oag.gov.bt/x/", same_host_only=False)
    assert any(l["url"].startswith("https://other-site.example") for l in links)


def test_safe_filename_stable_and_clean():
    url = "https://oag.gov.bt/files/Evidence%20Act%202005.pdf"
    name = _safe_filename(url)
    assert name == _safe_filename(url)  # deterministic
    assert name.endswith("Evidence_Act_2005.pdf")
    assert "/" not in name and "%" not in name and " " not in name
