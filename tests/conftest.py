import textwrap
from pathlib import Path

import fitz
import pytest

# Mimics the drafting style of a Bhutanese Act: chapter headings, marginal
# notes, numbered sections with sub-clauses, page headers/footers.
PAGE1 = """\
PENAL CODE OF BHUTAN (FIXTURE)
CHAPTER 1
PRELIMINARY
Title
1. This Code is the Penal Code of Bhutan, 2004.
Commencement
2. This Code shall come into force on the day of its enact-
ment by Parliament.
Penal Code of Bhutan
1
"""

PAGE2 = """\
PENAL CODE OF BHUTAN (FIXTURE)
CHAPTER 3
OFFENCES AGAINST THE PERSON
Murder
92. A defendant shall be guilty of the offence of murder if the
defendant commits homicide under the following circumstances:
(a) The homicide is premeditated; or
(b) The homicide is committed in the course of a felony.
93. The offence of murder shall be a felony of the first degree.
Penal Code of Bhutan
2
"""

PAGE3 = """\
PENAL CODE OF BHUTAN (FIXTURE)
Assault
94. A defendant shall be guilty of the offence of assault if the
defendant unlawfully applies force to another person.
95. The offence of assault shall be a petty misdemeanour.
Penal Code of Bhutan
3
"""

PAGE4 = """\
PENAL CODE OF BHUTAN (FIXTURE)
CHAPTER 4
GENERAL
Repeal
96. The provisions of any law inconsistent with this Code are
hereby repealed.
Penal Code of Bhutan
4
"""

FIXTURE_PAGES = [PAGE1, PAGE2, PAGE3, PAGE4]


@pytest.fixture(scope="session")
def fixture_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdf") / "fixture_act.pdf"
    doc = fitz.open()
    for text in FIXTURE_PAGES:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path
