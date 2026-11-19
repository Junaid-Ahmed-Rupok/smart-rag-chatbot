import io

import docx
import pytest
from reportlab.pdfgen import canvas

from smartrag.documents import IngestError, load_file

BIG = 10 * 1024 * 1024


def make_pdf(pages):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        if text:
            c.drawString(72, 720, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_text_file():
    f = load_file("n.txt", b"Hello world, this is a note.", BIG)
    assert f.docs[0].metadata["source"] == "n.txt" and "Hello world" in f.docs[0].page_content


def test_markdown_sections():
    f = load_file("a.md", b"# One\nalpha\n\n## Two\nbeta", BIG)
    assert [d.metadata["section"] for d in f.docs] == ["One", "Two"]


def test_pdf_keeps_page_numbers():
    f = load_file("r.pdf", make_pdf(["First page text here", "Second page text here"]), BIG)
    assert [d.metadata["page"] for d in f.docs] == [1, 2]
    assert f.pages == 2


def test_scanned_pdf_gives_clear_error():
    with pytest.raises(IngestError, match="scanned"):
        load_file("scan.pdf", make_pdf(["", ""]), BIG)


def test_damaged_pdf():
    with pytest.raises(IngestError, match="could not be read"):
        load_file("bad.pdf", b"%PDF-1.4 not really a pdf", BIG)


def test_docx_headings_and_tables():
    d = docx.Document()
    d.add_heading("Methods", level=1)
    d.add_paragraph("We measured things carefully.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "alpha", "beta"
    buf = io.BytesIO()
    d.save(buf)
    f = load_file("m.docx", buf.getvalue(), BIG)
    assert f.docs[0].metadata["section"] == "Methods"
    assert "alpha | beta" in f.docs[0].page_content


def test_corrupt_docx():
    with pytest.raises(IngestError, match="could not be read"):
        load_file("x.docx", b"not a zip", BIG)


def test_csv_becomes_readable_rows():
    f = load_file("t.csv", b"name,age\nAnn,31\nBob,42\n", BIG)
    assert "name: Ann; age: 31" in f.docs[0].page_content


def test_csv_without_rows():
    with pytest.raises(IngestError):
        load_file("t.csv", b"name,age\n", BIG)


def test_limits_and_types():
    with pytest.raises(IngestError, match="unsupported"):
        load_file("a.exe", b"x", BIG)
    with pytest.raises(IngestError, match="empty"):
        load_file("a.txt", b"", BIG)
    with pytest.raises(IngestError, match="larger"):
        load_file("a.txt", b"x" * 100, 50)
    with pytest.raises(IngestError, match="empty"):
        load_file("a.txt", b"   \n  ", BIG)


def test_path_in_name_is_removed():
    assert load_file("../../etc/notes.txt", b"hello there friend", BIG).name == "notes.txt"


def test_non_utf8_text():
    f = load_file("l.txt", "café résumé".encode("cp1252"), BIG)
    assert "café" in f.docs[0].page_content


def test_hyphenated_words_are_rejoined():
    f = load_file("h.txt", b"inter-\nnational trade is good", BIG)
    assert "international" in f.docs[0].page_content
