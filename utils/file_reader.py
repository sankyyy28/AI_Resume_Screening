"""
utils/file_reader.py
Extracts plain text from uploaded resume files (PDF, DOCX, TXT).

Dependencies (install as needed):
    pip install pymupdf python-docx
"""

import io
from typing import Tuple, Optional


def extract_text_from_file(uploaded_file) -> Tuple[str, Optional[str]]:
    """
    Extract text from a Streamlit UploadedFile object.

    Supports:
        .pdf  — via PyMuPDF (fitz); falls back to pdfminer if unavailable
        .docx — via python-docx
        .txt  — direct decode

    Returns
    -------
    (text, error)  where error is None on success, or an error message string.
    """
    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    # ── TXT ──────────────────────────────────────────────────────────────────
    if filename.endswith(".txt"):
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
            return text.strip(), None
        except Exception as e:
            return "", str(e)

    # ── DOCX ─────────────────────────────────────────────────────────────────
    if filename.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Also pull text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paragraphs.append(cell.text.strip())
            return "\n".join(paragraphs).strip(), None
        except ImportError:
            return "", "python-docx not installed. Run: pip install python-docx"
        except Exception as e:
            return "", str(e)

    # ── PDF ───────────────────────────────────────────────────────────────────
    if filename.endswith(".pdf"):
        # Try PyMuPDF first (fastest, best layout)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()
            text = "\n".join(pages).strip()
            if text:
                return text, None
        except ImportError:
            pass  # fall through to pdfminer
        except Exception as e:
            return "", f"PyMuPDF error: {e}"

        # Fallback: pdfminer.six
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(io.BytesIO(raw_bytes))
            if text and text.strip():
                return text.strip(), None
        except ImportError:
            pass
        except Exception as e:
            return "", f"pdfminer error: {e}"

        return "", (
            "Could not extract PDF text. Install a PDF library:\n"
            "  pip install pymupdf          # recommended\n"
            "  pip install pdfminer.six     # fallback"
        )

    return "", f"Unsupported file type: {filename}"
