"""Render a Markdown document to PDF with markdown-it and headless Chrome.

Usage: python md2pdf.py doc.md [doc.pdf]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
@page { size: Letter; margin: 2cm 2cm 2.2cm 2cm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10pt; line-height: 1.45; color: #1d2330; }
h1 { font-size: 20pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #c9ced8; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; }
p, li { orphans: 3; widows: 3; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f2f4f7; padding: 0 2px; }
pre { background: #f6f7f9; border: 1px solid #e1e5ea; padding: 8pt; font-size: 8.3pt; line-height: 1.3;
      overflow-wrap: anywhere; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9ced8; padding: 3pt 6pt; vertical-align: top; text-align: left; }
th { background: #eef1f5; }
em { color: #3c4556; }
"""


def main():
    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else source.with_suffix(".pdf")
    body = MarkdownIt("commonmark").enable("table").render(source.read_text())
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "doc.html"
        page.write_text(html)
        subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={target}", page.as_uri()], check=True, capture_output=True)
    print(target)


if __name__ == "__main__":
    main()
